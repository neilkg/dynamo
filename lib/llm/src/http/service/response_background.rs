// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background lifecycle belongs to the frontend, independently of the inference backend.
use super::response_store::{PreparedResponse, ResponseStorage, StoreError, StoredResponse};
use crate::protocols::openai::responses::NvResponse;
use dynamo_protocols::types::responses::{ErrorObject, Status};
use dynamo_runtime::pipeline::AsyncEngineContext;
use std::{
    future::Future,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

pub struct BackgroundJobs {
    slots: Arc<Semaphore>,
    pub timeout: Duration,
}

impl Default for BackgroundJobs {
    fn default() -> Self {
        Self {
            slots: Arc::new(Semaphore::new(64)),
            timeout: Duration::from_secs(600),
        }
    }
}

impl BackgroundJobs {
    pub fn from_env() -> anyhow::Result<Self> {
        use dynamo_runtime::config::environment_names::llm as env;
        let parse = |name: &str, default: u64| -> anyhow::Result<u64> {
            match std::env::var(name) {
                Ok(value) => Ok(value.parse()?),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(_) => anyhow::bail!("{name} must be valid Unicode"),
            }
        };
        let slots = parse(env::DYN_RESPONSE_BACKGROUND_MAX_JOBS, 64)?;
        let timeout = parse(env::DYN_RESPONSE_BACKGROUND_TIMEOUT_SECS, 600)?;
        anyhow::ensure!(
            (1..=100_000).contains(&slots),
            "background max jobs must be between 1 and 100000"
        );
        anyhow::ensure!(
            (1..=86400).contains(&timeout),
            "background timeout must be between 1 and 86400 seconds"
        );
        Ok(Self {
            slots: Arc::new(Semaphore::new(slots as usize)),
            timeout: Duration::from_secs(timeout),
        })
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
fn encode(record: &StoredResponse) -> Result<Vec<u8>, StoreError> {
    serde_json::to_vec(record).map_err(|_| StoreError::InvalidRecord)
}
fn pending(record: &StoredResponse) -> bool {
    matches!(
        record.response.inner.status,
        Status::Queued | Status::InProgress
    )
}
fn failed(record: &mut StoredResponse, message: &str) {
    record.response.inner.status = Status::Failed;
    record.response.inner.error = Some(ErrorObject {
        code: "server_error".into(),
        message: message.into(),
    });
    record.deadline = None;
}

// A full store must still be able to finish an admitted pending response. If
// even the error detail cannot fit, status-only failure is smaller than the
// pending record (its deadline is removed), and preserves the original history.
async fn replace_terminal(
    storage: &ResponseStorage,
    key: &str,
    expected: &[u8],
    record: &mut StoredResponse,
) -> Result<bool, StoreError> {
    let result = storage
        .store
        .compare_exchange(key, expected, encode(record)?, storage.config.ttl)
        .await;
    if record.response.inner.status == Status::Failed
        && matches!(result, Err(StoreError::Capacity | StoreError::TooLarge))
    {
        record.response.inner.error = None;
        storage
            .store
            .compare_exchange(key, expected, encode(record)?, storage.config.ttl)
            .await
    } else {
        result
    }
}

pub(super) async fn expire_abandoned(
    storage: &ResponseStorage,
    scope: &str,
    mut record: StoredResponse,
    bytes: Vec<u8>,
) -> Result<StoredResponse, StoreError> {
    if pending(&record) && record.deadline.is_some_and(|deadline| deadline <= now()) {
        failed(
            &mut record,
            "Background response exceeded its execution deadline",
        );
        if !replace_terminal(
            storage,
            &format!("{scope}:{}", record.response.inner.id),
            &bytes,
            &mut record,
        )
        .await?
        {
            // A concurrent terminal transition won. Return its actual state.
            let value = storage
                .store
                .get(&format!("{scope}:{}", record.response.inner.id))
                .await?
                .ok_or(StoreError::NotFound)?;
            return serde_json::from_slice(&value).map_err(|_| StoreError::InvalidRecord);
        }
    }
    Ok(record)
}

pub(super) async fn cancel(
    storage: &ResponseStorage,
    scope: &str,
    id: &str,
) -> Result<NvResponse, StoreError> {
    let key = format!("{scope}:{id}");
    // Each competing update moves forward in a finite lifecycle.
    for _ in 0..8 {
        let mut record = storage.get(scope, id).await?;
        if record.response.inner.background != Some(true) {
            return Err(StoreError::NotBackground);
        }
        if !pending(&record) {
            return Ok(record.response);
        }
        let expected = encode(&record)?;
        record.response.inner.status = Status::Cancelled;
        record.deadline = None;
        if storage
            .store
            .compare_exchange(&key, &expected, encode(&record)?, storage.config.ttl)
            .await?
        {
            return Ok(record.response);
        }
    }
    Err(StoreError::Unavailable)
}

pub(super) struct BackgroundJob {
    prepared: PreparedResponse,
    record: StoredResponse,
    bytes: Vec<u8>,
    _permit: OwnedSemaphorePermit,
}

impl PreparedResponse {
    pub(super) async fn background(
        self,
        mut response: NvResponse,
    ) -> Result<BackgroundJob, StoreError> {
        let permit = self
            .storage
            .background
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| StoreError::Busy)?;
        response.inner.background = Some(true);
        response.inner.status = Status::Queued;
        response.inner.completed_at = None;
        let mut record = self.record(&response)?;
        record.deadline = Some(now() + self.storage.background.timeout.as_secs());
        let bytes = encode(&record)?;
        self.storage
            .store
            .put(
                &format!("{}:{}", self.key, response.inner.id),
                bytes.clone(),
                self.storage.background.timeout + self.storage.config.ttl,
            )
            .await?;
        Ok(BackgroundJob {
            prepared: self,
            record,
            bytes,
            _permit: permit,
        })
    }
}

struct CancelOnDrop(Arc<dyn AsyncEngineContext>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.kill();
    }
}

impl BackgroundJob {
    pub(super) fn response(&self) -> NvResponse {
        self.record.response.clone()
    }

    pub(super) fn spawn<F>(
        mut self,
        generation: F,
        context: Arc<dyn AsyncEngineContext>,
        shutdown: CancellationToken,
    ) where
        F: Future<Output = Result<NvResponse, String>> + Send + 'static,
    {
        tokio::spawn(async move {
            let _cancel = CancelOnDrop(context);
            let storage = &self.prepared.storage;
            let key = format!("{}:{}", self.prepared.key, self.record.response.inner.id);
            self.record.response.inner.status = Status::InProgress;
            let active = match encode(&self.record) {
                Ok(bytes) => bytes,
                Err(_) => return,
            };
            match storage
                .store
                .compare_exchange(
                    &key,
                    &self.bytes,
                    active.clone(),
                    storage.background.timeout + storage.config.ttl,
                )
                .await
            {
                Ok(true) => self.bytes = active,
                _ => return,
            }
            tokio::pin!(generation);
            let deadline = tokio::time::sleep(storage.background.timeout);
            tokio::pin!(deadline);
            let mut poll = tokio::time::interval(Duration::from_millis(250));
            let result = loop {
                tokio::select! {
                    biased;
                    _ = shutdown.cancelled() => break Err("Frontend shut down during background generation".to_owned()),
                    _ = &mut deadline => break Err("Background response exceeded its execution deadline".to_owned()),
                    _ = poll.tick() => {
                        match storage.store.get(&key).await {
                            Ok(Some(value)) if value == self.bytes => {},
                            // Cancellation, deletion, expiry, or unavailable storage stops work.
                            _ => return,
                        }
                    }
                    result = &mut generation => break result,
                }
            };
            let mut terminal = match result {
                Ok(mut response) => {
                    response.inner.id = self.record.response.inner.id.clone();
                    response.inner.created_at = self.record.response.inner.created_at;
                    response.inner.background = Some(true);
                    response.store = true;
                    match self.prepared.record(&response) {
                        Ok(record) => record,
                        Err(_) => {
                            failed(&mut self.record, "Could not encode background response");
                            self.record.clone()
                        }
                    }
                }
                Err(message) => {
                    failed(&mut self.record, &message);
                    self.record.clone()
                }
            };
            terminal.deadline = None;
            let bytes = match encode(&terminal) {
                Ok(bytes) => bytes,
                Err(_) => return,
            };
            let result = storage
                .store
                .compare_exchange(&key, &self.bytes, bytes, storage.config.ttl)
                .await;
            if matches!(result, Err(StoreError::Capacity | StoreError::TooLarge)) {
                failed(
                    &mut self.record,
                    "Background response exceeded storage capacity",
                );
                let _ = replace_terminal(storage, &key, &self.bytes, &mut self.record).await;
            } else if let Err(error) = result {
                tracing::warn!(%error, "Could not persist background response");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::response_store::{self, StoreConfig};
    use super::*;
    use crate::protocols::openai::responses::{
        NvCreateResponse, ResponseParams, stream_converter::ResponseStreamConverter,
    };

    fn response() -> NvResponse {
        ResponseStreamConverter::new(
            "test".into(),
            ResponseParams {
                store: Some(true),
                ..Default::default()
            },
        )
        .final_response()
    }

    #[tokio::test]
    async fn admission_is_bounded_and_releases_its_permit() {
        let mut storage = ResponseStorage::memory(StoreConfig::default()).unwrap();
        Arc::get_mut(&mut storage).unwrap().background.slots = Arc::new(Semaphore::new(1));
        let mut request: NvCreateResponse =
            serde_json::from_value(serde_json::json!({"model":"test","input":"hello"})).unwrap();
        let first = response_store::prepare(Some(&storage), "scope", &mut request)
            .await
            .unwrap()
            .unwrap()
            .background(response())
            .await
            .unwrap();
        let second = response_store::prepare(Some(&storage), "scope", &mut request)
            .await
            .unwrap()
            .unwrap()
            .background(response())
            .await;
        assert!(matches!(second, Err(StoreError::Busy)));
        drop(first);
        assert!(
            response_store::prepare(Some(&storage), "scope", &mut request)
                .await
                .unwrap()
                .unwrap()
                .background(response())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn abandoned_job_can_fail_even_when_storage_has_no_free_bytes() {
        let mut response = response();
        response.inner.status = Status::InProgress;
        response.inner.completed_at = None;
        response.inner.background = Some(true);
        let id = response.inner.id.clone();
        let key = format!("scope:{id}");
        let record = StoredResponse {
            schema_version: 1,
            response,
            input: Vec::new(),
            deadline: Some(now() - 1),
        };
        let bytes = encode(&record).unwrap();
        let size = key.len() + bytes.len();
        let storage = ResponseStorage::memory(StoreConfig {
            max_bytes: size,
            max_record_bytes: size,
            ..Default::default()
        })
        .unwrap();
        storage
            .store
            .put(&key, bytes.clone(), Duration::from_secs(60))
            .await
            .unwrap();
        let failed = expire_abandoned(&storage, "scope", record, bytes)
            .await
            .unwrap();
        assert_eq!(failed.response.inner.status, Status::Failed);
        assert!(failed.response.inner.error.is_none());
        assert_eq!(
            storage
                .get("scope", &id)
                .await
                .unwrap()
                .response
                .inner
                .status,
            Status::Failed
        );
    }
}
