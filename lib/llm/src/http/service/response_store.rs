// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, frontend-owned Responses records. Inference workers do not use this store.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use axum::http::HeaderMap;
use dynamo_protocols::types::responses::{InputParam, Status};
use dynamo_runtime::config::environment_names::llm as env;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::time::Instant;

use crate::protocols::openai::responses::{NvCreateResponse, NvResponse};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Response storage is disabled")]
    Disabled,
    #[error("Background response capacity exceeded")]
    Busy,
    #[error("Only background responses can be cancelled")]
    NotBackground,
    #[error("Response not found or expired")]
    NotFound,
    #[error("The previous response is not finished")]
    NotFinished,
    #[error("Response storage capacity exceeded")]
    Capacity,
    #[error("Response record exceeds the configured size limit")]
    TooLarge,
    #[error("Response storage is unavailable")]
    Unavailable,
    #[error("Invalid stored response")]
    InvalidRecord,
}

/// Limits count serialized record and key bytes, not allocator or process RSS.
#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub ttl: Duration,
    pub max_bytes: usize,
    pub max_entries: usize,
    pub max_record_bytes: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(3600),
            max_bytes: 256 * 1024 * 1024,
            max_entries: 10_000,
            max_record_bytes: 8 * 1024 * 1024,
        }
    }
}

impl StoreConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.ttl.is_zero() && self.ttl <= Duration::from_secs(365 * 86400),
            "response store TTL must be between 1 second and 365 days"
        );
        anyhow::ensure!(
            self.max_entries > 0
                && self.max_record_bytes > 0
                && self.max_bytes >= self.max_record_bytes,
            "response store limits must be positive and max bytes must cover one record"
        );
        Ok(())
    }
}

/// A complete response and an independent replay snapshot. Expiring an ancestor
/// must not invalidate a newer response. Request instructions are not inherited.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredResponse {
    pub schema_version: u32,
    pub response: NvResponse,
    pub input: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<u64>,
}

/// Atomic record operations. Reads never return expired values. Implementations
/// must reject writes over their limits, rather than evicting unexpired records.
/// The bytes are versioned JSON, not engine state or serialized Python objects.
#[async_trait]
pub trait ResponseStore: Send + Sync {
    /// Verify remote connectivity/configuration before the frontend starts serving.
    async fn initialize(&self) -> Result<(), StoreError> {
        Ok(())
    }
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError>;
    async fn put(&self, key: &str, value: Vec<u8>, ttl: Duration) -> Result<(), StoreError>;
    /// Replace only the exact current value. Missing/expired records never reappear.
    async fn compare_exchange(
        &self,
        key: &str,
        expected: &[u8],
        value: Vec<u8>,
        ttl: Duration,
    ) -> Result<bool, StoreError>;
    async fn delete(&self, key: &str) -> Result<bool, StoreError>;
    async fn purge_expired(&self) -> Result<(), StoreError>;
}

struct Entry {
    value: Vec<u8>,
    expires: Instant,
}

#[derive(Default)]
struct MemoryData {
    entries: HashMap<String, Entry>,
    bytes: usize,
}

impl MemoryData {
    fn purge(&mut self) {
        let now = Instant::now();
        self.entries.retain(|key, entry| {
            if entry.expires <= now {
                self.bytes -= key.len() + entry.value.len();
                false
            } else {
                true
            }
        });
    }
}

pub struct MemoryResponseStore {
    config: StoreConfig,
    data: Mutex<MemoryData>,
}

impl MemoryResponseStore {
    pub fn new(config: StoreConfig) -> anyhow::Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            data: Mutex::new(MemoryData::default()),
        })
    }
}

impl MemoryResponseStore {
    fn put_locked(
        &self,
        data: &mut MemoryData,
        key: &str,
        value: Vec<u8>,
        ttl: Duration,
    ) -> Result<(), StoreError> {
        let size = key
            .len()
            .checked_add(value.len())
            .ok_or(StoreError::TooLarge)?;
        if size > self.config.max_record_bytes {
            return Err(StoreError::TooLarge);
        }
        let expires = Instant::now()
            .checked_add(ttl)
            .ok_or(StoreError::TooLarge)?;
        let old_size = data
            .entries
            .get(key)
            .map_or(0, |entry| key.len() + entry.value.len());
        let bytes = data.bytes - old_size + size;
        if bytes > self.config.max_bytes
            || (old_size == 0 && data.entries.len() >= self.config.max_entries)
        {
            return Err(StoreError::Capacity);
        }
        data.entries
            .insert(key.to_owned(), Entry { value, expires });
        data.bytes = bytes;
        Ok(())
    }
}

#[async_trait]
impl ResponseStore for MemoryResponseStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let mut data = self.data.lock();
        if data
            .entries
            .get(key)
            .is_some_and(|entry| entry.expires <= Instant::now())
        {
            let entry = data.entries.remove(key).unwrap();
            data.bytes -= key.len() + entry.value.len();
        }
        Ok(data.entries.get(key).map(|entry| entry.value.clone()))
    }

    async fn put(&self, key: &str, value: Vec<u8>, ttl: Duration) -> Result<(), StoreError> {
        let mut data = self.data.lock();
        data.purge();
        self.put_locked(&mut data, key, value, ttl)
    }

    async fn compare_exchange(
        &self,
        key: &str,
        expected: &[u8],
        value: Vec<u8>,
        ttl: Duration,
    ) -> Result<bool, StoreError> {
        let mut data = self.data.lock();
        data.purge();
        if !data
            .entries
            .get(key)
            .is_some_and(|entry| entry.value == expected)
        {
            return Ok(false);
        }
        self.put_locked(&mut data, key, value, ttl)?;
        Ok(true)
    }

    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        let mut data = self.data.lock();
        data.purge();
        if let Some(entry) = data.entries.remove(key) {
            data.bytes -= key.len() + entry.value.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn purge_expired(&self) -> Result<(), StoreError> {
        self.data.lock().purge();
        Ok(())
    }
}

pub struct ResponseStorage {
    pub store: Arc<dyn ResponseStore>,
    pub config: StoreConfig,
    pub background: super::response_background::BackgroundJobs,
}

impl ResponseStorage {
    pub fn memory(config: StoreConfig) -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            store: Arc::new(MemoryResponseStore::new(config.clone())?),
            config,
            background: super::response_background::BackgroundJobs::default(),
        }))
    }

    pub fn from_env() -> anyhow::Result<Option<Arc<Self>>> {
        let backend = match std::env::var(env::DYN_RESPONSE_STORE_BACKEND) {
            Ok(value) => value,
            Err(std::env::VarError::NotPresent) => "disabled".into(),
            Err(_) => anyhow::bail!("DYN_RESPONSE_STORE_BACKEND must be valid Unicode"),
        };
        if backend == "disabled" {
            return Ok(None);
        }
        anyhow::ensure!(
            matches!(backend.as_str(), "memory" | "redis"),
            "unsupported DYN_RESPONSE_STORE_BACKEND; expected disabled, memory, or redis"
        );
        let mut config = StoreConfig::default();
        fn value(name: &str, default: usize) -> anyhow::Result<usize> {
            match std::env::var(name) {
                Ok(v) => v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("{name} must be a positive integer")),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(_) => anyhow::bail!("{name} must be valid Unicode"),
            }
        }
        config.ttl = Duration::from_secs(value(env::DYN_RESPONSE_STORE_TTL_SECS, 3600)? as u64);
        config.max_bytes = value(env::DYN_RESPONSE_STORE_MAX_BYTES, config.max_bytes)?;
        config.max_entries = value(env::DYN_RESPONSE_STORE_MAX_ENTRIES, config.max_entries)?;
        config.max_record_bytes = value(
            env::DYN_RESPONSE_STORE_MAX_RECORD_BYTES,
            config.max_record_bytes,
        )?;
        let mut storage = match backend.as_str() {
            "memory" => Self::memory(config)?,
            #[cfg(feature = "response-store-redis")]
            "redis" => {
                let url = std::env::var(env::DYN_RESPONSE_STORE_REDIS_URL)
                    .map_err(|_| anyhow::anyhow!("DYN_RESPONSE_STORE_REDIS_URL is required"))?;
                let namespace = std::env::var(env::DYN_RESPONSE_STORE_REDIS_NAMESPACE)
                    .unwrap_or_else(|_| "default".into());
                Self::redis(&url, &namespace, config)?
            }
            _ => anyhow::bail!(
                "Redis Responses storage requires a build with response-store-redis enabled"
            ),
        };
        Arc::get_mut(&mut storage).unwrap().background =
            super::response_background::BackgroundJobs::from_env()?;
        Ok(Some(storage))
    }

    pub async fn get(&self, scope: &str, id: &str) -> Result<StoredResponse, StoreError> {
        if !id.starts_with("resp_") || id.len() > 128 {
            return Err(StoreError::NotFound);
        }
        let bytes = self
            .store
            .get(&format!("{scope}:{id}"))
            .await?
            .ok_or(StoreError::NotFound)?;
        let record: StoredResponse =
            serde_json::from_slice(&bytes).map_err(|_| StoreError::InvalidRecord)?;
        if record.schema_version != 1 {
            return Err(StoreError::InvalidRecord);
        }
        super::response_background::expire_abandoned(self, scope, record, bytes).await
    }

    pub async fn delete(&self, scope: &str, id: &str) -> Result<bool, StoreError> {
        if !id.starts_with("resp_") || id.len() > 128 {
            return Err(StoreError::NotFound);
        }
        self.store.delete(&format!("{scope}:{id}")).await
    }
}

/// Scope by the supplied authorization credential, never by client metadata.
/// This is isolation, not authentication: deployment authentication remains required.
pub fn credential_scope(headers: &HeaderMap) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            headers
                .get(axum::http::header::AUTHORIZATION)
                .map_or(&b""[..], |v| v.as_bytes())
        )
    )
}

pub struct PreparedResponse {
    pub(super) storage: Arc<ResponseStorage>,
    pub(super) key: String,
    pub(super) input: Vec<Value>,
}

impl PreparedResponse {
    pub(super) fn record(&self, response: &NvResponse) -> Result<StoredResponse, StoreError> {
        let mut input = self.input.clone();
        for item in &response.inner.output {
            input.push(serde_json::to_value(item).map_err(|_| StoreError::InvalidRecord)?);
        }
        // Validate the round trip before acknowledging storage.
        let _: InputParam = serde_json::from_value(Value::Array(input.clone()))
            .map_err(|_| StoreError::InvalidRecord)?;
        let record = StoredResponse {
            schema_version: 1,
            response: response.clone(),
            input,
            deadline: None,
        };
        Ok(record)
    }

    pub async fn persist(&self, response: &NvResponse) -> Result<(), StoreError> {
        let record = self.record(response)?;
        let bytes = serde_json::to_vec(&record).map_err(|_| StoreError::InvalidRecord)?;
        self.storage
            .store
            .put(
                &format!("{}:{}", self.key, response.inner.id),
                bytes,
                self.storage.config.ttl,
            )
            .await
    }
}

pub async fn prepare(
    storage: Option<&Arc<ResponseStorage>>,
    scope: &str,
    request: &mut NvCreateResponse,
) -> Result<Option<PreparedResponse>, StoreError> {
    let Some(storage) = storage else {
        if request.inner.previous_response_id.is_some() {
            return Err(StoreError::Disabled);
        }
        request.inner.store = Some(false);
        return Ok(None);
    };
    let mut input = if let Some(id) = &request.inner.previous_response_id {
        let previous = storage.get(scope, id).await?;
        if matches!(
            previous.response.inner.status,
            Status::Queued | Status::InProgress
        ) {
            return Err(StoreError::NotFinished);
        }
        previous.input
    } else {
        Vec::new()
    };
    match &request.inner.input {
        InputParam::Text(text) => input.push(json!({"role":"user", "content":text})),
        InputParam::Items(items) => {
            for item in items {
                input.push(serde_json::to_value(item).map_err(|_| StoreError::InvalidRecord)?);
            }
        }
    }
    // Bound history before it is sent through tokenization and generation.
    if serde_json::to_vec(&input)
        .map_err(|_| StoreError::InvalidRecord)?
        .len()
        > storage.config.max_record_bytes
    {
        return Err(StoreError::TooLarge);
    }
    request.inner.input = serde_json::from_value(Value::Array(input.clone()))
        .map_err(|_| StoreError::InvalidRecord)?;
    let store = request.inner.store.unwrap_or(true);
    request.inner.store = Some(store);
    Ok(store.then(|| PreparedResponse {
        storage: storage.clone(),
        key: scope.to_owned(),
        input,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> StoreConfig {
        StoreConfig {
            max_bytes: 16,
            max_record_bytes: 12,
            max_entries: 2,
            ..StoreConfig::default()
        }
    }

    #[tokio::test]
    async fn capacity_updates_and_deletion_are_atomic() {
        let store = MemoryResponseStore::new(small_config()).unwrap();
        let ttl = Duration::from_secs(10);
        store.put("a", vec![0; 7], ttl).await.unwrap();
        store.put("b", vec![0; 7], ttl).await.unwrap();
        assert!(matches!(
            store.put("c", vec![0], ttl).await,
            Err(StoreError::Capacity)
        ));
        assert!(matches!(
            store.put("a", vec![0; 11], ttl).await,
            Err(StoreError::Capacity)
        ));
        assert_eq!(store.get("a").await.unwrap().unwrap().len(), 7);
        store.put("a", vec![1], ttl).await.unwrap();
        assert!(store.delete("b").await.unwrap());
        assert!(!store.delete("b").await.unwrap());
        store.put("c", vec![2; 11], ttl).await.unwrap();
        assert!(matches!(
            store.put("d", vec![0; 12], ttl).await,
            Err(StoreError::TooLarge)
        ));
    }

    #[tokio::test]
    async fn expiration_reclaims_capacity() {
        let store = MemoryResponseStore::new(small_config()).unwrap();
        store.put("a", vec![0; 11], Duration::ZERO).await.unwrap();
        assert!(store.get("a").await.unwrap().is_none());
        store
            .put("b", vec![1; 11], Duration::from_secs(1))
            .await
            .unwrap();
        store.purge_expired().await.unwrap();
        assert!(store.get("b").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn concurrent_writes_cannot_exceed_capacity() {
        let store = Arc::new(MemoryResponseStore::new(small_config()).unwrap());
        let mut tasks = tokio::task::JoinSet::new();
        for id in 0..32 {
            let store = store.clone();
            tasks.spawn(async move {
                store
                    .put(&format!("{id:02}"), vec![0; 6], Duration::from_secs(1))
                    .await
            });
        }
        let mut successes = 0;
        while let Some(result) = tasks.join_next().await {
            successes += usize::from(result.unwrap().is_ok());
        }
        assert_eq!(successes, 2);
    }

    #[tokio::test]
    async fn compare_exchange_never_overwrites_a_winner_or_restores_deleted_records() {
        let store = MemoryResponseStore::new(small_config()).unwrap();
        let ttl = Duration::from_secs(10);
        store.put("a", vec![1], ttl).await.unwrap();
        assert!(
            store
                .compare_exchange("a", &[1], vec![2], ttl)
                .await
                .unwrap()
        );
        assert!(
            !store
                .compare_exchange("a", &[1], vec![3], ttl)
                .await
                .unwrap()
        );
        assert_eq!(store.get("a").await.unwrap(), Some(vec![2]));
        store.delete("a").await.unwrap();
        assert!(
            !store
                .compare_exchange("a", &[2], vec![3], ttl)
                .await
                .unwrap()
        );
        assert!(store.get("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn disabled_storage_does_not_claim_to_store() {
        let mut request: NvCreateResponse =
            serde_json::from_value(json!({"model":"test", "input":"hello", "store":true})).unwrap();
        assert!(
            prepare(None, "scope", &mut request)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(request.inner.store, Some(false));
        request.inner.previous_response_id = Some("resp_missing".into());
        assert!(matches!(
            prepare(None, "scope", &mut request).await,
            Err(StoreError::Disabled)
        ));
    }
}
