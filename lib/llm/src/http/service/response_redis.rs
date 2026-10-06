// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional Redis storage. No server configuration or keys outside our namespace are changed.
use super::response_store::{ResponseStorage, ResponseStore, StoreConfig, StoreError};
use async_trait::async_trait;
use redis::{Client, Script, aio::ConnectionManager};
use std::{sync::Arc, time::Duration};
use tokio::sync::OnceCell;

pub struct RedisResponseStore {
    client: Client,
    connection: OnceCell<ConnectionManager>,
    prefix: String,
    config: StoreConfig,
}

impl RedisResponseStore {
    pub fn new(url: &str, namespace: &str, config: StoreConfig) -> anyhow::Result<Self> {
        config.validate()?;
        anyhow::ensure!(
            !namespace.is_empty()
                && namespace.len() <= 128
                && namespace
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
            "response Redis namespace must contain 1-128 letters, digits, hyphens, or underscores"
        );
        // Never propagate a URL/parser error: it may include credentials.
        let client =
            Client::open(url).map_err(|_| anyhow::anyhow!("Invalid response Redis URL"))?;
        Ok(Self {
            client,
            connection: OnceCell::new(),
            prefix: format!("dynamo:responses:{{{namespace}}}:"),
            config,
        })
    }

    async fn execute(
        &self,
        operation: &str,
        key: &str,
        value: &[u8],
        expected: &[u8],
        ttl: Duration,
    ) -> Result<(bool, Vec<u8>), StoreError> {
        let result = tokio::time::timeout(Duration::from_secs(3), async {
            let mut connection = self
                .connection
                .get_or_try_init(|| self.client.get_connection_manager())
                .await?
                .clone();
            Script::new(include_str!("response_redis.lua"))
                .key(format!("{}record:{key}", self.prefix))
                .key(format!("{}expires", self.prefix))
                .key(format!("{}sizes", self.prefix))
                .key(format!("{}accounting", self.prefix))
                .arg(operation)
                .arg(key)
                .arg(value)
                .arg(expected)
                .arg(ttl.as_millis().min(u64::MAX as u128) as u64)
                .arg(self.config.max_bytes)
                .arg(self.config.max_entries)
                .arg(self.config.max_record_bytes)
                .invoke_async::<(i64, Vec<u8>)>(&mut connection)
                .await
        })
        .await
        .map_err(|_| StoreError::Unavailable)?
        .map_err(|_| StoreError::Unavailable)?;
        match result {
            (1, value) => Ok((true, value)),
            (0, value) => Ok((false, value)),
            (-1, _) => Err(StoreError::Capacity),
            (-2, _) => Err(StoreError::TooLarge),
            _ => Err(StoreError::Unavailable),
        }
    }
}

impl ResponseStorage {
    pub fn redis(url: &str, namespace: &str, config: StoreConfig) -> anyhow::Result<Arc<Self>> {
        Ok(Arc::new(Self {
            store: Arc::new(RedisResponseStore::new(url, namespace, config.clone())?),
            config,
            background: super::response_background::BackgroundJobs::default(),
        }))
    }
}

#[async_trait]
impl ResponseStore for RedisResponseStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let (found, value) = self.execute("get", key, &[], &[], Duration::ZERO).await?;
        Ok(found.then_some(value))
    }
    async fn put(&self, key: &str, value: Vec<u8>, ttl: Duration) -> Result<(), StoreError> {
        if key.len().saturating_add(value.len()) > self.config.max_record_bytes {
            return Err(StoreError::TooLarge);
        }
        self.execute("put", key, &value, &[], ttl).await.map(|_| ())
    }
    async fn compare_exchange(
        &self,
        key: &str,
        expected: &[u8],
        value: Vec<u8>,
        ttl: Duration,
    ) -> Result<bool, StoreError> {
        if key.len().saturating_add(value.len()) > self.config.max_record_bytes {
            return Err(StoreError::TooLarge);
        }
        self.execute("cas", key, &value, expected, ttl)
            .await
            .map(|(changed, _)| changed)
    }
    async fn delete(&self, key: &str) -> Result<bool, StoreError> {
        self.execute("delete", key, &[], &[], Duration::ZERO)
            .await
            .map(|(deleted, _)| deleted)
    }
    async fn purge_expired(&self) -> Result<(), StoreError> {
        self.execute("purge", "", &[], &[], Duration::ZERO)
            .await
            .map(|_| ())
    }
    async fn initialize(&self) -> Result<(), StoreError> {
        self.purge_expired().await
    }
}
