// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::response_store::{
    ResponseStorage, ResponseStore, StoreConfig, StoreError,
};
use std::{sync::Arc, time::Duration};

fn config() -> StoreConfig {
    StoreConfig {
        max_bytes: 20,
        max_entries: 2,
        max_record_bytes: 16,
        ..StoreConfig::default()
    }
}

async fn contract(first: Arc<dyn ResponseStore>, second: Arc<dyn ResponseStore>) {
    first.initialize().await.unwrap();
    second.initialize().await.unwrap();
    let ttl = Duration::from_secs(2);
    first.put("a", vec![1; 9], ttl).await.unwrap();
    assert_eq!(second.get("a").await.unwrap(), Some(vec![1; 9]));
    second.put("b", vec![2; 9], ttl).await.unwrap();
    assert!(matches!(
        first.put("c", vec![3], ttl).await,
        Err(StoreError::Capacity)
    ));
    assert!(matches!(
        first.put("a", vec![3; 15], ttl).await,
        Err(StoreError::Capacity)
    ));
    assert_eq!(second.get("a").await.unwrap(), Some(vec![1; 9]));
    assert!(
        !first
            .compare_exchange("a", &[0], vec![4], ttl)
            .await
            .unwrap()
    );
    assert!(
        second
            .compare_exchange("a", &[1; 9], vec![4], ttl)
            .await
            .unwrap()
    );
    assert!(
        !first
            .compare_exchange("a", &[1; 9], vec![5], ttl)
            .await
            .unwrap()
    );
    assert!(second.delete("a").await.unwrap());
    assert!(
        !first
            .compare_exchange("a", &[4], vec![5], ttl)
            .await
            .unwrap()
    );
    assert!(!first.delete("a").await.unwrap());
    second.delete("b").await.unwrap();
    assert!(matches!(
        first.put("huge", vec![0; 16], ttl).await,
        Err(StoreError::TooLarge)
    ));
    first
        .put("exp", vec![1; 12], Duration::from_millis(20))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(35)).await;
    assert!(second.get("exp").await.unwrap().is_none());
    second.purge_expired().await.unwrap();
    first.put("next", vec![2; 12], ttl).await.unwrap();
    first.delete("next").await.unwrap();
    let mut tasks = tokio::task::JoinSet::new();
    for i in 0..32 {
        let store = if i % 2 == 0 {
            first.clone()
        } else {
            second.clone()
        };
        tasks.spawn(async move { store.put(&format!("{i:02}"), vec![1; 8], ttl).await });
    }
    let mut successes = 0;
    while let Some(result) = tasks.join_next().await {
        successes += usize::from(result.unwrap().is_ok());
    }
    assert_eq!(successes, 2);
}

#[tokio::test]
async fn memory_contract() {
    let storage = ResponseStorage::memory(config()).unwrap();
    contract(storage.store.clone(), storage.store.clone()).await;
}

#[cfg(feature = "response-store-redis")]
#[tokio::test]
#[ignore = "requires TEST_RESPONSE_REDIS_URL"]
async fn redis_contract_across_independent_clients() {
    let url = std::env::var("TEST_RESPONSE_REDIS_URL").expect("set TEST_RESPONSE_REDIS_URL");
    let namespace = format!("contract-{}", uuid::Uuid::new_v4());
    let first = ResponseStorage::redis(&url, &namespace, config()).unwrap();
    let second = ResponseStorage::redis(&url, &namespace, config()).unwrap();
    contract(first.store.clone(), second.store.clone()).await;
    let mut different = config();
    different.max_entries = 3;
    let mismatch = ResponseStorage::redis(&url, &namespace, different).unwrap();
    assert!(matches!(
        mismatch.store.initialize().await,
        Err(StoreError::Unavailable)
    ));
}

#[cfg(feature = "response-store-redis")]
#[tokio::test]
#[ignore = "requires TEST_RESPONSE_REDIS_URL"]
async fn redis_namespace_isolation_and_redacted_connection_failures() {
    let url = std::env::var("TEST_RESPONSE_REDIS_URL").expect("set TEST_RESPONSE_REDIS_URL");
    let first = ResponseStorage::redis(
        &url,
        &format!("isolation-{}", uuid::Uuid::new_v4()),
        config(),
    )
    .unwrap();
    let second = ResponseStorage::redis(
        &url,
        &format!("isolation-{}", uuid::Uuid::new_v4()),
        config(),
    )
    .unwrap();
    first
        .store
        .put("a", vec![1], Duration::from_secs(2))
        .await
        .unwrap();
    assert!(second.store.get("a").await.unwrap().is_none());
    let unavailable = ResponseStorage::redis(
        "redis://:secret-must-not-leak@127.0.0.1:1/",
        "unavailable",
        config(),
    )
    .unwrap();
    let error = unavailable.store.initialize().await.unwrap_err();
    assert_eq!(error.to_string(), "Response storage is unavailable");
    assert!(!format!("{error:?}").contains("secret-must-not-leak"));
}
