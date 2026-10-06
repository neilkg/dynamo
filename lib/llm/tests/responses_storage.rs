// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::http::service::response_store::{ResponseStorage, StoreConfig};
use serde_json::{Value, json};
use std::sync::Arc;

#[path = "common/http_harness.rs"]
mod http_harness;
#[path = "common/ports.rs"]
mod ports;
#[path = "common/scripted_chat_engine.rs"]
mod scripted_chat_engine;
use http_harness::{HarnessService, MODEL, load_agent_fixture, parse_json_sse};
use scripted_chat_engine::ScriptedChatEngine;

async fn service(fixtures: &[&str]) -> HarnessService {
    let mut scripts = Vec::new();
    for fixture in fixtures {
        scripts.push(Ok(load_agent_fixture(fixture).await.unwrap()));
    }
    HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::new(scripts)),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await
}

async fn create(svc: &HarnessService, body: Value) -> Value {
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert!(status.is_success(), "{status}: {body}");
    body
}

#[tokio::test]
async fn retrieve_continue_delete_and_keep_descendant_independent() {
    let svc = service(&["text.sse", "text.sse", "text.sse"]).await;
    let first = create(
        &svc,
        json!({"model":MODEL,"input":"Remember Alice", "instructions":"Only this turn"}),
    )
    .await;
    let first_url = format!(
        "{}/v1/responses/{}",
        svc.base_url,
        first["id"].as_str().unwrap()
    );
    let fetched: Value = svc
        .client
        .get(&first_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first, fetched);
    let second = create(
        &svc,
        json!({"model":MODEL,"input":"Who?","previous_response_id":first["id"]}),
    )
    .await;
    assert_eq!(second["previous_response_id"], first["id"]);
    assert!(second["instructions"].is_null());
    assert_eq!(
        svc.client.delete(&first_url).send().await.unwrap().status(),
        200
    );
    assert_eq!(
        svc.client.get(&first_url).send().await.unwrap().status(),
        404
    );
    create(
        &svc,
        json!({"model":MODEL,"input":"Again?","previous_response_id":second["id"],"store":false}),
    )
    .await;
    let requests = svc.engine.take_requests().await;
    assert_eq!(requests[1].inner.messages.len(), 3);
    assert_eq!(requests[2].inner.messages.len(), 5);
    svc.shutdown().await;
}

#[tokio::test]
async fn streaming_record_matches_terminal_event_and_continues_tool_call() {
    let svc = service(&["fragmented-tool.sse", "text.sse"]).await;
    let response = svc.client.post(format!("{}/v1/responses", svc.base_url)).json(&json!({
        "model":MODEL,"input":"List files", "stream":true,
        "tools":[{"type":"function","name":"list_directory","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}]
    })).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let events = parse_json_sse(&response.text().await.unwrap())
        .await
        .unwrap();
    let terminal = events
        .iter()
        .find(|event| event.event == "response.completed")
        .unwrap();
    let response = &terminal.data["response"];
    let fetched: Value = svc
        .client
        .get(format!(
            "{}/v1/responses/{}",
            svc.base_url,
            response["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(&fetched, response);
    let call = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| item["type"] == "function_call")
        .unwrap();
    create(&svc,json!({"model":MODEL,"previous_response_id":response["id"],"input":[{"type":"function_call_output","call_id":call["call_id"],"output":"[]"}]})).await;
    assert_eq!(svc.engine.take_requests().await[1].inner.messages.len(), 3);
    svc.shutdown().await;
}

#[tokio::test]
async fn stateless_and_credential_scoped_records() {
    let svc = service(&["text.sse", "text.sse"]).await;
    let first = create(&svc, json!({"model":MODEL,"input":"hello","store":false})).await;
    assert_eq!(first["store"], false);
    assert_eq!(
        svc.client
            .get(format!(
                "{}/v1/responses/{}",
                svc.base_url,
                first["id"].as_str().unwrap()
            ))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .bearer_auth("scope-a")
        .json(&json!({"model":MODEL,"input":"hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: Value = response.json().await.unwrap();
    let url = format!(
        "{}/v1/responses/{}",
        svc.base_url,
        response["id"].as_str().unwrap()
    );
    assert_eq!(
        svc.client
            .get(&url)
            .bearer_auth("scope-a")
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        svc.client
            .get(&url)
            .bearer_auth("scope-b")
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(svc.client.delete(&url).send().await.unwrap().status(), 404);
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .bearer_auth("scope-b")
        .json(&json!({"model":MODEL,"input":"hello","previous_response_id":response["id"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(svc.engine.remaining_scripts().await, 0);
    svc.shutdown().await;
}

#[tokio::test]
async fn capacity_failure_never_acknowledges_storage_and_unknown_history_skips_engine() {
    let config = StoreConfig {
        max_entries: 1,
        ..StoreConfig::default()
    };
    let script = load_agent_fixture("text.sse").await.unwrap();
    let svc = HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::new(vec![
            Ok(script.clone()),
            Ok(script),
        ])),
        Some(ResponseStorage::memory(config).unwrap()),
    )
    .await;
    let first = create(&svc, json!({"model":MODEL,"input":"first"})).await;
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"second"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 507);
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"third","previous_response_id":"resp_missing"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(svc.engine.take_requests().await.len(), 2);
    let retained: Value = svc
        .client
        .get(format!(
            "{}/v1/responses/{}",
            svc.base_url,
            first["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(retained, first);
    svc.shutdown().await;
}

async fn poll_terminal(svc: &HarnessService, id: &Value) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let response: Value = svc
                .client
                .get(format!(
                    "{}/v1/responses/{}",
                    svc.base_url,
                    id.as_str().unwrap()
                ))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            if !matches!(response["status"].as_str(), Some("queued" | "in_progress")) {
                return response;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("background response did not terminate")
}

#[tokio::test]
async fn background_outlives_creating_connection() {
    let script = load_agent_fixture("text.sse").await.unwrap();
    let (engine, gate) = ScriptedChatEngine::with_gated_tail(script, 1);
    let svc = HarnessService::start_with_storage(
        Arc::new(engine),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await;
    // Dedicated client closes its connection while the generation is still blocked.
    let client = reqwest::Client::new();
    let queued: Value = client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"hello","background":true}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    drop(client);
    assert_eq!(queued["status"], "queued");
    assert_eq!(queued["background"], true);
    gate.release();
    let completed = poll_terminal(&svc, &queued["id"]).await;
    assert_eq!(completed["status"], "completed", "{completed}");
    assert_eq!(completed["id"], queued["id"]);
    assert_eq!(completed["created_at"], queued["created_at"]);
    assert_eq!(completed["background"], true);
    assert!(!completed["output"].as_array().unwrap().is_empty());
    svc.shutdown().await;
}

#[tokio::test]
async fn background_cancel_stops_backend_and_late_completion_cannot_overwrite() {
    let script = load_agent_fixture("text.sse").await.unwrap();
    let (engine, gate) = ScriptedChatEngine::with_gated_tail(script, 1);
    let svc = HarnessService::start_with_storage(
        Arc::new(engine),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await;
    let queued = create(
        &svc,
        json!({"model":MODEL,"input":"hello","background":true}),
    )
    .await;
    let contexts = tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            let contexts = svc.engine.take_contexts().await;
            if !contexts.is_empty() {
                break contexts;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let cancel_url = format!(
        "{}/v1/responses/{}/cancel",
        svc.base_url,
        queued["id"].as_str().unwrap()
    );
    let response: Value = svc
        .client
        .post(&cancel_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(response["status"], "cancelled");
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !contexts[0].is_killed() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    gate.release();
    assert_eq!(
        poll_terminal(&svc, &queued["id"]).await["status"],
        "cancelled"
    );
    let again: Value = svc
        .client
        .post(&cancel_url)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(again, response);
    svc.shutdown().await;
}

#[tokio::test]
async fn background_rejects_incompatible_flags_before_dispatch() {
    let svc = service(&[]).await;
    for extra in [json!({"store":false}), json!({"stream":true})] {
        let mut request = json!({"model":MODEL,"input":"hello","background":true});
        request
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert_eq!(
            svc.client
                .post(format!("{}/v1/responses", svc.base_url))
                .json(&request)
                .send()
                .await
                .unwrap()
                .status(),
            400
        );
    }
    assert!(svc.engine.take_requests().await.is_empty());
    svc.shutdown().await;
}

#[tokio::test]
async fn background_timeout_becomes_a_retrievable_failure() {
    let script = load_agent_fixture("text.sse").await.unwrap();
    let (engine, _gate) = ScriptedChatEngine::with_gated_tail(script, 1);
    let mut storage = ResponseStorage::memory(StoreConfig::default()).unwrap();
    Arc::get_mut(&mut storage).unwrap().background.timeout = std::time::Duration::from_secs(1);
    let svc = HarnessService::start_with_storage(Arc::new(engine), Some(storage)).await;
    let queued = create(
        &svc,
        json!({"model":MODEL,"input":"hello","background":true}),
    )
    .await;
    let failed = poll_terminal(&svc, &queued["id"]).await;
    assert_eq!(failed["status"], "failed");
    assert!(
        failed["error"]["message"]
            .as_str()
            .unwrap()
            .contains("deadline")
    );
    svc.shutdown().await;
}

#[tokio::test]
async fn deleting_background_record_stops_generation_without_resurrection() {
    let (engine, gate) =
        ScriptedChatEngine::with_gated_tail(load_agent_fixture("text.sse").await.unwrap(), 1);
    let svc = HarnessService::start_with_storage(
        Arc::new(engine),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await;
    let queued = create(
        &svc,
        json!({"model":MODEL,"input":"hello","background":true}),
    )
    .await;
    let url = format!(
        "{}/v1/responses/{}",
        svc.base_url,
        queued["id"].as_str().unwrap()
    );
    assert_eq!(
        svc.client
            .get(format!("{url}?stream=true"))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    assert_eq!(svc.client.delete(&url).send().await.unwrap().status(), 200);
    gate.release();
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(svc.client.get(&url).send().await.unwrap().status(), 404);
    svc.shutdown().await;
}

#[tokio::test]
async fn background_backend_failure_is_stored_and_foreground_cancel_is_rejected() {
    let storage = ResponseStorage::memory(StoreConfig::default()).unwrap();
    let svc = HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::new([
            Err(anyhow::anyhow!("scripted backend failure")),
            Ok(load_agent_fixture("text.sse").await.unwrap()),
        ])),
        Some(storage),
    )
    .await;
    let queued = create(
        &svc,
        json!({"model":MODEL,"input":"hello","background":true}),
    )
    .await;
    let response = poll_terminal(&svc, &queued["id"]).await;
    assert_eq!(response["status"], "failed");
    assert!(response["error"].is_object());
    let foreground = create(&svc, json!({"model":MODEL,"input":"hello"})).await;
    assert_eq!(
        svc.client
            .post(format!(
                "{}/v1/responses/{}/cancel",
                svc.base_url,
                foreground["id"].as_str().unwrap()
            ))
            .send()
            .await
            .unwrap()
            .status(),
        400
    );
    svc.shutdown().await;
}

#[tokio::test]
async fn failed_stream_is_retrievable_with_the_same_partial_output_and_error() {
    use dynamo_runtime::error::{BackendError, DynamoError, ErrorType as DynamoErrorType};
    let mut script = load_agent_fixture("text.sse").await.unwrap();
    let finish = script
        .iter()
        .position(|chunk| {
            chunk.data.as_ref().is_some_and(|data| {
                data.inner
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason.is_some())
            })
        })
        .unwrap();
    script.truncate(finish);
    let error = DynamoError::builder()
        .error_type(DynamoErrorType::Backend(BackendError::InvalidArgument))
        .message("scripted failure")
        .build();
    let svc = HarnessService::start_with_storage(
        Arc::new(ScriptedChatEngine::with_backend_error(script, error)),
        Some(ResponseStorage::memory(StoreConfig::default()).unwrap()),
    )
    .await;
    let response = svc
        .client
        .post(format!("{}/v1/responses", svc.base_url))
        .json(&json!({"model":MODEL,"input":"hello","stream":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let events = parse_json_sse(&response.text().await.unwrap())
        .await
        .unwrap();
    let failed = &events
        .iter()
        .find(|event| event.event == "response.failed")
        .unwrap()
        .data["response"];
    let stored: Value = svc
        .client
        .get(format!(
            "{}/v1/responses/{}",
            svc.base_url,
            failed["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(&stored, failed);
    svc.shutdown().await;
}
