//! SPEC §8.7 invariant: each agent step starts a *fresh* subagent
//! conversation, with no message history carried across steps.
//!
//! This test runs a workflow with two sequential agent steps against
//! a mock SSE LLM server that captures every chat-completions request
//! body. After the run, we assert that the second step's payload
//! contains exactly one `user`-role message — i.e. no leakage from the
//! first step's prompt or completion into the second step's history.

#![expect(clippy::unwrap_used, reason = "test assertions")]
#![expect(clippy::expect_used, reason = "test assertions")]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::{Mutex, mpsc};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

use tmg_agents::SubagentManager;
use tmg_llm::{LlmPool, PoolConfig};
use tmg_sandbox::{SandboxConfig, SandboxContext, SandboxMode};
use tmg_tools::ToolRegistry;
use tmg_workflow::{WorkflowConfig, WorkflowEngine, parse_workflow_str};

/// A minimal HTTP/1.1 mock that:
///   - Accepts `POST /v1/chat/completions`
///   - Reads the request body up to `Content-Length`
///   - Captures the body for later inspection
///   - Returns an SSE response that yields a single content delta
///     ("ok") followed by `[DONE]`.
///
/// One client connection per request — no keep-alive, no
/// persistence. Adequate for capturing the two POSTs the workflow
/// will issue.
struct MockLlm {
    addr: SocketAddr,
    captured: Arc<Mutex<Vec<String>>>,
}

impl MockLlm {
    async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let captured_clone = Arc::clone(&captured);

        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let captured = Arc::clone(&captured_clone);
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 16 * 1024];
                    let mut total = Vec::new();

                    // Read until we have a complete request: parse
                    // headers, find Content-Length, then keep reading
                    // until total body length is satisfied.
                    let body = loop {
                        match sock.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => {
                                total.extend_from_slice(&buf[..n]);
                                if let Some(idx) = find_subsequence(&total, b"\r\n\r\n") {
                                    let header = &total[..idx];
                                    let body_start = idx + 4;
                                    let header_str = std::str::from_utf8(header).unwrap_or("");
                                    let content_length =
                                        parse_content_length(header_str).unwrap_or(0);
                                    if total.len() - body_start >= content_length {
                                        break total[body_start..body_start + content_length]
                                            .to_vec();
                                    }
                                    // need more bytes
                                }
                            }
                        }
                    };

                    let body_str = String::from_utf8_lossy(&body).to_string();
                    {
                        let mut guard = captured.lock().await;
                        guard.push(body_str);
                    }

                    // Build a tiny SSE stream:
                    //   data: {chunk with content "ok"}
                    //   data: [DONE]
                    let chunk = serde_json::json!({
                        "id": "mock",
                        "object": "chat.completion.chunk",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": "ok"},
                            "finish_reason": null
                        }]
                    });
                    let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                    let len = body.len();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}"
                    );
                    let _ = sock.write_all(response.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });

        Self { addr, captured }
    }

    fn endpoint(&self) -> String {
        format!("http://{}", self.addr)
    }

    async fn captured(&self) -> Vec<String> {
        self.captured.lock().await.clone()
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_content_length(headers: &str) -> Option<usize> {
    for line in headers.lines() {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            return rest.trim().parse().ok();
        }
    }
    None
}

/// Run a two-step agent workflow and verify the second step's payload
/// contains exactly one `user` turn — no history leakage.
#[tokio::test]
async fn two_sequential_agent_steps_have_isolated_histories() {
    let mock = MockLlm::start().await;
    let endpoint = mock.endpoint();

    let tmp = tempfile::tempdir().unwrap();
    let workspace = tmp.path().to_path_buf();

    let llm_pool = Arc::new(LlmPool::new(&PoolConfig::single(&endpoint), "mock-model").unwrap());
    let sandbox = Arc::new(SandboxContext::new(
        SandboxConfig::new(workspace).with_mode(SandboxMode::WorkspaceWrite),
    ));
    let tool_registry = Arc::new(ToolRegistry::new());

    let llm_client_cfg = tmg_llm::LlmClientConfig::new(&endpoint, "mock-model");
    let llm_client = tmg_llm::LlmClient::new(llm_client_cfg).unwrap();
    let cancel = CancellationToken::new();
    let manager = SubagentManager::new(
        llm_client,
        cancel,
        &endpoint,
        "mock-model",
        Arc::clone(&sandbox),
    );
    let subagent_manager = Arc::new(Mutex::new(manager));

    let engine = WorkflowEngine::new(
        llm_pool,
        sandbox,
        tool_registry,
        subagent_manager,
        WorkflowConfig::default(),
        serde_json::Value::Null,
    );

    let yaml = r#"
id: two_step
steps:
  - id: first
    type: agent
    subagent: explore
    prompt: "first task"
  - id: second
    type: agent
    subagent: explore
    prompt: "second task"
"#;
    let wf = parse_workflow_str(yaml, "<inline>").unwrap();
    let (tx, _rx) = mpsc::channel(32);

    // Run with a generous timeout so the test fails clearly if the
    // mock server hangs.
    let outputs = timeout(
        Duration::from_secs(20),
        engine.run(&wf, BTreeMap::new(), tx),
    )
    .await
    .expect("workflow run timed out")
    .expect("workflow run failed");
    assert!(outputs.values.is_empty());

    let payloads = mock.captured().await;
    // Each step issues one chat-completions POST in its single round
    // (no tool calls in the canned response). We expect exactly two
    // bodies overall.
    assert_eq!(
        payloads.len(),
        2,
        "expected 2 captured chat-completions bodies, got {}: {:?}",
        payloads.len(),
        payloads,
    );

    let second: serde_json::Value = serde_json::from_str(&payloads[1]).expect("body 2 is JSON");
    let messages = second
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages is an array");

    // Count user-role turns. SPEC §8.7 invariant: a fresh agent step
    // sees only the system prompt + the new task — i.e. exactly one
    // user-role message, with no history from step 1.
    let user_turns: Vec<&serde_json::Value> = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
        .collect();
    assert_eq!(
        user_turns.len(),
        1,
        "expected exactly 1 user turn in step 2's payload, got {}: {:?}",
        user_turns.len(),
        messages,
    );
    // And that turn must be the *second* prompt, not the first.
    let user_content = user_turns[0]
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("");
    assert!(
        user_content.contains("second task"),
        "expected step 2's user turn to mention 'second task', got: {user_content}",
    );
    assert!(
        !user_content.contains("first task"),
        "step 2's user turn must not contain step 1's prompt; got: {user_content}",
    );
    // No assistant-role turns from step 1 should leak in.
    let assistant_turns = messages
        .iter()
        .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .count();
    assert_eq!(
        assistant_turns, 0,
        "step 2's payload must not carry assistant turns from step 1: {messages:?}",
    );
}
