//! Context compression and token counting for managing the LLM context window.
//!
//! This module provides:
//! - [`ContextConfig`]: configuration for context window management
//! - [`TokenCounter`]: async token counting via llama-server's `/tokenize` endpoint
//! - [`ContextCompressor`]: automatic and manual context compression

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use tmg_llm::LlmClient;

use crate::error::CoreError;
use crate::message::Message;

/// Default maximum context tokens.
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 16384;

/// Default compression threshold as a fraction of max context.
const DEFAULT_COMPRESSION_THRESHOLD: f64 = 0.8;

/// Default maximum tokens for a single tool result.
const DEFAULT_MAX_TOOL_RESULT_TOKENS: usize = 4096;

/// Default token threshold above which `file_read` results are
/// rewritten via tree-sitter signature extraction (issue #49).
const DEFAULT_SIGNATURE_THRESHOLD_TOKENS: usize = 1500;

// ---------------------------------------------------------------------------
// ContextConfig
// ---------------------------------------------------------------------------

/// Configuration for context window management.
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum number of tokens in the context window.
    pub max_context_tokens: usize,
    /// Fraction of max context at which compression auto-triggers (0.0..=1.0).
    pub compression_threshold: f64,
    /// Maximum tokens allowed in a single tool result before truncation.
    pub max_tool_result_tokens: usize,
    /// Token threshold above which a `file_read` tool result is replaced
    /// with a tree-sitter signature summary instead of being kept verbatim
    /// (or tail-truncated).
    ///
    /// When the estimated token count of the output exceeds this value
    /// **and** the file extension is recognised by
    /// [`tmg_tools::extract_signatures`], the tool result stored in the
    /// agent loop's history is rewritten as a structural summary. The
    /// tail-truncation path is used for unrecognised file types or when
    /// signature extraction yields no symbols.
    pub signature_threshold_tokens: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            compression_threshold: DEFAULT_COMPRESSION_THRESHOLD,
            max_tool_result_tokens: DEFAULT_MAX_TOOL_RESULT_TOKENS,
            signature_threshold_tokens: DEFAULT_SIGNATURE_THRESHOLD_TOKENS,
        }
    }
}

impl ContextConfig {
    /// The token count at which compression should trigger.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "threshold is clamped to 0..1 and max_context_tokens fits in usize; precision loss is acceptable"
    )]
    pub fn compression_trigger(&self) -> usize {
        let threshold = self.compression_threshold.clamp(0.0, 1.0);
        (self.max_context_tokens as f64 * threshold) as usize
    }
}

// ---------------------------------------------------------------------------
// TokenCounter
// ---------------------------------------------------------------------------

/// Async token counter backed by llama-server's `/tokenize` endpoint.
///
/// Maintains a cached count that is updated asynchronously so TUI rendering
/// is never blocked.
pub struct TokenCounter {
    /// The LLM client used for tokenization requests.
    client: LlmClient,
    /// Cached token count, updated asynchronously.
    cached_count: Arc<AtomicU64>,
    /// Background task set for managing tokenization requests.
    tasks: Arc<Mutex<JoinSet<()>>>,
}

impl std::fmt::Debug for TokenCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCounter")
            .field("cached_count", &self.cached_count.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl TokenCounter {
    /// Create a new token counter using the given LLM client.
    pub fn new(client: LlmClient) -> Self {
        Self {
            client,
            cached_count: Arc::new(AtomicU64::new(0)),
            tasks: Arc::new(Mutex::new(JoinSet::new())),
        }
    }

    /// Return the last known token count.
    ///
    /// This value is updated asynchronously and may lag behind the
    /// actual message count by one turn.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "token counts are always well within usize range"
    )]
    pub fn cached_count(&self) -> usize {
        self.cached_count.load(Ordering::Relaxed) as usize
    }

    /// Request an async token count update for the given messages.
    ///
    /// Spawns a background task to count tokens via the LLM server.
    /// The result updates `cached_count` when complete. Does not block.
    pub async fn request_update(&self, messages: &[Message]) {
        // Serialize all messages into a single string for counting.
        let content = serialize_messages_for_counting(messages);
        let client = self.client.clone();
        let cached = Arc::clone(&self.cached_count);

        let mut tasks = self.tasks.lock().await;
        // Only the latest count matters; abort all prior tasks and drain them.
        tasks.abort_all();
        while tasks.join_next().await.is_some() {}
        tasks.spawn(async move {
            let count = match client.tokenize(&content).await {
                Ok(count) => count,
                Err(_) => {
                    // Fall back to heuristic estimation on API failure.
                    content.len() / 4
                }
            };
            // Token counts always fit in u64.
            cached.store(count as u64, Ordering::Relaxed);
        });
    }

    /// Estimate token count using a simple heuristic (characters / 4).
    ///
    /// Useful when the tokenize API is unavailable.
    #[must_use]
    pub fn estimate_tokens(text: &str) -> usize {
        // Integer division: 4 chars per token heuristic.
        (text.len() / 4).max(1)
    }
}

/// Serialize conversation messages into a string for token counting.
///
/// Approximates the chat template format used by llama-server.
fn serialize_messages_for_counting(messages: &[Message]) -> String {
    let mut buf = String::new();
    for msg in messages {
        let role_tag = match msg.role() {
            tmg_llm::Role::System => "system",
            tmg_llm::Role::User => "user",
            tmg_llm::Role::Assistant => "assistant",
            tmg_llm::Role::Tool => "tool",
        };
        buf.push_str("<|");
        buf.push_str(role_tag);
        buf.push_str("|>\n");
        buf.push_str(msg.content());
        buf.push('\n');

        // Include tool call arguments in the count.
        if let Some(tool_calls) = msg.tool_calls() {
            for tc in tool_calls {
                buf.push_str(&tc.function.name);
                buf.push('(');
                buf.push_str(&tc.function.arguments);
                buf.push_str(")\n");
            }
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

/// Outcome of a single [`ContextCompressor::compress_tool_result`] call.
///
/// Returned alongside the compressed string so the caller (the agent
/// loop wiring on the TUI side) can stamp a "compressed" marker on the
/// matching activity entry without re-deriving the kind from the
/// rewritten text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedToolResult {
    /// The (possibly rewritten) tool result text suitable for storage
    /// in the conversation history.
    pub text: String,
    /// Whether the result was rewritten via tree-sitter signature
    /// extraction. `false` when the input was small enough to keep
    /// verbatim, when the tool was not `file_read`, or when signature
    /// extraction yielded no symbols and tail-truncation took over.
    pub compressed_via_signatures: bool,
    /// Number of symbols extracted when
    /// [`Self::compressed_via_signatures`] is `true`.
    pub symbol_count: usize,
}

/// Context compressor that summarizes old conversation turns to free
/// context window space.
///
/// Compression works by:
/// 1. Identifying old tool-result messages with large outputs
/// 2. Asking the LLM to summarize them (compaction turn)
/// 3. Replacing the originals with shorter summaries
///
/// Compression is cancellable via [`CancellationToken`].
pub struct ContextCompressor {
    /// The LLM client for generating summaries.
    client: LlmClient,
    /// Context-window configuration. Used by
    /// [`Self::compress_tool_result`] to decide when to rewrite a
    /// `file_read` body via tree-sitter signature extraction and at
    /// which token budget tail-truncation kicks in.
    config: ContextConfig,
}

/// The prompt used to ask the LLM to summarize old context.
const COMPACTION_PROMPT: &str = "\
Summarize the following conversation context concisely, preserving \
all key technical details, file paths, code changes, and decisions. \
Remove verbose tool outputs but keep their essential findings. \
Output only the summary, no preamble.";

impl ContextCompressor {
    /// Create a new compressor using the given LLM client and the
    /// default [`ContextConfig`].
    ///
    /// Existing call sites that do not yet need to thread a custom
    /// config through to the compressor keep working unchanged; new
    /// call sites that want signature-based tool-result compression
    /// (issue #49) should use [`Self::with_config`] instead.
    pub fn new(client: LlmClient) -> Self {
        Self::with_config(client, ContextConfig::default())
    }

    /// Create a new compressor with a custom [`ContextConfig`].
    ///
    /// The config is used by [`Self::compress_tool_result`] only; the
    /// LLM-driven [`Self::compress`] path does not consume it.
    pub fn with_config(client: LlmClient, config: ContextConfig) -> Self {
        Self { client, config }
    }

    /// Borrow the active [`ContextConfig`].
    #[must_use]
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }

    /// Compress a single tool result for inclusion in the conversation
    /// history.
    ///
    /// Behaviour, by precedence:
    ///
    /// 1. If `tool_name == "file_read"`, the output's estimated token
    ///    count exceeds [`ContextConfig::signature_threshold_tokens`],
    ///    `params["path"]` is a string, and
    ///    [`tmg_tools::extract_signatures`] returns at least one
    ///    symbol — the output is replaced with a structural summary
    ///    block headed by
    ///    `[file content compressed via tree-sitter — N symbols extracted]`.
    /// 2. Otherwise, the output is passed through
    ///    [`truncate_tool_result`] using
    ///    [`ContextConfig::max_tool_result_tokens`] as the budget. This
    ///    matches the pre-#49 behaviour for non-`file_read` tools and
    ///    `file_read` results below the signature threshold.
    ///
    /// The returned [`CompressedToolResult::compressed_via_signatures`]
    /// flag tells the caller whether path 1 fired so a "compressed via
    /// tree-sitter: N symbols" hint can be surfaced in the TUI activity
    /// pane.
    #[must_use]
    pub fn compress_tool_result(
        &self,
        tool_name: &str,
        params: &serde_json::Value,
        output: &str,
    ) -> CompressedToolResult {
        let threshold = self.config.signature_threshold_tokens;
        let estimated = TokenCounter::estimate_tokens(output);
        if tool_name == "file_read"
            && threshold > 0
            && estimated > threshold
            && let Some(path) = params.get("path").and_then(|v| v.as_str())
            && let Some(sigs) = tmg_tools::extract_signatures(path, output)
            && !sigs.is_empty()
        {
            let symbol_count = sigs.len();
            let formatted = tmg_tools::format_signatures(&sigs);
            let text = format!(
                "[file content compressed via tree-sitter — {symbol_count} symbols extracted]\n\n{formatted}"
            );
            return CompressedToolResult {
                text,
                compressed_via_signatures: true,
                symbol_count,
            };
        }

        // Fallback: tail-truncate at the existing tool-result budget.
        CompressedToolResult {
            text: truncate_tool_result(output, self.config.max_tool_result_tokens),
            compressed_via_signatures: false,
            symbol_count: 0,
        }
    }

    /// Compress the conversation history by summarizing old turns.
    ///
    /// Keeps the system prompt (first message) and the most recent
    /// `preserve_recent` messages intact. Everything in between is
    /// summarized into a single user-role message.
    ///
    /// # Cancel safety
    ///
    /// If the [`CancellationToken`] is cancelled during the LLM
    /// summarization call, the original history is returned unchanged
    /// and [`CoreError::Cancelled`] is returned.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] on LLM communication failure or cancellation.
    pub async fn compress(
        &self,
        history: &[Message],
        cancel: CancellationToken,
        preserve_recent: usize,
    ) -> Result<Vec<Message>, CoreError> {
        // We need at least a system prompt + some messages to compress.
        let min_messages = 1 + preserve_recent + 1; // system + recent + at least 1 to compress
        if history.len() <= min_messages {
            // Nothing to compress.
            return Ok(history.to_vec());
        }

        // Split: [system_prompt] [compressible...] [recent...]
        let system_msg = &history[0];
        let split_point = history.len().saturating_sub(preserve_recent);
        let compressible = &history[1..split_point];
        let recent = &history[split_point..];

        if compressible.is_empty() {
            return Ok(history.to_vec());
        }

        // Build a text representation of the compressible section.
        let mut context_text = String::new();
        for msg in compressible {
            let role_label = match msg.role() {
                tmg_llm::Role::System => "System",
                tmg_llm::Role::User => "User",
                tmg_llm::Role::Assistant => "Assistant",
                tmg_llm::Role::Tool => "Tool result",
            };
            context_text.push_str(role_label);
            context_text.push_str(": ");
            // Truncate very long messages for the summary request.
            let content = msg.content();
            if content.len() > 2000 {
                let head_end = find_floor_char_boundary(content, 1000);
                let tail_start =
                    find_ceil_char_boundary(content, content.len().saturating_sub(500));
                context_text.push_str(&content[..head_end]);
                context_text.push_str("\n...(truncated)...\n");
                context_text.push_str(&content[tail_start..]);
            } else {
                context_text.push_str(content);
            }
            context_text.push_str("\n\n");
        }

        // Ask the LLM to summarize.
        let summary_messages = vec![
            tmg_llm::ChatMessage {
                role: tmg_llm::Role::System,
                content: Some(COMPACTION_PROMPT.to_owned()),
                tool_calls: None,
                tool_call_id: None,
            },
            tmg_llm::ChatMessage {
                role: tmg_llm::Role::User,
                content: Some(context_text),
                tool_calls: None,
                tool_call_id: None,
            },
        ];

        if cancel.is_cancelled() {
            return Err(CoreError::Cancelled);
        }

        let response = tokio::select! {
            () = cancel.cancelled() => {
                return Err(CoreError::Cancelled);
            }
            result = self.client.chat(summary_messages, vec![]) => {
                result.map_err(CoreError::Llm)?
            }
        };

        let summary_text = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("[compression failed: no response]")
            .to_owned();

        // Build the new history: system + summary + recent
        let mut new_history = Vec::with_capacity(2 + recent.len());
        new_history.push(system_msg.clone());
        new_history.push(Message::user(format!(
            "[Context summary from previous conversation]\n\n{summary_text}"
        )));
        new_history.extend_from_slice(recent);

        Ok(new_history)
    }
}

// ---------------------------------------------------------------------------
// Tool result truncation
// ---------------------------------------------------------------------------

/// Truncate a tool result string if it exceeds the token limit.
///
/// Uses heuristic token estimation (chars / 4) for fast, non-blocking
/// truncation. The truncated text preserves the beginning and end with
/// an indicator of how much was removed.
pub fn truncate_tool_result(output: &str, max_tokens: usize) -> String {
    let estimated_tokens = TokenCounter::estimate_tokens(output);
    if estimated_tokens <= max_tokens {
        return output.to_owned();
    }

    // Approximate character budget from token limit (4 chars per token).
    let char_budget = max_tokens * 4;
    let half = char_budget / 2;

    if output.len() <= char_budget {
        return output.to_owned();
    }

    // Find safe char boundaries.
    let start_end = find_floor_char_boundary(output, half);
    let tail_start = find_ceil_char_boundary(output, output.len().saturating_sub(half));
    let truncated = output.len() - start_end - (output.len() - tail_start);

    format!(
        "{}\n\n... ({truncated} bytes truncated, tool result exceeded {max_tokens} token limit) ...\n\n{}",
        &output[..start_end],
        &output[tail_start..],
    )
}

/// Find the largest byte index <= `index` that is a char boundary.
fn find_floor_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Find the smallest byte index >= `index` that is a char boundary.
fn find_ceil_char_boundary(s: &str, index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    let mut i = index;
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Context usage formatting
// ---------------------------------------------------------------------------

/// Format token count for display in the TUI header.
///
/// Produces strings like `ctx: 2.1k/8k` or `ctx: 512/8192`.
#[must_use]
pub fn format_context_usage(current: usize, max: usize) -> String {
    format!(
        "ctx: {}/{}",
        format_token_count(current),
        format_token_count(max)
    )
}

/// Format a token count with k-suffix for values >= 1000.
fn format_token_count(count: usize) -> String {
    if count >= 1000 {
        let whole = count / 1000;
        let frac = (count % 1000) / 100; // single decimal digit
        if frac == 0 {
            format!("{whole}k")
        } else {
            format!("{whole}.{frac}k")
        }
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_config_defaults() {
        let config = ContextConfig::default();
        assert_eq!(config.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS);
        assert!(
            (config.compression_threshold - DEFAULT_COMPRESSION_THRESHOLD).abs() < f64::EPSILON
        );
        assert_eq!(
            config.max_tool_result_tokens,
            DEFAULT_MAX_TOOL_RESULT_TOKENS
        );
        assert_eq!(
            config.signature_threshold_tokens,
            DEFAULT_SIGNATURE_THRESHOLD_TOKENS
        );
    }

    #[test]
    fn compression_trigger_calculation() {
        let config = ContextConfig {
            max_context_tokens: 10_000,
            compression_threshold: 0.8,
            max_tool_result_tokens: 4096,
            signature_threshold_tokens: 1500,
        };
        assert_eq!(config.compression_trigger(), 8000);
    }

    #[test]
    fn compression_trigger_zero_threshold() {
        let config = ContextConfig {
            max_context_tokens: 10_000,
            compression_threshold: 0.0,
            max_tool_result_tokens: 4096,
            signature_threshold_tokens: 1500,
        };
        assert_eq!(config.compression_trigger(), 0);
    }

    #[test]
    fn format_context_usage_small() {
        let result = format_context_usage(512, 8192);
        assert_eq!(result, "ctx: 512/8.1k");
    }

    #[test]
    fn format_context_usage_large() {
        let result = format_context_usage(2100, 8000);
        assert_eq!(result, "ctx: 2.1k/8k");
    }

    #[test]
    fn format_context_usage_exact_k() {
        let result = format_context_usage(2000, 8000);
        assert_eq!(result, "ctx: 2k/8k");
    }

    #[test]
    fn format_token_count_below_1k() {
        assert_eq!(format_token_count(0), "0");
        assert_eq!(format_token_count(999), "999");
    }

    #[test]
    fn format_token_count_at_and_above_1k() {
        assert_eq!(format_token_count(1000), "1k");
        assert_eq!(format_token_count(1500), "1.5k");
        assert_eq!(format_token_count(2100), "2.1k");
        assert_eq!(format_token_count(10000), "10k");
    }

    #[test]
    fn estimate_tokens_basic() {
        assert_eq!(TokenCounter::estimate_tokens("abcd"), 1);
        assert_eq!(TokenCounter::estimate_tokens("a".repeat(400).as_str()), 100);
    }

    #[test]
    fn estimate_tokens_empty() {
        // Empty string should return at least 1.
        assert_eq!(TokenCounter::estimate_tokens(""), 1);
    }

    /// Build a no-network `LlmClient` suitable for unit tests of
    /// `ContextCompressor` paths that do not actually contact the
    /// server (e.g. the tool-result compression path).
    #[expect(
        clippy::expect_used,
        reason = "test-only helper; failure to construct an offline LlmClient indicates a broken test runtime"
    )]
    fn dummy_client() -> tmg_llm::LlmClient {
        let cfg = tmg_llm::LlmClientConfig::new("http://localhost:0", "test-model");
        tmg_llm::LlmClient::new(cfg).expect("offline LlmClient construction must not fail")
    }

    #[test]
    fn compress_tool_result_passthrough_when_short() {
        let cfg = ContextConfig {
            max_context_tokens: 8192,
            compression_threshold: 0.8,
            max_tool_result_tokens: 1024,
            signature_threshold_tokens: 1500,
        };
        let comp = ContextCompressor::with_config(dummy_client(), cfg);
        let params = serde_json::json!({"path": "lib.rs"});
        let out = comp.compress_tool_result("file_read", &params, "fn small() {}");
        assert!(!out.compressed_via_signatures);
        assert_eq!(out.symbol_count, 0);
        assert_eq!(out.text, "fn small() {}");
    }

    #[test]
    fn compress_tool_result_rewrites_large_file_read() {
        let cfg = ContextConfig {
            max_context_tokens: 8192,
            compression_threshold: 0.8,
            max_tool_result_tokens: 1024,
            // Set the threshold low so a tiny synthetic body trips it.
            signature_threshold_tokens: 4,
        };
        let comp = ContextCompressor::with_config(dummy_client(), cfg);
        let params = serde_json::json!({"path": "lib.rs"});
        let body = "pub fn alpha(x: i32) -> i32 { x }\npub struct Beta;\n";
        let out = comp.compress_tool_result("file_read", &params, body);
        assert!(out.compressed_via_signatures);
        assert!(out.symbol_count >= 2);
        assert!(out.text.contains("compressed via tree-sitter"));
        assert!(out.text.contains("alpha"));
        assert!(out.text.contains("Beta"));
    }

    #[test]
    fn compress_tool_result_falls_back_for_non_file_read() {
        let cfg = ContextConfig {
            max_context_tokens: 8192,
            compression_threshold: 0.8,
            // Force tail-truncation on the long output below.
            max_tool_result_tokens: 8,
            signature_threshold_tokens: 4,
        };
        let comp = ContextCompressor::with_config(dummy_client(), cfg);
        let params = serde_json::json!({});
        let body = "x".repeat(10_000);
        let out = comp.compress_tool_result("shell_exec", &params, &body);
        assert!(!out.compressed_via_signatures);
        assert!(out.text.contains("truncated"));
    }

    #[test]
    fn compress_tool_result_falls_back_for_unknown_extension() {
        let cfg = ContextConfig {
            max_context_tokens: 8192,
            compression_threshold: 0.8,
            max_tool_result_tokens: 8,
            signature_threshold_tokens: 4,
        };
        let comp = ContextCompressor::with_config(dummy_client(), cfg);
        let params = serde_json::json!({"path": "data.bin"});
        let body = "x".repeat(10_000);
        let out = comp.compress_tool_result("file_read", &params, &body);
        assert!(!out.compressed_via_signatures);
        assert!(out.text.contains("truncated"));
    }

    #[test]
    fn truncate_tool_result_short() {
        let short = "hello world";
        let result = truncate_tool_result(short, 4096);
        assert_eq!(result, short);
    }

    #[test]
    fn truncate_tool_result_long() {
        let long = "x".repeat(100_000);
        let result = truncate_tool_result(&long, 100);
        assert!(result.len() < long.len());
        assert!(result.contains("truncated"));
        assert!(result.contains("100 token limit"));
    }

    #[test]
    fn serialize_messages_includes_tool_calls() {
        let tc = tmg_llm::ToolCall {
            id: "call_1".to_owned(),
            kind: tmg_llm::ToolKind::Function,
            function: tmg_llm::FunctionCall {
                name: "file_read".to_owned(),
                arguments: r#"{"path":"/tmp"}"#.to_owned(),
            },
        };
        let msg = Message::assistant_with_tool_calls("thinking", vec![tc]);
        let serialized = serialize_messages_for_counting(&[msg]);
        assert!(serialized.contains("file_read"));
        assert!(serialized.contains("/tmp"));
    }

    #[test]
    fn serialize_messages_basic() {
        let msgs = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there"),
        ];
        let serialized = serialize_messages_for_counting(&msgs);
        assert!(serialized.contains("<|system|>"));
        assert!(serialized.contains("<|user|>"));
        assert!(serialized.contains("<|assistant|>"));
        assert!(serialized.contains("You are helpful"));
        assert!(serialized.contains("Hello"));
        assert!(serialized.contains("Hi there"));
    }
}
