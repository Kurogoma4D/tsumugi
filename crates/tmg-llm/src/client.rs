//! SSE streaming client for OpenAI-compatible chat completions.

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use eventsource_stream::Eventsource;
use futures_core::Stream;
use reqwest::Client;
use tokio_util::sync::CancellationToken;

use crate::error::LlmError;
use crate::types::{
    ChatCompletionChunk, ChatRequest, ChatResponse, StreamEvent, ToolCallAccumulator,
};

/// Default connection timeout in seconds.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Default overall request timeout in seconds.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 300;

/// Configuration for building an [`LlmClient`].
#[derive(Debug, Clone)]
pub struct LlmClientConfig {
    /// Base URL of the LLM server (e.g. `http://localhost:8080`).
    pub endpoint: String,

    /// Model name to use in requests.
    pub model: String,

    /// Connection timeout.
    pub connect_timeout: Duration,

    /// Overall request timeout.
    pub request_timeout: Duration,
}

impl LlmClientConfig {
    /// Create a new configuration with the given endpoint and model.
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            model: model.into(),
            connect_timeout: Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS),
            request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
        }
    }

    /// Set the connection timeout.
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// Set the overall request timeout.
    #[must_use]
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }
}

/// Client for communicating with an OpenAI-compatible LLM server.
#[derive(Debug, Clone)]
pub struct LlmClient {
    http: Client,
    config: LlmClientConfig,
}

impl LlmClient {
    /// Create a new client from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `LlmError::Http` if the underlying HTTP client cannot be built.
    pub fn new(config: LlmClientConfig) -> Result<Self, LlmError> {
        let http = Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .build()?;

        Ok(Self { http, config })
    }

    /// The chat completions endpoint URL.
    fn completions_url(&self) -> String {
        let base = self.config.endpoint.trim_end_matches('/');
        format!("{base}/v1/chat/completions")
    }

    /// Send a non-streaming chat completion request.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError`] on connection failure, timeout, or invalid response.
    pub async fn chat(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Vec<crate::types::ToolDefinition>,
    ) -> Result<ChatResponse, LlmError> {
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: false,
            tools,
            temperature: None,
        };

        let response = self
            .http
            .post(self.completions_url())
            .json(&request)
            .send()
            .await
            .map_err(classify_reqwest_error)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ServerError {
                status: status.as_u16(),
                body,
            });
        }

        response
            .json::<ChatResponse>()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))
    }

    /// Send a streaming chat completion request.
    ///
    /// Returns a [`ChatStream`] that yields [`StreamEvent`]s. The stream
    /// respects the provided [`CancellationToken`] and will terminate with
    /// [`LlmError::Cancelled`] if the token is cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError`] if the initial HTTP request fails.
    ///
    /// # Cancel safety
    ///
    /// The returned stream is cancel-safe: dropping it at any point will not
    /// lose data or leave the connection in an inconsistent state.
    pub async fn chat_streaming(
        &self,
        messages: Vec<crate::types::ChatMessage>,
        tools: Vec<crate::types::ToolDefinition>,
        cancel: CancellationToken,
    ) -> Result<ChatStream, LlmError> {
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
            tools,
            temperature: None,
        };

        let response = self
            .http
            .post(self.completions_url())
            .json(&request)
            .send()
            .await
            .map_err(classify_reqwest_error)?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::ServerError {
                status: status.as_u16(),
                body,
            });
        }

        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        Ok(ChatStream {
            inner: Box::pin(event_stream),
            accumulator: ToolCallAccumulator::new(),
            cancel,
            finished: false,
            pending: VecDeque::new(),
        })
    }
}

/// A stream of [`StreamEvent`]s from a chat completion request.
///
/// This stream assembles incremental tool call deltas into complete
/// [`ToolCall`](crate::types::ToolCall)s automatically.
pub struct ChatStream {
    inner: Pin<
        Box<
            dyn Stream<
                    Item = Result<
                        eventsource_stream::Event,
                        eventsource_stream::EventStreamError<reqwest::Error>,
                    >,
                > + Send,
        >,
    >,
    accumulator: ToolCallAccumulator,
    cancel: CancellationToken,
    finished: bool,
    /// Buffered events to yield before polling the inner stream again.
    pending: VecDeque<Result<StreamEvent, LlmError>>,
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStream")
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

impl Stream for ChatStream {
    type Item = Result<StreamEvent, LlmError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Drain any buffered events first.
        if let Some(event) = self.pending.pop_front() {
            return Poll::Ready(Some(event));
        }

        if self.finished {
            return Poll::Ready(None);
        }

        // Check cancellation.
        if self.cancel.is_cancelled() {
            self.finished = true;
            return Poll::Ready(Some(Err(LlmError::Cancelled)));
        }

        loop {
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.finished = true;
                    // Flush any remaining tool calls.
                    let remaining = self.accumulator.finish();
                    for tc in remaining {
                        self.pending
                            .push_back(Ok(StreamEvent::ToolCallComplete(tc)));
                    }
                    return if let Some(event) = self.pending.pop_front() {
                        Poll::Ready(Some(event))
                    } else {
                        Poll::Ready(None)
                    };
                }
                Poll::Ready(Some(Err(e))) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(LlmError::StreamError(e.to_string()))));
                }
                Poll::Ready(Some(Ok(event))) => {
                    self.process_sse_event(&event);

                    // If processing produced any events, return the first one.
                    if let Some(event) = self.pending.pop_front() {
                        return Poll::Ready(Some(event));
                    }

                    // No actionable content (e.g. role-only delta); loop to
                    // poll the inner stream for the next SSE event.
                }
            }
        }
    }
}

impl ChatStream {
    /// Process a single SSE event, pushing results into the pending buffer.
    fn process_sse_event(&mut self, event: &eventsource_stream::Event) {
        let data = &event.data;

        // The stream ends with a [DONE] sentinel.
        if data == "[DONE]" {
            self.finished = true;
            let remaining = self.accumulator.finish();
            for tc in remaining {
                self.pending
                    .push_back(Ok(StreamEvent::ToolCallComplete(tc)));
            }
            self.pending.push_back(Ok(StreamEvent::Done(None)));
            return;
        }

        // Parse the chunk.
        let chunk = match serde_json::from_str::<ChatCompletionChunk>(data) {
            Ok(c) => c,
            Err(e) => {
                self.pending
                    .push_back(Err(LlmError::InvalidResponse(format!(
                        "failed to parse SSE chunk: {e}"
                    ))));
                return;
            }
        };

        // Process each choice (typically just one).
        for choice in &chunk.choices {
            // Handle tool call deltas.
            if let Some(tc_deltas) = &choice.delta.tool_calls {
                let completed = self.accumulator.feed(tc_deltas);
                for tc in completed {
                    self.pending
                        .push_back(Ok(StreamEvent::ToolCallComplete(tc)));
                }
            }

            // Handle content delta.
            if let Some(content) = &choice.delta.content {
                if !content.is_empty() {
                    self.pending
                        .push_back(Ok(StreamEvent::ContentDelta(content.clone())));
                }
            }

            // Handle finish reason.
            if let Some(reason) = &choice.finish_reason {
                self.finished = true;
                let remaining = self.accumulator.finish();
                for tc in remaining {
                    self.pending
                        .push_back(Ok(StreamEvent::ToolCallComplete(tc)));
                }
                self.pending
                    .push_back(Ok(StreamEvent::Done(Some(reason.clone()))));
            }
        }
    }
}

/// Classify a `reqwest::Error` into an appropriate `LlmError` variant.
fn classify_reqwest_error(e: reqwest::Error) -> LlmError {
    if e.is_connect() {
        LlmError::ConnectionFailed(e.to_string())
    } else if e.is_timeout() {
        LlmError::Timeout
    } else {
        LlmError::Http(e)
    }
}
