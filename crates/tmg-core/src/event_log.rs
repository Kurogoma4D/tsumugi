//! Structured event logging for agent loop diagnostics.
//!
//! Provides [`EventLogWriter`] for writing JSON Lines event logs, and
//! [`TeeStreamSink`] for transparently logging events while forwarding
//! them to another [`StreamSink`].

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::agent_loop::StreamSink;
use crate::error::CoreError;

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// A single logged event with elapsed time.
#[derive(Serialize)]
struct EventRecord<'a> {
    elapsed_ms: u128,
    event: &'a EventKind,
}

/// The kind of event that occurred during a conversation turn.
#[derive(Serialize)]
#[serde(tag = "type")]
enum EventKind {
    #[serde(rename = "thinking")]
    Thinking { token: String },
    #[serde(rename = "token")]
    Token { token: String },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "tool_call")]
    ToolCall { name: String, arguments: String },
    #[serde(rename = "tool_result")]
    ToolResult {
        name: String,
        output: String,
        is_error: bool,
    },
    #[serde(rename = "warning")]
    Warning { message: String },
}

// ---------------------------------------------------------------------------
// EventLogWriter
// ---------------------------------------------------------------------------

/// Writes structured events to a JSON Lines file.
pub struct EventLogWriter {
    writer: BufWriter<File>,
    start: Instant,
}

impl EventLogWriter {
    /// Open (or create) a file at `path` for writing events.
    /// Truncates the file if it already exists.
    pub fn new(path: &Path) -> std::io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            start: Instant::now(),
        })
    }

    /// Open a file at `path` in append mode for writing events.
    /// Creates the file if it does not exist.
    pub fn open_append(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            start: Instant::now(),
        })
    }

    /// Log a thinking token event.
    pub fn write_thinking(&mut self, token: &str) {
        let event = EventKind::Thinking {
            token: token.to_owned(),
        };
        self.write_event(&event);
    }

    /// Log a content token event.
    pub fn write_token(&mut self, token: &str) {
        let event = EventKind::Token {
            token: token.to_owned(),
        };
        self.write_event(&event);
    }

    /// Log a tool call event.
    pub fn write_tool_call(&mut self, name: &str, arguments: &str) {
        let event = EventKind::ToolCall {
            name: name.to_owned(),
            arguments: arguments.to_owned(),
        };
        self.write_event(&event);
    }

    /// Log a tool result event.
    pub fn write_tool_result(&mut self, name: &str, output: &str, is_error: bool) {
        let event = EventKind::ToolResult {
            name: name.to_owned(),
            output: output.to_owned(),
            is_error,
        };
        self.write_event(&event);
    }

    /// Log a done event.
    pub fn write_done(&mut self) {
        self.write_event(&EventKind::Done);
    }

    /// Write one event record as a JSON line. Flushes immediately for
    /// real-time observability. Errors are silently ignored so that
    /// logging never disrupts the agent loop.
    fn write_event(&mut self, event: &EventKind) {
        let record = EventRecord {
            elapsed_ms: self.start.elapsed().as_millis(),
            event,
        };
        // Best-effort: ignore serialization and I/O errors.
        if serde_json::to_writer(&mut self.writer, &record).is_ok() {
            let _ = self.writer.write_all(b"\n");
            let _ = self.writer.flush();
        }
    }
}

// ---------------------------------------------------------------------------
// TeeStreamSink
// ---------------------------------------------------------------------------

/// A [`StreamSink`] that logs every event to an [`EventLogWriter`] before
/// forwarding it to an inner sink.
pub struct TeeStreamSink<S> {
    inner: S,
    log: EventLogWriter,
}

impl<S: StreamSink> TeeStreamSink<S> {
    /// Create a new tee that writes events to `log` and forwards to `inner`.
    pub fn new(inner: S, log: EventLogWriter) -> Self {
        Self { inner, log }
    }
}

impl<S: StreamSink> StreamSink for TeeStreamSink<S> {
    fn on_thinking(&mut self, token: &str) -> Result<(), CoreError> {
        self.log.write_thinking(token);
        self.inner.on_thinking(token)
    }

    fn on_token(&mut self, token: &str) -> Result<(), CoreError> {
        self.log.write_token(token);
        self.inner.on_token(token)
    }

    fn on_done(&mut self) -> Result<(), CoreError> {
        self.log.write_done();
        self.inner.on_done()
    }

    fn on_tool_call(
        &mut self,
        call_id: &str,
        name: &str,
        arguments: &str,
    ) -> Result<(), CoreError> {
        self.log.write_tool_call(name, arguments);
        self.inner.on_tool_call(call_id, name, arguments)
    }

    fn on_tool_result(
        &mut self,
        call_id: &str,
        name: &str,
        output: &str,
        is_error: bool,
    ) -> Result<(), CoreError> {
        self.log.write_tool_result(name, output, is_error);
        self.inner.on_tool_result(call_id, name, output, is_error)
    }

    fn on_tool_result_compressed(
        &mut self,
        call_id: &str,
        name: &str,
        symbol_count: usize,
    ) -> Result<(), CoreError> {
        self.inner
            .on_tool_result_compressed(call_id, name, symbol_count)
    }

    fn on_warning(&mut self, message: &str) -> Result<(), CoreError> {
        let event = EventKind::Warning {
            message: message.to_owned(),
        };
        self.log.write_event(&event);
        self.inner.on_warning(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;

    /// A no-op sink for testing the tee.
    struct NullSink;

    impl StreamSink for NullSink {
        fn on_token(&mut self, _token: &str) -> Result<(), CoreError> {
            Ok(())
        }
    }

    #[test]
    fn event_log_writes_jsonl() -> Result<(), Box<dyn std::error::Error>> {
        let dir = std::env::temp_dir().join("tmg-event-log-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.jsonl");

        {
            let log = EventLogWriter::new(&path)?;
            let mut tee = TeeStreamSink::new(NullSink, log);
            tee.on_token("hello")?;
            tee.on_tool_call("call_1", "file_read", r#"{"path":"Cargo.toml"}"#)?;
            tee.on_tool_result("call_1", "file_read", "[workspace]", false)?;
            tee.on_done()?;
        }

        let mut contents = String::new();
        File::open(&path)?.read_to_string(&mut contents)?;

        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 4);

        // Verify each line is valid JSON with expected type field.
        let first: serde_json::Value = serde_json::from_str(lines[0])?;
        assert_eq!(first["event"]["type"], "token");
        assert_eq!(first["event"]["token"], "hello");
        assert!(first["elapsed_ms"].is_number());

        let second: serde_json::Value = serde_json::from_str(lines[1])?;
        assert_eq!(second["event"]["type"], "tool_call");

        let third: serde_json::Value = serde_json::from_str(lines[2])?;
        assert_eq!(third["event"]["type"], "tool_result");

        let fourth: serde_json::Value = serde_json::from_str(lines[3])?;
        assert_eq!(fourth["event"]["type"], "done");

        // Cleanup.
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);

        Ok(())
    }
}
