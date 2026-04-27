//! Error types for the workflow crate.

use std::path::PathBuf;

/// Errors that can occur during workflow loading, expression evaluation,
/// or step execution.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorkflowError {
    /// I/O error while reading workflow files.
    #[error("{context}: {source}")]
    Io {
        /// Description of what was being done.
        context: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Workflow YAML failed to deserialize.
    #[error("yaml parse error in {path}: {source}")]
    YamlParse {
        /// Path to the workflow file.
        path: PathBuf,
        /// The underlying YAML error.
        #[source]
        source: serde_yml::Error,
    },

    /// Workflow definition failed structural validation.
    #[error("invalid workflow {path}: {reason}")]
    InvalidWorkflow {
        /// Path to the workflow file (or "<inline>" for inline parsing).
        path: String,
        /// Reason for the validation failure.
        reason: String,
    },

    /// Expression evaluation failed.
    #[error("expression error: {message}")]
    Expression {
        /// Human-readable error message.
        message: String,
    },

    /// A required input was not provided.
    #[error("missing required input: {name}")]
    MissingInput {
        /// Name of the missing input.
        name: String,
    },

    /// A workflow step failed.
    #[error("step '{step_id}' failed: {message}")]
    StepFailed {
        /// The id of the failing step.
        step_id: String,
        /// Human-readable error message.
        message: String,
    },

    /// Path validation rejected a path (e.g. `..` traversal).
    #[error("invalid path '{path}': {reason}")]
    InvalidPath {
        /// The offending path.
        path: String,
        /// Reason for rejection.
        reason: String,
    },

    /// Sandbox rejected a path or command.
    #[error("sandbox error: {0}")]
    Sandbox(#[from] tmg_sandbox::SandboxError),

    /// LLM error during agent step execution.
    #[error("llm error: {0}")]
    Llm(#[from] tmg_llm::LlmError),

    /// Subagent execution error.
    #[error("agent error: {0}")]
    Agent(#[from] tmg_agents::AgentError),

    /// A workflow with the requested id was not found.
    #[error("workflow not found: {id}")]
    NotFound {
        /// The workflow id that was looked up.
        id: String,
    },
}

impl WorkflowError {
    /// Build an I/O error with descriptive context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    /// Build an `InvalidWorkflow` error with a path-string + reason.
    pub fn invalid_workflow(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidWorkflow {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Build an `Expression` error.
    pub fn expression(message: impl Into<String>) -> Self {
        Self::Expression {
            message: message.into(),
        }
    }
}

/// Convenience alias for `Result<T, WorkflowError>`.
pub type Result<T> = std::result::Result<T, WorkflowError>;
