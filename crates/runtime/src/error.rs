//! Errors returned at the runtime's validation, artifact, tensor, and I/O boundaries.

use crate::{PlanDType, SupportedModelId};
use thiserror::Error;

/// The result type returned by `dlir-runtime` operations.
pub type Result<T> = std::result::Result<T, DlirError>;

/// A failure detected while planning, loading, or executing one inference request.
///
/// Variants preserve the boundary that rejected the request. In particular, registry and
/// checkpoint mismatches fail explicitly rather than surfacing later as ambiguous tensor errors.
#[derive(Debug, Error)]
pub enum DlirError {
    /// The supplied identifier is not present in the closed model registry.
    #[error("unsupported model '{0}'; run `dlir models` to list supported models")]
    UnsupportedModel(String),
    /// The requested dtype is not validated for CPU execution of the selected model.
    #[error("model {model} does not support {dtype} execution on CPU")]
    UnsupportedExecution {
        /// The selected registered model.
        model: SupportedModelId,
        /// The unsupported runtime dtype.
        dtype: PlanDType,
    },
    /// A byte count used an unsupported suffix, form, or numeric range.
    #[error("invalid memory size '{value}': {reason}")]
    InvalidMemorySize {
        /// The original user-supplied value.
        value: String,
        /// A human-readable explanation of the rejected form.
        reason: String,
    },
    /// A model, cache, or generation invariant is invalid.
    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),
    /// Downloaded metadata or safetensors differ from the compiled model specification.
    #[error("downloaded checkpoint does not match the registry: {0}")]
    CheckpointMismatch(String),
    /// The user prompt is empty or contains only whitespace.
    #[error("prompt must not be empty")]
    EmptyPrompt,
    /// Tokenization consumed the model context and left no position for generation.
    #[error(
        "prompt uses {prompt_tokens} tokens but model context is {max_context}; no room remains for generation"
    )]
    PromptTooLong {
        /// Number of tokens in the rendered and encoded prompt.
        prompt_tokens: usize,
        /// Maximum positions declared by the registered model.
        max_context: usize,
    },
    /// A cache append or model forward would exceed its preallocated token capacity.
    #[error("KV cache capacity exceeded: attempted {attempted} tokens with capacity {capacity}")]
    CacheCapacityExceeded {
        /// Populated token count that the operation attempted to reach.
        attempted: usize,
        /// Allocated cache capacity in token positions.
        capacity: usize,
    },
    /// The logical persistent state exceeds the user-declared memory budget.
    #[error(
        "placement failed: persistent model state needs {required_bytes} bytes but the budget is {budget_bytes} bytes"
    )]
    PlacementFailed {
        /// Logical weight plus allocated KV-cache bytes.
        required_bytes: u64,
        /// User-declared host-domain budget in bytes.
        budget_bytes: u64,
    },
    /// Hugging Face API, cache, or artifact-resolution failure.
    #[error("artifact error: {0}")]
    Artifact(String),
    /// Failure while constructing a registered fixed chat prompt.
    #[error("chat-template error: {0}")]
    PromptTemplate(String),
    /// Failure while loading, encoding, or decoding with the tokenizer.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    /// Candle tensor construction or execution failure.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
    /// Local filesystem failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing or serialization failure.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
