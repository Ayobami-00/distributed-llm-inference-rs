use crate::{PlanDType, SupportedModelId};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, DlirError>;

#[derive(Debug, Error)]
pub enum DlirError {
    #[error("unsupported model '{0}'; run `dlir models` to list supported models")]
    UnsupportedModel(String),
    #[error("model {model} does not support {dtype} execution on CPU")]
    UnsupportedExecution {
        model: SupportedModelId,
        dtype: PlanDType,
    },
    #[error("invalid memory size '{value}': {reason}")]
    InvalidMemorySize { value: String, reason: String },
    #[error("invalid model configuration: {0}")]
    InvalidConfig(String),
    #[error("downloaded checkpoint does not match the registry: {0}")]
    CheckpointMismatch(String),
    #[error("prompt must not be empty")]
    EmptyPrompt,
    #[error(
        "prompt uses {prompt_tokens} tokens but model context is {max_context}; no room remains for generation"
    )]
    PromptTooLong {
        prompt_tokens: usize,
        max_context: usize,
    },
    #[error("KV cache capacity exceeded: attempted {attempted} tokens with capacity {capacity}")]
    CacheCapacityExceeded { attempted: usize, capacity: usize },
    #[error(
        "placement failed: persistent model state needs {required_bytes} bytes but the budget is {budget_bytes} bytes"
    )]
    PlacementFailed {
        required_bytes: u64,
        budget_bytes: u64,
    },
    #[error("artifact error: {0}")]
    Artifact(String),
    #[error("chat-template error: {0}")]
    PromptTemplate(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
