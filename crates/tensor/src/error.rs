//! Tensor-parallel planning and execution errors.

use thiserror::Error;

/// Result type used by tensor-parallel orchestration.
pub type Result<T> = std::result::Result<T, TensorParallelError>;

/// Failure at a TP plan, artifact, transport, model, or serialization boundary.
#[derive(Debug, Error)]
pub enum TensorParallelError {
    /// Invalid request topology or model dimension.
    #[error("invalid tensor-parallel request: {0}")]
    InvalidRequest(String),
    /// The rank's persistent state exceeds its enforced budget.
    #[error(
        "tensor shard placement failed: rank {rank} needs {required_bytes} bytes but has {budget_bytes} bytes"
    )]
    PlacementFailed {
        /// Rejected global rank.
        rank: usize,
        /// Planned logical persistent bytes.
        required_bytes: u64,
        /// Enforced container memory maximum.
        budget_bytes: u64,
    },
    /// Existing runtime error.
    #[error(transparent)]
    Runtime(#[from] dlir_runtime::DlirError),
    /// Point-to-point or collective error.
    #[error(transparent)]
    Collectives(#[from] dlir_collectives::CollectivesError),
    /// Candle tensor error.
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    /// JSON encoding or decoding error.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Tokenizer loading or decoding error.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}
