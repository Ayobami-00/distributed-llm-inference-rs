//! Pipeline planning, execution, and report errors.

use thiserror::Error;

/// Result returned by pipeline operations.
pub type Result<T> = std::result::Result<T, PipelineError>;

/// Failure detected while planning or executing a pipeline rank.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// The requested topology or stage assignment is invalid.
    #[error("invalid pipeline topology: {0}")]
    InvalidTopology(String),
    /// A control message did not match the expected phase or step.
    #[error("pipeline protocol error: {0}")]
    Protocol(String),
    /// Stage placement exceeded its enforced memory budget.
    #[error("rank {rank} requires {required_bytes} bytes but its limit is {budget_bytes} bytes")]
    PlacementFailed {
        /// Stage rank that cannot fit.
        rank: usize,
        /// Planned persistent bytes.
        required_bytes: u64,
        /// Enforced per-rank budget.
        budget_bytes: u64,
    },
    /// Runtime model execution failed.
    #[error(transparent)]
    Runtime(#[from] dlir_runtime::DlirError),
    /// Rank communication failed.
    #[error(transparent)]
    Collectives(#[from] dlir_collectives::CollectivesError),
    /// Candle tensor execution failed.
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    /// Control JSON could not be encoded or decoded.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Tokenizer loading or incremental decoding failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}
