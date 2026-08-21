//! The single-process runtime for the distributed LLM inference laboratory.

mod artifacts;
mod error;
mod generation;
mod inspect;
mod memory;
mod model;
mod prompt;
mod registry;
mod report;

pub use error::{DlirError, Result};
pub use generation::{EventObserver, GenerationRequest, NoopObserver, generate};
pub use inspect::{InspectionReport, InspectionRequest, inspect};
pub use memory::{
    BudgetSource, MemoryBudget, MemoryComponentBreakdown, MemoryDomain, PlacementVerdict,
    RankMemoryPlan, format_bytes, parse_byte_size,
};
pub use registry::{
    CheckpointDType, ExecutionSupport, ModelConfig, ModelSpec, PlanDType, PromptTemplate,
    SupportedModelId, TensorLayout, supported_models,
};
pub use report::{
    GenerationReport, RunEvent, RunEventKind, StopReason, TimingReport, TopologyReport,
};
