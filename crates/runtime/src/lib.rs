//! The single-process runtime for the distributed LLM inference laboratory.
//!
//! `dlir-runtime` owns the complete `v0.1-single` execution path: a closed model registry,
//! artifact validation, prompt rendering, logical memory planning, an owned Llama forward pass,
//! KV-cached greedy generation, structured events, and versioned reports. The crate contains no
//! terminal presentation logic; [`generate`] reports progress through [`EventObserver`] and
//! returns structured data for any caller to present.
//! `v0.4-pipeline` also reuses this crate's [`LlamaStage`] and [`StageKvCache`] APIs so a physical
//! rank can materialize and execute only a contiguous transformer slice without moving Docker or
//! transport concerns into the model implementation.
//!
//! # Inspect without model downloads
//!
//! Inspection uses only embedded registry metadata:
//!
//! ```
//! use dlir_runtime::{InspectionRequest, PlanDType, SupportedModelId, inspect};
//!
//! let report = inspect(&InspectionRequest {
//!     model: SupportedModelId::SmolLm2_135MInstruct,
//!     dtype: PlanDType::F32,
//!     context_length: 512,
//!     device_memory_budget: None,
//! })?;
//! assert_eq!(report.memory.rank, 0);
//! assert_eq!(report.memory.context_length, 512);
//! # Ok::<(), dlir_runtime::DlirError>(())
//! ```
//!
//! # Generate with structured events
//!
//! Generation resolves the selected model's pinned Hugging Face artifacts, so examples are
//! compile-only:
//!
//! ```no_run
//! use dlir_runtime::{
//!     EventObserver, GenerationRequest, PlanDType, RunEvent, SupportedModelId, generate,
//! };
//!
//! struct Observer;
//!
//! impl EventObserver for Observer {
//!     fn on_event(&mut self, event: &RunEvent) {
//!         eprintln!("rank {} event {}", event.rank, event.sequence);
//!     }
//! }
//!
//! let request = GenerationRequest {
//!     model: SupportedModelId::SmolLm2_135MInstruct,
//!     dtype: PlanDType::F32,
//!     prompt: "Explain tensor parallelism.".into(),
//!     max_new_tokens: 32,
//!     device_memory_budget: None,
//! };
//! let report = generate(&request, &mut Observer)?;
//! println!("{}", report.completion);
//! # Ok::<(), dlir_runtime::DlirError>(())
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod artifacts;
mod error;
mod generation;
mod inspect;
mod memory;
mod model;
mod parallel;
mod prompt;
mod registry;
mod report;

pub use artifacts::{
    ArtifactRepository, MetadataArtifacts, validate_checkpoint, validate_metadata,
};
pub use error::{DlirError, Result};
pub use generation::{EventObserver, GenerationRequest, NoopObserver, generate};
pub use inspect::{InspectionReport, InspectionRequest, inspect};
pub use memory::{
    BudgetSource, MemoryBudget, MemoryComponentBreakdown, MemoryDomain, PlacementVerdict,
    RankMemoryPlan, format_bytes, parse_byte_size,
};
pub use model::{
    LayerObserver, LlamaStage, NoopLayerObserver, NoopTensorParallelObserver, StageKvCache,
    TensorParallelKvCache, TensorParallelLlama, TensorParallelObserver,
};
pub use parallel::{
    ColumnParallelLinear, ParallelContext, RowParallelLinear, ShardRange, TensorShardPlan,
    VocabParallelEmbedding, VocabParallelLmHead, plan_tensor_shard,
};
pub use prompt::render_prompt;
pub use registry::{
    CheckpointDType, ExecutionSupport, ModelConfig, ModelSpec, PlanDType, PromptTemplate,
    SupportedModelId, TensorLayout, supported_models,
};
pub use report::{
    CollectiveKind, ControlPurpose, ExecutionPhase, GenerationReport, RunEvent, RunEventKind,
    StopReason, TensorPurpose, TimingReport, TopologyReport,
};
