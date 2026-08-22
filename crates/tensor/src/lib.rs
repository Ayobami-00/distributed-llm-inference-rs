//! Tensor-parallel planning, reports, manifests, and rank execution.
//!
//! The crate is the orchestration boundary for v0.5. Mathematical tensor-parallel model
//! components remain in `dlir-runtime`; native collective algorithms remain in
//! `dlir-collectives`.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod error;
mod event;
mod plan;
mod report;
mod runner;

pub use error::{Result, TensorParallelError};
pub use event::{NoopTensorEventSink, TensorEventSink};
pub use plan::{TensorParallelMemoryPlan, TensorParallelPartition};
pub use report::{
    TensorParallelManifest, TensorParallelRankReport, TensorParallelRankTimings,
    TensorParallelReport, TensorParallelResourcePlan, TensorParallelStreamRecord,
};
pub use runner::{run_tensor_parallel_rank, run_tensor_parallel_rank_observed};
