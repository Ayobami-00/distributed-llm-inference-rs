//! Correctness-first pipeline partitioning, rank execution, reports, and event streams.
//!
//! The crate is transport-generic and contains no Docker or terminal lifecycle logic. A pipeline
//! rank owns one contiguous model stage, exchanges copied F32 activations, and participates in
//! final-stage token feedback and rank-0 continuation decisions.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod error;
mod event;
mod partition;
mod protocol;
mod report;
mod runner;

pub use error::{PipelineError, Result};
pub(crate) use event::EventRecorder;
pub use event::{NoopPipelineEventSink, PipelineEvent, PipelineEventSink, ReceivedPipelineEvent};
pub use partition::{PipelinePartition, StageAssignment, StageMemoryPlan};
pub use protocol::{PipelineControl, PipelineDecision, activation_tag, decision_tag, token_tag};
pub use report::{
    CommunicationReport, PipelineManifest, PipelineRankReport, PipelineRankTimings, PipelineReport,
    PipelineResourcePlan, PipelineStreamRecord, PipelineTimingReport, ResourceSnapshot,
};
pub use runner::run_pipeline_rank;
