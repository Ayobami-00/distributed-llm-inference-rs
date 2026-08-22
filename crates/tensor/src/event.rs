//! Live observational event sink used by tensor ranks.

use dlir_pipeline::PipelineEvent;

/// Thread-safe consumer of the shared schema-v1 distributed event envelope.
pub trait TensorEventSink: Send + Sync {
    /// Observes one rank event without controlling execution.
    fn publish(&self, event: &PipelineEvent);
}

/// Event sink that discards tensor-rank notifications.
#[derive(Default)]
pub struct NoopTensorEventSink;

impl TensorEventSink for NoopTensorEventSink {
    fn publish(&self, _event: &PipelineEvent) {}
}
