//! Rank-local event publication and host receive ordering.

use dlir_runtime::{RunEvent, RunEventKind};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Schema-versioned event published by one pipeline rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineEvent {
    /// Event-stream schema version; currently `1`.
    pub schema_version: u32,
    /// Stable Docker/TCP run identity.
    pub run_id: String,
    /// Identity of the generation request within the run.
    pub request_id: String,
    /// Existing rank-aware runtime event payload.
    pub event: RunEvent,
}

/// Event annotated with deterministic host receive order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceivedPipelineEvent {
    /// Zero-based order in which the launcher accepted the event.
    pub receive_sequence: u64,
    /// Rank-published event.
    pub published: PipelineEvent,
}

/// Synchronous observational consumer of rank events.
pub trait PipelineEventSink {
    /// Consumes one event without controlling rank execution.
    fn publish(&mut self, event: &PipelineEvent);
}

/// Event sink that discards live notifications.
#[derive(Default)]
pub struct NoopPipelineEventSink;

impl PipelineEventSink for NoopPipelineEventSink {
    fn publish(&mut self, _event: &PipelineEvent) {}
}

pub(crate) struct EventRecorder<'a> {
    start: Instant,
    rank: usize,
    sequence: u64,
    run_id: &'a str,
    request_id: &'a str,
    sink: &'a mut dyn PipelineEventSink,
    pub(crate) events: Vec<PipelineEvent>,
}

impl<'a> EventRecorder<'a> {
    pub(crate) fn new(
        rank: usize,
        run_id: &'a str,
        request_id: &'a str,
        sink: &'a mut dyn PipelineEventSink,
    ) -> Self {
        Self {
            start: Instant::now(),
            rank,
            sequence: 0,
            run_id,
            request_id,
            sink,
            events: Vec::new(),
        }
    }

    pub(crate) fn emit(&mut self, kind: RunEventKind) {
        let event = PipelineEvent {
            schema_version: 1,
            run_id: self.run_id.to_owned(),
            request_id: self.request_id.to_owned(),
            event: RunEvent {
                sequence: self.sequence,
                rank: self.rank,
                elapsed_ns: self.start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                event: kind,
            },
        };
        self.sequence += 1;
        self.sink.publish(&event);
        self.events.push(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Sink(Vec<PipelineEvent>);

    impl PipelineEventSink for Sink {
        fn publish(&mut self, event: &PipelineEvent) {
            self.0.push(event.clone());
        }
    }

    #[test]
    fn recorder_preserves_rank_local_sequence_and_observer_identity() {
        let mut sink = Sink::default();
        let mut recorder = EventRecorder::new(2, "run", "request", &mut sink);
        recorder.emit(RunEventKind::ModelLoadStarted);
        recorder.emit(RunEventKind::ModelLoadFinished);
        assert_eq!(recorder.events.len(), 2);
        assert_eq!(recorder.events[0].event.sequence, 0);
        assert_eq!(recorder.events[1].event.sequence, 1);
        assert!(recorder.events.iter().all(|event| {
            event.schema_version == 1
                && event.run_id == "run"
                && event.request_id == "request"
                && event.event.rank == 2
        }));
        drop(recorder);
        assert_eq!(sink.0.len(), 2);
    }
}
