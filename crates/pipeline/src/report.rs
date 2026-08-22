//! Pipeline request manifests, rank results, and aggregate schema-v1 reports.

use crate::{PipelineEvent, PipelinePartition, ReceivedPipelineEvent, StageMemoryPlan};
use dlir_collectives::PeerInfo;
use dlir_runtime::{PlanDType, StopReason, SupportedModelId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Resource limits observed inside one rank container.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// CPU quota expressed in thousandths of one CPU.
    pub cpu_millis: Option<u64>,
    /// Current cgroup memory usage.
    pub memory_current_bytes: Option<u64>,
    /// Enforced cgroup memory maximum.
    pub memory_limit_bytes: Option<u64>,
    /// Effective cpuset string, when exposed.
    pub cpuset_cpus: Option<String>,
    /// Detected cgroup version.
    pub cgroup_version: Option<String>,
}

/// Host Docker capacity and equal per-rank allocation used for pipeline placement.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineResourcePlan {
    /// Docker Engine CPU capacity in millicpus.
    pub engine_cpu_millis: u64,
    /// Docker Engine memory capacity.
    pub engine_memory_bytes: u64,
    /// User-requested total CPU quota.
    pub requested_cpu_millis: u64,
    /// User-requested total memory.
    pub requested_memory_bytes: u64,
    /// Equal CPU quota assigned to every rank.
    pub per_rank_cpu_millis: u64,
    /// Equal whole-MiB memory maximum assigned to every rank.
    pub per_rank_memory_bytes: u64,
    /// Requested CPU not allocated after integer division.
    pub unused_cpu_millis: u64,
    /// Requested memory not allocated after whole-MiB division.
    pub unused_memory_bytes: u64,
}

/// Immutable request data bind-mounted into rank containers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineManifest {
    /// Manifest schema version; currently `1`.
    pub schema_version: u32,
    /// Stable topology run identity.
    pub run_id: String,
    /// Stable request identity within the run.
    pub request_id: String,
    /// Closed-registry model ID.
    pub model: SupportedModelId,
    /// CPU runtime dtype; F32 in v0.4.
    pub dtype: PlanDType,
    /// Tokenized fixed-template prompt.
    pub prompt_token_ids: Vec<u32>,
    /// Unicode scalar count in the original user prompt.
    pub prompt_characters: usize,
    /// User-requested output-token limit.
    pub requested_max_new_tokens: usize,
    /// Output-token limit after available-context clipping.
    pub effective_max_new_tokens: usize,
    /// Prompt plus effective output capacity.
    pub context_capacity: usize,
    /// Container-visible checkpoint path.
    pub checkpoint_path: PathBuf,
    /// Container-visible tokenizer path.
    pub tokenizer_path: PathBuf,
    /// Ordered stage partition.
    pub partition: PipelinePartition,
    /// Rank-ordered persistent-state plans.
    pub memory_plans: Vec<StageMemoryPlan>,
    /// Expected equal CPU quota per rank.
    pub expected_cpu_millis: u64,
    /// Expected equal enforced memory limit per rank.
    pub expected_memory_bytes: u64,
}

/// Per-rank communication counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunicationReport {
    /// Tensor messages sent.
    pub tensor_messages_sent: u64,
    /// Tensor messages received.
    pub tensor_messages_received: u64,
    /// Logical tensor payload bytes sent.
    pub tensor_bytes_sent: u64,
    /// Logical tensor payload bytes received.
    pub tensor_bytes_received: u64,
    /// Control messages sent.
    pub control_messages_sent: u64,
    /// Control messages received.
    pub control_messages_received: u64,
    /// Control payload bytes sent.
    pub control_bytes_sent: u64,
    /// Control payload bytes received.
    pub control_bytes_received: u64,
    /// Total measured tensor/control communication nanoseconds.
    pub communication_ns: u64,
}

impl CommunicationReport {
    /// Returns all logical tensor and control bytes handled by this rank.
    pub fn total_bytes(&self) -> u64 {
        self.tensor_bytes_sent
            + self.tensor_bytes_received
            + self.control_bytes_sent
            + self.control_bytes_received
    }
}

/// Rank-local pipeline timing aggregates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelineRankTimings {
    /// Stage weight loading and cache construction.
    pub model_load_ns: u64,
    /// Rank-local end-to-end prefill phase.
    pub prefill_ns: u64,
    /// Sum of rank-local decode phases.
    pub decode_total_ns: u64,
    /// Sum of synchronized local transformer-layer compute durations.
    pub layer_compute_ns: u64,
    /// Number of cached decode forwards.
    pub decode_forward_count: usize,
    /// Rank-0 time through first token feedback.
    pub time_to_first_token_ns: Option<u64>,
    /// Entire rank request from entry through completion barrier.
    pub total_ns: u64,
}

/// Final result emitted by one physical pipeline rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRankReport {
    /// Report schema version; currently `1`.
    pub schema_version: u32,
    /// Run identity.
    pub run_id: String,
    /// Request identity.
    pub request_id: String,
    /// Global rank.
    pub rank: usize,
    /// Ordered TCP peer table established through rendezvous.
    pub peers: Vec<PeerInfo>,
    /// Rank's stage assignment.
    pub assignment: crate::StageAssignment,
    /// Rank's persistent-state plan.
    pub memory: StageMemoryPlan,
    /// Resource limits and usage observed in the rank.
    pub resources: ResourceSnapshot,
    /// Final populated local KV-cache bytes.
    pub final_kv_cache_bytes: u64,
    /// Rank-local communication counters.
    pub communication: CommunicationReport,
    /// Rank-local timing aggregates.
    pub timings: PipelineRankTimings,
    /// Non-EOS tokens, populated by rank 0.
    pub generated_tokens: Vec<u32>,
    /// Final decoded completion, populated by rank 0.
    pub completion: String,
    /// Successful termination condition.
    pub stop_reason: StopReason,
    /// Whether startup and completion barriers both passed.
    pub barriers_passed: bool,
    /// Complete event stream published by this rank.
    pub events: Vec<PipelineEvent>,
    /// Rank-local correctness verdict.
    pub success: bool,
}

/// Aggregate timing values calculated by the host launcher.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineTimingReport {
    /// Host preflight, artifact resolution, and validation duration.
    pub artifact_resolution_ns: u64,
    /// Rank-0 prefill through first-token feedback.
    pub prefill_ns: u64,
    /// Rank-0 time to first generated token.
    pub time_to_first_token_ns: u64,
    /// Sum of rank-0 decode-step durations.
    pub decode_total_ns: u64,
    /// Count of cached decode forwards.
    pub decode_forward_count: usize,
    /// Mean rank-0 decode-step duration.
    pub mean_decode_ns: Option<u64>,
    /// Host command duration including artifacts and containers.
    pub cold_start_total_ns: u64,
}

/// Complete schema-v1 result of one Docker pipeline generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineReport {
    /// Report schema version; currently `1`.
    pub schema_version: u32,
    /// Run identity.
    pub run_id: String,
    /// Request identity.
    pub request_id: String,
    /// Closed-registry model ID.
    pub model: SupportedModelId,
    /// Pinned repository.
    pub repository: String,
    /// Pinned revision.
    pub revision: String,
    /// Execution device.
    pub device: String,
    /// Runtime dtype.
    pub dtype: PlanDType,
    /// TCP backend identifier.
    pub backend: String,
    /// World and pipeline size.
    pub world_size: usize,
    /// Tensor-parallel size; one in v0.4.
    pub tensor_parallel: usize,
    /// Pipeline-parallel size; equal to `world_size`.
    pub pipeline_parallel: usize,
    /// Expert-parallel size; one in v0.4.
    pub expert_parallel: usize,
    /// Stage assignment.
    pub partition: PipelinePartition,
    /// Original architectural parameter count.
    pub model_parameters: u64,
    /// Sum of physically materialized parameters across ranks.
    pub materialized_parameters: u64,
    /// Cross-rank parameter copies beyond the architectural model count.
    pub duplicated_parameters: u64,
    /// Docker capacity and equal resource allocation.
    pub resources: PipelineResourcePlan,
    /// Prompt tokens after fixed chat templating.
    pub prompt_tokens: usize,
    /// User-requested output-token limit.
    pub requested_max_new_tokens: usize,
    /// Generated non-EOS token IDs.
    pub generated_tokens: Vec<u32>,
    /// Final decoded assistant text.
    pub completion: String,
    /// Successful termination condition.
    pub stop_reason: StopReason,
    /// Rank-ordered reports.
    pub ranks: Vec<PipelineRankReport>,
    /// Host-received event order across rank-local clocks.
    pub events: Vec<ReceivedPipelineEvent>,
    /// Aggregate timing values.
    pub timings: PipelineTimingReport,
    /// Logical tensor/control traffic summed from sends only.
    pub communication_bytes: u64,
    /// Rank, container, stream, or aggregation failures.
    pub failures: Vec<String>,
    /// Overall correctness and lifecycle verdict.
    pub success: bool,
}

/// One line in a rank's stdout event/result stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum PipelineStreamRecord {
    /// Live rank event.
    Event {
        /// Published event.
        event: PipelineEvent,
    },
    /// Final rank result.
    Result {
        /// Rank report.
        result: Box<PipelineRankReport>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PipelinePartition;

    #[test]
    fn aggregate_report_round_trips_with_an_independent_schema() {
        let model = SupportedModelId::SmolLm2_135MInstruct;
        let report = PipelineReport {
            schema_version: 1,
            run_id: "run".into(),
            request_id: "request".into(),
            model,
            repository: model.spec().repository.into(),
            revision: model.spec().revision.into(),
            device: "cpu".into(),
            dtype: PlanDType::F32,
            backend: "tcp".into(),
            world_size: 2,
            tensor_parallel: 1,
            pipeline_parallel: 2,
            expert_parallel: 1,
            partition: PipelinePartition::balanced(model.spec(), 2).unwrap(),
            model_parameters: model.spec().expected_parameters,
            materialized_parameters: model.spec().expected_parameters,
            duplicated_parameters: 0,
            resources: PipelineResourcePlan::default(),
            prompt_tokens: 4,
            requested_max_new_tokens: 1,
            generated_tokens: vec![3],
            completion: "token".into(),
            stop_reason: StopReason::MaxNewTokens,
            ranks: Vec::new(),
            events: Vec::new(),
            timings: PipelineTimingReport::default(),
            communication_bytes: 128,
            failures: Vec::new(),
            success: true,
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["pipeline_parallel"], 2);
        let decoded: PipelineReport = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.model, model);
        assert_eq!(decoded.generated_tokens, vec![3]);
        assert!(decoded.success);
    }
}
