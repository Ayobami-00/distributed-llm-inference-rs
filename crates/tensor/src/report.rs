//! Tensor-parallel request manifests and schema-v1 reports.

use crate::{TensorParallelMemoryPlan, TensorParallelPartition};
use dlir_collectives::{AllReduceAlgorithm, CollectiveTrace, PeerInfo};
use dlir_pipeline::{PipelineEvent, ReceivedPipelineEvent, ResourceSnapshot};
use dlir_runtime::{PlanDType, StopReason, SupportedModelId};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Immutable request bind-mounted into every tensor-rank container.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelManifest {
    /// Manifest schema; currently `1`.
    pub schema_version: u32,
    /// Docker/TCP run identity.
    pub run_id: String,
    /// Generation request identity.
    pub request_id: String,
    /// Closed-registry model.
    pub model: SupportedModelId,
    /// F32 execution dtype.
    pub dtype: PlanDType,
    /// Tensor rank count.
    pub tensor_parallel: usize,
    /// Native all-reduce implementation.
    pub all_reduce: AllReduceAlgorithm,
    /// Rendered prompt token IDs.
    pub prompt_token_ids: Vec<u32>,
    /// User maximum output length.
    pub requested_max_new_tokens: usize,
    /// Output length clipped by remaining context.
    pub effective_max_new_tokens: usize,
    /// Prompt plus effective output positions.
    pub context_capacity: usize,
    /// Container-visible checkpoint.
    pub checkpoint_path: PathBuf,
    /// Container-visible tokenizer.
    pub tokenizer_path: PathBuf,
    /// Exact rank shard and memory plans.
    pub partition: TensorParallelPartition,
    /// Expected cgroup CPU quota per rank.
    pub expected_cpu_millis: u64,
    /// Expected cgroup memory maximum per rank.
    pub expected_memory_bytes: u64,
}

/// Rank-local timing aggregates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorParallelRankTimings {
    /// Mmap shard loading and local cache allocation.
    pub model_load_ns: u64,
    /// First distributed forward through token selection.
    pub prefill_ns: u64,
    /// Sum of cached distributed decode forwards.
    pub decode_total_ns: u64,
    /// Count of cached forwards.
    pub decode_forward_count: usize,
    /// Complete rank request duration.
    pub total_ns: u64,
}

/// Final result emitted by one tensor rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelRankReport {
    /// Report schema; currently `1`.
    pub schema_version: u32,
    /// Run identity.
    pub run_id: String,
    /// Request identity.
    pub request_id: String,
    /// Global/tensor rank.
    pub rank: usize,
    /// Rendezvous-ordered peer table.
    pub peers: Vec<PeerInfo>,
    /// Rank-local shard and memory plan.
    pub memory: TensorParallelMemoryPlan,
    /// Resource limits and usage observed inside the rank container.
    pub resources: ResourceSnapshot,
    /// Final populated compact KV bytes.
    pub final_kv_cache_bytes: u64,
    /// Rank-local collective calls in sequence order.
    pub collectives: Vec<CollectiveTrace>,
    /// Logical tensor bytes sent by collective calls.
    pub sent_bytes: u64,
    /// Logical tensor bytes received by collective calls.
    pub received_bytes: u64,
    /// Generated non-EOS token IDs.
    pub generated_tokens: Vec<u32>,
    /// Completion text, populated on rank 0.
    pub completion: String,
    /// Termination condition.
    pub stop_reason: StopReason,
    /// Rank-local timing aggregates.
    pub timings: TensorParallelRankTimings,
    /// Complete rank-local event stream.
    pub events: Vec<PipelineEvent>,
    /// Whether startup and completion barriers passed.
    pub barriers_passed: bool,
    /// Rank correctness verdict.
    pub success: bool,
}

/// Host resource allocation recorded with a tensor run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TensorParallelResourcePlan {
    /// Docker Engine CPU capacity in millicpus.
    pub engine_cpu_millis: u64,
    /// Docker Engine memory capacity.
    pub engine_memory_bytes: u64,
    /// Requested CPU total.
    pub requested_cpu_millis: u64,
    /// Requested memory total.
    pub requested_memory_bytes: u64,
    /// Equal per-rank CPU quota.
    pub per_rank_cpu_millis: u64,
    /// Equal per-rank memory maximum.
    pub per_rank_memory_bytes: u64,
    /// Unallocated CPU remainder.
    pub unused_cpu_millis: u64,
    /// Unallocated memory remainder.
    pub unused_memory_bytes: u64,
}

/// Complete host-aggregated schema-v1 tensor-parallel generation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelReport {
    /// Report schema; currently `1`.
    pub schema_version: u32,
    /// Run identity.
    pub run_id: String,
    /// Request identity.
    pub request_id: String,
    /// Model identity.
    pub model: SupportedModelId,
    /// Pinned repository.
    pub repository: String,
    /// Pinned revision.
    pub revision: String,
    /// Execution dtype.
    pub dtype: PlanDType,
    /// Transport backend; `tcp`.
    pub transport: String,
    /// Collective backend; `native`.
    pub collective_backend: String,
    /// Selected all-reduce.
    pub all_reduce: AllReduceAlgorithm,
    /// Tensor group/world size.
    pub tensor_parallel: usize,
    /// Pipeline group size; one.
    pub pipeline_parallel: usize,
    /// Expert group size; one.
    pub expert_parallel: usize,
    /// Exact partition and logical memory plan.
    pub partition: TensorParallelPartition,
    /// Docker resource allocation.
    pub resources: TensorParallelResourcePlan,
    /// Prompt token count.
    pub prompt_tokens: usize,
    /// Generated token IDs.
    pub generated_tokens: Vec<u32>,
    /// Decoded completion.
    pub completion: String,
    /// Stop reason.
    pub stop_reason: StopReason,
    /// Rank reports ordered by rank.
    pub ranks: Vec<TensorParallelRankReport>,
    /// Cross-rank event stream annotated with host receive order.
    pub events: Vec<ReceivedPipelineEvent>,
    /// Logical collective bytes, counting sends once.
    pub communication_bytes: u64,
    /// Host cold-start duration.
    pub cold_start_total_ns: u64,
    /// Aggregation or container failures.
    pub failures: Vec<String>,
    /// Overall verdict.
    pub success: bool,
}

/// One JSON line written by an internal tensor rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum TensorParallelStreamRecord {
    /// Live rank event.
    Event {
        /// Published shared event envelope.
        event: PipelineEvent,
    },
    /// Final rank result.
    Result {
        /// Completed rank report.
        result: Box<TensorParallelRankReport>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlir_runtime::{RunEvent, RunEventKind};

    #[test]
    fn collective_event_json_has_distinct_event_and_collective_sequences() {
        let record = TensorParallelStreamRecord::Event {
            event: PipelineEvent {
                schema_version: 1,
                run_id: "run".into(),
                request_id: "request".into(),
                event: RunEvent {
                    sequence: 7,
                    rank: 1,
                    elapsed_ns: 9,
                    event: RunEventKind::TensorCollectiveStarted {
                        collective: "allreduce".into(),
                        algorithm: "ring".into(),
                        collective_sequence: 3,
                        shape: vec![1, 4],
                    },
                },
            },
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"sequence\":7"));
        assert!(json.contains("\"collective_sequence\":3"));
        let decoded: TensorParallelStreamRecord = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded, TensorParallelStreamRecord::Event { .. }));
    }
}
