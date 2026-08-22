//! Schema-versioned generation results, timings, topology, and event records.
//!
//! Human CLI text is a presentation concern. These serializable records are the runtime's
//! machine-readable observability contract.

use crate::{PlanDType, RankMemoryPlan, SupportedModelId};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Reason the autoregressive loop stopped successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model selected its registered end-of-sequence token.
    Eos,
    /// The requested number of non-EOS tokens was emitted.
    MaxNewTokens,
    /// Available model context was smaller than the requested generation length.
    ContextLimit,
}

impl fmt::Display for StopReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Eos => "eos",
            Self::MaxNewTokens => "max_new_tokens",
            Self::ContextLimit => "context_limit",
        })
    }
}

/// Parallel topology identity captured in every generation report.
///
/// v0.1 serializes the single-rank baseline explicitly so later reports can preserve the same
/// conceptual fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyReport {
    /// Number of participating ranks; `1` in v0.1.
    pub world_size: usize,
    /// Global rank producing the report; `0` in v0.1.
    pub rank: usize,
    /// Tensor-parallel group size; `1` in v0.1.
    pub tensor_parallel: usize,
    /// Pipeline-parallel group size; `1` in v0.1.
    pub pipeline_parallel: usize,
    /// Expert-parallel group size; `1` in v0.1.
    pub expert_parallel: usize,
}

impl Default for TopologyReport {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            tensor_parallel: 1,
            pipeline_parallel: 1,
            expert_parallel: 1,
        }
    }
}

/// Pipeline execution phase associated with a layer or tensor event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    /// Multi-token prompt processing.
    Prefill,
    /// One-token cached autoregressive processing.
    Decode,
}

/// Purpose assigned to a distributed tensor transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorPurpose {
    /// Residual-stream activations passed to the next pipeline stage.
    Activation,
}

/// Purpose assigned to a bounded pipeline control transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlPurpose {
    /// Greedy token ID returned from the final stage to rank 0.
    TokenFeedback,
    /// Rank-0 continue/stop decision broadcast to the other stages.
    Decision,
}

/// Collective operation represented in the shared event vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveKind {
    /// Reusable centralized rank barrier.
    Barrier,
}

/// Payload of one event in the ordered runtime timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEventKind {
    /// Artifact resolution or validation has started.
    ArtifactResolutionStarted,
    /// The current artifact resolution/validation phase has finished.
    ArtifactResolutionFinished,
    /// Model and cache construction has started.
    ModelLoadStarted,
    /// Model and cache construction has finished.
    ModelLoadFinished,
    /// One global transformer layer is about to execute on its owning rank.
    LayerStarted {
        /// Global zero-based transformer-layer index.
        layer: usize,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
    },
    /// One global transformer layer finished executing.
    LayerCompleted {
        /// Global zero-based transformer-layer index.
        layer: usize,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
        /// Rank-local synchronized compute duration.
        duration_ns: u64,
    },
    /// A collective operation is about to block.
    CollectiveStarted {
        /// Collective implementation being entered.
        collective: CollectiveKind,
        /// Reusable collective generation.
        generation: u64,
    },
    /// A collective operation completed.
    CollectiveCompleted {
        /// Collective implementation that completed.
        collective: CollectiveKind,
        /// Reusable collective generation.
        generation: u64,
        /// Rank-local collective duration.
        duration_ns: u64,
    },
    /// A tensor was copied and sent to another rank.
    TensorSent {
        /// Destination global rank.
        peer: usize,
        /// Semantic role of the tensor.
        purpose: TensorPurpose,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
        /// Tensor dimensions.
        shape: Vec<usize>,
        /// Logical F32 payload bytes.
        bytes: u64,
        /// Rank-local copy and send duration.
        duration_ns: u64,
    },
    /// A tensor was received and reconstructed from another rank.
    TensorReceived {
        /// Source global rank.
        peer: usize,
        /// Semantic role of the tensor.
        purpose: TensorPurpose,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
        /// Tensor dimensions.
        shape: Vec<usize>,
        /// Logical F32 payload bytes.
        bytes: u64,
        /// Rank-local receive and reconstruction duration.
        duration_ns: u64,
    },
    /// A bounded typed control payload was sent to another rank.
    ControlSent {
        /// Destination global rank.
        peer: usize,
        /// Semantic role of the control message.
        purpose: ControlPurpose,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
        /// Logical serialized payload bytes.
        bytes: u64,
        /// Rank-local serialization and send duration.
        duration_ns: u64,
    },
    /// A bounded typed control payload was received from another rank.
    ControlReceived {
        /// Source global rank.
        peer: usize,
        /// Semantic role of the control message.
        purpose: ControlPurpose,
        /// Prefill or decode phase.
        phase: ExecutionPhase,
        /// Zero for prefill and one-based for decode.
        step: usize,
        /// Logical serialized payload bytes.
        bytes: u64,
        /// Rank-local receive and deserialization duration.
        duration_ns: u64,
    },
    /// A rank sampled its current enforced-container memory state.
    MemorySample {
        /// Current cgroup memory usage, when available.
        current_bytes: Option<u64>,
        /// Enforced cgroup memory maximum, when available.
        limit_bytes: Option<u64>,
    },
    /// The synchronized prompt forward is about to run.
    PrefillStarted {
        /// Number of tokens in the rendered prompt.
        prompt_tokens: usize,
    },
    /// The synchronized prompt forward has finished.
    PrefillFinished,
    /// One cached single-token model forward is about to run.
    DecodeStepStarted {
        /// One-based decode-forward index.
        step: usize,
    },
    /// One cached single-token model forward has finished.
    DecodeStepFinished {
        /// One-based decode-forward index.
        step: usize,
    },
    /// A non-EOS token has been selected and decoded incrementally.
    TokenGenerated {
        /// Vocabulary token ID selected by greedy argmax.
        token_id: u32,
        /// Text fragment currently available from the tokenizer decode stream.
        text: String,
    },
    /// The generation loop completed successfully.
    GenerationFinished {
        /// Condition that ended generation.
        stop_reason: StopReason,
    },
}

/// One monotonically ordered, rank-aware runtime event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    /// Zero-based event order within this request.
    pub sequence: u64,
    /// Emitting global rank; zero for single-rank generation.
    pub rank: usize,
    /// Nanoseconds elapsed since the emitting rank entered the request.
    pub elapsed_ns: u64,
    /// Event-specific discriminator and payload, flattened into JSON.
    #[serde(flatten)]
    pub event: RunEventKind,
}

/// Measured operation durations and derived throughput values.
///
/// Duration fields serialize as integer nanoseconds. Optional decode aggregates are absent when
/// the run requires no one-token decode forward.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingReport {
    /// Metadata plus checkpoint resolution and validation time.
    pub artifact_resolution_ns: u64,
    /// Model construction, cache allocation, and final synchronization time.
    pub model_load_ns: u64,
    /// Fixed-template rendering and tokenizer encoding time.
    pub tokenization_ns: u64,
    /// Synchronized multi-token prompt forward time.
    pub prefill_ns: u64,
    /// Tokenization plus post-load work through first-token selection.
    pub time_to_first_token_ns: u64,
    /// Sum of synchronized one-token model-forward durations.
    pub decode_total_ns: u64,
    /// Number of one-token forwards, including a forward that discovers EOS.
    pub decode_forward_count: usize,
    /// Mean synchronized model-forward duration, if any decode occurred.
    pub mean_decode_ns: Option<u64>,
    /// Prompt tokens divided by prefill model-forward seconds.
    pub prefill_tokens_per_second: f64,
    /// Decode-forward count divided by decode model-forward seconds, when defined.
    pub decode_tokens_per_second: Option<f64>,
    /// Tokenization plus post-load generation work, excluding artifacts and model loading.
    pub generation_total_ns: u64,
    /// Request entry through final completion decoding and cache accounting, including artifacts
    /// and model loading and ending immediately before report construction.
    pub cold_start_total_ns: u64,
}

/// Complete schema-versioned result of one generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationReport {
    /// Serialization contract version; currently `1`.
    pub schema_version: u32,
    /// Closed-registry model ID.
    pub model: SupportedModelId,
    /// Pinned Hugging Face repository.
    pub repository: String,
    /// Pinned repository revision.
    pub revision: String,
    /// Execution device name; `cpu` in v0.1.
    pub device: String,
    /// Runtime model and cache dtype.
    pub dtype: PlanDType,
    /// Explicit single-rank topology identity.
    pub topology: TopologyReport,
    /// Prompt-dependent rank-0 logical memory plan.
    pub memory: RankMemoryPlan,
    /// Logical bytes populated in the KV cache when generation ended.
    pub final_kv_cache_bytes: u64,
    /// Unicode scalar-value count in the unformatted user prompt.
    pub prompt_characters: usize,
    /// Token count after chat-template rendering and encoding.
    pub prompt_tokens: usize,
    /// User-requested maximum count of non-EOS generated tokens.
    pub requested_max_new_tokens: usize,
    /// Non-EOS token IDs selected in generation order.
    pub generated_tokens: Vec<u32>,
    /// Final decoded assistant text.
    pub completion: String,
    /// Successful condition that ended the generation loop.
    pub stop_reason: StopReason,
    /// Measured and derived timing values.
    pub timings: TimingReport,
    /// Complete ordered event stream also delivered live to the observer.
    pub events: Vec<RunEvent>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryBudget, PlanDType, RankMemoryPlan};

    #[test]
    fn generation_report_is_schema_versioned_and_round_trips() {
        let model = SupportedModelId::SmolLm2_135MInstruct;
        let report = GenerationReport {
            schema_version: 1,
            model,
            repository: model.spec().repository.into(),
            revision: model.spec().revision.into(),
            device: "cpu".into(),
            dtype: PlanDType::F32,
            topology: TopologyReport::default(),
            memory: RankMemoryPlan::for_model(
                model.spec(),
                PlanDType::F32,
                8,
                Some(MemoryBudget::user_declared(u64::MAX)),
            )
            .unwrap(),
            final_kv_cache_bytes: 1,
            prompt_characters: 5,
            prompt_tokens: 2,
            requested_max_new_tokens: 1,
            generated_tokens: vec![42],
            completion: "hello".into(),
            stop_reason: StopReason::MaxNewTokens,
            timings: TimingReport::default(),
            events: vec![RunEvent {
                sequence: 0,
                rank: 0,
                elapsed_ns: 12,
                event: RunEventKind::GenerationFinished {
                    stop_reason: StopReason::MaxNewTokens,
                },
            }],
        };
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["topology"]["world_size"], 1);
        assert_eq!(value["events"][0]["rank"], 0);
        let decoded: GenerationReport = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.generated_tokens, vec![42]);
        assert_eq!(decoded.stop_reason, StopReason::MaxNewTokens);
    }
}
