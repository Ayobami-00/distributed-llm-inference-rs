use crate::{PlanDType, RankMemoryPlan, SupportedModelId};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Eos,
    MaxNewTokens,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyReport {
    pub world_size: usize,
    pub rank: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunEventKind {
    ArtifactResolutionStarted,
    ArtifactResolutionFinished,
    ModelLoadStarted,
    ModelLoadFinished,
    PrefillStarted { prompt_tokens: usize },
    PrefillFinished,
    DecodeStepStarted { step: usize },
    DecodeStepFinished { step: usize },
    TokenGenerated { token_id: u32, text: String },
    GenerationFinished { stop_reason: StopReason },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    pub sequence: u64,
    pub rank: usize,
    pub elapsed_ns: u64,
    #[serde(flatten)]
    pub event: RunEventKind,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingReport {
    pub artifact_resolution_ns: u64,
    pub model_load_ns: u64,
    pub tokenization_ns: u64,
    pub prefill_ns: u64,
    pub time_to_first_token_ns: u64,
    pub decode_total_ns: u64,
    pub decode_forward_count: usize,
    pub mean_decode_ns: Option<u64>,
    pub prefill_tokens_per_second: f64,
    pub decode_tokens_per_second: Option<f64>,
    pub generation_total_ns: u64,
    pub cold_start_total_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationReport {
    pub schema_version: u32,
    pub model: SupportedModelId,
    pub repository: String,
    pub revision: String,
    pub device: String,
    pub dtype: PlanDType,
    pub topology: TopologyReport,
    pub memory: RankMemoryPlan,
    pub final_kv_cache_bytes: u64,
    pub prompt_characters: usize,
    pub prompt_tokens: usize,
    pub requested_max_new_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub completion: String,
    pub stop_reason: StopReason,
    pub timings: TimingReport,
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
