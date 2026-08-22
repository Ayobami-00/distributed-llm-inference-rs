//! Closed registry of model identities, architecture contracts, and execution support.
//!
//! Registry data drives CLI validation, artifact resolution, tensor-manifest validation, memory
//! formulas, model construction, prompt rendering, and report identity. Arbitrary Hub IDs and
//! aliases are intentionally not accepted.

use crate::{DlirError, Result};
use candle_core::DType;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

const SMOL_CHAT_TEMPLATE: &str = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
const TINY_LLAMA_CHAT_TEMPLATE: &str = "<|user|>\n{prompt}</s>\n<|assistant|>\n";

/// Stable CLI and report identifier for a model supported by this release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupportedModelId {
    /// Hugging Face SmolLM2 135M instruction-tuned checkpoint.
    #[serde(rename = "smollm2-135m-instruct")]
    SmolLm2_135MInstruct,
    /// TinyLlama 1.1B chat checkpoint.
    #[serde(rename = "tinyllama-1.1b-chat")]
    TinyLlama1_1BChat,
}

impl SupportedModelId {
    /// Returns the exact lowercase identifier accepted by the CLI and emitted in JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SmolLm2_135MInstruct => "smollm2-135m-instruct",
            Self::TinyLlama1_1BChat => "tinyllama-1.1b-chat",
        }
    }

    /// Returns the immutable compiled specification associated with this identifier.
    pub fn spec(self) -> &'static ModelSpec {
        SUPPORTED_MODELS
            .iter()
            .find(|spec| spec.id == self)
            .expect("every model id must have a registry entry")
    }
}

impl fmt::Display for SupportedModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SupportedModelId {
    type Err = DlirError;

    fn from_str(value: &str) -> Result<Self> {
        SUPPORTED_MODELS
            .iter()
            .find(|spec| spec.id.as_str() == value)
            .map(|spec| spec.id)
            .ok_or_else(|| DlirError::UnsupportedModel(value.to_owned()))
    }
}

/// Numeric dtype used for logical planning and, when supported, model execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanDType {
    /// IEEE 754 half precision, two bytes per element.
    F16,
    /// Brain floating point, two bytes per element.
    Bf16,
    /// IEEE 754 single precision, four bytes per element.
    F32,
}

impl PlanDType {
    /// Returns the logical bytes occupied by one value of this dtype.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// Converts the planning dtype into Candle's dtype representation.
    pub const fn candle(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::Bf16 => DType::BF16,
            Self::F32 => DType::F32,
        }
    }
}

impl fmt::Display for PlanDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        };
        f.write_str(value)
    }
}

impl FromStr for PlanDType {
    type Err = DlirError;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "f32" => Ok(Self::F32),
            _ => Err(DlirError::InvalidConfig(format!(
                "unsupported dtype '{value}'; expected f16, bf16, or f32"
            ))),
        }
    }
}

/// Validation state of an execution backend/dtype combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSupport {
    /// The combination has passed release acceptance tests.
    Validated,
    /// The combination is represented in the roadmap but cannot execute yet.
    Planned,
    /// The combination is intentionally not supported.
    Unsupported,
}

/// On-disk naming and shape convention expected from checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorLayout {
    /// Hugging Face `LlamaForCausalLM` safetensor names and shapes.
    HuggingFaceLlama,
}

/// Element dtype required in a registered checkpoint file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointDType {
    /// IEEE 754 half-precision checkpoint tensors.
    F16,
    /// Brain floating-point checkpoint tensors.
    Bf16,
}

impl CheckpointDType {
    /// Converts the registry value to the safetensors dtype discriminator.
    pub const fn safetensors(self) -> safetensors::Dtype {
        match self {
            Self::F16 => safetensors::Dtype::F16,
            Self::Bf16 => safetensors::Dtype::BF16,
        }
    }
}

/// Fixed one-user-message chat template selected by the model registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplate {
    /// SmolLM2 ChatML-style system/user/assistant template.
    SmolChatMl,
    /// TinyLlama user/assistant template.
    TinyLlamaChat,
}

impl PromptTemplate {
    /// Returns the fixed template source containing a `{prompt}` placeholder.
    pub const fn source(self) -> &'static str {
        match self {
            Self::SmolChatMl => SMOL_CHAT_TEMPLATE,
            Self::TinyLlamaChat => TINY_LLAMA_CHAT_TEMPLATE,
        }
    }

    /// Returns the stable template identity emitted by `dlir models`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::SmolChatMl => "smollm2-chatml",
            Self::TinyLlamaChat => "tinyllama-chat",
        }
    }
}

/// Llama architecture values required by model execution and memory planning.
///
/// The registry stores this independently of downloaded `config.json`; generation reconstructs
/// the same value from that file and requires exact equality before loading weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of token IDs and output logits.
    pub vocab_size: usize,
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Width of the SwiGLU intermediate projections.
    pub intermediate_size: usize,
    /// Number of transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of query/attention heads.
    pub num_attention_heads: usize,
    /// Number of compact key/value heads used by GQA.
    pub num_key_value_heads: usize,
    /// Epsilon added by RMSNorm for numerical stability.
    pub rms_norm_eps: f64,
    /// Base frequency parameter for unscaled RoPE.
    pub rope_theta: f64,
    /// Maximum supported token positions.
    pub max_position_embeddings: usize,
    /// Beginning-of-sequence token ID from the checkpoint configuration.
    pub bos_token_id: u32,
    /// End-of-sequence token ID that terminates generation.
    pub eos_token_id: u32,
    /// Whether the output projection reuses the token embedding matrix.
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Validates head divisibility and returns `hidden_size / num_attention_heads`.
    ///
    /// GQA additionally requires the number of query heads to be divisible by the number of KV
    /// heads so every KV head serves an equal-sized query group.
    pub fn head_dim(&self) -> Result<usize> {
        if self.num_attention_heads == 0 || self.hidden_size % self.num_attention_heads != 0 {
            return Err(DlirError::InvalidConfig(format!(
                "hidden size {} is not divisible by {} attention heads",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err(DlirError::InvalidConfig(format!(
                "{} attention heads are not divisible by {} KV heads",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

/// Complete immutable contract for one supported model checkpoint.
#[derive(Debug)]
pub struct ModelSpec {
    /// Stable local identifier.
    pub id: SupportedModelId,
    /// Hugging Face model repository.
    pub repository: &'static str,
    /// Exact repository commit used for every artifact.
    pub revision: &'static str,
    /// Safetensor filename at the pinned revision.
    pub weight_file: &'static str,
    /// Architectural parameter total expected from formulas and checkpoint metadata.
    pub expected_parameters: u64,
    /// Known on-disk checkpoint size used for download status.
    pub expected_checkpoint_bytes: u64,
    /// Required dtype of every safetensor in the checkpoint.
    pub checkpoint_dtype: CheckpointDType,
    /// Tensor naming and shape convention.
    pub tensor_layout: TensorLayout,
    /// Architecture values used by planning and execution.
    pub config: ModelConfig,
    /// Fixed prompt template used before tokenization.
    pub prompt_template: PromptTemplate,
    /// CPU execution support state.
    pub cpu_support: ExecutionSupport,
    /// CUDA execution support state.
    pub cuda_support: ExecutionSupport,
}

impl ModelSpec {
    /// Requires this release's validated CPU/F32 execution combination.
    pub fn validate_cpu_dtype(&self, dtype: PlanDType) -> Result<()> {
        if self.cpu_support != ExecutionSupport::Validated || dtype != PlanDType::F32 {
            return Err(DlirError::UnsupportedExecution {
                model: self.id,
                dtype,
            });
        }
        Ok(())
    }
}

static SUPPORTED_MODELS: [ModelSpec; 2] = [
    ModelSpec {
        id: SupportedModelId::SmolLm2_135MInstruct,
        repository: "HuggingFaceTB/SmolLM2-135M-Instruct",
        revision: "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        weight_file: "model.safetensors",
        expected_parameters: 134_515_008,
        expected_checkpoint_bytes: 269_060_552,
        checkpoint_dtype: CheckpointDType::Bf16,
        tensor_layout: TensorLayout::HuggingFaceLlama,
        config: ModelConfig {
            vocab_size: 49_152,
            hidden_size: 576,
            intermediate_size: 1_536,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000.0,
            max_position_embeddings: 8_192,
            bos_token_id: 1,
            eos_token_id: 2,
            tie_word_embeddings: true,
        },
        prompt_template: PromptTemplate::SmolChatMl,
        cpu_support: ExecutionSupport::Validated,
        cuda_support: ExecutionSupport::Planned,
    },
    ModelSpec {
        id: SupportedModelId::TinyLlama1_1BChat,
        repository: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision: "5243d158d6f4b356f1142ea8fd6a99cb5ac2c0e1",
        weight_file: "model.safetensors",
        expected_parameters: 1_100_048_384,
        expected_checkpoint_bytes: 2_200_119_864,
        checkpoint_dtype: CheckpointDType::Bf16,
        tensor_layout: TensorLayout::HuggingFaceLlama,
        config: ModelConfig {
            vocab_size: 32_000,
            hidden_size: 2_048,
            intermediate_size: 5_632,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 2_048,
            bos_token_id: 1,
            eos_token_id: 2,
            tie_word_embeddings: false,
        },
        prompt_template: PromptTemplate::TinyLlamaChat,
        cpu_support: ExecutionSupport::Validated,
        cuda_support: ExecutionSupport::Planned,
    },
];

/// Returns the complete closed registry in stable display order.
pub fn supported_models() -> &'static [ModelSpec] {
    &SUPPORTED_MODELS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_ids_are_unique_and_valid() {
        let mut ids = HashSet::new();
        for spec in supported_models() {
            assert!(ids.insert(spec.id.as_str()));
            assert_eq!(spec.id.spec().repository, spec.repository);
            assert_eq!(
                spec.config.head_dim().unwrap() * spec.config.num_attention_heads,
                spec.config.hidden_size
            );
        }
    }

    #[test]
    fn registry_revisions_and_counts_are_pinned() {
        let smol = SupportedModelId::SmolLm2_135MInstruct.spec();
        assert_eq!(smol.revision, "12fd25f77366fa6b3b4b768ec3050bf629380bac");
        assert_eq!(smol.expected_parameters, 134_515_008);

        let tiny = SupportedModelId::TinyLlama1_1BChat.spec();
        assert_eq!(tiny.revision, "5243d158d6f4b356f1142ea8fd6a99cb5ac2c0e1");
        assert_eq!(tiny.expected_parameters, 1_100_048_384);
        assert_eq!(smol.checkpoint_dtype, CheckpointDType::Bf16);
        assert_eq!(tiny.checkpoint_dtype, CheckpointDType::Bf16);
    }

    #[test]
    fn model_ids_accept_no_aliases() {
        for alias in [
            "smollm2",
            "tinyllama",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
        ] {
            assert!(alias.parse::<SupportedModelId>().is_err());
        }
    }

    #[test]
    fn invalid_head_divisibility_is_rejected() {
        let mut config = SupportedModelId::SmolLm2_135MInstruct.spec().config;
        config.num_attention_heads = 7;
        assert!(config.head_dim().is_err());
        config.num_attention_heads = 9;
        config.num_key_value_heads = 2;
        assert!(config.head_dim().is_err());
    }

    #[test]
    fn unknown_model_is_rejected() {
        assert!(
            "someone/arbitrary-model"
                .parse::<SupportedModelId>()
                .is_err()
        );
    }
}
