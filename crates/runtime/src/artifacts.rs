use crate::{DlirError, ModelConfig, ModelSpec, Result};
use candle_core::safetensors::MmapedSafetensors;
use hf_hub::{
    Repo, RepoType,
    api::sync::{Api, ApiRepo},
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

pub struct ArtifactRepository {
    repo: ApiRepo,
}

#[derive(Debug)]
pub struct MetadataArtifacts {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub tokenizer_config: PathBuf,
}

impl ArtifactRepository {
    pub fn new(spec: &ModelSpec) -> Result<Self> {
        let api = Api::new().map_err(|err| DlirError::Artifact(err.to_string()))?;
        let repo = api.repo(Repo::with_revision(
            spec.repository.to_owned(),
            RepoType::Model,
            spec.revision.to_owned(),
        ));
        Ok(Self { repo })
    }

    pub fn download_metadata(&self) -> Result<MetadataArtifacts> {
        Ok(MetadataArtifacts {
            config: self.get("config.json")?,
            tokenizer: self.get("tokenizer.json")?,
            tokenizer_config: self.get("tokenizer_config.json")?,
        })
    }

    pub fn download_weights(&self, spec: &ModelSpec) -> Result<PathBuf> {
        self.get(spec.weight_file)
    }

    fn get(&self, file: &str) -> Result<PathBuf> {
        self.repo
            .get(file)
            .map_err(|err| DlirError::Artifact(format!("could not resolve {file}: {err}")))
    }
}

#[derive(Debug, Deserialize)]
struct HubLlamaConfig {
    architectures: Vec<String>,
    model_type: String,
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    max_position_embeddings: usize,
    bos_token_id: u32,
    eos_token_id: u32,
    tie_word_embeddings: bool,
    rope_scaling: Option<serde_json::Value>,
    vocab_size: usize,
}

pub fn validate_metadata(spec: &ModelSpec, artifacts: &MetadataArtifacts) -> Result<()> {
    let raw = fs::read(&artifacts.config)?;
    let config: HubLlamaConfig = serde_json::from_slice(&raw)?;
    if config.architectures.as_slice() != ["LlamaForCausalLM"]
        || config.model_type != "llama"
        || config.hidden_act != "silu"
        || config.attention_bias
        || config.mlp_bias
        || config.rope_scaling.is_some()
    {
        return Err(DlirError::CheckpointMismatch(
            "only unbiased LlamaForCausalLM checkpoints with SiLU and unscaled RoPE are supported"
                .into(),
        ));
    }
    let actual = ModelConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        max_position_embeddings: config.max_position_embeddings,
        bos_token_id: config.bos_token_id,
        eos_token_id: config.eos_token_id,
        tie_word_embeddings: config.tie_word_embeddings,
    };
    if actual != spec.config {
        return Err(DlirError::CheckpointMismatch(format!(
            "config for {} differs from the compiled registry\nexpected: {:?}\nactual:   {:?}",
            spec.id, spec.config, actual
        )));
    }

    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(&fs::read(&artifacts.tokenizer_config)?)?;
    let chat_template = tokenizer_config
        .get("chat_template")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            DlirError::CheckpointMismatch("tokenizer_config.json has no chat_template".into())
        })?;
    for marker in required_template_markers(spec) {
        if !chat_template.contains(marker) {
            return Err(DlirError::CheckpointMismatch(format!(
                "chat template for {} is missing expected marker {marker:?}",
                spec.id
            )));
        }
    }
    Ok(())
}

fn required_template_markers(spec: &ModelSpec) -> &'static [&'static str] {
    match spec.prompt_template {
        crate::PromptTemplate::SmolChatMl => &["<|im_start|>", "<|im_end|>"],
        crate::PromptTemplate::TinyLlamaChat => &["<|user|>", "<|assistant|>"],
    }
}

pub fn validate_checkpoint(spec: &ModelSpec, path: &PathBuf) -> Result<()> {
    // SAFETY: the immutable Hub cache path remains present for the duration of this function.
    // MmapedSafetensors owns the mapping, and no mutable file handle is exposed here.
    let checkpoint = unsafe { MmapedSafetensors::new(path) }?;
    let tensors = checkpoint
        .tensors()
        .into_iter()
        .map(|(name, view)| (name, (view.shape().to_vec(), view.dtype())))
        .collect::<BTreeMap<_, _>>();
    validate_tensor_metadata(spec, &tensors)
}

fn validate_tensor_metadata(
    spec: &ModelSpec,
    tensors: &BTreeMap<String, (Vec<usize>, safetensors::Dtype)>,
) -> Result<()> {
    let expected = expected_tensor_shapes(spec)?;
    let wrong_dtypes = tensors
        .iter()
        .filter(|(_, (_, dtype))| *dtype != spec.checkpoint_dtype.safetensors())
        .map(|(name, (_, dtype))| (name, dtype))
        .collect::<Vec<_>>();
    if !wrong_dtypes.is_empty() {
        let preview = &wrong_dtypes[..wrong_dtypes.len().min(8)];
        return Err(DlirError::CheckpointMismatch(format!(
            "expected {:?} tensors but {} tensors differed; first mismatches: {preview:?}",
            spec.checkpoint_dtype,
            wrong_dtypes.len(),
        )));
    }
    let actual: BTreeMap<String, Vec<usize>> = tensors
        .iter()
        .map(|(name, (shape, _))| (name.clone(), shape.clone()))
        .collect();

    if actual != expected {
        let missing: Vec<_> = expected
            .keys()
            .filter(|key| !actual.contains_key(*key))
            .collect();
        let unexpected: Vec<_> = actual
            .keys()
            .filter(|key| !expected.contains_key(*key))
            .collect();
        let wrong_shapes: Vec<_> = expected
            .iter()
            .filter_map(|(name, shape)| {
                actual
                    .get(name)
                    .filter(|actual_shape| *actual_shape != shape)
                    .map(|actual_shape| (name, shape, actual_shape))
            })
            .collect();
        return Err(DlirError::CheckpointMismatch(format!(
            "tensor manifest mismatch; missing={missing:?}, unexpected={unexpected:?}, wrong_shapes={wrong_shapes:?}"
        )));
    }

    let parameters = tensors
        .values()
        .try_fold(0u64, |total, (shape, _)| {
            let count = shape
                .iter()
                .try_fold(1u64, |value, dim| value.checked_mul(*dim as u64).ok_or(()));
            count.and_then(|count| total.checked_add(count).ok_or(()))
        })
        .map_err(|_| DlirError::CheckpointMismatch("parameter count overflow".into()))?;
    if parameters != spec.expected_parameters {
        return Err(DlirError::CheckpointMismatch(format!(
            "checkpoint has {parameters} parameters but {} are expected",
            spec.expected_parameters
        )));
    }
    Ok(())
}

fn expected_tensor_shapes(spec: &ModelSpec) -> Result<BTreeMap<String, Vec<usize>>> {
    let cfg = &spec.config;
    let head_dim = cfg.head_dim()?;
    let kv_size = cfg.num_key_value_heads * head_dim;
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.embed_tokens.weight".into(),
        vec![cfg.vocab_size, cfg.hidden_size],
    );
    tensors.insert("model.norm.weight".into(), vec![cfg.hidden_size]);
    if !cfg.tie_word_embeddings {
        tensors.insert(
            "lm_head.weight".into(),
            vec![cfg.vocab_size, cfg.hidden_size],
        );
    }
    for layer in 0..cfg.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![cfg.hidden_size, cfg.hidden_size],
        );
        for projection in ["k_proj", "v_proj"] {
            tensors.insert(
                format!("{prefix}.self_attn.{projection}.weight"),
                vec![kv_size, cfg.hidden_size],
            );
        }
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![cfg.hidden_size, cfg.hidden_size],
        );
        for projection in ["gate_proj", "up_proj"] {
            tensors.insert(
                format!("{prefix}.mlp.{projection}.weight"),
                vec![cfg.intermediate_size, cfg.hidden_size],
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![cfg.hidden_size, cfg.intermediate_size],
        );
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![cfg.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![cfg.hidden_size],
        );
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SupportedModelId;

    fn metadata_files(
        config: serde_json::Value,
        template: &str,
    ) -> (tempfile::TempDir, MetadataArtifacts) {
        let directory = tempfile::tempdir().unwrap();
        let config_path = directory.path().join("config.json");
        let tokenizer = directory.path().join("tokenizer.json");
        let tokenizer_config = directory.path().join("tokenizer_config.json");
        fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        fs::write(&tokenizer, b"{}").unwrap();
        fs::write(
            &tokenizer_config,
            serde_json::to_vec(&serde_json::json!({ "chat_template": template })).unwrap(),
        )
        .unwrap();
        (
            directory,
            MetadataArtifacts {
                config: config_path,
                tokenizer,
                tokenizer_config,
            },
        )
    }

    fn smol_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "hidden_size": 576,
            "intermediate_size": 1536,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "rms_norm_eps": 0.00001,
            "rope_theta": 100000.0,
            "max_position_embeddings": 8192,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": true,
            "rope_scaling": null,
            "vocab_size": 49152,
        })
    }

    #[test]
    fn metadata_must_match_registry_and_supported_llama_subset() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let (_directory, artifacts) = metadata_files(
            smol_config(),
            "{% if true %}<|im_start|><|im_end|>{% endif %}",
        );
        validate_metadata(spec, &artifacts).unwrap();

        let mut invalid = smol_config();
        invalid["hidden_act"] = serde_json::json!("gelu");
        let (_directory, artifacts) =
            metadata_files(invalid, "{% if true %}<|im_start|><|im_end|>{% endif %}");
        assert!(matches!(
            validate_metadata(spec, &artifacts),
            Err(DlirError::CheckpointMismatch(_))
        ));
    }

    #[test]
    fn tensor_manifest_checks_names_shapes_dtypes_and_parameter_count() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let expected = expected_tensor_shapes(spec).unwrap();
        let mut tensors = expected
            .into_iter()
            .map(|(name, shape)| (name, (shape, spec.checkpoint_dtype.safetensors())))
            .collect::<BTreeMap<_, _>>();
        validate_tensor_metadata(spec, &tensors).unwrap();

        let key = tensors.keys().next().unwrap().clone();
        tensors.get_mut(&key).unwrap().0[0] += 1;
        assert!(matches!(
            validate_tensor_metadata(spec, &tensors),
            Err(DlirError::CheckpointMismatch(_))
        ));

        tensors.get_mut(&key).unwrap().0[0] -= 1;
        tensors.get_mut(&key).unwrap().1 = safetensors::Dtype::F32;
        assert!(matches!(
            validate_tensor_metadata(spec, &tensors),
            Err(DlirError::CheckpointMismatch(_))
        ));
    }
}
