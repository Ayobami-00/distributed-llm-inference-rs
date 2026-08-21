use super::KvCache;
use crate::{DlirError, ModelConfig, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};

#[derive(Debug)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &ModelConfig, capacity: usize, dtype: DType, device: &Device) -> Result<Self> {
        let head_dim = config.head_dim()?;
        let inverse_frequency: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|index| (1.0 / config.rope_theta.powf(index as f64 / head_dim as f64)) as f32)
            .collect();
        let theta = Tensor::new(inverse_frequency, device)?;
        let positions = Tensor::arange(0u32, capacity as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((capacity, 1))?;
        let frequencies = positions.matmul(&theta.reshape((1, head_dim / 2))?)?;
        Ok(Self {
            cos: frequencies.cos()?.to_dtype(dtype)?,
            sin: frequencies.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, tensor: &Tensor, position: usize) -> Result<Tensor> {
        let sequence = tensor.dim(2)?;
        let cos = self.cos.narrow(0, position, sequence)?;
        let sin = self.sin.narrow(0, position, sequence)?;
        Ok(candle_nn::rotary_emb::rope(tensor, &cos, &sin)?)
    }
}

#[derive(Debug)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn load(vb: VarBuilder<'_>, config: &ModelConfig) -> Result<Self> {
        let head_dim = config.head_dim()?;
        let kv_size = config.num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(config.hidden_size, kv_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(config.hidden_size, kv_size, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("o_proj"))?,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        position: usize,
        layer: usize,
        rope: &RotaryEmbedding,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch, sequence, hidden) = input.dims3()?;
        let q = self
            .q_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(input)?
            .reshape((batch, sequence, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = rope.apply(&q, position)?;
        let k = rope.apply(&k, position)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let k = repeat_kv(&k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(&v, self.num_attention_heads / self.num_key_value_heads)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let scores = if sequence > 1 {
            apply_causal_mask(&scores, position, sequence)?
        } else {
            scores
        };
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let output = probabilities
            .matmul(&v.contiguous()?)?
            .to_dtype(input.dtype())?
            .transpose(1, 2)?
            .reshape((batch, sequence, hidden))?;
        Ok(self.o_proj.forward(&output)?)
    }
}

fn repeat_kv(tensor: &Tensor, repetitions: usize) -> Result<Tensor> {
    if repetitions == 1 {
        return Ok(tensor.clone());
    }
    let (batch, kv_heads, sequence, head_dim) = tensor.dims4()?;
    Ok(tensor
        .unsqueeze(2)?
        .expand((batch, kv_heads, repetitions, sequence, head_dim))?
        .reshape((batch, kv_heads * repetitions, sequence, head_dim))?)
}

fn apply_causal_mask(scores: &Tensor, position: usize, sequence: usize) -> Result<Tensor> {
    let key_length = position + sequence;
    let mut values = Vec::with_capacity(sequence * key_length);
    for query in 0..sequence {
        let absolute_query = position + query;
        for key in 0..key_length {
            values.push(u8::from(key > absolute_query));
        }
    }
    let mask = Tensor::from_vec(values, (sequence, key_length), scores.device())?
        .broadcast_as(scores.shape())?;
    let negative_infinity =
        Tensor::new(f32::NEG_INFINITY, scores.device())?.broadcast_as(scores.shape())?;
    Ok(mask.where_cond(&negative_infinity, scores)?)
}

#[derive(Debug)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn load(vb: VarBuilder<'_>, config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gated = candle_nn::ops::silu(&self.gate_proj.forward(input)?)?;
        Ok(self
            .down_proj
            .forward(&(gated * self.up_proj.forward(input)?)?)?)
    }
}

#[derive(Debug)]
struct Block {
    input_norm: RmsNorm,
    attention: CausalSelfAttention,
    post_attention_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: VarBuilder<'_>, config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            input_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            attention: CausalSelfAttention::load(vb.pp("self_attn"), config)?,
            post_attention_norm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), config)?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        position: usize,
        layer: usize,
        rope: &RotaryEmbedding,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let attention = self.attention.forward(
            &self.input_norm.forward(input)?,
            position,
            layer,
            rope,
            cache,
        )?;
        let hidden = (input + attention)?;
        Ok((&hidden
            + self
                .mlp
                .forward(&self.post_attention_norm.forward(&hidden)?)?)?)
    }
}

#[derive(Debug)]
pub struct Llama {
    embeddings: Embedding,
    blocks: Vec<Block>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
}

impl Llama {
    pub fn load(vb: VarBuilder<'_>, config: &ModelConfig, capacity: usize) -> Result<Self> {
        let embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let blocks = (0..config.num_hidden_layers)
            .map(|index| Block::load(vb.pp(format!("model.layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let rope = RotaryEmbedding::new(config, capacity, vb.dtype(), vb.device())?;
        Ok(Self {
            embeddings,
            blocks,
            final_norm,
            lm_head,
            rope,
        })
    }

    pub fn forward(
        &self,
        token_ids: &Tensor,
        position: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch, sequence) = token_ids.dims2()?;
        if batch != 1 {
            return Err(DlirError::InvalidConfig(format!(
                "v0.1 requires batch size one, received {batch}"
            )));
        }
        if sequence == 0 {
            return Err(DlirError::InvalidConfig(
                "a model forward pass requires at least one token".into(),
            ));
        }
        if position != cache.len() {
            return Err(DlirError::InvalidConfig(format!(
                "forward position {position} does not match cache length {}",
                cache.len()
            )));
        }
        if position + sequence > cache.capacity() {
            return Err(DlirError::CacheCapacityExceeded {
                attempted: position + sequence,
                capacity: cache.capacity(),
            });
        }
        let mut hidden = self.embeddings.forward(token_ids)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, position, layer, &self.rope, cache)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
        Ok(self.lm_head.forward(&last)?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::llama::{
        Cache as OracleCache, Config as OracleConfig, Llama as OracleLlama,
    };
    use std::collections::HashMap;

    fn config(tied: bool) -> ModelConfig {
        ModelConfig {
            vocab_size: 19,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 8,
            bos_token_id: 1,
            eos_token_id: 2,
            tie_word_embeddings: tied,
        }
    }

    fn oracle_config(config: &ModelConfig) -> OracleConfig {
        OracleConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            use_flash_attn: false,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta as f32,
            bos_token_id: Some(config.bos_token_id),
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: config.max_position_embeddings,
            tie_word_embeddings: config.tie_word_embeddings,
        }
    }

    fn fixture(config: &ModelConfig, device: &Device) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        let mut seed = 1usize;
        let mut add = |name: String, shape: Vec<usize>, norm: bool| {
            let count = shape.iter().product();
            let values = (0..count)
                .map(|index| {
                    if norm {
                        0.9 + ((index + seed) % 17) as f32 / 100.0
                    } else {
                        (((index * 29 + seed * 13) % 101) as f32 - 50.0) / 250.0
                    }
                })
                .collect::<Vec<_>>();
            tensors.insert(name, Tensor::from_vec(values, shape, device).unwrap());
            seed += 1;
        };
        add(
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, config.hidden_size],
            false,
        );
        add("model.norm.weight".into(), vec![config.hidden_size], true);
        if !config.tie_word_embeddings {
            add(
                "lm_head.weight".into(),
                vec![config.vocab_size, config.hidden_size],
                false,
            );
        }
        let kv_size = config.num_key_value_heads * config.head_dim().unwrap();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            add(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                false,
            );
            for projection in ["k_proj", "v_proj"] {
                add(
                    format!("{prefix}.self_attn.{projection}.weight"),
                    vec![kv_size, config.hidden_size],
                    false,
                );
            }
            add(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                false,
            );
            for projection in ["gate_proj", "up_proj"] {
                add(
                    format!("{prefix}.mlp.{projection}.weight"),
                    vec![config.intermediate_size, config.hidden_size],
                    false,
                );
            }
            add(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![config.hidden_size, config.intermediate_size],
                false,
            );
            add(
                format!("{prefix}.input_layernorm.weight"),
                vec![config.hidden_size],
                true,
            );
            add(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![config.hidden_size],
                true,
            );
        }
        tensors
    }

    fn assert_close(left: &Tensor, right: &Tensor) {
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(left.len(), right.len());
        let maximum = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0f32, f32::max);
        assert!(maximum <= 1e-4, "maximum logit difference was {maximum}");
    }

    fn compare_with_oracle(tied: bool) {
        let device = Device::Cpu;
        let config = config(tied);
        let tensors = fixture(&config, &device);
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
        let model = Llama::load(vb, &config, 8).unwrap();
        let mut cache = KvCache::new(&config, 8, DType::F32, &device).unwrap();

        let oracle_config = oracle_config(&config);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let oracle = OracleLlama::load(vb, &oracle_config).unwrap();
        let mut oracle_cache = OracleCache::new(true, DType::F32, &oracle_config, &device).unwrap();

        let prompt = Tensor::new(&[[1u32, 3, 5]], &device).unwrap();
        let ours = model.forward(&prompt, 0, &mut cache).unwrap();
        let expected = oracle.forward(&prompt, 0, &mut oracle_cache).unwrap();
        assert_close(&ours, &expected);

        let token = Tensor::new(&[[7u32]], &device).unwrap();
        let ours = model.forward(&token, 3, &mut cache).unwrap();
        let expected = oracle.forward(&token, 3, &mut oracle_cache).unwrap();
        assert_close(&ours, &expected);

        let vb = VarBuilder::from_tensors(fixture(&config, &device), DType::F32, &device);
        let recompute_model = Llama::load(vb, &config, 8).unwrap();
        let mut recompute_cache = KvCache::new(&config, 8, DType::F32, &device).unwrap();
        let full = Tensor::new(&[[1u32, 3, 5, 7]], &device).unwrap();
        let recomputed = recompute_model
            .forward(&full, 0, &mut recompute_cache)
            .unwrap();
        assert_close(&ours, &recomputed);
    }

    #[test]
    fn grouped_query_cached_logits_match_candle_oracle_with_tied_head() {
        compare_with_oracle(true);
    }

    #[test]
    fn grouped_query_cached_logits_match_candle_oracle_with_untied_head() {
        compare_with_oracle(false);
    }

    #[test]
    fn rejects_cache_overflow_and_position_mismatch() {
        let device = Device::Cpu;
        let config = config(true);
        let vb = VarBuilder::from_tensors(fixture(&config, &device), DType::F32, &device);
        let model = Llama::load(vb, &config, 3).unwrap();
        let mut cache = KvCache::new(&config, 3, DType::F32, &device).unwrap();
        let prompt = Tensor::new(&[[1u32, 3, 5]], &device).unwrap();
        model.forward(&prompt, 0, &mut cache).unwrap();
        let token = Tensor::new(&[[7u32]], &device).unwrap();
        assert!(matches!(
            model.forward(&token, 3, &mut cache),
            Err(DlirError::CacheCapacityExceeded { .. })
        ));

        let vb = VarBuilder::from_tensors(fixture(&config, &device), DType::F32, &device);
        let model = Llama::load(vb, &config, 3).unwrap();
        let mut cache = KvCache::new(&config, 3, DType::F32, &device).unwrap();
        assert!(matches!(
            model.forward(&token, 1, &mut cache),
            Err(DlirError::InvalidConfig(_))
        ));
    }
}
