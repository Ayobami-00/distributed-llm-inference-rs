//! Tensor-sharded construction of the owned Llama forward path.

use super::{
    KvCache,
    llama::{RotaryEmbedding, apply_causal_mask, repeat_kv},
};
use crate::{
    DlirError, ModelConfig, ParallelContext, Result, RowParallelLinear, VocabParallelEmbedding,
    VocabParallelLmHead, plan_tensor_shard,
};
use candle_core::{DType, IndexOp, Shape, Tensor};
use candle_nn::var_builder::{Shard, ShardedVarBuilder};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};
use dlir_collectives::{AllReduceAlgorithm, CollectiveCommunicator};
use std::time::{Duration, Instant};

/// Observes rank-local tensor-parallel transformer-layer boundaries.
pub trait TensorParallelObserver {
    /// Called immediately before every rank executes one transformer layer.
    fn layer_started(&mut self, layer: usize);

    /// Called after local compute and the layer's attention/MLP collectives complete.
    fn layer_completed(&mut self, layer: usize, duration: Duration);
}

/// Tensor-parallel layer observer that discards notifications.
#[derive(Default)]
pub struct NoopTensorParallelObserver;

impl TensorParallelObserver for NoopTensorParallelObserver {
    fn layer_started(&mut self, _layer: usize) {}

    fn layer_completed(&mut self, _layer: usize, _duration: Duration) {}
}

trait WeightSource {
    fn tensor(
        &self,
        name: &str,
        shape: &[usize],
        shard_dim: Option<usize>,
        parallel: ParallelContext,
    ) -> Result<Tensor>;
    fn dtype(&self) -> DType;
    fn device(&self) -> &candle_core::Device;
}

struct FullWeights<'a>(VarBuilder<'a>);

impl WeightSource for FullWeights<'_> {
    fn tensor(
        &self,
        name: &str,
        shape: &[usize],
        shard_dim: Option<usize>,
        parallel: ParallelContext,
    ) -> Result<Tensor> {
        let tensor = self.0.get(Shape::from(shape.to_vec()), name)?;
        if let Some(dim) = shard_dim {
            let length = shape[dim] / parallel.tp_size();
            Ok(tensor
                .narrow(dim, parallel.tp_rank() * length, length)?
                .contiguous()?)
        } else {
            Ok(tensor)
        }
    }

    fn dtype(&self) -> DType {
        self.0.dtype()
    }
    fn device(&self) -> &candle_core::Device {
        self.0.device()
    }
}

struct MmapShardedWeights<'a>(ShardedVarBuilder<'a>);

impl WeightSource for MmapShardedWeights<'_> {
    fn tensor(
        &self,
        name: &str,
        shape: &[usize],
        shard_dim: Option<usize>,
        parallel: ParallelContext,
    ) -> Result<Tensor> {
        let hint = shard_dim.map_or_else(Shard::default, |dim| Shard {
            dim,
            rank: parallel.tp_rank(),
            world_size: parallel.tp_size(),
        });
        Ok(self
            .0
            .get_with_hints(Shape::from(shape.to_vec()), name, hint)?)
    }

    fn dtype(&self) -> DType {
        self.0.dtype()
    }
    fn device(&self) -> &candle_core::Device {
        self.0.device()
    }
}

#[derive(Debug)]
struct TensorParallelAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: RowParallelLinear,
    local_query_heads: usize,
    local_kv_heads: usize,
    head_dim: usize,
}

impl TensorParallelAttention {
    fn load(
        source: &impl WeightSource,
        prefix: &str,
        config: &ModelConfig,
        parallel: ParallelContext,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        let head_dim = config.head_dim()?;
        let kv_width = config.num_key_value_heads * head_dim;
        let q = source.tensor(
            &format!("{prefix}.q_proj.weight"),
            &[config.hidden_size, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let k = source.tensor(
            &format!("{prefix}.k_proj.weight"),
            &[kv_width, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let v = source.tensor(
            &format!("{prefix}.v_proj.weight"),
            &[kv_width, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let o = source.tensor(
            &format!("{prefix}.o_proj.weight"),
            &[config.hidden_size, config.hidden_size],
            Some(1),
            parallel,
        )?;
        Ok(Self {
            q_proj: Linear::new(q, None),
            k_proj: Linear::new(k, None),
            v_proj: Linear::new(v, None),
            o_proj: RowParallelLinear::from_weight(o, algorithm)?,
            local_query_heads: config.num_attention_heads / parallel.tp_size(),
            local_kv_heads: config.num_key_value_heads / parallel.tp_size(),
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
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        let (batch, sequence, _) = input.dims3()?;
        let q = self
            .q_proj
            .forward(input)?
            .reshape((batch, sequence, self.local_query_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(input)?
            .reshape((batch, sequence, self.local_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(input)?
            .reshape((batch, sequence, self.local_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = rope.apply(&q, position)?;
        let k = rope.apply(&k, position)?;
        let (k, v) = cache.append(layer, &k, &v)?;
        let repetitions = self.local_query_heads / self.local_kv_heads;
        let k = repeat_kv(&k, repetitions)?;
        let v = repeat_kv(&v, repetitions)?;
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
        let local = probabilities
            .matmul(&v.contiguous()?)?
            .to_dtype(input.dtype())?
            .transpose(1, 2)?
            .reshape((batch, sequence, self.local_query_heads * self.head_dim))?;
        self.o_proj.forward(&local, collectives)
    }
}

#[derive(Debug)]
struct TensorParallelMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: RowParallelLinear,
}

impl TensorParallelMlp {
    fn load(
        source: &impl WeightSource,
        prefix: &str,
        config: &ModelConfig,
        parallel: ParallelContext,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        let gate = source.tensor(
            &format!("{prefix}.gate_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let up = source.tensor(
            &format!("{prefix}.up_proj.weight"),
            &[config.intermediate_size, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let down = source.tensor(
            &format!("{prefix}.down_proj.weight"),
            &[config.hidden_size, config.intermediate_size],
            Some(1),
            parallel,
        )?;
        Ok(Self {
            gate_proj: Linear::new(gate, None),
            up_proj: Linear::new(up, None),
            down_proj: RowParallelLinear::from_weight(down, algorithm)?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        let gated = candle_nn::ops::silu(&self.gate_proj.forward(input)?)?;
        let local = (gated * self.up_proj.forward(input)?)?;
        self.down_proj.forward(&local, collectives)
    }
}

#[derive(Debug)]
struct TensorParallelBlock {
    input_norm: RmsNorm,
    attention: TensorParallelAttention,
    post_attention_norm: RmsNorm,
    mlp: TensorParallelMlp,
}

impl TensorParallelBlock {
    fn load(
        source: &impl WeightSource,
        layer: usize,
        config: &ModelConfig,
        parallel: ParallelContext,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{layer}");
        let input_norm = source.tensor(
            &format!("{prefix}.input_layernorm.weight"),
            &[config.hidden_size],
            None,
            parallel,
        )?;
        let post_norm = source.tensor(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[config.hidden_size],
            None,
            parallel,
        )?;
        Ok(Self {
            input_norm: RmsNorm::new(input_norm, config.rms_norm_eps),
            attention: TensorParallelAttention::load(
                source,
                &format!("{prefix}.self_attn"),
                config,
                parallel,
                algorithm,
            )?,
            post_attention_norm: RmsNorm::new(post_norm, config.rms_norm_eps),
            mlp: TensorParallelMlp::load(
                source,
                &format!("{prefix}.mlp"),
                config,
                parallel,
                algorithm,
            )?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        position: usize,
        layer: usize,
        rope: &RotaryEmbedding,
        cache: &mut KvCache,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        let attention = self.attention.forward(
            &self.input_norm.forward(input)?,
            position,
            layer,
            rope,
            cache,
            collectives,
        )?;
        let hidden = (input + attention)?;
        Ok((&hidden
            + self
                .mlp
                .forward(&self.post_attention_norm.forward(&hidden)?, collectives)?)?)
    }
}

/// Existing owned Llama graph constructed with rank-local tensor shards.
///
/// Every rank executes every transformer layer. Q/K/V, output projections, MLP projections,
/// embeddings, logits, and KV state are sharded; RMSNorm weights are replicated.
#[derive(Debug)]
pub struct TensorParallelLlama {
    parallel: ParallelContext,
    embeddings: VocabParallelEmbedding,
    blocks: Vec<TensorParallelBlock>,
    final_norm: RmsNorm,
    lm_head: VocabParallelLmHead,
    rope: RotaryEmbedding,
}

impl TensorParallelLlama {
    /// Loads from ordinary full tensors, then slices them locally. Intended for offline fixtures.
    pub fn load(
        vb: VarBuilder<'_>,
        config: &ModelConfig,
        parallel: ParallelContext,
        capacity: usize,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        Self::load_from(&FullWeights(vb), config, parallel, capacity, algorithm)
    }

    /// Loads only local slices from Candle's mmap-backed sharded safetensor builder.
    pub fn load_sharded(
        vb: ShardedVarBuilder<'_>,
        config: &ModelConfig,
        parallel: ParallelContext,
        capacity: usize,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        Self::load_from(
            &MmapShardedWeights(vb),
            config,
            parallel,
            capacity,
            algorithm,
        )
    }

    fn load_from(
        source: &impl WeightSource,
        config: &ModelConfig,
        parallel: ParallelContext,
        capacity: usize,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Self> {
        let plan = plan_tensor_shard(config, parallel)?;
        let embedding_weight = source.tensor(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
            Some(0),
            parallel,
        )?;
        let embeddings = VocabParallelEmbedding::from_weight(
            embedding_weight.clone(),
            plan.vocabulary,
            config.vocab_size,
        )?
        .with_all_reduce(algorithm);
        let blocks = (0..config.num_hidden_layers)
            .map(|layer| TensorParallelBlock::load(source, layer, config, parallel, algorithm))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = RmsNorm::new(
            source.tensor("model.norm.weight", &[config.hidden_size], None, parallel)?,
            config.rms_norm_eps,
        );
        let head_weight = if config.tie_word_embeddings {
            embedding_weight
        } else {
            source.tensor(
                "lm_head.weight",
                &[config.vocab_size, config.hidden_size],
                Some(0),
                parallel,
            )?
        };
        let lm_head = VocabParallelLmHead::from_weight(head_weight, config.vocab_size)?;
        let rope = RotaryEmbedding::new(config, capacity, source.dtype(), source.device())?;
        Ok(Self {
            parallel,
            embeddings,
            blocks,
            final_norm,
            lm_head,
            rope,
        })
    }

    /// Runs prefill or one cached decode step and returns gathered F32 logits on every rank.
    pub fn forward(
        &self,
        token_ids: &Tensor,
        position: usize,
        cache: &mut KvCache,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        self.forward_observed(
            token_ids,
            position,
            cache,
            collectives,
            &mut NoopTensorParallelObserver,
        )
    }

    /// Runs prefill or decode while publishing each rank-local layer boundary.
    pub fn forward_observed(
        &self,
        token_ids: &Tensor,
        position: usize,
        cache: &mut KvCache,
        collectives: &mut dyn CollectiveCommunicator,
        observer: &mut dyn TensorParallelObserver,
    ) -> Result<Tensor> {
        if collectives.rank().global_rank() != self.parallel.tp_rank()
            || collectives.rank().world_size() != self.parallel.tp_size()
        {
            return Err(DlirError::InvalidConfig(
                "collective world and ParallelContext disagree".into(),
            ));
        }
        let (batch, sequence) = token_ids.dims2()?;
        if batch != 1 || sequence == 0 {
            return Err(DlirError::InvalidConfig(format!(
                "TP token input must have shape [1,S] with S>0, got {:?}",
                token_ids.dims()
            )));
        }
        if position != cache.len() || cache.layer_count() != self.blocks.len() {
            return Err(DlirError::InvalidConfig(
                "TP cache position or layer count mismatch".into(),
            ));
        }
        if position + sequence > cache.capacity() {
            return Err(DlirError::CacheCapacityExceeded {
                attempted: position + sequence,
                capacity: cache.capacity(),
            });
        }
        let mut hidden = self.embeddings.forward(token_ids, collectives)?;
        for (layer, block) in self.blocks.iter().enumerate() {
            observer.layer_started(layer);
            let started = Instant::now();
            hidden = block.forward(&hidden, position, layer, &self.rope, cache, collectives)?;
            observer.layer_completed(layer, started.elapsed());
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let last = hidden.i((.., sequence - 1, ..))?.contiguous()?;
        Ok(self
            .lm_head
            .forward(&last, collectives)?
            .to_dtype(DType::F32)?)
    }

    /// Returns this model's tensor-process context.
    pub const fn parallel_context(&self) -> ParallelContext {
        self.parallel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Llama;
    use candle_core::Device;
    use dlir_collectives::{CollectivesError, NativeCollectives, run_in_memory};
    use std::{collections::HashMap, time::Duration};

    fn config(tied: bool) -> ModelConfig {
        ModelConfig {
            vocab_size: 20,
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            max_position_embeddings: 8,
            bos_token_id: 1,
            eos_token_id: 2,
            tie_word_embeddings: tied,
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
                        0.9 + ((index + seed) % 17) as f32 / 100.
                    } else {
                        (((index * 29 + seed * 13) % 101) as f32 - 50.) / 250.
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
        let kv_width = config.num_key_value_heads * config.head_dim().unwrap();
        for layer in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            add(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                false,
            );
            for name in ["k_proj", "v_proj"] {
                add(
                    format!("{prefix}.self_attn.{name}.weight"),
                    vec![kv_width, config.hidden_size],
                    false,
                );
            }
            add(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![config.hidden_size, config.hidden_size],
                false,
            );
            for name in ["gate_proj", "up_proj"] {
                add(
                    format!("{prefix}.mlp.{name}.weight"),
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

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert_eq!(actual.len(), expected.len());
        assert!(maximum <= 1e-4, "maximum logit difference was {maximum}");
    }

    fn compare(tied: bool, algorithm: AllReduceAlgorithm) {
        let device = Device::Cpu;
        let config = config(tied);
        let tensors = fixture(&config, &device);
        let serial = Llama::load(
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
            &config,
            8,
        )
        .unwrap();
        let mut serial_cache = KvCache::new(&config, 8, DType::F32, &device).unwrap();
        let prompt = Tensor::new(&[[1u32, 3, 5]], &device).unwrap();
        let token = Tensor::new(&[[7u32]], &device).unwrap();
        let expected_prefill = values(&serial.forward(&prompt, 0, &mut serial_cache).unwrap());
        let expected_decode = values(&serial.forward(&token, 3, &mut serial_cache).unwrap());

        let results = run_in_memory(2, Duration::from_secs(3), move |communicator| {
            let rank = communicator.rank().global_rank();
            let parallel = ParallelContext::tensor_parallel(rank, 2)
                .map_err(|error| CollectivesError::Collective(error.to_string()))?;
            let model = TensorParallelLlama::load(
                VarBuilder::from_tensors(tensors.clone(), DType::F32, &device),
                &config,
                parallel,
                8,
                algorithm,
            )
            .map_err(|error| CollectivesError::Collective(error.to_string()))?;
            let mut cache = KvCache::new_tensor_parallel(&config, 2, 8, DType::F32, &device)
                .map_err(|error| CollectivesError::Collective(error.to_string()))?;
            let mut native = NativeCollectives::new(communicator);
            let prefill = model
                .forward(&prompt, 0, &mut cache, &mut native)
                .map_err(|error| CollectivesError::Collective(error.to_string()))?;
            let decode = model
                .forward(&token, 3, &mut cache, &mut native)
                .map_err(|error| CollectivesError::Collective(error.to_string()))?;
            Ok((values(&prefill), values(&decode), native.take_traces()))
        })
        .unwrap();

        for (prefill, decode, traces) in results {
            assert_close(&prefill, &expected_prefill);
            assert_close(&decode, &expected_decode);
            // embedding + two per layer + logits, once for prefill and once for decode.
            assert_eq!(traces.len(), 12);
        }
    }

    #[test]
    fn tp2_prefill_and_cached_decode_match_owned_llama() {
        for tied in [true, false] {
            compare(tied, AllReduceAlgorithm::Centralized);
            compare(tied, AllReduceAlgorithm::Ring);
        }
    }
}
