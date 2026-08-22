//! Reusable tensor-parallel topology and sharded linear/embedding primitives.

use crate::{DlirError, ModelConfig, Result};
use candle_core::{DType, Tensor};
use candle_nn::{Embedding, Linear, Module};
use dlir_collectives::{AllReduceAlgorithm, CollectiveCommunicator, ReduceOp};
use serde::{Deserialize, Serialize};

/// Coordinates one rank's tensor, pipeline, and expert process-group identities.
///
/// v0.5 uses only the tensor dimensions: `tp_rank=global_rank`, `tp_size=world_size`, while
/// pipeline and expert sizes remain one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelContext {
    tp_rank: usize,
    tp_size: usize,
    pp_rank: usize,
    pp_size: usize,
    ep_rank: usize,
    ep_size: usize,
}

impl ParallelContext {
    /// Constructs the v0.5 TP-only context for one rank.
    pub fn tensor_parallel(tp_rank: usize, tp_size: usize) -> Result<Self> {
        let context = Self {
            tp_rank,
            tp_size,
            pp_rank: 0,
            pp_size: 1,
            ep_rank: 0,
            ep_size: 1,
        };
        context.validate()?;
        Ok(context)
    }

    /// Validates nonzero group sizes and rank bounds.
    pub fn validate(self) -> Result<()> {
        for (name, rank, size) in [
            ("tensor", self.tp_rank, self.tp_size),
            ("pipeline", self.pp_rank, self.pp_size),
            ("expert", self.ep_rank, self.ep_size),
        ] {
            if size == 0 || rank >= size {
                return Err(DlirError::InvalidConfig(format!(
                    "{name} rank {rank} is outside group size {size}"
                )));
            }
        }
        Ok(())
    }

    /// Tensor-parallel rank.
    pub const fn tp_rank(self) -> usize {
        self.tp_rank
    }
    /// Tensor-parallel world size.
    pub const fn tp_size(self) -> usize {
        self.tp_size
    }
    /// Pipeline-parallel rank.
    pub const fn pp_rank(self) -> usize {
        self.pp_rank
    }
    /// Pipeline-parallel world size.
    pub const fn pp_size(self) -> usize {
        self.pp_size
    }
    /// Expert-parallel rank.
    pub const fn ep_rank(self) -> usize {
        self.ep_rank
    }
    /// Expert-parallel world size.
    pub const fn ep_size(self) -> usize {
        self.ep_size
    }
}

/// Half-open equal shard range within one global tensor dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardRange {
    /// Inclusive global start index.
    pub start: usize,
    /// Exclusive global end index.
    pub end: usize,
}

impl ShardRange {
    /// Returns the number of local elements.
    pub const fn len(self) -> usize {
        self.end - self.start
    }
    /// Returns whether the range is empty.
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Exact model-dimension shard assignment for one tensor rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShardPlan {
    /// Rank context.
    pub parallel: ParallelContext,
    /// Vocabulary rows owned by this rank.
    pub vocabulary: ShardRange,
    /// Query heads owned by this rank.
    pub query_heads: ShardRange,
    /// Compact GQA key/value heads owned by this rank.
    pub kv_heads: ShardRange,
    /// SwiGLU intermediate features owned by this rank.
    pub intermediate: ShardRange,
    /// Architectural checkpoint parameters, before cross-rank replication.
    pub architectural_parameters: u64,
    /// Parameters physically materialized by this rank.
    pub local_parameters: u64,
    /// RMSNorm parameters replicated on every rank.
    pub replicated_parameters: u64,
    /// Sum of local materialized parameters across all ranks.
    pub aggregate_materialized_parameters: u64,
}

/// Validates strict equal TP sharding and constructs one rank's exact ranges and parameter plan.
pub fn plan_tensor_shard(
    config: &ModelConfig,
    parallel: ParallelContext,
) -> Result<TensorShardPlan> {
    parallel.validate()?;
    if parallel.pp_size != 1 || parallel.ep_size != 1 {
        return Err(DlirError::InvalidConfig(
            "v0.5 requires PP=1 and EP=1".into(),
        ));
    }
    let tp = parallel.tp_size;
    for (name, value) in [
        ("vocabulary size", config.vocab_size),
        ("hidden width", config.hidden_size),
        ("intermediate width", config.intermediate_size),
        ("query heads", config.num_attention_heads),
        ("KV heads", config.num_key_value_heads),
    ] {
        if value % tp != 0 {
            return Err(DlirError::InvalidConfig(format!(
                "{name} {value} is not divisible by TP={tp}"
            )));
        }
    }
    let head_dim = config.head_dim()?;
    let range = |length: usize| {
        let local = length / tp;
        ShardRange {
            start: parallel.tp_rank * local,
            end: (parallel.tp_rank + 1) * local,
        }
    };
    let h = config.hidden_size as u64;
    let i = config.intermediate_size as u64;
    let v = config.vocab_size as u64;
    let k = config.num_key_value_heads as u64;
    let d = head_dim as u64;
    let l = config.num_hidden_layers as u64;
    let embeddings = v * h;
    let attention = h * h + 2 * k * d * h + h * h;
    let mlp = 3 * h * i;
    let replicated_parameters = 2 * l * h + h;
    let lm_head = if config.tie_word_embeddings { 0 } else { v * h };
    let architectural_parameters = embeddings + l * (attention + mlp + 2 * h) + h + lm_head;
    let sharded_parameters = architectural_parameters - replicated_parameters;
    let local_parameters = sharded_parameters / tp as u64 + replicated_parameters;
    let aggregate_materialized_parameters = local_parameters * tp as u64;
    Ok(TensorShardPlan {
        parallel,
        vocabulary: range(config.vocab_size),
        query_heads: range(config.num_attention_heads),
        kv_heads: range(config.num_key_value_heads),
        intermediate: range(config.intermediate_size),
        architectural_parameters,
        local_parameters,
        replicated_parameters,
        aggregate_materialized_parameters,
    })
}

/// Column-sharded linear transform whose local weight is `[O/TP, I]`.
#[derive(Debug)]
pub struct ColumnParallelLinear {
    linear: Linear,
    global_output_features: usize,
    gather_output: bool,
}

impl ColumnParallelLinear {
    /// Constructs a column-parallel linear layer from one rank-local row shard.
    pub fn from_weight(
        weight: Tensor,
        global_output_features: usize,
        gather_output: bool,
    ) -> Result<Self> {
        let (local_output, _) = weight.dims2()?;
        if local_output == 0 || global_output_features % local_output != 0 {
            return Err(DlirError::InvalidConfig(format!(
                "local output width {local_output} does not evenly shard global width {global_output_features}"
            )));
        }
        Ok(Self {
            linear: Linear::new(weight, None),
            global_output_features,
            gather_output,
        })
    }

    /// Computes the local features and optionally all-gathers the final feature axis.
    pub fn forward(
        &self,
        input: &Tensor,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        let local = self.linear.forward(input)?;
        if self.gather_output {
            let axis = local
                .rank()
                .checked_sub(1)
                .ok_or_else(|| DlirError::InvalidConfig("linear output cannot be scalar".into()))?;
            let gathered = collectives.all_gather(&local, axis)?;
            if gathered.dim(axis)? != self.global_output_features {
                return Err(DlirError::InvalidConfig(
                    "column-parallel gather produced the wrong output width".into(),
                ));
            }
            Ok(gathered)
        } else {
            Ok(local)
        }
    }

    /// Returns this rank's output-feature width.
    pub fn local_output_features(&self) -> usize {
        self.linear.weight().dims()[0]
    }
}

/// Row-sharded linear transform whose local weight is `[O, I/TP]`.
#[derive(Debug)]
pub struct RowParallelLinear {
    linear: Linear,
    algorithm: AllReduceAlgorithm,
}

impl RowParallelLinear {
    /// Constructs a row-parallel linear from one input-column weight shard.
    pub fn from_weight(weight: Tensor, algorithm: AllReduceAlgorithm) -> Result<Self> {
        weight.dims2()?;
        Ok(Self {
            linear: Linear::new(weight, None),
            algorithm,
        })
    }

    /// Computes a partial full-width result and sums it across tensor ranks.
    pub fn forward(
        &self,
        local_input: &Tensor,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        let partial = self.linear.forward(local_input)?;
        Ok(collectives.all_reduce(&partial, ReduceOp::Sum, self.algorithm)?)
    }
}

/// Vocabulary-row-sharded embedding with zero masking and all-reduce reconstruction.
#[derive(Debug)]
pub struct VocabParallelEmbedding {
    embedding: Embedding,
    range: ShardRange,
    global_vocabulary: usize,
    algorithm: AllReduceAlgorithm,
}

impl VocabParallelEmbedding {
    /// Constructs an embedding from local `[V/TP,H]` rows and their global range.
    pub fn from_weight(
        weight: Tensor,
        range: ShardRange,
        global_vocabulary: usize,
    ) -> Result<Self> {
        let (rows, hidden) = weight.dims2()?;
        if rows != range.len() || range.end > global_vocabulary {
            return Err(DlirError::InvalidConfig(
                "embedding weight and vocabulary shard range disagree".into(),
            ));
        }
        Ok(Self {
            embedding: Embedding::new(weight, hidden),
            range,
            global_vocabulary,
            algorithm: AllReduceAlgorithm::Centralized,
        })
    }

    /// Selects the native all-reduce used to reconstruct the residual-stream embedding.
    pub const fn with_all_reduce(mut self, algorithm: AllReduceAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Looks up locally owned IDs, masks all other IDs to zero, then all-reduces embeddings.
    pub fn forward(
        &self,
        token_ids: &Tensor,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        if token_ids.dtype() != DType::U32 {
            return Err(DlirError::InvalidConfig("token IDs must use u32".into()));
        }
        let (batch, sequence) = token_ids.dims2()?;
        let ids = token_ids.flatten_all()?.to_vec1::<u32>()?;
        let mut local_ids = Vec::with_capacity(ids.len());
        let mut mask = Vec::with_capacity(ids.len());
        for id in ids {
            let id = id as usize;
            if id >= self.global_vocabulary {
                return Err(DlirError::InvalidConfig(format!(
                    "token ID {id} is outside vocabulary {}",
                    self.global_vocabulary
                )));
            }
            let owned = id >= self.range.start && id < self.range.end;
            local_ids.push(if owned {
                (id - self.range.start) as u32
            } else {
                0
            });
            mask.push(f32::from(owned));
        }
        let local_ids = Tensor::from_vec(local_ids, (batch, sequence), token_ids.device())?;
        let local = self.embedding.forward(&local_ids)?;
        let mask = Tensor::from_vec(mask, (batch, sequence, 1), token_ids.device())?;
        let local = local.broadcast_mul(&mask)?;
        Ok(collectives.all_reduce(&local, ReduceOp::Sum, self.algorithm)?)
    }
}

/// Vocabulary-sharded language-model head that gathers full logits in rank order.
#[derive(Debug)]
pub struct VocabParallelLmHead(ColumnParallelLinear);

impl VocabParallelLmHead {
    /// Constructs the head from local vocabulary rows.
    pub fn from_weight(weight: Tensor, global_vocabulary: usize) -> Result<Self> {
        Ok(Self(ColumnParallelLinear::from_weight(
            weight,
            global_vocabulary,
            true,
        )?))
    }

    /// Computes local logits and all-gathers the vocabulary axis.
    pub fn forward(
        &self,
        hidden: &Tensor,
        collectives: &mut dyn CollectiveCommunicator,
    ) -> Result<Tensor> {
        self.0.forward(hidden, collectives)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SupportedModelId;
    use candle_core::Device;
    use dlir_collectives::{CollectivesError, NativeCollectives, run_in_memory};
    use std::time::Duration;

    fn flat(tensor: &Tensor) -> Vec<f32> {
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn collective<T>(result: Result<T>) -> dlir_collectives::Result<T> {
        result.map_err(|error| CollectivesError::Collective(error.to_string()))
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(left, right)| (left - right).abs() <= 1e-5)
        );
    }

    #[test]
    fn registered_models_enforce_strict_gqa_tp_divisibility() {
        let smol = &SupportedModelId::SmolLm2_135MInstruct.spec().config;
        assert!(plan_tensor_shard(smol, ParallelContext::tensor_parallel(0, 3).unwrap()).is_ok());
        for tp in [2, 4] {
            assert!(
                plan_tensor_shard(smol, ParallelContext::tensor_parallel(0, tp).unwrap()).is_err()
            );
        }
        let tiny = &SupportedModelId::TinyLlama1_1BChat.spec().config;
        for tp in [2, 4] {
            assert!(
                plan_tensor_shard(tiny, ParallelContext::tensor_parallel(0, tp).unwrap()).is_ok()
            );
        }
    }

    #[test]
    fn shard_parameter_accounting_matches_registry_and_norm_replication() {
        for (model, tp) in [
            (SupportedModelId::SmolLm2_135MInstruct, 3),
            (SupportedModelId::TinyLlama1_1BChat, 4),
        ] {
            let spec = model.spec();
            let plan = plan_tensor_shard(
                &spec.config,
                ParallelContext::tensor_parallel(0, tp).unwrap(),
            )
            .unwrap();
            assert_eq!(plan.architectural_parameters, spec.expected_parameters);
            assert_eq!(
                plan.aggregate_materialized_parameters,
                plan.architectural_parameters + plan.replicated_parameters * (tp as u64 - 1)
            );
        }
    }

    #[test]
    fn parallel_linears_embedding_and_head_match_monolithic_results() {
        let device = Device::Cpu;
        let column_weight = Tensor::from_vec(
            (0..24).map(|value| value as f32 / 10.).collect::<Vec<_>>(),
            (6, 4),
            &device,
        )
        .unwrap();
        let row_weight = Tensor::from_vec(
            (0..12).map(|value| value as f32 / 7.).collect::<Vec<_>>(),
            (3, 4),
            &device,
        )
        .unwrap();
        let embedding_weight = Tensor::from_vec(
            (0..18).map(|value| value as f32 / 5.).collect::<Vec<_>>(),
            (6, 3),
            &device,
        )
        .unwrap();
        let input = Tensor::new(&[[1f32, 2., 3., 4.]], &device).unwrap();
        let row_input = input.clone();
        let token_ids = Tensor::new(&[[0u32, 4, 2]], &device).unwrap();

        let expected_column = Linear::new(column_weight.clone(), None)
            .forward(&input)
            .unwrap();
        let expected_row = Linear::new(row_weight.clone(), None)
            .forward(&row_input)
            .unwrap();
        let expected_embedding = Embedding::new(embedding_weight.clone(), 3)
            .forward(&token_ids)
            .unwrap();

        let results = run_in_memory(2, Duration::from_secs(2), move |communicator| {
            let rank = communicator.rank().global_rank();
            let mut native = NativeCollectives::new(communicator);
            let column = collective(ColumnParallelLinear::from_weight(
                column_weight.narrow(0, rank * 3, 3)?,
                6,
                true,
            ))?;
            let actual_column = collective(column.forward(&input, &mut native))?;

            let row = collective(RowParallelLinear::from_weight(
                row_weight.narrow(1, rank * 2, 2)?.contiguous()?,
                AllReduceAlgorithm::Centralized,
            ))?;
            let local_input = row_input.narrow(1, rank * 2, 2)?.contiguous()?;
            let actual_row = collective(row.forward(&local_input, &mut native))?;

            let range = ShardRange {
                start: rank * 3,
                end: (rank + 1) * 3,
            };
            let embedding = collective(VocabParallelEmbedding::from_weight(
                embedding_weight.narrow(0, range.start, range.len())?,
                range,
                6,
            ))?;
            let actual_embedding = collective(embedding.forward(&token_ids, &mut native))?;

            let head = collective(VocabParallelLmHead::from_weight(
                column_weight.narrow(0, rank * 3, 3)?,
                6,
            ))?;
            let actual_head = collective(head.forward(&input, &mut native))?;
            Ok((
                flat(&actual_column),
                flat(&actual_row),
                flat(&actual_embedding),
                flat(&actual_head),
            ))
        })
        .unwrap();

        for result in results {
            assert_close(&result.0, &flat(&expected_column));
            assert_close(&result.1, &flat(&expected_row));
            assert_close(&result.2, &flat(&expected_embedding));
            assert_close(&result.3, &flat(&expected_column));
        }
    }
}
