//! Strict equal tensor-shard and persistent-memory planning.

use crate::{Result, TensorParallelError};
use dlir_runtime::{
    ParallelContext, PlacementVerdict, PlanDType, SupportedModelId, TensorShardPlan,
    plan_tensor_shard,
};
use serde::{Deserialize, Serialize};

/// Rank-local tensor ranges and logical persistent-state accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorParallelMemoryPlan {
    /// Global rank.
    pub rank: usize,
    /// Exact tensor dimension shards and parameter counts.
    pub shard: TensorShardPlan,
    /// Runtime dtype; F32 in v0.5.
    pub dtype: PlanDType,
    /// KV capacity in token positions.
    pub context_capacity: usize,
    /// Local sharded and replicated weight bytes.
    pub local_weight_bytes: u64,
    /// Local compact-GQA KV cache capacity bytes.
    pub kv_cache_capacity_bytes: u64,
    /// Weight plus cache estimate.
    pub persistent_bytes: u64,
    /// Enforced container memory limit used for placement.
    pub budget_bytes: Option<u64>,
    /// Exact boundary verdict; equality fits.
    pub placement: PlacementVerdict,
}

/// Complete equal tensor partition for one registered model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorParallelPartition {
    /// Closed-registry model.
    pub model: SupportedModelId,
    /// Tensor rank count.
    pub tensor_parallel: usize,
    /// Rank-ordered plans.
    pub ranks: Vec<TensorParallelMemoryPlan>,
    /// Architectural parameter count.
    pub architectural_parameters: u64,
    /// Sum of physical parameters across ranks, including replicated norms.
    pub aggregate_materialized_parameters: u64,
    /// Cross-rank duplicates caused by replicated norms.
    pub duplicated_parameters: u64,
}

impl TensorParallelPartition {
    /// Creates a strict equal F32 partition and optionally compares every rank with one budget.
    pub fn plan(
        model: SupportedModelId,
        tp: usize,
        context_capacity: usize,
        per_rank_budget: Option<u64>,
    ) -> Result<Self> {
        if !(2..=64).contains(&tp) {
            return Err(TensorParallelError::InvalidRequest(format!(
                "TP must be in 2..=64, got {tp}"
            )));
        }
        let spec = model.spec();
        if context_capacity == 0 || context_capacity > spec.config.max_position_embeddings {
            return Err(TensorParallelError::InvalidRequest(format!(
                "context capacity {context_capacity} is outside 1..={}",
                spec.config.max_position_embeddings
            )));
        }
        let head_dim = spec.config.head_dim()? as u64;
        let ranks = (0..tp)
            .map(|rank| {
                let shard =
                    plan_tensor_shard(&spec.config, ParallelContext::tensor_parallel(rank, tp)?)?;
                let local_weight_bytes =
                    shard.local_parameters.checked_mul(4).ok_or_else(|| {
                        TensorParallelError::InvalidRequest("local weight byte overflow".into())
                    })?;
                let kv_cache_capacity_bytes = 2u64
                    .checked_mul(spec.config.num_hidden_layers as u64)
                    .and_then(|v| v.checked_mul(context_capacity as u64))
                    .and_then(|v| v.checked_mul((spec.config.num_key_value_heads / tp) as u64))
                    .and_then(|v| v.checked_mul(head_dim))
                    .and_then(|v| v.checked_mul(4))
                    .ok_or_else(|| {
                        TensorParallelError::InvalidRequest("KV byte overflow".into())
                    })?;
                let persistent_bytes = local_weight_bytes
                    .checked_add(kv_cache_capacity_bytes)
                    .ok_or_else(|| {
                        TensorParallelError::InvalidRequest("persistent byte overflow".into())
                    })?;
                let placement = match per_rank_budget {
                    Some(budget) if persistent_bytes <= budget => {
                        PlacementVerdict::FitsPersistentEstimate
                    }
                    Some(_) => PlacementVerdict::DoesNotFit,
                    None => PlacementVerdict::NotEvaluated,
                };
                Ok(TensorParallelMemoryPlan {
                    rank,
                    shard,
                    dtype: PlanDType::F32,
                    context_capacity,
                    local_weight_bytes,
                    kv_cache_capacity_bytes,
                    persistent_bytes,
                    budget_bytes: per_rank_budget,
                    placement,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let architectural_parameters = ranks[0].shard.architectural_parameters;
        let aggregate_materialized_parameters =
            ranks.iter().map(|rank| rank.shard.local_parameters).sum();
        Ok(Self {
            model,
            tensor_parallel: tp,
            ranks,
            architectural_parameters,
            aggregate_materialized_parameters,
            duplicated_parameters: aggregate_materialized_parameters - architectural_parameters,
        })
    }

    /// Rejects the first rank whose persistent estimate exceeds its budget.
    pub fn require_placement(&self) -> Result<()> {
        if let Some(rank) = self
            .ranks
            .iter()
            .find(|rank| rank.placement == PlacementVerdict::DoesNotFit)
        {
            return Err(TensorParallelError::PlacementFailed {
                rank: rank.rank,
                required_bytes: rank.persistent_bytes,
                budget_bytes: rank.budget_bytes.expect("failed placement has a budget"),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_tp_rules_memory_formula_and_exact_budget_boundary() {
        let smol =
            TensorParallelPartition::plan(SupportedModelId::SmolLm2_135MInstruct, 3, 64, None)
                .unwrap();
        assert_eq!(smol.ranks.len(), 3);
        let exact = smol.ranks[0].persistent_bytes;
        assert_eq!(
            TensorParallelPartition::plan(
                SupportedModelId::SmolLm2_135MInstruct,
                3,
                64,
                Some(exact)
            )
            .unwrap()
            .ranks[0]
                .placement,
            PlacementVerdict::FitsPersistentEstimate
        );
        let failed = TensorParallelPartition::plan(
            SupportedModelId::SmolLm2_135MInstruct,
            3,
            64,
            Some(exact - 1),
        )
        .unwrap();
        assert!(failed.require_placement().is_err());
        assert!(
            TensorParallelPartition::plan(SupportedModelId::SmolLm2_135MInstruct, 2, 64, None)
                .is_err()
        );
    }
}
