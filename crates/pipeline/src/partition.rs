//! Deterministic contiguous layer partitioning and rank-local memory planning.

use crate::{PipelineError, Result};
use dlir_runtime::{ModelSpec, PlacementVerdict, PlanDType};
use serde::{Deserialize, Serialize};

/// One rank's contiguous pipeline-stage assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageAssignment {
    /// Global rank and pipeline-stage index.
    pub rank: usize,
    /// Total number of pipeline stages.
    pub world_size: usize,
    /// Inclusive first global transformer layer.
    pub layer_start: usize,
    /// Exclusive final global transformer layer.
    pub layer_end: usize,
    /// Whether this rank materializes and applies token embeddings.
    pub owns_embeddings: bool,
    /// Whether this rank materializes final normalization and the LM head.
    pub owns_lm_head: bool,
    /// Whether an originally tied embedding matrix is duplicated on the final rank.
    pub duplicates_tied_embeddings: bool,
}

impl StageAssignment {
    /// Returns the number of transformer layers executed locally.
    pub fn layer_count(&self) -> usize {
        self.layer_end - self.layer_start
    }
}

/// Complete ordered stage assignment for one model and pipeline world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PipelinePartition {
    /// Total transformer layers covered exactly once.
    pub total_layers: usize,
    /// Rank-ordered stage assignments.
    pub stages: Vec<StageAssignment>,
}

impl PipelinePartition {
    /// Balances contiguous layers, assigning remainders to the lowest ranks.
    pub fn balanced(spec: &ModelSpec, world_size: usize) -> Result<Self> {
        if !(2..=64).contains(&world_size) {
            return Err(PipelineError::InvalidTopology(format!(
                "world size must be between 2 and 64, got {world_size}"
            )));
        }
        let layers = spec.config.num_hidden_layers;
        if world_size > layers {
            return Err(PipelineError::InvalidTopology(format!(
                "world size {world_size} exceeds {layers} transformer layers"
            )));
        }
        let base = layers / world_size;
        let remainder = layers % world_size;
        let mut next = 0usize;
        let stages = (0..world_size)
            .map(|rank| {
                let count = base + usize::from(rank < remainder);
                let layer_start = next;
                let layer_end = layer_start + count;
                next = layer_end;
                StageAssignment {
                    rank,
                    world_size,
                    layer_start,
                    layer_end,
                    owns_embeddings: rank == 0,
                    owns_lm_head: rank + 1 == world_size,
                    duplicates_tied_embeddings: rank + 1 == world_size
                        && spec.config.tie_word_embeddings,
                }
            })
            .collect::<Vec<_>>();
        debug_assert_eq!(next, layers);
        Ok(Self {
            total_layers: layers,
            stages,
        })
    }

    /// Returns one validated rank assignment.
    pub fn stage(&self, rank: usize) -> Result<&StageAssignment> {
        self.stages
            .get(rank)
            .ok_or_else(|| PipelineError::InvalidTopology(format!("partition has no rank {rank}")))
    }
}

/// Logical persistent-state plan for one pipeline rank.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageMemoryPlan {
    /// Global rank described by this plan.
    pub rank: usize,
    /// Assigned global layer range.
    pub layer_start: usize,
    /// Exclusive end of the assigned layer range.
    pub layer_end: usize,
    /// Parameters materialized by local transformer blocks.
    pub block_parameters: u64,
    /// Parameters materialized for input embeddings.
    pub embedding_parameters: u64,
    /// Parameters materialized for final normalization.
    pub final_norm_parameters: u64,
    /// Parameters materialized for the output head.
    pub lm_head_parameters: u64,
    /// Total rank-local materialized parameters.
    pub materialized_parameters: u64,
    /// Logical rank-local weight bytes.
    pub weight_bytes: u64,
    /// Preallocated local KV-cache bytes.
    pub kv_cache_capacity_bytes: u64,
    /// Weights plus local KV capacity.
    pub persistent_bytes: u64,
    /// Enforced Docker memory limit used as placement budget.
    pub budget_bytes: u64,
    /// Exact persistent-estimate comparison with the budget.
    pub placement: PlacementVerdict,
    /// Whether tied embedding weights are duplicated on this rank.
    pub duplicates_tied_embeddings: bool,
}

impl StageMemoryPlan {
    /// Builds a rank-local plan from one assignment and enforced memory limit.
    pub fn for_stage(
        spec: &ModelSpec,
        assignment: &StageAssignment,
        dtype: PlanDType,
        context_length: usize,
        budget_bytes: u64,
    ) -> Result<Self> {
        if context_length == 0 || context_length > spec.config.max_position_embeddings {
            return Err(PipelineError::InvalidTopology(format!(
                "context length {context_length} is outside model maximum {}",
                spec.config.max_position_embeddings
            )));
        }
        let cfg = &spec.config;
        let h = cfg.hidden_size as u64;
        let i = cfg.intermediate_size as u64;
        let kv = (cfg.num_key_value_heads * cfg.head_dim()?) as u64;
        let per_block = 2 * h * h + 2 * h * kv + 3 * h * i + 2 * h;
        let block_parameters = per_block
            .checked_mul(assignment.layer_count() as u64)
            .ok_or_else(|| PipelineError::InvalidTopology("stage parameter overflow".into()))?;
        let embedding_parameters = if assignment.owns_embeddings {
            cfg.vocab_size as u64 * h
        } else {
            0
        };
        let final_norm_parameters = u64::from(assignment.owns_lm_head) * h;
        let lm_head_parameters = if assignment.owns_lm_head {
            cfg.vocab_size as u64 * h
        } else {
            0
        };
        let materialized_parameters = block_parameters
            .checked_add(embedding_parameters)
            .and_then(|v| v.checked_add(final_norm_parameters))
            .and_then(|v| v.checked_add(lm_head_parameters))
            .ok_or_else(|| PipelineError::InvalidTopology("stage parameter overflow".into()))?;
        let weight_bytes = materialized_parameters
            .checked_mul(dtype.bytes())
            .ok_or_else(|| PipelineError::InvalidTopology("stage weight overflow".into()))?;
        let kv_cache_capacity_bytes = 2u64
            .checked_mul(assignment.layer_count() as u64)
            .and_then(|v| v.checked_mul(context_length as u64))
            .and_then(|v| v.checked_mul(cfg.num_key_value_heads as u64))
            .and_then(|v| v.checked_mul(cfg.head_dim().ok()? as u64))
            .and_then(|v| v.checked_mul(dtype.bytes()))
            .ok_or_else(|| PipelineError::InvalidTopology("stage KV byte overflow".into()))?;
        let persistent_bytes = weight_bytes
            .checked_add(kv_cache_capacity_bytes)
            .ok_or_else(|| PipelineError::InvalidTopology("stage memory overflow".into()))?;
        let placement = if persistent_bytes <= budget_bytes {
            PlacementVerdict::FitsPersistentEstimate
        } else {
            PlacementVerdict::DoesNotFit
        };
        Ok(Self {
            rank: assignment.rank,
            layer_start: assignment.layer_start,
            layer_end: assignment.layer_end,
            block_parameters,
            embedding_parameters,
            final_norm_parameters,
            lm_head_parameters,
            materialized_parameters,
            weight_bytes,
            kv_cache_capacity_bytes,
            persistent_bytes,
            budget_bytes,
            placement,
            duplicates_tied_embeddings: assignment.duplicates_tied_embeddings,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlir_runtime::SupportedModelId;

    #[test]
    fn balances_both_registry_layer_counts() {
        let smol =
            PipelinePartition::balanced(SupportedModelId::SmolLm2_135MInstruct.spec(), 4).unwrap();
        assert_eq!(
            smol.stages
                .iter()
                .map(StageAssignment::layer_count)
                .collect::<Vec<_>>(),
            vec![8, 8, 7, 7]
        );
        let tiny =
            PipelinePartition::balanced(SupportedModelId::TinyLlama1_1BChat.spec(), 4).unwrap();
        assert_eq!(
            tiny.stages
                .iter()
                .map(StageAssignment::layer_count)
                .collect::<Vec<_>>(),
            vec![6, 6, 5, 5]
        );
    }

    #[test]
    fn tied_head_duplication_is_explicit() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let partition = PipelinePartition::balanced(spec, 2).unwrap();
        let first =
            StageMemoryPlan::for_stage(spec, &partition.stages[0], PlanDType::F32, 8, u64::MAX)
                .unwrap();
        let last =
            StageMemoryPlan::for_stage(spec, &partition.stages[1], PlanDType::F32, 8, u64::MAX)
                .unwrap();
        assert!(last.duplicates_tied_embeddings);
        assert_eq!(first.embedding_parameters, last.lm_head_parameters);
        let total = [first, last]
            .iter()
            .map(|plan| plan.materialized_parameters)
            .sum::<u64>();
        assert_eq!(
            total,
            spec.expected_parameters
                + spec.config.vocab_size as u64 * spec.config.hidden_size as u64
        );
    }

    #[test]
    fn partitions_cover_every_layer_once_and_reject_empty_stages() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        for world_size in [2, 3, 4, spec.config.num_hidden_layers] {
            let partition = PipelinePartition::balanced(spec, world_size).unwrap();
            assert_eq!(partition.stages.first().unwrap().layer_start, 0);
            assert_eq!(
                partition.stages.last().unwrap().layer_end,
                spec.config.num_hidden_layers
            );
            for pair in partition.stages.windows(2) {
                assert_eq!(pair[0].layer_end, pair[1].layer_start);
                assert!(pair[0].layer_count() > 0);
            }
            assert!(partition.stages.last().unwrap().layer_count() > 0);
        }
        assert!(PipelinePartition::balanced(spec, 1).is_err());
        assert!(PipelinePartition::balanced(spec, spec.config.num_hidden_layers + 1).is_err());
        assert!(PipelinePartition::balanced(spec, 65).is_err());
    }

    #[test]
    fn stage_placement_uses_an_inclusive_exact_boundary() {
        let spec = SupportedModelId::TinyLlama1_1BChat.spec();
        let partition = PipelinePartition::balanced(spec, 2).unwrap();
        let unconstrained =
            StageMemoryPlan::for_stage(spec, &partition.stages[0], PlanDType::F32, 32, u64::MAX)
                .unwrap();
        let exact = StageMemoryPlan::for_stage(
            spec,
            &partition.stages[0],
            PlanDType::F32,
            32,
            unconstrained.persistent_bytes,
        )
        .unwrap();
        assert_eq!(exact.placement, PlacementVerdict::FitsPersistentEstimate);
        let short = StageMemoryPlan::for_stage(
            spec,
            &partition.stages[0],
            PlanDType::F32,
            32,
            unconstrained.persistent_bytes - 1,
        )
        .unwrap();
        assert_eq!(short.placement, PlacementVerdict::DoesNotFit);
        assert!(!partition.stages.last().unwrap().duplicates_tied_embeddings);
    }
}
