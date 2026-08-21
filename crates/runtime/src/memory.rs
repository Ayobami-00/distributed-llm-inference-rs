use crate::{DlirError, ModelSpec, PlanDType, Result};
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryDomain {
    Host,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetSource {
    UserDeclared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub bytes: u64,
    pub domain: MemoryDomain,
    pub source: BudgetSource,
    pub os_enforced: bool,
}

impl MemoryBudget {
    pub const fn user_declared(bytes: u64) -> Self {
        Self {
            bytes,
            domain: MemoryDomain::Host,
            source: BudgetSource::UserDeclared,
            os_enforced: false,
        }
    }
}

impl FromStr for MemoryBudget {
    type Err = DlirError;

    fn from_str(value: &str) -> Result<Self> {
        parse_byte_size(value).map(Self::user_declared)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementVerdict {
    FitsPersistentEstimate,
    DoesNotFit,
    NotEvaluated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryComponentBreakdown {
    pub token_embedding_parameters: u64,
    pub attention_parameters: u64,
    pub mlp_parameters: u64,
    pub layer_norm_parameters: u64,
    pub final_norm_parameters: u64,
    pub lm_head_parameters: u64,
}

impl MemoryComponentBreakdown {
    pub fn total(&self) -> u64 {
        self.token_embedding_parameters
            + self.attention_parameters
            + self.mlp_parameters
            + self.layer_norm_parameters
            + self.final_norm_parameters
            + self.lm_head_parameters
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankMemoryPlan {
    pub rank: usize,
    pub parameter_count: u64,
    pub dtype: PlanDType,
    pub context_length: usize,
    pub weight_bytes: u64,
    pub kv_cache_capacity_bytes: u64,
    pub persistent_bytes: u64,
    pub breakdown: MemoryComponentBreakdown,
    pub budget: Option<MemoryBudget>,
    pub placement: PlacementVerdict,
}

impl RankMemoryPlan {
    pub fn for_model(
        spec: &ModelSpec,
        dtype: PlanDType,
        context_length: usize,
        budget: Option<MemoryBudget>,
    ) -> Result<Self> {
        if context_length == 0 || context_length > spec.config.max_position_embeddings {
            return Err(DlirError::InvalidConfig(format!(
                "context length must be between 1 and {} for {}",
                spec.config.max_position_embeddings, spec.id
            )));
        }

        let breakdown = parameter_breakdown(spec)?;
        if breakdown.total() != spec.expected_parameters {
            return Err(DlirError::InvalidConfig(format!(
                "registry parameter formula produced {} for {} but expected {}",
                breakdown.total(),
                spec.id,
                spec.expected_parameters
            )));
        }

        let weight_bytes = spec
            .expected_parameters
            .checked_mul(dtype.bytes())
            .ok_or_else(|| DlirError::InvalidConfig("weight byte count overflow".into()))?;
        let head_dim = spec.config.head_dim()? as u64;
        let kv_cache_capacity_bytes = 2u64
            .checked_mul(spec.config.num_hidden_layers as u64)
            .and_then(|v| v.checked_mul(context_length as u64))
            .and_then(|v| v.checked_mul(spec.config.num_key_value_heads as u64))
            .and_then(|v| v.checked_mul(head_dim))
            .and_then(|v| v.checked_mul(dtype.bytes()))
            .ok_or_else(|| DlirError::InvalidConfig("KV-cache byte count overflow".into()))?;
        let persistent_bytes = weight_bytes
            .checked_add(kv_cache_capacity_bytes)
            .ok_or_else(|| DlirError::InvalidConfig("persistent byte count overflow".into()))?;
        let placement = match budget {
            None => PlacementVerdict::NotEvaluated,
            Some(budget) if persistent_bytes <= budget.bytes => {
                PlacementVerdict::FitsPersistentEstimate
            }
            Some(_) => PlacementVerdict::DoesNotFit,
        };

        Ok(Self {
            rank: 0,
            parameter_count: spec.expected_parameters,
            dtype,
            context_length,
            weight_bytes,
            kv_cache_capacity_bytes,
            persistent_bytes,
            breakdown,
            budget,
            placement,
        })
    }
}

fn parameter_breakdown(spec: &ModelSpec) -> Result<MemoryComponentBreakdown> {
    let cfg = &spec.config;
    let h = cfg.hidden_size as u64;
    let i = cfg.intermediate_size as u64;
    let l = cfg.num_hidden_layers as u64;
    let v = cfg.vocab_size as u64;
    let kv = (cfg.num_key_value_heads * cfg.head_dim()?) as u64;

    Ok(MemoryComponentBreakdown {
        token_embedding_parameters: v * h,
        attention_parameters: l * (2 * h * h + 2 * h * kv),
        mlp_parameters: l * 3 * h * i,
        layer_norm_parameters: l * 2 * h,
        final_norm_parameters: h,
        lm_head_parameters: if cfg.tie_word_embeddings { 0 } else { v * h },
    })
}

pub fn parse_byte_size(value: &str) -> Result<u64> {
    let value = value.trim();
    if value.is_empty() {
        return Err(invalid_size(value, "value is empty"));
    }

    let split = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, suffix) = value.split_at(split);
    if digits.is_empty() {
        return Err(invalid_size(value, "expected an integer byte count"));
    }
    let amount = digits
        .parse::<u64>()
        .map_err(|_| invalid_size(value, "integer is out of range"))?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "kib" => 1u64 << 10,
        "mib" => 1u64 << 20,
        "gib" => 1u64 << 30,
        _ => {
            return Err(invalid_size(
                value,
                "expected B, KiB, MiB, or GiB; decimal and ambiguous suffixes are rejected",
            ));
        }
    };
    amount
        .checked_mul(multiplier)
        .ok_or_else(|| invalid_size(value, "byte count overflow"))
}

fn invalid_size(value: &str, reason: &str) -> DlirError {
    DlirError::InvalidMemorySize {
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

pub fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    const KIB: f64 = (1u64 << 10) as f64;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.1} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.1} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

impl fmt::Display for PlacementVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FitsPersistentEstimate => "PASSES PERSISTENT-STATE ESTIMATE",
            Self::DoesNotFit => "FAILED",
            Self::NotEvaluated => "NOT EVALUATED",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SupportedModelId;

    #[test]
    fn parses_only_unambiguous_iec_sizes() {
        assert_eq!(parse_byte_size("500MiB").unwrap(), 500 << 20);
        assert_eq!(parse_byte_size("2gib").unwrap(), 2 << 30);
        assert_eq!(parse_byte_size("1024").unwrap(), 1024);
        assert!(parse_byte_size("500M").is_err());
        assert!(parse_byte_size("1.5GiB").is_err());
    }

    #[test]
    fn smol_parameter_formula_and_budget_match_the_checkpoint() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let plan = RankMemoryPlan::for_model(
            spec,
            PlanDType::F32,
            512,
            Some(MemoryBudget::user_declared(500 << 20)),
        )
        .unwrap();
        assert_eq!(plan.breakdown.total(), 134_515_008);
        assert_eq!(plan.breakdown.token_embedding_parameters, 28_311_552);
        assert_eq!(plan.breakdown.attention_parameters, 26_542_080);
        assert_eq!(plan.breakdown.mlp_parameters, 79_626_240);
        assert_eq!(plan.breakdown.layer_norm_parameters, 34_560);
        assert_eq!(plan.breakdown.final_norm_parameters, 576);
        assert_eq!(plan.breakdown.lm_head_parameters, 0);
        assert_eq!(plan.weight_bytes, 538_060_032);
        assert_eq!(plan.kv_cache_capacity_bytes, 23_592_960);
        assert_eq!(plan.persistent_bytes, 561_652_992);
        assert_eq!(plan.placement, PlacementVerdict::DoesNotFit);
    }

    #[test]
    fn budget_boundary_is_inclusive() {
        let spec = SupportedModelId::TinyLlama1_1BChat.spec();
        let first = RankMemoryPlan::for_model(spec, PlanDType::Bf16, 1, None).unwrap();
        assert_eq!(first.breakdown.lm_head_parameters, 65_536_000);
        let plan = RankMemoryPlan::for_model(
            spec,
            PlanDType::Bf16,
            1,
            Some(MemoryBudget::user_declared(first.persistent_bytes)),
        )
        .unwrap();
        assert_eq!(plan.placement, PlacementVerdict::FitsPersistentEstimate);
    }
}
