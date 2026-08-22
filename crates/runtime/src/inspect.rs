//! Network-free model inspection backed entirely by the compiled registry.
//!
//! Inspection and generation share [`RankMemoryPlan`], ensuring that a reported placement
//! boundary is the same boundary enforced before generation downloads checkpoint weights.

use crate::{MemoryBudget, PlanDType, RankMemoryPlan, Result, SupportedModelId};
use serde::{Deserialize, Serialize};

/// Inputs for a registry-only model inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionRequest {
    /// Closed-registry model to inspect.
    pub model: SupportedModelId,
    /// Logical dtype used for weight and KV-cache byte calculations.
    pub dtype: PlanDType,
    /// KV-cache capacity to model, in token positions.
    pub context_length: usize,
    /// Optional user-declared per-rank host memory budget.
    pub device_memory_budget: Option<MemoryBudget>,
}

/// Schema-versioned architecture and memory result returned by [`inspect`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionReport {
    /// Serialization contract version; currently `1`.
    pub schema_version: u32,
    /// Closed-registry model ID.
    pub model: SupportedModelId,
    /// Pinned Hugging Face repository.
    pub repository: String,
    /// Pinned repository revision.
    pub revision: String,
    /// Hugging Face architecture identifier validated by the runtime.
    pub architecture: String,
    /// Embedded architecture values used for all calculations.
    pub config: crate::ModelConfig,
    /// Rank-0 logical memory plan and placement verdict.
    pub memory: RankMemoryPlan,
    /// Explanation of memory categories excluded from the estimate.
    pub caveat: String,
}

/// Inspects one registered model without resolving or downloading artifacts.
///
/// A [`crate::PlacementVerdict::DoesNotFit`] verdict is successful report data. Errors are
/// reserved for invalid configuration such as a zero or oversized context.
pub fn inspect(request: &InspectionRequest) -> Result<InspectionReport> {
    let spec = request.model.spec();
    let memory = RankMemoryPlan::for_model(
        spec,
        request.dtype,
        request.context_length,
        request.device_memory_budget,
    )?;
    Ok(InspectionReport {
        schema_version: 1,
        model: request.model,
        repository: spec.repository.to_owned(),
        revision: spec.revision.to_owned(),
        architecture: "LlamaForCausalLM".to_owned(),
        config: spec.config,
        memory,
        caveat:
            "Activations, workspaces, allocator fragmentation, and runtime memory are excluded."
                .to_owned(),
    })
}
