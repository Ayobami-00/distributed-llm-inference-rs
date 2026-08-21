use crate::{MemoryBudget, PlanDType, RankMemoryPlan, Result, SupportedModelId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionRequest {
    pub model: SupportedModelId,
    pub dtype: PlanDType,
    pub context_length: usize,
    pub device_memory_budget: Option<MemoryBudget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionReport {
    pub schema_version: u32,
    pub model: SupportedModelId,
    pub repository: String,
    pub revision: String,
    pub architecture: String,
    pub config: crate::ModelConfig,
    pub memory: RankMemoryPlan,
    pub caveat: String,
}

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
