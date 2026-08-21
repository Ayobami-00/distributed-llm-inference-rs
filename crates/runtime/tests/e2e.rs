use dlir_runtime::{
    GenerationRequest, NoopObserver, PlanDType, RunEventKind, StopReason, SupportedModelId,
    generate,
};

fn request(model: SupportedModelId, max_new_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        model,
        dtype: PlanDType::F32,
        prompt: "Explain tensor parallelism.".into(),
        max_new_tokens,
        device_memory_budget: None,
    }
}

#[test]
#[ignore = "downloads the 269 MB pinned SmolLM2 checkpoint"]
fn smollm2_generates_deterministically_and_reports_schema_v1() {
    let request = request(SupportedModelId::SmolLm2_135MInstruct, 4);
    let first = generate(&request, &mut NoopObserver).unwrap();
    let second = generate(&request, &mut NoopObserver).unwrap();

    assert_eq!(first.schema_version, 1);
    assert_eq!(first.generated_tokens, second.generated_tokens);
    assert!(first.generated_tokens.len() >= 2 || first.stop_reason == StopReason::Eos);
    assert!(first.events.iter().all(|event| event.rank == 0));
    assert!(
        first
            .events
            .iter()
            .any(|event| matches!(event.event, RunEventKind::PrefillFinished))
    );
    assert!(serde_json::to_vec(&first).is_ok());
}

#[test]
#[ignore = "downloads the 2.2 GB pinned TinyLlama checkpoint"]
fn tinyllama_loads_and_generates_one_token() {
    let report = generate(
        &request(SupportedModelId::TinyLlama1_1BChat, 1),
        &mut NoopObserver,
    )
    .unwrap();
    assert_eq!(report.schema_version, 1);
    assert!(report.generated_tokens.len() == 1 || report.stop_reason == StopReason::Eos);
}
