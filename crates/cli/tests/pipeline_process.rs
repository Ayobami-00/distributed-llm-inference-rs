use dlir_pipeline::{
    PipelineManifest, PipelinePartition, PipelineRankReport, PipelineStreamRecord, StageMemoryPlan,
};
use dlir_runtime::{
    ArtifactRepository, GenerationRequest, NoopObserver, PlanDType, SupportedModelId, generate,
    render_prompt, validate_checkpoint, validate_metadata,
};
use std::{
    fs,
    net::TcpListener,
    process::{Command, Stdio},
    thread,
};
use tokenizers::Tokenizer;

#[test]
#[ignore = "downloads TinyLlama and materializes two CPU/F32 process stages"]
fn tinyllama_loopback_process_pipeline_matches_single_rank_token() {
    let model = SupportedModelId::TinyLlama1_1BChat;
    let spec = model.spec();
    let artifacts = ArtifactRepository::new(spec).unwrap();
    let metadata = artifacts.download_metadata().unwrap();
    validate_metadata(spec, &metadata).unwrap();
    let checkpoint = artifacts.download_weights(spec).unwrap();
    validate_checkpoint(spec, &checkpoint).unwrap();

    let prompt = "Explain pipeline parallelism.";
    let rendered = render_prompt(spec.prompt_template, prompt).unwrap();
    let tokenizer = Tokenizer::from_file(&metadata.tokenizer).unwrap();
    let prompt_token_ids = tokenizer
        .encode(rendered, false)
        .unwrap()
        .get_ids()
        .to_vec();
    let context_capacity = prompt_token_ids.len() + 1;
    let partition = PipelinePartition::balanced(spec, 2).unwrap();
    let memory_plans = partition
        .stages
        .iter()
        .map(|stage| {
            StageMemoryPlan::for_stage(spec, stage, PlanDType::F32, context_capacity, u64::MAX)
                .unwrap()
        })
        .collect();
    let directory = tempfile::tempdir().unwrap();
    let manifest_path = directory.path().join("pipeline.json");
    let manifest = PipelineManifest {
        schema_version: 1,
        run_id: "tinyllama-loopback".into(),
        request_id: "one-token".into(),
        model,
        dtype: PlanDType::F32,
        prompt_token_ids,
        prompt_characters: prompt.chars().count(),
        requested_max_new_tokens: 1,
        effective_max_new_tokens: 1,
        context_capacity,
        checkpoint_path: checkpoint,
        tokenizer_path: metadata.tokenizer,
        partition,
        memory_plans,
        expected_cpu_millis: 0,
        expected_memory_bytes: 0,
    };
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let rendezvous_port = free_port();
    let peer_ports = [free_port(), free_port()];
    let child = |rank: usize| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_dlir"));
        command.args([
            "rank",
            "--workload",
            "pipeline",
            "--pipeline-manifest",
            manifest_path.to_str().unwrap(),
            "--rank",
            &rank.to_string(),
            "--world-size",
            "2",
            "--run-id",
            "tinyllama-loopback",
            "--rendezvous-addr",
            &format!("127.0.0.1:{rendezvous_port}"),
            "--listen-addr",
            &format!("127.0.0.1:{}", peer_ports[rank]),
            "--advertise-addr",
            &format!("127.0.0.1:{}", peer_ports[rank]),
            "--startup-timeout-seconds",
            "120",
            "--operation-timeout-seconds",
            "120",
        ]);
        if rank == 0 {
            command.args([
                "--rendezvous-bind-addr",
                &format!("127.0.0.1:{rendezvous_port}"),
            ]);
        }
        command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap()
    };
    let rank_zero = child(0);
    let rank_one = child(1);
    let zero = thread::spawn(move || rank_zero.wait_with_output().unwrap());
    let one = thread::spawn(move || rank_one.wait_with_output().unwrap());
    let outputs = [zero.join().unwrap(), one.join().unwrap()];
    let rank_reports = outputs
        .iter()
        .map(|output| {
            assert!(
                output.status.success(),
                "rank failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| serde_json::from_str::<PipelineStreamRecord>(line).ok())
                .find_map(|record| match record {
                    PipelineStreamRecord::Result { result } => Some(*result),
                    PipelineStreamRecord::Event { .. } => None,
                })
                .expect("rank emitted a final pipeline result")
        })
        .collect::<Vec<PipelineRankReport>>();
    assert!(
        rank_reports
            .iter()
            .all(|report| report.success && report.barriers_passed)
    );

    let single = generate(
        &GenerationRequest {
            model,
            dtype: PlanDType::F32,
            prompt: prompt.into(),
            max_new_tokens: 1,
            device_memory_budget: None,
        },
        &mut NoopObserver,
    )
    .unwrap();
    assert_eq!(rank_reports[0].generated_tokens, single.generated_tokens);
    assert_eq!(rank_reports[0].completion, single.completion);
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
