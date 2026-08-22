use dlir_runtime::{
    ArtifactRepository, GenerationRequest, NoopObserver, PlanDType, SupportedModelId, generate,
    render_prompt, validate_checkpoint, validate_metadata,
};
use dlir_tensor::{
    TensorParallelManifest, TensorParallelPartition, TensorParallelRankReport,
    TensorParallelStreamRecord,
};
use std::{
    fs,
    net::TcpListener,
    process::{Command, Stdio},
    thread,
};
use tokenizers::Tokenizer;

#[test]
#[ignore = "downloads TinyLlama and materializes four CPU/F32 tensor shards"]
fn tinyllama_tp4_loopback_matches_single_rank_token() {
    run_tinyllama_loopback(4, 1, "tinyllama-tp4-loopback");
}

#[test]
#[ignore = "downloads TinyLlama and materializes two CPU/F32 tensor shards"]
fn tinyllama_tp2_loopback_matches_single_rank_tokens() {
    run_tinyllama_loopback(2, 2, "tinyllama-tp2-loopback");
}

fn run_tinyllama_loopback(tp: usize, max_new_tokens: usize, run_id: &str) {
    let model = SupportedModelId::TinyLlama1_1BChat;
    let spec = model.spec();
    let artifacts = ArtifactRepository::new(spec).unwrap();
    let metadata = artifacts.download_metadata().unwrap();
    validate_metadata(spec, &metadata).unwrap();
    let checkpoint = artifacts.download_weights(spec).unwrap();
    validate_checkpoint(spec, &checkpoint).unwrap();
    let prompt = "Explain tensor parallelism.";
    let tokenizer = Tokenizer::from_file(&metadata.tokenizer).unwrap();
    let prompt_token_ids = tokenizer
        .encode(render_prompt(spec.prompt_template, prompt).unwrap(), false)
        .unwrap()
        .get_ids()
        .to_vec();
    let context_capacity = prompt_token_ids.len() + max_new_tokens;
    let partition = TensorParallelPartition::plan(model, tp, context_capacity, None).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let manifest_path = directory.path().join("tensor.json");
    let manifest = TensorParallelManifest {
        schema_version: 1,
        run_id: run_id.into(),
        request_id: "acceptance".into(),
        model,
        dtype: PlanDType::F32,
        tensor_parallel: tp,
        all_reduce: dlir_collectives::AllReduceAlgorithm::Ring,
        prompt_token_ids,
        requested_max_new_tokens: max_new_tokens,
        effective_max_new_tokens: max_new_tokens,
        context_capacity,
        checkpoint_path: checkpoint,
        tokenizer_path: metadata.tokenizer,
        partition,
        expected_cpu_millis: 0,
        expected_memory_bytes: 0,
    };
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let rendezvous_port = free_port();
    let peer_ports = (0..tp).map(|_| free_port()).collect::<Vec<_>>();
    let mut children = Vec::new();
    for (rank, peer_port) in peer_ports.iter().enumerate() {
        let mut command = Command::new(env!("CARGO_BIN_EXE_dlir"));
        command.args([
            "rank",
            "--workload",
            "tensor",
            "--tensor-manifest",
            manifest_path.to_str().unwrap(),
            "--rank",
            &rank.to_string(),
            "--world-size",
            &tp.to_string(),
            "--run-id",
            run_id,
            "--rendezvous-addr",
            &format!("127.0.0.1:{rendezvous_port}"),
            "--listen-addr",
            &format!("127.0.0.1:{peer_port}"),
            "--advertise-addr",
            &format!("127.0.0.1:{peer_port}"),
            "--startup-timeout-seconds",
            "180",
            "--operation-timeout-seconds",
            "180",
        ]);
        if rank == 0 {
            command.args([
                "--rendezvous-bind-addr",
                &format!("127.0.0.1:{rendezvous_port}"),
            ]);
        }
        children.push(
            command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let outputs = children
        .into_iter()
        .map(|child| thread::spawn(move || child.wait_with_output().unwrap()))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    let ranks = outputs
        .iter()
        .map(|output| {
            assert!(
                output.status.success(),
                "rank failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| serde_json::from_str::<TensorParallelStreamRecord>(line).ok())
                .find_map(|record| match record {
                    TensorParallelStreamRecord::Result { result } => Some(*result),
                    TensorParallelStreamRecord::Event { .. } => None,
                })
                .unwrap()
        })
        .collect::<Vec<TensorParallelRankReport>>();
    let single = generate(
        &GenerationRequest {
            model,
            dtype: PlanDType::F32,
            prompt: prompt.into(),
            max_new_tokens,
            device_memory_budget: None,
        },
        &mut NoopObserver,
    )
    .unwrap();
    assert!(
        ranks
            .iter()
            .all(|rank| rank.success && rank.barriers_passed)
    );
    assert_eq!(ranks[0].generated_tokens, single.generated_tokens);
    assert_eq!(ranks[0].completion, single.completion);
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
