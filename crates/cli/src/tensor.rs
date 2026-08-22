//! Docker tensor-parallel generation and internal rank-process entry point.

use crate::launch::{
    CpuAmount, DockerResources, PEER_PORT, RENDEZVOUS_PORT, RUN_LABEL, ResourcePlan,
    default_run_id, docker_checked, docker_engine_info, docker_output, ensure_image, format_cpu,
    install_interrupt_handler, observe_cgroup_resources, plan_resources, request_interrupt,
    reset_interrupted, stop_containers, validate_run_id, verify_resources,
};
use anyhow::{Result, bail};
use dlir_collectives::{
    AllReduceAlgorithm, DEFAULT_MAX_TENSOR_BYTES, Rank, TcpTransport, TcpTransportConfig,
};
use dlir_pipeline::{PipelineEvent, ReceivedPipelineEvent, ResourceSnapshot};
use dlir_runtime::{
    ArtifactRepository, PlanDType, SupportedModelId, format_bytes, render_prompt,
    validate_checkpoint, validate_metadata,
};
use dlir_tensor::{
    TensorEventSink, TensorParallelManifest, TensorParallelPartition, TensorParallelReport,
    TensorParallelResourcePlan, TensorParallelStreamRecord, run_tensor_parallel_rank_observed,
};
use dlir_tui::{DashboardExit, DashboardMessage, DashboardState, run_dashboard};
use std::{
    fs,
    io::{self, BufRead, BufReader, IsTerminal, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::Tokenizer;

const CONTAINER_MANIFEST: &str = "/opt/dlir/request/tensor.json";
const CONTAINER_CHECKPOINT: &str = "/opt/dlir/artifacts/model.safetensors";
const CONTAINER_TOKENIZER: &str = "/opt/dlir/artifacts/tokenizer.json";

pub(crate) struct TensorLaunchRequest {
    pub(crate) model: SupportedModelId,
    pub(crate) tp: usize,
    pub(crate) dtype: PlanDType,
    pub(crate) prompt: String,
    pub(crate) max_new_tokens: usize,
    pub(crate) total_cpus: CpuAmount,
    pub(crate) total_memory: String,
    pub(crate) all_reduce: AllReduceAlgorithm,
    pub(crate) tui: bool,
    pub(crate) report: Option<PathBuf>,
    pub(crate) image: String,
    pub(crate) build_context: PathBuf,
    pub(crate) rebuild: bool,
    pub(crate) run_id: Option<String>,
    pub(crate) startup_timeout: Duration,
    pub(crate) operation_timeout: Duration,
    pub(crate) keep_containers: bool,
}

pub(crate) fn run_tensor(request: TensorLaunchRequest) -> Result<()> {
    let cold_started = Instant::now();
    if request.tui && !io::stderr().is_terminal() {
        bail!("--tui requires stderr to be connected to an interactive terminal");
    }
    if request.prompt.trim().is_empty() || request.max_new_tokens == 0 {
        bail!("tensor prompt must be non-empty and max-new-tokens must be at least one");
    }
    if request.dtype != PlanDType::F32 {
        bail!("v0.5 tensor execution supports only CPU/F32");
    }
    let spec = request.model.spec();
    spec.validate_cpu_dtype(request.dtype)?;
    let total_memory = dlir_runtime::parse_byte_size(&request.total_memory)?;
    let run_id = request.run_id.unwrap_or_else(default_run_id);
    validate_run_id(&run_id)?;
    let request_id = request_id();

    eprintln!(
        "model: {} ({})\nrevision: {}\nTP={} PP=1 EP=1 all-reduce={:?}\nresolving configuration and tokenizer...",
        spec.id, spec.repository, spec.revision, request.tp, request.all_reduce
    );
    let repository = ArtifactRepository::new(spec)?;
    let metadata = repository.download_metadata()?;
    validate_metadata(spec, &metadata)?;
    let tokenizer = Tokenizer::from_file(&metadata.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer error: {error}"))?;
    let rendered = render_prompt(spec.prompt_template, &request.prompt)?;
    let encoded = tokenizer
        .encode(rendered, false)
        .map_err(|error| anyhow::anyhow!("tokenizer error: {error}"))?;
    let prompt_token_ids = encoded.get_ids().to_vec();
    if prompt_token_ids.is_empty() || prompt_token_ids.len() >= spec.config.max_position_embeddings
    {
        bail!(
            "prompt has {} tokens but model context is {}",
            prompt_token_ids.len(),
            spec.config.max_position_embeddings
        );
    }
    let effective_max_new_tokens = request
        .max_new_tokens
        .min(spec.config.max_position_embeddings - prompt_token_ids.len());
    let context_capacity = prompt_token_ids.len() + effective_max_new_tokens;

    let engine = docker_engine_info()?;
    let resources = plan_resources(request.tp, request.total_cpus, total_memory, &engine)?;
    let partition = TensorParallelPartition::plan(
        request.model,
        request.tp,
        context_capacity,
        Some(resources.per_rank_memory_bytes),
    )?;
    partition.require_placement()?;
    for plan in &partition.ranks {
        eprintln!(
            "rank {}: vocab {}..{}, Q heads {}..{}, KV heads {}..{}, intermediate {}..{}, persistent {}",
            plan.rank,
            plan.shard.vocabulary.start,
            plan.shard.vocabulary.end,
            plan.shard.query_heads.start,
            plan.shard.query_heads.end,
            plan.shard.kv_heads.start,
            plan.shard.kv_heads.end,
            plan.shard.intermediate.start,
            plan.shard.intermediate.end,
            format_bytes(plan.persistent_bytes),
        );
    }
    eprintln!("placement passed; resolving checkpoint weights once on the host...");
    let checkpoint = repository.download_weights(spec)?;
    validate_checkpoint(spec, &checkpoint)?;

    let run_directory = tempfile::Builder::new()
        .prefix(&format!("dlir-{run_id}-"))
        .tempdir()?;
    let manifest_path = run_directory.path().join("tensor.json");
    let manifest = TensorParallelManifest {
        schema_version: 1,
        run_id: run_id.clone(),
        request_id: request_id.clone(),
        model: request.model,
        dtype: request.dtype,
        tensor_parallel: request.tp,
        all_reduce: request.all_reduce,
        prompt_token_ids,
        requested_max_new_tokens: request.max_new_tokens,
        effective_max_new_tokens,
        context_capacity,
        checkpoint_path: PathBuf::from(CONTAINER_CHECKPOINT),
        tokenizer_path: PathBuf::from(CONTAINER_TOKENIZER),
        partition: partition.clone(),
        expected_cpu_millis: resources.per_rank_cpu_millis,
        expected_memory_bytes: resources.per_rank_memory_bytes,
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    install_interrupt_handler()?;
    reset_interrupted();
    ensure_image(&request.image, &request.build_context, request.rebuild)?;
    let network = format!("dlir-{run_id}");
    docker_checked(&[
        "network".into(),
        "create".into(),
        "--label".into(),
        format!("{RUN_LABEL}={run_id}"),
        network.clone(),
    ])?;
    let mut docker_resources = DockerResources {
        network,
        containers: Vec::new(),
        keep: request.keep_containers,
    };
    let mounts = ArtifactMounts {
        manifest: fs::canonicalize(&manifest_path)?,
        checkpoint: fs::canonicalize(checkpoint)?,
        tokenizer: fs::canonicalize(&metadata.tokenizer)?,
    };
    for rank in 0..request.tp {
        let name = format!("dlir-{run_id}-rank-{rank}");
        eprintln!(
            "starting tensor rank {rank}: {} CPU, {}",
            format_cpu(resources.per_rank_cpu_millis),
            format_bytes(resources.per_rank_memory_bytes)
        );
        let output = docker_checked(&tensor_container_arguments(
            &request.image,
            &docker_resources.network,
            &name,
            &run_id,
            rank,
            request.tp,
            &resources,
            request.startup_timeout,
            request.operation_timeout,
            &mounts,
        ))?;
        let id = String::from_utf8(output.stdout)?.trim().to_owned();
        if id.is_empty() {
            bail!("Docker returned an empty container ID for rank {rank}");
        }
        docker_resources.containers.push(name);
    }

    let (sender, receiver) = mpsc::channel();
    for (rank, name) in docker_resources.containers.iter().enumerate() {
        follow_container(rank, name.clone(), sender.clone());
        wait_container(rank, name.clone(), sender.clone());
    }
    drop(sender);
    let (dashboard_sender, dashboard_receiver) = mpsc::channel();
    let tui_active = Arc::new(AtomicBool::new(request.tui));
    let collector_active = Arc::clone(&tui_active);
    let container_names = docker_resources.containers.clone();
    let world_size = request.tp;
    let collector = thread::spawn(move || {
        collect_rank_streams(
            receiver,
            dashboard_sender,
            collector_active,
            world_size,
            &container_names,
        )
    });
    if request.tui {
        let mut dashboard = DashboardState::new_tensor(
            spec.id.as_str(),
            &partition.ranks,
            format!("{:?}", request.all_reduce).to_ascii_lowercase(),
        );
        match run_dashboard(&mut dashboard, &dashboard_receiver)? {
            DashboardExit::Disabled => {
                tui_active.store(false, Ordering::SeqCst);
                eprintln!("TUI disabled; continuing with text progress...");
            }
            DashboardExit::Interrupted => request_interrupt(),
            DashboardExit::Finished => {}
        }
    }
    let collected = collector
        .join()
        .map_err(|_| anyhow::anyhow!("tensor stream collector panicked"))?;
    let mut ranks = collected.ranks.into_iter().flatten().collect::<Vec<_>>();
    ranks.sort_by_key(|rank| rank.rank);
    let rank_zero = ranks.iter().find(|rank| rank.rank == 0);
    let generated_tokens = rank_zero
        .map(|rank| rank.generated_tokens.clone())
        .unwrap_or_default();
    let completion = rank_zero
        .map(|rank| rank.completion.clone())
        .unwrap_or_default();
    let stop_reason = rank_zero
        .map(|rank| rank.stop_reason)
        .unwrap_or(dlir_runtime::StopReason::MaxNewTokens);
    let communication_bytes = ranks.iter().map(|rank| rank.sent_bytes).sum();
    let success = collected.failures.is_empty()
        && ranks.len() == request.tp
        && ranks.iter().all(|rank| rank.success)
        && collected.exit_codes.iter().all(|code| *code == Some(0));
    let report = TensorParallelReport {
        schema_version: 1,
        run_id: run_id.clone(),
        request_id,
        model: request.model,
        repository: spec.repository.into(),
        revision: spec.revision.into(),
        dtype: request.dtype,
        transport: "tcp".into(),
        collective_backend: "native".into(),
        all_reduce: request.all_reduce,
        tensor_parallel: request.tp,
        pipeline_parallel: 1,
        expert_parallel: 1,
        partition,
        resources: resource_report(&engine, &resources),
        prompt_tokens: manifest.prompt_token_ids.len(),
        generated_tokens,
        completion: completion.clone(),
        stop_reason,
        ranks,
        events: collected.events,
        communication_bytes,
        cold_start_total_ns: ns(cold_started.elapsed()),
        failures: collected.failures,
        success,
    };
    println!("{}", completion);
    io::stdout().flush()?;
    eprintln!(
        "\nTENSOR PARALLEL SUMMARY\nTP={} PP=1 EP=1\nall-reduce: {:?}\ngenerated tokens: {}\ncommunication: {}\nresult: {}",
        report.tensor_parallel,
        report.all_reduce,
        report.generated_tokens.len(),
        format_bytes(report.communication_bytes),
        if report.success { "PASS" } else { "FAIL" },
    );
    if let Some(path) = &request.report {
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("report: {}", path.display());
    }
    if request.keep_containers {
        let kept = run_directory.keep();
        eprintln!("retained tensor request manifest at {}", kept.display());
    } else {
        docker_resources.cleanup();
    }
    if !report.success {
        bail!("Docker tensor-parallel generation failed");
    }
    Ok(())
}

pub(crate) fn run_tensor_rank_process(
    rank_request: crate::launch::RankRequest,
    manifest_path: &Path,
) -> Result<()> {
    let manifest: TensorParallelManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    validate_checkpoint(manifest.model.spec(), &manifest.checkpoint_path)?;
    let observed = observe_cgroup_resources();
    verify_resources(
        &observed,
        rank_request.expected_cpu_millis,
        rank_request.expected_memory_bytes,
    )?;
    let rank = Rank::new(rank_request.rank, rank_request.world_size)?;
    let transport = TcpTransport::connect(TcpTransportConfig {
        rank,
        run_id: rank_request.run_id,
        rendezvous_addr: rank_request.rendezvous_addr,
        rendezvous_bind_addr: rank_request.rendezvous_bind_addr,
        listen_addr: rank_request.listen_addr,
        advertise_addr: rank_request.advertise_addr,
        startup_timeout: rank_request.startup_timeout,
        operation_timeout: rank_request.operation_timeout,
        max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
    })?;
    let peers = transport.peers().to_vec();
    let sink: Arc<dyn TensorEventSink> = Arc::new(JsonLineEventSink);
    let mut report = run_tensor_parallel_rank_observed(
        transport,
        &manifest,
        peers,
        resource_snapshot(observed),
        Arc::clone(&sink),
    )?;
    let final_observed = observe_cgroup_resources();
    verify_resources(
        &final_observed,
        rank_request.expected_cpu_millis,
        rank_request.expected_memory_bytes,
    )?;
    report.resources = resource_snapshot(final_observed);
    append_final_memory_sample(&mut report, sink.as_ref());
    println!(
        "{}",
        serde_json::to_string(&TensorParallelStreamRecord::Result {
            result: Box::new(report)
        })?
    );
    io::stdout().flush()?;
    Ok(())
}

struct JsonLineEventSink;

impl TensorEventSink for JsonLineEventSink {
    fn publish(&self, event: &dlir_pipeline::PipelineEvent) {
        let record = TensorParallelStreamRecord::Event {
            event: event.clone(),
        };
        let mut stdout = io::stdout().lock();
        if serde_json::to_writer(&mut stdout, &record).is_ok() {
            let _ = stdout.write_all(b"\n");
            let _ = stdout.flush();
        }
    }
}

enum HostMessage {
    Record {
        rank: usize,
        record: TensorParallelStreamRecord,
    },
    Log {
        rank: usize,
        line: String,
    },
    StreamEnded {
        rank: usize,
    },
    Exit {
        rank: usize,
        code: i32,
    },
}

fn follow_container(rank: usize, name: String, sender: mpsc::Sender<HostMessage>) {
    thread::spawn(move || {
        let child = Command::new("docker")
            .args(["logs", "--follow", &name])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let Ok(mut child) = child else {
            let _ = sender.send(HostMessage::Log {
                rank,
                line: format!("could not follow Docker logs for {name}"),
            });
            let _ = sender.send(HostMessage::StreamEnded { rank });
            return;
        };
        if let Some(stderr) = child.stderr.take() {
            let stderr_sender = sender.clone();
            thread::spawn(move || {
                for line in BufReader::new(stderr)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    let _ = stderr_sender.send(HostMessage::Log { rank, line });
                }
            });
        }
        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout)
                .lines()
                .map_while(std::result::Result::ok)
            {
                match serde_json::from_str::<TensorParallelStreamRecord>(&line) {
                    Ok(record) => {
                        let _ = sender.send(HostMessage::Record { rank, record });
                    }
                    Err(error) => {
                        let _ = sender.send(HostMessage::Log {
                            rank,
                            line: format!("invalid rank JSON: {error}: {line}"),
                        });
                    }
                }
            }
        }
        let _ = child.wait();
        let _ = sender.send(HostMessage::StreamEnded { rank });
    });
}

fn wait_container(rank: usize, name: String, sender: mpsc::Sender<HostMessage>) {
    thread::spawn(move || {
        let code = docker_output(&["wait".to_owned(), name])
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(-1);
        let _ = sender.send(HostMessage::Exit { rank, code });
    });
}

struct Collected {
    ranks: Vec<Option<dlir_tensor::TensorParallelRankReport>>,
    events: Vec<ReceivedPipelineEvent>,
    exit_codes: Vec<Option<i32>>,
    failures: Vec<String>,
}

fn collect_rank_streams(
    receiver: mpsc::Receiver<HostMessage>,
    dashboard: mpsc::Sender<DashboardMessage>,
    tui_active: Arc<AtomicBool>,
    world_size: usize,
    container_names: &[String],
) -> Collected {
    let mut ranks = (0..world_size).map(|_| None).collect::<Vec<_>>();
    let mut events = Vec::new();
    let mut exit_codes = vec![None; world_size];
    let mut expected_sequence = vec![0u64; world_size];
    let mut failures = Vec::new();
    let mut ended = 0;
    let mut exited = 0;
    for message in receiver {
        match message {
            HostMessage::Record { rank, record } => match record {
                TensorParallelStreamRecord::Event { event } => {
                    if event.event.rank != rank || event.event.sequence != expected_sequence[rank] {
                        failures.push(format!(
                            "rank {rank} event sequence mismatch: expected {}, got {}",
                            expected_sequence[rank], event.event.sequence
                        ));
                    }
                    expected_sequence[rank] = event.event.sequence.saturating_add(1);
                    events.push(ReceivedPipelineEvent {
                        receive_sequence: events.len() as u64,
                        published: event.clone(),
                    });
                    if tui_active.load(Ordering::SeqCst) {
                        let _ = dashboard.send(DashboardMessage::Event(event));
                    } else {
                        print_text_event(&event);
                    }
                }
                TensorParallelStreamRecord::Result { result } => ranks[rank] = Some(*result),
            },
            HostMessage::Log { rank, line } => eprintln!("[rank {rank}] {line}"),
            HostMessage::StreamEnded { rank } => {
                ended += 1;
                if ranks[rank].is_none() {
                    failures.push(format!("rank {rank} stream ended without a result"));
                }
            }
            HostMessage::Exit { rank, code } => {
                exited += 1;
                exit_codes[rank] = Some(code);
                if code != 0 {
                    failures.push(format!("rank {rank} exited with status {code}"));
                    stop_containers(container_names);
                }
            }
        }
        if ended == world_size && exited == world_size {
            break;
        }
    }
    let _ = dashboard.send(DashboardMessage::Finished);
    Collected {
        ranks,
        events,
        exit_codes,
        failures,
    }
}

fn print_text_event(event: &dlir_pipeline::PipelineEvent) {
    match &event.event.event {
        dlir_runtime::RunEventKind::ModelLoadStarted => {
            eprintln!("[rank {}] loading tensor shard...", event.event.rank)
        }
        dlir_runtime::RunEventKind::TokenGenerated { token_id, .. } => {
            eprintln!("[rank 0] generated token {token_id}")
        }
        dlir_runtime::RunEventKind::GenerationFinished { stop_reason } => {
            eprintln!("[rank {}] finished: {stop_reason}", event.event.rank)
        }
        _ => {}
    }
}

struct ArtifactMounts {
    manifest: PathBuf,
    checkpoint: PathBuf,
    tokenizer: PathBuf,
}

#[allow(clippy::too_many_arguments)]
fn tensor_container_arguments(
    image: &str,
    network: &str,
    name: &str,
    run_id: &str,
    rank: usize,
    world_size: usize,
    resources: &ResourcePlan,
    startup: Duration,
    operation: Duration,
    mounts: &ArtifactMounts,
) -> Vec<String> {
    let mut args = vec![
        "run".into(),
        "--detach".into(),
        "--name".into(),
        name.into(),
        "--network".into(),
        network.into(),
        "--network-alias".into(),
        format!("rank-{rank}"),
        "--label".into(),
        format!("{RUN_LABEL}={run_id}"),
        "--cpus".into(),
        format_cpu(resources.per_rank_cpu_millis),
        "--memory".into(),
        format!("{}b", resources.per_rank_memory_bytes),
        "--memory-swap".into(),
        format!("{}b", resources.per_rank_memory_bytes),
    ];
    for (source, target) in [
        (&mounts.manifest, CONTAINER_MANIFEST),
        (&mounts.checkpoint, CONTAINER_CHECKPOINT),
        (&mounts.tokenizer, CONTAINER_TOKENIZER),
    ] {
        args.push("--mount".into());
        args.push(format!(
            "type=bind,source={},target={target},readonly",
            source.display()
        ));
    }
    args.extend([
        image.into(),
        "rank".into(),
        "--workload".into(),
        "tensor".into(),
        "--tensor-manifest".into(),
        CONTAINER_MANIFEST.into(),
        "--rank".into(),
        rank.to_string(),
        "--world-size".into(),
        world_size.to_string(),
        "--run-id".into(),
        run_id.into(),
        "--rendezvous-addr".into(),
        format!("rank-0:{RENDEZVOUS_PORT}"),
        "--listen-addr".into(),
        format!("0.0.0.0:{PEER_PORT}"),
        "--advertise-addr".into(),
        format!("rank-{rank}:{PEER_PORT}"),
        "--startup-timeout-seconds".into(),
        startup.as_secs().to_string(),
        "--operation-timeout-seconds".into(),
        operation.as_secs().to_string(),
        "--expected-cpu-millis".into(),
        resources.per_rank_cpu_millis.to_string(),
        "--expected-memory-bytes".into(),
        resources.per_rank_memory_bytes.to_string(),
    ]);
    if rank == 0 {
        args.extend([
            "--rendezvous-bind-addr".into(),
            format!("0.0.0.0:{RENDEZVOUS_PORT}"),
        ]);
    }
    args
}

fn resource_report(
    engine: &crate::launch::DockerEngineReport,
    resources: &ResourcePlan,
) -> TensorParallelResourcePlan {
    TensorParallelResourcePlan {
        engine_cpu_millis: engine.cpu_millis,
        engine_memory_bytes: engine.memory_bytes,
        requested_cpu_millis: resources.requested_cpu_millis,
        requested_memory_bytes: resources.requested_memory_bytes,
        per_rank_cpu_millis: resources.per_rank_cpu_millis,
        per_rank_memory_bytes: resources.per_rank_memory_bytes,
        unused_cpu_millis: resources.unused_cpu_millis,
        unused_memory_bytes: resources.unused_memory_bytes,
    }
}

fn resource_snapshot(observed: crate::launch::ResourceObservation) -> ResourceSnapshot {
    ResourceSnapshot {
        cpu_millis: observed.cpu_millis,
        memory_current_bytes: observed.memory_current_bytes,
        memory_limit_bytes: observed.memory_limit_bytes,
        cpuset_cpus: observed.cpuset_cpus,
        cgroup_version: observed.cgroup_version,
    }
}

fn append_final_memory_sample(
    report: &mut dlir_tensor::TensorParallelRankReport,
    sink: &dyn TensorEventSink,
) {
    let event = PipelineEvent {
        schema_version: 1,
        run_id: report.run_id.clone(),
        request_id: report.request_id.clone(),
        event: dlir_runtime::RunEvent {
            sequence: report.events.len() as u64,
            rank: report.rank,
            elapsed_ns: report.timings.total_ns,
            event: dlir_runtime::RunEventKind::MemorySample {
                current_bytes: report.resources.memory_current_bytes,
                limit_bytes: report.resources.memory_limit_bytes,
            },
        },
    };
    sink.publish(&event);
    report.events.push(event);
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("request-{nanos:x}")
}

fn ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
