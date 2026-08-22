//! Docker pipeline generation, live rank-stream aggregation, and report presentation.

use crate::launch::{
    CpuAmount, DockerResources, PEER_PORT, RENDEZVOUS_PORT, RUN_LABEL, ResourcePlan,
    default_run_id, docker_checked, docker_engine_info, docker_output, ensure_image, format_cpu,
    install_interrupt_handler, is_interrupted, observe_cgroup_resources, plan_resources,
    request_interrupt, reset_interrupted, stop_containers, validate_run_id, verify_resources,
};
use anyhow::{Context, Result, bail};
use dlir_collectives::{DEFAULT_MAX_TENSOR_BYTES, Rank, TcpTransport, TcpTransportConfig};
use dlir_pipeline::{
    PipelineEvent, PipelineEventSink, PipelineManifest, PipelinePartition, PipelineRankReport,
    PipelineReport, PipelineResourcePlan, PipelineStreamRecord, PipelineTimingReport,
    ReceivedPipelineEvent, ResourceSnapshot, StageMemoryPlan, run_pipeline_rank,
};
use dlir_runtime::{
    ArtifactRepository, PlacementVerdict, PlanDType, RunEvent, RunEventKind, SupportedModelId,
    format_bytes, render_prompt, validate_checkpoint, validate_metadata,
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

const CONTAINER_MANIFEST: &str = "/opt/dlir/request/pipeline.json";
const CONTAINER_CHECKPOINT: &str = "/opt/dlir/artifacts/model.safetensors";
const CONTAINER_TOKENIZER: &str = "/opt/dlir/artifacts/tokenizer.json";
const CONTAINER_CONFIG: &str = "/opt/dlir/artifacts/config.json";
const CONTAINER_TOKENIZER_CONFIG: &str = "/opt/dlir/artifacts/tokenizer_config.json";

pub(crate) struct PipelineLaunchRequest {
    pub(crate) model: SupportedModelId,
    pub(crate) dtype: PlanDType,
    pub(crate) prompt: String,
    pub(crate) max_new_tokens: usize,
    pub(crate) nproc: usize,
    pub(crate) total_cpus: CpuAmount,
    pub(crate) total_memory: String,
    pub(crate) image: String,
    pub(crate) build_context: PathBuf,
    pub(crate) rebuild: bool,
    pub(crate) run_id: Option<String>,
    pub(crate) startup_timeout: Duration,
    pub(crate) operation_timeout: Duration,
    pub(crate) tui: bool,
    pub(crate) report: Option<PathBuf>,
    pub(crate) keep_containers: bool,
}

pub(crate) fn run_pipeline(request: PipelineLaunchRequest) -> Result<()> {
    let cold_started = Instant::now();
    if request.tui && !io::stderr().is_terminal() {
        bail!("--tui requires stderr to be connected to an interactive terminal");
    }
    if request.max_new_tokens == 0 || request.prompt.trim().is_empty() {
        bail!("pipeline prompt must be non-empty and max-new-tokens must be at least one");
    }
    if request.dtype != PlanDType::F32 {
        bail!("v0.4 pipeline execution supports only CPU/F32");
    }
    let spec = request.model.spec();
    spec.validate_cpu_dtype(request.dtype)?;
    let partition = PipelinePartition::balanced(spec, request.nproc)?;
    let total_memory = dlir_runtime::parse_byte_size(&request.total_memory)?;
    let run_id = request.run_id.unwrap_or_else(default_run_id);
    validate_run_id(&run_id)?;
    let request_id = request_id();

    let artifact_started = Instant::now();
    eprintln!(
        "model: {} ({})\nrevision: {}\nresolving configuration and tokenizer...",
        spec.id, spec.repository, spec.revision
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
    let available = spec.config.max_position_embeddings - prompt_token_ids.len();
    let effective_max_new_tokens = request.max_new_tokens.min(available);
    let context_capacity = prompt_token_ids.len() + effective_max_new_tokens;

    let engine = docker_engine_info()?;
    let resources = plan_resources(request.nproc, request.total_cpus, total_memory, &engine)?;
    let memory_plans = partition
        .stages
        .iter()
        .map(|stage| {
            StageMemoryPlan::for_stage(
                spec,
                stage,
                request.dtype,
                context_capacity,
                resources.per_rank_memory_bytes,
            )
        })
        .collect::<dlir_pipeline::Result<Vec<_>>>()?;
    if let Some(failed) = memory_plans
        .iter()
        .find(|plan| plan.placement == PlacementVerdict::DoesNotFit)
    {
        bail!(
            "rank {} placement failed before checkpoint download: {} required, {} available",
            failed.rank,
            format_bytes(failed.persistent_bytes),
            format_bytes(failed.budget_bytes)
        );
    }
    eprintln!("resolving checkpoint weights once on the host...");
    let weights = repository.download_weights(spec)?;
    validate_checkpoint(spec, &weights)?;
    let artifact_resolution_ns = ns(artifact_started.elapsed());

    let run_directory = tempfile::Builder::new()
        .prefix(&format!("dlir-{run_id}-"))
        .tempdir()
        .context("could not create pipeline run directory")?;
    let manifest_path = run_directory.path().join("pipeline.json");
    let manifest = PipelineManifest {
        schema_version: 1,
        run_id: run_id.clone(),
        request_id: request_id.clone(),
        model: request.model,
        dtype: request.dtype,
        prompt_token_ids,
        prompt_characters: request.prompt.chars().count(),
        requested_max_new_tokens: request.max_new_tokens,
        effective_max_new_tokens,
        context_capacity,
        checkpoint_path: PathBuf::from(CONTAINER_CHECKPOINT),
        tokenizer_path: PathBuf::from(CONTAINER_TOKENIZER),
        partition: partition.clone(),
        memory_plans: memory_plans.clone(),
        expected_cpu_millis: resources.per_rank_cpu_millis,
        expected_memory_bytes: resources.per_rank_memory_bytes,
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    install_interrupt_handler()?;
    reset_interrupted();
    ensure_image(&request.image, &request.build_context, request.rebuild)?;
    let network = format!("dlir-{run_id}");
    docker_checked(&[
        "network".to_owned(),
        "create".to_owned(),
        "--label".to_owned(),
        format!("{RUN_LABEL}={run_id}"),
        network.clone(),
    ])?;
    let mut docker_resources = DockerResources {
        network,
        containers: Vec::new(),
        keep: request.keep_containers,
    };

    let mounts = ArtifactMounts::new(&manifest_path, &metadata, &weights)?;
    let mut identities = Vec::with_capacity(request.nproc);
    for rank in 0..request.nproc {
        let name = format!("dlir-{run_id}-rank-{rank}");
        eprintln!(
            "starting pipeline rank {rank}: layers {}..{}, {} CPU, {}",
            partition.stages[rank].layer_start,
            partition.stages[rank].layer_end,
            format_cpu(resources.per_rank_cpu_millis),
            format_bytes(resources.per_rank_memory_bytes)
        );
        let arguments = pipeline_container_arguments(
            &request.image,
            &docker_resources.network,
            &name,
            &run_id,
            rank,
            request.nproc,
            &resources,
            request.startup_timeout,
            request.operation_timeout,
            &mounts,
        );
        let output = docker_checked(&arguments)?;
        let container_id = String::from_utf8(output.stdout)
            .context("Docker returned a non-UTF-8 container ID")?
            .trim()
            .to_owned();
        docker_resources.containers.push(name.clone());
        identities.push((rank, name, container_id));
    }

    let (sender, receiver) = mpsc::channel();
    for (rank, name, _) in &identities {
        follow_container(*rank, name.clone(), sender.clone());
        wait_container(*rank, name.clone(), sender.clone());
    }
    drop(sender);
    let (dashboard_sender, dashboard_receiver) = mpsc::channel();
    let tui_active = Arc::new(AtomicBool::new(request.tui));
    let collector_active = Arc::clone(&tui_active);
    let container_names = docker_resources.containers.clone();
    let world_size = request.nproc;
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
        let mut dashboard = DashboardState::new(spec.id.as_str(), &partition.stages, &memory_plans);
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
        .map_err(|_| anyhow::anyhow!("pipeline stream collector panicked"))?;

    let rank_zero = collected.ranks.first().and_then(Option::as_ref);
    let generated_tokens = rank_zero
        .map(|rank| rank.generated_tokens.clone())
        .unwrap_or_default();
    let completion = rank_zero
        .map(|rank| rank.completion.clone())
        .unwrap_or_default();
    let stop_reason = rank_zero
        .map(|rank| rank.stop_reason)
        .unwrap_or(dlir_runtime::StopReason::MaxNewTokens);
    let ranks = collected.ranks.into_iter().flatten().collect::<Vec<_>>();
    let communication_bytes = ranks
        .iter()
        .map(|rank| rank.communication.tensor_bytes_sent + rank.communication.control_bytes_sent)
        .sum();
    let rank_zero_timing = ranks.iter().find(|rank| rank.rank == 0);
    let prefill_ns = rank_zero_timing
        .map(|rank| rank.timings.prefill_ns)
        .unwrap_or(0);
    let time_to_first_token_ns = rank_zero_timing
        .and_then(|rank| rank.timings.time_to_first_token_ns)
        .unwrap_or(0);
    let decode_forward_count = rank_zero_timing
        .map(|rank| rank.timings.decode_forward_count)
        .unwrap_or(0);
    let decode_total_ns = rank_zero_timing
        .map(|rank| rank.timings.decode_total_ns)
        .unwrap_or(0);
    let success = collected.failures.is_empty()
        && ranks.len() == request.nproc
        && ranks.iter().all(|rank| rank.success)
        && collected.exit_codes.iter().all(|code| *code == Some(0));
    let materialized_parameters = memory_plans
        .iter()
        .map(|plan| plan.materialized_parameters)
        .sum::<u64>();
    let report = PipelineReport {
        schema_version: 1,
        run_id: run_id.clone(),
        request_id,
        model: request.model,
        repository: spec.repository.to_owned(),
        revision: spec.revision.to_owned(),
        device: "cpu".to_owned(),
        dtype: request.dtype,
        backend: "tcp".to_owned(),
        world_size: request.nproc,
        tensor_parallel: 1,
        pipeline_parallel: request.nproc,
        expert_parallel: 1,
        partition,
        model_parameters: spec.expected_parameters,
        materialized_parameters,
        duplicated_parameters: materialized_parameters.saturating_sub(spec.expected_parameters),
        resources: pipeline_resource_report(&engine, &resources),
        prompt_tokens: manifest.prompt_token_ids.len(),
        requested_max_new_tokens: request.max_new_tokens,
        generated_tokens,
        completion: completion.clone(),
        stop_reason,
        ranks,
        events: collected.events,
        timings: PipelineTimingReport {
            artifact_resolution_ns,
            prefill_ns,
            time_to_first_token_ns,
            decode_total_ns,
            decode_forward_count,
            mean_decode_ns: (decode_forward_count > 0)
                .then_some(decode_total_ns / decode_forward_count as u64),
            cold_start_total_ns: ns(cold_started.elapsed()),
        },
        communication_bytes,
        failures: collected.failures,
        success,
    };

    println!("{}", report.completion);
    io::stdout().flush()?;
    print_pipeline_summary(&report);
    if let Some(path) = &request.report {
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        eprintln!("report: {}", path.display());
    }

    if request.keep_containers {
        let kept_path = run_directory.keep();
        eprintln!("retained request manifest at {}", kept_path.display());
    } else {
        docker_resources.cleanup();
    }
    if !report.success {
        bail!("Docker pipeline generation failed");
    }
    Ok(())
}

pub(crate) fn run_pipeline_rank_process(
    rank_request: crate::launch::RankRequest,
    manifest_path: &Path,
) -> Result<()> {
    let manifest: PipelineManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported pipeline manifest schema {}",
            manifest.schema_version
        );
    }
    validate_checkpoint(manifest.model.spec(), &manifest.checkpoint_path)?;
    let rank = Rank::new(rank_request.rank, rank_request.world_size)?;
    let observed = observe_cgroup_resources();
    verify_resources(
        &observed,
        rank_request.expected_cpu_millis,
        rank_request.expected_memory_bytes,
    )?;
    let resources = resource_snapshot(observed);
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
    let mut sink = JsonLineSink;
    let mut report = run_pipeline_rank(transport, &manifest, resources, peers, &mut sink)?;
    let final_observed = observe_cgroup_resources();
    verify_resources(
        &final_observed,
        rank_request.expected_cpu_millis,
        rank_request.expected_memory_bytes,
    )?;
    report.resources = resource_snapshot(final_observed);
    let final_sample = PipelineEvent {
        schema_version: 1,
        run_id: manifest.run_id.clone(),
        request_id: manifest.request_id.clone(),
        event: RunEvent {
            sequence: report.events.len() as u64,
            rank: report.rank,
            elapsed_ns: report.timings.total_ns,
            event: RunEventKind::MemorySample {
                current_bytes: report.resources.memory_current_bytes,
                limit_bytes: report.resources.memory_limit_bytes,
            },
        },
    };
    sink.publish(&final_sample);
    report.events.push(final_sample);
    println!(
        "{}",
        serde_json::to_string(&PipelineStreamRecord::Result {
            result: Box::new(report),
        })?
    );
    io::stdout().flush()?;
    Ok(())
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

struct JsonLineSink;

impl PipelineEventSink for JsonLineSink {
    fn publish(&mut self, event: &PipelineEvent) {
        if let Ok(line) = serde_json::to_string(&PipelineStreamRecord::Event {
            event: event.clone(),
        }) {
            println!("{line}");
            let _ = io::stdout().flush();
        }
    }
}

struct ArtifactMounts {
    manifest: PathBuf,
    checkpoint: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
    tokenizer_config: PathBuf,
}

impl ArtifactMounts {
    fn new(
        manifest: &Path,
        metadata: &dlir_runtime::MetadataArtifacts,
        checkpoint: &Path,
    ) -> Result<Self> {
        Ok(Self {
            manifest: fs::canonicalize(manifest)?,
            checkpoint: fs::canonicalize(checkpoint)?,
            tokenizer: fs::canonicalize(&metadata.tokenizer)?,
            config: fs::canonicalize(&metadata.config)?,
            tokenizer_config: fs::canonicalize(&metadata.tokenizer_config)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn pipeline_container_arguments(
    image: &str,
    network: &str,
    name: &str,
    run_id: &str,
    rank: usize,
    world_size: usize,
    resources: &ResourcePlan,
    startup_timeout: Duration,
    operation_timeout: Duration,
    mounts: &ArtifactMounts,
) -> Vec<String> {
    let mut arguments = vec![
        "run".to_owned(),
        "--detach".to_owned(),
        "--name".to_owned(),
        name.to_owned(),
        "--network".to_owned(),
        network.to_owned(),
        "--network-alias".to_owned(),
        format!("rank-{rank}"),
        "--label".to_owned(),
        format!("{RUN_LABEL}={run_id}"),
        "--cpus".to_owned(),
        format_cpu(resources.per_rank_cpu_millis),
        "--memory".to_owned(),
        format!("{}b", resources.per_rank_memory_bytes),
        "--memory-swap".to_owned(),
        format!("{}b", resources.per_rank_memory_bytes),
    ];
    for (source, destination) in [
        (&mounts.manifest, CONTAINER_MANIFEST),
        (&mounts.checkpoint, CONTAINER_CHECKPOINT),
        (&mounts.tokenizer, CONTAINER_TOKENIZER),
        (&mounts.config, CONTAINER_CONFIG),
        (&mounts.tokenizer_config, CONTAINER_TOKENIZER_CONFIG),
    ] {
        arguments.push("--mount".to_owned());
        arguments.push(format!(
            "type=bind,source={},target={destination},readonly",
            source.display()
        ));
    }
    arguments.extend([
        image.to_owned(),
        "rank".to_owned(),
        "--workload".to_owned(),
        "pipeline".to_owned(),
        "--pipeline-manifest".to_owned(),
        CONTAINER_MANIFEST.to_owned(),
        "--rank".to_owned(),
        rank.to_string(),
        "--world-size".to_owned(),
        world_size.to_string(),
        "--run-id".to_owned(),
        run_id.to_owned(),
        "--rendezvous-addr".to_owned(),
        format!("rank-0:{RENDEZVOUS_PORT}"),
        "--listen-addr".to_owned(),
        format!("0.0.0.0:{PEER_PORT}"),
        "--advertise-addr".to_owned(),
        format!("rank-{rank}:{PEER_PORT}"),
        "--startup-timeout-seconds".to_owned(),
        startup_timeout.as_secs().to_string(),
        "--operation-timeout-seconds".to_owned(),
        operation_timeout.as_secs().to_string(),
        "--expected-cpu-millis".to_owned(),
        resources.per_rank_cpu_millis.to_string(),
        "--expected-memory-bytes".to_owned(),
        resources.per_rank_memory_bytes.to_string(),
    ]);
    if rank == 0 {
        arguments.push("--rendezvous-bind-addr".to_owned());
        arguments.push(format!("0.0.0.0:{RENDEZVOUS_PORT}"));
    }
    arguments
}

#[derive(Debug)]
enum HostMessage {
    Record {
        rank: usize,
        record: PipelineStreamRecord,
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
                match serde_json::from_str::<PipelineStreamRecord>(&line) {
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
    ranks: Vec<Option<PipelineRankReport>>,
    events: Vec<ReceivedPipelineEvent>,
    exit_codes: Vec<Option<i32>>,
    failures: Vec<String>,
}

fn collect_rank_streams(
    receiver: mpsc::Receiver<HostMessage>,
    dashboard: mpsc::Sender<DashboardMessage>,
    tui_active: Arc<AtomicBool>,
    world_size: usize,
    containers: &[String],
) -> Collected {
    let mut collected = Collected {
        ranks: (0..world_size).map(|_| None).collect(),
        events: Vec::new(),
        exit_codes: vec![None; world_size],
        failures: Vec::new(),
    };
    let mut sequences = vec![0u64; world_size];
    let mut ended = vec![false; world_size];
    let mut exited = vec![false; world_size];
    let mut stopped = false;
    while ended.iter().any(|done| !done) || exited.iter().any(|done| !done) {
        if is_interrupted() && !stopped {
            stop_containers(containers);
            stopped = true;
        }
        let message = match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(message) => message,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        match message {
            HostMessage::Record { rank, record } => match record {
                PipelineStreamRecord::Event { event } => {
                    if event.event.rank != rank || event.event.sequence != sequences[rank] {
                        collected.failures.push(format!(
                            "rank {rank} event sequence {}, expected {}",
                            event.event.sequence, sequences[rank]
                        ));
                    } else {
                        sequences[rank] += 1;
                    }
                    let receive_sequence = collected.events.len() as u64;
                    collected.events.push(ReceivedPipelineEvent {
                        receive_sequence,
                        published: event.clone(),
                    });
                    let _ = dashboard.send(DashboardMessage::Event(event.clone()));
                    if !tui_active.load(Ordering::SeqCst) {
                        print_text_event(&event);
                    }
                }
                PipelineStreamRecord::Result { result } => {
                    let result = *result;
                    if result.rank != rank {
                        collected.failures.push(format!(
                            "rank {rank} stream reported result for rank {}",
                            result.rank
                        ));
                    } else {
                        collected.ranks[rank] = Some(result);
                    }
                }
            },
            HostMessage::Log { rank, line } => {
                eprintln!("[rank {rank}] {line}");
                if line.starts_with("invalid rank JSON") {
                    collected.failures.push(format!("rank {rank}: {line}"));
                }
            }
            HostMessage::StreamEnded { rank } => ended[rank] = true,
            HostMessage::Exit { rank, code } => {
                exited[rank] = true;
                collected.exit_codes[rank] = Some(code);
                if code != 0 && !stopped {
                    collected
                        .failures
                        .push(format!("rank {rank} exited with code {code}"));
                    stop_containers(containers);
                    stopped = true;
                }
            }
        }
    }
    if is_interrupted() {
        collected
            .failures
            .push("pipeline launch interrupted".to_owned());
    }
    for rank in 0..world_size {
        if collected.ranks[rank].is_none() {
            collected
                .failures
                .push(format!("rank {rank} emitted no final report"));
        }
    }
    let _ = dashboard.send(DashboardMessage::Finished);
    collected
}

fn print_text_event(event: &PipelineEvent) {
    match &event.event.event {
        dlir_runtime::RunEventKind::ModelLoadStarted => {
            eprintln!("[rank {}] loading stage...", event.event.rank)
        }
        dlir_runtime::RunEventKind::ModelLoadFinished => {
            eprintln!("[rank {}] stage ready", event.event.rank)
        }
        _ => {}
    }
}

fn pipeline_resource_report(
    engine: &crate::launch::DockerEngineReport,
    plan: &ResourcePlan,
) -> PipelineResourcePlan {
    PipelineResourcePlan {
        engine_cpu_millis: engine.cpu_millis,
        engine_memory_bytes: engine.memory_bytes,
        requested_cpu_millis: plan.requested_cpu_millis,
        requested_memory_bytes: plan.requested_memory_bytes,
        per_rank_cpu_millis: plan.per_rank_cpu_millis,
        per_rank_memory_bytes: plan.per_rank_memory_bytes,
        unused_cpu_millis: plan.unused_cpu_millis,
        unused_memory_bytes: plan.unused_memory_bytes,
    }
}

fn print_pipeline_summary(report: &PipelineReport) {
    eprintln!(
        "\n\nPIPELINE RUN SUMMARY\nworld size / PP:    {}\nprompt tokens:      {}\ngenerated tokens:   {}\nstop reason:        {}\nmodel parameters:   {}\nmaterialized:       {} ({} duplicated)\ncommunication:      {}\nprefill:            {:.3} ms\nTTFT:               {:.3} ms\nmean decode:        {}\ncold start:         {:.3} s",
        report.world_size,
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.stop_reason,
        report.model_parameters,
        report.materialized_parameters,
        report.duplicated_parameters,
        format_bytes(report.communication_bytes),
        report.timings.prefill_ns as f64 / 1_000_000.0,
        report.timings.time_to_first_token_ns as f64 / 1_000_000.0,
        report.timings.mean_decode_ns.map_or_else(
            || "n/a".to_owned(),
            |ns| format!("{:.3} ms", ns as f64 / 1_000_000.0)
        ),
        report.timings.cold_start_total_ns as f64 / 1_000_000_000.0,
    );
    for rank in &report.ranks {
        eprintln!(
            "rank {} layers {}..{}: weights {}, KV {}/{} used, cgroup {}/{}, load {:.3} s, layers {:.3} ms, comm {}",
            rank.rank,
            rank.assignment.layer_start,
            rank.assignment.layer_end,
            format_bytes(rank.memory.weight_bytes),
            format_bytes(rank.final_kv_cache_bytes),
            format_bytes(rank.memory.kv_cache_capacity_bytes),
            rank.resources
                .memory_current_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "unknown".to_owned()),
            rank.resources
                .memory_limit_bytes
                .map(format_bytes)
                .unwrap_or_else(|| "unknown".to_owned()),
            rank.timings.model_load_ns as f64 / 1_000_000_000.0,
            rank.timings.layer_compute_ns as f64 / 1_000_000.0,
            format_bytes(
                rank.communication.tensor_bytes_sent + rank.communication.control_bytes_sent
            ),
        );
    }
    eprintln!(
        "result:             {}",
        if report.success { "PASS" } else { "FAIL" }
    );
}

fn request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("request-{nanos}")
}

fn ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_container_is_read_only_resource_limited_and_selects_workload() {
        let resources = ResourcePlan {
            requested_cpu_millis: 1000,
            requested_memory_bytes: 512 * crate::launch::MIB,
            per_rank_cpu_millis: 500,
            per_rank_memory_bytes: 256 * crate::launch::MIB,
            allocated_cpu_millis: 1000,
            allocated_memory_bytes: 512 * crate::launch::MIB,
            unused_cpu_millis: 0,
            unused_memory_bytes: 0,
            engine_cpu_headroom_millis: 1000,
            engine_memory_headroom_bytes: 512 * crate::launch::MIB,
        };
        let mounts = ArtifactMounts {
            manifest: "/tmp/manifest.json".into(),
            checkpoint: "/tmp/model.safetensors".into(),
            tokenizer: "/tmp/tokenizer.json".into(),
            config: "/tmp/config.json".into(),
            tokenizer_config: "/tmp/tokenizer_config.json".into(),
        };
        let arguments = pipeline_container_arguments(
            "dlir:v0.4-pipeline",
            "network",
            "rank-0",
            "run",
            0,
            2,
            &resources,
            Duration::from_secs(30),
            Duration::from_secs(30),
            &mounts,
        );
        assert!(arguments.windows(2).any(|pair| pair == ["--cpus", "0.5"]));
        assert!(
            arguments
                .windows(2)
                .any(|pair| pair == ["--workload", "pipeline"])
        );
        assert_eq!(
            arguments
                .iter()
                .filter(|argument| argument.contains(",readonly"))
                .count(),
            5
        );
        assert!(arguments.iter().any(|argument| {
            argument.contains("pipeline.json") && argument.contains(CONTAINER_MANIFEST)
        }));
    }
}
