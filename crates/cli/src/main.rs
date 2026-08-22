//! Command-line presentation layer for the dlir inference laboratory.
//!
//! The binary converts CLI arguments into runtime requests, renders model, inspection, and
//! point-to-point results, launches Docker rank processes, orchestrates pipeline requests, streams
//! assistant text and rank events, writes optional JSON reports, and owns exit behavior. Model
//! execution remains in `dlir-runtime`/`dlir-pipeline`/`dlir-tensor`; communication remains in
//! `dlir-collectives`; terminal reduction remains in `dlir-tui`.

mod benchmark;
mod launch;
mod pipeline;
mod tensor;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dlir_collectives::{
    AllReduceAlgorithm, CollectiveCheckReport, P2pReport, run_collective_check, run_p2p_ring,
};
use dlir_runtime::{
    EventObserver, GenerationRequest, InspectionReport, InspectionRequest, MemoryBudget, ModelSpec,
    PlanDType, RankMemoryPlan, RunEvent, RunEventKind, SupportedModelId, format_bytes, generate,
    inspect, supported_models,
};
use serde_json::json;
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use benchmark::CollectiveBenchRequest;
use launch::{CpuAmount, LaunchRequest, RankRequest};
use pipeline::PipelineLaunchRequest;
use tensor::TensorLaunchRequest;

const P2P_RECEIVE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Parser)]
#[command(
    name = "dlir",
    version,
    about = "A distributed LLM inference laboratory"
)]
struct Cli {
    /// Operation to perform.
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Check or benchmark native collectives derived from send and receive.
    Collectives {
        /// Collective operation to run.
        #[command(subcommand)]
        command: CollectivesCommand,
    },
    /// Generate one completion with tensor-sharded Llama ranks in Docker.
    Tensor {
        /// Exact identifier from `dlir models`.
        #[arg(long)]
        model: SupportedModelId,
        /// Tensor rank and container count; must be 2..=64 and divide model dimensions.
        #[arg(long)]
        tp: usize,
        /// Execution device; v0.5 accepts only cpu.
        #[arg(long, value_enum, default_value_t)]
        device: DeviceArg,
        /// Runtime dtype; v0.5 accepts only f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// Non-empty user message wrapped in the registered chat template.
        #[arg(long)]
        prompt: String,
        /// Maximum generated non-EOS tokens.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// Total CPU quota divided equally across tensor ranks.
        #[arg(long)]
        total_cpus: CpuAmount,
        /// Total enforced memory divided equally across tensor ranks.
        #[arg(long)]
        total_memory: String,
        /// Native all-reduce algorithm used inside embeddings and every block.
        #[arg(long, value_enum, default_value_t)]
        all_reduce: AllReduceArg,
        /// Display the read-only tensor-parallel dashboard on stderr.
        #[arg(long)]
        tui: bool,
        /// Write the complete schema-v1 report.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Docker image to reuse or build.
        #[arg(long, default_value = "dlir:v0.5-tensor")]
        image: String,
        /// Directory containing the Dockerfile and workspace.
        #[arg(long, default_value = ".")]
        build_context: PathBuf,
        /// Force a no-cache image rebuild.
        #[arg(long)]
        rebuild: bool,
        /// Stable run identifier; generated when omitted.
        #[arg(long)]
        run_id: Option<String>,
        /// Artifact and topology startup deadline.
        #[arg(long, default_value_t = 30)]
        startup_timeout_seconds: u64,
        /// Collective/control/barrier receive deadline.
        #[arg(long, default_value_t = 30)]
        operation_timeout_seconds: u64,
        /// Retain stopped resources and request manifest.
        #[arg(long)]
        keep_containers: bool,
    },
    /// Generate one completion across CPU pipeline stages in Docker containers.
    Pipeline {
        /// Exact identifier from `dlir models`; arbitrary Hub IDs are rejected.
        #[arg(long)]
        model: SupportedModelId,
        /// Execution device; v0.4 pipeline generation accepts only cpu.
        #[arg(long, value_enum, default_value_t)]
        device: DeviceArg,
        /// Runtime dtype; v0.4 pipeline generation accepts only f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// Non-empty user message to wrap in the model's registered chat template.
        #[arg(long)]
        prompt: String,
        /// Maximum number of non-EOS tokens to emit; must be at least one.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// Number of CPU pipeline-stage containers; must be between 2 and 64.
        #[arg(long)]
        nproc: usize,
        /// Total CPU quota to divide equally, with up to three decimal places.
        #[arg(long)]
        total_cpus: CpuAmount,
        /// Total memory to divide equally using bytes, KiB, MiB, or GiB.
        #[arg(long)]
        total_memory: String,
        /// Docker image to reuse or build when missing.
        #[arg(long, default_value = "dlir:v0.4-pipeline")]
        image: String,
        /// Directory containing the checked-in Dockerfile and workspace.
        #[arg(long, default_value = ".")]
        build_context: PathBuf,
        /// Force a no-cache image rebuild.
        #[arg(long)]
        rebuild: bool,
        /// Stable run identifier; generated when omitted.
        #[arg(long)]
        run_id: Option<String>,
        /// Artifact, rendezvous, and connection-establishment deadline.
        #[arg(long, default_value_t = 30)]
        startup_timeout_seconds: u64,
        /// Activation, control, and barrier receive deadline.
        #[arg(long, default_value_t = 30)]
        operation_timeout_seconds: u64,
        /// Display the read-only live pipeline dashboard on stderr.
        #[arg(long)]
        tui: bool,
        /// Write the complete schema-v1 distributed pipeline report as JSON.
        #[arg(long)]
        report: Option<PathBuf>,
        /// Retain stopped containers, network, and request manifest for inspection.
        #[arg(long)]
        keep_containers: bool,
    },
    /// Start one TCP rank per Docker container and verify the topology.
    Launch {
        /// Number of rank containers; must be between 2 and 64.
        #[arg(long)]
        nproc: usize,
        /// Total CPU quota to divide equally, with up to three decimal places.
        #[arg(long)]
        total_cpus: CpuAmount,
        /// Total memory to divide equally using bytes, KiB, MiB, or GiB.
        #[arg(long)]
        total_memory: String,
        /// Docker image to reuse or build when missing.
        #[arg(long, default_value = "dlir:v0.3-tcp")]
        image: String,
        /// Directory containing the checked-in Dockerfile and workspace.
        #[arg(long, default_value = ".")]
        build_context: PathBuf,
        /// Force a no-cache image rebuild.
        #[arg(long)]
        rebuild: bool,
        /// Stable run identifier; generated when omitted.
        #[arg(long)]
        run_id: Option<String>,
        /// Rendezvous and connection-establishment deadline.
        #[arg(long, default_value_t = 30)]
        startup_timeout_seconds: u64,
        /// Point-to-point receive and barrier deadline.
        #[arg(long, default_value_t = 10)]
        operation_timeout_seconds: u64,
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
        /// Retain stopped containers and the run network for inspection.
        #[arg(long)]
        keep_containers: bool,
    },
    /// Run exactly one TCP rank; normally invoked as container PID 1 by `dlir launch`.
    Rank {
        /// Internal rank workload. Docker topology checks use `topology`.
        #[arg(long, value_enum, default_value_t)]
        workload: RankWorkloadArg,
        /// Read-only pipeline request manifest, required by the pipeline workload.
        #[arg(long, required_if_eq("workload", "pipeline"))]
        pipeline_manifest: Option<PathBuf>,
        /// Read-only tensor request manifest, required by the tensor workload.
        #[arg(long, required_if_eq("workload", "tensor"))]
        tensor_manifest: Option<PathBuf>,
        /// Read-only collective benchmark manifest.
        #[arg(long, required_if_eq("workload", "collective-benchmark"))]
        benchmark_manifest: Option<PathBuf>,
        /// Zero-based global rank.
        #[arg(long)]
        rank: usize,
        /// Number of ranks in the rendezvous world.
        #[arg(long)]
        world_size: usize,
        /// Run identity shared by every rank.
        #[arg(long)]
        run_id: String,
        /// Rank-0 rendezvous address reachable by every container.
        #[arg(long)]
        rendezvous_addr: String,
        /// Rank-0-only local rendezvous bind address.
        #[arg(long)]
        rendezvous_bind_addr: Option<String>,
        /// Local peer-listener bind address.
        #[arg(long)]
        listen_addr: String,
        /// Peer-listener address advertised through rendezvous.
        #[arg(long)]
        advertise_addr: String,
        /// Rendezvous and connection-establishment deadline.
        #[arg(long, default_value_t = 30)]
        startup_timeout_seconds: u64,
        /// Point-to-point receive and barrier deadline.
        #[arg(long, default_value_t = 10)]
        operation_timeout_seconds: u64,
        /// CPU quota the launcher expects this cgroup to expose, in millicpus.
        #[arg(long, requires = "expected_memory_bytes")]
        expected_cpu_millis: Option<u64>,
        /// Memory maximum the launcher expects this cgroup to expose.
        #[arg(long, requires = "expected_cpu_millis")]
        expected_memory_bytes: Option<u64>,
    },
    /// Exchange copied CPU/F32 tensors between in-memory rank workers.
    P2p {
        /// Number of logical ranks in the ring; must be at least two.
        #[arg(long, default_value_t = 2)]
        world_size: usize,
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
    },
    /// List the closed set of model checkpoints supported by this release.
    Models {
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
    },
    /// Inspect architecture and logical memory without downloading artifacts.
    Inspect {
        /// Exact identifier from `dlir models`; arbitrary Hub IDs are rejected.
        #[arg(long)]
        model: SupportedModelId,
        /// Logical planning dtype: f16, bf16, or f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// KV-cache capacity to model in token positions.
        #[arg(long, default_value_t = 512)]
        context_length: usize,
        /// Optional advisory per-rank host budget using bytes, KiB, MiB, or GiB.
        #[arg(long)]
        device_memory_budget: Option<MemoryBudget>,
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
        /// Write the selected representation to a file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Generate one deterministic assistant completion on CPU.
    Generate {
        /// Exact identifier from `dlir models`; arbitrary Hub IDs are rejected.
        #[arg(long)]
        model: SupportedModelId,
        /// Execution device; v0.1 accepts only cpu.
        #[arg(long, value_enum, default_value_t)]
        device: DeviceArg,
        /// Runtime dtype; v0.1 CPU generation accepts only f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// Non-empty user message to wrap in the model's registered chat template.
        #[arg(long)]
        prompt: String,
        /// Maximum number of non-EOS tokens to emit; must be at least one.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// Advisory per-rank host budget in bytes/KiB/MiB/GiB, checked before weight download.
        #[arg(long)]
        device_memory_budget: Option<MemoryBudget>,
        /// Write the complete schema-v1 generation report as JSON.
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum CollectivesCommand {
    /// Verify all six native collectives over deterministic in-memory tensors.
    Check {
        /// Number of logical ranks; must be at least two.
        #[arg(long, default_value_t = 4)]
        world_size: usize,
        /// Render human-readable text or schema-v1 JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
        /// Write output to a file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Benchmark native all-reduce over reproducible Docker/TCP rank processes.
    Bench {
        /// Rank count.
        #[arg(long)]
        nproc: usize,
        /// Total Docker CPU budget divided and enforced across ranks.
        #[arg(long)]
        total_cpus: CpuAmount,
        /// Total Docker memory budget divided and enforced across ranks.
        #[arg(long)]
        total_memory: String,
        /// Algorithms to benchmark.
        #[arg(long, value_enum, default_value_t)]
        all_reduce: AllReduceSelection,
        /// Comma-separated IEC payload sizes.
        #[arg(long, default_value = "4KiB,64KiB,1MiB,16MiB")]
        sizes: String,
        /// Discarded iterations per case.
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        /// Measured iterations per case.
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        /// Text or schema-v1 JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
        /// Write output to a file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Docker image to reuse or build.
        #[arg(long, default_value = "dlir:v0.5-tensor")]
        image: String,
        /// Directory containing the Dockerfile and workspace.
        #[arg(long, default_value = ".")]
        build_context: PathBuf,
        /// Force a no-cache image rebuild.
        #[arg(long)]
        rebuild: bool,
        /// Stable run identity; generated when omitted.
        #[arg(long)]
        run_id: Option<String>,
        /// Rendezvous and connection deadline.
        #[arg(long, default_value_t = 30)]
        startup_timeout_seconds: u64,
        /// Collective and barrier receive deadline.
        #[arg(long, default_value_t = 30)]
        operation_timeout_seconds: u64,
        /// Retain stopped containers and network.
        #[arg(long)]
        keep_containers: bool,
    },
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum DeviceArg {
    #[default]
    Cpu,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum RankWorkloadArg {
    #[default]
    Topology,
    Pipeline,
    Tensor,
    CollectiveBenchmark,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum AllReduceArg {
    Centralized,
    #[default]
    Ring,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum AllReduceSelection {
    Centralized,
    Ring,
    #[default]
    Both,
}

impl From<AllReduceArg> for dlir_collectives::AllReduceAlgorithm {
    fn from(value: AllReduceArg) -> Self {
        match value {
            AllReduceArg::Centralized => Self::Centralized,
            AllReduceArg::Ring => Self::Ring,
        }
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Collectives { command } => match command {
            CollectivesCommand::Check {
                world_size,
                format,
                output,
            } => run_collectives_check(world_size, format, output.as_deref()),
            CollectivesCommand::Bench {
                nproc,
                total_cpus,
                total_memory,
                all_reduce,
                sizes,
                warmup,
                iterations,
                format,
                output,
                image,
                build_context,
                rebuild,
                run_id,
                startup_timeout_seconds,
                operation_timeout_seconds,
                keep_containers,
            } => benchmark::run_collective_bench(CollectiveBenchRequest {
                nproc,
                total_cpus,
                total_memory,
                algorithms: match all_reduce {
                    AllReduceSelection::Centralized => vec![AllReduceAlgorithm::Centralized],
                    AllReduceSelection::Ring => vec![AllReduceAlgorithm::Ring],
                    AllReduceSelection::Both => {
                        vec![AllReduceAlgorithm::Centralized, AllReduceAlgorithm::Ring]
                    }
                },
                sizes,
                warmup,
                iterations,
                json: matches!(format, OutputFormat::Json),
                output,
                image,
                build_context,
                rebuild,
                run_id,
                startup_timeout: Duration::from_secs(startup_timeout_seconds),
                operation_timeout: Duration::from_secs(operation_timeout_seconds),
                keep_containers,
            }),
        },
        Command::Tensor {
            model,
            tp,
            device: DeviceArg::Cpu,
            dtype,
            prompt,
            max_new_tokens,
            total_cpus,
            total_memory,
            all_reduce,
            tui,
            report,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout_seconds,
            operation_timeout_seconds,
            keep_containers,
        } => tensor::run_tensor(TensorLaunchRequest {
            model,
            tp,
            dtype,
            prompt,
            max_new_tokens,
            total_cpus,
            total_memory,
            all_reduce: all_reduce.into(),
            tui,
            report,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout: Duration::from_secs(startup_timeout_seconds),
            operation_timeout: Duration::from_secs(operation_timeout_seconds),
            keep_containers,
        }),
        Command::Pipeline {
            model,
            device: DeviceArg::Cpu,
            dtype,
            prompt,
            max_new_tokens,
            nproc,
            total_cpus,
            total_memory,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout_seconds,
            operation_timeout_seconds,
            tui,
            report,
            keep_containers,
        } => pipeline::run_pipeline(PipelineLaunchRequest {
            model,
            dtype,
            prompt,
            max_new_tokens,
            nproc,
            total_cpus,
            total_memory,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout: Duration::from_secs(startup_timeout_seconds),
            operation_timeout: Duration::from_secs(operation_timeout_seconds),
            tui,
            report,
            keep_containers,
        }),
        Command::Launch {
            nproc,
            total_cpus,
            total_memory,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout_seconds,
            operation_timeout_seconds,
            format,
            keep_containers,
        } => launch::run_launch(LaunchRequest {
            nproc,
            total_cpus,
            total_memory,
            image,
            build_context,
            rebuild,
            run_id,
            startup_timeout: Duration::from_secs(startup_timeout_seconds),
            operation_timeout: Duration::from_secs(operation_timeout_seconds),
            json: matches!(format, OutputFormat::Json),
            keep_containers,
        }),
        Command::Rank {
            workload,
            pipeline_manifest,
            tensor_manifest,
            benchmark_manifest,
            rank,
            world_size,
            run_id,
            rendezvous_addr,
            rendezvous_bind_addr,
            listen_addr,
            advertise_addr,
            startup_timeout_seconds,
            operation_timeout_seconds,
            expected_cpu_millis,
            expected_memory_bytes,
        } => {
            let rank_request = RankRequest {
                rank,
                world_size,
                run_id,
                rendezvous_addr,
                rendezvous_bind_addr,
                listen_addr,
                advertise_addr,
                startup_timeout: Duration::from_secs(startup_timeout_seconds),
                operation_timeout: Duration::from_secs(operation_timeout_seconds),
                expected_cpu_millis,
                expected_memory_bytes,
            };
            match workload {
                RankWorkloadArg::Topology => launch::run_rank(rank_request),
                RankWorkloadArg::Pipeline => pipeline::run_pipeline_rank_process(
                    rank_request,
                    pipeline_manifest
                        .as_deref()
                        .context("--pipeline-manifest is required for pipeline ranks")?,
                ),
                RankWorkloadArg::Tensor => tensor::run_tensor_rank_process(
                    rank_request,
                    tensor_manifest
                        .as_deref()
                        .context("--tensor-manifest is required for tensor ranks")?,
                ),
                RankWorkloadArg::CollectiveBenchmark => {
                    benchmark::run_collective_benchmark_rank_process(
                        rank_request,
                        benchmark_manifest.as_deref().context(
                            "--benchmark-manifest is required for collective benchmark ranks",
                        )?,
                    )
                }
            }
        }
        Command::P2p { world_size, format } => run_p2p(world_size, format),
        Command::Models { format } => print_models(format),
        Command::Inspect {
            model,
            dtype,
            context_length,
            device_memory_budget,
            format,
            output,
        } => run_inspect(
            InspectionRequest {
                model,
                dtype,
                context_length,
                device_memory_budget,
            },
            format,
            output.as_deref(),
        ),
        Command::Generate {
            model,
            device: DeviceArg::Cpu,
            dtype,
            prompt,
            max_new_tokens,
            device_memory_budget,
            report,
        } => run_generate(
            GenerationRequest {
                model,
                dtype,
                prompt,
                max_new_tokens,
                device_memory_budget,
            },
            report.as_deref(),
        ),
    }
}

fn run_collectives_check(
    world_size: usize,
    format: OutputFormat,
    output: Option<&Path>,
) -> Result<()> {
    if world_size < 2 {
        anyhow::bail!("collective correctness checks require at least two ranks");
    }
    let report = run_collective_check(world_size, P2P_RECEIVE_TIMEOUT)?;
    let rendered = match format {
        OutputFormat::Text => collective_check_text(&report),
        OutputFormat::Json => format!("{}\n", serde_json::to_string_pretty(&report)?),
    };
    if let Some(path) = output {
        fs::write(path, rendered)?;
    } else {
        print!("{rendered}");
    }
    if !report.success {
        anyhow::bail!("native collective correctness check failed");
    }
    Ok(())
}

fn collective_check_text(report: &CollectiveCheckReport) -> String {
    let mut text = format!(
        "NATIVE COLLECTIVE CHECK\nBackend:    {}/{}\nWorld size: {}\n\n",
        report.collective_backend, report.backend, report.world_size
    );
    for rank in &report.ranks {
        text.push_str(&format!("rank {}\n", rank.rank));
        for operation in &rank.operations {
            let algorithm = operation
                .algorithm
                .map(|value| format!("/{value:?}"))
                .unwrap_or_default();
            text.push_str(&format!(
                "  {:?}{}: {}\n",
                operation.kind,
                algorithm,
                if operation.passed { "PASS" } else { "FAIL" }
            ));
        }
    }
    text.push_str(&format!(
        "\nResult: {}\n",
        if report.success { "PASS" } else { "FAIL" }
    ));
    text
}

fn run_p2p(world_size: usize, format: OutputFormat) -> Result<()> {
    let report = run_p2p_ring(world_size, P2P_RECEIVE_TIMEOUT)?;
    match format {
        OutputFormat::Text => print!("{}", p2p_text(&report)),
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }
    if !report.success {
        anyhow::bail!("point-to-point tensor verification failed");
    }
    Ok(())
}

fn p2p_text(report: &P2pReport) -> String {
    let mut text = format!(
        "P2P TENSOR EXCHANGE\n\
         Backend:    {}\n\
         Pattern:    {}\n\
         World size: {}\n",
        report.backend, report.pattern, report.world_size,
    );
    for rank in &report.ranks {
        text.push_str(&format!(
            "\nrank {} sent {} to rank {}\n\
             rank {} received {} from rank {}\n\
             rank {} verification: {}\n",
            rank.rank,
            tensor_values(&rank.sent.values),
            rank.sent_to,
            rank.rank,
            tensor_values(&rank.received.values),
            rank.received_from,
            rank.rank,
            if rank.matches_expected {
                "PASS"
            } else {
                "FAIL"
            },
        ));
    }
    text.push_str(&format!(
        "\nResult: {}\n",
        if report.success { "PASS" } else { "FAIL" }
    ));
    text
}

fn tensor_values(values: &[f32]) -> String {
    let values = values
        .iter()
        .map(|value| {
            if value.fract() == 0.0 {
                format!("{value:.0}")
            } else {
                value.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}

fn print_models(format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Text => {
            println!("SUPPORTED MODELS");
            for spec in supported_models() {
                println!("\n{}", spec.id);
                println!("  repository:       {}", spec.repository);
                println!("  revision:         {}", spec.revision);
                println!("  parameters:       {}", grouped(spec.expected_parameters));
                println!(
                    "  max context:      {}",
                    spec.config.max_position_embeddings
                );
                println!("  CPU/F32:          validated");
                println!("  CUDA:             planned");
                println!("  chat template:    {}", spec.prompt_template.name());
            }
        }
        OutputFormat::Json => {
            let models = supported_models()
                .iter()
                .map(model_json)
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema_version": 1,
                    "models": models,
                }))?
            );
        }
    }
    Ok(())
}

fn model_json(spec: &ModelSpec) -> serde_json::Value {
    json!({
        "id": spec.id,
        "repository": spec.repository,
        "revision": spec.revision,
        "weight_file": spec.weight_file,
        "expected_parameters": spec.expected_parameters,
        "expected_checkpoint_bytes": spec.expected_checkpoint_bytes,
        "checkpoint_dtype": spec.checkpoint_dtype,
        "tensor_layout": spec.tensor_layout,
        "configuration": spec.config,
        "chat_template": spec.prompt_template.name(),
        "execution": {
            "cpu_f32": "validated",
            "cuda": "planned",
        },
    })
}

fn run_inspect(
    request: InspectionRequest,
    format: OutputFormat,
    output: Option<&Path>,
) -> Result<()> {
    let report = inspect(&request)?;
    let rendered = match format {
        OutputFormat::Text => inspection_text(&report),
        OutputFormat::Json => format!("{}\n", serde_json::to_string_pretty(&report)?),
    };
    if let Some(path) = output {
        fs::write(path, rendered)
            .with_context(|| format!("failed to write inspection to {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn inspection_text(report: &InspectionReport) -> String {
    let cfg = &report.config;
    let memory = &report.memory;
    let mut text = format!(
        "MODEL INSPECTION\n\
         Model:              {}\n\
         Repository:         {}\n\
         Revision:           {}\n\
         Parameters:         {}\n\
         Layers:             {}\n\
         Attention heads:    {}\n\
         KV heads:           {}\n\
         Hidden dimension:   {}\n\
         Head dimension:     {}\n\
         Maximum context:    {}\n\
         Planned context:    {}\n\
         Dtype:              {}\n\
         Logical weights:    {} ({})\n\
         KV cache capacity:  {} ({})\n\
         Persistent minimum: {} ({})\n",
        report.model,
        report.repository,
        report.revision,
        grouped(memory.parameter_count),
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
        cfg.hidden_size / cfg.num_attention_heads,
        cfg.max_position_embeddings,
        memory.context_length,
        memory.dtype,
        format_bytes(memory.weight_bytes),
        memory.weight_bytes,
        format_bytes(memory.kv_cache_capacity_bytes),
        memory.kv_cache_capacity_bytes,
        format_bytes(memory.persistent_bytes),
        memory.persistent_bytes,
    );
    if let Some(budget) = memory.budget {
        text.push_str(&format!(
            "Device budget:      {} ({})\n\
             Budget semantics:   host, per-rank, user-declared, not OS-enforced\n",
            format_bytes(budget.bytes),
            budget.bytes,
        ));
    }
    text.push_str(&format!("Placement:          {}\n", memory.placement));
    text.push_str(&format!("Note: {0}\n", report.caveat));
    text
}

fn run_generate(request: GenerationRequest, report_path: Option<&Path>) -> Result<()> {
    let spec = request.model.spec();
    eprintln!(
        "model: {} ({})\nrevision: {}\ncheckpoint: {} (known download size {})",
        request.model,
        spec.repository,
        spec.revision,
        spec.weight_file,
        format_bytes(spec.expected_checkpoint_bytes),
    );
    let mut observer = CliObserver::default();
    let report = generate(&request, &mut observer)?;
    observer.finish(&report.completion)?;

    if let Some(path) = report_path {
        let json = serde_json::to_vec_pretty(&report)?;
        fs::write(path, json)
            .with_context(|| format!("failed to write run report to {}", path.display()))?;
        eprintln!("report: {}", path.display());
    }
    eprint!("{}", generation_summary(&report.memory, &report));
    Ok(())
}

#[derive(Default)]
struct CliObserver {
    artifact_phase: usize,
    streamed: String,
}

impl CliObserver {
    fn finish(&mut self, completion: &str) -> Result<()> {
        if let Some(suffix) = completion.strip_prefix(&self.streamed) {
            print!("{suffix}");
        }
        println!();
        io::stdout().flush()?;
        Ok(())
    }
}

impl EventObserver for CliObserver {
    fn on_event(&mut self, event: &RunEvent) {
        match &event.event {
            RunEventKind::ArtifactResolutionStarted => {
                self.artifact_phase += 1;
                let phase = if self.artifact_phase == 1 {
                    "resolving configuration and tokenizer"
                } else {
                    "resolving checkpoint weights"
                };
                eprintln!("{phase}...");
            }
            RunEventKind::ModelLoadStarted => eprintln!("loading model on CPU..."),
            RunEventKind::PrefillStarted { prompt_tokens } => {
                eprintln!("prefill: {prompt_tokens} prompt tokens");
            }
            RunEventKind::TokenGenerated { text, .. } => {
                print!("{text}");
                let _ = io::stdout().flush();
                self.streamed.push_str(text);
            }
            _ => {}
        }
    }
}

fn generation_summary(memory: &RankMemoryPlan, report: &dlir_runtime::GenerationReport) -> String {
    let timings = &report.timings;
    format!(
        "\nRUN SUMMARY\n\
         prompt tokens:       {}\n\
         generated tokens:    {}\n\
         stop reason:         {}\n\
         logical weights:     {}\n\
         KV capacity:         {} for {} tokens\n\
         final KV used:       {}\n\
         placement:           {}\n\
         artifacts:           {:.3} s\n\
         model load:          {:.3} s\n\
         tokenization:        {:.3} ms\n\
         prefill:             {:.3} ms ({:.2} tok/s)\n\
         TTFT:                {:.3} ms\n\
         decode forwards:     {}\n\
         mean decode latency: {}\n\
         decode throughput:   {}\n\
         generation total:    {:.3} s\n\
         cold start total:    {:.3} s\n",
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.stop_reason,
        format_bytes(memory.weight_bytes),
        format_bytes(memory.kv_cache_capacity_bytes),
        memory.context_length,
        format_bytes(report.final_kv_cache_bytes),
        memory.placement,
        ns_seconds(timings.artifact_resolution_ns),
        ns_seconds(timings.model_load_ns),
        ns_millis(timings.tokenization_ns),
        ns_millis(timings.prefill_ns),
        timings.prefill_tokens_per_second,
        ns_millis(timings.time_to_first_token_ns),
        timings.decode_forward_count,
        timings
            .mean_decode_ns
            .map(|ns| format!("{:.3} ms", ns_millis(ns)))
            .unwrap_or_else(|| "n/a".into()),
        timings
            .decode_tokens_per_second
            .map(|rate| format!("{rate:.2} tok/s"))
            .unwrap_or_else(|| "n/a".into()),
        ns_seconds(timings.generation_total_ns),
        ns_seconds(timings.cold_start_total_ns),
    )
}

fn ns_millis(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn ns_seconds(ns: u64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut result = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            result.push(',');
        }
        result.push(char::from(byte));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parser_rejects_unknown_models_and_ambiguous_budgets() {
        assert!(Cli::try_parse_from(["dlir", "inspect", "--model", "someone/model"]).is_err());
        assert!(
            Cli::try_parse_from([
                "dlir",
                "inspect",
                "--model",
                "smollm2-135m-instruct",
                "--device-memory-budget",
                "500M",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "dlir",
                "generate",
                "--model",
                "smollm2-135m-instruct",
                "--device",
                "cuda",
                "--prompt",
                "hello",
            ])
            .is_err()
        );
    }

    #[test]
    fn pipeline_parser_exposes_only_the_supported_cpu_execution_shape() {
        let parsed = Cli::try_parse_from([
            "dlir",
            "pipeline",
            "--model",
            "smollm2-135m-instruct",
            "--prompt",
            "hello",
            "--nproc",
            "2",
            "--total-cpus",
            "1",
            "--total-memory",
            "512MiB",
            "--report",
            "run.json",
        ]);
        assert!(parsed.is_ok());
        assert!(
            Cli::try_parse_from([
                "dlir",
                "pipeline",
                "--model",
                "smollm2-135m-instruct",
                "--device",
                "cuda",
                "--prompt",
                "hello",
                "--nproc",
                "2",
                "--total-cpus",
                "1",
                "--total-memory",
                "512MiB",
            ])
            .is_err()
        );
    }

    #[test]
    fn pipeline_rank_requires_a_manifest() {
        assert!(
            Cli::try_parse_from([
                "dlir",
                "rank",
                "--workload",
                "pipeline",
                "--rank",
                "0",
                "--world-size",
                "2",
                "--run-id",
                "run",
                "--rendezvous-addr",
                "rank-0:29500",
                "--listen-addr",
                "0.0.0.0:29501",
                "--advertise-addr",
                "rank-0:29501",
            ])
            .is_err()
        );
    }

    #[test]
    fn p2p_text_and_json_are_deterministic_and_versioned() {
        let report = run_p2p_ring(2, Duration::from_millis(250)).unwrap();
        assert_eq!(
            p2p_text(&report),
            "P2P TENSOR EXCHANGE\n\
             Backend:    in_memory\n\
             Pattern:    ring\n\
             World size: 2\n\
             \n\
             rank 0 sent [1, 2, 3, 4] to rank 1\n\
             rank 0 received [5, 6, 7, 8] from rank 1\n\
             rank 0 verification: PASS\n\
             \n\
             rank 1 sent [5, 6, 7, 8] to rank 0\n\
             rank 1 received [1, 2, 3, 4] from rank 0\n\
             rank 1 verification: PASS\n\
             \n\
             Result: PASS\n"
        );
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["backend"], "in_memory");
        assert_eq!(value["pattern"], "ring");
        assert_eq!(
            value["ranks"][0]["sent"]["values"],
            json!([1.0, 2.0, 3.0, 4.0])
        );
        assert_eq!(value["success"], true);
    }

    #[test]
    fn p2p_requires_at_least_two_ranks() {
        assert!(run_p2p_ring(1, Duration::from_millis(10)).is_err());
    }

    #[test]
    fn inspect_text_matches_the_acceptance_baseline() {
        let report = inspect(&InspectionRequest {
            model: SupportedModelId::SmolLm2_135MInstruct,
            dtype: PlanDType::F32,
            context_length: 512,
            device_memory_budget: Some(MemoryBudget::user_declared(500 << 20)),
        })
        .unwrap();
        let text = inspection_text(&report);
        assert!(text.contains("Parameters:         134,515,008"));
        assert!(text.contains("Logical weights:    513.1 MiB (538060032)"));
        assert!(text.contains("KV cache capacity:  22.5 MiB (23592960)"));
        assert!(text.contains("Persistent minimum: 535.6 MiB (561652992)"));
        assert!(text.contains("Placement:          FAILED"));
    }

    #[test]
    fn model_json_is_versionable_and_includes_pin_and_support() {
        let value = model_json(SupportedModelId::TinyLlama1_1BChat.spec());
        assert_eq!(value["id"], "tinyllama-1.1b-chat");
        assert_eq!(
            value["revision"],
            "5243d158d6f4b356f1142ea8fd6a99cb5ac2c0e1"
        );
        assert_eq!(value["execution"]["cpu_f32"], "validated");
        assert_eq!(value["execution"]["cuda"], "planned");
    }

    #[test]
    fn generation_rejects_non_f32_before_resolving_artifacts() {
        let error = generate(
            &GenerationRequest {
                model: SupportedModelId::SmolLm2_135MInstruct,
                dtype: PlanDType::Bf16,
                prompt: "hello".into(),
                max_new_tokens: 1,
                device_memory_budget: None,
            },
            &mut dlir_runtime::NoopObserver,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            dlir_runtime::DlirError::UnsupportedExecution { .. }
        ));
    }

    #[test]
    fn inspect_json_can_be_written_when_placement_fails() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inspection.json");
        run_inspect(
            InspectionRequest {
                model: SupportedModelId::SmolLm2_135MInstruct,
                dtype: PlanDType::F32,
                context_length: 512,
                device_memory_budget: Some(MemoryBudget::user_declared(500 << 20)),
            },
            OutputFormat::Json,
            Some(&path),
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["memory"]["placement"], "does_not_fit");
    }
}
