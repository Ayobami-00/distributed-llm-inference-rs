//! Docker/TCP native all-reduce benchmark launcher and rank workload.

use crate::launch::{
    CpuAmount, DockerResources, PEER_PORT, RENDEZVOUS_PORT, RUN_LABEL, ResourcePlan,
    default_run_id, docker_checked, docker_engine_info, docker_output, ensure_image, format_cpu,
    install_interrupt_handler, observe_cgroup_resources, plan_resources, reset_interrupted,
    validate_run_id, verify_resources,
};
use anyhow::{Result, bail};
use dlir_collectives::{
    AllReduceAlgorithm, CollectiveBenchmarkManifest, CollectiveBenchmarkRankReport,
    CollectiveBenchmarkReport, CollectiveBenchmarkResources, DEFAULT_MAX_TENSOR_BYTES, Rank,
    TcpTransport, TcpTransportConfig, aggregate_all_reduce_benchmark,
    run_all_reduce_benchmark_rank,
};
use dlir_runtime::format_bytes;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const CONTAINER_MANIFEST: &str = "/opt/dlir/request/benchmark.json";

pub(crate) struct CollectiveBenchRequest {
    pub(crate) nproc: usize,
    pub(crate) total_cpus: CpuAmount,
    pub(crate) total_memory: String,
    pub(crate) algorithms: Vec<AllReduceAlgorithm>,
    pub(crate) sizes: String,
    pub(crate) warmup: usize,
    pub(crate) iterations: usize,
    pub(crate) json: bool,
    pub(crate) output: Option<PathBuf>,
    pub(crate) image: String,
    pub(crate) build_context: PathBuf,
    pub(crate) rebuild: bool,
    pub(crate) run_id: Option<String>,
    pub(crate) startup_timeout: Duration,
    pub(crate) operation_timeout: Duration,
    pub(crate) keep_containers: bool,
}

pub(crate) fn run_collective_bench(request: CollectiveBenchRequest) -> Result<()> {
    if !(2..=64).contains(&request.nproc) || request.iterations == 0 {
        bail!("benchmark requires 2..=64 ranks and at least one measured iteration");
    }
    let sizes = request
        .sizes
        .split(',')
        .map(dlir_runtime::parse_byte_size)
        .collect::<dlir_runtime::Result<Vec<_>>>()?;
    if sizes.is_empty() {
        bail!("benchmark requires at least one payload size");
    }
    let total_memory = dlir_runtime::parse_byte_size(&request.total_memory)?;
    let engine = docker_engine_info()?;
    let resources = plan_resources(request.nproc, request.total_cpus, total_memory, &engine)?;
    let run_id = request.run_id.unwrap_or_else(default_run_id);
    validate_run_id(&run_id)?;
    let manifest = CollectiveBenchmarkManifest {
        schema_version: 1,
        run_id: run_id.clone(),
        world_size: request.nproc,
        sizes,
        algorithms: request.algorithms,
        warmup: request.warmup,
        iterations: request.iterations,
    };
    // Validate ring split requirements before creating Docker resources.
    for algorithm in &manifest.algorithms {
        for size in &manifest.sizes {
            let elements = size / 4;
            if *size == 0
                || size % 4 != 0
                || (*algorithm == AllReduceAlgorithm::Ring
                    && elements % manifest.world_size as u64 != 0)
            {
                bail!(
                    "payload {} is invalid for {algorithm:?} with {} ranks",
                    format_bytes(*size),
                    manifest.world_size
                );
            }
        }
    }
    let directory = tempfile::Builder::new()
        .prefix(&format!("dlir-{run_id}-"))
        .tempdir()?;
    let manifest_path = directory.path().join("benchmark.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    let manifest_path = fs::canonicalize(manifest_path)?;

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
    for rank in 0..request.nproc {
        let name = format!("dlir-{run_id}-rank-{rank}");
        eprintln!(
            "starting benchmark rank {rank}: {} CPU, {}",
            format_cpu(resources.per_rank_cpu_millis),
            format_bytes(resources.per_rank_memory_bytes)
        );
        docker_checked(&container_arguments(
            &request.image,
            &docker_resources.network,
            &name,
            &run_id,
            rank,
            request.nproc,
            &resources,
            request.startup_timeout,
            request.operation_timeout,
            &manifest_path,
        ))?;
        docker_resources.containers.push(name);
    }

    let mut ranks = Vec::new();
    let mut failures = Vec::new();
    for (rank, name) in docker_resources.containers.iter().enumerate() {
        let waited = docker_output(&["wait".into(), name.clone()])?;
        let code = String::from_utf8_lossy(&waited.stdout)
            .trim()
            .parse::<i32>()
            .unwrap_or(-1);
        let logs = docker_output(&["logs".into(), name.clone()])?;
        for line in String::from_utf8_lossy(&logs.stderr).lines() {
            eprintln!("[rank {rank}] {line}");
        }
        if code != 0 {
            failures.push(format!("rank {rank} exited with status {code}"));
            continue;
        }
        let result = String::from_utf8_lossy(&logs.stdout)
            .lines()
            .find_map(|line| serde_json::from_str::<CollectiveBenchmarkRankReport>(line).ok());
        if let Some(result) = result {
            ranks.push(result);
        } else {
            failures.push(format!("rank {rank} emitted no benchmark report"));
        }
    }
    if !failures.is_empty() {
        bail!("Docker benchmark failed: {}", failures.join("; "));
    }
    let report =
        aggregate_all_reduce_benchmark(&manifest, ranks, benchmark_resources(&engine, &resources))?;
    let rendered = if request.json {
        format!("{}\n", serde_json::to_string_pretty(&report)?)
    } else {
        benchmark_text(&report)
    };
    if let Some(path) = &request.output {
        fs::write(path, rendered)?;
    } else {
        print!("{rendered}");
    }
    if request.keep_containers {
        let kept = directory.keep();
        eprintln!("retained benchmark manifest at {}", kept.display());
    } else {
        docker_resources.cleanup();
    }
    if !report.success {
        bail!("native TCP all-reduce correctness failed during benchmark");
    }
    Ok(())
}

pub(crate) fn run_collective_benchmark_rank_process(
    rank_request: crate::launch::RankRequest,
    manifest_path: &Path,
) -> Result<()> {
    let manifest: CollectiveBenchmarkManifest = serde_json::from_slice(&fs::read(manifest_path)?)?;
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
    let report = run_all_reduce_benchmark_rank(transport, &manifest)?;
    let final_observed = observe_cgroup_resources();
    verify_resources(
        &final_observed,
        rank_request.expected_cpu_millis,
        rank_request.expected_memory_bytes,
    )?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn container_arguments(
    image: &str,
    network: &str,
    name: &str,
    run_id: &str,
    rank: usize,
    world_size: usize,
    resources: &ResourcePlan,
    startup: Duration,
    operation: Duration,
    manifest: &Path,
) -> Vec<String> {
    let mut arguments = vec![
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
        "--mount".into(),
        format!(
            "type=bind,source={},target={CONTAINER_MANIFEST},readonly",
            manifest.display()
        ),
        image.into(),
        "rank".into(),
        "--workload".into(),
        "collective-benchmark".into(),
        "--benchmark-manifest".into(),
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
    ];
    if rank == 0 {
        arguments.extend([
            "--rendezvous-bind-addr".into(),
            format!("0.0.0.0:{RENDEZVOUS_PORT}"),
        ]);
    }
    arguments
}

fn benchmark_resources(
    engine: &crate::launch::DockerEngineReport,
    resources: &ResourcePlan,
) -> CollectiveBenchmarkResources {
    CollectiveBenchmarkResources {
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

fn benchmark_text(report: &CollectiveBenchmarkReport) -> String {
    let mut text = format!(
        "NATIVE ALL-REDUCE BENCHMARK\nBackend: {}/{}\nWorld size: {}\n\n",
        report.collective_backend, report.backend, report.world_size
    );
    for case in &report.cases {
        text.push_str(&format!(
            "{:?} {:>9}: mean {:.3} ms, p50 {:.3} ms, p95 {:.3} ms, {:.2} MiB/s, wire {}\n",
            case.algorithm,
            format_bytes(case.payload_bytes),
            case.mean_latency_ns as f64 / 1_000_000.,
            case.p50_latency_ns as f64 / 1_000_000.,
            case.p95_latency_ns as f64 / 1_000_000.,
            case.effective_payload_bytes_per_second / 1_048_576.,
            format_bytes(case.observed_wire_bytes),
        ));
    }
    text.push_str(&format!(
        "\nResult: {}\n",
        if report.success { "PASS" } else { "FAIL" }
    ));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_container_selects_tcp_workload_and_enforced_limits() {
        let resources = ResourcePlan {
            requested_cpu_millis: 1000,
            requested_memory_bytes: 512 << 20,
            per_rank_cpu_millis: 500,
            per_rank_memory_bytes: 256 << 20,
            allocated_cpu_millis: 1000,
            allocated_memory_bytes: 512 << 20,
            unused_cpu_millis: 0,
            unused_memory_bytes: 0,
            engine_cpu_headroom_millis: 0,
            engine_memory_headroom_bytes: 0,
        };
        let args = container_arguments(
            "dlir:v0.5-tensor",
            "network",
            "rank-0",
            "run",
            0,
            2,
            &resources,
            Duration::from_secs(30),
            Duration::from_secs(10),
            Path::new("/tmp/benchmark.json"),
        );
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--workload", "collective-benchmark"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--cpus", "0.5"]));
        assert!(args.iter().any(|arg| arg.contains("readonly")));
    }
}
