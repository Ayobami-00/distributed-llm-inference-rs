//! Docker process orchestration and the one-process TCP rank entry point.

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use dlir_collectives::{
    Communicator, DEFAULT_MAX_TENSOR_BYTES, MessageTag, PeerInfo, Rank, RankExchangeReport,
    TcpTransport, TcpTransportConfig, TensorSummary,
};
use dlir_runtime::{format_bytes, parse_byte_size};
use serde::{Deserialize, Serialize};
use std::{
    fmt, fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    str::FromStr,
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MIN_WORLD_SIZE: usize = 2;
const MAX_WORLD_SIZE: usize = 64;
const MIN_CPU_MILLIS: u64 = 100;
const MIN_MEMORY_BYTES: u64 = 128 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
const RUN_LABEL: &str = "io.dlir.run_id";
const RENDEZVOUS_PORT: u16 = 29_500;
const PEER_PORT: u16 = 29_501;
const RING_TAG: MessageTag = MessageTag(0);

static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_HANDLER: OnceLock<()> = OnceLock::new();

/// Positive CPU quantity stored in thousandths of one CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CpuAmount {
    millis: u64,
}

impl CpuAmount {
    fn docker_value(self) -> String {
        format_cpu(self.millis)
    }
}

impl FromStr for CpuAmount {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let value = value.trim();
        if value.is_empty() || value.starts_with(['-', '+']) {
            return Err("CPU total must be a positive decimal".to_owned());
        }
        let mut parts = value.split('.');
        let whole = parts
            .next()
            .expect("split always returns one part")
            .parse::<u64>()
            .map_err(|_| "CPU total must be a positive decimal".to_owned())?;
        let fraction = parts.next().unwrap_or("");
        if parts.next().is_some()
            || fraction.len() > 3
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err("CPU total accepts at most three decimal places".to_owned());
        }
        let fraction = if fraction.is_empty() {
            0
        } else {
            fraction
                .parse::<u64>()
                .map_err(|_| "CPU total must be a positive decimal".to_owned())?
                * 10u64.pow((3 - fraction.len()) as u32)
        };
        let millis = whole
            .checked_mul(1000)
            .and_then(|value| value.checked_add(fraction))
            .ok_or_else(|| "CPU total is too large".to_owned())?;
        if millis == 0 {
            return Err("CPU total must be greater than zero".to_owned());
        }
        Ok(Self { millis })
    }
}

impl fmt::Display for CpuAmount {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.docker_value())
    }
}

pub(crate) struct LaunchRequest {
    pub(crate) nproc: usize,
    pub(crate) total_cpus: CpuAmount,
    pub(crate) total_memory: String,
    pub(crate) image: String,
    pub(crate) build_context: PathBuf,
    pub(crate) rebuild: bool,
    pub(crate) run_id: Option<String>,
    pub(crate) startup_timeout: Duration,
    pub(crate) operation_timeout: Duration,
    pub(crate) json: bool,
    pub(crate) keep_containers: bool,
}

pub(crate) struct RankRequest {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
    pub(crate) run_id: String,
    pub(crate) rendezvous_addr: String,
    pub(crate) rendezvous_bind_addr: Option<String>,
    pub(crate) listen_addr: String,
    pub(crate) advertise_addr: String,
    pub(crate) startup_timeout: Duration,
    pub(crate) operation_timeout: Duration,
    pub(crate) expected_cpu_millis: Option<u64>,
    pub(crate) expected_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct DockerEngineReport {
    server_version: String,
    operating_system: String,
    architecture: String,
    cgroup_version: String,
    cpu_millis: u64,
    memory_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct DockerInfoJson {
    #[serde(rename = "ServerVersion", default)]
    server_version: String,
    #[serde(rename = "OperatingSystem", default)]
    operating_system: String,
    #[serde(rename = "Architecture", default)]
    architecture: String,
    #[serde(rename = "CgroupVersion", default)]
    cgroup_version: String,
    #[serde(rename = "NCPU")]
    ncpu: u64,
    #[serde(rename = "MemTotal")]
    mem_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ResourcePlan {
    requested_cpu_millis: u64,
    requested_memory_bytes: u64,
    per_rank_cpu_millis: u64,
    per_rank_memory_bytes: u64,
    allocated_cpu_millis: u64,
    allocated_memory_bytes: u64,
    unused_cpu_millis: u64,
    unused_memory_bytes: u64,
    engine_cpu_headroom_millis: u64,
    engine_memory_headroom_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ResourceObservation {
    cpu_millis: Option<u64>,
    memory_limit_bytes: Option<u64>,
    memory_current_bytes: Option<u64>,
    cpuset_cpus: Option<String>,
    cpuset_cpu_count: Option<usize>,
    cgroup_version: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ResourceVerification {
    Passed,
    NotEvaluated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RankProcessReport {
    schema_version: u32,
    protocol_version: u16,
    run_id: String,
    rank: usize,
    world_size: usize,
    process_id: u32,
    backend: String,
    peers: Vec<PeerInfo>,
    expected_cpu_millis: Option<u64>,
    expected_memory_bytes: Option<u64>,
    observed_resources: ResourceObservation,
    resource_verification: ResourceVerification,
    startup_barrier: bool,
    exchange: RankExchangeReport,
    completion_barrier: bool,
    success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContainerRankReport {
    rank: usize,
    container_name: String,
    container_id: String,
    exit_code: i32,
    report: Option<RankProcessReport>,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DockerLaunchReport {
    schema_version: u32,
    run_id: String,
    launcher: String,
    backend: String,
    pattern: String,
    world_size: usize,
    image: String,
    engine: DockerEngineReport,
    resources: ResourcePlan,
    ranks: Vec<ContainerRankReport>,
    success: bool,
}

pub(crate) fn run_rank(request: RankRequest) -> Result<()> {
    let rank = Rank::new(request.rank, request.world_size)?;
    let observed_resources = observe_cgroup_resources();
    let resource_verification = verify_resources(
        &observed_resources,
        request.expected_cpu_millis,
        request.expected_memory_bytes,
    )?;
    let transport = TcpTransport::connect(TcpTransportConfig {
        rank,
        run_id: request.run_id.clone(),
        rendezvous_addr: request.rendezvous_addr,
        rendezvous_bind_addr: request.rendezvous_bind_addr,
        listen_addr: request.listen_addr,
        advertise_addr: request.advertise_addr,
        startup_timeout: request.startup_timeout,
        operation_timeout: request.operation_timeout,
        max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
    })?;
    let peers = transport.peers().to_vec();
    let communicator = Communicator::new(transport);
    communicator.barrier()?;

    let sent_to = (request.rank + 1) % request.world_size;
    let received_from = (request.rank + request.world_size - 1) % request.world_size;
    let sent_values = values_for_rank(request.rank);
    let sent = Tensor::from_vec(sent_values.clone(), 4, &Device::Cpu)?;
    communicator.send_tensor(sent_to, RING_TAG, &sent)?;
    let received = communicator.recv_tensor(received_from, RING_TAG)?;
    let received_values = received.to_vec1::<f32>()?;
    let expected = values_for_rank(received_from);
    let matches_expected = received.dims() == [4] && received_values == expected;
    communicator.barrier()?;

    let exchange = RankExchangeReport {
        rank: request.rank,
        sent_to,
        received_from,
        sent: TensorSummary {
            dtype: "f32".to_owned(),
            shape: vec![4],
            values: sent_values,
        },
        received: TensorSummary {
            dtype: "f32".to_owned(),
            shape: received.dims().to_vec(),
            values: received_values,
        },
        matches_expected,
    };
    let report = RankProcessReport {
        schema_version: 1,
        protocol_version: dlir_collectives::PROTOCOL_VERSION,
        run_id: request.run_id,
        rank: request.rank,
        world_size: request.world_size,
        process_id: std::process::id(),
        backend: "tcp".to_owned(),
        peers,
        expected_cpu_millis: request.expected_cpu_millis,
        expected_memory_bytes: request.expected_memory_bytes,
        observed_resources,
        resource_verification,
        startup_barrier: true,
        exchange,
        completion_barrier: true,
        success: matches_expected,
    };
    println!("{}", serde_json::to_string(&report)?);
    if !report.success {
        bail!("rank {} tensor verification failed", report.rank);
    }
    Ok(())
}

pub(crate) fn run_launch(request: LaunchRequest) -> Result<()> {
    validate_world_size(request.nproc)?;
    if request.startup_timeout.is_zero() || request.operation_timeout.is_zero() {
        bail!("startup and operation timeouts must be greater than zero");
    }
    let total_memory = parse_byte_size(&request.total_memory)?;
    let engine = docker_engine_info()?;
    let resources = plan_resources(request.nproc, request.total_cpus, total_memory, &engine)?;
    let run_id = request.run_id.unwrap_or_else(default_run_id);
    validate_run_id(&run_id)?;
    install_interrupt_handler()?;
    INTERRUPTED.store(false, Ordering::SeqCst);

    ensure_image(&request.image, &request.build_context, request.rebuild)?;
    let network = format!("dlir-{run_id}");
    eprintln!("creating Docker network {network}...");
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

    let mut identities = Vec::with_capacity(request.nproc);
    for rank in 0..request.nproc {
        let name = format!("dlir-{run_id}-rank-{rank}");
        eprintln!(
            "starting rank {rank}: {} CPU, {}...",
            format_cpu(resources.per_rank_cpu_millis),
            format_bytes(resources.per_rank_memory_bytes)
        );
        let arguments = container_arguments(
            &request.image,
            &docker_resources.network,
            &name,
            &run_id,
            rank,
            request.nproc,
            &resources,
            request.startup_timeout,
            request.operation_timeout,
        );
        let output = docker_checked(&arguments)?;
        let container_id = String::from_utf8(output.stdout)
            .context("Docker returned a non-UTF-8 container ID")?
            .trim()
            .to_owned();
        docker_resources.containers.push(name.clone());
        identities.push((rank, name, container_id));
    }

    let exit_codes = wait_for_containers(&identities, &docker_resources.containers)?;
    let mut ranks = Vec::with_capacity(request.nproc);
    for (rank, name, container_id) in identities {
        let logs = docker_output(&["logs".to_owned(), name.clone()])?;
        let stderr = String::from_utf8_lossy(&logs.stderr).trim().to_owned();
        if !stderr.is_empty() {
            for line in stderr.lines() {
                eprintln!("[rank {rank}] {line}");
            }
        }
        let stdout = String::from_utf8_lossy(&logs.stdout).trim().to_owned();
        let report = if stdout.is_empty() {
            None
        } else {
            serde_json::from_str::<RankProcessReport>(&stdout).ok()
        };
        let exit_code = *exit_codes.get(rank).unwrap_or(&-1);
        let error = if exit_code == 0 && report.as_ref().is_some_and(|report| report.success) {
            None
        } else if !stderr.is_empty() {
            Some(stderr)
        } else if !stdout.is_empty() && report.is_none() {
            Some(format!("rank emitted invalid JSON: {stdout}"))
        } else {
            Some(format!("container exited with code {exit_code}"))
        };
        ranks.push(ContainerRankReport {
            rank,
            container_name: name,
            container_id,
            exit_code,
            report,
            error,
        });
    }
    ranks.sort_by_key(|rank| rank.rank);
    let success = ranks.iter().all(|rank| {
        rank.exit_code == 0
            && rank.error.is_none()
            && rank.report.as_ref().is_some_and(|report| {
                report.success
                    && report.resource_verification == ResourceVerification::Passed
                    && report.peers.len() == request.nproc
            })
    });
    let report = DockerLaunchReport {
        schema_version: 1,
        run_id,
        launcher: "docker_cli".to_owned(),
        backend: "tcp".to_owned(),
        pattern: "ring".to_owned(),
        world_size: request.nproc,
        image: request.image,
        engine,
        resources,
        ranks,
        success,
    };
    if request.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", launch_text(&report));
    }
    if docker_resources.keep {
        eprintln!(
            "retained Docker network {} and {} rank containers",
            docker_resources.network,
            docker_resources.containers.len()
        );
    } else {
        docker_resources.cleanup();
    }
    if !success {
        bail!("Docker TCP topology verification failed");
    }
    Ok(())
}

fn validate_world_size(world_size: usize) -> Result<()> {
    if !(MIN_WORLD_SIZE..=MAX_WORLD_SIZE).contains(&world_size) {
        bail!("nproc must be between {MIN_WORLD_SIZE} and {MAX_WORLD_SIZE}, got {world_size}");
    }
    Ok(())
}

fn plan_resources(
    world_size: usize,
    total_cpus: CpuAmount,
    total_memory: u64,
    engine: &DockerEngineReport,
) -> Result<ResourcePlan> {
    validate_world_size(world_size)?;
    if total_cpus.millis > engine.cpu_millis {
        bail!(
            "requested {} CPUs but Docker Engine exposes {}",
            total_cpus,
            format_cpu(engine.cpu_millis)
        );
    }
    if total_memory > engine.memory_bytes {
        bail!(
            "requested {} memory but Docker Engine exposes {}",
            format_bytes(total_memory),
            format_bytes(engine.memory_bytes)
        );
    }
    let world = world_size as u64;
    let per_rank_cpu_millis = total_cpus.millis / world;
    let per_rank_memory_bytes = (total_memory / world / MIB) * MIB;
    if per_rank_cpu_millis < MIN_CPU_MILLIS {
        bail!(
            "each rank requires at least {} CPU after division",
            format_cpu(MIN_CPU_MILLIS)
        );
    }
    if per_rank_memory_bytes < MIN_MEMORY_BYTES {
        bail!(
            "each rank requires at least {} after division",
            format_bytes(MIN_MEMORY_BYTES)
        );
    }
    let allocated_cpu_millis = per_rank_cpu_millis * world;
    let allocated_memory_bytes = per_rank_memory_bytes * world;
    Ok(ResourcePlan {
        requested_cpu_millis: total_cpus.millis,
        requested_memory_bytes: total_memory,
        per_rank_cpu_millis,
        per_rank_memory_bytes,
        allocated_cpu_millis,
        allocated_memory_bytes,
        unused_cpu_millis: total_cpus.millis - allocated_cpu_millis,
        unused_memory_bytes: total_memory - allocated_memory_bytes,
        engine_cpu_headroom_millis: engine.cpu_millis - allocated_cpu_millis,
        engine_memory_headroom_bytes: engine.memory_bytes - allocated_memory_bytes,
    })
}

fn docker_engine_info() -> Result<DockerEngineReport> {
    let output = docker_checked(&[
        "info".to_owned(),
        "--format".to_owned(),
        "{{json .}}".to_owned(),
    ])?;
    let info: DockerInfoJson =
        serde_json::from_slice(&output.stdout).context("could not parse `docker info` JSON")?;
    if info.ncpu == 0 || info.mem_total == 0 {
        bail!("Docker Engine reported zero CPU or memory capacity");
    }
    Ok(DockerEngineReport {
        server_version: info.server_version,
        operating_system: info.operating_system,
        architecture: info.architecture,
        cgroup_version: info.cgroup_version,
        cpu_millis: info.ncpu.saturating_mul(1000),
        memory_bytes: info.mem_total,
    })
}

fn ensure_image(image: &str, context: &Path, rebuild: bool) -> Result<()> {
    if image.trim().is_empty() {
        bail!("Docker image name cannot be empty");
    }
    let present = docker_output(&["image".to_owned(), "inspect".to_owned(), image.to_owned()])?
        .status
        .success();
    if present && !rebuild {
        eprintln!("reusing Docker image {image}");
        return Ok(());
    }
    if !context.join("Dockerfile").is_file() {
        bail!(
            "build context {} does not contain Dockerfile",
            context.display()
        );
    }
    eprintln!("building Docker image {image}...");
    let mut arguments = vec!["build".to_owned(), "--tag".to_owned(), image.to_owned()];
    if rebuild {
        arguments.push("--no-cache".to_owned());
    }
    arguments.push(context.display().to_string());
    let output = docker_output(&arguments)?;
    if !output.stdout.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stdout));
    }
    if !output.stderr.is_empty() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if !output.status.success() {
        bail!("Docker image build failed with status {}", output.status);
    }
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
    startup_timeout: Duration,
    operation_timeout: Duration,
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
        image.to_owned(),
        "rank".to_owned(),
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
    ];
    if rank == 0 {
        arguments.push("--rendezvous-bind-addr".to_owned());
        arguments.push(format!("0.0.0.0:{RENDEZVOUS_PORT}"));
    }
    arguments
}

fn wait_for_containers(
    identities: &[(usize, String, String)],
    all_names: &[String],
) -> Result<Vec<i32>> {
    let (sender, receiver) = mpsc::channel();
    for (rank, name, _) in identities {
        let sender = sender.clone();
        let name = name.clone();
        let rank = *rank;
        thread::spawn(move || {
            let result = docker_output(&["wait".to_owned(), name]);
            let _ = sender.send((rank, result));
        });
    }
    drop(sender);
    let mut exit_codes = vec![-1; identities.len()];
    let mut remaining = identities.len();
    let mut stopped = false;
    while remaining > 0 {
        if INTERRUPTED.load(Ordering::SeqCst) && !stopped {
            stop_containers(all_names);
            stopped = true;
        }
        match receiver.recv_timeout(Duration::from_millis(100)) {
            Ok((rank, output)) => {
                remaining -= 1;
                let output = output?;
                let code = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<i32>()
                    .unwrap_or(-1);
                exit_codes[rank] = code;
                if code != 0 && !stopped {
                    stop_containers(all_names);
                    stopped = true;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if INTERRUPTED.load(Ordering::SeqCst) {
        bail!("Docker launch interrupted");
    }
    Ok(exit_codes)
}

fn stop_containers(names: &[String]) {
    for name in names {
        let _ = docker_output(&["kill".to_owned(), name.clone()]);
    }
}

fn docker_checked(arguments: &[String]) -> Result<Output> {
    let output = docker_output(arguments)?;
    if output.status.success() {
        Ok(output)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        bail!(
            "docker {} failed with status {}{}",
            arguments.join(" "),
            output.status,
            if stderr.is_empty() {
                String::new()
            } else {
                format!(": {stderr}")
            }
        );
    }
}

fn docker_output(arguments: &[String]) -> Result<Output> {
    Command::new("docker")
        .args(arguments)
        .output()
        .with_context(|| format!("could not execute `docker {}`", arguments.join(" ")))
}

struct DockerResources {
    network: String,
    containers: Vec<String>,
    keep: bool,
}

impl DockerResources {
    fn cleanup(&mut self) {
        if self.keep {
            return;
        }
        for name in self.containers.drain(..) {
            let _ = docker_output(&["rm".to_owned(), "--force".to_owned(), name]);
        }
        if !self.network.is_empty() {
            let network = std::mem::take(&mut self.network);
            let _ = docker_output(&["network".to_owned(), "rm".to_owned(), network]);
        }
    }
}

impl Drop for DockerResources {
    fn drop(&mut self) {
        if self.keep {
            stop_containers(&self.containers);
        } else {
            self.cleanup();
        }
    }
}

fn install_interrupt_handler() -> Result<()> {
    if INTERRUPT_HANDLER.get().is_none() {
        ctrlc::set_handler(|| INTERRUPTED.store(true, Ordering::SeqCst))
            .context("could not install interrupt handler")?;
        let _ = INTERRUPT_HANDLER.set(());
    }
    Ok(())
}

fn observe_cgroup_resources() -> ResourceObservation {
    let v2_cpu = read_optional("/sys/fs/cgroup/cpu.max");
    let v2_memory = read_optional("/sys/fs/cgroup/memory.max");
    if v2_cpu.is_some() || v2_memory.is_some() {
        let cpuset = read_optional("/sys/fs/cgroup/cpuset.cpus.effective")
            .or_else(|| read_optional("/sys/fs/cgroup/cpuset.cpus"));
        return ResourceObservation {
            cpu_millis: v2_cpu.as_deref().and_then(parse_cpu_max),
            memory_limit_bytes: v2_memory.as_deref().and_then(parse_memory_limit),
            memory_current_bytes: read_optional("/sys/fs/cgroup/memory.current")
                .as_deref()
                .and_then(parse_memory_limit),
            cpuset_cpu_count: cpuset.as_deref().and_then(parse_cpuset_count),
            cpuset_cpus: cpuset,
            cgroup_version: Some("v2".to_owned()),
        };
    }

    let quota = read_optional("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    let period = read_optional("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
    let memory = read_optional("/sys/fs/cgroup/memory/memory.limit_in_bytes");
    let cpuset = read_optional("/sys/fs/cgroup/cpuset/cpuset.cpus");
    let present = quota.is_some() || memory.is_some() || cpuset.is_some();
    ResourceObservation {
        cpu_millis: quota
            .as_deref()
            .zip(period.as_deref())
            .and_then(|(quota, period)| parse_cpu_v1(quota, period)),
        memory_limit_bytes: memory.as_deref().and_then(parse_memory_limit),
        memory_current_bytes: read_optional("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            .as_deref()
            .and_then(parse_memory_limit),
        cpuset_cpu_count: cpuset.as_deref().and_then(parse_cpuset_count),
        cpuset_cpus: cpuset,
        cgroup_version: present.then(|| "v1".to_owned()),
    }
}

fn verify_resources(
    observed: &ResourceObservation,
    expected_cpu_millis: Option<u64>,
    expected_memory_bytes: Option<u64>,
) -> Result<ResourceVerification> {
    match (expected_cpu_millis, expected_memory_bytes) {
        (None, None) => Ok(ResourceVerification::NotEvaluated),
        (Some(cpu), Some(memory)) => {
            let actual_cpu = observed
                .cpu_millis
                .context("container cgroup does not expose an enforced CPU quota")?;
            let actual_memory = observed
                .memory_limit_bytes
                .context("container cgroup does not expose an enforced memory maximum")?;
            if actual_cpu.abs_diff(cpu) > 1 {
                bail!(
                    "cgroup CPU quota is {} but launcher requested {}",
                    format_cpu(actual_cpu),
                    format_cpu(cpu)
                );
            }
            if actual_memory != memory {
                bail!(
                    "cgroup memory maximum is {} but launcher requested {}",
                    format_bytes(actual_memory),
                    format_bytes(memory)
                );
            }
            Ok(ResourceVerification::Passed)
        }
        _ => bail!("expected CPU and memory limits must be supplied together"),
    }
}

fn read_optional(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
}

fn parse_cpu_max(value: &str) -> Option<u64> {
    let mut fields = value.split_whitespace();
    let quota = fields.next()?;
    let period = fields.next()?.parse::<u64>().ok()?;
    if quota == "max" || period == 0 || fields.next().is_some() {
        return None;
    }
    quota
        .parse::<u64>()
        .ok()?
        .checked_mul(1000)?
        .checked_div(period)
}

fn parse_cpu_v1(quota: &str, period: &str) -> Option<u64> {
    let quota = quota.trim().parse::<i64>().ok()?;
    let period = period.trim().parse::<u64>().ok()?;
    if quota < 0 || period == 0 {
        return None;
    }
    (quota as u64).checked_mul(1000)?.checked_div(period)
}

fn parse_memory_limit(value: &str) -> Option<u64> {
    let value = value.trim();
    if value == "max" {
        None
    } else {
        value.parse().ok()
    }
}

fn parse_cpuset_count(value: &str) -> Option<usize> {
    if value.trim().is_empty() {
        return None;
    }
    value.split(',').try_fold(0usize, |count, group| {
        let mut bounds = group.trim().split('-');
        let start = bounds.next()?.parse::<usize>().ok()?;
        let end = bounds
            .next()
            .map_or(Some(start), |value| value.parse().ok())?;
        if bounds.next().is_some() || end < start {
            return None;
        }
        count.checked_add(end - start + 1)
    })
}

fn values_for_rank(rank: usize) -> Vec<f32> {
    let base = rank * 4 + 1;
    (base..base + 4).map(|value| value as f32).collect()
}

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty()
        || run_id.len() > 48
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("run ID must contain 1-48 ASCII letters, digits, '-' or '_'");
    }
    Ok(())
}

fn default_run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{nanos}", std::process::id())
}

fn format_cpu(millis: u64) -> String {
    let whole = millis / 1000;
    let fraction = millis % 1000;
    if fraction == 0 {
        whole.to_string()
    } else {
        format!("{whole}.{fraction:03}")
            .trim_end_matches('0')
            .to_owned()
    }
}

fn launch_text(report: &DockerLaunchReport) -> String {
    let mut text = format!(
        "TCP DOCKER TOPOLOGY\n\
         Run ID:         {}\n\
         Backend:        {}\n\
         Pattern:        {}\n\
         World size:     {}\n\
         Docker engine:  {} CPU, {}\n\
         Requested:      {} CPU, {}\n\
         Per rank:       {} CPU, {}\n",
        report.run_id,
        report.backend,
        report.pattern,
        report.world_size,
        format_cpu(report.engine.cpu_millis),
        format_bytes(report.engine.memory_bytes),
        format_cpu(report.resources.requested_cpu_millis),
        format_bytes(report.resources.requested_memory_bytes),
        format_cpu(report.resources.per_rank_cpu_millis),
        format_bytes(report.resources.per_rank_memory_bytes),
    );
    for rank in &report.ranks {
        if let Some(rank_report) = &rank.report {
            text.push_str(&format!(
                "\nrank {} peers: {}\n\
                 rank {} resources: {}\n\
                 rank {} sent {:?} to rank {}\n\
                 rank {} received {:?} from rank {}\n\
                 rank {} barriers: {}\n\
                 rank {} verification: {}\n",
                rank.rank,
                rank_report.peers.len(),
                rank.rank,
                if rank_report.resource_verification == ResourceVerification::Passed {
                    "PASS"
                } else {
                    "NOT EVALUATED"
                },
                rank.rank,
                rank_report.exchange.sent.values,
                rank_report.exchange.sent_to,
                rank.rank,
                rank_report.exchange.received.values,
                rank_report.exchange.received_from,
                rank.rank,
                if rank_report.startup_barrier && rank_report.completion_barrier {
                    "PASS"
                } else {
                    "FAIL"
                },
                rank.rank,
                if rank_report.success { "PASS" } else { "FAIL" },
            ));
        } else {
            text.push_str(&format!(
                "\nrank {}: FAIL ({})\n",
                rank.rank,
                rank.error.as_deref().unwrap_or("missing rank report")
            ));
        }
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

    fn engine() -> DockerEngineReport {
        DockerEngineReport {
            server_version: "test".to_owned(),
            operating_system: "linux".to_owned(),
            architecture: "x86_64".to_owned(),
            cgroup_version: "2".to_owned(),
            cpu_millis: 8_000,
            memory_bytes: 8 << 30,
        }
    }

    #[test]
    fn cpu_parser_accepts_millicpus_and_rejects_ambiguous_values() {
        assert_eq!("2".parse::<CpuAmount>().unwrap().millis, 2_000);
        assert_eq!("0.125".parse::<CpuAmount>().unwrap().millis, 125);
        assert_eq!("1.5".parse::<CpuAmount>().unwrap().millis, 1_500);
        assert!("0".parse::<CpuAmount>().is_err());
        assert!("1.0001".parse::<CpuAmount>().is_err());
        assert!("-1".parse::<CpuAmount>().is_err());
    }

    #[test]
    fn resource_plan_divides_equally_and_retains_remainders() {
        let plan = plan_resources(3, "2".parse().unwrap(), 1 << 30, &engine()).unwrap();
        assert_eq!(plan.per_rank_cpu_millis, 666);
        assert_eq!(plan.unused_cpu_millis, 2);
        assert_eq!(plan.per_rank_memory_bytes, 341 * MIB);
        assert_eq!(plan.unused_memory_bytes, MIB);
    }

    #[test]
    fn resource_plan_checks_engine_capacity_and_per_rank_minimums() {
        assert!(plan_resources(2, "9".parse().unwrap(), 1 << 30, &engine()).is_err());
        assert!(plan_resources(2, "1".parse().unwrap(), 9 << 30, &engine()).is_err());
        assert!(plan_resources(4, "0.3".parse().unwrap(), 1 << 30, &engine()).is_err());
        assert!(plan_resources(4, "1".parse().unwrap(), 256 * MIB, &engine()).is_err());
    }

    #[test]
    fn cgroup_parsers_cover_v1_v2_and_cpuset_ranges() {
        assert_eq!(parse_cpu_max("50000 100000"), Some(500));
        assert_eq!(parse_cpu_max("max 100000"), None);
        assert_eq!(parse_cpu_v1("25000", "100000"), Some(250));
        assert_eq!(parse_cpu_v1("-1", "100000"), None);
        assert_eq!(parse_memory_limit("536870912"), Some(536_870_912));
        assert_eq!(parse_memory_limit("max"), None);
        assert_eq!(parse_cpuset_count("0-3,8,10-11"), Some(7));
    }

    #[test]
    fn strict_resource_verification_detects_mismatches() {
        let observed = ResourceObservation {
            cpu_millis: Some(500),
            memory_limit_bytes: Some(256 * MIB),
            memory_current_bytes: Some(1),
            cpuset_cpus: Some("0-3".to_owned()),
            cpuset_cpu_count: Some(4),
            cgroup_version: Some("v2".to_owned()),
        };
        assert_eq!(
            verify_resources(&observed, Some(500), Some(256 * MIB)).unwrap(),
            ResourceVerification::Passed
        );
        assert!(verify_resources(&observed, Some(600), Some(256 * MIB)).is_err());
        assert!(verify_resources(&observed, Some(500), Some(512 * MIB)).is_err());
    }

    #[test]
    fn container_command_has_scoped_network_limits_and_rank_identity() {
        let plan = plan_resources(2, "1".parse().unwrap(), 512 * MIB, &engine()).unwrap();
        let arguments = container_arguments(
            "dlir:test",
            "dlir-run",
            "dlir-run-rank-0",
            "run",
            0,
            2,
            &plan,
            Duration::from_secs(30),
            Duration::from_secs(10),
        );
        let joined = arguments.join(" ");
        assert!(joined.contains("--cpus 0.5"));
        assert!(joined.contains("--memory 268435456b"));
        assert!(joined.contains("--memory-swap 268435456b"));
        assert!(joined.contains("--network dlir-run"));
        assert!(joined.contains("rank --rank 0 --world-size 2"));
        assert!(joined.contains("--rendezvous-bind-addr 0.0.0.0:29500"));
    }
}
