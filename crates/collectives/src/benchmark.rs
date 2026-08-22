//! Reproducible native all-reduce benchmark reports.

use crate::{
    AllReduceAlgorithm, BarrierTransport, CollectiveCommunicator, CollectiveTrace, Communicator,
    NativeCollectives, ReduceOp, Result, Transport, run_in_memory,
};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Immutable workload shared by Docker/TCP benchmark ranks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkManifest {
    /// Manifest schema; currently `1`.
    pub schema_version: u32,
    /// Docker/TCP run identity.
    pub run_id: String,
    /// Expected rank count.
    pub world_size: usize,
    /// Per-rank logical F32 payload sizes.
    pub sizes: Vec<u64>,
    /// Native algorithms in execution order.
    pub algorithms: Vec<AllReduceAlgorithm>,
    /// Discarded iterations per case.
    pub warmup: usize,
    /// Measured iterations per case.
    pub iterations: usize,
}

/// One rank's measured values for one benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkRankCase {
    /// Algorithm under test.
    pub algorithm: AllReduceAlgorithm,
    /// Per-rank input bytes.
    pub payload_bytes: u64,
    /// Rank-local latency for each synchronized measured iteration.
    pub latencies_ns: Vec<u64>,
    /// Logical bytes sent by this rank across measured iterations.
    pub sent_bytes: u64,
    /// Complete measured collective traces.
    pub traces: Vec<CollectiveTrace>,
}

/// Final benchmark result emitted by one TCP rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkRankReport {
    /// Report schema; currently `1`.
    pub schema_version: u32,
    /// Global rank.
    pub rank: usize,
    /// Cases in manifest order.
    pub cases: Vec<CollectiveBenchmarkRankCase>,
    /// Whether every reduced value matched the deterministic sum.
    pub success: bool,
}

/// Docker capacity and equal limits used for a TCP benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkResources {
    /// Docker Engine CPU capacity in millicpus.
    pub engine_cpu_millis: u64,
    /// Docker Engine memory capacity.
    pub engine_memory_bytes: u64,
    /// Requested CPU total.
    pub requested_cpu_millis: u64,
    /// Requested memory total.
    pub requested_memory_bytes: u64,
    /// Equal rank CPU quota.
    pub per_rank_cpu_millis: u64,
    /// Equal rank memory maximum.
    pub per_rank_memory_bytes: u64,
    /// Unallocated CPU remainder.
    pub unused_cpu_millis: u64,
    /// Unallocated memory remainder.
    pub unused_memory_bytes: u64,
}

/// Metrics for one tensor size and all-reduce algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkCase {
    /// Algorithm under test.
    pub algorithm: AllReduceAlgorithm,
    /// Per-rank logical input bytes.
    pub payload_bytes: u64,
    /// Number of measured synchronized iterations.
    pub iterations: usize,
    /// Mean of maximum-rank iteration latency.
    pub mean_latency_ns: u64,
    /// Median maximum-rank latency.
    pub p50_latency_ns: u64,
    /// 95th percentile maximum-rank latency.
    pub p95_latency_ns: u64,
    /// Payload divided by mean maximum-rank latency.
    pub effective_payload_bytes_per_second: f64,
    /// Logical payload bytes sent across all ranks and measured iterations.
    pub observed_wire_bytes: u64,
}

/// Schema-v1 native all-reduce benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveBenchmarkReport {
    /// Schema version; currently `1`.
    pub schema_version: u32,
    /// Transport used for this runner.
    pub backend: String,
    /// Native collective implementation identity.
    pub collective_backend: String,
    /// Participant count.
    pub world_size: usize,
    /// Discarded iterations per case.
    pub warmup: usize,
    /// Rank-synchronized measured iterations per case.
    pub iterations: usize,
    /// Cases in requested algorithm/size order.
    pub cases: Vec<CollectiveBenchmarkCase>,
    /// Docker resource allocation for TCP runs; absent for the offline runner.
    pub resources: Option<CollectiveBenchmarkResources>,
    /// Rank-local measurements for auditability; empty for the offline runner.
    pub ranks: Vec<CollectiveBenchmarkRankReport>,
    /// Overall completion verdict.
    pub success: bool,
}

/// Benchmarks native all-reduce over in-memory ranks.
///
/// Docker/TCP rank processes use the same [`NativeCollectives`] implementation; this local runner
/// is deterministic and useful for offline CI without requiring a Docker daemon.
pub fn run_in_memory_all_reduce_benchmark(
    world_size: usize,
    sizes: &[u64],
    algorithms: &[AllReduceAlgorithm],
    warmup: usize,
    iterations: usize,
    timeout: Duration,
) -> Result<CollectiveBenchmarkReport> {
    let mut cases = Vec::new();
    for &algorithm in algorithms {
        for &payload_bytes in sizes {
            let elements = usize::try_from(payload_bytes / 4).map_err(|_| {
                crate::CollectivesError::Collective("benchmark size is too large".into())
            })?;
            if payload_bytes == 0
                || payload_bytes % 4 != 0
                || (algorithm == AllReduceAlgorithm::Ring && elements % world_size != 0)
            {
                return Err(crate::CollectivesError::Collective(format!(
                    "benchmark size {payload_bytes} is invalid for {algorithm:?} with {world_size} ranks"
                )));
            }
            let ranks = run_in_memory(world_size, timeout, move |communicator| {
                let rank = communicator.rank().global_rank();
                let input =
                    Tensor::from_vec(vec![rank as f32 + 1.; elements], elements, &Device::Cpu)?;
                let mut native = NativeCollectives::new(communicator);
                for _ in 0..warmup {
                    native.barrier()?;
                    native.all_reduce(&input, ReduceOp::Sum, algorithm)?;
                }
                native.take_traces();
                let mut latencies = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    native.barrier()?;
                    let started = Instant::now();
                    native.all_reduce(&input, ReduceOp::Sum, algorithm)?;
                    latencies.push(ns(started.elapsed()));
                }
                let sent = native
                    .traces()
                    .iter()
                    .map(|trace| trace.sent_bytes)
                    .sum::<u64>();
                Ok((latencies, sent))
            })?;
            let mut maximums = (0..iterations)
                .map(|iteration| {
                    ranks
                        .iter()
                        .map(|rank| rank.0[iteration])
                        .max()
                        .unwrap_or(0)
                })
                .collect::<Vec<_>>();
            maximums.sort_unstable();
            let mean = maximums.iter().sum::<u64>() / iterations.max(1) as u64;
            cases.push(CollectiveBenchmarkCase {
                algorithm,
                payload_bytes,
                iterations,
                mean_latency_ns: mean,
                p50_latency_ns: percentile(&maximums, 50),
                p95_latency_ns: percentile(&maximums, 95),
                effective_payload_bytes_per_second: if mean == 0 {
                    0.
                } else {
                    payload_bytes as f64 * 1e9 / mean as f64
                },
                observed_wire_bytes: ranks.iter().map(|rank| rank.1).sum(),
            });
        }
    }
    Ok(CollectiveBenchmarkReport {
        schema_version: 1,
        backend: "in_memory".into(),
        collective_backend: "native".into(),
        world_size,
        warmup,
        iterations,
        cases,
        resources: None,
        ranks: Vec::new(),
        success: true,
    })
}

/// Executes a synchronized benchmark workload on one established transport rank.
pub fn run_all_reduce_benchmark_rank<T>(
    transport: T,
    manifest: &CollectiveBenchmarkManifest,
) -> Result<CollectiveBenchmarkRankReport>
where
    T: Transport + BarrierTransport,
{
    let rank = transport.rank();
    if manifest.schema_version != 1
        || manifest.world_size != rank.world_size()
        || manifest.iterations == 0
    {
        return Err(crate::CollectivesError::Collective(
            "benchmark manifest and transport topology disagree".into(),
        ));
    }
    let mut native = NativeCollectives::new(Communicator::new(transport));
    let expected = (rank.world_size() * (rank.world_size() + 1) / 2) as f32;
    let mut cases = Vec::new();
    let mut success = true;
    for &algorithm in &manifest.algorithms {
        for &payload_bytes in &manifest.sizes {
            let elements = usize::try_from(payload_bytes / 4).map_err(|_| {
                crate::CollectivesError::Collective("benchmark size is too large".into())
            })?;
            if payload_bytes == 0
                || payload_bytes % 4 != 0
                || (algorithm == AllReduceAlgorithm::Ring && elements % rank.world_size() != 0)
            {
                return Err(crate::CollectivesError::Collective(format!(
                    "benchmark size {payload_bytes} is invalid for {algorithm:?} with {} ranks",
                    rank.world_size()
                )));
            }
            let input = Tensor::from_vec(
                vec![rank.global_rank() as f32 + 1.; elements],
                elements,
                &Device::Cpu,
            )?;
            for _ in 0..manifest.warmup {
                native.barrier()?;
                native.all_reduce(&input, ReduceOp::Sum, algorithm)?;
            }
            native.take_traces();
            let mut latencies_ns = Vec::with_capacity(manifest.iterations);
            for _ in 0..manifest.iterations {
                native.barrier()?;
                let started = Instant::now();
                let output = native.all_reduce(&input, ReduceOp::Sum, algorithm)?;
                latencies_ns.push(ns(started.elapsed()));
                let values = output.flatten_all()?.to_vec1::<f32>()?;
                success &= values.iter().all(|value| *value == expected);
            }
            let traces = native.take_traces();
            let sent_bytes = traces.iter().map(|trace| trace.sent_bytes).sum();
            cases.push(CollectiveBenchmarkRankCase {
                algorithm,
                payload_bytes,
                latencies_ns,
                sent_bytes,
                traces,
            });
        }
    }
    native.barrier()?;
    Ok(CollectiveBenchmarkRankReport {
        schema_version: 1,
        rank: rank.global_rank(),
        cases,
        success,
    })
}

/// Aggregates rank-local TCP samples using maximum-rank latency per iteration.
pub fn aggregate_all_reduce_benchmark(
    manifest: &CollectiveBenchmarkManifest,
    mut ranks: Vec<CollectiveBenchmarkRankReport>,
    resources: CollectiveBenchmarkResources,
) -> Result<CollectiveBenchmarkReport> {
    ranks.sort_by_key(|rank| rank.rank);
    if ranks.len() != manifest.world_size
        || ranks
            .iter()
            .enumerate()
            .any(|(rank, report)| report.rank != rank)
    {
        return Err(crate::CollectivesError::Collective(
            "benchmark rank reports are incomplete or unordered".into(),
        ));
    }
    let case_count = manifest.algorithms.len() * manifest.sizes.len();
    if ranks.iter().any(|rank| rank.cases.len() != case_count) {
        return Err(crate::CollectivesError::Collective(
            "benchmark ranks emitted different case counts".into(),
        ));
    }
    if ranks.iter().any(|rank| {
        rank.schema_version != 1
            || rank
                .cases
                .iter()
                .any(|case| case.latencies_ns.len() != manifest.iterations)
    }) {
        return Err(crate::CollectivesError::Collective(
            "benchmark rank reports have an unsupported schema or sample count".into(),
        ));
    }
    let mut cases = Vec::with_capacity(case_count);
    for case_index in 0..case_count {
        let first = &ranks[0].cases[case_index];
        if ranks.iter().any(|rank| {
            let case = &rank.cases[case_index];
            case.algorithm != first.algorithm || case.payload_bytes != first.payload_bytes
        }) {
            return Err(crate::CollectivesError::Collective(
                "benchmark ranks emitted inconsistent case identities".into(),
            ));
        }
        let mut maximums = (0..manifest.iterations)
            .map(|iteration| {
                ranks
                    .iter()
                    .map(|rank| rank.cases[case_index].latencies_ns[iteration])
                    .max()
                    .unwrap_or(0)
            })
            .collect::<Vec<_>>();
        maximums.sort_unstable();
        let mean = maximums.iter().sum::<u64>() / manifest.iterations as u64;
        cases.push(CollectiveBenchmarkCase {
            algorithm: first.algorithm,
            payload_bytes: first.payload_bytes,
            iterations: manifest.iterations,
            mean_latency_ns: mean,
            p50_latency_ns: percentile(&maximums, 50),
            p95_latency_ns: percentile(&maximums, 95),
            effective_payload_bytes_per_second: if mean == 0 {
                0.
            } else {
                first.payload_bytes as f64 * 1e9 / mean as f64
            },
            observed_wire_bytes: ranks
                .iter()
                .map(|rank| rank.cases[case_index].sent_bytes)
                .sum(),
        });
    }
    let success = ranks.iter().all(|rank| rank.success);
    Ok(CollectiveBenchmarkReport {
        schema_version: 1,
        backend: "tcp".into(),
        collective_backend: "native".into(),
        world_size: manifest.world_size,
        warmup: manifest.warmup,
        iterations: manifest.iterations,
        cases,
        resources: Some(resources),
        ranks,
        success,
    })
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    let index = ((sorted.len().saturating_sub(1)) * percent).div_ceil(100);
    sorted.get(index).copied().unwrap_or(0)
}

fn ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> CollectiveBenchmarkManifest {
        CollectiveBenchmarkManifest {
            schema_version: 1,
            run_id: "test".into(),
            world_size: 2,
            sizes: vec![4096],
            algorithms: vec![AllReduceAlgorithm::Ring],
            warmup: 1,
            iterations: 2,
        }
    }

    fn rank(rank: usize, latencies_ns: Vec<u64>) -> CollectiveBenchmarkRankReport {
        CollectiveBenchmarkRankReport {
            schema_version: 1,
            rank,
            cases: vec![CollectiveBenchmarkRankCase {
                algorithm: AllReduceAlgorithm::Ring,
                payload_bytes: 4096,
                latencies_ns,
                sent_bytes: 8192,
                traces: Vec::new(),
            }],
            success: true,
        }
    }

    fn resources() -> CollectiveBenchmarkResources {
        CollectiveBenchmarkResources {
            engine_cpu_millis: 8000,
            engine_memory_bytes: 4 << 30,
            requested_cpu_millis: 1000,
            requested_memory_bytes: 512 << 20,
            per_rank_cpu_millis: 500,
            per_rank_memory_bytes: 256 << 20,
            unused_cpu_millis: 0,
            unused_memory_bytes: 0,
        }
    }

    #[test]
    fn tcp_aggregation_uses_maximum_rank_samples_and_rejects_missing_samples() {
        let report = aggregate_all_reduce_benchmark(
            &manifest(),
            vec![rank(1, vec![20, 30]), rank(0, vec![10, 40])],
            resources(),
        )
        .unwrap();
        assert_eq!(report.cases[0].mean_latency_ns, 30);
        assert_eq!(report.cases[0].p50_latency_ns, 40);
        assert_eq!(report.cases[0].observed_wire_bytes, 16_384);
        assert_eq!(report.ranks[0].rank, 0);

        let error = aggregate_all_reduce_benchmark(
            &manifest(),
            vec![rank(0, vec![10]), rank(1, vec![20, 30])],
            resources(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("sample count"));
    }
}
