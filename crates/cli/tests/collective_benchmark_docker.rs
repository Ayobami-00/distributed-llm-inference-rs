use std::{fs, process::Command};

#[test]
#[ignore = "requires Docker"]
fn two_rank_tcp_all_reduce_benchmark_reports_metrics_and_cleans_up() {
    let directory = tempfile::tempdir().unwrap();
    let report_path = directory.path().join("benchmark.json");
    let run_id = format!("benchmark-e2e-{}", std::process::id());
    let output = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "collectives",
            "bench",
            "--nproc",
            "2",
            "--total-cpus",
            "1",
            "--total-memory",
            "512MiB",
            "--sizes",
            "4KiB,64KiB",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--format",
            "json",
            "--output",
            report_path.to_str().unwrap(),
            "--run-id",
            &run_id,
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "benchmark failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(report_path).unwrap()).unwrap();
    assert_eq!(report["backend"], "tcp");
    assert_eq!(report["world_size"], 2);
    assert_eq!(report["success"], true);
    assert_eq!(report["cases"].as_array().unwrap().len(), 4);
    assert!(report["cases"].as_array().unwrap().iter().all(|case| {
        case["mean_latency_ns"]
            .as_u64()
            .is_some_and(|value| value > 0)
            && case["observed_wire_bytes"]
                .as_u64()
                .is_some_and(|value| value > 0)
    }));
    let containers = Command::new("docker")
        .args([
            "ps",
            "--all",
            "--quiet",
            "--filter",
            &format!("label=io.dlir.run_id={run_id}"),
        ])
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&containers.stdout)
            .trim()
            .is_empty()
    );
    let networks = Command::new("docker")
        .args([
            "network",
            "ls",
            "--quiet",
            "--filter",
            &format!("label=io.dlir.run_id={run_id}"),
        ])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&networks.stdout).trim().is_empty());
}
