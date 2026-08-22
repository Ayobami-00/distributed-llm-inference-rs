use std::{fs, process::Command};

#[test]
#[ignore = "downloads SmolLM2 and requires Docker with at least 1536 MiB"]
fn smollm2_tp3_docker_matches_single_rank_tokens_and_reports_events() {
    let directory = tempfile::tempdir().unwrap();
    let tensor_report = directory.path().join("tensor.json");
    let single_report = directory.path().join("single.json");
    let prompt = "Explain tensor parallelism.";
    let run_id = format!("tensor-e2e-{}", std::process::id());
    let tensor = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "tensor",
            "--model",
            "smollm2-135m-instruct",
            "--tp",
            "3",
            "--prompt",
            prompt,
            "--max-new-tokens",
            "4",
            "--total-cpus",
            "3",
            "--total-memory",
            "1536MiB",
            "--run-id",
            &run_id,
            "--report",
            tensor_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        tensor.status.success(),
        "tensor run failed: {}",
        String::from_utf8_lossy(&tensor.stderr)
    );
    let single = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "generate",
            "--model",
            "smollm2-135m-instruct",
            "--prompt",
            prompt,
            "--max-new-tokens",
            "4",
            "--report",
            single_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(single.status.success());
    let tensor: serde_json::Value =
        serde_json::from_slice(&fs::read(tensor_report).unwrap()).unwrap();
    let single: serde_json::Value =
        serde_json::from_slice(&fs::read(single_report).unwrap()).unwrap();
    assert_eq!(tensor["generated_tokens"], single["generated_tokens"]);
    assert_eq!(tensor["completion"], single["completion"]);
    assert_eq!(tensor["tensor_parallel"], 3);
    assert_eq!(tensor["success"], true);
    assert!(
        tensor["events"]
            .as_array()
            .is_some_and(|events| !events.is_empty())
    );
    assert!(tensor["ranks"].as_array().unwrap().iter().all(|rank| {
        rank["success"] == true
            && rank["barriers_passed"] == true
            && rank["resources"]["cpu_millis"] == 1000
            && rank["resources"]["memory_limit_bytes"] == 536_870_912u64
            && rank["events"].as_array().is_some_and(|events| {
                events
                    .iter()
                    .filter(|event| event["event"]["kind"] == "memory_sample")
                    .count()
                    == 2
            })
            && rank["collectives"]
                .as_array()
                .is_some_and(|values| !values.is_empty())
    }));
}
