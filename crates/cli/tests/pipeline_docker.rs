use std::{fs, process::Command};

fn assert_pipeline_matches_single(model: &str, max_new_tokens: usize, total_memory: &str) {
    let directory = tempfile::tempdir().unwrap();
    let pipeline_report = directory.path().join("pipeline.json");
    let single_report = directory.path().join("single.json");
    let prompt = "Explain pipeline parallelism.";
    let run_id = format!("pipeline-e2e-{}", std::process::id());

    let pipeline = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "pipeline",
            "--model",
            model,
            "--prompt",
            prompt,
            "--max-new-tokens",
            &max_new_tokens.to_string(),
            "--nproc",
            "2",
            "--total-cpus",
            "2",
            "--total-memory",
            total_memory,
            "--run-id",
            &run_id,
            "--report",
            pipeline_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        pipeline.status.success(),
        "pipeline failed: {}",
        String::from_utf8_lossy(&pipeline.stderr)
    );

    let single = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "generate",
            "--model",
            model,
            "--prompt",
            prompt,
            "--max-new-tokens",
            &max_new_tokens.to_string(),
            "--report",
            single_report.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        single.status.success(),
        "single generation failed: {}",
        String::from_utf8_lossy(&single.stderr)
    );

    let pipeline: serde_json::Value =
        serde_json::from_slice(&fs::read(pipeline_report).unwrap()).unwrap();
    let single: serde_json::Value =
        serde_json::from_slice(&fs::read(single_report).unwrap()).unwrap();
    assert_eq!(pipeline["schema_version"], 1);
    assert_eq!(pipeline["generated_tokens"], single["generated_tokens"]);
    assert_eq!(pipeline["completion"], single["completion"]);
    assert_eq!(pipeline["success"], true);
    assert!(pipeline["ranks"].as_array().unwrap().iter().all(|rank| {
        rank["barriers_passed"] == true
            && rank["success"] == true
            && rank["peers"]
                .as_array()
                .is_some_and(|peers| peers.len() == 2)
    }));
}

#[test]
#[ignore = "downloads the pinned SmolLM2 checkpoint and requires Docker"]
fn smollm2_two_rank_pipeline_matches_single_rank_tokens() {
    assert_pipeline_matches_single("smollm2-135m-instruct", 4, "2GiB");
}

#[test]
#[ignore = "downloads the pinned TinyLlama checkpoint and requires Docker with at least 6 GiB"]
fn tinyllama_two_rank_pipeline_matches_single_rank_token() {
    assert_pipeline_matches_single("tinyllama-1.1b-chat", 1, "6GiB");
}
