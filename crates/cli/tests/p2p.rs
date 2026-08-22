use std::process::Command;

#[test]
fn text_demo_shows_bidirectional_reference_exchange() {
    let output = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args(["p2p", "--world-size", "2", "--format", "text"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("rank 0 sent [1, 2, 3, 4] to rank 1"));
    assert!(stdout.contains("rank 0 received [5, 6, 7, 8] from rank 1"));
    assert!(stdout.ends_with("Result: PASS\n"));
}

#[test]
fn json_demo_is_schema_versioned_and_rank_ordered() {
    let output = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args(["p2p", "--world-size", "4", "--format", "json"])
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["backend"], "in_memory");
    assert_eq!(report["pattern"], "ring");
    assert_eq!(report["world_size"], 4);
    assert_eq!(report["ranks"][0]["rank"], 0);
    assert_eq!(report["ranks"][3]["rank"], 3);
    assert_eq!(report["success"], true);
}

#[test]
fn invalid_world_size_exits_nonzero() {
    let output = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args(["p2p", "--world-size", "1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("p2p requires world size 2 or greater")
    );
}
