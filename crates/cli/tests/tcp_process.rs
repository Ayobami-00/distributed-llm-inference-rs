use std::{
    net::TcpListener,
    process::{Command, Stdio},
};

#[test]
fn two_os_processes_form_a_tcp_world() {
    let rendezvous_port = free_port();
    let peer_ports = [free_port(), free_port()];
    let binary = env!("CARGO_BIN_EXE_dlir");
    let common = |rank: usize| {
        vec![
            "rank".to_owned(),
            "--rank".to_owned(),
            rank.to_string(),
            "--world-size".to_owned(),
            "2".to_owned(),
            "--run-id".to_owned(),
            "process-test".to_owned(),
            "--rendezvous-addr".to_owned(),
            format!("127.0.0.1:{rendezvous_port}"),
            "--listen-addr".to_owned(),
            format!("127.0.0.1:{}", peer_ports[rank]),
            "--advertise-addr".to_owned(),
            format!("127.0.0.1:{}", peer_ports[rank]),
            "--startup-timeout-seconds".to_owned(),
            "5".to_owned(),
            "--operation-timeout-seconds".to_owned(),
            "2".to_owned(),
        ]
    };

    let mut rank_zero_args = common(0);
    rank_zero_args.push("--rendezvous-bind-addr".to_owned());
    rank_zero_args.push(format!("127.0.0.1:{rendezvous_port}"));
    let rank_zero = Command::new(binary)
        .args(&rank_zero_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let rank_one = Command::new(binary)
        .args(common(1))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let outputs = [
        rank_zero.wait_with_output().unwrap(),
        rank_one.wait_with_output().unwrap(),
    ];
    let reports = outputs
        .iter()
        .map(|output| {
            assert!(
                output.status.success(),
                "rank failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
        })
        .collect::<Vec<_>>();
    assert_ne!(reports[0]["process_id"], reports[1]["process_id"]);
    assert_eq!(reports[0]["rank"], 0);
    assert_eq!(reports[1]["rank"], 1);
    assert_eq!(reports[0]["peers"].as_array().unwrap().len(), 2);
    assert_eq!(reports[1]["peers"].as_array().unwrap().len(), 2);
    assert_eq!(reports[0]["resource_verification"], "not_evaluated");
    assert_eq!(reports[1]["resource_verification"], "not_evaluated");
    assert_eq!(reports[0]["success"], true);
    assert_eq!(reports[1]["success"], true);
}

#[test]
fn launch_rejects_invalid_world_before_contacting_docker() {
    let output = Command::new(env!("CARGO_BIN_EXE_dlir"))
        .args([
            "launch",
            "--nproc",
            "1",
            "--total-cpus",
            "1",
            "--total-memory",
            "512MiB",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("nproc must be between 2 and 64"));
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
