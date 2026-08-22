use candle_core::{Device, Tensor};
use dlir_collectives::{
    CollectivesError, Communicator, DEFAULT_MAX_TENSOR_BYTES, MessageTag, Rank, TcpTransport,
    TcpTransportConfig,
};
use std::{net::TcpListener, thread, time::Duration};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const OPERATION_TIMEOUT: Duration = Duration::from_secs(2);

#[test]
fn full_mesh_ring_and_reusable_barrier_work_for_multiple_worlds() {
    for world_size in [2, 3, 4] {
        let rendezvous_port = free_port();
        let peer_ports = (0..world_size).map(|_| free_port()).collect::<Vec<_>>();
        let handles = (0..world_size)
            .map(|global_rank| {
                let peer_ports = peer_ports.clone();
                thread::spawn(move || {
                    let rank = Rank::new(global_rank, world_size).unwrap();
                    let transport = TcpTransport::connect(TcpTransportConfig {
                        rank,
                        run_id: format!("tcp-test-{world_size}"),
                        rendezvous_addr: format!("127.0.0.1:{rendezvous_port}"),
                        rendezvous_bind_addr: (global_rank == 0)
                            .then(|| format!("127.0.0.1:{rendezvous_port}")),
                        listen_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        advertise_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        startup_timeout: STARTUP_TIMEOUT,
                        operation_timeout: OPERATION_TIMEOUT,
                        max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
                    })
                    .unwrap();
                    assert_eq!(transport.peers().len(), world_size);
                    let communicator = Communicator::new(transport);
                    communicator.barrier().unwrap_or_else(|error| {
                        panic!("world {world_size} rank {global_rank} startup barrier: {error}")
                    });

                    let destination = (global_rank + 1) % world_size;
                    let source = (global_rank + world_size - 1) % world_size;
                    let base = global_rank * 4 + 1;
                    let sent = (base..base + 4)
                        .map(|value| value as f32)
                        .collect::<Vec<_>>();
                    communicator
                        .send_tensor(
                            destination,
                            MessageTag(7),
                            &Tensor::from_vec(sent, 4, &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                    let received = communicator
                        .recv_tensor(source, MessageTag(7))
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    communicator.barrier().unwrap_or_else(|error| {
                        panic!("world {world_size} rank {global_rank} completion barrier: {error}")
                    });
                    communicator.barrier().unwrap_or_else(|error| {
                        panic!("world {world_size} rank {global_rank} reuse barrier: {error}")
                    });
                    (global_rank, received)
                })
            })
            .collect::<Vec<_>>();

        let mut results = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        results.sort_by_key(|(rank, _)| *rank);
        for (rank, received) in results {
            let source = (rank + world_size - 1) % world_size;
            let base = source * 4 + 1;
            assert_eq!(
                received,
                (base..base + 4)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn tcp_receive_preserves_out_of_order_tags() {
    let rendezvous_port = free_port();
    let peer_ports = [free_port(), free_port()];
    let handles = (0..2)
        .map(|global_rank| {
            thread::spawn(move || {
                let communicator = Communicator::new(
                    TcpTransport::connect(TcpTransportConfig {
                        rank: Rank::new(global_rank, 2).unwrap(),
                        run_id: "tcp-tags".to_owned(),
                        rendezvous_addr: format!("127.0.0.1:{rendezvous_port}"),
                        rendezvous_bind_addr: (global_rank == 0)
                            .then(|| format!("127.0.0.1:{rendezvous_port}")),
                        listen_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        advertise_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        startup_timeout: STARTUP_TIMEOUT,
                        operation_timeout: OPERATION_TIMEOUT,
                        max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
                    })
                    .unwrap(),
                );
                if global_rank == 0 {
                    for (tag, value) in [(MessageTag(10), 10f32), (MessageTag(11), 11f32)] {
                        communicator
                            .send_tensor(1, tag, &Tensor::new(&[value], &Device::Cpu).unwrap())
                            .unwrap();
                    }
                    Vec::new()
                } else {
                    let second = communicator
                        .recv_tensor(0, MessageTag(11))
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()[0];
                    let first = communicator
                        .recv_tensor(0, MessageTag(10))
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()[0];
                    vec![first, second]
                }
            })
        })
        .collect::<Vec<_>>();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results[1], vec![10., 11.]);
}

#[test]
fn tcp_control_and_tensor_frames_match_independently_and_out_of_order() {
    let rendezvous_port = free_port();
    let peer_ports = [free_port(), free_port()];
    let handles = (0..2)
        .map(|global_rank| {
            thread::spawn(move || {
                let communicator = Communicator::new(
                    TcpTransport::connect(TcpTransportConfig {
                        rank: Rank::new(global_rank, 2).unwrap(),
                        run_id: "tcp-control-tags".to_owned(),
                        rendezvous_addr: format!("127.0.0.1:{rendezvous_port}"),
                        rendezvous_bind_addr: (global_rank == 0)
                            .then(|| format!("127.0.0.1:{rendezvous_port}")),
                        listen_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        advertise_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                        startup_timeout: STARTUP_TIMEOUT,
                        operation_timeout: OPERATION_TIMEOUT,
                        max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
                    })
                    .unwrap(),
                );
                if global_rank == 0 {
                    communicator
                        .send_control(1, MessageTag(20), b"first".to_vec())
                        .unwrap();
                    communicator
                        .send_tensor(
                            1,
                            MessageTag(20),
                            &Tensor::new(&[3f32, 4.], &Device::Cpu).unwrap(),
                        )
                        .unwrap();
                    communicator
                        .send_control(1, MessageTag(21), b"second".to_vec())
                        .unwrap();
                    Vec::new()
                } else {
                    let second = communicator.recv_control(0, MessageTag(21)).unwrap();
                    let tensor = communicator
                        .recv_tensor(0, MessageTag(20))
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let first = communicator.recv_control(0, MessageTag(20)).unwrap();
                    [first, second]
                        .into_iter()
                        .flatten()
                        .chain(tensor.into_iter().map(|value| value as u8))
                        .collect()
                }
            })
        })
        .collect::<Vec<_>>();
    let results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(results[1], b"firstsecond\x03\x04");
}

#[test]
fn tcp_receive_and_barrier_use_total_deadlines() {
    let rendezvous_port = free_port();
    let peer_ports = [free_port(), free_port()];
    let handles = (0..2)
        .map(|global_rank| {
            thread::spawn(move || {
                let transport = TcpTransport::connect(TcpTransportConfig {
                    rank: Rank::new(global_rank, 2).unwrap(),
                    run_id: "tcp-timeout".to_owned(),
                    rendezvous_addr: format!("127.0.0.1:{rendezvous_port}"),
                    rendezvous_bind_addr: (global_rank == 0)
                        .then(|| format!("127.0.0.1:{rendezvous_port}")),
                    listen_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                    advertise_addr: format!("127.0.0.1:{}", peer_ports[global_rank]),
                    startup_timeout: STARTUP_TIMEOUT,
                    operation_timeout: Duration::from_millis(50),
                    max_tensor_bytes: DEFAULT_MAX_TENSOR_BYTES,
                })
                .unwrap();
                let communicator = Communicator::new(transport);
                if global_rank == 0 {
                    assert!(matches!(
                        communicator.recv_tensor(1, MessageTag(99)),
                        Err(CollectivesError::ReceiveTimeout { .. })
                    ));
                    assert!(matches!(
                        communicator.barrier(),
                        Err(CollectivesError::BarrierTimeout { .. })
                            | Err(CollectivesError::ReceiveDisconnected { .. })
                            | Err(CollectivesError::Io { .. })
                    ));
                } else {
                    thread::sleep(Duration::from_millis(150));
                }
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}
