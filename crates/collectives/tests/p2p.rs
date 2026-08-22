use candle_core::{Device, Tensor};
use dlir_collectives::{
    CollectivesError, MessageTag, TensorPacket, Transport, create_in_memory_world, run_in_memory,
    run_p2p_ring,
};
use std::time::Duration;

const TEST_TIMEOUT: Duration = Duration::from_millis(250);

#[test]
fn rank_zero_sends_the_reference_tensor_to_rank_one() {
    let results = run_in_memory(2, TEST_TIMEOUT, |communicator| {
        if communicator.rank().global_rank() == 0 {
            let tensor = Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu)?;
            communicator.send_tensor(1, MessageTag(1), &tensor)?;
            Ok(Vec::new())
        } else {
            Ok(communicator
                .recv_tensor(0, MessageTag(1))?
                .to_vec1::<f32>()?)
        }
    })
    .unwrap();
    assert_eq!(results[1], vec![1., 2., 3., 4.]);
}

#[test]
fn two_ranks_exchange_tensors_bidirectionally() {
    let results = run_in_memory(2, TEST_TIMEOUT, |communicator| {
        let rank = communicator.rank().global_rank();
        let peer = 1 - rank;
        let tensor = Tensor::new(&[rank as f32], &Device::Cpu)?;
        communicator.send_tensor(peer, MessageTag(2), &tensor)?;
        Ok(communicator
            .recv_tensor(peer, MessageTag(2))?
            .to_vec1::<f32>()?)
    })
    .unwrap();
    assert_eq!(results, vec![vec![1.], vec![0.]]);
}

#[test]
fn ring_exchange_is_correct_and_ordered_for_multiple_world_sizes() {
    for world_size in [2, 3, 4] {
        let report = run_p2p_ring(world_size, TEST_TIMEOUT).unwrap();
        assert!(report.success);
        assert_eq!(report.world_size, world_size);
        assert_eq!(
            report
                .ranks
                .iter()
                .map(|rank| rank.rank)
                .collect::<Vec<_>>(),
            (0..world_size).collect::<Vec<_>>()
        );
        assert_eq!(report.ranks[0].sent.values, vec![1., 2., 3., 4.]);
    }
}

#[test]
fn recv_preserves_messages_with_other_tags() {
    let results = run_in_memory(2, TEST_TIMEOUT, |communicator| {
        if communicator.rank().global_rank() == 0 {
            for (tag, value) in [(MessageTag(10), 10f32), (MessageTag(11), 11f32)] {
                communicator.send_tensor(1, tag, &Tensor::new(&[value], &Device::Cpu)?)?;
            }
            Ok(Vec::new())
        } else {
            let second = communicator
                .recv_tensor(0, MessageTag(11))?
                .to_vec1::<f32>()?[0];
            let first = communicator
                .recv_tensor(0, MessageTag(10))?
                .to_vec1::<f32>()?[0];
            Ok(vec![first, second])
        }
    })
    .unwrap();
    assert_eq!(results[1], vec![10., 11.]);
}

#[test]
fn repeated_tags_do_not_consume_stale_messages() {
    let results = run_in_memory(2, TEST_TIMEOUT, |communicator| {
        let rank = communicator.rank().global_rank();
        let peer = 1 - rank;
        let mut received = Vec::new();
        for iteration in 0..32u64 {
            let value = (rank as u64 * 100 + iteration) as f32;
            communicator.send_tensor(
                peer,
                MessageTag(iteration),
                &Tensor::new(&[value], &Device::Cpu)?,
            )?;
            received.push(
                communicator
                    .recv_tensor(peer, MessageTag(iteration))?
                    .to_vec1::<f32>()?[0],
            );
        }
        Ok(received)
    })
    .unwrap();
    assert_eq!(
        results[0],
        (100..132).map(|value| value as f32).collect::<Vec<_>>()
    );
    assert_eq!(
        results[1],
        (0..32).map(|value| value as f32).collect::<Vec<_>>()
    );
}

#[test]
fn receive_times_out_and_detects_disconnected_source() {
    let mut endpoints = create_in_memory_world(2, Duration::from_millis(10)).unwrap();
    let rank_one = endpoints.pop().unwrap();
    assert!(matches!(
        rank_one.recv(0, MessageTag(1)),
        Err(CollectivesError::ReceiveTimeout { .. })
    ));

    let mut endpoints = create_in_memory_world(2, TEST_TIMEOUT).unwrap();
    let rank_one = endpoints.pop().unwrap();
    drop(endpoints.pop().unwrap());
    assert!(matches!(
        rank_one.recv(0, MessageTag(1)),
        Err(CollectivesError::ReceiveDisconnected { .. })
    ));
}

#[test]
fn send_detects_disconnected_destination() {
    let mut endpoints = create_in_memory_world(2, TEST_TIMEOUT).unwrap();
    let rank_one = endpoints.pop().unwrap();
    let rank_zero = endpoints.pop().unwrap();
    drop(rank_one);
    assert!(matches!(
        rank_zero.send(
            1,
            MessageTag(1),
            TensorPacket::new(vec![1], vec![1.]).unwrap()
        ),
        Err(CollectivesError::SendDisconnected { .. })
    ));
}

#[test]
fn invalid_peers_and_self_send_are_rejected() {
    let mut endpoints = create_in_memory_world(2, TEST_TIMEOUT).unwrap();
    let rank_zero = endpoints.remove(0);
    let packet = TensorPacket::new(vec![1], vec![1.]).unwrap();
    assert!(matches!(
        rank_zero.send(0, MessageTag(0), packet.clone()),
        Err(CollectivesError::SelfSend { .. })
    ));
    assert!(matches!(
        rank_zero.send(2, MessageTag(0), packet),
        Err(CollectivesError::InvalidPeer { .. })
    ));
}

#[test]
fn runner_propagates_worker_errors_and_panics() {
    let error = run_in_memory::<(), _>(2, TEST_TIMEOUT, |communicator| {
        if communicator.rank().global_rank() == 0 {
            Err(CollectivesError::P2pWorldTooSmall { world_size: 1 })
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(
        error,
        CollectivesError::WorkerFailed { rank: 0, .. }
    ));

    let error = run_in_memory::<(), _>(2, TEST_TIMEOUT, |communicator| {
        if communicator.rank().global_rank() == 1 {
            panic!("intentional worker panic");
        }
        Ok(())
    })
    .unwrap_err();
    assert!(matches!(
        error,
        CollectivesError::WorkerPanicked { rank: 1, .. }
    ));
}

#[test]
fn failed_sender_disconnects_a_waiting_receiver() {
    let error = run_in_memory::<(), _>(2, TEST_TIMEOUT, |communicator| {
        if communicator.rank().global_rank() == 0 {
            Err(CollectivesError::P2pWorldTooSmall { world_size: 1 })
        } else {
            communicator.recv_tensor(0, MessageTag(99))?;
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(
        error,
        CollectivesError::WorkerFailed { rank: 0, .. }
    ));
}

#[test]
fn p2p_report_round_trips_through_json() {
    let report = run_p2p_ring(2, TEST_TIMEOUT).unwrap();
    let json = serde_json::to_string(&report).unwrap();
    let decoded: dlir_collectives::P2pReport = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, report);
}
