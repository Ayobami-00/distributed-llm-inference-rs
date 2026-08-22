use candle_core::{Device, Tensor};
use dlir_collectives::{Communicator, MessageTag, create_in_memory_world};
use std::time::Duration;

#[test]
fn in_memory_control_matches_tags_and_preserves_tensor_messages() {
    let mut world = create_in_memory_world(2, Duration::from_secs(1)).unwrap();
    let rank_one = Communicator::new(world.pop().unwrap());
    let rank_zero = Communicator::new(world.pop().unwrap());

    rank_zero
        .send_control(1, MessageTag(10), b"first".to_vec())
        .unwrap();
    rank_zero
        .send_tensor(
            1,
            MessageTag(10),
            &Tensor::new(&[1f32, 2.], &Device::Cpu).unwrap(),
        )
        .unwrap();
    rank_zero
        .send_control(1, MessageTag(11), b"second".to_vec())
        .unwrap();

    assert_eq!(rank_one.recv_control(0, MessageTag(11)).unwrap(), b"second");
    assert_eq!(
        rank_one
            .recv_tensor(0, MessageTag(10))
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        vec![1., 2.]
    );
    assert_eq!(rank_one.recv_control(0, MessageTag(10)).unwrap(), b"first");
}

#[test]
fn in_memory_control_rejects_oversized_payloads() {
    let mut world = create_in_memory_world(2, Duration::from_secs(1)).unwrap();
    let rank_zero = Communicator::new(world.remove(0));
    assert!(
        rank_zero
            .send_control(
                1,
                MessageTag(1),
                vec![0; dlir_collectives::MAX_CONTROL_BYTES + 1],
            )
            .is_err()
    );
}
