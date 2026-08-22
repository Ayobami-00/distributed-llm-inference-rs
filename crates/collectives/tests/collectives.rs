use candle_core::{Device, Tensor};
use dlir_collectives::{
    AllReduceAlgorithm, CollectiveCommunicator, NativeCollectives, ReduceOp, run_collective_check,
    run_in_memory,
};
use std::time::Duration;

const TIMEOUT: Duration = Duration::from_secs(2);

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn all_native_collectives_are_correct_for_two_three_and_four_ranks() {
    for world_size in [2, 3, 4] {
        let results = run_in_memory(world_size, TIMEOUT, move |communicator| {
            let rank = communicator.rank().global_rank();
            let mut native = NativeCollectives::new(communicator);

            let broadcast_input = Tensor::new(
                if rank == 1 {
                    &[1f32, 2., 3., 4.]
                } else {
                    &[0f32, 0., 0., 0.]
                },
                &Device::Cpu,
            )?;
            let broadcast = native.broadcast(&broadcast_input, 1)?;

            let local = Tensor::from_vec(
                vec![rank as f32 + 1.; world_size * 2],
                world_size * 2,
                &Device::Cpu,
            )?;
            let reduced = native.reduce(&local, 0, ReduceOp::Sum)?;
            let gathered = native.all_gather(
                &Tensor::new(&[rank as f32 * 2., rank as f32 * 2. + 1.], &Device::Cpu)?,
                0,
            )?;
            let centralized =
                native.all_reduce(&local, ReduceOp::Sum, AllReduceAlgorithm::Centralized)?;
            let ring = native.all_reduce(&local, ReduceOp::Sum, AllReduceAlgorithm::Ring)?;

            let full = Tensor::from_vec(
                vec![rank as f32 + 1.; world_size * 2],
                world_size * 2,
                &Device::Cpu,
            )?;
            let scattered = native.reduce_scatter(&full, 0, ReduceOp::Sum)?;

            let all_to_all_input = Tensor::from_vec(
                (0..world_size)
                    .map(|destination| (rank * 100 + destination) as f32)
                    .collect::<Vec<_>>(),
                world_size,
                &Device::Cpu,
            )?;
            let all_to_all = native.all_to_all(&all_to_all_input, 0)?;

            Ok((
                values(&broadcast),
                reduced.as_ref().map(values),
                values(&gathered),
                values(&centralized),
                values(&ring),
                values(&scattered),
                values(&all_to_all),
                native.take_traces(),
            ))
        })
        .unwrap();

        let sum = (world_size * (world_size + 1) / 2) as f32;
        let gathered = (0..world_size * 2)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        for (rank, result) in results.into_iter().enumerate() {
            assert_eq!(result.0, vec![1., 2., 3., 4.]);
            if rank == 0 {
                assert_eq!(result.1, Some(vec![sum; world_size * 2]));
            } else {
                assert_eq!(result.1, None);
            }
            assert_eq!(result.2, gathered);
            assert_eq!(result.3, vec![sum; world_size * 2]);
            assert_eq!(result.4, vec![sum; world_size * 2]);
            assert_eq!(result.5, vec![sum; 2]);
            assert_eq!(
                result.6,
                (0..world_size)
                    .map(|source| (source * 100 + rank) as f32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(result.7.len(), 7);
            assert!(result.7.iter().all(|trace| trace.duration_ns > 0));
        }
    }
}

#[test]
fn collectives_preserve_multidimensional_axis_order() {
    let results = run_in_memory(2, TIMEOUT, |communicator| {
        let rank = communicator.rank().global_rank();
        let mut native = NativeCollectives::new(communicator);
        let local = Tensor::from_vec(vec![rank as f32, rank as f32 + 10.], (2, 1), &Device::Cpu)?;
        let gathered = native.all_gather(&local, 1)?;
        Ok((gathered.dims().to_vec(), values(&gathered)))
    })
    .unwrap();
    assert_eq!(results[0].0, vec![2, 2]);
    assert_eq!(results[0].1, vec![0., 1., 10., 11.]);
    assert_eq!(results[1], results[0]);
}

#[test]
fn ring_rejects_a_non_divisible_flattened_tensor() {
    let error = run_in_memory::<(), _>(2, TIMEOUT, |communicator| {
        let mut native = NativeCollectives::new(communicator);
        let tensor = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        native.all_reduce(&tensor, ReduceOp::Sum, AllReduceAlgorithm::Ring)?;
        Ok(())
    })
    .unwrap_err();
    assert!(error.to_string().contains("divide evenly"));
}

#[test]
fn sharded_collectives_reject_non_divisible_axes() {
    let error = run_in_memory::<(), _>(2, TIMEOUT, |communicator| {
        let mut native = NativeCollectives::new(communicator);
        let tensor = Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?;
        native.reduce_scatter(&tensor, 0, ReduceOp::Sum)?;
        Ok(())
    })
    .unwrap_err();
    assert!(error.to_string().contains("not divisible"));
}

#[test]
fn correctness_report_covers_all_operations_and_round_trips() {
    let report = run_collective_check(4, TIMEOUT).unwrap();
    assert!(report.success);
    assert_eq!(report.ranks.len(), 4);
    assert!(report.ranks.iter().all(|rank| rank.operations.len() == 7));
    let json = serde_json::to_string(&report).unwrap();
    let decoded: dlir_collectives::CollectiveCheckReport = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, report);
}
