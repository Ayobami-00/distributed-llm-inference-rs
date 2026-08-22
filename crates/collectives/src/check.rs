//! Deterministic in-memory correctness runner for the native collective algorithms.

use crate::{
    AllReduceAlgorithm, CollectiveCommunicator, CollectiveKind, NativeCollectives, ReduceOp,
    Result, TensorSummary, run_in_memory,
};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One rank's verification of one collective call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCheckOperation {
    /// Collective that was exercised.
    pub kind: CollectiveKind,
    /// Named native algorithm when the collective has more than one implementation.
    pub algorithm: Option<AllReduceAlgorithm>,
    /// Expected local result; absent for non-root participants of `reduce`.
    pub expected: Option<TensorSummary>,
    /// Actual local result; absent for non-root participants of `reduce`.
    pub actual: Option<TensorSummary>,
    /// Whether shape, dtype, and values matched exactly.
    pub passed: bool,
}

/// Ordered correctness records produced by one logical rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCheckRankReport {
    /// Global rank.
    pub rank: usize,
    /// Calls in collective-sequence order.
    pub operations: Vec<CollectiveCheckOperation>,
}

/// Schema-v1 result of `dlir collectives check`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CollectiveCheckReport {
    /// Serialization contract version; currently `1`.
    pub schema_version: u32,
    /// Point-to-point transport backend used by the check.
    pub backend: String,
    /// Collective implementation family.
    pub collective_backend: String,
    /// Number of in-memory participants.
    pub world_size: usize,
    /// Per-rank records ordered by global rank.
    pub ranks: Vec<CollectiveCheckRankReport>,
    /// True only when every local operation matched its expected result.
    pub success: bool,
}

/// Runs all six collective kinds over deterministic CPU/F32 tensors.
///
/// Both centralized and ring algorithms are checked for all-reduce, so every rank records seven
/// calls while covering the six public collective operations.
pub fn run_collective_check(
    world_size: usize,
    receive_timeout: Duration,
) -> Result<CollectiveCheckReport> {
    let ranks = run_in_memory(world_size, receive_timeout, move |communicator| {
        let rank = communicator.rank().global_rank();
        let mut native = NativeCollectives::new(communicator);
        let mut operations = Vec::new();

        let broadcast_input = tensor(if rank == 0 {
            vec![1., 2., 3., 4.]
        } else {
            vec![0.; 4]
        })?;
        let actual = native.broadcast(&broadcast_input, 0)?;
        push_check(
            &mut operations,
            CollectiveKind::Broadcast,
            None,
            Some(tensor(vec![1., 2., 3., 4.])?),
            Some(actual),
        )?;

        let rank_input = tensor(vec![rank as f32 + 1.; world_size * 2])?;
        let actual = native.reduce(&rank_input, 0, ReduceOp::Sum)?;
        let total = (world_size * (world_size + 1) / 2) as f32;
        let expected = (rank == 0)
            .then(|| tensor(vec![total; world_size * 2]))
            .transpose()?;
        push_check(
            &mut operations,
            CollectiveKind::Reduce,
            None,
            expected,
            actual,
        )?;

        let local_shard = tensor(vec![rank as f32 * 2., rank as f32 * 2. + 1.])?;
        let actual = native.all_gather(&local_shard, 0)?;
        push_check(
            &mut operations,
            CollectiveKind::AllGather,
            None,
            Some(tensor(
                (0..world_size * 2).map(|value| value as f32).collect(),
            )?),
            Some(actual),
        )?;

        for algorithm in [AllReduceAlgorithm::Centralized, AllReduceAlgorithm::Ring] {
            let actual = native.all_reduce(&rank_input, ReduceOp::Sum, algorithm)?;
            push_check(
                &mut operations,
                CollectiveKind::AllReduce,
                Some(algorithm),
                Some(tensor(vec![total; world_size * 2])?),
                Some(actual),
            )?;
        }

        let actual = native.reduce_scatter(&rank_input, 0, ReduceOp::Sum)?;
        push_check(
            &mut operations,
            CollectiveKind::ReduceScatter,
            None,
            Some(tensor(vec![total; 2])?),
            Some(actual),
        )?;

        let all_to_all_input = tensor(
            (0..world_size)
                .map(|destination| (rank * 100 + destination) as f32)
                .collect(),
        )?;
        let actual = native.all_to_all(&all_to_all_input, 0)?;
        push_check(
            &mut operations,
            CollectiveKind::AllToAll,
            None,
            Some(tensor(
                (0..world_size)
                    .map(|source| (source * 100 + rank) as f32)
                    .collect(),
            )?),
            Some(actual),
        )?;

        Ok(CollectiveCheckRankReport { rank, operations })
    })?;
    let success = ranks
        .iter()
        .flat_map(|rank| &rank.operations)
        .all(|operation| operation.passed);
    Ok(CollectiveCheckReport {
        schema_version: 1,
        backend: "in_memory".to_owned(),
        collective_backend: "native".to_owned(),
        world_size,
        ranks,
        success,
    })
}

fn tensor(values: Vec<f32>) -> Result<Tensor> {
    let length = values.len();
    Ok(Tensor::from_vec(values, length, &Device::Cpu)?)
}

fn push_check(
    operations: &mut Vec<CollectiveCheckOperation>,
    kind: CollectiveKind,
    algorithm: Option<AllReduceAlgorithm>,
    expected: Option<Tensor>,
    actual: Option<Tensor>,
) -> Result<()> {
    let expected = expected
        .as_ref()
        .map(TensorSummary::from_tensor)
        .transpose()?;
    let actual = actual
        .as_ref()
        .map(TensorSummary::from_tensor)
        .transpose()?;
    operations.push(CollectiveCheckOperation {
        kind,
        algorithm,
        passed: expected == actual,
        expected,
        actual,
    });
    Ok(())
}
