//! Deterministic report types for the point-to-point ring demonstration.

use crate::{CollectivesError, MessageTag, Result, run_in_memory};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const RING_TAG: MessageTag = MessageTag(0);

/// Serializable summary of one CPU/F32 tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSummary {
    /// Stable dtype name; always `f32` in v0.2.
    pub dtype: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Flattened values in row-major order.
    pub values: Vec<f32>,
}

impl TensorSummary {
    /// Copies a CPU/F32 tensor into a serializable shape/value summary.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let packet = crate::TensorPacket::from_tensor(tensor)?;
        Ok(Self {
            dtype: "f32".to_owned(),
            shape: packet.shape().to_vec(),
            values: packet.values().to_vec(),
        })
    }
}

/// Result recorded by one rank in a ring exchange.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankExchangeReport {
    /// Worker global rank.
    pub rank: usize,
    /// Destination of this rank's tensor.
    pub sent_to: usize,
    /// Source of this rank's received tensor.
    pub received_from: usize,
    /// Tensor created and sent by this rank.
    pub sent: TensorSummary,
    /// Newly reconstructed tensor received from the previous rank.
    pub received: TensorSummary,
    /// Whether `received` exactly matches the deterministic source-rank tensor.
    pub matches_expected: bool,
}

/// Schema-versioned result of a deterministic in-memory p2p ring exchange.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct P2pReport {
    /// Serialization contract version; currently `1`.
    pub schema_version: u32,
    /// Transport backend identity; `in_memory` in v0.2.
    pub backend: String,
    /// Exchange pattern; `ring` in v0.2.
    pub pattern: String,
    /// Number of logical rank workers.
    pub world_size: usize,
    /// Rank records ordered by `rank`.
    pub ranks: Vec<RankExchangeReport>,
    /// True only when every rank received its expected tensor.
    pub success: bool,
}

/// Runs the deterministic CPU/F32 ring used by `dlir p2p`.
pub fn run_p2p_ring(world_size: usize, receive_timeout: Duration) -> Result<P2pReport> {
    if world_size < 2 {
        return Err(CollectivesError::P2pWorldTooSmall { world_size });
    }

    let ranks = run_in_memory(world_size, receive_timeout, move |communicator| {
        let rank = communicator.rank().global_rank();
        let sent_to = (rank + 1) % world_size;
        let received_from = (rank + world_size - 1) % world_size;
        let sent_values = values_for_rank(rank);
        let sent = Tensor::from_vec(sent_values, 4, &Device::Cpu)?;

        // Channels are unbounded, so every rank can send before any rank begins its receive.
        communicator.send_tensor(sent_to, RING_TAG, &sent)?;
        let received = communicator.recv_tensor(received_from, RING_TAG)?;
        let expected = values_for_rank(received_from);
        let received_summary = TensorSummary::from_tensor(&received)?;
        let matches_expected = received_summary.shape == [4]
            && received_summary.dtype == "f32"
            && received_summary.values == expected;

        Ok(RankExchangeReport {
            rank,
            sent_to,
            received_from,
            sent: TensorSummary::from_tensor(&sent)?,
            received: received_summary,
            matches_expected,
        })
    })?;
    let success = ranks.iter().all(|rank| rank.matches_expected);

    Ok(P2pReport {
        schema_version: 1,
        backend: "in_memory".to_owned(),
        pattern: "ring".to_owned(),
        world_size,
        ranks,
        success,
    })
}

fn values_for_rank(rank: usize) -> Vec<f32> {
    let base = rank * 4 + 1;
    (base..base + 4).map(|value| value as f32).collect()
}
