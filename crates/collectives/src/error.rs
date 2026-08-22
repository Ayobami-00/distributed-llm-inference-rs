//! Errors produced by rank validation, tensor transfer, and worker execution.

use crate::MessageTag;
use std::time::Duration;
use thiserror::Error;

/// Result type returned by `dlir-collectives` operations.
pub type Result<T> = std::result::Result<T, CollectivesError>;

/// Failure detected at a point-to-point communication boundary.
#[derive(Debug, Error)]
pub enum CollectivesError {
    /// A world must contain at least one rank.
    #[error("world size must be greater than zero")]
    InvalidWorldSize,
    /// A global rank falls outside the world's contiguous rank range.
    #[error("rank {rank} is outside world size {world_size}")]
    InvalidRank {
        /// Rejected global rank.
        rank: usize,
        /// Number of ranks in the world.
        world_size: usize,
    },
    /// A source or destination rank falls outside the caller's world.
    #[error("peer rank {peer} is outside rank {rank}'s world size {world_size}")]
    InvalidPeer {
        /// Calling global rank.
        rank: usize,
        /// Rejected peer rank.
        peer: usize,
        /// Number of ranks in the caller's world.
        world_size: usize,
    },
    /// Point-to-point operations deliberately reject sending to the same rank.
    #[error("rank {rank} cannot send to or receive from itself")]
    SelfSend {
        /// Calling global rank.
        rank: usize,
    },
    /// The p2p CLI demonstration needs at least two distinct ranks.
    #[error("p2p requires world size 2 or greater, got {world_size}")]
    P2pWorldTooSmall {
        /// Rejected world size.
        world_size: usize,
    },
    /// The implemented transports transfer only CPU tensors.
    #[error("point-to-point transfer supports only CPU tensors, got {device}")]
    UnsupportedTensorDevice {
        /// Debug representation of the supplied Candle device.
        device: String,
    },
    /// The implemented transports transfer only F32 elements.
    #[error("point-to-point transfer supports only f32 tensors, got {dtype}")]
    UnsupportedTensorDType {
        /// Display representation of the supplied Candle dtype.
        dtype: String,
    },
    /// Tensor dimensions overflowed `usize` while their element count was calculated.
    #[error("tensor shape {shape:?} overflows the addressable element count")]
    ShapeOverflow {
        /// Rejected tensor shape.
        shape: Vec<usize>,
    },
    /// A tensor packet's shape and value count disagree.
    #[error("tensor shape {shape:?} requires {expected} values but packet contains {actual}")]
    ElementCountMismatch {
        /// Packet tensor shape.
        shape: Vec<usize>,
        /// Element count implied by the shape.
        expected: usize,
        /// Actual value count.
        actual: usize,
    },
    /// A rank-pair channel could not accept another message.
    #[error("rank {rank} could not send tag {tag} to rank {destination}: channel disconnected")]
    SendDisconnected {
        /// Sending global rank.
        rank: usize,
        /// Destination global rank.
        destination: usize,
        /// Message tag being sent.
        tag: MessageTag,
    },
    /// The requested source endpoint disappeared before a matching message arrived.
    #[error(
        "rank {rank} could not receive tag {tag} from rank {source_rank}: channel disconnected"
    )]
    ReceiveDisconnected {
        /// Receiving global rank.
        rank: usize,
        /// Expected source global rank.
        source_rank: usize,
        /// Expected message tag.
        tag: MessageTag,
    },
    /// No matching message arrived before the endpoint's total receive deadline.
    #[error(
        "rank {rank} timed out after {timeout:?} waiting for tag {tag} from rank {source_rank}"
    )]
    ReceiveTimeout {
        /// Receiving global rank.
        rank: usize,
        /// Expected source global rank.
        source_rank: usize,
        /// Expected message tag.
        tag: MessageTag,
        /// Configured total receive timeout.
        timeout: Duration,
    },
    /// Not every rank reached a barrier before its total deadline.
    #[error("rank {rank} timed out after {timeout:?} in barrier generation {generation}")]
    BarrierTimeout {
        /// Calling global rank.
        rank: usize,
        /// Reusable barrier generation.
        generation: u64,
        /// Configured total deadline.
        timeout: Duration,
    },
    /// A previous participant failure made the current barrier generation unusable.
    #[error("rank {rank} entered broken barrier generation {generation}")]
    BarrierBroken {
        /// Calling global rank.
        rank: usize,
        /// Broken barrier generation.
        generation: u64,
    },
    /// A socket, listener, or stream operation failed.
    #[error("{context}: {source}")]
    Io {
        /// Operation-specific context.
        context: String,
        /// Operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// A peer sent an invalid or incompatible wire message.
    #[error("protocol error: {0}")]
    Protocol(String),
    /// Rendezvous failed before the peer world was established.
    #[error("rendezvous error: {0}")]
    Rendezvous(String),
    /// A collective call violated an operation, shape, rank, or sequence invariant.
    #[error("collective error: {0}")]
    Collective(String),
    /// JSON control-plane encoding or decoding failed.
    #[error("control-plane JSON error: {0}")]
    ControlJson(#[from] serde_json::Error),
    /// A mutex was poisoned because another worker panicked while holding it.
    #[error("rank {rank} communication state was poisoned")]
    Synchronization {
        /// Rank whose state could not be locked.
        rank: usize,
    },
    /// A worker returned a normal collectives error.
    #[error("rank {rank} worker failed: {source}")]
    WorkerFailed {
        /// Failed worker's global rank.
        rank: usize,
        /// Rank-local cause.
        #[source]
        source: Box<CollectivesError>,
    },
    /// A worker unwound instead of returning a result.
    #[error("rank {rank} worker panicked: {message}")]
    WorkerPanicked {
        /// Panicking worker's global rank.
        rank: usize,
        /// Extracted panic payload or a fallback description.
        message: String,
    },
    /// Candle tensor access or construction failed.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
}
