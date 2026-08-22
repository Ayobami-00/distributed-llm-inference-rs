//! Transport-independent point-to-point message contract.

use crate::{MessageTag, Rank, Result, TensorPacket};

/// Synchronous point-to-point transport for one rank endpoint.
///
/// Implementations own source identity. A receive matches one explicit source and [`MessageTag`].
/// The v0.2 implementation is in-memory; a later TCP backend will implement the same contract.
pub trait Transport: Send + Sync + 'static {
    /// Returns the rank represented by this endpoint.
    fn rank(&self) -> Rank;

    /// Sends one owned tensor packet to a distinct destination rank.
    fn send(&self, destination: usize, tag: MessageTag, packet: TensorPacket) -> Result<()>;

    /// Receives the next packet matching `source` and `tag` before the endpoint deadline.
    fn recv(&self, source: usize, tag: MessageTag) -> Result<TensorPacket>;
}

#[derive(Debug)]
pub(crate) struct MessageFrame {
    pub(crate) source: usize,
    pub(crate) destination: usize,
    pub(crate) tag: MessageTag,
    pub(crate) packet: TensorPacket,
}
