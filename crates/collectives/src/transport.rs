//! Transport-independent point-to-point message contract.

use crate::{ControlPacket, MessageTag, Rank, Result, TensorPacket};

/// Synchronous point-to-point transport for one rank endpoint.
///
/// Implementations own source identity. A receive matches one explicit source and [`MessageTag`].
/// Both in-memory and TCP backends implement this contract. Additional capabilities such as
/// bounded control packets and barriers are expressed by separate traits.
pub trait Transport: Send + Sync + 'static {
    /// Returns the rank represented by this endpoint.
    fn rank(&self) -> Rank;

    /// Sends one owned tensor packet to a distinct destination rank.
    fn send(&self, destination: usize, tag: MessageTag, packet: TensorPacket) -> Result<()>;

    /// Receives the next packet matching `source` and `tag` before the endpoint deadline.
    fn recv(&self, source: usize, tag: MessageTag) -> Result<TensorPacket>;
}

/// Capability for transports that carry bounded non-tensor control messages.
pub trait ControlTransport: Transport {
    /// Sends one owned control packet to a distinct destination rank.
    fn send_control(
        &self,
        destination: usize,
        tag: MessageTag,
        packet: ControlPacket,
    ) -> Result<()>;

    /// Receives the next control packet matching `source` and `tag`.
    fn recv_control(&self, source: usize, tag: MessageTag) -> Result<ControlPacket>;
}

#[derive(Debug)]
pub(crate) enum MessagePayload {
    Tensor(TensorPacket),
    Control(ControlPacket),
}

#[derive(Debug)]
pub(crate) struct MessageFrame {
    pub(crate) source: usize,
    pub(crate) destination: usize,
    pub(crate) tag: MessageTag,
    pub(crate) payload: MessagePayload,
}
