//! Candle tensor operations layered over an owned-message transport.

use crate::{MessageTag, Rank, Result, TensorPacket, Transport};
use candle_core::Tensor;

/// Tensor-level point-to-point communicator for one rank.
pub struct Communicator<T: Transport> {
    transport: T,
}

impl<T: Transport> Communicator<T> {
    /// Wraps one transport endpoint.
    pub const fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Returns this communicator's rank identity.
    pub fn rank(&self) -> Rank {
        self.transport.rank()
    }

    /// Copies and sends one CPU/F32 Candle tensor.
    pub fn send_tensor(&self, destination: usize, tag: MessageTag, tensor: &Tensor) -> Result<()> {
        self.transport
            .send(destination, tag, TensorPacket::from_tensor(tensor)?)
    }

    /// Receives and reconstructs a new CPU/F32 Candle tensor.
    pub fn recv_tensor(&self, source: usize, tag: MessageTag) -> Result<Tensor> {
        self.transport.recv(source, tag)?.to_tensor()
    }

    /// Returns the owned lower-level transport endpoint.
    pub fn into_transport(self) -> T {
        self.transport
    }
}
