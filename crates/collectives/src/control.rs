//! Owned bounded control payloads transferred independently of tensor packets.

use crate::{CollectivesError, Result};

/// Maximum application control payload accepted by the transport.
pub const MAX_CONTROL_BYTES: usize = 64 * 1024;

/// An owned control-plane payload associated with a source and message tag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPacket(Vec<u8>);

impl ControlPacket {
    /// Constructs a bounded payload.
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() > MAX_CONTROL_BYTES {
            return Err(CollectivesError::Protocol(format!(
                "control payload is {} bytes, maximum is {MAX_CONTROL_BYTES}",
                bytes.len()
            )));
        }
        Ok(Self(bytes))
    }

    /// Returns the payload bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// Consumes the packet and returns its owned bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}
