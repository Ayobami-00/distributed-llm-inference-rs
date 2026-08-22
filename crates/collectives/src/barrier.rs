//! Barrier capability shared by in-memory and network transports.

use crate::{Result, Transport};

/// A transport that can synchronize every rank in its world.
///
/// Barriers are reusable and generation ordered. A caller must not overlap a barrier with another
/// operation on the same communicator.
pub trait BarrierTransport: Transport {
    /// Blocks until every rank reaches the same barrier generation.
    fn barrier(&self) -> Result<()>;
}
