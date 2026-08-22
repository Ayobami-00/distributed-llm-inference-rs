//! Validated identity for one participant in a communication world.

use crate::{CollectivesError, Result};
use std::fmt;

/// Identity and world membership of one logical distributed worker.
///
/// Valid ranks form the contiguous range `0..world_size`. In v0.2 each rank is hosted by one
/// worker thread and owns one logical CPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rank {
    global_rank: usize,
    world_size: usize,
}

impl Rank {
    /// Constructs a rank after validating its world membership.
    pub fn new(global_rank: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 {
            return Err(CollectivesError::InvalidWorldSize);
        }
        if global_rank >= world_size {
            return Err(CollectivesError::InvalidRank {
                rank: global_rank,
                world_size,
            });
        }
        Ok(Self {
            global_rank,
            world_size,
        })
    }

    /// Returns this worker's zero-based global rank.
    pub const fn global_rank(self) -> usize {
        self.global_rank
    }

    /// Returns the number of ranks participating in this world.
    pub const fn world_size(self) -> usize {
        self.world_size
    }

    /// Validates that `peer` is a distinct rank in the same world.
    pub fn validate_peer(self, peer: usize) -> Result<()> {
        if peer >= self.world_size {
            return Err(CollectivesError::InvalidPeer {
                rank: self.global_rank,
                peer,
                world_size: self.world_size,
            });
        }
        if peer == self.global_rank {
            return Err(CollectivesError::SelfSend {
                rank: self.global_rank,
            });
        }
        Ok(())
    }
}

/// Application-selected identifier used to match point-to-point messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageTag(
    /// Stable numeric tag value.
    pub u64,
);

impl fmt::Display for MessageTag {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_world_and_rank_boundaries() {
        assert!(matches!(
            Rank::new(0, 0),
            Err(CollectivesError::InvalidWorldSize)
        ));
        assert!(matches!(
            Rank::new(2, 2),
            Err(CollectivesError::InvalidRank { .. })
        ));

        let rank = Rank::new(1, 3).unwrap();
        assert_eq!(rank.global_rank(), 1);
        assert_eq!(rank.world_size(), 3);
        assert!(rank.validate_peer(0).is_ok());
        assert!(matches!(
            rank.validate_peer(1),
            Err(CollectivesError::SelfSend { .. })
        ));
        assert!(matches!(
            rank.validate_peer(3),
            Err(CollectivesError::InvalidPeer { .. })
        ));
    }
}
