//! Typed pipeline control messages carried by protocol-v2 control packets.

use crate::{PipelineError, Result};
use dlir_collectives::MessageTag;
use dlir_runtime::StopReason;
use serde::{Deserialize, Serialize};

const ACTIVATION_BASE: u64 = 0x1000_0000_0000_0000;
const TOKEN_BASE: u64 = 0x2000_0000_0000_0000;
const DECISION_BASE: u64 = 0x3000_0000_0000_0000;

/// Rank-0 decision following one final-stage token selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PipelineDecision {
    /// Begin the next cached decode forward.
    Continue,
    /// End generation successfully.
    Stop {
        /// Successful termination condition.
        reason: StopReason,
    },
}

/// Small typed messages exchanged outside the activation tensor stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PipelineControl {
    /// Token selected by the final pipeline stage.
    Token {
        /// Zero for the prefill result and one-based for cached decode results.
        step: usize,
        /// Greedy vocabulary token ID.
        token_id: u32,
    },
    /// Rank-0 continuation or termination decision.
    Decision {
        /// Step to which this decision applies.
        step: usize,
        /// Continue or stop action.
        decision: PipelineDecision,
    },
}

impl PipelineControl {
    /// Serializes one bounded control payload.
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Decodes one typed control payload.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }
}

/// Returns the activation tensor tag for a phase step.
pub fn activation_tag(step: usize) -> Result<MessageTag> {
    step_tag(ACTIVATION_BASE, step)
}

/// Returns the final-stage token feedback tag for a phase step.
pub fn token_tag(step: usize) -> Result<MessageTag> {
    step_tag(TOKEN_BASE, step)
}

/// Returns the rank-0 decision tag for a phase step.
pub fn decision_tag(step: usize) -> Result<MessageTag> {
    step_tag(DECISION_BASE, step)
}

fn step_tag(base: u64, step: usize) -> Result<MessageTag> {
    let step = u64::try_from(step)
        .map_err(|_| PipelineError::Protocol("pipeline step does not fit u64".into()))?;
    base.checked_add(step)
        .filter(|tag| *tag < base + (1u64 << 60))
        .map(MessageTag)
        .ok_or_else(|| PipelineError::Protocol("pipeline message-tag overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_controls_round_trip_and_tags_use_independent_namespaces() {
        let token = PipelineControl::Token {
            step: 7,
            token_id: 42,
        };
        assert_eq!(
            PipelineControl::decode(&token.encode().unwrap()).unwrap(),
            token
        );
        let decision = PipelineControl::Decision {
            step: 7,
            decision: PipelineDecision::Stop {
                reason: StopReason::ContextLimit,
            },
        };
        assert_eq!(
            PipelineControl::decode(&decision.encode().unwrap()).unwrap(),
            decision
        );
        assert_ne!(activation_tag(7).unwrap(), token_tag(7).unwrap());
        assert_ne!(token_tag(7).unwrap(), decision_tag(7).unwrap());
    }

    #[test]
    fn malformed_control_is_rejected() {
        assert!(PipelineControl::decode(b"not-json").is_err());
    }
}
