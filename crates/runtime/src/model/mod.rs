//! Owned Llama execution and its persistent per-layer attention cache.
//!
//! Candle supplies tensor storage and primitive operations. This module owns the supported model
//! graph, tensor-shape transitions, causal attention behavior, and cache invariants.

mod cache;
mod llama;

pub use cache::{KvCache, StageKvCache};
pub use llama::{LayerObserver, Llama, LlamaStage, NoopLayerObserver};
