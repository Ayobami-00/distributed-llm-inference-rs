//! Owned Llama execution and its persistent per-layer attention cache.
//!
//! Candle supplies tensor storage and primitive operations. This module owns the supported model
//! graph, tensor-shape transitions, causal attention behavior, and cache invariants.

mod cache;
mod llama;
mod tensor_parallel;

pub use cache::{KvCache, StageKvCache};
pub use llama::{LayerObserver, Llama, LlamaStage, NoopLayerObserver};
pub use tensor_parallel::{
    NoopTensorParallelObserver, TensorParallelLlama, TensorParallelObserver,
};

/// KV cache containing every layer but only one rank's compact GQA heads.
pub type TensorParallelKvCache = KvCache;
