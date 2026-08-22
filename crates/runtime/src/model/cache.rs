//! Preallocated batch-one key/value storage for every transformer layer.
//!
//! The cache stores compact GQA heads as `[1, KV heads, capacity, head dimension]`. Appends mutate
//! the sequence axis in place and expose only the populated prefix to attention.

use crate::{DlirError, ModelConfig, Result};
use candle_core::{DType, Device, IndexOp, Tensor};

#[derive(Debug)]
struct LayerKvCache {
    keys: Tensor,
    values: Tensor,
    length: usize,
    capacity: usize,
}

impl LayerKvCache {
    fn new(config: &ModelConfig, capacity: usize, dtype: DType, device: &Device) -> Result<Self> {
        let shape = (1, config.num_key_value_heads, capacity, config.head_dim()?);
        Ok(Self {
            keys: Tensor::zeros(shape, dtype, device)?,
            values: Tensor::zeros(shape, dtype, device)?,
            length: 0,
            capacity,
        })
    }

    fn append(&mut self, keys: &Tensor, values: &Tensor) -> Result<(Tensor, Tensor)> {
        if keys.dims() != values.dims() {
            return Err(DlirError::InvalidConfig(
                "key and value cache chunks have different shapes".into(),
            ));
        }
        let (_, _, sequence, _) = keys.dims4()?;
        let attempted = self.length + sequence;
        if attempted > self.capacity {
            return Err(DlirError::CacheCapacityExceeded {
                attempted,
                capacity: self.capacity,
            });
        }
        // The sequence axis is dimension 2 in [1, KV heads, capacity, head dimension].
        self.keys.slice_set(keys, 2, self.length)?;
        self.values.slice_set(values, 2, self.length)?;
        self.length = attempted;
        // Attention must not observe the unpopulated zero-filled tail of the allocation.
        Ok((
            self.keys.i((.., .., ..self.length, ..))?,
            self.values.i((.., .., ..self.length, ..))?,
        ))
    }
}

/// Preallocated per-layer key/value state for one autoregressive sequence.
///
/// Every layer has the same capacity. After a complete model forward, every layer has also
/// appended the same number of token positions.
#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    capacity: usize,
}

impl KvCache {
    /// Allocates zero-filled key and value tensors for every registered transformer layer.
    ///
    /// Capacity must be within `1..=config.max_position_embeddings`.
    pub fn new(
        config: &ModelConfig,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if capacity == 0 || capacity > config.max_position_embeddings {
            return Err(DlirError::InvalidConfig(format!(
                "cache capacity {capacity} is outside model context 1..={}",
                config.max_position_embeddings
            )));
        }
        Self::new_for_layers(config, config.num_hidden_layers, capacity, dtype, device)
    }

    /// Allocates cache storage for only `layer_count` transformer layers.
    ///
    /// Pipeline stages use this constructor so a rank does not allocate KV state for layers it
    /// does not execute.
    pub fn new_for_layers(
        config: &ModelConfig,
        layer_count: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if layer_count == 0 || layer_count > config.num_hidden_layers {
            return Err(DlirError::InvalidConfig(format!(
                "cache layer count must be between 1 and {}, got {layer_count}",
                config.num_hidden_layers
            )));
        }
        if capacity == 0 || capacity > config.max_position_embeddings {
            return Err(DlirError::InvalidConfig(format!(
                "cache capacity {capacity} is outside model context 1..={}",
                config.max_position_embeddings
            )));
        }
        let layers = (0..layer_count)
            .map(|_| LayerKvCache::new(config, capacity, dtype, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, capacity })
    }

    /// Allocates every transformer layer while storing only this TP rank's compact KV heads.
    pub fn new_tensor_parallel(
        config: &ModelConfig,
        tp_size: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if tp_size == 0 || config.num_key_value_heads % tp_size != 0 {
            return Err(DlirError::InvalidConfig(format!(
                "{} KV heads cannot be divided across TP={tp_size}",
                config.num_key_value_heads
            )));
        }
        let mut local = *config;
        local.hidden_size = config.hidden_size / tp_size;
        local.num_attention_heads = config.num_attention_heads / tp_size;
        local.num_key_value_heads = config.num_key_value_heads / tp_size;
        Self::new(&local, capacity, dtype, device)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the populated token positions visible to the next forward call.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.length)
    }

    /// Returns true when no token positions have been cached.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of local layers stored by this cache.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn append(
        &mut self,
        layer: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.layers
            .get_mut(layer)
            .ok_or_else(|| DlirError::InvalidConfig(format!("cache has no layer {layer}")))?
            .append(keys, values)
    }

    /// Calculates logical bytes populated across all layer keys and values.
    pub fn used_bytes(&self, config: &ModelConfig, dtype: DType) -> Result<u64> {
        let bytes = dtype.size_in_bytes() as u64;
        Ok(2 * self.layers.len() as u64
            * self.len() as u64
            * config.num_key_value_heads as u64
            * config.head_dim()? as u64
            * bytes)
    }
}

/// Stage-local cache alias emphasizing that only assigned layers are allocated.
pub type StageKvCache = KvCache;
