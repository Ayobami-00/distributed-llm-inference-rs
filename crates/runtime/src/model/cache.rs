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
        self.keys.slice_set(keys, 2, self.length)?;
        self.values.slice_set(values, 2, self.length)?;
        self.length = attempted;
        Ok((
            self.keys.i((.., .., ..self.length, ..))?,
            self.values.i((.., .., ..self.length, ..))?,
        ))
    }
}

#[derive(Debug)]
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    capacity: usize,
}

impl KvCache {
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
        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerKvCache::new(config, capacity, dtype, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, capacity })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.length)
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

    pub fn used_bytes(&self, config: &ModelConfig, dtype: DType) -> Result<u64> {
        let bytes = dtype.size_in_bytes() as u64;
        Ok(2 * config.num_hidden_layers as u64
            * self.len() as u64
            * config.num_key_value_heads as u64
            * config.head_dim()? as u64
            * bytes)
    }
}
