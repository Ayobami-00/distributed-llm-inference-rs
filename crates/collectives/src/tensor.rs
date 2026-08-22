//! Owned CPU/F32 tensor representation used at rank boundaries.

use crate::{CollectivesError, Result};
use candle_core::{DType, Device, Tensor};

/// Owned tensor metadata and values transferred between ranks.
///
/// A packet deliberately owns a `Vec<f32>` instead of retaining a Candle [`Tensor`] handle. This
/// makes the in-memory backend exercise the same copy boundary required by a future network
/// transport.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPacket {
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl TensorPacket {
    /// Constructs a packet and validates that `shape` describes exactly `values.len()` elements.
    pub fn new(shape: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        let expected = element_count(&shape)?;
        if expected != values.len() {
            return Err(CollectivesError::ElementCountMismatch {
                shape,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { shape, values })
    }

    /// Copies a CPU/F32 Candle tensor into an owned packet.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        if !matches!(tensor.device(), Device::Cpu) {
            return Err(CollectivesError::UnsupportedTensorDevice {
                device: format!("{:?}", tensor.device()),
            });
        }
        if tensor.dtype() != DType::F32 {
            return Err(CollectivesError::UnsupportedTensorDType {
                dtype: format!("{:?}", tensor.dtype()),
            });
        }
        let shape = tensor.dims().to_vec();
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        Self::new(shape, values)
    }

    /// Reconstructs a new CPU Candle tensor from this packet.
    pub fn to_tensor(&self) -> Result<Tensor> {
        let expected = element_count(&self.shape)?;
        if expected != self.values.len() {
            return Err(CollectivesError::ElementCountMismatch {
                shape: self.shape.clone(),
                expected,
                actual: self.values.len(),
            });
        }
        Ok(Tensor::from_vec(
            self.values.clone(),
            self.shape.as_slice(),
            &Device::Cpu,
        )?)
    }

    /// Returns the packet's tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the flattened F32 values in row-major order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| CollectivesError::ShapeOverflow {
                shape: shape.to_vec(),
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_multidimensional_f32_tensor() {
        let tensor =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &Device::Cpu).unwrap();
        let packet = TensorPacket::from_tensor(&tensor).unwrap();
        assert_eq!(packet.shape(), &[2, 3]);
        assert_eq!(packet.values(), &[1., 2., 3., 4., 5., 6.]);

        let rebuilt = packet.to_tensor().unwrap();
        assert_eq!(rebuilt.dims(), &[2, 3]);
        assert_eq!(
            rebuilt.to_vec2::<f32>().unwrap(),
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
    }

    #[test]
    fn rejects_non_f32_and_malformed_packets() {
        let tensor = Tensor::new(&[1u32, 2, 3], &Device::Cpu).unwrap();
        assert!(matches!(
            TensorPacket::from_tensor(&tensor),
            Err(CollectivesError::UnsupportedTensorDType { .. })
        ));
        assert!(matches!(
            TensorPacket::new(vec![2, 2], vec![1., 2., 3.]),
            Err(CollectivesError::ElementCountMismatch { .. })
        ));
        assert!(matches!(
            TensorPacket::new(vec![usize::MAX, 2], Vec::new()),
            Err(CollectivesError::ShapeOverflow { .. })
        ));
    }
}
