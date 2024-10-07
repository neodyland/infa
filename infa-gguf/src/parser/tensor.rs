use std::borrow::Cow;

use half::{bf16, f16};

pub struct GGUFFloatTensor {
    pub shape: Vec<u64>,
    pub bytes: Box<dyn crate::BaseGGUFBlock>,
    pub data_type: crate::GGMLType,
}

impl GGUFFloatTensor {
    pub fn f32_size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
    fn from_raw_parts<T>(
        bytes: Vec<u8>,
        size: usize,
        shape: &Vec<u64>,
        data_type: &crate::GGMLType,
    ) -> Self
    where
        T: crate::GGUFBlock + 'static,
    {
        let raw_data_ptr = bytes.as_ptr();
        let n_blocks = size / std::mem::size_of::<T>();
        let bytes = unsafe { std::slice::from_raw_parts(raw_data_ptr as *const T, n_blocks) };
        Self {
            shape: shape.clone(),
            bytes: Box::new(bytes.to_vec()),
            data_type: data_type.clone(),
        }
    }
    pub(crate) fn data(&self) -> crate::Result<Cow<[u8]>> {
        let data_ptr = self.bytes.as_ptr();
        let size_in_bytes = self.bytes.bytes_size();
        let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
        Ok(Cow::from(data))
    }
    pub(crate) fn from_data(
        data_type: &crate::GGMLType,
        shape: Vec<u64>,
        size: usize,
        bytes: Vec<u8>,
    ) -> Self {
        match data_type {
            crate::GGMLType::Q4_0 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ4_0>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q4_1 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ4_1>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q5_0 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ5_0>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q5_1 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ5_1>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q8_0 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ8_0>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q8_1 => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ8_1>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q2_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ2K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q3_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ3K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q4_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ4K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q5_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ5K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q6_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ6K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::Q8_K => crate::GGUFFloatTensor::from_raw_parts::<crate::BlockQ8K>(
                bytes, size, &shape, &data_type,
            ),
            crate::GGMLType::BF16 => {
                crate::GGUFFloatTensor::from_raw_parts::<bf16>(bytes, size, &shape, &data_type)
            }
            crate::GGMLType::F16 => {
                crate::GGUFFloatTensor::from_raw_parts::<f16>(bytes, size, &shape, &data_type)
            }
            crate::GGMLType::F32 => {
                crate::GGUFFloatTensor::from_raw_parts::<f32>(bytes, size, &shape, &data_type)
            }
            _ => unimplemented!(),
        }
    }
}

impl infa_impl::BaseTensorOps for GGUFFloatTensor {
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
}

impl infa_impl::Dequantize<infa_impl::Float32Tensor> for GGUFFloatTensor {
    fn dequantize(self) -> infa_impl::Result<infa_impl::Float32Tensor> {
        let v = self
            .bytes
            .to_f32(self.shape.iter().product::<u64>() as usize)
            .map_err(|e| infa_impl::Error::DequantizeError(e.to_string()))?;
        Ok(infa_impl::Float32Tensor {
            shape: self.shape.clone(),
            data: v,
        })
    }
}
