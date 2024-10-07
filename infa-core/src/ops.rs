use infa_impl::Dequantize;
use infa_impl::TensorOps;

pub enum FloatTensor {
    #[cfg(feature = "gguf")]
    GGUFFloatTensor(infa_gguf::GGUFFloatTensor),
    Float32Tensor(infa_impl::Float32Tensor),
}

impl infa_impl::TensorOps<FloatTensor> for FloatTensor {
    fn add(&self, rhs: &FloatTensor) -> infa_impl::Result<FloatTensor> {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.add(t2)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::Float32Tensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t2.add(t1)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t1.add(&t2.dequantize()?)?)
            }
            (FloatTensor::Float32Tensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.add(t2)?)
            }
        })
    }
    fn sum(&self) -> infa_impl::Result<FloatTensor> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => FloatTensor::Float32Tensor(t.sum()?),
            FloatTensor::Float32Tensor(t) => FloatTensor::Float32Tensor(t.sum()?),
        })
    }

    fn mul(&self, rhs: &FloatTensor) -> infa_impl::Result<FloatTensor> {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.mul(t2)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::Float32Tensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t2.mul(t1)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t1.mul(&t2.dequantize()?)?)
            }
            (FloatTensor::Float32Tensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.mul(t2)?)
            }
        })
    }

    type Item = f32;

    fn item(&self) -> infa_impl::Result<Vec<Self::Item>> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => t.item()?,
            FloatTensor::Float32Tensor(t) => t.item()?,
        })
    }
}

impl infa_impl::BaseTensorOps for FloatTensor {
    fn shape(&self) -> &Vec<u64> {
        match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => t.shape(),
            FloatTensor::Float32Tensor(t) => t.shape(),
        }
    }

    fn reshape(&self, shape: Vec<u64>) -> infa_impl::Result<Self> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => FloatTensor::GGUFFloatTensor(t.reshape(shape)?),
            FloatTensor::Float32Tensor(t) => FloatTensor::Float32Tensor(t.reshape(shape)?),
        })
    }
}

impl<'a> std::ops::Add<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOps::add(self, rhs)
    }
}

impl<'a> std::ops::Mul<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOps::mul(self, rhs)
    }
}
