use infa_impl::Dequantize;
use infa_impl::Float32Tensor;
use infa_impl::TensorOps;

pub enum FloatTensor {
    #[cfg(feature = "gguf")]
    GGUFFloatTensor(infa_gguf::GGUFFloatTensor),
    Float32Tensor(infa_impl::Float32Tensor),
}

impl infa_impl::TensorOps<FloatTensor, f32> for FloatTensor {
    fn sum(&self, dim: i64) -> infa_impl::Result<FloatTensor> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => FloatTensor::Float32Tensor(t.sum(dim)?),
            FloatTensor::Float32Tensor(t) => FloatTensor::Float32Tensor(t.sum(dim)?),
        })
    }
    fn item(&self) -> infa_impl::Result<Vec<Self::Item>> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => t.item()?,
            FloatTensor::Float32Tensor(t) => t.item()?,
        })
    }
    fn size(&self) -> infa_impl::Result<usize> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => t.size()?,
            FloatTensor::Float32Tensor(t) => t.size()?,
        })
    }
    fn apply(&self, f: impl Fn(Self::Item) -> Self::Item) -> infa_impl::Result<FloatTensor> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t1) => FloatTensor::Float32Tensor(t1.apply(f)?),
            FloatTensor::Float32Tensor(t1) => FloatTensor::Float32Tensor(t1.apply(f)?),
        })
    }
    fn apply_xy(
        &self,
        rhs: &FloatTensor,
        f: impl Fn(Self::Item, Self::Item) -> Self::Item,
    ) -> infa_impl::Result<FloatTensor> {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.apply_xy(t2, f)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::Float32Tensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t2.apply_xy(t1, f)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t1.apply_xy(&t2.dequantize()?, f)?)
            }
            (FloatTensor::Float32Tensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.apply_xy(t2, f)?)
            }
        })
    }

    fn matmul(&self, rhs: &FloatTensor) -> infa_impl::Result<FloatTensor> {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.matmul(t2)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::Float32Tensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t1.matmul(&t2.dequantize()?)?)
            }
            #[cfg(feature = "gguf")]
            (FloatTensor::GGUFFloatTensor(t1), FloatTensor::GGUFFloatTensor(t2)) => {
                FloatTensor::Float32Tensor(t1.matmul(&t2.dequantize()?)?)
            }
            (FloatTensor::Float32Tensor(t1), FloatTensor::Float32Tensor(t2)) => {
                FloatTensor::Float32Tensor(t1.matmul(t2)?)
            }
        })
    }
}

impl infa_impl::BaseTensorOps for FloatTensor {
    type Item = f32;
    fn shape(&self) -> &Vec<u64> {
        match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => t.shape(),
            FloatTensor::Float32Tensor(t) => t.shape(),
        }
    }

    fn reshape(&self, shape: Vec<i64>) -> infa_impl::Result<Self> {
        Ok(match self {
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(t) => FloatTensor::GGUFFloatTensor(t.reshape(shape)?),
            FloatTensor::Float32Tensor(t) => FloatTensor::Float32Tensor(t.reshape(shape)?),
        })
    }
    fn from_values(shape: Vec<u64>, values: Vec<Self::Item>) -> infa_impl::Result<Self> {
        Float32Tensor::from_values(shape, values).map(FloatTensor::Float32Tensor)
    }
}

impl<'a> std::ops::Add<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn add(self, rhs: Self) -> Self::Output {
        TensorOps::add(self, rhs)
    }
}
impl<'a> std::ops::Add<&'a f32> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn add(self, rhs: &f32) -> Self::Output {
        TensorOps::add_item(self, rhs)
    }
}
impl<'a> std::ops::Sub<&'a f32> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn sub(self, rhs: &f32) -> Self::Output {
        TensorOps::sub_item(self, rhs)
    }
}
impl<'a> std::ops::Sub<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn sub(self, rhs: &FloatTensor) -> Self::Output {
        TensorOps::sub(self, rhs)
    }
}

impl<'a> std::ops::Mul<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn mul(self, rhs: Self) -> Self::Output {
        TensorOps::mul(self, rhs)
    }
}

impl<'a> std::ops::Mul<&'a f32> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn mul(self, rhs: &f32) -> Self::Output {
        TensorOps::mul_item(self, rhs)
    }
}
impl<'a> std::ops::Div<&'a f32> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn div(self, rhs: &f32) -> Self::Output {
        TensorOps::div_item(self, rhs)
    }
}
impl<'a> std::ops::Div<&'a FloatTensor> for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn div(self, rhs: Self) -> Self::Output {
        TensorOps::div(self, rhs)
    }
}

impl<'a> std::ops::Neg for &'a FloatTensor {
    type Output = infa_impl::Result<FloatTensor>;

    fn neg(self) -> Self::Output {
        Ok(match self {
            FloatTensor::Float32Tensor(s) => FloatTensor::Float32Tensor(s.neg()?),
            #[cfg(feature = "gguf")]
            FloatTensor::GGUFFloatTensor(s) => FloatTensor::Float32Tensor(s.neg()?),
        })
    }
}
