pub enum Tensor {
    #[cfg(feature = "gguf")]
    GGUFTensor(infa_gguf::GGUFTensor),
}

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = crate::Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (Tensor::GGUFTensor(lhs), Tensor::GGUFTensor(rhs)) => Tensor::GGUFTensor((lhs + rhs)?),
        })
    }
}
impl std::ops::Add<&Tensor> for Tensor {
    type Output = crate::Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (Tensor::GGUFTensor(lhs), Tensor::GGUFTensor(rhs)) => Tensor::GGUFTensor((&lhs + rhs)?),
        })
    }
}
impl std::ops::Sub<&Tensor> for Tensor {
    type Output = crate::Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (Tensor::GGUFTensor(lhs), Tensor::GGUFTensor(rhs)) => Tensor::GGUFTensor((&lhs - rhs)?),
        })
    }
}
impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = crate::Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        Ok(match (self, rhs) {
            #[cfg(feature = "gguf")]
            (Tensor::GGUFTensor(lhs), Tensor::GGUFTensor(rhs)) => Tensor::GGUFTensor((lhs - rhs)?),
        })
    }
}

impl infa_impl::TensorOps<'_, Tensor, crate::Error> for &Tensor {
    fn shape(&self) -> Vec<u64> {
        match self {
            #[cfg(feature = "gguf")]
            Tensor::GGUFTensor(t) => t.shape(),
        }
    }
}

impl infa_impl::TensorOps<'_, Tensor, crate::Error> for Tensor {
    fn shape(&self) -> Vec<u64> {
        match self {
            #[cfg(feature = "gguf")]
            Tensor::GGUFTensor(t) => t.shape(),
        }
    }
}
