pub enum Tensor {
    #[cfg(feature = "gguf")]
    GGUFTensor(infa_gguf::GGUFTensor),
}

impl std::ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            #[cfg(feature = "gguf")]
            (Tensor::GGUFTensor(lhs), Tensor::GGUFTensor(rhs)) => Tensor::GGUFTensor(lhs + rhs),
        }
    }
}

impl infa_impl::TensorOps<Tensor> for Tensor {
    fn shape(&self) -> &Vec<u64> {
        match self {
            #[cfg(feature = "gguf")]
            Tensor::GGUFTensor(t) => t.shape(),
        }
    }
}
