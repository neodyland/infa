use infa_impl::TensorOps;

pub struct Linear {
    pub weights: infa_core::FloatTensor,
    pub biases: Option<infa_core::FloatTensor>,
}

impl Linear {
    pub fn forward(&self, x: infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
        let mut output = x.matmul(&self.weights)?;
        if let Some(biases) = &self.biases {
            output = (&output + biases)?;
        }
        Ok(output)
    }
}
