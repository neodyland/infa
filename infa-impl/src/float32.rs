#[derive(Debug)]
pub struct Float32Tensor {
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

impl crate::TensorOps<Float32Tensor> for Float32Tensor {
    fn add(self, rhs: Float32Tensor) -> Result<Float32Tensor, crate::Error> {
        if self.shape != rhs.shape {
            return Err(crate::Error::ShapeMismatch(self.shape, rhs.shape));
        }
        let mut result_data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(rhs.data.iter()) {
            result_data.push(a + b);
        }
        Ok(Float32Tensor {
            shape: self.shape,
            data: result_data,
        })
    }

    fn sum(self) -> crate::Result<Float32Tensor> {
        let sum = self.data.iter().sum();
        Ok(Float32Tensor {
            shape: vec![1],
            data: vec![sum],
        })
    }
}

impl crate::BaseTensorOps for Float32Tensor {
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
    fn reshape(&self, shape: Vec<u64>) -> crate::Result<Self> {
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }
}
