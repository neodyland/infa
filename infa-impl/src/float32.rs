#[derive(Debug)]
pub struct Float32Tensor {
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

impl crate::TensorOps<Float32Tensor, f32> for Float32Tensor {
    fn sum(&self) -> crate::Result<Float32Tensor> {
        let sum = self.data.iter().sum();
        Ok(Float32Tensor {
            shape: vec![1],
            data: vec![sum],
        })
    }
    fn item(&self) -> crate::Result<Vec<Self::Item>> {
        Ok(self.data.clone())
    }
    fn size(&self) -> crate::Result<usize> {
        Ok(self.data.len())
    }
    fn apply(&self, f: impl Fn(Self::Item) -> Self::Item) -> crate::Result<Float32Tensor> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for item in self.data.iter() {
            result_data.push(f(*item));
        }
        Ok(Float32Tensor {
            shape: self.shape.clone(),
            data: result_data,
        })
    }
    fn apply_xy(
        &self,
        rhs: &Float32Tensor,
        f: impl Fn(Self::Item, Self::Item) -> Self::Item,
    ) -> crate::Result<Float32Tensor> {
        let mut result_data = Vec::with_capacity(self.data.len());
        for (x, y) in self.data.iter().zip(rhs.data.iter()) {
            result_data.push(f(*x, *y));
        }
        Ok(Float32Tensor {
            shape: self.shape.clone(),
            data: result_data,
        })
    }
}

impl crate::BaseTensorOps for Float32Tensor {
    type Item = f32;
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
    fn reshape(&self, shape: Vec<u64>) -> crate::Result<Self> {
        Ok(Self {
            shape,
            data: self.data.clone(),
        })
    }
    fn from_values(shape: Vec<u64>, values: Vec<Self::Item>) -> crate::Result<Self> {
        let size: u64 = shape.iter().product();
        if size != values.len() as u64 {
            return Err(crate::Error::InvalidShape(
                shape.clone(),
                vec![values.len() as u64],
            ));
        }
        Ok(Self {
            shape,
            data: values,
        })
    }
}
