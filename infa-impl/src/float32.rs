#[derive(Debug)]
pub struct Float32Tensor {
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

impl crate::TensorOps<Float32Tensor, f32> for Float32Tensor {
    fn add(&self, rhs: &Float32Tensor) -> Result<Float32Tensor, crate::Error> {
        if self.shape != rhs.shape {
            return Err(crate::Error::ShapeMismatch(
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        let mut result_data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(rhs.data.iter()) {
            result_data.push(a + b);
        }
        Ok(Float32Tensor {
            shape: self.shape.clone(),
            data: result_data,
        })
    }

    fn sum(&self) -> crate::Result<Float32Tensor> {
        let sum = self.data.iter().sum();
        Ok(Float32Tensor {
            shape: vec![1],
            data: vec![sum],
        })
    }

    fn mul(&self, rhs: &Float32Tensor) -> crate::Result<Float32Tensor> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(crate::Error::InvalidShape(
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        if self.shape[1] != rhs.shape[0] {
            return Err(crate::Error::ShapeMismatch(
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        let mut result_data = vec![0.0; (self.shape[0] * rhs.shape[1]) as usize];
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                for k in 0..self.shape[1] {
                    result_data[(i * rhs.shape[1] + j) as usize] += self.data
                        [(i * self.shape[1] + k) as usize]
                        * rhs.data[(k * rhs.shape[1] + j) as usize];
                }
            }
        }
        Ok(Float32Tensor {
            shape: vec![self.shape[0], rhs.shape[1]],
            data: result_data,
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
    fn div(&self, rhs: &Float32Tensor) -> crate::Result<Float32Tensor> {
        if self.shape != rhs.shape {
            return Err(crate::Error::ShapeMismatch(
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        let mut result_data = Vec::with_capacity(self.data.len());
        for (a, b) in self.data.iter().zip(rhs.data.iter()) {
            result_data.push(a / b);
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
