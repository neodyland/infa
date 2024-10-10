use crate::BaseTensorOps;

#[derive(Debug)]
pub struct Float32Tensor {
    pub shape: Vec<u64>,
    pub data: Vec<f32>,
}

impl crate::TensorOps<Float32Tensor, f32> for Float32Tensor {
    fn sum(&self, dim: i64) -> crate::Result<Float32Tensor> {
        let dim = self.resolve_dim(dim)?;
        if dim == 0 {
            let sum = self.data.iter().sum();
            return Ok(Float32Tensor {
                shape: vec![1],
                data: vec![sum],
            });
        }
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i % self.shape[dim as usize] as usize] += self.data[i];
        }
        let mut shape = self.shape.clone();
        shape[dim as usize] = 1;
        Ok(Float32Tensor { shape, data })
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
        let rhs = if rhs.data.len() == 1 {
            &Float32Tensor {
                shape: self.shape.clone(),
                data: vec![rhs.data[0]; self.size()? as usize],
            }
        } else {
            rhs
        };
        let mut result_data = Vec::with_capacity(self.data.len());
        for (x, y) in self.data.iter().zip(rhs.data.iter()) {
            result_data.push(f(*x, *y));
        }
        Ok(Float32Tensor {
            shape: self.shape.clone(),
            data: result_data,
        })
    }

    fn matmul(&self, rhs: &Float32Tensor) -> crate::Result<Float32Tensor> {
        let ar = self.shape[0] as usize;
        let ac = self.shape[1] as usize;
        let br = rhs.shape[0] as usize;
        let bc = rhs.shape[1] as usize;
        if ac != br {
            return Err(crate::Error::InvalidShape(
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        let mut data = vec![0.0; ar * bc];
        for i in 0..ar {
            for j in 0..bc {
                for k in 0..ac {
                    data[i * bc + j] += self.data[i * ac + k] * rhs.data[k * bc + j];
                }
            }
        }
        let shape = vec![ar as u64, bc as u64];
        Ok(Float32Tensor { shape, data })
    }
}

impl crate::BaseTensorOps for Float32Tensor {
    type Item = f32;
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
    fn reshape(&self, shape: Vec<i64>) -> crate::Result<Self> {
        let shape = self.resolve_shape(shape)?;
        Ok(Float32Tensor {
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
