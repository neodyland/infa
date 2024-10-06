pub struct GGUFTensor {
    pub shape: Vec<u64>,
    pub bytes: Box<dyn crate::BaseGGUFBlock>,
    pub data_type: crate::GGMLType,
}

impl GGUFTensor {
    pub fn f32_size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
    pub(crate) fn from_raw_parts<T>(
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
            bytes: Box::new(bytes),
            data_type: data_type.clone(),
        }
    }
}

impl std::ops::Add<GGUFTensor> for GGUFTensor {
    type Output = crate::Result<Self>;

    fn add(self, rhs: GGUFTensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(crate::Error::OpError(format!(
                "Cannot add tensors with different shapes"
            )));
        }
        let rhs = rhs.bytes.to_f32(rhs.f32_size()).unwrap();
        let mut lhs = self.bytes.to_f32(self.f32_size()).unwrap();
        for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = *l + *r;
        }
        Ok(Self {
            shape: self.shape,
            bytes: Box::new(lhs),
            data_type: crate::GGMLType::F32,
        })
    }
}
impl std::ops::Sub<GGUFTensor> for GGUFTensor {
    type Output = crate::Result<Self>;

    fn sub(self, rhs: GGUFTensor) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(crate::Error::OpError(format!(
                "Cannot add tensors with different shapes"
            )));
        }
        let rhs = rhs.bytes.to_f32(rhs.f32_size()).unwrap();
        let mut lhs = self.bytes.to_f32(self.f32_size()).unwrap();
        for (l, r) in lhs.iter_mut().zip(rhs.iter()) {
            *l = *l - *r;
        }
        Ok(Self {
            shape: self.shape,
            bytes: Box::new(lhs),
            data_type: crate::GGMLType::F32,
        })
    }
}

impl infa_impl::TensorOps<GGUFTensor, crate::Error> for GGUFTensor {
    fn shape(&self) -> &Vec<u64> {
        &self.shape
    }
}
