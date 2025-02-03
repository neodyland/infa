pub trait FloatOps
where
    Self: Sized + Clone,
{
    fn sqrt(&self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.sqrt_()?;
        Ok(res)
    }
    // element-wise square root
    fn sqrt_(&self) -> anyhow::Result<()>;
    fn fill(&self, value: f32) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.fill_(value)?;
        Ok(res)
    }
    // fill with the value
    fn fill_(&self, value: f32) -> anyhow::Result<()>;
    fn rand(&self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.rand_()?;
        Ok(res)
    }
    // fill with random values from 0 to 1
    fn rand_(&self) -> anyhow::Result<()>;
    fn add(&self, other: &Self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.add_(other)?;
        Ok(res)
    }
    // element-wise addition
    fn add_(&self, other: &Self) -> anyhow::Result<()>;
    fn sub(&self, other: &Self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.sub_(other)?;
        Ok(res)
    }
    // element-wise subtraction
    fn sub_(&self, other: &Self) -> anyhow::Result<()>;
    fn mul(&self, other: &Self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.mul_(other)?;
        Ok(res)
    }
    // element-wise multiplication
    fn mul_(&self, other: &Self) -> anyhow::Result<()>;
    fn div(&self, other: &Self) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.div_(other)?;
        Ok(res)
    }
    // element-wise division
    fn div_(&self, other: &Self) -> anyhow::Result<()>;
    fn add_scalar(&self, value: f32) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.add_scalar_(value)?;
        Ok(res)
    }
    // element-wise addition with scalar
    fn add_scalar_(&self, value: f32) -> anyhow::Result<()>;
    fn sub_scalar(&self, value: f32) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.sub_scalar_(value)?;
        Ok(res)
    }
    // element-wise subtraction with scalar
    fn sub_scalar_(&self, value: f32) -> anyhow::Result<()>;
    fn mul_scalar(&self, value: f32) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.mul_scalar_(value)?;
        Ok(res)
    }
    // element-wise multiplication with scalar
    fn mul_scalar_(&self, value: f32) -> anyhow::Result<()>;
    fn div_scalar(&self, value: f32) -> anyhow::Result<Self> {
        let res = self.clone()?;
        res.div_scalar_(value)?;
        Ok(res)
    }
    // element-wise division with scalar
    fn div_scalar_(&self, value: f32) -> anyhow::Result<()>;
    fn matmul(&self, other: &Self, bias: Option<Self>) -> anyhow::Result<Self>;
}

pub trait Free {
    fn free(self) -> anyhow::Result<()>;
}

pub trait Clone
where
    Self: Sized,
{
    fn clone(&self) -> anyhow::Result<Self>;
}

pub trait Display {
    fn display(&self) -> anyhow::Result<String>;
}

pub trait ToF32 {
    type F32;
    // copy the data from self to f32
    fn to_f32(&self, f32: &Self::F32) -> anyhow::Result<()>;
}

pub trait FromF32
where
    Self: Sized,
{
    type F32;
    // copy the data from f32 to self
    fn from_f32(&self, f32: &Self::F32) -> anyhow::Result<()>;
}
