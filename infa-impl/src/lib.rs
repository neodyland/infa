mod int64;
pub use int64::*;
mod int16;
pub use int16::*;
mod int32;
pub use int32::*;
mod int8;
pub use int8::*;
mod float16;
pub use float16::*;
mod float32;
pub use float32::*;
mod float64;
pub use float64::*;
mod bfloat16;
pub use bfloat16::*;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Dequantize error: {0}")]
    DequantizeError(String),
    #[error("Shape mismatch: {0:?} {1:?}")]
    ShapeMismatch(Vec<u64>, Vec<u64>),
    #[error("Invalid shape: {0:?} {1:?}")]
    InvalidShape(Vec<u64>, Vec<u64>),
    #[error("Other error: {0}")]
    OtherError(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait TensorOps<T, I>: BaseTensorOps<Item = I>
where
    I: NumberOps,
{
    fn item(&self) -> Result<Vec<I>>;
    fn add(&self, rhs: &T) -> Result<T>;
    fn add_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(|x| x.add(rhs))
    }
    fn mul(&self, rhs: &T) -> Result<T>;
    fn mul_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(|x| x.mul(rhs))
    }
    fn div_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(|x| x.div(rhs))
    }
    fn div(&self, rhs: &T) -> Result<T>;
    fn sum(&self) -> Result<T>;
    fn size(&self) -> Result<usize>;
    fn dim(&self, dim: i64) -> Result<u64> {
        let shape = self.shape();
        let index = if dim < 0 {
            let index = shape.len() as i64 + dim;
            if index < 0 {
                return Err(Error::InvalidShape(shape.clone(), vec![index as u64]));
            }
            index as usize
        } else {
            dim as usize
        };
        if index >= shape.len() {
            return Err(Error::InvalidShape(shape.clone(), vec![index as u64]));
        }
        Ok(shape[index])
    }
    fn apply(&self, f: impl Fn(Self::Item) -> Self::Item) -> Result<T>;
    fn sqrt(&self) -> Result<T> {
        self.apply(|x| x.sqrt())
    }
    fn tanh(&self) -> Result<T> {
        self.apply(|x| x.tanh())
    }
    fn unary(&self) -> Result<T> {
        self.apply(|x| x.minus())
    }
    fn exp(&self) -> Result<T> {
        self.apply(|x| x.exp())
    }
}

pub trait NumberOps {
    fn zero() -> Self;
    fn one() -> Self;
    fn half() -> Self;
    fn rand(len: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized;
    fn exp(&self) -> Self;
    fn tanh(&self) -> Self;
    fn minus(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn mul(&self, r: &Self) -> Self;
    fn add(&self, r: &Self) -> Self;
    fn div(&self, r: &Self) -> Self;
}

impl NumberOps for f32 {
    #[inline(always)]
    fn div(&self, r: &Self) -> Self {
        self / r
    }
    #[inline(always)]
    fn mul(&self, r: &Self) -> Self {
        self * r
    }
    #[inline(always)]
    fn add(&self, r: &Self) -> Self {
        self + r
    }
    #[inline(always)]
    fn minus(&self) -> Self {
        -self
    }
    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
    #[inline(always)]
    fn exp(&self) -> Self {
        self.exp2()
    }
    #[inline(always)]
    fn tanh(&self) -> Self {
        self.tan()
    }
    #[inline(always)]
    fn half() -> Self {
        0.5
    }
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn one() -> Self {
        1.0
    }
    #[inline(always)]
    fn rand(len: usize, rng: &mut impl rand::Rng) -> Vec<Self> {
        (0..len).map(|_| rng.gen()).collect()
    }
}

pub trait BaseTensorOps
where
    Self::Item: NumberOps + Clone,
{
    type Item;
    fn shape(&self) -> &Vec<u64>;
    fn reshape(&self, shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized;
    fn from_values(shape: Vec<u64>, values: Vec<Self::Item>) -> Result<Self>
    where
        Self: Sized;
    fn zeros(shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized,
    {
        let values = vec![Self::Item::zero(); shape.iter().product::<u64>() as usize];
        Self::from_values(shape, values)
    }
    fn ones(shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized,
    {
        let values = vec![Self::Item::one(); shape.iter().product::<u64>() as usize];
        Self::from_values(shape, values)
    }
    fn of(shape: Vec<u64>, v: Self::Item) -> Result<Self>
    where
        Self: Sized,
    {
        let values = vec![v; shape.iter().product::<u64>() as usize];
        Self::from_values(shape, values)
    }
    fn rand(shape: Vec<u64>, rng: &mut impl rand::Rng) -> Result<Self>
    where
        Self: Sized,
    {
        let size = shape.iter().product::<u64>() as usize;
        Self::from_values(shape, Self::Item::rand(size, rng))
    }
}

pub trait Dequantize<T> {
    fn dequantize(&self) -> Result<T>;
}

impl<T, U, I> TensorOps<T, I> for U
where
    T: TensorOps<T, I>,
    U: Dequantize<T> + BaseTensorOps<Item = I>,
    I: NumberOps + Clone,
{
    fn item(&self) -> Result<Vec<T::Item>> {
        self.dequantize()?.item()
    }
    fn add(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.add(rhs)
    }
    fn mul(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.mul(rhs)
    }
    fn sum(&self) -> Result<T> {
        self.dequantize()?.sum()
    }
    fn size(&self) -> Result<usize> {
        self.dequantize()?.size()
    }
    fn apply(&self, f: impl Fn(Self::Item) -> Self::Item) -> Result<T> {
        self.dequantize()?.apply(f)
    }
    fn div(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.div(rhs)
    }
}
