mod int64;
use core::num;

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
    #[error("Shape mismatch: {0:?} {1:?}")]
    ShapeMismatch_(Vec<u64>, Vec<i64>),
    #[error("Invalid shape: {0:?} {1:?}")]
    InvalidShape(Vec<u64>, Vec<u64>),
    #[error("Invalid shape: {0:?} {1:?}")]
    InvalidShape_(Vec<u64>, Vec<i64>),
    #[error("Other error: {0}")]
    OtherError(String),
    #[error("Invalid dim: {0}")]
    InvalidDimension(i64),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait TensorOps<T, I>: BaseTensorOps<Item = I>
where
    I: NumberOps,
{
    fn matmul(&self, rhs: &T) -> Result<T>;
    fn item(&self) -> Result<Vec<I>>;
    fn max(&self) -> Result<I>
    where
        I: std::cmp::PartialOrd + num_traits::float::FloatCore,
    {
        let mut it = self.item()?;
        if it.len() == 0 {
            return Err(Error::OtherError("What? empty".to_string()));
        }
        for e in it.iter() {
            if e.is_nan() {
                return Err(Error::OtherError("Why is there NaN?".to_string()));
            }
        }
        it.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok((*it.first().ok_or(Error::OtherError("What?".to_string()))?).clone())
    }
    fn log(&self, i: I) -> Result<T>
    where
        I: num_traits::real::Real,
    {
        self.apply(
            #[inline(always)]
            |x| x.log(i),
        )
    }
    fn ln(&self) -> Result<T>
    where
        I: num_traits::real::Real,
    {
        self.apply(
            #[inline(always)]
            |x| x.ln(),
        )
    }
    fn add(&self, rhs: &T) -> Result<T> {
        self.apply_xy(
            rhs,
            #[inline(always)]
            |x, y| x.add(y),
        )
    }
    fn add_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(
            #[inline(always)]
            |x| x.add((*rhs).clone()),
        )
    }
    fn sub(&self, rhs: &T) -> Result<T> {
        self.apply_xy(
            rhs,
            #[inline(always)]
            |x, y| x.sub(y),
        )
    }
    fn sub_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(
            #[inline(always)]
            |x| x.sub((*rhs).clone()),
        )
    }
    fn mul(&self, rhs: &T) -> Result<T> {
        self.apply_xy(
            rhs,
            #[inline(always)]
            |x, y| x.mul(y),
        )
    }
    fn mul_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(
            #[inline(always)]
            |x| x.mul((*rhs).clone()),
        )
    }
    fn div_item(&self, rhs: &Self::Item) -> Result<T> {
        self.apply(
            #[inline(always)]
            |x| x.div((*rhs).clone()),
        )
    }
    fn div(&self, rhs: &T) -> Result<T> {
        self.apply_xy(
            rhs,
            #[inline(always)]
            |x, y| x.div(y),
        )
    }
    fn sum(&self, dim: i64) -> Result<T>;
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
    fn apply_xy(&self, rhs: &T, f: impl Fn(Self::Item, Self::Item) -> Self::Item) -> Result<T>;
    fn sqrt(&self) -> Result<T>
    where
        I: num_traits::real::Real,
    {
        self.apply(
            #[inline(always)]
            |x| x.sqrt(),
        )
    }
    fn tanh(&self) -> Result<T>
    where
        I: num_traits::real::Real,
    {
        self.apply(
            #[inline(always)]
            |x| x.tanh(),
        )
    }
    fn neg(&self) -> Result<T>
    where
        I: std::ops::Neg<Output = I>,
    {
        self.apply(
            #[inline(always)]
            |x| -x,
        )
    }
    fn exp(&self) -> Result<T>
    where
        I: num_traits::real::Real,
    {
        self.apply(
            #[inline(always)]
            |x| x.exp(),
        )
    }
    fn size(&self) -> Result<usize>;
}

pub trait NumberOps: num_traits::Num + Clone {
    fn rand(len: usize, rng: &mut impl rand::Rng) -> Vec<Self>
    where
        Self: Sized;
}

impl NumberOps for f32 {
    #[inline(always)]
    fn rand(len: usize, rng: &mut impl rand::Rng) -> Vec<Self> {
        (0..len).map(|_| rng.gen_range(0.0..1.0)).collect()
    }
}

pub trait BaseTensorOps
where
    Self::Item: NumberOps + Clone,
{
    type Item;
    fn shape(&self) -> &Vec<u64>;
    fn reshape(&self, shape: Vec<i64>) -> Result<Self>
    where
        Self: Sized;
    fn resolve_dim(&self, dim: i64) -> Result<u64> {
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
        Ok(index as u64)
    }
    fn resolve_shape(&self, shape2: Vec<i64>) -> Result<Vec<u64>> {
        let shape = self.shape().clone();
        let mut minus_index = None;
        let size: u64 = shape.iter().product();
        let mut new_shape = vec![0; shape2.len()];
        for (i, a) in shape2.iter().enumerate() {
            if minus_index.is_some() && *a == -1 {
                return Err(Error::InvalidShape(shape.clone(), shape));
            }
            if *a == -1 {
                minus_index = Some(i);
                new_shape[i] = 1;
            } else {
                new_shape[i] = *a as u64;
            }
        }
        if let Some(i) = minus_index {
            new_shape[i] = size / new_shape.iter().product::<u64>();
        }
        Ok(new_shape)
    }
    fn from_values(shape: Vec<u64>, values: Vec<Self::Item>) -> Result<Self>
    where
        Self: Sized;
    fn zeros(shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized,
        Self::Item: num_traits::ConstZero,
    {
        use num_traits::ConstZero;
        let values = vec![Self::Item::ZERO; shape.iter().product::<u64>() as usize];
        Self::from_values(shape, values)
    }
    fn ones(shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized,
        Self::Item: num_traits::ConstOne,
    {
        use num_traits::ConstOne;
        let values = vec![Self::Item::ONE; shape.iter().product::<u64>() as usize];
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
    fn sum(&self, dim: i64) -> Result<T> {
        self.dequantize()?.sum(dim)
    }
    fn apply(&self, f: impl Fn(Self::Item) -> Self::Item) -> Result<T> {
        self.dequantize()?.apply(f)
    }
    fn apply_xy(&self, rhs: &T, f: impl Fn(Self::Item, Self::Item) -> Self::Item) -> Result<T> {
        self.dequantize()?.apply_xy(rhs, f)
    }
    fn div(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.div(rhs)
    }
    fn size(&self) -> Result<usize> {
        self.dequantize()?.size()
    }
    fn matmul(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.matmul(rhs)
    }
}
