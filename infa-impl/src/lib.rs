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

pub trait TensorOps<T>: BaseTensorOps {
    type Item;
    fn item(&self) -> Result<Vec<Self::Item>>;
    fn add(&self, rhs: &T) -> Result<T>;
    fn add_item(&self, rhs: &Self::Item) -> Result<T>;
    fn mul(&self, rhs: &T) -> Result<T>;
    fn sum(&self) -> Result<T>;
    fn size(&self) -> Result<usize>;
}

pub trait BaseTensorOps {
    fn shape(&self) -> &Vec<u64>;
    fn reshape(&self, shape: Vec<u64>) -> Result<Self>
    where
        Self: Sized;
}

pub trait Dequantize<T> {
    fn dequantize(&self) -> Result<T>;
}

impl<T, U> TensorOps<T> for U
where
    T: TensorOps<T>,
    U: Dequantize<T> + BaseTensorOps,
{
    type Item = T::Item;
    fn item(&self) -> Result<Vec<Self::Item>> {
        self.dequantize()?.item()
    }
    fn add(&self, rhs: &T) -> Result<T> {
        self.dequantize()?.add(rhs)
    }
    fn add_item(&self, rhs: &Self::Item) -> Result<T> {
        self.dequantize()?.add_item(rhs)
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
}
