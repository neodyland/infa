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

use std::ops::{Add, Sub};

pub trait TensorOps<'a, T: 'a, E>:
    Add<&'a T, Output = Result<T, E>> + Sub<&'a T, Output = Result<T, E>> + Sized
{
    fn shape(&self) -> Vec<u64>;
    fn reshape(&self, shape: Vec<u64>) -> Result<T, E>;
}
