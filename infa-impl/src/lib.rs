use std::ops::{Add, Sub};

pub trait TensorOps<'a, T: 'a, E>:
    Add<&'a T, Output = Result<T, E>> + Sub<&'a T, Output = Result<T, E>> + Sized
{
    fn shape(&self) -> Vec<u64>;
}
