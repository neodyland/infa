use std::ops::{Add, Sub};

pub trait TensorOps<T, E>:
    Add<T, Output = Result<T, E>> + Sub<T, Output = Result<T, E>> + Sized
{
    fn shape(&self) -> &Vec<u64>;
}
