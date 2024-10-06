use std::ops::Add;

pub trait TensorOps<T, E>: Add<T, Output = Result<T, E>> + Sized {
    fn shape(&self) -> &Vec<u64>;
}
