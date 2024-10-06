use std::ops::Add;

pub trait TensorOps<T>: Add<T> + Sized {
    fn shape(&self) -> &Vec<u64>;
}
