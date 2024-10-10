use infa_impl::TensorOps;

#[inline]
pub fn relu(v: &infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
    Ok(v.apply(|x| if x > 0.0 { x } else { 0.0 })?)
}
