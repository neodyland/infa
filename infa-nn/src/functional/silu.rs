use infa_impl::TensorOps;

pub fn silu(v: &infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
    Ok((v / &((&(-v)?.exp()? + &1.0)?))?)
}
