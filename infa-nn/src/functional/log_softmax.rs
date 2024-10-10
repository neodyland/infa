use infa_impl::TensorOps;

pub fn log_softmax(
    x: &infa_core::FloatTensor,
    dim: i64,
) -> infa_core::Result<infa_core::FloatTensor> {
    let x = super::softmax(x, dim)?;
    Ok(x.ln()?)
}
