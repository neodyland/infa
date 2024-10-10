use infa_impl::TensorOps;

pub fn log_softmax(x: &infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
    let x = x.ln()?;
    let x = super::softmax(&x)?;
    Ok(x)
}
