use infa_impl::TensorOps;

pub fn softmax(x: &infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
    let exp_x = (x - &x.max()?)?;
    let exp_x = exp_x.exp()?;
    Ok((&exp_x / &exp_x.sum(0)?)?)
}
