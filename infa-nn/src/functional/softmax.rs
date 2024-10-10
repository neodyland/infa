use infa_impl::TensorOps;

pub fn softmax(x: &infa_core::FloatTensor, dim: i64) -> infa_core::Result<infa_core::FloatTensor> {
    let exp_x = (x - &x.max()?)?.exp()?;
    Ok((&exp_x / &exp_x.sum(dim)?)?)
}
