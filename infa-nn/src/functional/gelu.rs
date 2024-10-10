use infa_impl::TensorOps;

const SQRT_TWO_OVER_PI_F32: f32 = 0.79788456080286535587989211986876373;

#[inline]
pub fn gelu(v: &infa_core::FloatTensor) -> infa_core::Result<infa_core::FloatTensor> {
    Ok((&(v * &0.5)?
        * &(&(&(v * &SQRT_TWO_OVER_PI_F32)? * &(&(&(v * v)? * &0.044715)? + &1.0)?)?.tanh()?
            + &1.0)?)?)
}
