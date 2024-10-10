use infa_core::FloatTensor;

pub struct LayerNorm {
    pub weight: FloatTensor,
    pub bias: Option<FloatTensor>,
    pub remove_mean: bool,
    pub eps: f64,
}

impl LayerNorm {
    pub fn forward(&self, x: &FloatTensor) -> infa_core::Result<FloatTensor> {
        unimplemented!()
    }
}
