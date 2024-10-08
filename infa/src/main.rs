use infa::core::FloatTensor;
use infa::nn::module::Linear;
use infa::r#impl::{BaseTensorOps, TensorOps};

struct Model {
    linear: Linear,
}

impl Model {
    fn new() -> infa::Result<Self> {
        let mut rng = infa::rand::thread_rng();
        let linear = Linear {
            weights: FloatTensor::rand(vec![1, 2], &mut rng)?,
            biases: Some(FloatTensor::rand(vec![2, 2], &mut rng)?),
        };
        Ok(Self { linear })
    }
    fn forward(&self, x: FloatTensor) -> infa::Result<FloatTensor> {
        Ok(self.linear.forward(x)?)
    }
}

fn main() {
    let x = FloatTensor::of(vec![2, 1], 2.0).unwrap();
    let model = Model::new().unwrap();
    let y = model.forward(x).unwrap();
    println!("{:?} {:?}", y.shape(), &y.item().unwrap());
}
