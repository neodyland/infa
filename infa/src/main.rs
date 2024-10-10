use infa::core::FloatTensor;
use infa::nn::functional as F;
use infa::nn::module::Linear;
use infa::r#impl::{BaseTensorOps, TensorOps};

struct Model {
    linear: Linear,
    linear2: Linear,
    linear3: Linear,
}

impl Model {
    fn new() -> infa::Result<Self> {
        let mut rng = infa::rand::thread_rng();
        let linear = Linear {
            weights: FloatTensor::rand(vec![28 * 28, 512], &mut rng)?,
            biases: None,
        };
        let linear2 = Linear {
            weights: FloatTensor::rand(vec![512, 256], &mut rng)?,
            biases: None,
        };
        let linear3 = Linear {
            weights: FloatTensor::rand(vec![256, 10], &mut rng)?,
            biases: None,
        };
        Ok(Self {
            linear,
            linear2,
            linear3,
        })
    }
    fn forward(&self, x: FloatTensor) -> infa::Result<FloatTensor> {
        let x = x.reshape(vec![-1, 28 * 28])?;
        let x = F::relu(&self.linear.forward(x)?)?;
        let x = F::relu(&self.linear2.forward(x)?)?;
        let x = self.linear3.forward(x)?;
        let x = F::log_softmax(&x, 1)?;
        Ok(x)
    }
}

fn main() {
    let x = FloatTensor::of(vec![28 * 28], 2.0).unwrap();
    let model = Model::new().unwrap();
    let y = model.forward(x).unwrap();
    println!("{:?} {:?}", y.shape(), &y.item().unwrap());
}
