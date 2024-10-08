use infa::core::FloatTensor;
use infa::gguf::GGUFParser;
use infa::nn::module::Linear;
use infa::r#impl::{BaseTensorOps, Float32Tensor, TensorOps};

struct Model {
    linear: Linear,
}

impl Model {
    fn new() -> infa::Result<Self> {
        let linear = Linear {
            weights: FloatTensor::Float32Tensor(Float32Tensor::new(vec![1, 2048], 1.0)?),
            biases: Some(FloatTensor::Float32Tensor(Float32Tensor::new(
                vec![2048, 2048],
                1.0,
            )?)),
        };
        Ok(Self { linear })
    }
    fn forward(&self, x: FloatTensor) -> infa::Result<FloatTensor> {
        Ok(self.linear.forward(x)?)
    }
}

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    let x = FloatTensor::GGUFFloatTensor(gguf.get_tensor("output_norm.weight").unwrap());
    let model = Model::new().unwrap();
    let x = x.reshape(vec![2048, 1]).unwrap();
    let y = model.forward(x).unwrap();
    println!("{:?} {:?}", y.shape(), &y.item().unwrap()[..10]);
}
