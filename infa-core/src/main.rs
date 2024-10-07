use std::time::Instant;

use infa_core::FloatTensor;
use infa_gguf::GGUFParser;
use infa_impl::{BaseTensorOps, TensorOps};

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    let ts = FloatTensor::GGUFFloatTensor(gguf.get_tensor("output_norm.weight").unwrap());
    let ts2 = FloatTensor::GGUFFloatTensor(gguf.get_tensor("output_norm.weight").unwrap());
    println!("{:?}", (ts2 * ts).unwrap().sum().unwrap().shape());
    println!("{}", now.elapsed().as_millis());
}
