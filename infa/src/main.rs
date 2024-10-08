use std::time::Instant;

use infa::core::FloatTensor;
use infa::gguf::GGUFParser;
use infa::r#impl::TensorOps;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    let ts = FloatTensor::GGUFFloatTensor(gguf.get_tensor("output_norm.weight").unwrap());
    let ts2 = FloatTensor::GGUFFloatTensor(gguf.get_tensor("output_norm.weight").unwrap());
    println!(
        "{:?}",
        (&(&ts2 + &ts).unwrap() + &1.0_f32).unwrap().item().unwrap()
    );
    println!("{}", now.elapsed().as_millis());
}
