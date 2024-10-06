use std::time::Instant;

use infa_gguf::GGUFParser;
use infa_impl::TensorOps;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    let ts = gguf.get_tensor("output_norm.weight").unwrap();
    let ts2 = gguf.get_tensor("output_norm.weight").unwrap();
    let tsa = (&(&ts + &(&ts2).reshape(vec![16, 128]).unwrap()).unwrap() - &ts2).unwrap();
    println!("{:?}", tsa.shape);
    println!("{}", now.elapsed().as_millis());
}
