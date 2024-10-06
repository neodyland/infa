use std::time::Instant;

use infa_gguf::GGUFParser;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    let ts = gguf.get_tensor("output_norm.weight").unwrap();
    let ts2 = gguf.get_tensor("output_norm.weight").unwrap();
    let tsa = (&(&ts + &ts2).unwrap() - &ts2).unwrap();
    println!("{:?}", tsa.shape);
    println!("{}", now.elapsed().as_millis());
}
