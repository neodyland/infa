use std::time::Instant;

use pureinfer_gguf::GGUFParser;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    println!("{:?}", gguf.tensor_names());
    println!("{}", now.elapsed().as_millis());
    if let Ok(tensor) = gguf.get_tensor("output_norm.weight") {
        println!("{:?}", tensor.shape);
    };
    println!("{}", now.elapsed().as_millis());
}
