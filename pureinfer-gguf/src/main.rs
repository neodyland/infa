use std::time::Instant;

use pureinfer_gguf::GGUFParser;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./ggml-model-Q4_0.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    for name in gguf.tensor_names() {
        if let Ok(tensor) = gguf.get_tensor(&name) {
            println!("{:?}", tensor.data_type);
        }
    }
    println!("{}", now.elapsed().as_millis());
}
