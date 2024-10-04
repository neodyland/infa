use std::time::Instant;

use pureinfer_gguf::GGUFParser;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    for name in gguf.tensor_names() {
        if let Some(tensor) = gguf.get_tensor_meta(&name) {
            println!("{} {:?}", name, tensor.data_type);
        } else {
            println!("Tensor {} not found", name);
        };
    }
    println!("{}", now.elapsed().as_millis());
    println!("{}", now.elapsed().as_millis());
}
