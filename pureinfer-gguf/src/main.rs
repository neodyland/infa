use std::time::Instant;

use pureinfer_gguf::GGUFParser;

fn main() {
    let file = std::io::BufReader::new(std::fs::File::open("./example.gguf").unwrap());
    let now = Instant::now();
    let mut gguf = GGUFParser::new(file).parse().unwrap();
    for name in gguf.tensor_names() {
        if let Ok(tensor) = gguf.get_tensor(&name) {
            if matches!(tensor.data_type, pureinfer_gguf::GGMLType::BF16) {
                let t32 = tensor.bytes.to_f32(tensor.f32_size()).unwrap();
                println!("{}", t32.len());
            }
            println!("{:?}", tensor.data_type);
        }
    }
    println!("{}", now.elapsed().as_millis());
}
