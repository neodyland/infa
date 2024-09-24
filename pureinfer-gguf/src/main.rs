use pureinfer_gguf::GGUFParser;

fn main() {
    let b = std::fs::read("./example.gguf").unwrap();
    let mut parser = GGUFParser::from_bytes(&b);
    let header = parser.read_header().unwrap();
    println!("{:?}", header);
}
