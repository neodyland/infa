use std::{fs::File, io::BufReader, time::Instant};

use infa_safetensors::STParser;

fn main() {
    let now = Instant::now();
    let parser = STParser::new(BufReader::new(File::open("./model.safetensors").unwrap()))
        .parse()
        .unwrap();
    println!("{:?}", parser.header);
    println!("{}", now.elapsed().as_millis());
}
