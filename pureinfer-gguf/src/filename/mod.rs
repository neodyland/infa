use once_cell::sync::Lazy;
use regex::Regex;

pub static GGUF_FILENAME_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(include_str!("./re.txt")).unwrap());
