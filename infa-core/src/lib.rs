mod ops;
pub use ops::*;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "gguf")]
    #[error("GGUF Error: {0}")]
    GGUFError(#[from] infa_gguf::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
