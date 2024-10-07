mod ops;
pub use ops::*;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[cfg(feature = "gguf")]
    #[error("GGUF error: {0}")]
    GGUFError(#[from] infa_gguf::Error),
    #[error("Impl error: {0}")]
    ImplError(#[from] infa_impl::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
