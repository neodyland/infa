#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("GGUF error: {0}")]
    GGUF(#[from] infa_gguf::Error),
    #[error("Core error: {0}")]
    Core(#[from] infa_core::Error),
    #[error("Impl error: {0}")]
    Impl(#[from] infa_impl::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
