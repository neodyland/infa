#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Magic mismatch")]
    MagicMismatch,
    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Utf8 error: {0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
    #[error("Invalid GGUF metadata value type: {0}")]
    InvalidGGUFMetadataValueType(u32),
}
pub(crate) type Result<T> = std::result::Result<T, Error>;
