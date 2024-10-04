use crate::GGUFMetadataValue;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Magic mismatch")]
    MagicMismatch,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Utf8 error: {0}")]
    FromUtf8Error(#[from] std::string::FromUtf8Error),
    #[error("Invalid GGUF metadata value type: {0}")]
    InvalidGGUFMetadataValueType(u32),
    #[error("Invalid GGML type: {0}")]
    InvalidGGMLType(u32),
    #[error("Invalid alignment metaType: {0:?}")]
    InvalidAlignmentMetaType(GGUFMetadataValue),
    #[error("No such tensor: {0}")]
    NoSuchTensor(String),
}
pub(crate) type Result<T> = std::result::Result<T, Error>;
