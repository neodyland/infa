#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Io error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serde json error: {0}")]
    SerdeJsonError(#[from] serde_json::Error),
}
pub(crate) type Result<T> = std::result::Result<T, Error>;
