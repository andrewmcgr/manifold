//! Crate-wide error type.

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid mesh: {0}")]
    InvalidMesh(String),

    #[error("slicing failed: {0}")]
    Slicing(String),

    #[error("toolpath planning failed: {0}")]
    Toolpath(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("3mf error: {0}")]
    ThreeMf(#[from] lib3mf::Error),
}
