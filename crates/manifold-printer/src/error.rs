use thiserror::Error;

#[derive(Error, Debug)]
pub enum MoonrakerError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("WebSocket error: {0}")]
    WebSocket(#[from] tokio_tungstenite::tungstenite::Error),

    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Upload error: {0}")]
    UploadFailed(String),

    #[error("Moonraker API error (code {code}): {message}")]
    ApiError { code: i64, message: String },

    #[error("Klipper error: {0}")]
    KlipperError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
