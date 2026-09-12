use thiserror::Error;

#[derive(Error, Debug)]
pub enum MoonrakerError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("WebSocket error: {0}")]
    WebSocket(Box<tokio_tungstenite::tungstenite::Error>),
    #[error("URL parse error: {0}")]
    Url(#[from] url::ParseError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{operation} rejected: {message}")]
    Rejected { operation: String, message: String },
    #[error("{operation}: Moonraker error {code}: {message}")]
    Api {
        operation: String,
        code: i64,
        message: String,
    },
    #[error("{operation}: outcome unknown; do not automatically retry: {message}")]
    OutcomeUnknown { operation: String, message: String },
    #[error("{0} deadline exceeded")]
    Timeout(&'static str),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl MoonrakerError {
    pub fn rejected(operation: &str, message: &str) -> Self {
        Self::Rejected {
            operation: operation.into(),
            message: message.into(),
        }
    }
    pub fn is_auth(&self) -> bool {
        matches!(
            self,
            Self::Api {
                code: 401 | 403,
                ..
            }
        )
    }
    pub fn outcome_unknown(&self) -> bool {
        matches!(self, Self::OutcomeUnknown { .. })
    }
}
impl From<tokio_tungstenite::tungstenite::Error> for MoonrakerError {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        Self::WebSocket(Box::new(err))
    }
}
