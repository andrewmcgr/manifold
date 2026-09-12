use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ConnectionState {
    #[default]
    Disconnected,
    Connecting,
    Connected,
    Reconnecting {
        attempt: u32,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PrintState {
    #[default]
    Unknown,
    Standby,
    Cancelled,
    Printing,
    Paused,
    Complete,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureState {
    pub current: f32,
    pub target: f32,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PrinterTelemetry {
    pub connection_state: ConnectionState,
    /// True only for a live, authenticated subscription snapshot.
    pub fresh: bool,
    pub generation: u64,
    pub klippy_ready: bool,
    pub display_progress: Option<f32>,
    pub sd_progress: Option<f32>,
    pub print_state: PrintState,
    pub filename: Option<String>,
    pub progress_fraction: f32,
    pub print_duration_secs: u64,
    pub total_duration_secs: u64,
    pub estimated_remaining_secs: Option<u64>,
    pub toolhead_z: Option<f64>,
    pub current_layer: Option<u32>,
    pub total_layers: Option<u32>,
    pub hotend: Option<TemperatureState>,
    pub bed: Option<TemperatureState>,
    pub klipper_message: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct MoonrakerConfig {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_true")]
    pub auto_connect: bool,
}

fn default_true() -> bool {
    true
}

impl Default for MoonrakerConfig {
    fn default() -> Self {
        Self {
            url: "http://127.0.0.1:7125".to_string(),
            api_key: None,
            auto_connect: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moonraker_config_defaults() {
        let json = r#"{"url": "http://192.168.1.100:7125"}"#;
        let config: MoonrakerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.url, "http://192.168.1.100:7125");
        assert!(config.api_key.is_none());
        assert!(config.auto_connect);
    }

    #[test]
    fn test_printer_telemetry_defaults() {
        let telemetry = PrinterTelemetry::default();
        assert_eq!(telemetry.connection_state, ConnectionState::Disconnected);
        assert_eq!(telemetry.print_state, PrintState::Unknown);
        assert_eq!(telemetry.progress_fraction, 0.0);
    }
}

impl std::fmt::Debug for MoonrakerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoonrakerConfig")
            .field("url", &self.url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("auto_connect", &self.auto_connect)
            .finish()
    }
}

impl PrintState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Printing | Self::Paused)
    }
    pub fn parse(value: &str) -> Self {
        match value {
            "standby" => Self::Standby,
            "printing" => Self::Printing,
            "paused" => Self::Paused,
            "complete" => Self::Complete,
            "cancelled" => Self::Cancelled,
            "error" => Self::Error,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    pub klippy_state: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UploadDisposition {
    Uploaded,
    Started,
    Queued,
    StartNotConfirmed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadOutcome {
    pub path: String,
    pub root: String,
    pub disposition: UploadDisposition,
}
