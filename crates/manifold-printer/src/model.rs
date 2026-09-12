use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting { attempt: u32 },
    Error(String),
}

impl Default for ConnectionState {
    fn default() -> Self {
        Self::Disconnected
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PrintState {
    #[default]
    Standby,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        assert_eq!(telemetry.print_state, PrintState::Standby);
        assert_eq!(telemetry.progress_fraction, 0.0);
    }
}
