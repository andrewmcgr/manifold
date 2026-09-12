# Moonraker Printer Connectivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide native 3D printer connectivity to Moonraker/Klipper from Manifold, enabling direct G-code upload, one-click print starting, real-time WebSocket telemetry monitoring (progress, thermals, layers, ETA), and essential job controls (pause, resume, cancel, emergency stop) in both GUI and CLI.

**Architecture:** A new workspace crate `crates/manifold-printer` encapsulates Moonraker HTTP REST and JSON-RPC WebSocket client logic, protocol parsing, and an aggregated telemetry state machine. `manifold-cli` uses this crate for `--upload`, `--print`, and `--monitor` CLI flags. `manifold-gui` integrates `MoonrakerConfig` into its profile system, runs a background worker thread via channels, adds top-toolbar upload actions, and renders a modular, collapsible printer control and telemetry panel.

**Tech Stack:** Rust (edition 2021), `reqwest` (HTTP multipart/REST), `tokio` (async runtime), `tokio-tungstenite` (WebSocket), `serde`/`serde_json`, `thiserror`, `indicatif` (CLI progress), `egui`/`eframe` (GUI).

**Spec:** `docs/superpowers/specs/2025-05-13-moonraker-printer-connectivity-design.md`

## Global Constraints

- Slicing domain logic in `manifold-core` must remain strictly untouched by network I/O.
- `manifold-printer` uses `thiserror::Error` for library errors; `manifold-cli` and `manifold-gui` consume them with `anyhow::Result` at the application boundary.
- Serialization compatibility: existing `profile.json` files must continue loading without error when `moonraker` is absent (`#[serde(default)]`).
- GUI UI thread must never block on network operations or WebSocket event loops.
- Follow Rust workspace checks: `cargo fmt --all` -> `cargo clippy --workspace --all-targets` -> `cargo test --workspace`.

---

### File Structure Map

- **`Cargo.toml`**: Register `crates/manifold-printer` in workspace members.
- **`crates/manifold-printer/`** (New crate):
  - `Cargo.toml`: Declare dependencies (`reqwest`, `tokio`, `tokio-tungstenite`, `serde`, `serde_json`, `thiserror`, `tracing`, `url`).
  - `src/lib.rs`: Re-export public types, error enum, client, and session.
  - `src/error.rs`: `MoonrakerError` enum with `thiserror`.
  - `src/model.rs`: Core types: `ConnectionState`, `PrintState`, `TemperatureState`, `PrinterTelemetry`, `MoonrakerConfig`, Moonraker JSON-RPC payload structures.
  - `src/client.rs`: `MoonrakerClient` providing REST calls (`upload_gcode`, `start_print`, `pause_print`, `resume_print`, `cancel_print`, `emergency_stop`) and WebSocket connection.
  - `src/session.rs`: `PrinterSessionHandle`, background thread event loop, exponential backoff reconnects, state aggregation, action command dispatch.
  - `tests/client_tests.rs`: Integration tests against mock HTTP and WebSocket endpoints.
- **`crates/manifold-cli/`**:
  - `Cargo.toml`: Add `manifold-printer` and `indicatif`.
  - `src/main.rs`: Add CLI arguments (`--printer-url`, `--printer-api-key`, `--upload`, `--print`, `--monitor`), orchestrate slicing -> uploading -> monitoring.
- **`crates/manifold-gui/`**:
  - `Cargo.toml`: Add `manifold-printer`.
  - `src/profile.rs`: Add `MoonrakerConfig` to `Profile`.
  - `src/printer_panel.rs`: Modular egui panel rendering connection state, live thermals, progress, and controls.
  - `src/app.rs`: Wire `PrinterSessionHandle`, top toolbar buttons ("Upload", "Upload & Print"), and render `printer_panel`.

---

### Task 1: Scaffold `manifold-printer` Crate & Define Core Domain Models

**Files:**

- Modify: `Cargo.toml:1-10`
- Create: `crates/manifold-printer/Cargo.toml`
- Create: `crates/manifold-printer/src/lib.rs`
- Create: `crates/manifold-printer/src/error.rs`
- Create: `crates/manifold-printer/src/model.rs`

**Interfaces:**

- Produces:
  - `MoonrakerError`: `enum` covering HTTP, WebSocket, JSON parsing, URL, and Klipper errors.
  - `ConnectionState`: `enum { Disconnected, Connecting, Connected, Reconnecting { attempt: u32 }, Error(String) }`
  - `PrintState`: `enum { Standby, Printing, Paused, Complete, Error }`
  - `TemperatureState`: `struct { current: f32, target: f32 }`
  - `PrinterTelemetry`: `struct { connection_state, print_state, filename, progress_fraction, print_duration_secs, total_duration_secs, estimated_remaining_secs, toolhead_z, current_layer, total_layers, hotend, bed, klipper_message }`
  - `MoonrakerConfig`: `struct { url: String, api_key: Option<String>, auto_connect: bool }`

- [ ] **Step 1: Write unit tests for models and deserialization**

Create `crates/manifold-printer/src/model.rs` with tests verifying `MoonrakerConfig` defaults and JSON deserialization of Moonraker status messages:

```rust
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
```

- [ ] **Step 2: Run test to verify it fails (crate doesn't exist yet)**

Run: `cargo test -p manifold-printer`
Expected: FAIL (cannot find crate or compile)

- [ ] **Step 3: Implement workspace registration, `Cargo.toml`, `error.rs`, `model.rs`, and `lib.rs`**

Update `Cargo.toml` to add `"crates/manifold-printer"` to `workspace.members`.

Create `crates/manifold-printer/Cargo.toml`:

```toml
[package]
name = "manifold-printer"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
description = "Moonraker API client and telemetry integration for the Manifold slicer"

[dependencies]
reqwest = { version = "0.12", default-features = false, features = ["json", "multipart", "rustls-tls"] }
tokio = { version = "1", features = ["rt-multi-thread", "sync", "time", "macros"] }
tokio-tungstenite = { version = "0.24", features = ["rustls-tls-native-roots"] }
futures-util = { version = "0.3", default-features = false, features = ["sink", "std"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
url = "2"

[dev-dependencies]
tokio = { version = "1", features = ["full"] }
```

Create `crates/manifold-printer/src/error.rs`:

```rust
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
```

Create `crates/manifold-printer/src/model.rs`:

```rust
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
```

Create `crates/manifold-printer/src/lib.rs`:

```rust
pub mod error;
pub mod model;

pub use error::MoonrakerError;
pub use model::{
    ConnectionState, MoonrakerConfig, PrintState, PrinterTelemetry, TemperatureState,
};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-printer`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/manifold-printer
git commit -m "feat(printer): scaffold manifold-printer crate and core domain models"
```

---

### Task 2: Implement REST API Client in `manifold-printer`

**Files:**

- Create: `crates/manifold-printer/src/client.rs`
- Modify: `crates/manifold-printer/src/lib.rs`
- Create: `crates/manifold-printer/tests/rest_tests.rs`

**Interfaces:**

- Produces:
  - `MoonrakerClient`:
    - `new(config: MoonrakerConfig) -> Result<Self, MoonrakerError>`
    - `check_connection(&self) -> Result<(), MoonrakerError>`
    - `upload_gcode(&self, filename: &str, gcode_bytes: Vec<u8>, start_print: bool) -> Result<(), MoonrakerError>`
    - `start_print(&self, filename: &str) -> Result<(), MoonrakerError>`
    - `pause_print(&self) -> Result<(), MoonrakerError>`
    - `resume_print(&self) -> Result<(), MoonrakerError>`
    - `cancel_print(&self) -> Result<(), MoonrakerError>`
    - `emergency_stop(&self) -> Result<(), MoonrakerError>`

- [ ] **Step 1: Write integration test with mockito / wiremock or hyper local mock**

Add `wiremock = "0.6"` to `crates/manifold-printer/Cargo.toml` under `[dev-dependencies]`.
Create `crates/manifold-printer/tests/rest_tests.rs`:

```rust
use manifold_printer::{MoonrakerClient, MoonrakerConfig};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_check_connection_success() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/server/info"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "result": { "klippy_state": "ready" }
        })))
        .mount(&mock_server)
        .await;

    let config = MoonrakerConfig {
        url: mock_server.uri(),
        api_key: None,
        auto_connect: false,
    };
    let client = MoonrakerClient::new(config).unwrap();
    assert!(client.check_connection().await.is_ok());
}

#[tokio::test]
async fn test_upload_gcode_and_start_print() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/server/files/upload"))
        .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
            "result": { "item": { "path": "test.gcode" } }
        })))
        .mount(&mock_server)
        .await;

    let config = MoonrakerConfig {
        url: mock_server.uri(),
        api_key: None,
        auto_connect: false,
    };
    let client = MoonrakerClient::new(config).unwrap();
    let result = client
        .upload_gcode("test.gcode", b"G28\nG1 Z10\n".to_vec(), true)
        .await;
    assert!(result.is_ok());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test rest_tests -p manifold-printer`
Expected: FAIL (`MoonrakerClient` not implemented)

- [ ] **Step 3: Implement `MoonrakerClient` in `crates/manifold-printer/src/client.rs`**

```rust
use crate::error::MoonrakerError;
use crate::model::MoonrakerConfig;
use reqwest::multipart::{Form, Part};
use reqwest::{Client, StatusCode};
use url::Url;

#[derive(Debug, Clone)]
pub struct MoonrakerClient {
    config: MoonrakerConfig,
    base_url: Url,
    http: Client,
}

impl MoonrakerClient {
    pub fn new(config: MoonrakerConfig) -> Result<Self, MoonrakerError> {
        let mut base_url = Url::parse(&config.url)?;
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path().trim_end_matches('/')));
        }
        let http = Client::builder().build()?;
        Ok(Self {
            config,
            base_url,
            http,
        })
    }

    pub fn config(&self) -> &MoonrakerConfig {
        &self.config
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn apply_auth(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req = req.header("X-Api-Key", key);
        }
        req
    }

    pub async fn check_connection(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("server/info")?;
        let req = self.apply_auth(self.http.get(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Server returned status {}",
                resp.status()
            )))
        }
    }

    pub async fn upload_gcode(
        &self,
        filename: &str,
        gcode_bytes: Vec<u8>,
        start_print: bool,
    ) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("server/files/upload")?;
        let part = Part::bytes(gcode_bytes)
            .file_name(filename.to_string())
            .mime_str("application/octet-stream")
            .map_err(|e| MoonrakerError::UploadFailed(e.to_string()))?;

        let mut form = Form::new().part("file", part);
        if start_print {
            form = form.text("print", "true");
        }

        let req = self.apply_auth(self.http.post(url).multipart(form));
        let resp = req.send().await?;

        if resp.status().is_success() || resp.status() == StatusCode::CREATED {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(MoonrakerError::UploadFailed(format!(
                "Upload failed (status {status}): {body}"
            )))
        }
    }

    pub async fn start_print(&self, filename: &str) -> Result<(), MoonrakerError> {
        let url = self
            .base_url
            .join(&format!("printer/print/start?filename={filename}"))?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Start print failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn pause_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/pause")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Pause failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn resume_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/resume")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Resume failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn cancel_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/cancel")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Cancel failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn emergency_stop(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/emergency_stop")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Emergency stop failed: {}",
                resp.status()
            )))
        }
    }
}
```

Update `crates/manifold-printer/src/lib.rs` to export `MoonrakerClient`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test rest_tests -p manifold-printer`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-printer
git commit -m "feat(printer): implement Moonraker HTTP REST client"
```

---

### Task 3: Implement JSON-RPC WebSocket Subscription & Telemetry Aggregator

**Files:**

- Create: `crates/manifold-printer/src/subscription.rs`
- Modify: `crates/manifold-printer/src/lib.rs`
- Create: `crates/manifold-printer/tests/subscription_tests.rs`

**Interfaces:**

- Produces:
  - `SubscriptionHandler`:
    - Handles WebSocket handshakes, subscribing to `print_stats`, `toolhead`, `extruder`, `heater_bed`, `display_status`, `virtual_sdcard`.
    - Parses `notify_status_update` deltas and updates `PrinterTelemetry`.
    - Maps Klipper print state strings (`"printing"` -> `PrintState::Printing`, `"paused"` -> `PrintState::Paused`, `"complete"` -> `PrintState::Complete`, `"error"` -> `PrintState::Error`, `"standby"` -> `PrintState::Standby`).

- [ ] **Step 1: Write unit tests for telemetry delta updating**

Create `crates/manifold-printer/tests/subscription_tests.rs`:

```rust
use manifold_printer::model::{PrintState, PrinterTelemetry};
use manifold_printer::subscription::apply_status_delta;

#[test]
fn test_apply_status_delta_temperatures_and_progress() {
    let mut telemetry = PrinterTelemetry::default();
    let delta = serde_json::json!({
        "extruder": { "temperature": 215.2, "target": 215.0 },
        "heater_bed": { "temperature": 60.1, "target": 60.0 },
        "display_status": { "progress": 0.45 },
        "print_stats": {
            "state": "printing",
            "filename": "benchy.gcode",
            "print_duration": 120.0,
            "total_duration": 140.0
        },
        "toolhead": {
            "position": [100.0, 100.0, 12.4, 45.0]
        }
    });

    apply_status_delta(&mut telemetry, &delta);

    assert_eq!(telemetry.print_state, PrintState::Printing);
    assert_eq!(telemetry.filename.as_deref(), Some("benchy.gcode"));
    assert!((telemetry.progress_fraction - 0.45).abs() < 1e-4);
    assert_eq!(telemetry.print_duration_secs, 120);
    assert_eq!(telemetry.total_duration_secs, 140);
    assert_eq!(telemetry.toolhead_z, Some(12.4));
    assert_eq!(telemetry.hotend.as_ref().unwrap().current, 215.2);
    assert_eq!(telemetry.hotend.as_ref().unwrap().target, 215.0);
    assert_eq!(telemetry.bed.as_ref().unwrap().current, 60.1);
    assert_eq!(telemetry.bed.as_ref().unwrap().target, 60.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test subscription_tests -p manifold-printer`
Expected: FAIL (`apply_status_delta` not defined)

- [ ] **Step 3: Implement status delta updates and WebSocket subscription payloads**

Create `crates/manifold-printer/src/subscription.rs`:

```rust
use crate::model::{PrintState, PrinterTelemetry, TemperatureState};
use serde_json::Value;

pub fn build_subscribe_request(id: u64) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": "printer.objects.subscribe",
        "params": {
            "objects": {
                "print_stats": null,
                "display_status": null,
                "toolhead": null,
                "extruder": null,
                "heater_bed": null,
                "virtual_sdcard": null
            }
        },
        "id": id
    })
}

pub fn apply_status_delta(telemetry: &mut PrinterTelemetry, delta: &Value) {
    if let Some(extruder) = delta.get("extruder") {
        let current = extruder.get("temperature").and_then(|v| v.as_f64()).map(|v| v as f32);
        let target = extruder.get("target").and_then(|v| v.as_f64()).map(|v| v as f32);
        if let (Some(cur), Some(tgt)) = (current, target) {
            telemetry.hotend = Some(TemperatureState { current: cur, target: tgt });
        } else if let Some(ref mut hotend) = telemetry.hotend {
            if let Some(c) = current { hotend.current = c; }
            if let Some(t) = target { hotend.target = t; }
        }
    }

    if let Some(bed) = delta.get("heater_bed") {
        let current = bed.get("temperature").and_then(|v| v.as_f64()).map(|v| v as f32);
        let target = bed.get("target").and_then(|v| v.as_f64()).map(|v| v as f32);
        if let (Some(cur), Some(tgt)) = (current, target) {
            telemetry.bed = Some(TemperatureState { current: cur, target: tgt });
        } else if let Some(ref mut bed_state) = telemetry.bed {
            if let Some(c) = current { bed_state.current = c; }
            if let Some(t) = target { bed_state.target = t; }
        }
    }

    if let Some(display) = delta.get("display_status") {
        if let Some(progress) = display.get("progress").and_then(|v| v.as_f64()) {
            telemetry.progress_fraction = progress.clamp(0.0, 1.0) as f32;
        }
    }

    if let Some(stats) = delta.get("print_stats") {
        if let Some(state_str) = stats.get("state").and_then(|v| v.as_str()) {
            telemetry.print_state = match state_str.to_lowercase().as_str() {
                "printing" => PrintState::Printing,
                "paused" => PrintState::Paused,
                "complete" => PrintState::Complete,
                "error" => PrintState::Error,
                _ => PrintState::Standby,
            };
        }
        if let Some(filename) = stats.get("filename").and_then(|v| v.as_str()) {
            if !filename.is_empty() {
                telemetry.filename = Some(filename.to_string());
            }
        }
        if let Some(dur) = stats.get("print_duration").and_then(|v| v.as_f64()) {
            telemetry.print_duration_secs = dur.max(0.0) as u64;
        }
        if let Some(total) = stats.get("total_duration").and_then(|v| v.as_f64()) {
            telemetry.total_duration_secs = total.max(0.0) as u64;
        }
        if let Some(msg) = stats.get("message").and_then(|v| v.as_str()) {
            telemetry.klipper_message = if msg.is_empty() { None } else { Some(msg.to_string()) };
        }
    }

    if let Some(toolhead) = delta.get("toolhead") {
        if let Some(pos) = toolhead.get("position").and_then(|v| v.as_array()) {
            if pos.len() >= 3 {
                if let Some(z) = pos[2].as_f64() {
                    telemetry.toolhead_z = Some(z);
                }
            }
        }
        if let Some(est) = toolhead.get("estimated_print_time").and_then(|v| v.as_f64()) {
            if est > 0.0 && telemetry.print_duration_secs > 0 {
                let remaining = (est as u64).saturating_sub(telemetry.print_duration_secs);
                telemetry.estimated_remaining_secs = Some(remaining);
            }
        }
    }

    if let Some(sdcard) = delta.get("virtual_sdcard") {
        if telemetry.progress_fraction == 0.0 {
            if let Some(progress) = sdcard.get("progress").and_then(|v| v.as_f64()) {
                telemetry.progress_fraction = progress.clamp(0.0, 1.0) as f32;
            }
        }
    }
}
```

Update `crates/manifold-printer/src/lib.rs` to export `subscription`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test subscription_tests -p manifold-printer`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-printer
git commit -m "feat(printer): implement JSON-RPC subscription parsing and telemetry aggregation"
```

---

### Task 4: Implement High-Level Background Worker (`PrinterSessionHandle`)

**Files:**

- Create: `crates/manifold-printer/src/session.rs`
- Modify: `crates/manifold-printer/src/lib.rs`
- Create: `crates/manifold-printer/tests/session_tests.rs`

**Interfaces:**

- Produces:
  - `PrinterAction`:
    - `UploadAndPrint { filename: String, gcode: Vec<u8> }`
    - `UploadOnly { filename: String, gcode: Vec<u8> }`
    - `Pause`
    - `Resume`
    - `Cancel`
    - `EmergencyStop`
    - `Disconnect`
    - `Reconnect`
  - `PrinterSessionHandle`:
    - `spawn(config: MoonrakerConfig) -> Self`
    - `send_action(&self, action: PrinterAction) -> Result<(), ...>`
    - `latest_telemetry(&self) -> PrinterTelemetry`
    - `upload_in_progress(&self) -> bool`

- [ ] **Step 1: Write integration test for `PrinterSessionHandle`**

Create `crates/manifold-printer/tests/session_tests.rs`:

```rust
use manifold_printer::model::{ConnectionState, MoonrakerConfig};
use manifold_printer::session::{PrinterAction, PrinterSessionHandle};
use std::time::Duration;

#[tokio::test]
async fn test_printer_session_spawns_and_handles_disconnect() {
    let config = MoonrakerConfig {
        url: "http://127.0.0.1:9".to_string(), // Unreachable port
        api_key: None,
        auto_connect: true,
    };
    let session = PrinterSessionHandle::spawn(config);
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Should indicate connecting or reconnecting, without panicking
    let telemetry = session.latest_telemetry();
    assert!(matches!(
        telemetry.connection_state,
        ConnectionState::Connecting
            | ConnectionState::Reconnecting { .. }
            | ConnectionState::Error(_)
    ));

    session.send_action(PrinterAction::Disconnect).unwrap();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test session_tests -p manifold-printer`
Expected: FAIL (`PrinterSessionHandle` not implemented)

- [ ] **Step 3: Implement `PrinterSessionHandle` and the background Tokio event loop in `session.rs`**

Implement connection management, WebSocket stream reading, sending pings, exponential backoff reconnects, action dispatching, and thread-safe telemetry snapshot sharing via `Arc<RwLock<PrinterTelemetry>>`.

Create `crates/manifold-printer/src/session.rs`:

```rust
use crate::client::MoonrakerClient;
use crate::error::MoonrakerError;
use crate::model::{ConnectionState, MoonrakerConfig, PrinterTelemetry};
use crate::subscription::{apply_status_delta, build_subscribe_request};
use futures_util::{SinkExt, StreamExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};
use url::Url;

#[derive(Debug)]
pub enum PrinterAction {
    UploadAndPrint { filename: String, gcode: Vec<u8> },
    UploadOnly { filename: String, gcode: Vec<u8> },
    Pause,
    Resume,
    Cancel,
    EmergencyStop,
    Disconnect,
    Reconnect,
}

#[derive(Clone)]
pub struct PrinterSessionHandle {
    action_tx: mpsc::UnboundedSender<PrinterAction>,
    telemetry: Arc<RwLock<PrinterTelemetry>>,
    uploading: Arc<AtomicBool>,
}

impl PrinterSessionHandle {
    pub fn spawn(config: MoonrakerConfig) -> Self {
        let (action_tx, action_rx) = mpsc::unbounded_channel();
        let telemetry = Arc::new(RwLock::new(PrinterTelemetry::default()));
        let uploading = Arc::new(AtomicBool::new(false));

        let tele_clone = Arc::clone(&telemetry);
        let up_clone = Arc::clone(&uploading);

        std::thread::Builder::new()
            .name("manifold-printer-worker".to_string())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!("Failed to create printer Tokio runtime: {e}");
                        return;
                    }
                };

                rt.block_on(run_worker(config, action_rx, tele_clone, up_clone));
            })
            .expect("Failed to spawn printer worker thread");

        Self {
            action_tx,
            telemetry,
            uploading,
        }
    }

    pub fn send_action(&self, action: PrinterAction) -> Result<(), mpsc::error::SendError<PrinterAction>> {
        self.action_tx.send(action)
    }

    pub fn latest_telemetry(&self) -> PrinterTelemetry {
        self.telemetry.read().unwrap().clone()
    }

    pub fn is_uploading(&self) -> bool {
        self.uploading.load(Ordering::Relaxed)
    }
}

async fn run_worker(
    config: MoonrakerConfig,
    mut action_rx: mpsc::UnboundedReceiver<PrinterAction>,
    telemetry: Arc<RwLock<PrinterTelemetry>>,
    uploading: Arc<AtomicBool>,
) {
    let client = match MoonrakerClient::new(config.clone()) {
        Ok(c) => c,
        Err(e) => {
            if let Ok(mut t) = telemetry.write() {
                t.connection_state = ConnectionState::Error(e.to_string());
            }
            return;
        }
    };

    let mut auto_reconnect = config.auto_connect;
    let mut reconnect_attempt = 0;

    let ws_url = {
        let mut u = match Url::parse(&config.url) {
            Ok(u) => u,
            Err(e) => {
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Error(e.to_string());
                }
                return;
            }
        };
        let ws_scheme = if u.scheme() == "https" { "wss" } else { "ws" };
        let _ = u.set_scheme(ws_scheme);
        match u.join("websocket") {
            Ok(ws_u) => ws_u,
            Err(e) => {
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Error(e.to_string());
                }
                return;
            }
        }
    };

    'main_loop: loop {
        if !auto_reconnect {
            if let Ok(mut t) = telemetry.write() {
                t.connection_state = ConnectionState::Disconnected;
            }
            while let Some(action) = action_rx.recv().await {
                match action {
                    PrinterAction::Reconnect => {
                        auto_reconnect = true;
                        reconnect_attempt = 0;
                        break;
                    }
                    _ => warn!("Ignored action while disconnected: {action:?}"),
                }
            }
        }

        if let Ok(mut t) = telemetry.write() {
            if reconnect_attempt == 0 {
                t.connection_state = ConnectionState::Connecting;
            } else {
                t.connection_state = ConnectionState::Reconnecting {
                    attempt: reconnect_attempt,
                };
            }
        }

        let ws_stream_res = connect_async(ws_url.as_str()).await;
        let mut ws_stream = match ws_stream_res {
            Ok((stream, _)) => {
                reconnect_attempt = 0;
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Connected;
                }
                stream
            }
            Err(err) => {
                reconnect_attempt += 1;
                let backoff_secs = (1u64 << reconnect_attempt.min(4)).min(10);
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Reconnecting {
                        attempt: reconnect_attempt,
                    };
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(backoff_secs)) => {
                        continue 'main_loop;
                    }
                    Some(action) = action_rx.recv() => {
                        match action {
                            PrinterAction::Disconnect => {
                                auto_reconnect = false;
                                continue 'main_loop;
                            }
                            _ => {}
                        }
                    }
                }
                continue 'main_loop;
            }
        };

        // Subscribe to objects
        let sub_req = build_subscribe_request(1);
        let _ = ws_stream.send(Message::Text(sub_req.to_string())).await;

        let mut ping_interval = tokio::time::interval(Duration::from_secs(15));

        'ws_loop: loop {
            tokio::select! {
                _ = ping_interval.tick() => {
                    if let Err(e) = ws_stream.send(Message::Ping(vec![].into())).await {
                        warn!("WebSocket ping failed: {e}");
                        break 'ws_loop;
                    }
                }
                Some(msg_res) = ws_stream.next() => {
                    match msg_res {
                        Ok(Message::Text(text)) => {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                                if val.get("method").and_then(|v| v.as_str()) == Some("notify_status_update") {
                                    if let Some(params) = val.get("params").and_then(|v| v.as_array()) {
                                        if let Some(delta) = params.first() {
                                            if let Ok(mut t) = telemetry.write() {
                                                apply_status_delta(&mut t, delta);
                                            }
                                        }
                                    }
                                } else if let Some(res) = val.get("result").and_then(|v| v.get("status")) {
                                    if let Ok(mut t) = telemetry.write() {
                                        apply_status_delta(&mut t, res);
                                    }
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            info!("WebSocket closed by server");
                            break 'ws_loop;
                        }
                        Err(e) => {
                            warn!("WebSocket stream error: {e}");
                            break 'ws_loop;
                        }
                        _ => {}
                    }
                }
                Some(action) = action_rx.recv() => {
                    match action {
                        PrinterAction::UploadAndPrint { filename, gcode } => {
                            let client = client.clone();
                            let uploading = Arc::clone(&uploading);
                            tokio::spawn(async move {
                                uploading.store(true, Ordering::Relaxed);
                                let res = client.upload_gcode(&filename, gcode, true).await;
                                uploading.store(false, Ordering::Relaxed);
                                if let Err(e) = res {
                                    error!("Upload & Print failed: {e}");
                                }
                            });
                        }
                        PrinterAction::UploadOnly { filename, gcode } => {
                            let client = client.clone();
                            let uploading = Arc::clone(&uploading);
                            tokio::spawn(async move {
                                uploading.store(true, Ordering::Relaxed);
                                let res = client.upload_gcode(&filename, gcode, false).await;
                                uploading.store(false, Ordering::Relaxed);
                                if let Err(e) = res {
                                    error!("Upload Only failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Pause => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.pause_print().await {
                                    error!("Pause failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Resume => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.resume_print().await {
                                    error!("Resume failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Cancel => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.cancel_print().await {
                                    error!("Cancel failed: {e}");
                                }
                            });
                        }
                        PrinterAction::EmergencyStop => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.emergency_stop().await {
                                    error!("Emergency stop failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Disconnect => {
                            auto_reconnect = false;
                            break 'ws_loop;
                        }
                        PrinterAction::Reconnect => {
                            auto_reconnect = true;
                            break 'ws_loop;
                        }
                    }
                }
            }
        }
    }
}
```

Update `crates/manifold-printer/src/lib.rs` to re-export `session` types:

```rust
pub mod client;
pub mod error;
pub mod model;
pub mod session;
pub mod subscription;

pub use client::MoonrakerClient;
pub use error::MoonrakerError;
pub use model::{
    ConnectionState, MoonrakerConfig, PrintState, PrinterTelemetry, TemperatureState,
};
pub use session::{PrinterAction, PrinterSessionHandle};
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test session_tests -p manifold-printer`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-printer
git commit -m "feat(printer): implement PrinterSessionHandle background worker"
```

---

### Task 5: Add Printer CLI Options to `manifold-cli`

**Files:**

- Modify: `crates/manifold-cli/Cargo.toml`
- Modify: `crates/manifold-cli/src/main.rs`

**Interfaces:**

- Consumes: `manifold_printer::{MoonrakerClient, MoonrakerConfig, PrintState, ConnectionState}`
- Produces: CLI flags `--printer-url`, `--printer-api-key`, `--upload`, `--print`, `--monitor`.

- [ ] **Step 1: Write CLI argument parsing unit test**

In `crates/manifold-cli/src/main.rs`:

```rust
#[test]
fn test_cli_printer_args() {
    use clap::Parser;
    let args = vec![
        "manifold",
        "model.stl",
        "--printer-url",
        "http://192.168.1.50:7125",
        "--upload",
        "--print",
        "--monitor",
    ];
    let cli = Cli::parse_from(args);
    assert_eq!(cli.printer_url.as_deref(), Some("http://192.168.1.50:7125"));
    assert!(cli.upload);
    assert!(cli.print);
    assert!(cli.monitor);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-cli -- test_cli_printer_args`
Expected: FAIL (unrecognized arguments)

- [ ] **Step 3: Update `crates/manifold-cli/Cargo.toml` and implement CLI workflow in `main.rs`**

Add dependencies to `crates/manifold-cli/Cargo.toml`:

```toml
manifold-printer = { path = "../manifold-printer" }
indicatif = "0.17"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time"] }
```

In `crates/manifold-cli/src/main.rs`:
Add fields to `Cli`:

```rust
    /// Moonraker printer URL (e.g. http://192.168.1.50:7125)
    #[arg(long)]
    printer_url: Option<String>,

    /// Moonraker API key (if authentication is enabled)
    #[arg(long)]
    printer_api_key: Option<String>,

    /// Upload the sliced Gcode to the printer via Moonraker
    #[arg(long)]
    upload: bool,

    /// Start printing immediately after upload
    #[arg(long)]
    print: bool,

    /// Tail print progress and temperatures in the terminal until completion
    #[arg(long)]
    monitor: bool,
```

Implement the upload and monitor block after slicing and G-code generation:

```rust
    if cli.upload || cli.print || cli.printer_url.is_some() {
        if let Some(ref url) = cli.printer_url {
            let config = manifold_printer::MoonrakerConfig {
                url: url.clone(),
                api_key: cli.printer_api_key.clone(),
                auto_connect: true,
            };
            let client = manifold_printer::MoonrakerClient::new(config)?;
            let filename = cli.output.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("out.gcode");

            info!("Uploading Gcode to Moonraker ({url})...");
            client.upload_gcode(filename, gcode.as_bytes().to_vec(), cli.print).await?;
            info!("Upload successful!");

            if cli.monitor {
                info!("Monitoring print job...");
                let session = manifold_printer::PrinterSessionHandle::spawn(client.config().clone());
                let pb = indicatif::ProgressBar::new(100);
                pb.set_style(
                    indicatif::ProgressStyle::default_bar()
                        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% ({eta}) {msg}")?
                        .progress_chars("#>-")
                );

                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    let t = session.latest_telemetry();
                    let pct = (t.progress_fraction * 100.0) as u64;
                    pb.set_position(pct);

                    let temp_str = match (&t.hotend, &t.bed) {
                        (Some(h), Some(b)) => format!("E:{:.0}/{:.0}°C B:{:.0}/{:.0}°C", h.current, h.target, b.current, b.target),
                        _ => String::new(),
                    };
                    pb.set_message(format!("{:?} {}", t.print_state, temp_str));

                    if matches!(t.print_state, manifold_printer::PrintState::Complete | manifold_printer::PrintState::Error) {
                        pb.finish_with_message(format!("Finished: {:?}", t.print_state));
                        break;
                    }
                }
            }
        } else {
            bail!("--upload or --print was passed, but no --printer-url was provided");
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-cli`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-cli
git commit -m "feat(cli): add Moonraker --upload, --print, and --monitor flags"
```

---

### Task 6: Integrate `MoonrakerConfig` into `manifold-gui` Profiles

**Files:**

- Modify: `crates/manifold-gui/Cargo.toml`
- Modify: `crates/manifold-gui/src/profile.rs`

**Interfaces:**

- Consumes: `manifold_printer::MoonrakerConfig`
- Produces: `Profile.moonraker: Option<MoonrakerConfig>` with backward compatibility.

- [ ] **Step 1: Write profile backward-compatibility test**

In `crates/manifold-gui/src/profile.rs`:

```rust
#[test]
fn test_profile_deserialization_without_moonraker() {
    let json = r#"{
        "machine": {
            "build_volume": {
                "Aabb": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [200.0, 200.0, 200.0]
                }
            },
            "tools": [{"id": 0, "nozzle_diameter": 0.4}]
        },
        "config": {}
    }"#;
    let profile: Profile = serde_json::from_str(json).unwrap();
    assert!(profile.moonraker.is_none());
}

#[test]
fn test_profile_roundtrip_with_moonraker() {
    let mut profile = sample_profile();
    profile.moonraker = Some(manifold_printer::MoonrakerConfig {
        url: "http://voron.local:7125".to_string(),
        api_key: Some("secret123".to_string()),
        auto_connect: true,
    });
    let json = serde_json::to_string_pretty(&profile).unwrap();
    let loaded: Profile = serde_json::from_str(&json).unwrap();
    assert_eq!(profile, loaded);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-gui -- test_profile_deserialization_without_moonraker`
Expected: FAIL (profile.moonraker doesn't exist)

- [ ] **Step 3: Update `crates/manifold-gui/Cargo.toml` and `profile.rs`**

Add `manifold-printer` to `crates/manifold-gui/Cargo.toml`:

```toml
manifold-printer = { path = "../manifold-printer" }
```

In `crates/manifold-gui/src/profile.rs`:

```rust
use manifold_printer::MoonrakerConfig;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Profile {
    pub machine: Machine,
    pub config: SlicerConfig,
    #[serde(default)]
    pub moonraker: Option<MoonrakerConfig>,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-gui -- test_profile`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-gui
git commit -m "feat(gui): integrate MoonrakerConfig into settings profiles"
```

---

### Task 7: Implement Modular `printer_panel` in `manifold-gui`

**Files:**

- Create: `crates/manifold-gui/src/printer_panel.rs`
- Modify: `crates/manifold-gui/src/main.rs` (or `lib.rs` / `app.rs`)

**Interfaces:**

- Consumes: `manifold_printer::{PrinterSessionHandle, PrinterTelemetry, ConnectionState, PrintState, PrinterAction}`
- Produces: `PrinterPanel` component with `ui(&mut self, ui: &mut egui::Ui, session: &mut Option<PrinterSessionHandle>)`

- [ ] **Step 1: Write test for `PrinterPanel` state struct**

Create `crates/manifold-gui/src/printer_panel.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_printer_panel_defaults() {
        let panel = PrinterPanel::default();
        assert!(panel.collapsed);
        assert!(!panel.confirming_cancel);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p manifold-gui -- test_printer_panel_defaults`
Expected: FAIL (`PrinterPanel` not defined)

- [ ] **Step 3: Implement `PrinterPanel`**

In `crates/manifold-gui/src/printer_panel.rs`:

```rust
use egui::{Color32, RichText, Ui};
use manifold_printer::{
    ConnectionState, MoonrakerConfig, PrintState, PrinterAction, PrinterSessionHandle,
};

#[derive(Default)]
pub struct PrinterPanel {
    pub collapsed: bool,
    pub confirming_cancel: bool,
    pub url_input: String,
    pub api_key_input: String,
}

impl PrinterPanel {
    pub fn new(config: Option<&MoonrakerConfig>) -> Self {
        Self {
            collapsed: false,
            confirming_cancel: false,
            url_input: config.map(|c| c.url.clone()).unwrap_or_else(|| "http://127.0.0.1:7125".to_string()),
            api_key_input: config.and_then(|c| c.api_key.clone()).unwrap_or_default(),
        }
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        session_opt: &mut Option<PrinterSessionHandle>,
        on_save_config: impl FnOnce(MoonrakerConfig),
    ) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let header_text = if self.collapsed { "▶ Printer" } else { "▼ Printer" };
                if ui.button(RichText::new(header_text).strong()).clicked() {
                    self.collapsed = !self.collapsed;
                }

                if let Some(session) = session_opt.as_ref() {
                    let tele = session.latest_telemetry();
                    match &tele.connection_state {
                        ConnectionState::Connected => {
                            ui.colored_label(Color32::from_rgb(0, 200, 0), "● Connected");
                        }
                        ConnectionState::Connecting => {
                            ui.colored_label(Color32::YELLOW, "◌ Connecting...");
                        }
                        ConnectionState::Reconnecting { attempt } => {
                            ui.colored_label(Color32::YELLOW, format!("◌ Reconnecting ({attempt})..."));
                        }
                        ConnectionState::Disconnected => {
                            ui.colored_label(Color32::GRAY, "○ Disconnected");
                        }
                        ConnectionState::Error(err) => {
                            ui.colored_label(Color32::RED, format!("⚠ Error: {err}"));
                        }
                    }
                } else {
                    ui.colored_label(Color32::GRAY, "○ Disconnected");
                }
            });

            if self.collapsed {
                return;
            }

            ui.separator();

            // Connection settings
            ui.horizontal(|ui| {
                ui.label("URL:");
                ui.text_edit_singleline(&mut self.url_input);
                if session_opt.is_none() {
                    if ui.button("Connect").clicked() {
                        let config = MoonrakerConfig {
                            url: self.url_input.clone(),
                            api_key: if self.api_key_input.is_empty() { None } else { Some(self.api_key_input.clone()) },
                            auto_connect: true,
                        };
                        on_save_config(config.clone());
                        *session_opt = Some(PrinterSessionHandle::spawn(config));
                    }
                } else if ui.button("Disconnect").clicked() {
                    if let Some(session) = session_opt.take() {
                        let _ = session.send_action(PrinterAction::Disconnect);
                    }
                }
            });

            if let Some(session) = session_opt.as_ref() {
                let tele = session.latest_telemetry();
                if matches!(tele.connection_state, ConnectionState::Connected) {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(format!("Status: {:?}", tele.print_state));
                        if let Some(ref name) = tele.filename {
                            ui.label(format!("File: {name}"));
                        }
                    });

                    // Progress bar
                    let progress = tele.progress_fraction;
                    ui.add(egui::ProgressBar::new(progress).show_percentage());

                    // Thermals & Z
                    ui.horizontal(|ui| {
                        if let Some(ref h) = tele.hotend {
                            ui.label(format!("Hotend: {:.1} / {:.0} °C", h.current, h.target));
                        }
                        if let Some(ref b) = tele.bed {
                            ui.label(format!("Bed: {:.1} / {:.0} °C", b.current, b.target));
                        }
                        if let Some(z) = tele.toolhead_z {
                            ui.label(format!("Z: {:.2} mm", z));
                        }
                    });

                    // Controls
                    ui.horizontal(|ui| {
                        match tele.print_state {
                            PrintState::Printing => {
                                if ui.button("⏸ Pause").clicked() {
                                    let _ = session.send_action(PrinterAction::Pause);
                                }
                            }
                            PrintState::Paused => {
                                if ui.button("▶ Resume").clicked() {
                                    let _ = session.send_action(PrinterAction::Resume);
                                }
                            }
                            _ => {}
                        }

                        if matches!(tele.print_state, PrintState::Printing | PrintState::Paused) {
                            if !self.confirming_cancel {
                                if ui.button("⏹ Cancel").clicked() {
                                    self.confirming_cancel = true;
                                }
                            } else {
                                ui.colored_label(Color32::RED, "Sure?");
                                if ui.button("Yes, Cancel").clicked() {
                                    let _ = session.send_action(PrinterAction::Cancel);
                                    self.confirming_cancel = false;
                                }
                                if ui.button("No").clicked() {
                                    self.confirming_cancel = false;
                                }
                            }
                        }

                        if ui.button(RichText::new("⛔ Emergency Stop").color(Color32::RED)).clicked() {
                            let _ = session.send_action(PrinterAction::EmergencyStop);
                        }
                    });
                }
            }
        });
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p manifold-gui -- test_printer_panel`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-gui
git commit -m "feat(gui): implement modular PrinterPanel component"
```

---

### Task 8: Wire Printer Actions into `manifold-gui` Toolbar and App Loop

**Files:**

- Modify: `crates/manifold-gui/src/app.rs`
- Modify: `crates/manifold-gui/src/main.rs`

**Interfaces:**

- Consumes: `PrinterPanel`, `PrinterSessionHandle`, `PrinterAction`
- Produces: Top toolbar buttons ("Upload", "Upload & Print") and docked `printer_panel` view.

- [ ] **Step 1: Write unit / compilation test in `manifold-gui`**

Verify `App` holds `printer_session: Option<PrinterSessionHandle>` and `printer_panel: PrinterPanel`.

- [ ] **Step 2: Run `cargo check -p manifold-gui` to verify failure before adding fields**

- [ ] **Step 3: Wire `printer_session` and `printer_panel` into `App`**

In `crates/manifold-gui/src/app.rs`:

- Add fields to `App`:

  ```rust
  printer_session: Option<manifold_printer::PrinterSessionHandle>,
  printer_panel: crate::printer_panel::PrinterPanel,
  ```

- Initialize in `App::new()`:
  - If loaded profile has `moonraker: Some(ref config)` with `auto_connect: true`, spawn `PrinterSessionHandle`.
  - Initialize `printer_panel: PrinterPanel::new(profile.moonraker.as_ref())`.
- In top toolbar next to "Slice" and "Export…":
  - Add button `"Upload to Printer"`
  - Add button `"Upload & Print"`
  - Enabled when `self.gcode.is_some() && self.printer_session.as_ref().map_or(false, |s| s.latest_telemetry().connection_state == manifold_printer::ConnectionState::Connected) && !self.printer_session.as_ref().map_or(false, |s| s.is_uploading())`.
  - Clicking dispatches `PrinterAction::UploadOnly` or `PrinterAction::UploadAndPrint` with the current G-code.
- In side or bottom panel:
  - Render `self.printer_panel.show(ui, &mut self.printer_session, |cfg| { ... update profile ... });`

- [ ] **Step 4: Verify workspace builds and tests pass**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/manifold-gui
git commit -m "feat(gui): wire Moonraker upload and printer panel into main application"
```

---

### Task 9: Full Workspace Lint, Format, and Verification Gate

**Files:**

- All touched files across workspace.

- [ ] **Step 1: Run format check**

Run: `cargo fmt --all -- --check`
If formatting needed: `cargo fmt --all`

- [ ] **Step 2: Run clippy across all workspace targets**

Run: `cargo clippy --workspace --all-targets`
Expected: Zero warnings or errors.

- [ ] **Step 3: Run full workspace test suite**

Run: `cargo test --workspace`
Expected: 100% tests pass.

- [ ] **Step 4: Commit any cleanup**

```bash
git add -u
git commit -m "chore: format and clean workspace lints for Moonraker integration"
```
