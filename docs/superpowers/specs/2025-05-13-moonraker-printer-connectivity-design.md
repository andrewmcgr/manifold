# Moonraker Printer Connectivity Design

**Status:** Approved Design
**Date:** 2025-05-13
**Target Crates:** `crates/manifold-printer` (new), `crates/manifold-cli`, `crates/manifold-gui`

---

## 1. Overview & Goals

Manifold is a non-planar 3D printer slicer that converts meshes into G-code toolpaths. Currently, users slice meshes and must manually export G-code to disk and upload it to their printer via an external browser interface (such as Mainsail, Fluidd, or OctoPrint).

This feature introduces native Moonraker integration to Manifold:

- **Direct Upload:** Upload generated G-code directly to a Moonraker-managed printer from both the GUI and CLI.
- **One-Click Print:** Option to upload and immediately start printing.
- **Real-Time Telemetry & Monitoring:** Live tracking of print state, progress percentage, remaining time, current layer/Z-height, and hotend/bed temperatures via Moonraker's WebSocket notifications.
- **Essential Print Controls:** Safe Pause, Resume, Cancel (with confirmation), and Emergency Stop actions.
- **Profile Persistence:** Persist printer connection endpoints and API keys alongside machine and slicer profiles.

---

## 2. Architecture & Crate Boundaries

Per `AGENTS.md` and `CODE_STYLE.md`, `manifold-core` is strictly reserved for slicing domain logic and must have no UI or network dependencies. To maintain clean boundaries, all networking and Moonraker protocol implementation is encapsulated in a new workspace crate:

```text
crates/
├── manifold-core       (pure slicing domain logic, zero network I/O)
├── manifold-printer    (NEW: Moonraker HTTP + WebSocket API client & state types)
├── manifold-cli        (CLI entrypoint: slices, uploads, optionally monitors)
└── manifold-gui        (egui/wgpu frontend: profile config, upload button, live printer panel)
```

### Dependencies

- **`crates/manifold-printer`**:
  - `reqwest = { version = "0.12", default-features = false, features = ["json", "multipart", "rustls-tls"] }`
  - `tokio = { version = "1", features = ["rt-multi-thread", "sync", "time", "macros"] }`
  - `tokio-tungstenite = { version = "0.24", features = ["rustls-tls-native-roots"] }`
  - `serde = { version = "1", features = ["derive"] }`
  - `serde_json = "1"`
  - `thiserror = "2"`
  - `tracing = "0.1"`
  - `url = "2"`
- **`crates/manifold-cli`**:
  - `manifold-printer = { path = "../manifold-printer" }`
  - `indicatif = "0.17"` (terminal progress bar for `--monitor`)
- **`crates/manifold-gui`**:
  - `manifold-printer = { path = "../manifold-printer" }`
  - `crossbeam-channel = "0.5"` (or standard `std::sync::mpsc`) for UI <-> background worker communication

---

## 3. Detailed Component Design

### 3.1. `manifold-printer` Crate

#### Data Types & Telemetry Models

```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting { attempt: u32 },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PrintState {
    #[default]
    Standby,
    Printing,
    Paused,
    Complete,
    Error,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TemperatureState {
    pub current: f32,
    pub target: f32,
}

#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct PrinterTelemetry {
    pub connection_state: ConnectionState,
    pub print_state: PrintState,
    pub filename: Option<String>,
    pub progress_fraction: f32,            // 0.0 ..= 1.0
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MoonrakerConfig {
    pub url: String,
    pub api_key: Option<String>,
    #[serde(default = "default_true")]
    pub auto_connect: bool,
}

fn default_true() -> bool {
    true
}
```

#### Low-Level Async Client (`MoonrakerClient`)

Implements REST and raw WebSocket connections:

- `check_connection(&self) -> Result<ServerInfo, MoonrakerError>`
- `upload_gcode(&self, filename: &str, gcode_bytes: Vec<u8>, start_print: bool, progress: Option<mpsc::Sender<f32>>) -> Result<(), MoonrakerError>`
- `start_print(&self, filename: &str) -> Result<(), MoonrakerError>`
- `pause_print(&self) -> Result<(), MoonrakerError>`
- `resume_print(&self) -> Result<(), MoonrakerError>`
- `cancel_print(&self) -> Result<(), MoonrakerError>`
- `emergency_stop(&self) -> Result<(), MoonrakerError>`
- `subscribe_objects(&self) -> Result<impl Stream<Item = Result<StatusNotification, MoonrakerError>>, MoonrakerError>`

#### High-Level Worker Session (`PrinterSessionHandle`)

Designed for non-blocking UI integration:

- Spawns a Tokio background thread managing the WebSocket loop, keepalive pings, reconnects with exponential backoff (1s to 10s), and REST action dispatches.
- Exposes:
  - `command_tx: Sender<PrinterCommand>`
  - `telemetry_rx: Receiver<PrinterTelemetry>`
  - `upload_progress_rx: Receiver<f32>`

---

## 4. Front-End Integrations

### 4.1. CLI (`manifold-cli`)

CLI flags added to `Cli`:

```text
PRINTER OPTIONS:
  --printer-url <URL>        Moonraker endpoint (e.g. http://192.168.1.100:7125)
  --printer-api-key <KEY>    Optional API key for Moonraker authentication
  --upload                   Upload the generated G-code file to the printer
  --print                    Upload and immediately start printing
  --monitor                  Tail live progress, temperatures, and state in the terminal until completion
```

**Flow:**

1. Slice objects to G-code.
2. If `--upload` or `--print` is set, instantiate `MoonrakerClient` and upload with multipart stream.
3. If `--print` is set, trigger `start_print`.
4. If `--monitor` is requested, establish the WebSocket subscription and render an `indicatif` progress bar showing progress, ETA, and temperatures until the job completes or fails.

### 4.2. GUI (`manifold-gui`)

#### Profile Schema Evolution

`crates/manifold-gui/src/profile.rs` adds an optional `MoonrakerConfig`:

```rust
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Profile {
    pub machine: Machine,
    pub config: SlicerConfig,
    #[serde(default)]
    pub moonraker: Option<MoonrakerConfig>,
}
```

Existing `profile.json` files continue loading safely due to `#[serde(default)]`.

#### UI Components

1. **Toolbar Actions (Top Pane)**:
   - Next to "Slice" and "Export…", add "Upload" and "Upload & Print" buttons.
   - Disabled unless `gcode.is_some()` AND `printer.is_connected()`.
   - Disabled while upload is actively inflight to prevent double-submissions.
2. **Modular Printer Panel (`crates/manifold-gui/src/printer_panel.rs`)**:
   - Collapsible panel (side pane or bottom dock) designed for iterative UI refinement.
   - **Connection header:** URL text input, API key input, Connect/Disconnect button, status badge (green/yellow/red).
   - **Upload bar:** Active upload progress bar when transferring G-code.
   - **Live Job Card (active during print):**
     - Filename and print status pill (`Printing`, `Paused`, etc.).
     - Progress bar with percentage and ETA.
     - Thermals: Nozzle and Bed target/current readouts.
     - Z-height and layer indicators.
   - **Action Bar:**
     - Toggle `Pause` / `Resume`.
     - `Cancel Print` with confirmation modal/dialog.
     - Distinct red `Emergency Stop` button.

---

## 5. Error Handling & Resilience

- **Upload Failures:** Categorize network connection drops, HTTP 4xx/5xx, and Moonraker disk full errors with clear user feedback.
- **WebSocket Reconnection:** Automatic retry on disconnect with exponential backoff (1s, 2s, 4s, up to 10s) and state indication (`Reconnecting { attempt }`).
- **Klipper Error Handling:** Extract error messages from Klipper `print_stats.message` and display prominently.
- **Safe State Transitions:** Guard against starting a print while another job is active or while heaters are in an unhandled state.

---

## 6. Testing Strategy

1. **Unit Tests in `manifold-printer`:**
   - JSON deserialization of Moonraker REST responses (`server/info`, `server/files/upload`).
   - JSON-RPC delta notification parsing (`notify_status_update`) for `print_stats`, `toolhead`, `extruder`, `heater_bed`, `virtual_sdcard`.
   - Telemetry aggregator state transitions.
2. **Mock Server Integration Tests:**
   - Test `upload_gcode`, `start_print`, and WebSocket subscriptions using a local HTTP mock server (e.g. `wiremock`) and mock WebSocket stream.
3. **GUI Profile Tests:**
   - Verify serialization/deserialization backward compatibility of `Profile` with and without `MoonrakerConfig`.
4. **Manual & Local Printer Testing:**
   - Validation against a real local Moonraker instance or Dockerized Klipper/Moonraker emulator.
