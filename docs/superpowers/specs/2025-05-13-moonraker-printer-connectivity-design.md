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
  - `reqwest = { version = "0.12", default-features = false, features = ["json", "multipart", "rustls-tls", "stream"] }`
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

The public model includes connection state, explicit freshness/readiness and
subscription/job generations, unknown/cancelled print states, filename, separate
display/SD progress availability, durations, approximate remaining time, optional
reported layers/Z/thermals, and Klipper messages. Unknown state never means ready.
`MoonrakerConfig` preserves optional plaintext API-key persistence (redacted Debug)
and the backward-compatible auto-connect field.

#### Shared protocol and session contract (corrected after whole-branch review)

- `MoonrakerClient` validates one normalized HTTP(S) directory endpoint for both
  transports. REST uses X-Api-Key; WS authenticates with correlated
  server.connection.identify (desktop), then readiness and subscription requests.
- Upload returns a typed canonical path/root and Uploaded/Started/Queued/
  StartNotConfirmed outcome decoded from Moonraker's direct response. Upload chunks
  expose client bytes read, not server completion. Controls validate acknowledgments.
- `PrinterSessionHandle` owns cancellable work, offers immutable active endpoint,
  bounded action admission/operation history, fresh telemetry snapshots, atomic
  upload/start reservation, serialized normal actions and independent emergency HTTP.
  A snapshot interface is approved; no unused subscribe stream wrapper is required.
- Disconnect/last-handle drop terminates owned work and prevents reconnection;
  explicit Reconnect may reopen. Already transmitted effects cannot be undone.
- Errors identify the operation, server code/message and uncertainty after possible
  transmission. Never automatically retry a mutating request.

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

1. Validate intent before slicing. A URL alone does not upload. Monitor-only with
   a URL attaches without mesh input, slicing or file writes.
2. Upload only for `--upload`/`--print`; `--print` uses the single combined multipart
   `print=true` request, **never an extra start after upload**.
3. Parse the direct upload outcome. Queued/unconfirmed immediate printing is not
   successful `--print`; report canonical filename and return nonzero.
4. `--print --monitor` binds to that filename and an active-job observation before
   accepting completion. Monitor-only identifies the existing active file. Reject
   `--upload --monitor` without `--print`. Failure/cancellation/auth error exits nonzero;
   Ctrl-C exits monitoring without cancelling the print. Readiness/start/disconnection
   waits are bounded. Progress-based ETA is explicitly approximate or unknown.

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
   - Disabled unless G-code exists and telemetry is fresh and subscribed.
   - Disabled during upload/pending start. Start additionally requires ready Klipper
     and an inactive known job state; fresh server queries enforce guards as well.
2. **Modular Printer Panel (`crates/manifold-gui/src/printer_panel.rs`)**:
   - Collapsible panel (side pane or bottom dock) designed for iterative UI refinement.
   - **Connection header:** Separate draft and active target labels; masked API key,
     explicit auto-connect checkbox, validated settings saved independently of Connect.
     Every profile replacement retires the old session even when auto-connect is off.
   - **Upload bar:** Client-byte-read upload progress separate from print progress;
     operation IDs/results/errors retained for acknowledgment. Bounded repaint while
     session/action exists, including collapsed panel.
   - **Live Job Card (active during print):**
     - Filename and print status pill (`Printing`, `Paused`, etc.).
     - Progress bar with percentage and ETA.
     - Thermals: Nozzle and Bed target/current readouts.
     - Z-height and layer indicators.
   - **Action Bar:**
     - Toggle `Pause` / `Resume`.
     - `Cancel Print` with confirmation modal/dialog.
     - Distinct red `Emergency Stop` for the immutable configured active target,
       independent of WS readiness or normal upload queue; best effort, not hardware safety.
     - Clear cancel confirmation on session/job changes; disconnect is not cancellation.

---

## 5. Error Handling & Resilience

- **Upload Failures:** Categorize network connection drops, HTTP 4xx/5xx, and Moonraker disk full errors with clear user feedback.
- **WebSocket Reconnection:** Automatic retry on disconnect with exponential backoff (1s, 2s, 4s, up to 10s) and state indication (`Reconnecting { attempt }`).
- **Klipper Error Handling:** Extract error messages from Klipper `print_stats.message` and display prominently.
- **Safe State Transitions:** Query fresh Klipper readiness/job state, reject starts
  during active/paused/error/unknown states and protect the active filename from overwrite.
  Do not invent a cold-heater threshold: start G-code normally performs heating.
- **Telemetry semantics:** Replace snapshots on reconnect; handle Klipper lifecycle,
  clear reset fields, advance SD fallback across deltas, and read layers only from
  print_stats.info. toolhead.estimated_print_time is a motion clock, not job ETA.
- **Ownership/deadlines:** Bounded connect/RPC/HTTP/pong deadlines; EOF is explicit;
  all unexpected disconnects use cancellable 1,2,4,8,10-second backoff. Authentication
  errors require correction/reconnect. Never replay mutations automatically.

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
   - Optional operator-authorized validation, read-only first; see `docs/moonraker.md`.
     No real-printer or visual validation is claimed by loopback/headless tests.

The corrected implementation plan replaces defective executable samples with these
contracts and named behavioral regressions. Tests run in release mode; normal GUI
binary unit tests do not inherently launch application main.
