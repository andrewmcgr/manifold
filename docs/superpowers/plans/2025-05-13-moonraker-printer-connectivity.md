# Moonraker Printer Connectivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide native 3D printer connectivity to Moonraker/Klipper from Manifold, enabling direct G-code upload, one-click print starting, real-time WebSocket telemetry monitoring (progress, thermals, layers, ETA), and GUI job controls (pause, resume, cancel, emergency stop), with upload/print/monitor in both front ends.

**Architecture:** A new workspace crate `crates/manifold-printer` encapsulates Moonraker HTTP REST and JSON-RPC WebSocket client logic, protocol parsing, and an aggregated telemetry state machine. `manifold-cli` uses this crate for `--upload`, `--print`, and `--monitor` CLI flags. `manifold-gui` integrates `MoonrakerConfig` into its profile system, runs a background worker thread via channels, adds top-toolbar upload actions, and renders a modular, collapsible printer control and telemetry panel.

**Tech Stack:** Rust (edition 2021), `reqwest` (HTTP multipart/REST), `tokio` (async runtime), `tokio-tungstenite` (WebSocket), `serde`/`serde_json`, `thiserror`, `indicatif` (CLI progress), `egui`/`eframe` (GUI).

**Spec:** `docs/superpowers/specs/2025-05-13-moonraker-printer-connectivity-design.md`

## Global Constraints

- Slicing domain logic in `manifold-core` must remain strictly untouched by network I/O.
- `manifold-printer` uses `thiserror::Error` for library errors; `manifold-cli` and `manifold-gui` consume them with `anyhow::Result` at the application boundary.
- Serialization compatibility: existing `profile.json` files must continue loading without error when `moonraker` is absent (`#[serde(default)]`).
- GUI UI thread must never block on network operations or WebSocket event loops.
- Follow Rust workspace checks: `cargo fmt --all` -> `cargo clippy --workspace --all-targets` -> `cargo test --workspace --release`.

---

## Corrected execution contract (whole-branch review I1–I15, M1–M2)

The original copy-paste implementations have been removed. They authenticated only
REST, detached uncancellable work, defaulted unknown state to standby, discarded
upload responses, uploaded merely because a URL was configured, and used Klipper's
motion clock as ETA. **Do not restore those samples.** The approved behavior and
these corrections supersede them. No real printer actions are part of automated
verification. Public API spelling is subordinate to these contracts.

### Task 1: Models and errors

- [x] Keep slicing/network separation. `manifold-printer` owns protocol/session types.
- [x] Model unknown and cancelled print states explicitly. Snapshot freshness,
  subscription generation, job generation, progress-source availability and Klipper
  readiness must be explicit; default is unknown, not ready/standby.
- [x] Keep optional plaintext profile API-key compatibility, mask GUI input and
  redact credential Debug/error output. No keys in URL queries or redirects.
- [x] Use typed upload and operation outcomes; distinguish rejection, server error
  and outcome-unknown after possible transmission.

References: `src/model.rs`, `src/error.rs`, `src/operation.rs` under
`crates/manifold-printer`; `config_debug_redacts_key_and_rejects_credentials_in_url`.

### Task 2: Shared endpoint and HTTP protocol

- [x] Validate an HTTP(S) directory endpoint without embedded credentials, query
  or fragment; derive HTTP and WS paths from the same normalization, including
  reverse-proxy prefixes with or without trailing slash. Keep TLS verification.
- [x] REST uses `X-Api-Key`, bounded deadlines and no credential-forwarding redirects.
- [x] Query fresh server readiness and print state before upload/start. Reject
  starts while printing, paused, error or unknown, and uploads overwriting the
  active file. Cold heaters alone are not a blocker; start G-code normally heats.
- [x] Upload G-code in chunks with byte-read progress, multipart `root=gcodes`,
  `file`, and optional string `print=true`. Upload currently accepts a basename.
- [x] Decode the documented **direct** HTTP 201 object with `item.path`, `item.root`,
  `print_started`, `print_queued`. Preserve the canonical name and distinguish
  Uploaded, Started, Queued and StartNotConfirmed. Invalid/missing response fields
  after transmission mean unknown outcome. Never send an extra start automatically.
- [x] Encode public `start_print` filenames using the query serializer exactly once.
- [x] Preserve operation/code/message on structured server failures; do not classify
  pause/resume/cancel/stop errors as upload errors. Validate control acknowledgments.

References: `client.rs`; behavioral fixtures in `tests/rest_tests.rs`, especially
`upload_direct_outcomes_multipart_auth_and_chunk_progress`,
`active_paused_unknown_and_shutdown_guard_mutations`, `start_preserves_reserved_filename`.

### Task 3: Telemetry aggregation

- [x] Apply partial deltas but replace the entire snapshot on every new subscription.
- [x] Track display and SD progress availability separately. Successive SD-only
  deltas advance; null display progress enables fallback. Reset job-local values
  when the filename changes or a new run of the same file is observed.
- [x] Parse `print_stats.info.current_layer` / `total_layer`, never infer layers
  from Z. Show optional toolhead Z separately.
- [x] Never use `toolhead.estimated_print_time` as job ETA: it is a motion timebase.
  Progress-based remaining time is approximate and unknown for zero/invalid
  progress, pauses or stale data. Clear filename/message when explicitly emptied.

References: `subscription.rs`; `tests/subscription_tests.rs` sequential-delta,
source-fallback, layers/reset, same-filename restart and ETA tests.

### Task 4: Owned session, authentication and command dispatch

- [x] One owned worker and one shared implementation for CLI/GUI. Snapshot polling
  is sufficient; do not add unused stream interfaces or a parallel networking loop.
- [x] Authenticate native WS via correlated `server.connection.identify`, including
  client_name/version/type=desktop/url and optional api_key. Await success, query
  server readiness, then subscribe and await the correlated snapshot. A socket
  handshake alone is not Connected. Auth/subscription errors are visible, with
  explicit correction/reconnect rather than infinite auth retries.
- [x] Handle socket/channel EOF, Klipper ready/shutdown/disconnected notifications,
  connect/RPC/action/pong deadlines, stale snapshots and cancellable backoff
  1,2,4,8,10 seconds capped. Accept/close must not hot-loop or reset retry attempts.
- [x] Last-handle drop and explicit Disconnect invalidate unsent commands, cancel
  owned futures and prevent reconnects without joining on the UI thread. Explicit
  Reconnect can reopen a retained handle. Disconnect is **not print cancellation**;
  already transmitted side effects cannot be recalled and may be outcome-unknown.
- [x] Use bounded normal admission, serialized execution and atomic upload/start
  single-flight. Retain pending-start reservation until observed or reconciled.
- [x] Give Emergency Stop its own bounded immediate HTTP path, independent of WS
  handshake/backoff and a held upload/full normal queue. Invalidate unsent normal
  work. Report success/failure; network stop is not a hardware safety guarantee.
- [x] Store bounded operation IDs/states/progress/results; retain failures until
  acknowledgment and explicitly reject unavailable/full admissions.

References: `session.rs`, `transport.rs`, `operation.rs`; `tests/session_tests.rs`.
Loopback tests exercise successful authenticated WS, rejected subscription, EOF/drop,
reconnect replacement, held upload/identify emergency stop, normal ordering,
pending-start reservation, pong/RPC/action timeout and capped/backed-off retries.

### Task 5: CLI orchestration

- [x] Validate intent before loading/slicing. URL alone never uploads. Preserve
  offline mesh slicing; upload/print requires a URL and mesh inputs.
- [x] Monitor-only with URL needs no mesh or output file and attaches to the exact
  active filename. Reject upload+monitor without print. Upload+print is one combined
  upload/start, not two uploads or a separate start.
- [x] Print+monitor requires a Started upload outcome, canonical filename and an
  active-job observation before accepting completion; old terminal telemetry cannot
  satisfy it. Fail/queued/unconfirmed/cancelled/auth failure exits nonzero.
- [x] Bound initial readiness/job-start/disconnection waits. Ctrl-C exits monitoring
  without sending print cancellation. Display filename, stale state, thermals,
  elapsed time and explicitly approximate/unknown remaining time.

References: `crates/manifold-cli/src/printer.rs`, `src/main.rs`, and
`tests/printer_cli.rs` actual subprocess exit-code/no-write/target/interrupt tests.

### Tasks 6–8: GUI profiles, panel and application wiring

- [x] Preserve profiles without Moonraker. **Every** profile load retires the previous
  target, including missing Moonraker and auto_connect=false. Active endpoint/session
  identity is immutable and visually separate from editable draft settings.
- [x] Mask API-key input, expose explicit auto-connect preference, validate/save draft
  settings without requiring Connect, and use visible draft in Save Profile.
- [x] Keep the modular collapsible panel. Display action admission errors, retained
  operation results, upload-byte progress distinct from print progress, Klipper
  messages, elapsed/approximate remaining time, optional reported layers and Z.
- [x] Bounded 250ms repaint while any session exists, including collapsed/offscreen
  panel state. Keep repaint behavior in the GUI, not the slicing/protocol layer.
- [x] Guard toolbar starts with fresh readiness/job state and single-flight. The
  server query remains authoritative. Tie cancel confirmation to session/subscription/
  job identity and clear it on replacement, disconnection or a new job.

References: `crates/manifold-gui/src/{printer_panel,app,profile}.rs`; headless egui
shape/state/repaint tests plus profile compatibility tests. These tests do not launch
the desktop application's main loop or assert visual usability on actual hardware.

### Task 9: Verification and review

- [x] Before each commit: `cargo fmt --all` → `cargo clippy --workspace --all-targets`
  → `cargo test --workspace --release`; preserve failures and full command logs.
- [ ] Independent whole-branch reviewer acceptance after the coordinated correction
  pass. Passing tests and the checkboxes above are not independent review approval.
- [ ] Optional operator-authorized live verification, read-only first. Not part of
  unattended checks; no automatic motion/heating/cancel/emergency actions.

Usage and limitations: `docs/moonraker.md`. Current API references:

- <https://moonraker.readthedocs.io/en/latest/external_api/file_manager/#file-upload>
- <https://moonraker.readthedocs.io/en/latest/external_api/server/#identify-connection>
- <https://www.klipper3d.org/Status_Reference.html>
