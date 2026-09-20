# AGENTS.md

## Status: Active Rust workspace (Manifold)

Manifold is a non-planar slicer for 3D printers: it converts a mesh into
Gcode using toolpaths that are not restricted to flat horizontal layers.

See `ARCHITECTURE.md` and `CODE_STYLE.md` for full detail. This file only
lists what an agent would otherwise guess wrong.

### Commands

- Build: `CARGO_TARGET_DIR=target/build cargo build --workspace`
- Test (full suite): `CARGO_TARGET_DIR=target/test cargo nextest run --workspace`
- Test (single crate): `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core`
- Test (doc-tests, rarely needed — nextest does not run these): `CARGO_TARGET_DIR=target/test cargo test --doc --workspace`
- Lint: `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets`
- Format: `cargo fmt --all`
- Required order before committing: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy
  cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test cargo nextest run
  --workspace`
- Each command above uses its own `CARGO_TARGET_DIR` subdirectory
  (`target/build`, `target/test`, `target/clippy`) rather than sharing one
  `target/`. `cargo build`/`test`/`clippy` compile with different flags and
  produce different fingerprint metadata for the same crate; sharing one
  target dir means alternating between them invalidates each other's
  incremental-compilation cache and forces a full rebuild every time you
  switch commands. Separate directories let each command's cache stay warm
  across repeated runs. Always include the `CARGO_TARGET_DIR=...` prefix
  when running these commands manually or from a script — omitting it
  silently falls back to the shared default `target/` and reintroduces the
  thrashing.
- `sccache` (0.13.0) is the `rustc-wrapper` via the project-local
  `.cargo/config.toml` (local disk cache: `~/.cache/sccache`). Verified
  working on this machine: dev-profile (incremental) rustc calls ARE
  cached — after `rm -rf target/build` a full-workspace rebuild hits the
  cache on essentially every compilation (~48 s vs ~73 s for the first
  build that fills the cache). Check status with `sccache --show-stats`;
  zero counters with `sccache --zero-stats` (renamed from
  `--clear-stats` in 0.13). An earlier 0%-hit observation was from a
  different machine and was not reproduced here.
- `cargo-nextest` (0.9.145) is configured in `.config/nextest.toml`
  (`fail-fast = false`, `slow-timeout` period 30 s). Note its config
  schema no longer has `slow-timeout.final` (unknown keys are rejected)
  — use `terminate-after`/`on-timeout` instead if you need a kill switch.

### Architecture

- Entrypoints: `crates/manifold-cli/src/main.rs` (binary `manifold`),
  `crates/manifold-gui/src/main.rs` (binary `manifold-gui`).
- Crate boundaries:
  - `crates/manifold-core` — the slicing engine (mesh -> layers ->
    toolpaths -> Gcode). No UI/CLI dependencies; headless-capable. Only
    crate allowed to hold slicing domain logic.
  - `crates/manifold-cli` — thin CLI front-end over `manifold-core`.
  - `crates/manifold-gui` — egui/wgpu desktop front-end over
    `manifold-core`.

### Conventions

- Core geometry uses `glam::DVec3` (f64) everywhere in `manifold-core` —
  do not introduce `f32`/`Vec3` there.
- `manifold-core` uses `thiserror` for its `Error` enum; application
  crates (`manifold-cli`, `manifold-gui`) use `anyhow` at their
  boundaries. Don't mix the two the other way.
- Logging via `tracing`; only binaries call
  `tracing_subscriber::fmt::init()`, never `manifold-core`.
- No CI configured yet. `manifold-cli` loads real meshes via
  `manifold_core::stl::load_stl` (`.stl`, binary or ASCII) and
  `manifold_core::threemf::load_3mf` (`.3mf`), dispatched by file
  extension in `load_objects` (`crates/manifold-cli/src/main.rs`).
  `Mesh::default()` only appears in that file's own unit tests as a
  stand-in placeholder mesh, never in the production load path.
