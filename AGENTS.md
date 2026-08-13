# AGENTS.md

## Status: Active Rust workspace (Manifold)

Manifold is a non-planar slicer for 3D printers: it converts a mesh into
Gcode using toolpaths that are not restricted to flat horizontal layers.

See `ARCHITECTURE.md` and `CODE_STYLE.md` for full detail. This file only
lists what an agent would otherwise guess wrong.

### Commands

- Build: `cargo build --workspace`
- Test (full suite): `cargo test --workspace`
- Test (single crate): `cargo test -p manifold-core`
- Lint: `cargo clippy --workspace --all-targets`
- Format: `cargo fmt --all`
- Required order before committing: `cargo fmt --all` -> `cargo clippy
  --workspace --all-targets` -> `cargo test --workspace`

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
- No CI configured yet, no mesh-format loaders (STL/3MF) implemented yet
  — `Mesh` construction in the CLI is currently a placeholder
  (`Mesh::default()`).
