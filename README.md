# Manifold

A non-planar slicer for 3D printers: turns a mesh into Gcode using
toolpaths that deform off the flat XY plane, rather than strictly planar
layers.

## Crates

- `crates/manifold-core` — the slicing engine (mesh → layers → toolpaths →
  Gcode). No UI dependencies; usable headless or embedded.
- `crates/manifold-cli` — CLI front-end (`manifold`) for headless/batch use.
- `crates/manifold-gui` — desktop GUI (egui + wgpu) for interactive use.

## Building

```sh
cargo build --workspace
```

## Running

```sh
cargo run -p manifold-cli -- path/to/model.stl -o out.gcode
cargo run -p manifold-gui
```

For real/complex meshes, build and run in release mode — the debug
profile can be many times slower for this CPU-heavy geometry pipeline
(polygon offsetting, Eikonal fast-marching, parallel toolpath planning),
which can look like a hang rather than a slow slice:

```sh
cargo run --release -p manifold-cli -- path/to/model.stl -o out.gcode
cargo run --release -p manifold-gui
```

### Automation (dev only)

`manifold-gui` can optionally expose a local MCP automation server for
driving the GUI programmatically (agent/test harnesses) — see
`ROADMAP.md` Phase 9. Off by default; enable with:

```sh
cargo run -p manifold-gui --features mcp-server
```

Listens on `127.0.0.1:8931`. Never enable this feature in a release
build.

See `ARCHITECTURE.md` and `CODE_STYLE.md` for more detail.
