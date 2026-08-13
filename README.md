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

See `ARCHITECTURE.md` and `CODE_STYLE.md` for more detail.
