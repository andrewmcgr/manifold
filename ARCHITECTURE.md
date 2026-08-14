# ARCHITECTURE.md

## Overview

Manifold is a **non-planar slicer**: it converts a 3D mesh into Gcode for
FDM 3D printers, using toolpaths that are not restricted to flat horizontal
layers. It is a Rust Cargo workspace with a headless-capable core engine
plus separate CLI and GUI front-ends.

## Tech Stack

- Language: Rust (2021 edition), workspace with `resolver = "2"`.
- GUI: `egui` + `eframe` (wgpu renderer backend).
- CLI: `clap` (derive API).
- Math: `glam` (f64 vectors — `DVec3`).
- Error handling: `thiserror` (library errors), `anyhow` (application errors).
- Logging: `tracing` + `tracing-subscriber`.
- Serialization: `serde` / `serde_json` for config.
- Geometry backend: `manifold-fidget` (fidget-based SDF/order-field/contour
  extraction) — depended on by both `manifold-core` (the mesh -> order
  field -> contour-extraction slicing pipeline) and `manifold-gui`
  (visualization).

## Directory Structure

```
Cargo.toml                  # workspace manifest, shared deps/profile
crates/
  manifold-core/            # slicing engine — the only crate with domain logic
    src/
      lib.rs                # public API: SlicerConfig, slice_to_gcode()
      error.rs               # Error/Result
      mesh.rs                 # Mesh type (vertices/indices)
      slicing.rs              # mesh -> Layer[] (via manifold-fidget's MeshSdf + contour extraction)
      toolpath.rs              # Layer[] -> Path[]
      gcode.rs                  # Path[] -> Gcode string
  manifold-fidget/          # SDF/order-field/contour-extraction backend
    src/                    # depended on by manifold-core (slicing pipeline)
                            # and manifold-gui (visualization)
  manifold-cli/             # `manifold` binary — headless/batch driver
    src/main.rs
  manifold-gui/              # `manifold-gui` binary — egui/wgpu desktop app
    src/main.rs
```

## Core Components

- **`manifold-core`**: the only crate allowed to contain slicing domain
  logic. Structured as a pipeline:
  `Mesh -> slicing::slice_mesh -> Layer[] -> toolpath::plan -> Path[] -> gcode::emit -> String`.
  `slice_mesh` builds a `manifold_fidget::mesh_sdf::MeshSdf` from the mesh
  and walks a `manifold_fidget::order::OrderField`/
  `manifold_fidget::contour::extract_contours_at_order` isosurface-walk to
  produce each `Layer`'s cross-section loops — `manifold-core` depends on
  `manifold-fidget` for this. Each stage is its own module so slicing,
  toolpath planning, and Gcode emission can be developed/tested
  independently. `slice_to_gcode()` in `lib.rs` wires the stages together
  and is the primary entry point for both front-ends.
- **`manifold-fidget`**: SDF/order-field/contour-extraction backend (not
  slicing domain logic itself — a geometry-query library `manifold-core`
  builds its pipeline on top of). Depended on by both `manifold-core` and
  `manifold-gui`.
- **`manifold-cli`**: thin wrapper — parses args with `clap`, builds a
  `SlicerConfig`, calls `manifold_core::slice_to_gcode`, writes the result
  to a file. No slicing logic lives here.
- **`manifold-gui`**: `eframe`/`egui` desktop app (wgpu renderer). Currently
  a scaffold (`ManifoldApp`); will grow to load meshes, preview toolpaths,
  and invoke `manifold-core` interactively.

## Data Flow

1. Front-end (CLI or GUI) loads/obtains a `Mesh`.
2. Front-end builds a `SlicerConfig` (layer height, nozzle diameter, ...).
3. `manifold_core::slice_to_gcode(&mesh, &config)` runs the pipeline and
   returns a Gcode `String`.
4. CLI writes it to disk; GUI will render/preview it.

## External Integrations

None yet. Anticipated future integrations: mesh file formats (STL/3MF)
loaded inside `manifold-core::mesh`, and possibly direct printer
communication (serial/OctoPrint) from a front-end crate — not from
`manifold-core`.

## Configuration

`SlicerConfig` (`manifold-core::lib`) is the single source of slicing
parameters (`layer_height`, `nozzle_diameter`, ...). It derives
`serde::Serialize`/`Deserialize` so it can be persisted as JSON/RON by a
front-end. No environment variables are used yet.

## Build & Deploy

- Build: `cargo build --workspace` (add `--release` for optimized builds;
  the release profile uses `lto = true`, `codegen-units = 1`).
- Test: `cargo test --workspace`.
- Lint: `cargo clippy --workspace --all-targets`.
- Format: `cargo fmt --all`.
- No CI or packaging/deployment pipeline exists yet.
