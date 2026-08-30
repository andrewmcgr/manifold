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
- Serialization: `serde` / `serde_json` for config and profile persistence.
- Geometry backend: `manifold-fidget` (fidget-based SDF/order-field/contour
  extraction, narrow-band Fast Marching Eikonal solvers, GPU compute pipelines,
  and NanoVDB acceleration) — depended on by `manifold-core` (slicing pipeline)
  and `manifold-gui` (visualization and overlays).
- 2D polygon geometry: `i_overlay` (pure-Rust polygon boolean ops and
  inward offsetting) — used by `manifold-core::polygon2d` for per-layer
  infill/solid-fill boundary computation (no C/C++ FFI).
- Mesh loading: pure Rust STL / 3MF loaders in `manifold-core::mesh`.

## Directory Structure

```
Cargo.toml                  # workspace manifest, shared deps/profile
crates/
  manifold-core/            # slicing engine — the only crate with domain logic
    src/
      lib.rs                # public API: SlicerConfig, slice_to_gcode()
      error.rs              # Error/Result
      mesh.rs               # Mesh type & STL/3MF parsers
      machine.rs            # Machine envelope, tools, per-axis limits
      kinematics.rs         # Speeds, accelerations, stepper motor dynamics
      fluid_dynamics.rs     # 2-point PA, extrudate swell, adaptive retraction
      order_field.rs        # Height, Conical, and Eikonal order fields
      slicing.rs            # mesh -> Layer[] via manifold-fidget + contour extraction
      infill.rs             # Cubic, TPMS (Gyroid, Schwarz), Concentric, AllWalls
      wave_overhang.rs      # 2D/3D Huygens wavefront planner (LaSO)
      toolpath.rs           # Layer[] -> Path[] (planning, slope compensation, Z-hops)
      gcode.rs              # Path[] -> Gcode string (macros, SCV, Klipper headers)
  manifold-fidget/          # SDF/order-field/contour-extraction backend
    src/                    # Fast Marching Method, bilateral conformal solvers,
                            # NanoVDB buffers, and wgpu GPU compute pipelines
  manifold-cli/             # `manifold` binary — headless/batch driver
    src/main.rs
  manifold-gui/             # `manifold-gui` binary — egui/wgpu desktop app
    src/main.rs
```

## Core Components

- **`manifold-core`**: the only crate allowed to contain slicing domain
  logic. Structured as a modular pipeline:
  `Workspace -> slicing::slice_mesh -> Layer[] -> toolpath::plan -> Path[] -> gcode::emit -> String`.
  - **Slicing**: Builds a `manifold_fidget::mesh_sdf::MeshSdf` and evaluates the
    configured `OrderField` (Height, Conical, or bilateral conformal Eikonal).
    Contour level-sets are extracted via `extract_contours_at_order` and
    partitioned into outer walls, inner walls, solid exposures, and infill cavities.
  - **Infill**: Generates 3D Cubic truss lattices, TPMS minimal surfaces
    (Gyroid, Schwarz D, Schwarz P), boustrophedon monotonic scanlines, or concentric rings,
    chained with serpentine turnaround bridges to minimize retractions.
  - **Wave Overhangs (LaSO)**: Generates support-free horizontal and non-planar
    overhang passes using Huygens wavefront diffraction, lateral seed anchoring,
    and footprint masking.
  - **Toolpath Planning**: Applies wall simplification (Ramer-Douglas-Peucker),
    tangent flat nozzle slope clearance compensation, 3D trajectory slope cosine
    volumetric compensation, support-aware move sorting, scarf joint seams,
    perimeter wipes, parallel order-aware A* travel collision avoidance routing with
    temporal already-printed solid classification, machine kinematic Z-penalty weighting,
    bed-floor clearance clamping, and Z-hops.
  - **Kinematics & G-code Emission**: Integrates stepper motor torque roll-off
    ODEs, Klipper Square Corner Velocity (SCV) lookahead, per-axis velocity/accel
    clamping, and dynamic 2-point non-Newtonian pressure advance.
- **`manifold-fidget`**: SDF/order-field/contour-extraction backend. Provides
  narrow-band 3D Fast Marching, bilateral upward-downward fixed-point conformal
  iteration, NanoVDB volume grids, 3D DDA raymarching, and `wgpu` GPU compute
  pipelines (`GpuEikonalRelaxation`, 3D TPMS infill compute).
- **`manifold-cli`**: Thin CLI wrapper around `manifold_core::slice_to_gcode`
  with multi-file `:tool_id` syntax, batch processing support, and profile loading.
- **`manifold-gui`**: Hardware-accelerated `egui`/`wgpu` desktop application
  featuring interactive 3D orbit/pan/zoom, object transform gizmos, 3D mesh
  overlays (conformal seed regions, surface order gradients), 7 continuous data views
  (Line Type, Speed, Actual Speed, Flow Rate, Acceleration, Actual Acceleration, Travel Durations),
  interactive hover segment HUD, order scrubber, and an optional local MCP automation server.

## Data Flow

1. Front-end (CLI or GUI) constructs a `Workspace` containing one or more `Object` meshes,
   the printer's `Machine` definition (build volume, clearance profile, per-axis limits),
   and `SlicerConfig`.
2. `manifold_core::slice_to_gcode(&workspace)` runs the slicing pipeline:
   - Slices meshes along order field isosurfaces into `Layer[]`.
   - Plans perimeter walls, solid skins, TPMS/cubic infills, and wave overhangs.
   - Computes kinematic velocity profiles and fluid dynamics retractions.
   - Emits Klipper-compatible G-code with evaluated start/end macro placeholders.
3. CLI writes G-code directly to disk; GUI renders interactive 3D ribbon toolpaths,
   displays print statistics, and allows export.

## Configuration & Persistence

Configuration is bundled into a `Profile` (`manifold-gui::profile` / `docs/configuration/`):
- `Machine`: Hardware envelope, tools, clearance profiles, stepper dynamics, and per-axis limits.
- `SlicerConfig`: Layer heights, infill patterns, conformal Eikonal settings, wave overhangs, speeds, accelerations, and fluid parameters.

Profiles serialize cleanly to and from JSON (`profile.json`).

## Build & Deploy

- Build: `cargo build --workspace` (add `--release` for production performance).
- Test: `cargo test --workspace`.
- Lint: `cargo clippy --workspace --all-targets`.
- Format: `cargo fmt --all`.
