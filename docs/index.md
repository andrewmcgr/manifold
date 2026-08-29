# Manifold Documentation

Welcome to the **Manifold** non-planar slicing engine documentation. Manifold is a modern 3D slicer designed for FDM (Fused Deposition Modeling) 3D printers that converts 3D meshes into G-code using toolpaths that deform off the flat XY plane.

---

## Table of Contents

### 1. [Getting Started](getting-started.md)
- [System Requirements & Dependencies](getting-started.md#system-requirements)
- [Building from Source](getting-started.md#building-from-source)
- [Quickstart: Command Line](getting-started.md#quickstart-command-line)
- [Quickstart: Graphical Interface](getting-started.md#quickstart-graphical-interface)

### 2. [Command Line Interface (CLI)](cli.md)
- [CLI Synopsis & Usage](cli.md#synopsis)
- [Multi-Model & Multi-Tool Syntax](cli.md#multi-file--multi-tool-syntax)
- [CLI Flag & Option Reference](cli.md#command-line-options-reference)
- [Batch Scripting Examples](cli.md#batch-processing-examples)

### 3. [Graphical User Interface (GUI)](gui/index.md)
- [Interface Overview & 3D Canvas](gui/index.md#interface-overview)
- [3D Viewport Controls & Navigation](gui/index.md#viewport-controls)
- [Object Management & Scene Setup](gui/index.md#object-management--scene-setup)
- [Settings Panel Reference](gui/settings-panel.md)
- [3D Viewport Data Views, Overlays & Inspection HUD](gui/data-views.md)
- [Custom G-code Macros & Placeholders](gui/custom-gcode.md)

### 4. [Configuration & Profile Reference](configuration/index.md)
- [Profile JSON Structure](configuration/index.md#profile-json-structure)
- [Layering & Extrusion Parameters](configuration/layering-extrusion.md)
- [Order Fields & Clearance Envelopes (Bilateral Conformal Eikonal)](configuration/order-fields.md)
- [Infill Patterns (TPMS Gyroid, Schwarz D/P, 3D Cubic)](configuration/infill.md)
- [Wave Overhangs (3D Shell-Conformal Huygens LaSO Engine)](configuration/wave-overhangs.md)
- [Kinematics, Speeds, Per-Axis Limits & Stepper Dynamics](configuration/kinematics.md)
- [Retraction, Scarf Joints, Extrudate Swell & Fluid Dynamics](configuration/retraction-fluid.md)
- [Machine Envelopes, Tools & Kinematics](configuration/machine-tools.md)

---

## Architectural Highlights

- **Headless-First Engine (`manifold-core`)**: Pure Rust slicing pipeline (`Workspace -> slicing -> Layer[] -> toolpath -> Path[] -> gcode -> String`) capable of running on servers, batch clusters, or embedded.
- **Bilateral Conformal Order Fields (`manifold-fidget`)**: Fast Marching Eikonal fields with surface geodesic lower bounds, top/bottom surface conforming, and Lipschitz slope bounds.
- **Support-Free Wave Overhangs (LaSO)**: 2D & 3D Huygens wavefront diffraction planner with lateral seed anchoring and footprint masking.
- **TPMS Minimal Surface Infills**: Continuous Gyroid, Schwarz Diamond, and Schwarz Primitive minimal surface infills evaluated along non-planar order field surfaces.
- **Physical Kinematic & Fluid Dynamics Modeling**: Multi-axis vector projection, ODE integration of stepper motor torque roll-off, Klipper Square Corner Velocity lookahead planning, viscoelastic extrudate swell compensation, and 2-point power-law pressure advance.
- **Hardware-Accelerated 3D GUI & Compute**: `egui`/`wgpu` interface with screen-space ribbon quads, 7 continuous data views, interactive hover inspection HUD, mesh overlays, and optional GPU compute pipelines on NanoVDB grids.
