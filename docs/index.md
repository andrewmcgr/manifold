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
- [Multi-Model & Multi-Tool Slicing](cli.md#multi-file--multi-tool-syntax)
- [CLI Flag & Option Reference](cli.md#command-line-options-reference)
- [Batch Scripting Examples](cli.md#batch-processing-examples)

### 3. [Graphical User Interface (GUI)](gui/index.md)
- [Interface Overview & 3D Canvas](gui/index.md#interface-overview)
- [3D Viewport Controls & Navigation](gui/index.md#viewport-controls)
- [Object Management & Scene Setup](gui/index.md#object-management)
- [Settings Panel & Pop-outs](gui/settings-panel.md)
- [3D Viewport Data Views & Legends](gui/data-views.md)
- [Custom G-code Macros & Placeholders](gui/custom-gcode.md)

### 4. [Configuration & Profile Reference](configuration/index.md)
- [Profile JSON Structure](configuration/index.md#profile-structure)
- [Layering & Extrusion Parameters](configuration/layering-extrusion.md)
- [Order Fields & Toolhead Clearance Envelopes](configuration/order-fields.md)
- [Infill Patterns (AllWalls, Concentric, 3D Cubic)](configuration/infill.md)
- [Wave Overhangs (Huygens LaSO Engine)](configuration/wave-overhangs.md)
- [Kinematics, Speeds & Stepper Dynamics](configuration/kinematics.md)
- [Retraction, Scarf Joints & Thermodynamic Fluid Model](configuration/retraction-fluid.md)
- [Machine Envelopes, Tools & Temperatures](configuration/machine-tools.md)

---

## Architectural Highlights

- **Headless-First Engine (`manifold-core`)**: Pure Rust slicing pipeline (`Mesh -> slicing -> Layer[] -> toolpath -> Path[] -> gcode -> String`) capable of running on servers, batch clusters, or embedded.
- **Non-Planar Order Fields (`manifold-fidget`)**: Fast Marching Eikonal fields, conical order surfaces, and height fields with Lipschitz slope bounds.
- **Support-Free Wave Overhangs**: 2D Huygens-propagation wavefront planner generating laterally supported overhangs (LaSO) without scaffolding.
- **Physical Kinematic & Fluid Dynamics Modeling**: ODE integration of stepper motor torque roll-off, Klipper Square Corner Velocity lookahead planning, and 2-point power-law non-Newtonian pressure advance.
