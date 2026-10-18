# Manifold2 — Project Architecture & Kinematic Rules

## Target Hardware & Why Standard Slicer Assumptions Do Not Apply

Manifold2 targets multi-axis, non-planar additive manufacturing hardware where standard 3-axis planar slicer assumptions **never** hold:

1. **6-Axis Robot Arm with an FDM Toolhead**: Capable of positioning and orienting the nozzle in **arbitrary 3D directions** anywhere in the workspace.
2. **5-Axis Motion System with a Tilting Build Plate**: Capable of tilting and rotating the bed relative to the world, meaning the direction of gravity changes relative to bed coordinates.

All geometry, scalar fields (OpenVDB), and toolpaths in Manifold2 are represented and computed in **bed coordinates**.

---

## Mandatory Kinematic Invariants (Never Ignore)

### 1. Always Query the Nozzle's Local Orthonormal Frame — Never Assume Nozzle = +Z

- **Rule**: No slicing, field generation, toolpath generation, extrusion width/height calculation, or collision check may hardcode `(0, 0, 1)` as the nozzle axis.
- **Definition of the Nozzle Local Frame** (`NozzleFrame` in C++ and Rust):
  An orthonormal right-handed basis $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$ queried in bed coordinates:
  1. **`nozzle_axis` ($\hat{\mathbf{z}}_{\text{nozzle}}$)**: Unit vector pointing **opposite to the nozzle discharge direction** (i.e., from the nozzle tip up along the barrel of the nozzle). For a vertical nozzle above a flat bed, this is `(0, 0, 1)`, **but it will vary across space and time**.
  2. **`travel_axis` ($\hat{\mathbf{x}}_{\text{nozzle}}$)**: Unit vector pointing in the **direction of toolhead travel**, orthogonalized against `nozzle_axis`.
  3. **`lateral_axis` ($\hat{\mathbf{y}}_{\text{nozzle}}$)**: Unit vector completing the **right-handed basis**:
     $$\hat{\mathbf{y}}_{\text{nozzle}} = \hat{\mathbf{z}}_{\text{nozzle}} \times \hat{\mathbf{x}}_{\text{nozzle}}, \qquad \hat{\mathbf{x}}_{\text{nozzle}} \times \hat{\mathbf{y}}_{\text{nozzle}} = \hat{\mathbf{z}}_{\text{nozzle}}$$
- **Usage**:
  - In C++: Always call `SlicerBackend::queryNozzleFrame(positionBed, travelDirBed)` (or the FFI entry point `manifold::query_nozzle_frame`).
  - In Rust: Always call `SlicerConfig::query_nozzle_frame(position_bed, travel_dir_bed)` / `ffi::query_nozzle_frame(...)`.
  - Tool clearance profiles (XZ pairs of `(radius_x_mm, height_z_mm)`) are always defined in this local nozzle frame, with $Z \ge 0$ measured along `nozzle_axis` and radial distance $X \ge 0$ measured perpendicular to `nozzle_axis`.

### 2. Always Query the Gravity Vector in Bed Coordinates — Never Assume Gravity = -Z

- **Rule**: Because the machine may tilt the build plate (5-axis bed tilt) or mount the build plate in non-horizontal orientations (6-axis cell), gravity in bed coordinates is **not** guaranteed to be `(0, 0, -1)`.
- **Definition**:
  - Gravity is represented as a unit direction vector $\hat{\mathbf{g}}_{\text{bed}}$ in bed coordinates pointing in the direction of gravitational acceleration.
- **Usage**:
  - Any calculation involving overhang angles, support generation, bridging, melt-pool sag, or gravity-aware scalar fields (such as the `"overhang"` OpenVDB grid) **must** query the gravity vector via `SlicerBackend::queryGravityDirection(positionBed)` / `ffi::query_gravity_direction(...)` rather than reading the raw $Z$ component of surface normals or gradients.

### 3. Variable Layer Height & Multi-Tool Awareness

- **Layer Height**: Non-planar conformal layers do not have uniform thickness. Layer height is always configured and queried as a `(target_mm, min_mm, max_mm)` triple (`LayerHeightConfig`), never as a single global constant.
- **Multi-Tool Registry**: A machine may carry multiple extrusion tools (`ToolConfig`), each with its own `nozzle_diameter_mm`, `temperature_celsius`, and local-frame `clearance_profile_xz` (`Vec<ClearancePointXZ>`).
