# Manifold Slicer — Profile Variables Reference

This document provides a complete reference for every configuration variable in Manifold Slicer's `SlicerConfig` data model. These variables control build-plate geometry, 5/6-axis machine kinematics, toolhead clearance envelopes, volumetric OpenVDB Eikonal field generation, non-planar layer extraction, conformal shell/TPMS toolpathing, non-Newtonian polymer rheology, and Klipper G-code generation.

---

## 1. Profile Architecture & Preset Categories

A complete Manifold Slicer configuration (`SlicerConfig`) is organized into seven logical sub-tables plus active tool/preset metadata:

| Top-Level Section | Rust Struct | Primary Role |
| :--- | :--- | :--- |
| `bed` | `PrintBedConfig` | Build plate shape, X/Y/Z build volume dimensions, and viewport grid spacing |
| `machine` | `MachineKinematicsConfig` | 5/6-axis nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$, bed-coordinate gravity $\hat{\mathbf{g}}_{\text{bed}}$, and 5-axis bed tilt (`Rx`, `Ry`) |
| `tools` | `Vec<ToolConfig>` | Multi-tool registry: nozzle diameter, hotend temperature, rotational limits, and XZ clearance profile in the nozzle frame |
| `arrangement` | `ArrangementConfig` | Archimedean spiral auto-arrangement part clearance and bed edge margins |
| `eikonal` | `EikonalFieldConfig` | OpenVDB Dual-Grid resolution, constrained Eikonal solver parameters, overhang bias, and orientation hints |
| `slicing` | `SlicingProcessConfig` | Variable non-planar layer heights, 3D shell thickness, Arachne line widths, TPMS infill, bridging, wave overhangs, and supports |
| `gcode` | `GcodeProcessConfig` | Feedrates, accelerations, Klipper lookahead, 3D scarf seams, retractions, shear-thinning PA, die swell, cooling, and G-code templates |

### Four Orthogonal Preset Categories

In both the GUI and the CLI (`--printer`, `--tool`, `--filament`, `--process`), variables are grouped into four independent preset categories that can be mixed and matched without overwriting unrelated settings:

1. **Printer Preset (`printer`)**: `bed.*`, `machine.bed_tilt_*`, `gcode.gcode_origin_at_bed_center`, `gcode.emit_exclude_object`, `gcode.emit_thumbnails`, `gcode.outer_wall_accel_mms2`, `gcode.inner_wall_accel_mms2`, `gcode.infill_accel_mms2`, `gcode.travel_accel_mms2`, `gcode.square_corner_velocity_mms`, `gcode.minimum_cruise_ratio`, `gcode.use_stepper_dynamics`, `gcode.max_stepper_speed_mms`, `gcode.z_travel_penalty`, `gcode.travel_speed_mms`, `gcode.z_hop_speed_mms`, `gcode.start_gcode_template`, `gcode.end_gcode_template`.
2. **Tool Preset (`tool`)**: Active tool entry (`tools.<active>.*`): `nozzle_diameter_mm`, `has_rotational_axes`, `max_angle_from_bed_normal_deg`, `clearance_profile_xz`, plus `gcode.enable_flat_nozzle_comp`.
3. **Filament Preset (`filament`)**: Active tool's `temperature_celsius` (`nozzle_temperature_celsius`) and material/rheology settings in `gcode.*`: `filament_type`, `filament_diameter_mm`, `filament_density_g_cm3`, `bed_temperature_celsius`, `extrusion_multiplier`, `first_layer_flow_multiplier`, `bridge_flow_multiplier`, `wave_overhang_flow_multiplier`, `max_volumetric_flow_mm3s`, `retract_length_mm`, `retract_speed_mms`, `min_travel_for_retract_mm`, `z_hop_mm`, `wipe_length_mm`, `pre_retract_taper_mm`, `ooze_relaxation_extra_prime_mm`, `ooze_relaxation_tau_s`, `enable_dynamic_pa`, `enable_slicer_pa_subdivision`, `pa_low_flow_s`, `pa_high_flow_s`, `pa_ref_low_flow_mm3s`, `pa_ref_high_flow_mm3s`, `swell_low`, `swell_high`, `enable_corner_flow_comp`, `corner_flow_comp_ratio`, `enable_transient_pressure_comp`, `transient_pressure_exponent`, `transient_min_flow_multiplier`, `fan_speed_percent`, `first_layer_fan_speed_percent`, `disable_fan_first_layers`, `bridge_fan_speed_percent`, `wave_overhang_fan_speed_percent`.
4. **Process Preset (`process`)**: All `slicing.*` parameters, `eikonal.*` solver parameters (preserving user-placed `surface_hints` and `point_hints`), `arrangement.*` clearances, and feature speeds/seam settings in `gcode.*` (`wall_print_order`, `first_layer_speed_mms`, `outer_wall_speed_mms`, `inner_wall_speed_mms`, `solid_infill_speed_mms`, `sparse_infill_speed_mms`, `gap_fill_speed_mms`, `bridge_speed_mms`, `wave_overhang_speed_mms`, `support_speed_mms`, `support_interface_speed_mms`, `enable_scarf_joint`, `scarf_length_mm`, `scarf_start_height_ratio`, `seam_gap_mm`).

### Key Resolution & Shorthand Aliases

When using the CLI (`-s KEY=VALUE`, `profile get`, `profile set`) or partial JSON/TOML profile files, you can reference any parameter by:
- Its **canonical dotted path** (e.g., `slicing.layer_height.target_mm`, `gcode.bed_temperature_celsius`).
- Any **unambiguous suffix or leaf name** (e.g., `layer_height.target_mm`, `infill_density_percent`, `bed_temperature_celsius`).
- Dedicated **preset shorthand aliases** that automatically target the active tool (`tools.0`) or bed config:

| Shorthand Alias | Canonical Target Path |
| :--- | :--- |
| `nozzle_temperature_celsius` or `nozzle_temp` | `tools.0.temperature_celsius` |
| `nozzle_diameter_mm` or `nozzle_diameter` | `tools.0.nozzle_diameter_mm` |
| `has_rotational_axes` | `tools.0.has_rotational_axes` |
| `max_angle_from_bed_normal_deg` | `tools.0.max_angle_from_bed_normal_deg` |
| `clearance_profile_xz` | `tools.0.clearance_profile_xz` |
| `bed_shape` | `bed.shape` |

---

## 2. Print Bed Configuration (`bed.*`)

Controls the build volume geometry and 3D viewport visualization. In Manifold Slicer's scene coordinate system, `(0, 0, 0)` is always the center of the build plate (`z = 0`), while exported G-code coordinates translate to the printer's G-code origin (`gcode.gcode_origin_at_bed_center`).

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `bed.shape` | `"Rectangular"` \| `"Circular"` | `"Rectangular"` | Printer | Build plate footprint geometry. When `"Circular"`, `bed.width_x` specifies the bed diameter and the G-code origin is always at `(0, 0, 0)` bed center. |
| `bed.width_x` | `f32` (mm, `> 0`) | `256.0` | Printer | Printable bed width along the $X$ axis in millimeters (or bed diameter when `bed.shape = "Circular"`). |
| `bed.depth_y` | `f32` (mm, `> 0`) | `256.0` | Printer | Printable bed depth along the $Y$ axis in millimeters. |
| `bed.height_z` | `f32` (mm, `> 0`) | `256.0` | Printer | Maximum printable build height along the bed $Z$ axis in millimeters. |
| `bed.grid_spacing` | `f32` (mm, `> 0`) | `20.0` | Printer | Spacing between major grid lines drawn on the build plate in the 3D viewport. |
| `bed.show_volume_box` | `bool` | `true` | Printer | Whether to draw the 3D wireframe bounding box of the printable build volume in the viewport. |

---

## 3. Multi-Axis Machine Kinematics (`machine.*`)

Manifold Slicer never hardcodes `+Z` as the nozzle axis or `-Z` as gravity. All volumetric fields, overhang angles, and clearance checks query the machine's kinematic vectors in **bed coordinates**.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `machine.nozzle_axis_bed` | `[f32; 3]` (unit vector) | `[0.0, 0.0, 1.0]` | Printer / Runtime | Baseline unit vector $\hat{\mathbf{z}}_{\text{nozzle}}$ pointing **opposite to the nozzle discharge direction** (from the nozzle tip up along the nozzle barrel) in bed coordinates. Used to construct the right-handed orthonormal `NozzleFrame` $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$. |
| `machine.gravity_direction_bed` | `[f32; 3]` (unit vector) | `[0.0, 0.0, -1.0]` | Printer | Unit vector $\hat{\mathbf{g}}_{\text{bed}}$ pointing in the direction of gravitational acceleration in bed coordinates. Automatically recomputed as $R_{\text{bed}}^{-1}(0, 0, -1)$ whenever `bed_tilt_rx_deg` or `bed_tilt_ry_deg` is modified without an explicit `gravity_direction_bed`. |
| `machine.bed_tilt_rx_deg` | `f32` (deg, `-90.0..=90.0`) | `0.0` | Printer | 5-axis build-plate tilt angle around the bed $X$ axis in degrees. Tilting the bed rotates $\hat{\mathbf{g}}_{\text{bed}}$ relative to the part. |
| `machine.bed_tilt_ry_deg` | `f32` (deg, `-90.0..=90.0`) | `0.0` | Printer | 5-axis build-plate tilt angle around the bed $Y$ axis in degrees. |
| `machine.show_nozzle_clearance_gizmo` | `bool` | `false` | UI / Runtime | Renders the active tool's local orthonormal frame $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$, revolved XZ clearance cone, and gravity vector $\hat{\mathbf{g}}_{\text{bed}}$ in the 3D viewport. |

---

## 4. Multi-Tool Registry & Nozzle Clearance (`tools.*` & `active_tool_index`)

Manifold Slicer supports machines with multiple extrusion toolheads (`tools.0`, `tools.1`, ...). By default, two tools are configured:
- `tools.0`: `"T0 - 0.4mm Standard"` ($0.40\text{ mm}$, $215\,^\circ\text{C}$)
- `tools.1`: `"T1 - 0.6mm High-Flow"` ($0.60\text{ mm}$, $230\,^\circ\text{C}$)

| Canonical Key | Type / Allowed Values | Default (`tools.0`) | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `active_tool_index` | `usize` | `0` | Top-Level | 0-based index into `tools` selecting the active toolhead used for slicing, auto-voxel sizing, and G-code emission. |
| `tools.<N>.id` | `u32` | `0` | Tool | Numeric tool identifier substituted into `{tool_id}` (`T0`, `T1`, ...) in G-code commands. |
| `tools.<N>.name` | `String` | `"T0 - 0.4mm Standard"` | Tool | Human-readable tool label displayed in the UI tool selector. |
| `tools.<N>.nozzle_diameter_mm` | `f32` (mm, `>= 0.10`) | `0.40` | Tool | Orifice diameter $d_{\text{nozzle}}$ of the nozzle tip. Scales the auto fine voxel size, flat-nozzle land radius ($R_{\text{flat}} = d_{\text{nozzle}}$), and circular bridge cross-sectional area $\pi (d_{\text{nozzle}}/2)^2$. |
| `tools.<N>.temperature_celsius` | `f32` (°C) | `215.0` | Filament / Tool | Target hotend extrusion temperature substituted into `{nozzle_temp}` (`M104`/`M109`) and Moonraker header metadata. |
| `tools.<N>.has_rotational_axes` | `bool` | `true` | Tool | Indicates whether the toolhead has 5/6-axis rotational degrees of freedom allowing it to tilt $\hat{\mathbf{z}}_{\text{nozzle}}$ away from the bed normal. |
| `tools.<N>.max_angle_from_bed_normal_deg` | `f32` (deg, `0.0..=180.0`) | `45.0` | Tool | Maximum angle $\theta_{\text{max,tool}}$ that the toolhead axis $\hat{\mathbf{z}}_{\text{nozzle}}$ may tilt away from the bed normal $+Z_{\text{bed}}$. On a 3-axis gantry (`has_rotational_axes = false`), effective tool tilt is $0^\circ$; on a 6-axis robot arm, this can be set up to $90^\circ+$. |
| `tools.<N>.clearance_mode` | `"PiecewiseXZ"` \| `"ParametricCone"` \| `"RectilinearCone"` | `"PiecewiseXZ"` | Tool | Selects the tool clearance envelope mode in the local nozzle frame $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$: `PiecewiseXZ` (multi-point revolved `(X_radius, Z_height)` table), `ParametricCone` (straight-sided circular cone), or `RectilinearCone` (4-sided rectangular pyramid with independent X and Y half-widths and cone angles). |
| `tools.<N>.clearance_profile_xz` | `Vec<ClearancePointXZ>` | 5-point profile: `(0.5, 0.0)`, `(2.5, 2.0)`, `(9.0, 6.0)`, `(22.0, 18.0)`, `(28.0, 45.0)` | Tool | Piecewise-linear revolved clearance envelope defined in the **local nozzle frame** (`NozzleFrame`), ordered by ascending `z_height_mm` ($\ge 0$ along $\hat{\mathbf{z}}_{\text{nozzle}}$) and radial distance `x_radius_mm` ($\ge 0$ perpendicular to $\hat{\mathbf{z}}_{\text{nozzle}}$). Used when `clearance_mode = "PiecewiseXZ"`. |
| `tools.<N>.cone_tip_radius_mm` | `f32` (mm, `>= 0.1`) | `0.50` | Tool | Tip flat radius at $z = 0$ in the local nozzle frame when `clearance_mode = "ParametricCone"`. |
| `tools.<N>.cone_half_angle_deg` | `f32` (deg, `5.0..=80.0`) | `35.0` | Tool | Cone half-angle $\theta_{\text{cone}}$ measured from $\hat{\mathbf{z}}_{\text{nozzle}}$ when `clearance_mode = "ParametricCone"` ($\alpha_{\max} = 90^\circ - \theta_{\text{cone}}$). |
| `tools.<N>.cone_height_mm` | `f32` (mm, `>= 2.0`) | `45.0` | Tool | Axial height of the circular clearance cone along $\hat{\mathbf{z}}_{\text{nozzle}}$ when `clearance_mode = "ParametricCone"`. |
| `tools.<N>.rect_tip_half_width_x_mm` | `f32` (mm, `>= 0.1`) | `0.50` | Tool | Tip half-width along `travel_axis` ($\hat{\mathbf{x}}_{\text{nozzle}}$) at $z = 0$ when `clearance_mode = "RectilinearCone"`. |
| `tools.<N>.rect_tip_half_width_y_mm` | `f32` (mm, `>= 0.1`) | `0.50` | Tool | Tip half-width along `lateral_axis` ($\hat{\mathbf{y}}_{\text{nozzle}}$) at $z = 0$ when `clearance_mode = "RectilinearCone"`. |
| `tools.<N>.rect_half_angle_x_deg` | `f32` (deg, `5.0..=80.0`) | `35.0` | Tool | Rectilinear pyramid half-angle $\theta_X$ in the local $XZ$ plane measured from $\hat{\mathbf{z}}_{\text{nozzle}}$ ($\alpha_{\max,X} = 90^\circ - \theta_X$) when `clearance_mode = "RectilinearCone"`. |
| `tools.<N>.rect_half_angle_y_deg` | `f32` (deg, `5.0..=80.0`) | `25.0` | Tool | Rectilinear pyramid half-angle $\theta_Y$ in the local $YZ$ plane measured from $\hat{\mathbf{z}}_{\text{nozzle}}$ ($\alpha_{\max,Y} = 90^\circ - \theta_Y$) when `clearance_mode = "RectilinearCone"`. |
| `tools.<N>.rect_height_mm` | `f32` (mm, `>= 2.0`) | `45.0` | Tool | Axial height of the 4-sided rectilinear pyramid cone along $\hat{\mathbf{z}}_{\text{nozzle}}$ when `clearance_mode = "RectilinearCone"`. |

---

## 5. Auto-Arrangement (`arrangement.*`)

Parameters governing the Archimedean spiral collision-free part packer (`frontend/src/arrangement.rs`), which arranges parts around the build-plate center `(0, 0)` using each part's 2D convex-hull footprint.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `arrangement.part_clearance_mm` | `f32` (mm, `>= 0.0`) | `4.0` | Process | Minimum XY clearance distance maintained between the 2D convex hulls of any two parts on the build plate. Increase when printing tall non-planar parts with wide toolhead shrouds. |
| `arrangement.bed_edge_margin_mm` | `f32` (mm, `>= 0.0`) | `5.0` | Process | Minimum inward safety margin kept between part footprints and the outer boundary of the build plate (`Rectangular` or `Circular`). |
| `arrangement.spiral_angle_step_rad` | `f32` (rad, `> 0.0`) | `0.20` | Process | Angular step increment $\Delta\theta$ (radians) used when searching outward along the Archimedean placement spiral $r(\theta) = \frac{p_{\text{radial}}}{2\pi}\theta$. |
| `arrangement.spiral_radial_pitch_mm` | `f32` (mm, `> 0.0`) | `8.0` | Process | Radial expansion distance $p_{\text{radial}}$ per full $2\pi$ revolution of the Archimedean search spiral. |

---

## 6. Volumetric OpenVDB Fields & Constrained Eikonal Solver (`eikonal.*`)

Controls the C++ OpenVDB volumetric backend (`backend/src/slicer_backend.cpp`), which builds the Signed Distance Field $\phi_{\text{sdf}}(\mathbf{x})$, the gravity-aware Overhang Field $o(\mathbf{x}) = -\nabla\phi_{\text{sdf}}\cdot\hat{\mathbf{g}}_{\text{bed}}$, the Tool Orientation Field $\hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x})$, and the Constrained Eikonal Arrival-Time / Order Field $\tau(\mathbf{x})$.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `eikonal.auto_voxel_size` | `bool` | `true` | Process | When `true`, automatically computes `eikonal.voxel_size_mm` from the active tool's nozzle diameter $d_{\text{nozzle}}$ and target layer height $h_{\text{target}}$ via $\Delta x_{\text{fine}} = \text{clamp}(0.375\,d_{\text{nozzle}} + 0.50\,h_{\text{target}},\, 0.08,\, 2.00)$. For a $0.40\text{ mm}$ nozzle at $0.20\text{ mm}$ layers, this yields $0.25\text{ mm}$ ($\sim 3\text{--}5$ fine voxels across a 2-wall shell). |
| `eikonal.voxel_size_mm` | `f32` (mm, `0.05..=5.0`) | `0.25` | Process | Spatial resolution $\Delta x_{\text{fine}}$ of the fine OpenVDB boundary-shell grids (`sdf`, `overhang`, `eikonal`). Overridden automatically when `eikonal.auto_voxel_size = true`. |
| `eikonal.enable_multi_resolution_grid` | `bool` | `true` | Process | Enables the **Dual-Grid Multi-Resolution** architecture: solves global Eikonal propagation and sparse core infill on a coarse interior grid ($\Delta x_{\text{coarse}} = R \cdot \Delta x_{\text{fine}}$) while refining the outer shell ($-T_{\text{shell}} \le \phi_{\text{sdf}} \le 0$) on a fine narrow-band grid with conformal mesh subdivision. |
| `eikonal.coarse_grid_ratio` | `u32` (`2..=8`) | `4` | Process | Ratio $R = \Delta x_{\text{coarse}} / \Delta x_{\text{fine}}$ between the coarse interior core voxel size and the fine boundary-shell voxel size. At $R = 4$, interior voxel count and Fast-Sweeping cost are reduced by $4^3 = 64\times$. |
| `eikonal.narrow_band_width_voxels` | `u32` (`2..=32`) | `6` | Process | Half-width (in voxels) of the initial OpenVDB level-set narrow band allocated around the exterior surface before signed-distance interior flood filling. |
| `eikonal.curvature_regularization` | `f32` (`0.0..=1.0`) | `0.05` | Process | Laplacian smoothing weight applied during iterative director-field and Eikonal relaxation. Higher values smooth out sharp kinks or shocks where competing surface orientation hints meet. |
| `eikonal.solver_iterations` | `u32` (`1..=16`) | `4` | Process | Number of outer Gauss-Seidel / Fast-Sweeping + gradient-cone projection passes used to converge the constrained Eikonal field $\tau(\mathbf{x})$ and coupled 5/6-axis tool orientation field. |
| `eikonal.default_surface_hint_depth_mm` | `f32` (mm, `> 0.0`) | `12.0` | Process | Default inward Euclidean distance ($-\phi_{\text{sdf}}$) over which newly placed B-Rep face or planar mesh patch orientation hints steer $\nabla\tau$. |
| `eikonal.default_point_hint_radius_mm` | `f32` (mm, `> 0.0`) | `18.0` | Process | Default spherical radius (Wendland $C^2$ radial falloff) for newly placed 3D point orientation hints. |
| `eikonal.default_hint_weight` | `f32` (`0.0..=1.0`) | `0.85` | Process | Default soft blending weight assigned to newly placed surface or 3D point orientation hints. |
| `eikonal.enable_overhang_orientation_bias` | `bool` | `true` | Process | Automatically biases the target layer normal $\hat{\mathbf{d}}(\mathbf{x})$ near the part boundary between **orthogonal** ($\nabla\tau \perp \hat{\mathbf{n}}_{\text{surf}}$) and **conformal** ($\nabla\tau \parallel \pm\hat{\mathbf{n}}_{\text{surf}}$) based on the signed overhang angle $\theta_{\text{ovh}} = \arcsin(\nabla\phi_{\text{sdf}}\cdot\hat{\mathbf{g}}_{\text{bed}}) \in [-90^\circ, +90^\circ]$. |
| `eikonal.overhang_bias_weight` | `f32` (`0.0..=1.0`) | `0.75` | Process | Soft blend weight for the automatic overhang-driven surface orientation bias. Explicit user-placed `surface_hints` and `point_hints` always override this bias within their influence zones. |
| `eikonal.orthogonal_supported_angle_deg` | `f32` (deg, `0.0..=89.0`) | `30.0` | Process | Supported surface angle from vertical ($\theta_{\text{min}} = -\text{orthogonal\_supported\_angle\_deg}$, default $-30^\circ$) below which upward-facing domes/roofs ($\theta_{\text{ovh}} < -30^\circ$) bias toward **conformal** layers ($\nabla\tau \parallel +\hat{\mathbf{n}}_{\text{surf}}$) to eliminate stair-stepping, while steeper walls ($\theta_{\text{ovh}} \ge -30^\circ$) bias toward **orthogonal** layers ($\nabla\tau \perp \hat{\mathbf{n}}_{\text{surf}}$). |
| `eikonal.orthogonal_overhang_angle_deg` | `f32` (deg, `0.0..=89.0`) | `60.0` | Process | Overhang angle past vertical ($\theta_{\text{max}} = +\text{orthogonal\_overhang\_angle\_deg}$, default $+60^\circ$) up to which overhanging walls bias toward **orthogonal** layers ($\nabla\tau \perp \hat{\mathbf{n}}_{\text{surf}}$) so layers tilt to support the overhang at $90^\circ$, while steeper overhangs/ceilings ($\theta_{\text{ovh}} > +60^\circ$) transition to **conformal** ($\nabla\tau \parallel -\hat{\mathbf{n}}_{\text{surf}}$). |
| `eikonal.overhang_bias_transition_deg` | `f32` (deg, `1.0..=45.0`) | `15.0` | Process | Angular width $\Delta\theta$ of the $C^1$ cubic Hermite `smoothstep` transition around $\theta_{\text{min}}$ and $\theta_{\text{max}}$, preventing angular discontinuities in the target director field. |
| `eikonal.surface_hints` | `Vec<SurfaceFaceHint>` | `[]` | Project (`.3mf`) | List of soft orientation hints attached to selectable `SurfacePatch` IDs (`object_id`, `patch_id`, `mode`: `NormalOrthogonalToSurface` or `NormalTangentToSurface`, `influence_depth_mm`, `weight`). |
| `eikonal.point_hints` | `Vec<PointNormalHint>` | `[]` | Project (`.3mf`) | List of soft 3D point orientation hints stored in object-local coordinates (`id`, `object_id`, `local_position`, `local_direction`, `influence_radius_mm`, `weight`). |

---

## 7. Non-Planar Slicing, Shells, Infill, Overhangs & Supports (`slicing.*`)

Controls non-planar layer extraction (`backend/src/slicer_backend.cpp`), conformal shell vs. core splitting, Arachne-style medial-axis variable bead widths, 3D TPMS lattices, circular bridging, inside-out wave overhangs, and causal support generation (`frontend/src/toolpath.rs`).

### 7.1 Variable Non-Planar Layer Height (`slicing.layer_height.*` & Slope Limit)

Because conformal non-planar isosurfaces expand and contract over curved geometry, layer thickness is never a single constant; it is governed by a 5-parameter `LayerHeightConfig` evaluated along the local isosurface normal $\hat{\mathbf{n}}_{\text{layer}} = \nabla\tau / \|\nabla\tau\|$:

$$h(\mathbf{x}) = \text{clamp}\!\left(\frac{\Delta\tau}{\|\nabla\tau(\mathbf{x})\|},\; h_{\min},\; h_{\max}\right)$$

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `slicing.layer_height.target_mm` | `f32` (mm, `min_mm..=max_mm`) | `0.20` | Process | Nominal target layer thickness along the local surface normal between consecutive non-planar isosurfaces. |
| `slicing.layer_height.min_mm` | `f32` (mm, `>= 0.01`) | `0.08` | Process | Minimum allowable local layer thickness where conformal layers compress or pinch over high-curvature zones. |
| `slicing.layer_height.max_mm` | `f32` (mm, `>= min_mm`) | `0.32` | Process | Maximum allowable local layer thickness where conformal layers fan out. Should generally remain $\le 80\%$ of `nozzle_diameter_mm`. |
| `slicing.layer_height.first_layer_mm` | `f32` (mm, `0.01..=1.0`) | `0.20` | Process | Dedicated thickness for the initial build-plate adhesion layer (`layer_index == 0`). |
| `slicing.layer_height.top_layer_mm` | `f32` (mm, `0.01..=1.0`) | `0.20` | Process | Dedicated target thickness for the terminal top isosurface of each branch in the Reeb graph. |
| `slicing.max_nonplanar_slope_deg` | `f32` (deg, `0.0..=85.0`) | `35.0` | Process | Maximum allowable angle between the isosurface normal $\nabla\tau$ and the local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$. Clamps the Eikonal gradient cone so layers never tilt steeper than the toolhead's physical clearance cone. |

### 7.2 Conformal 3D Shell, Perimeters & Arachne Variable Line Widths

Manifold Slicer defines the outer shell as the true 3D Euclidean band $-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$, ensuring uniform shell thickness across vertical walls, sloped domes, roofs, and floors alike. Each extrusion type has an independent `LineWidthConfig` (`target_mm`, `min_mm`, `max_mm`) for medial-axis adaptive bead sizing.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `slicing.shell_thickness_mm` | `f32` (mm, `>= 0.05`) | `1.60` | Process | True 3D signed-distance thickness $T_{\text{shell}}$ of the dense outer skin ($-\phi_{\text{sdf}}(\mathbf{x}) \le T_{\text{shell}}$). Any part interior with $\phi_{\text{sdf}}(\mathbf{x}) < -T_{\text{shell}}$ is classified as sparse core interior. |
| `slicing.inner_wall_count` | `u32` | `2` | Process | Number of concentric inner perimeter loops printed immediately inside the single outer wall (`0` = outer wall only; `2` = 1 outer wall + 2 inner walls = 3 total walls). |
| `slicing.perimeter_count` | `u32` | `3` | Process | Legacy/auxiliary total perimeter count field maintained alongside `1 + inner_wall_count`. |
| `slicing.outer_wall_width.target_mm` | `f32` (mm) | `0.40` | Process | Nominal target bead width for outer walls contacting $\phi_{\text{sdf}} = 0$. |
| `slicing.outer_wall_width.min_mm` | `f32` (mm, `>= 0.05`) | `0.28` | Process | Minimum outer wall bead width when compressing into narrow thin-wall tips. |
| `slicing.outer_wall_width.max_mm` | `f32` (mm, `>= min_mm`) | `0.56` | Process | Maximum outer wall bead width when expanding along the medial axis to eliminate thin-wall voids. |
| `slicing.inner_wall_width.target_mm` | `f32` (mm) | `0.42` | Process | Nominal target bead width for inner walls. |
| `slicing.inner_wall_width.min_mm` | `f32` (mm, `>= 0.05`) | `0.28` | Process | Minimum adaptive bead width for inner walls. |
| `slicing.inner_wall_width.max_mm` | `f32` (mm, `>= min_mm`) | `0.58` | Process | Maximum adaptive bead width for inner walls. |
| `slicing.solid_infill_width.target_mm` | `f32` (mm) | `0.42` | Process | Nominal target bead width for dense shell solid infill and medial-axis gap fill. |
| `slicing.solid_infill_width.min_mm` | `f32` (mm, `>= 0.05`) | `0.26` | Process | Minimum adaptive bead width for solid infill and narrow gap-fill traces. |
| `slicing.solid_infill_width.max_mm` | `f32` (mm, `>= min_mm`) | `0.60` | Process | Maximum adaptive bead width for solid infill and gap fill. |
| `slicing.solid_infill_pattern` | `"Monotonic"` \| `"Concentric"` | `"Monotonic"` | Process | Pattern used to fill the remaining dense shell region between the innermost wall and $\phi_{\text{sdf}} = -T_{\text{shell}}$. `"Monotonic"` uses alternating-angle flood-ordered scanlines; `"Concentric"` continues inward geodesic/SDF insetting. |
| `slicing.min_extrusion_length_mm` | `f32` (mm, `>= 0.0`) | `0.80` | Process | Minimum total arc length for an isolated extrusion chain (defaults to $2\times d_{\text{nozzle}}$). Shorter isolated fragments are culled to avoid blobs and unnecessary retractions. |

### 7.3 Volumetric Sparse Core Infill (`slicing.sparse_infill_*` & `infill_*`)

Inside the core interior ($\phi_{\text{sdf}}(\mathbf{x}) < -T_{\text{shell}}$), Manifold Slicer intersects analytical 3D lattices or scanline fields directly with the curved non-planar layer surface.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `slicing.sparse_infill_pattern` | `"Gyroid"` \| `"SchwarzD"` \| `"SchwarzP"` \| `"Cubic"` \| `"Monotonic"` \| `"Concentric"` \| `"AllWalls"` | `"Gyroid"` | Process | Volumetric sparse infill pattern in the core interior:<br>• `"Gyroid"`: 3D TPMS $\sin X\cos Y + \sin Y\cos Z + \sin Z\cos X = 0$<br>• `"SchwarzD"`: 3D Diamond TPMS lattice<br>• `"SchwarzP"`: 3D Primitive TPMS lattice<br>• `"Cubic"`: 3D space-diagonal truss ($\pm 45^\circ$ orthogonal scanlines with $\sqrt{2}$ phase drift)<br>• `"Monotonic"`: Alternating-angle rectilinear scanlines<br>• `"Concentric"`: Inward geodesic/SDF offset loops at spacing $w/\rho$<br>• `"AllWalls"`: $100\%$ concentric inward offset loops at $1\times$ line width |
| `slicing.infill_density_percent` | `f32` (`0.0..=100.0` %) | `20.0` | Process | Target volume fraction $\rho \in [0, 1]$ of sparse infill in the core interior. Determines the spatial wavelength $L_{\text{tpms}}$ of 3D TPMS lattices or line spacing $s = w_{\text{target}} / \rho$ of scanline patterns. |
| `slicing.infill_angle_deg` | `f32` (`0.0..=180.0` deg) | `45.0` | Process | Base in-plane rotation angle around the bed $Z$ axis for infill lattices and scanlines. |
| `slicing.sparse_infill_width.target_mm` | `f32` (mm) | `0.45` | Process | Nominal target bead width for sparse core infill. |
| `slicing.sparse_infill_width.min_mm` | `f32` (mm, `>= 0.05`) | `0.30` | Process | Minimum adaptive bead width for sparse core infill. |
| `slicing.sparse_infill_width.max_mm` | `f32` (mm, `>= min_mm`) | `0.65` | Process | Maximum adaptive bead width for sparse core infill. |

### 7.4 Gravity-Aware Overhangs, Circular Bridging & Inside-Out Wave Overhangs

When an extrusion path lies in a region where the gravity-aware overhang field exceeds `overhang_angle_threshold_deg` and has no solid part or support structure within $1.75\,h_{\text{layer}}$ below it along $+\hat{\mathbf{g}}_{\text{bed}}$, Manifold Slicer classifies it into either **Circular Bridging** (two-sided straight spans $\le L_{\max}$) or **Inside-Out Wave Overhangs** (one-sided cantilevers or curved overhangs).

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `slicing.overhang_angle_threshold_deg` | `f32` (`5.0..=85.0` deg) | `45.0` | Process | Overhang angle from vertical (evaluated against true bed-coordinate gravity $\hat{\mathbf{g}}_{\text{bed}}$ via the `"overhang"` grid) above which unsupported extrusions switch to Bridging or Wave Overhangs. |
| `slicing.max_bridge_length_mm` | `f32` (`0.0..=500.0` mm) | `40.0` | Process | Maximum unsupported 3D straight-line span distance between two supported anchor piers that can be crossed with tensioned circular `Bridge` extrusions ($A_{\text{bridge}} = \pi (d_{\text{nozzle}}/2)^2$). Spans exceeding this length or lacking a second anchor pier use inside-out `WaveOverhang` passes (or supports). |
| `slicing.wave_overhang_overlap_fraction` | `f32` (`0.01..=0.35`) | `0.08` | Process | Fractional lateral overlap $\alpha_{\text{ov}} \in [0.01, 0.35]$ (default $8\%$) of a standard line width maintained between each inside-out `WaveOverhang` pass and the adjacent inner supported extrusion, with centerline spacing $d_{\text{step}} = \frac{1}{2}(w_{\text{prev}} + w_{\text{curr}}) - \Delta w_{\text{ov}}$ and compensated extrusion cross-section $A_{\text{wave}} = (w_{\text{curr}} - \Delta w_{\text{ov}})\,h$. |

### 7.5 Causal Support Generation (`slicing.*support*`)

Manifold Slicer identifies overhangs that exceed the constrained Eikonal slope cone or hang downward as causal local minima, projects support columns along $+\hat{\mathbf{g}}_{\text{bed}}$ to the build plate or lower part surfaces, and optionally seeds the Eikonal order field $\tau(\mathbf{x})$ from the support contact roofs.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `slicing.enable_support_generation` | `bool` | `true` | Process | Enables automatic detection of unsupported overhangs and generation of gravity-aligned `Support` (coarse base) and `SupportInterface` (dense roof/floor) toolpaths. |
| `slicing.seed_order_field_from_supports` | `bool` | `true` | Process | Seeds the constrained Eikonal arrival-time field $\tau(\mathbf{x})$ at support contact patches so downward-hanging islands and steep cantilevers are reached causally upward from the support roof. |
| `slicing.support_only_unreachable_and_bridges` | `bool` | `false` | Process | When `true`, generates supports **only** for regions that are causally unreachable by the constrained Eikonal field (floating islands / downward-hanging local minima) and two-pier bridges longer than `max_bridge_length_mm`, leaving reachable overhangs to conformal layers and inside-out wave overhangs. |
| `slicing.support_z_gap_mm` | `f32` (`0.05..=5.0` mm) | `0.20` | Process | Separation distance along $-\hat{\mathbf{g}}_{\text{bed}}$ between the top support interface (or bottom support floor) and the object surface so supports detach cleanly. |
| `slicing.support_xy_gap_mm` | `f32` (`0.10..=10.0` mm) | `0.50` | Process | Minimum lateral 3D signed-distance clearance ($\phi_{\text{sdf}} \ge d_{xy}$) maintained between support columns and adjacent part walls. |
| `slicing.support_interface_layers` | `u32` (`0..=16`) | `2` | Process | Number of dense `SupportInterface` layers printed immediately below overhanging contact roofs. |
| `slicing.support_bed_interface_layers` | `u32` (`0..=16`) | `1` | Process | Number of dense `SupportInterface` layers printed on the build plate at the base of support columns for strong bed adhesion. |
| `slicing.support_base_spacing_mm` | `f32` (`0.40..=20.0` mm) | `2.50` | Process | Centerline scanline spacing for coarse `Support` base columns. |
| `slicing.support_interface_spacing_mm` | `f32` (`0.20..=5.0` mm) | `0.60` | Process | Centerline scanline spacing for dense `SupportInterface` roofs/floors. |
| `slicing.support_line_width_mm` | `f32` (`0.15..=3.0` mm) | `0.40` | Process | Extruded bead width for support base and support interface lines. |

---

## 8. G-code Generation, Kinematics & Polymer Rheology (`gcode.*`)

Controls the toolpath-to-G-code compiler (`frontend/src/gcode.rs`), including coordinate origin placement, Moonraker metadata, Klipper lookahead velocity planning, 3D scarf joints, viscoelastic die swell, shear-thinning Pressure Advance, corner flow compensation, and transient melt-chamber pressure lag.

### 8.1 General G-code Output, Filament & Cooling

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gcode.gcode_origin_at_bed_center` | `bool` | `false` | Printer | When `false` (standard for rectangular CoreXY/Cartesian beds), G-code `(X0, Y0)` is placed at the front-left corner of the bed (`[0..width_x, 0..depth_y]`). When `true` (or on `Circular` beds), G-code `(X0, Y0)` is at the bed center. |
| `gcode.emit_exclude_object` | `bool` | `true` | Printer | Emits Klipper `EXCLUDE_OBJECT_DEFINE` (with 2D convex-hull `POLYGON` and `CENTER`), `EXCLUDE_OBJECT_START`, and `EXCLUDE_OBJECT_END` commands around per-object extrusions. |
| `gcode.emit_thumbnails` | `bool` | `true` | Printer | Renders `32x32` and `300x300` isometric 3D toolpath PNG thumbnails into base64 `; thumbnail begin` blocks at the top of the `.gcode` file for Moonraker/Mainsail/Fluidd/KlipperScreen. |
| `gcode.wall_print_order` | `"InnerOuterInner"` \| `"OutsideIn"` | `"InnerOuterInner"` | Process | Sequence for concentric perimeter loops within each island:<br>• `"InnerOuterInner"` (`In-Out-In`): prints deep inner walls `[n..2]`, then the visible outer wall `[0]` unconfined for dimensional accuracy, then backing inner wall `[1]` to lock it in.<br>• `"OutsideIn"`: prints outer wall `[0]` first, followed by inner walls `[1..n]`. |
| `gcode.filament_type` | `String` | `"PLA"` | Filament | Material identifier written to Moonraker header comments (`; filament_type = PLA`). |
| `gcode.filament_diameter_mm` | `f32` (`1.0..=3.5` mm) | `1.75` | Filament | Raw filament feedstock diameter $D_{\text{fil}}$ entering the extruder ($A_{\text{fil}} = \pi D_{\text{fil}}^2 / 4$). |
| `gcode.filament_density_g_cm3` | `f32` (`0.5..=4.0` g/cm³) | `1.24` | Filament | Bulk density of the filament in $\text{g/cm}^3$ used to report total filament weight in grams. |
| `gcode.bed_temperature_celsius` | `f32` (`0.0..=150.0` °C) | `60.0` | Filament | Target heated bed temperature substituted into `{bed_temp}` (`M140`/`M190`). |
| `gcode.extrusion_multiplier` | `f32` (`0.2..=2.0`) | `1.00` | Filament | Global volumetric extrusion multiplier applied to all extruding moves. |
| `gcode.first_layer_flow_multiplier` | `f32` (`0.5..=2.0`) | `1.02` | Filament | Additional flow multiplier applied on `layer_index == 0`. |
| `gcode.bridge_flow_multiplier` | `f32` (`0.2..=2.0`) | `0.95` | Filament | Additional flow multiplier applied to free-spanning circular `Bridge` extrusions to keep strands taut. |
| `gcode.wave_overhang_flow_multiplier` | `f32` (`0.2..=2.0`) | `1.00` | Filament | Additional flow multiplier applied to inside-out `WaveOverhang` extrusions on top of the overlap volume adjustment. |
| `gcode.fan_speed_percent` | `f32` (`0.0..=100.0` %) | `100.0` | Filament | Steady-state part cooling fan duty cycle (`M106 S0..255`) after `disable_fan_first_layers`. |
| `gcode.first_layer_fan_speed_percent` | `f32` (`0.0..=100.0` %) | `0.0` | Filament | Part cooling fan duty cycle during the initial `disable_fan_first_layers` layers (`0%` emits `M107`). |
| `gcode.disable_fan_first_layers` | `u32` (`0..=20`) | `1` | Filament | Number of initial layers that use `first_layer_fan_speed_percent` before switching to `fan_speed_percent`. |
| `gcode.bridge_fan_speed_percent` | `f32` (`0.0..=100.0` %) | `100.0` | Filament | Part cooling fan duty cycle automatically emitted (`M106`) during circular `Bridge` extrusions. |
| `gcode.wave_overhang_fan_speed_percent` | `f32` (`0.0..=100.0` %) | `100.0` | Filament | Part cooling fan duty cycle automatically emitted (`M106`) during `WaveOverhang` extrusions. |

### 8.2 Feature Speeds, Accelerations & Stepper Kinematics

Every extruding move's target feedrate is automatically capped by the hotend's volumetric ceiling $v \le Q_{\max} / (w \cdot h)$ and planned through a two-pass backward/forward Klipper lookahead solver with optional stepper back-EMF torque roll-off.

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gcode.first_layer_speed_mms` | `f32` (`5.0..=500.0` mm/s) | `45.0` | Process | Target linear speed for all extruding moves on `layer_index == 0`. |
| `gcode.outer_wall_speed_mms` | `f32` (`5.0..=600.0` mm/s) | `65.0` | Process | Target linear speed for visible `OuterWall` perimeters. |
| `gcode.inner_wall_speed_mms` | `f32` (`5.0..=600.0` mm/s) | `100.0` | Process | Target linear speed for structural `InnerWall` perimeters. |
| `gcode.solid_infill_speed_mms` | `f32` (`5.0..=800.0` mm/s) | `130.0` | Process | Target linear speed for dense 3D shell `SolidInfill`. |
| `gcode.sparse_infill_speed_mms` | `f32` (`5.0..=800.0` mm/s) | `160.0` | Process | Target linear speed for core interior `SparseInfill` (TPMS lattices, Cubic, etc.). |
| `gcode.gap_fill_speed_mms` | `f32` (`5.0..=500.0` mm/s) | `60.0` | Process | Target linear speed for medial-axis `GapFill` beads. |
| `gcode.bridge_speed_mms` | `f32` (`5.0..=500.0` mm/s) | `35.0` | Process | Target linear speed for free-spanning circular `Bridge` extrusions. |
| `gcode.wave_overhang_speed_mms` | `f32` (`5.0..=500.0` mm/s) | `30.0` | Process | Target linear speed for inside-out `WaveOverhang` extrusions. |
| `gcode.support_speed_mms` | `f32` (`5.0..=600.0` mm/s) | `80.0` | Process | Target linear speed for coarse `Support` base columns. |
| `gcode.support_interface_speed_mms` | `f32` (`5.0..=500.0` mm/s) | `45.0` | Process | Target linear speed for dense `SupportInterface` roofs/floors. |
| `gcode.travel_speed_mms` | `f32` (`20.0..=1200.0` mm/s) | `250.0` | Printer | Target linear speed for rapid non-extruding `G0` travel moves. |
| `gcode.z_hop_speed_mms` | `f32` (`2.0..=200.0` mm/s) | `25.0` | Printer | Target linear speed for clearance lifts along the local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$. |
| `gcode.outer_wall_accel_mms2` | `f32` (`100.0..=50000.0` mm/s²) | `2500.0` | Printer | Acceleration limit for `OuterWall`, `Bridge`, and `WaveOverhang` moves (`SET_VELOCITY_LIMIT ACCEL=...`). |
| `gcode.inner_wall_accel_mms2` | `f32` (`100.0..=50000.0` mm/s²) | `4500.0` | Printer | Acceleration limit for `InnerWall`, `GapFill`, and `SupportInterface` moves. |
| `gcode.infill_accel_mms2` | `f32` (`100.0..=50000.0` mm/s²) | `6500.0` | Printer | Acceleration limit for `SolidInfill`, `SparseInfill`, and `Support` moves. |
| `gcode.travel_accel_mms2` | `f32` (`100.0..=80000.0` mm/s²) | `9000.0` | Printer | Acceleration limit for rapid `G0` travel moves. |
| `gcode.square_corner_velocity_mms` | `f32` (`0.5..=50.0` mm/s) | `5.0` | Printer | Klipper `square_corner_velocity` ($v_{\text{scv}}$): instantaneous junction velocity at a $90^\circ$ corner. Used both for chained lookahead junction speeds and for computing the centripetal corner rounding radius $R_{\text{eff}}$. |
| `gcode.minimum_cruise_ratio` | `f32` (`0.0..=0.95`) | `0.50` | Printer | Klipper `minimum_cruise_ratio`: enforces a minimum constant-velocity cruise fraction on short back-to-back segments to suppress resonance. |
| `gcode.max_volumetric_flow_mm3s` | `f32` (`1.0..=100.0` mm³/s) | `18.0` | Filament | Maximum hotend volumetric melt capacity $Q_{\max}$ ($\text{mm}^3/\text{s}$). Dynamically caps segment feedrate to $v \le Q_{\max} / A_{\text{seg}}$. |
| `gcode.use_stepper_dynamics` | `bool` | `true` | Printer | Models stepper motor back-EMF torque roll-off at high speed: $a_{\text{eff}}(v) = a_0 (1 - v / v_{\max,\text{stepper}})$. |
| `gcode.max_stepper_speed_mms` | `f32` (`50.0..=2000.0` mm/s) | `400.0` | Printer | Asymptotic stepper pull-out velocity $v_{\max,\text{stepper}}$ where back-EMF reduces available acceleration torque to zero. |
| `gcode.z_travel_penalty` | `f32` (`1.0..=20.0`) | `4.0` | Printer | Weight multiplier applied to vertical $\Delta Z$ displacements when chaining toolpath islands and selecting loop seams, discouraging unnecessary non-planar jumps. |

### 8.3 Seams, 3D Conformal Scarf Joints, Wipes & Retractions

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gcode.enable_scarf_joint` | `bool` | `true` | Process | Replaces abrupt loop start/stop seams on closed perimeters with a 3D helical ramp along the local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$. |
| `gcode.scarf_length_mm` | `f32` (`1.0..=40.0` mm) | `8.0` | Process | Arc length $L_{\text{scarf}}$ along the closed perimeter over which the leading and trailing scarf wedges ramp height and flow linearly. |
| `gcode.scarf_start_height_ratio` | `f32` (`0.02..=0.50`) | `0.12` | Process | Starting fraction $r_0$ of local layer height at the leading tip of the scarf wedge (ramps from $r_0 h$ to $1.0 h$ at the start, and from $(1 - r_0) h$ down to $0$ at loop closure). |
| `gcode.seam_gap_mm` | `f32` (`0.0..=1.0` mm) | `0.06` | Process | Arc length trimmed from the end of a closed loop before the seam to prevent overlap bulges when `enable_scarf_joint = false` (or on loops shorter than $2.2\,L_{\text{scarf}}$). |
| `gcode.pre_retract_taper_mm` | `f32` (`0.0..=10.0` mm) | `1.0` | Filament | Distance before the end of an extrusion chain over which flow tapers down to bleed residual melt-chamber pressure prior to retraction. |
| `gcode.wipe_length_mm` | `f32` (`0.0..=10.0` mm) | `1.5` | Filament | Non-extruding wipe travel distance along the path before lifting the nozzle (wipes forward across closed-loop seams, or backward along open paths). |
| `gcode.retract_length_mm` | `f32` (`0.0..=10.0` mm) | `0.80` | Filament | Nominal filament retraction pullback distance before travels exceeding `min_travel_for_retract_mm`, dynamically scaled by residual PA pressure. |
| `gcode.retract_speed_mms` | `f32` (`5.0..=150.0` mm/s) | `45.0` | Filament | Extruder feedrate (`G1 E... F...`) for retract and unretract moves. |
| `gcode.min_travel_for_retract_mm` | `f32` (`0.1..=20.0` mm) | `1.50` | Filament | Minimum non-extruding travel distance required to trigger a retraction and clearance hop. |
| `gcode.z_hop_mm` | `f32` (`0.0..=5.0` mm) | `0.40` | Filament | Clearance lift distance along the local nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$ during retracted travel moves. |
| `gcode.ooze_relaxation_extra_prime_mm` | `f32` (`-0.5..=0.5` mm) | `-0.02` | Filament | Asymptotic extra prime offset $\Delta E_{\text{ooze}}$ applied on unretract as a function of travel duration $t_{\text{travel}}$: $\Delta E = \Delta E_{\text{ooze}} (1 - e^{-t_{\text{travel}} / \tau_{\text{ooze}}})$. |
| `gcode.ooze_relaxation_tau_s` | `f32` (`0.1..=30.0` s) | `1.50` | Filament | Viscoelastic relaxation time constant $\tau_{\text{ooze}}$ (seconds) for thermal ooze and die-swell decay during travel moves. |

### 8.4 Non-Newtonian Rheology, Die Swell, Shear-Thinning PA & Flow Compensation

| Canonical Key | Type / Allowed Values | Default | Preset | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gcode.enable_flat_nozzle_comp` | `bool` | `true` | Tool | Elevates the nozzle tip along $\hat{\mathbf{z}}_{\text{nozzle}}$ on sloped outer walls by $\Delta z = R_{\text{flat}}\sin\alpha\,(1 - \cos\alpha)$ so the flat brass land around the orifice never gouges the uphill side of a conformal layer. |
| `gcode.enable_dynamic_pa` | `bool` | `true` | Filament | Emits deadband-compacted `SET_PRESSURE_ADVANCE ADVANCE=...` updates as segment volumetric flow $Q$ varies across slow walls, variable-height layers, and fast infill. |
| `gcode.enable_slicer_pa_subdivision` | `bool` | `false` | Filament | Subdivides each extruding segment into entry-acceleration, cruise, and exit-deceleration sub-segments with pre-compensated $E$ values (`SET_PRESSURE_ADVANCE ADVANCE=0.0000`) for controllers without firmware PA. |
| `gcode.pa_low_flow_s` | `f32` (`0.0..=0.5` s) | `0.042` | Filament | Calibrated Pressure Advance time constant $k_{\text{low}}$ (seconds) at reference volumetric flow `pa_ref_low_flow_mm3s`. |
| `gcode.pa_high_flow_s` | `f32` (`0.0..=0.5` s) | `0.024` | Filament | Calibrated Pressure Advance time constant $k_{\text{high}}$ (seconds) at reference volumetric flow `pa_ref_high_flow_mm3s` (captures shear-thinning viscosity drop at high flow). |
| `gcode.pa_ref_low_flow_mm3s` | `f32` (`0.1..=30.0` mm³/s) | `2.0` | Filament | Lower reference volumetric flow rate $Q_{\text{low}}$ ($\text{mm}^3/\text{s}$) for power-law interpolation of $k(Q)$ and die swell $B(Q)$. |
| `gcode.pa_ref_high_flow_mm3s` | `f32` (`> pa_ref_low_flow_mm3s`) | `14.0` | Filament | Upper reference volumetric flow rate $Q_{\text{high}}$ ($\text{mm}^3/\text{s}$) for power-law interpolation of $k(Q)$ and die swell $B(Q)$. |
| `gcode.swell_low` | `f32` (`1.0..=2.0`) | `1.03` | Filament | Viscoelastic die-swell expansion ratio $B_{\text{low}} \ge 1.0$ at $Q_{\text{low}}$ (displayed in the UI as its reciprocal extrusion multiplier $1 / B_{\text{low}} \approx 0.971\times$). |
| `gcode.swell_high` | `f32` (`>= swell_low..=2.0`) | `1.10` | Filament | Viscoelastic die-swell expansion ratio $B_{\text{high}} \ge B_{\text{low}}$ at $Q_{\text{high}}$ (displayed in the UI as its reciprocal extrusion multiplier $1 / B_{\text{high}} \approx 0.909\times$). Scales down flow and insets outer walls by $\Delta y_{\text{swell}} = \frac{1}{2} w (B(Q) - 1)$ along $-\hat{\mathbf{y}}_{\text{nozzle}}$. |
| `gcode.enable_corner_flow_comp` | `bool` | `true` | Filament | Deducts both geometric inner-corner diamond overlap ($V_{\text{overlap}} = \frac{1}{4} w^2 h \tan(\theta/2)$) and Klipper SCV centripetal corner-rounding shortcut volume at sharp perimeter turns. |
| `gcode.corner_flow_comp_ratio` | `f32` (`0.0..=1.0`) | `0.85` | Filament | Scaling factor (`0.0` = disabled, `1.0` = full theoretical deduction) applied to corner overlap and SCV shortcut volume deductions. |
| `gcode.enable_transient_pressure_comp` | `bool` | `true` | Filament | Enables a first-order lag model of the hotend melt-chamber pressure state $P_{\text{state}}$ across variable-width and variable-height segments. |
| `gcode.transient_pressure_exponent` | `f32` (`0.05..=1.0`) | `0.35` | Filament | Sensitivity exponent controlling how strongly flow is throttled when stepping down from high-flow infill/thick layers to low-flow walls/thin layers while excess melt-chamber pressure discharges. |
| `gcode.transient_min_flow_multiplier` | `f32` (`0.40..=1.0`) | `0.72` | Filament | Lower clamp on the transient pressure flow multiplier so a sudden flow step-down never starves the nozzle below this fraction of nominal extrusion. |

### 8.5 Start & End G-code Templates (`gcode.*_template`)

| Canonical Key | Type | Preset | Supported Placeholders |
| :--- | :--- | :--- | :--- |
| `gcode.start_gcode_template` | `String` | Printer | `{bed_temp}`, `{nozzle_temp}`, `{tool_id}`, `{nozzle_diameter}`, `{bbox_min_x}`, `{bbox_max_x}`, `{bbox_min_y}`, `{bbox_max_y}`, `{bbox_min_z}`, `{bbox_max_z}`, `{bbox_center_x}`, `{bbox_center_y}`, `{layer_count}` |
| `gcode.end_gcode_template` | `String` | Printer | Same placeholder set as `start_gcode_template`. |

---

## 9. Top-Level Preset Tracking Fields

These four string fields record which named presets were most recently applied to the configuration:

| Canonical Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `printer_preset_name` | `String` | `"256³ CoreXY (Standard Klipper)"` | Name of the active Printer preset. |
| `tool_preset_name` | `String` | `"0.4mm Standard 5/6-Axis Conical"` | Name of the active Tool preset. |
| `filament_preset_name` | `String` | `"Generic PLA"` | Name of the active Filament preset. |
| `process_preset_name` | `String` | `"0.20mm Standard Non-Planar"` | Name of the active Process preset. |
