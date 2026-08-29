# Configuration & Profile Reference

Manifold encapsulates complete printer, material, and slicing configurations into a JSON structure (`Profile`), saved and loaded via `.json` profile files.

---

## Profile JSON Structure

A `Profile` JSON bundles two top-level objects:
1. **`machine`**: Physical printer kinematics, build envelope, toolhead clearance envelope, tools, per-axis limits, and stepper motor models.
2. **`config`**: Slicing parameters, layer heights, infill patterns (TPMS / Cubic), conformal Eikonal settings, wave overhang parameters, speeds, accelerations, and fluid dynamics properties.

```json
{
  "machine": {
    "substrate_transform": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "build_volume": {
      "Aabb": {
        "min": [0.0, 0.0, 0.0],
        "max": [300.0, 300.0, 300.0]
      }
    },
    "tools": [
      {
        "id": 0,
        "nozzle_diameter": 0.4,
        "nozzle_flat_diameter": 0.8,
        "mount": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "collision_envelope": {
          "Sphere": { "center": [0.0, 0.0, 0.0], "radius": 0.0 }
        },
        "extrusion_multiplier": 1.0,
        "nozzle_temperature": 245.0
      }
    ],
    "axis_count": 3,
    "eikonal_slope_profile": [
      [0.0, 0.0],
      [15.0, 5.0],
      [35.0, 20.0]
    ],
    "use_stepper_dynamics": true,
    "zero_speed_acceleration": 20000.0,
    "max_available_speed": 1000.0,
    "acceleration_limit": 10000.0,
    "speed_limit": 750.0,
    "axis_limits": {
      "Z": {
        "speed_limit": 40.0,
        "acceleration_limit": 1500.0,
        "use_stepper_dynamics": true,
        "zero_speed_acceleration": 3000.0,
        "max_available_speed": 60.0
      }
    }
  },
  "config": {
    "layer_height": 0.20,
    "first_layer_height": 0.25,
    "first_layer_line_width": 0.52,
    "wall_line_width": 0.4,
    "shell_thickness": 0.8,
    "wall_offset": 0.2,
    "solid_layers_top": 3,
    "solid_layers_bottom": 3,
    "sparse_infill_pattern": "Gyroid",
    "solid_infill_pattern": "AllWalls",
    "infill_density": 0.20,
    "infill_line_width": 0.40,
    "order_field": "Eikonal",
    "eikonal_surface_order_weight": 1.0,
    "eikonal_conform_top_surfaces": true,
    "eikonal_conformal_max_angle_deg": 45.0,
    "eikonal_conform_bottom_surfaces": true,
    "eikonal_conformal_bottom_max_angle_deg": 30.0,
    "eikonal_conformal_skin_depth_mm": 1.2,
    "eikonal_enforce_monotonic_growth": true,
    "wave_overhangs_enabled": true,
    "wave_overhang_overlap": 0.05,
    "wave_overhang_speed": 1500.0,
    "wave_overhang_flow": 1.05,
    "scarf_joint_enabled": true,
    "scarf_joint_length": 3.0,
    "fluid_dynamics": {
      "pa_calibration_low": [0.045, 2.0],
      "pa_calibration_high": [0.030, 15.0],
      "heater_block_temp_c": 245.0,
      "reference_temp_c": 240.0,
      "max_fan_temp_drop_c": 8.0,
      "ooze_time_constant_ref_s": 1.5,
      "ooze_max_length_ref_mm": 0.30,
      "static_retraction_mm": 0.25,
      "max_retraction_mm": 1.5,
      "pa_deadband": 0.10,
      "swell_ratio_low": 1.05,
      "swell_ratio_high": 1.20
    }
  }
}
```

---

## Configuration Modules

- [Layering & Extrusion](layering-extrusion.md) — Layer heights, first layer squish, wall counts, and filament properties.
- [Order Fields & Clearance Envelopes](order-fields.md) — Fast Marching Eikonal fields, top/bottom surface conforming, and toolhead clearance profiles.
- [Infill Patterns](infill.md) — TPMS (Gyroid, Schwarz D, Schwarz P), 3D Cubic Lattice, and offset perimeter infills.
- [Wave Overhangs](wave-overhangs.md) — 2D & 3D Huygens wave-propagation theory (LaSO), lateral anchoring, and teardrop bead math.
- [Kinematics & Stepper Dynamics](kinematics.md) — Speeds, accelerations, per-axis limits, SCV, and motor ODE modeling.
- [Retraction & Fluid Dynamics](retraction-fluid.md) — Scarf joints, extrudate swell, and non-Newtonian pressure advance.
- [Machine & Tools](machine-tools.md) — Printer build volume, multi-tool setups, flat nozzle diameters, and temperatures.
