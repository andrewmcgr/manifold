# Command Line Interface (CLI) Reference

The `manifold` binary provides batch, headless, and scriptable 3D slicing for FDM 3D printers.

---

## Synopsis

```sh
manifold [OPTIONS] <INPUTS>... -o <OUTPUT>
```

- `<INPUTS>...`: One or more input mesh files (`.stl` or `.3mf`).
- `-o, --output <OUTPUT>`: Output G-code destination path (default: `out.gcode`).

---

## Multi-File & Multi-Tool Syntax

Manifold supports slicing multiple object files simultaneously. Each input path can optionally specify an explicit tool assignment using the `:tool_id` suffix:

```sh
manifold base.stl:0 accent.stl:1 insert.stl:0 -o multi_material.gcode
```

- If no `:tool_id` suffix is provided, the object is automatically assigned to Tool `0`.
- Manifold automatically centers the combined workspace onto the build plate and generates sequential tool changes with extruder retraction and unretraction.

---

## Command Line Options Reference

### General Options

| Flag | Description | Default |
|---|---|---|
| `-o, --output <PATH>` | Output G-code file path. | `out.gcode` |
| `-h, --help` | Print help message and option descriptions. | — |
| `-V, --version` | Print version information. | — |

---

### Layering & Extrusion Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--layer-height <MM>` | Float | `0.20` | Nominal non-planar layer thickness in millimeters. |
| `--nozzle-diameter <MM>` | Float | `0.40` | Default nozzle orifice diameter in millimeters. |

---

### Order Fields & Non-Planar Geometry

| Option | Values | Default | Description |
|---|---|---|---|
| `--order-field <KIND>` | `height`, `conical`, `eikonal` | `height` | Slicing order field geometry. |
| `--eikonal-slope-profile <POINTS>` | `height_mm:max_deg,...` | *(unconstrained)* | Series of toolhead clearance envelope points (`x:z` or `height:angle`). E.g. `"0:45,4:2"` or `"0:0,15:5,35:20"`. |
| `--eikonal-conform-top-surfaces` | Flag | `false` | Monotonically warp Eikonal field parallel to upward-facing exterior surfaces. |
| `--eikonal-conform-bottom-surfaces` | Flag | `false` | Warp Eikonal field to conform to downward-facing (overhang) surfaces. |
| `--eikonal-conformal-top-angle <DEG>` | Float | `45.0` | Detach angle (degrees from horizontal) for top surface tracking; steeper surfaces revert to bulk slicing. |
| `--eikonal-conformal-bottom-angle <DEG>` | Float | `30.0` | Detach angle (degrees from horizontal) for bottom surface tracking; steeper surfaces revert to bulk slicing. |

---

### Infill & Overhangs

| Option | Values / Type | Default | Description |
|---|---|---|---|
| `--sparse-infill-pattern <KIND>` | `monotonic`, `concentric`, `all-walls`, `cubic`, `gyroid`, `schwarz-d`, `schwarz-p`, `none` | `cubic` | Infill pattern for interior sparse regions. |
| `--solid-infill-pattern <KIND>` | `monotonic`, `concentric`, `all-walls`, `cubic`, `gyroid`, `schwarz-d`, `schwarz-p`, `none` | `all-walls` | Infill pattern for solid top/bottom exposure layers. |
| `--infill-pattern <KIND>` | *(same as above)* | `cubic` | Legacy unified infill pattern setting. |
| `--wave-overhangs` | Flag | `true` | Enable support-free Huygens wave-propagation overhangs (LaSO). |
| `--wave-overhang-overlap <MM>` | Float | `0.05` | Lateral track overlap distance (mm) between wave passes. |
| `--wave-overhang-speed <MM_S>` | Float | `25.0` | Printing speed for overhang moves in mm/s ($1500\text{ mm/min}$). |
| `--wave-overhang-flow <MULT>` | Float | `1.05` | Teardrop bead extrusion flow multiplier for overhangs. |

---

### Speeds & Kinematics

| Option | Type | Default | Description |
|---|---|---|---|
| `--square-corner-velocity <MM_S>` | Float | `5.0` | Klipper Square Corner Velocity (SCV) limit (mm/s). |
| `--speed-deadband <PCT>` | Float | `10.0` | Speed deadband percentage to compact G-code feedrate outputs. |
| `--acceleration-deadband <PCT>` | Float | `20.0` | Acceleration deadband percentage to compact `SET_VELOCITY_LIMIT` outputs. |

---

### Temperatures & Cooling

| Option | Type | Default | Description |
|---|---|---|---|
| `--nozzle-temp <C>` | Float | `240.0` | Target nozzle/hotend temperature (°C). |
| `--bed-temp <C>` | Float | `60.0` | Target heated bed temperature (°C). |
| `--chamber-temp <C>` | Float | `0.0` | Target heated chamber temperature (°C, 0 = unheated). |
| `--fan-speed <PCT>` | Float | `100.0` | Part cooling fan speed percentage (0..100). |
| `--overhang-fan-speed <PCT>` | Float | `100.0` | Dedicated part cooling fan speed percentage for overhangs (0..100). |
| `--fan-layer-delay <N>` | Integer | `1` | Number of initial layers to keep cooling fan disabled (`M106 S0`). |

---

### Retraction & Fluid Dynamics

| Option | Type | Default | Description |
|---|---|---|---|
| `--fluid-dynamics` | Flag | `false` | Enable non-Newtonian 2-point PA, extrudate swell compensation, and adaptive fluid retraction. |
| `--static-retraction <MM>` | Float | `0.15` | Static mechanical break-away distance (mm) under fluid dynamics. |

---

## Batch Processing Examples

### Shell Script: Batch Slicing with Conformal Eikonal & TPMS Infill

```bash
#!/usr/bin/env bash
set -euo pipefail

PROFILE_SLOPE="0:45,5:15,20:5"

for stl in models/*.stl; do
  name=$(basename "$stl" .stl)
  echo "Slicing $name..."
  manifold "$stl" \
    --order-field eikonal \
    --eikonal-slope-profile "$PROFILE_SLOPE" \
    --eikonal-conform-top-surfaces \
    --eikonal-conform-bottom-surfaces \
    --sparse-infill-pattern gyroid \
    --solid-infill-pattern all-walls \
    --layer-height 0.20 \
    --wave-overhangs \
    --fluid-dynamics \
    --nozzle-temp 245 \
    --bed-temp 105 \
    -o "gcode/${name}.gcode"
done
```
