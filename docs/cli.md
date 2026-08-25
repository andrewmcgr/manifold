# Command Line Interface (CLI) Reference

The `manifold` binary provides batch, headless, and scriptable 3D slicing for FDM 3D printers.

---

## Synopsis

```sh
manifold [OPTIONS] <INPUTS>... -o <OUTPUT>
```

- `<INPUTS>...`: One or more input mesh files (`.stl` or `.3mf`).
- `-o, --output <OUTPUT>`: Output G-code destination path (required).

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
| `-o, --output <PATH>` | Output G-code file path. *(Required)* | — |
| `-h, --help` | Print help message and option descriptions. | — |
| `-V, --version` | Print version information. | — |

---

### Layering & Extrusion Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--layer-height <MM>` | Float | `0.20` | Nominal non-planar layer thickness in millimeters. |
| `--first-layer-height <MM>` | Float | *(layer-height)* | Layer thickness for the first bed-contact layer. |
| `--first-layer-width <MM>` | Float | `1.3 * nozzle` | Lateral bead line width for first-layer squish (mm). |
| `--first-layer-speed <MM_S>` | Float | `0.4 * speed` | Maximum printing speed on layer 0 (mm/s). |
| `--first-layer-flow <MULT>` | Float | `1.0` | Volumetric extrusion flow multiplier for layer 0. |
| `--nozzle-diameter <MM>` | Float | `0.40` | Default nozzle orifice diameter in millimeters. |
| `--wall-line-width <MM>` | Float | `0.40` | Nominal extrusion width for perimeters/walls. |
| `--shell-thickness <MM>` | Float | `0.80` | Total shell wall thickness in millimeters. |
| `--wall-offset <MM>` | Float | `0.20` | Outermost wall offset from geometric surface ($0.5 \times d_{\text{nozzle}}$). |
| `--solid-layers-top <N>` | Integer | `3` | Number of solid shell layers covering top surfaces. |
| `--solid-layers-bottom <N>` | Integer | `3` | Number of solid shell layers covering bottom surfaces. |

---

### Order Fields & Non-Planar Geometry

| Option | Values | Default | Description |
|---|---|---|---|
| `--order-field <KIND>` | `height`, `conical`, `eikonal` | `height` | Slicing order field geometry. |
| `--eikonal-slope-profile <POINTS>` | `x:z,x:z,...` | *(unconstrained)* | Series of toolhead clearance envelope points (mm). E.g. `"0:0,15:5,35:20"`. |
| `--eikonal-conform-top-surfaces` | Flag | `false` | Monotonically warp Eikonal field parallel to top exterior surfaces. |

---

### Infill & Overhangs

| Option | Values / Type | Default | Description |
|---|---|---|---|
| `--sparse-infill-pattern <KIND>` | `none`, `all-walls`, `concentric`, `cubic` | `cubic` | Infill pattern for interior sparse regions. |
| `--solid-infill-pattern <KIND>` | `none`, `all-walls`, `concentric`, `cubic` | `all-walls` | Infill pattern for solid top/bottom exposure layers. |
| `--infill-density <FRACTION>` | Float (`0.0`..`1.0`) | `0.20` | Interior sparse infill volume density. |
| `--infill-line-width <MM>` | Float | `0.40` | Extrusion line width for infill scanlines. |
| `--wave-overhangs` | Flag | `true` | Enable support-free Huygens wave-propagation overhangs. |
| `--wave-overhang-overlap <MM>` | Float | `0.05` | Lateral track overlap distance (mm) between wave passes. |
| `--wave-overhang-speed <MM_S>` | Float | `25.0` | Printing speed for overhang moves in mm/s. |
| `--wave-overhang-flow <MULT>` | Float | `1.05` | Teardrop bead extrusion flow multiplier for overhangs. |

---

### Speeds & Kinematics

| Option | Type | Default | Description |
|---|---|---|---|
| `--print-speed <MM_S>` | Float | `155.0` | Nominal printing speed in mm/s ($9300\text{ mm/min}$). |
| `--travel-speed <MM_S>` | Float | `200.0` | Non-extruding travel move speed in mm/s ($12000\text{ mm/min}$). |
| `--outer-wall-speed <MM_S>` | Float | `110.0` | Outer perimeter wall speed in mm/s ($6600\text{ mm/min}$). |
| `--inner-wall-speed <MM_S>` | Float | `155.0` | Inner perimeter wall speed in mm/s ($9300\text{ mm/min}$). |
| `--infill-speed <MM_S>` | Float | `155.0` | Infill printing speed in mm/s ($9300\text{ mm/min}$). |
| `--solid-infill-speed <MM_S>` | Float | `130.0` | Solid top/bottom fill speed in mm/s ($7800\text{ mm/min}$). |
| `--bridge-speed <MM_S>` | Float | `50.0` | Bridging speed in mm/s ($3000\text{ mm/min}$). |
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
| `--fluid-dynamics` | Flag | `false` | Enable non-Newtonian 2-point PA and adaptive fluid retraction. |
| `--static-retraction <MM>` | Float | `0.15` | Static mechanical break-away distance (mm) under fluid dynamics. |

---

## Batch Processing Examples

### Shell Script: Batch Slicing a Directory of Parts

```bash
#!/usr/bin/env bash
set -euo pipefail

PROFILE_SLOPE="0:0,12:4,28:15,50:35"

for stl in models/*.stl; do
  name=$(basename "$stl" .stl)
  echo "Slicing $name..."
  manifold "$stl" \
    --order-field eikonal \
    --eikonal-slope-profile "$PROFILE_SLOPE" \
    --layer-height 0.20 \
    --sparse-infill-pattern cubic \
    --infill-density 0.15 \
    --wave-overhangs \
    --fluid-dynamics \
    --nozzle-temp 245 \
    --bed-temp 105 \
    -o "gcode/${name}.gcode"
done
```
