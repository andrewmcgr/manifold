# Manifold Slicer — Command-Line Interface (CLI) Guide

`manifold-frontend` provides both an interactive 3D GUI and a full-featured headless Command-Line Interface (CLI) for slicing `.step`, `.stp`, `.stl`, and `.3mf` models to G-code, as well as creating, inspecting, diffing, and editing JSON/TOML slicer profiles and saved presets.

---

## 1. Invoking the GUI vs. Headless CLI

- **Launch Interactive 3D GUI (Default)**:
  ```bash
  manifold-frontend
  ```
- **Launch GUI with Preloaded Files and/or Profile Overrides**:
  ```bash
  manifold-frontend gui -p my_profile.json part1.step part2.stl
  # Or pass 3D files directly without `-o` / `slice`:
  manifold-frontend project.3mf
  ```
- **Run Headless CLI Commands**:
  Passing `slice`, `calibrate` (or alias `calibration`), `profile` (or aliases `preset`, `config`), `--help`, or `--version` runs headlessly in the terminal and exits without opening a window:
  ```bash
  manifold-frontend --help
  manifold-frontend slice --help
  manifold-frontend calibrate --help
  manifold-frontend profile --help
  ```

---

## 2. Headless Slicing (`manifold-frontend slice`)

### Synopsis

```bash
manifold-frontend slice [OPTIONS] <INPUT_FILES>...
```

### Supported Input Formats

- **`.step` / `.stp`**: Exact OpenCASCADE (OCCT) B-Rep CAD parts and multi-part assemblies (automatically decomposed into independent parts while preserving assembly relative placement).
- **`.stl`**: Binary and ASCII triangle meshes.
- **`.3mf`**: Single-plate and multi-plate 3MF project archives (including Manifold2 `.3mf` projects and third-party multi-plate `.3mf` files from OrcaSlicer, Bambu Studio, and PrusaSlicer).

### Slicing Options

| Option | Description |
| :--- | :--- |
| `-p, --profile <FILE>` | Load a full or partial profile from a `.json` or `.toml` file. Repeatable; later files override earlier ones. |
| `--printer <NAME_OR_FILE>` | Apply a **Printer** preset by name (factory or user-saved) or from a `.json`/`.toml` file. |
| `--tool <NAME_OR_FILE>` | Apply a **Tool** preset by name or file path. |
| `--filament <NAME_OR_FILE>` | Apply a **Filament** preset by name or file path. |
| `--process <NAME_OR_FILE>` | Apply a **Process** preset by name or file path. |
| `-s, --set <KEY=VALUE>` / `-D <KEY=VALUE>` | Override any parameter by dotted path or unambiguous leaf key name (repeatable). |
| `-o, --output <PATH>` | Destination `.gcode` file path, output directory (for multi-plate projects), or `-` for `stdout`. Defaults to `<first_input_stem>.gcode` (or `<stem>_plate<N>.gcode` when slicing multiple plates). |
| `--save-3mf <PATH>` | Also export the configured and arranged build plate(s) as a Manifold `.3mf` project archive. |
| `--plate <INDEX_OR_NAME>` | Slice only a specific 1-based plate number (e.g. `--plate 1`) or plate name (e.g. `--plate "Plate 2"`). By default, all non-empty plates are sliced. |
| `--arrange` / `--no-arrange` | Force or disable Archimedean spiral auto-arrangement on each plate before slicing. By default, auto-arrangement runs automatically when multiple objects on a plate have overlapping XY bounding boxes. |
| `--place-on-face <FACE_INDEX>` | Orient imported object(s) so that simplified convex-hull face `#FACE_INDEX` (`0..26`, ordered by descending facet area) rests flat on the print bed (`z = 0`). |
| `--rotate <RX,RY,RZ>` | Apply Euler XYZ rotation (in degrees, e.g. `--rotate 0,0,45`) to imported object(s) and drop them onto the print bed (`z = 0`) before slicing. |
| `-q, --quiet` | Suppress progress diagnostics on `stderr`. |

### Configuration Precedence During `slice`

When slicing, configuration is layered in the following order (later stages override earlier ones):

1. Base `SlicerConfig::default()` (or the embedded `/Metadata/manifold2_config.toml` if an input file is a saved Manifold `.3mf` project).
2. `--profile <FILE>` arguments, applied in the order given on the command line.
3. Category presets (`--printer`, `--tool`, `--filament`, `--process`).
4. Inline `-s` / `--set` / `-D` `KEY=VALUE` overrides, applied in order.
5. Automatic sanitization and synchronization (`sync_auto_voxel_size` and bed-tilt gravity vector updates).

### Slicing Examples

```bash
# 1. Slice a STEP CAD model with a JSON profile:
manifold-frontend slice -p profile.json "Test Objects/testobj1.step" -o testobj1.gcode

# 2. Slice an STL with named presets and inline parameter overrides:
manifold-frontend slice \
  --printer "5-Axis Tilting Build Plate (250×210)" \
  --tool "0.6mm High-Flow" \
  --filament "Generic PETG" \
  --process "0.20mm Standard Non-Planar" \
  -s slicing.layer_height.target_mm=0.18 \
  -s infill_density_percent=30 \
  bracket.stl -o bracket.gcode

# 3. Slice Plate 2 of a multi-plate OrcaSlicer 3MF file:
manifold-frontend slice "Test Objects/Iron heart MOD (deepergrain).3mf" --plate 2 -o plate2.gcode

# 4. Auto-arrange multiple STL files onto one plate, save the 3MF project, and slice to G-code:
manifold-frontend slice --arrange --save-3mf batch_plate.3mf -o batch_plate.gcode part_a.stl part_b.stl part_c.stl

# 5. Slice with Auto Conformal SDF Surface Skins and Obstacle-Avoiding Non-Planar Tree Supports:
manifold-frontend slice \
  -s slicing.enable_auto_conformal_skin=true \
  -s slicing.conformal_skin_layers=3 \
  -s slicing.support_style=tree \
  -s slicing.support_on_bed_only=true \
  benchy.stl -o benchy_conformal.gcode
```

---

## 3. Polymer Rheology & Flow Calibration (`manifold-frontend calibrate`)

Generates native, kinematics-aware G-code test patterns for calibrating Extrusion Multiplier (`em`), Shear-Thinning Pressure Advance (`pa`), High-Flow Viscoelastic Die Swell (`em --pass 3`), and Transient Melt-Zone Pressure Compensation (`transient`). For visual reading guides and worked examples, see **[`docs/calibration_guide.md`](calibration_guide.md)**.

### Synopsis

```bash
manifold-frontend calibrate <em|pa|transient> [OPTIONS]
```

### Calibration Options

| Option | Description |
| :--- | :--- |
| `<em\|pa\|transient>` | Calibration utility to generate:<br>• `em`: Extrusion Multiplier & High-Flow Die Swell coupons<br>• `pa`: Ellis $90^\circ$ Chevron Pressure Advance pattern<br>• `transient`: Constant-speed cross-section step-down ribs for $\gamma$ and $\tau_{\text{melt}}$ |
| `-o, --output <PATH>` | Output `.gcode` file path (or `-` for `stdout`). Defaults to `calibration_<type>.gcode`. |
| `-p, --profile <FILE>` | Load base `.json` or `.toml` profile file (repeatable). |
| `--printer / --tool / --filament / --process <NAME_OR_FILE>` | Apply category preset(s) before generating the calibration pattern. |
| `-t, --tool-id <ID_OR_NAME>` / `--tool-index <IDX>` | Select which extruder toolhead in `config.tools` to calibrate (e.g. `-t T0`, `-t T1`, `-t 1`, or `--tool-index 1`). Uses that tool's nozzle diameter, hotend temperature, `NozzleFrame`, and `{tool_id}` placeholder. |
| `-s, --set <KEY=VALUE>` | Override any `SlicerConfig` parameter inline (repeatable). |
| `--pass <1\|2\|3>` | Pass selection:<br>• For `em`: `1` = Coarse (`-10%..+10%`), `2` = Fine (`-2%..+2%`), `3` = High-Flow Die Swell (`0.86..0.98`)<br>• For `transient`: `1` = Stage 1 Exponent $\gamma$, `2` = Stage 2 Relaxation Time $\tau_{\text{melt}}$ |
| `--center <FLOAT>` | Center extrusion multiplier for `calibrate em --pass 2` (defaults to `gcode.extrusion_multiplier`). |
| `--mode <dual\|single>` | Mode for `calibrate pa`: `dual` (default, prints both `LOW Q` and `HIGH Q` chevron banks) or `single` (prints a single bank at `--fast-speed`). |
| `--start <FLOAT>` / `--end <FLOAT>` / `--step <FLOAT>` | Parameter sweep range and step size (`s` for `pa` and `transient --pass 2`, dimensionless $\gamma$ for `transient --pass 1`). |
| `--fast-speed <MM_S>` / `--slow-speed <MM_S>` / `--accel <MM_S2>` | Override test speeds and acceleration for `calibrate pa` or `calibrate transient`. |
| `--adhesion-ribbons` / `--no-adhesion-ribbons` | Enable or disable optional Layer 0 apex stabilization ribbons on `calibrate pa` (disabled by default for maximum corner contrast against the build plate). |

### Calibration Examples

```bash
# Generate Dual-Band Shear-Thinning PA pattern for Tool T1 (0.6mm High-Flow):
manifold-frontend calibrate pa -p my_profile.json -t T1 --mode dual -o calib_pa_t1.gcode

# Generate Stage 1 (γ) and Stage 2 (τ_melt) Transient Melt Pressure patterns:
manifold-frontend calibrate transient -p my_profile.json -t T0 --pass 1 -o calib_transient_gamma.gcode
manifold-frontend calibrate transient -p my_profile.json -t T0 --pass 2 -s transient_pressure_exponent=0.34 -o calib_transient_tau.gcode
```

---

## 4. Profile & Preset Management (`manifold-frontend profile`)

The `profile` command (also accessible via aliases `preset`, `presets`, `profiles`, or `config`) provides structured verbs for creating, inspecting, comparing, validating, and editing `.json` and `.toml` profiles as well as user presets stored in `~/.config/manifold2/presets/`.

### 3.1 `profile init` — Create a New Profile File

Generates a complete `.json` (default) or `.toml` profile file from factory defaults, named presets, existing profile files, and/or `-s` key overrides.

```bash
# Create a full JSON profile from defaults:
manifold-frontend profile init default_profile.json

# Create a JSON profile pre-configured for PETG on a 5-axis tilting bed with 0.16 mm target layers:
manifold-frontend profile init petg_5axis.json \
  --printer "5-Axis Tilting Build Plate (250×210)" \
  --filament "Generic PETG" \
  -s layer_height.target_mm=0.16 \
  -s enable_support_generation=true

# Export only a single preset category (printer | tool | filament | process) as JSON or TOML:
manifold-frontend profile init my_filament.json --category filament --filament "Generic ABS"
manifold-frontend profile init my_process.toml --category process --process "0.12mm Fine Conformal"
```

### 3.2 `profile keys` — Discover All Editable Parameters

Lists all dotted parameter paths in `SlicerConfig`, their default values, physical units, and documentation summaries from the built-in parameter catalog.

```bash
# List all parameter keys and summaries:
manifold-frontend profile keys

# Filter keys or documentation by keyword (case-insensitive):
manifold-frontend profile keys --filter overhang

# Include detailed physical/geometric tuning explanations:
manifold-frontend profile keys --filter eikonal --docs
```

### 3.3 `profile get` — Read a Parameter from a Profile or Preset

Reads any parameter (or nested sub-object) from a `.json`/`.toml` file, a named preset, or `"default"`.

```bash
# Query a specific dotted path from a JSON profile:
manifold-frontend profile get petg_5axis.json slicing.layer_height.target_mm

# Query using an unambiguous shorthand leaf key:
manifold-frontend profile get petg_5axis.json bed_temperature_celsius

# Query an entire sub-object (prints formatted JSON):
manifold-frontend profile get petg_5axis.json slicing.layer_height

# Query a value from a built-in or user preset by name:
manifold-frontend profile get "Generic PETG" pa_low_flow_s
```

### 3.4 `profile set` — Edit Parameters in a Profile File or Preset

Updates one or more parameters in-place in a `.json` or `.toml` profile file (or writes to a new file with `-o <OUT_FILE>`). Every edit is type-checked against `SlicerConfig`, sanitized to valid physical ranges, and automatically updates coupled fields:
- Editing `machine.bed_tilt_rx_deg` or `machine.bed_tilt_ry_deg` automatically recomputes `machine.gravity_direction_bed` in bed coordinates.
- Editing `slicing.layer_height.target_mm` or `tools.0.nozzle_diameter_mm` while `eikonal.auto_voxel_size` is enabled automatically updates `eikonal.voxel_size_mm`.
- Explicitly setting `eikonal.voxel_size_mm` automatically disables `eikonal.auto_voxel_size` so the manual voxel size is preserved.

```bash
# Edit multiple parameters in-place in a JSON profile:
manifold-frontend profile set petg_5axis.json \
  slicing.layer_height.target_mm=0.15 \
  infill_density_percent=25 \
  gcode.bed_temperature_celsius=80 \
  machine.bed_tilt_rx_deg=15

# Edit a profile and write the result to a new file without modifying the original:
manifold-frontend profile set petg_5axis.json -o petg_5axis_fast.json outer_wall_speed_mms=90
```

### 3.5 `profile diff` — Compare Profiles or Inspect Modifications

Shows all leaf parameters that differ between a profile/preset and `SlicerConfig::default()`, or between two profiles/presets `<A>` and `<B>`.

```bash
# Show what changed in `petg_5axis.json` relative to factory defaults:
manifold-frontend profile diff petg_5axis.json

# Compare two profile files or presets:
manifold-frontend profile diff default_profile.json petg_5axis.json
manifold-frontend profile diff "0.20mm Standard Non-Planar" "0.12mm Fine Conformal"
```

### 3.6 `profile show` — Display a Full Profile or Preset

Prints a profile file, named preset, or `"default"` to `stdout` as JSON or TOML.

```bash
# Print a preset as JSON:
manifold-frontend profile show "0.12mm Fine Conformal" --format json

# Print a specific section of a profile file:
manifold-frontend profile show petg_5axis.json --key eikonal
```

### 3.7 `profile list`, `save-preset`, `delete-preset`, and `validate`

```bash
# List all factory and user-saved presets across all 4 categories (or filter with `--category`):
manifold-frontend profile list
manifold-frontend profile list --category filament

# Save a new user preset into ~/.config/manifold2/presets/<category>/:
manifold-frontend profile save-preset filament "My Carbon PETG" \
  --from petg_5axis.json \
  -s nozzle_temperature_celsius=255 \
  -s bed_temperature_celsius=85

# Delete a user-saved preset:
manifold-frontend profile delete-preset filament "My Carbon PETG"

# Validate one or more .json / .toml profile files:
manifold-frontend profile validate petg_5axis.json
```

---

## 4. Partial JSON / Shorthand Profile Files

When passing `-p / --profile <FILE>` to `slice` (or `profile validate` / `profile show`), the JSON or TOML file does **not** need to contain every field of `SlicerConfig`. You can pass a minimal JSON file containing only the sections or shorthand keys you want to override:

```json
{
  "slicing": {
    "layer_height": {
      "target_mm": 0.16,
      "min_mm": 0.08,
      "max_mm": 0.24
    },
    "infill_density_percent": 25.0
  },
  "nozzle_diameter_mm": 0.4,
  "bed_temperature_celsius": 70.0,
  "bed_tilt_rx_deg": 20.0
}
```
