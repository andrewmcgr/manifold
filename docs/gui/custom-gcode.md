# Custom G-code Macros & Placeholders

Manifold allows customizing the **Start G-code** and **End G-code** scripts emitted in the generated program. During slicing, template placeholders wrapped in `{...}` are automatically evaluated and replaced with print-specific metadata and temperatures.

---

## Available Template Placeholders

### 1. Bounding Box Placeholders (Adaptive Bed Mesh / Purge Area)

These placeholders evaluate to the exact XY bounding box of the **first printed layer** (formatted to 3 decimal places):

| Placeholder | Description | Example Output |
|---|---|---|
| `{print_min_x}` | Minimum X coordinate of first layer | `152.420` |
| `{print_min_y}` | Minimum Y coordinate of first layer | `148.110` |
| `{print_max_x}` | Maximum X coordinate of first layer | `198.650` |
| `{print_max_y}` | Maximum Y coordinate of first layer | `204.300` |

---

### 2. Temperature Placeholders

| Placeholder | Aliases | Description | Example Output |
|---|---|---|---|
| `{bed_temperature}` | `{bed_temp}`, `{first_layer_bed_temperature}` | Target heated bed temperature in °C | `105` |
| `{chamber_temperature}` | `{chamber_temp}` | Target heated chamber temperature in °C | `50` |
| `{first_used_tool_temperature}` | `{nozzle_temperature}`, `{nozzle_temp}`, `{first_layer_temperature}` | Nozzle temperature of the initial printing tool | `245` |
| `{temperature_0}` | `{temperature[0]}`, `{nozzle_temperature_0}`, `{nozzle_temp_0}` | Target nozzle temperature for Tool 0 | `240` |
| `{temperature_1}` | `{temperature[1]}`, `{nozzle_temperature_1}`, `{nozzle_temp_1}` | Target nozzle temperature for Tool 1 | `255` |

---

### 3. Tool Selection Placeholders

| Placeholder | Aliases | Description | Example Output |
|---|---|---|---|
| `{first_used_tool}` | `{initial_tool}`, `{first_used_extruder}`, `{initial_extruder}` | Tool index of the first extruding path | `0` |

---

## Example Start G-code Macro for Klipper

```gcode
; Start G-code for Klipper
M140 S{bed_temperature}
M104 T{first_used_tool} S150 ; Pre-heat nozzle without oozing
M190 S{bed_temperature}
M141 S{chamber_temperature}

G28 ; Home all axes
BED_MESH_CALIBRATE PRINT_MIN={print_min_x},{print_min_y} PRINT_MAX={print_max_x},{print_max_y}

M109 T{first_used_tool} S{first_used_tool_temperature} ; Final hotend heating
VORON_PURGE PRINT_MIN={print_min_x},{print_min_y} PRINT_MAX={print_max_x},{print_max_y}
```

## Example End G-code Macro

```gcode
; End G-code
M400 ; Finish all moves
M104 S0 ; Turn off hotend
M140 S0 ; Turn off bed
M141 S0 ; Turn off chamber
M106 S0 ; Turn off cooling fan
G91
G1 Z10 F600 ; Lift nozzle
G90
G1 X10 Y290 F18000 ; Present print
M84 ; Disable steppers
```
