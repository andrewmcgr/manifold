# GUI Settings Panel Reference

The left sidebar in `manifold-gui` organizes slicing and machine parameters into logical collapsible pop-out sections.

---

## 1. Objects & Workspace
- **Object List**: Displays loaded mesh names, vertex/triangle counts, bounding dimensions, and assigned tool ID.
- **Auto-center on bed**: Calculates global AABB across all parts and translates coordinates to center the group on the bed.
- **Clear all objects**: Clears the workspace and unloads GPU vertex buffers.

---

## 2. Layering
- **Layer Height (mm)**: Nominal layer height ($0.05\text{--}0.60\text{ mm}$).
- **First Layer Height (mm)**: Thickness of the initial bed-contact layer (e.g. $0.25\text{ mm}$ on a $0.20\text{ mm}$ profile).
- **Top Solid Layers**: Number of solid shell layers covering upward-facing exposures.
- **Bottom Solid Layers**: Number of solid shell layers covering downward-facing exposures.

---

## 3. Extrusion & Walls
- **Nozzle Diameter (mm)**: Physical nozzle orifice size ($0.1\text{--}1.5\text{ mm}$).
- **Wall Line Width (mm)**: Nominal extrusion width for perimeter walls.
- **Shell Thickness (mm)**: Total thickness of solid perimeter walls ($N_{\text{walls}} = \text{round}(\text{thickness} / \text{width})$).
- **Wall Offset (mm)**: Outward perimeter offset ($0.5 \times d_{\text{nozzle}}$).
- **First Layer Line Width (mm)**: Lateral line width for bed contact squish ($1.3 \times d_{\text{nozzle}}$ default).
- **First Layer Flow Multiplier**: Volumetric extrusion multiplier on layer 0.
- **Filament Diameter (mm)**: Nominal filament diameter ($1.75\text{ mm}$ or $2.85\text{ mm}$).
- **Filament Density (g/cm³)**: Used for exact printed mass calculation ($1.24\text{ g/cm}^3$ for PLA, $1.04\text{ g/cm}^3$ for ABS/ASA, $1.27\text{ g/cm}^3$ for PETG).

---

## 4. Temperatures
- **Default Nozzle Temp (°C)**: Base target hotend temperature ($150\text{--}350^\circ\text{C}$).
- **Bed Temperature (°C)**: Target heated build plate temperature ($0\text{--}150^\circ\text{C}$).
- **Chamber Temperature (°C)**: Target heated enclosure temperature ($0\text{--}100^\circ\text{C}$, $0 = \text{unheated}$).

---

## 5. Wave Overhangs
- **Enable Wave Overhangs**: Activates 2D Huygens wave propagation for support-free horizontal overhangs (LaSO).
- **Track Overlap (mm)**: Lateral overlap distance between adjacent wave passes ($0.01\text{--}0.20\text{ mm}$, default $0.05\text{ mm}$).
- **Overhang Speed (mm/s)**: Printing speed for overhang moves ($5\text{--}100\text{ mm/s}$, default $25\text{ mm/s}$).
- **Flow Multiplier**: Teardrop bead extrusion multiplier (default $1.05$).
- **Overhang Fan Speed (%)**: Dedicated part cooling fan speed ($0\text{--}100\%$, default $100\%$).

---

## 6. Retraction & Seams
- **Use Dynamic Fluid Model**: Toggles the 2-point power-law non-Newtonian pressure advance and adaptive fluid retraction engine.
  - *When enabled*:
    - **Low-Flow PA & Q**: Calibration point $(C_{\text{PA\_low}}, Q_{\text{low}})$.
    - **High-Flow PA & Q**: Calibration point $(C_{\text{PA\_high}}, Q_{\text{high}})$.
    - **Static Break Distance (mm)**: Direct mechanical pull distance to snap capillary droplet bridge.
    - **Max Fan Temp Drop (°C)**: Maximum tip cooling drop under 100% fan.
    - **Ooze Time Constant $\tau$ (s)**: Asymptotic ooze rate.
    - **Max Ooze Prime (mm)**: Maximum extra filament re-primed after long travels.
- **Retraction Distance & Speed**: Traditional retraction parameters when fluid dynamics is disabled.
- **Pressure Advance (s)**: Static pressure advance constant.
- **Pre-Retract Taper Distance (mm)**: Bleeds melt-zone pressure before travel stops.
- **Scarf Joint Seams**: Eliminates vertical seam lines with a ramping 3.0 mm overlap lead-in and tail.
- **Wipe on Retract**: Wipes nozzle tip inward before travel lifts.
- **Use Firmware Retraction (G10/G11)**: Emits firmware retraction macros.

---

## 7. Travel & Simplification
- **Z-hop on Travel**: Lifts nozzle vertically during travel jumps to prevent dragging across printed beads.
- **Z-hop Height (mm)**: Lift clearance ($0.1\text{--}2.0\text{ mm}$).
- **Simplify Wall Toolpaths**: Ramer-Douglas-Peucker simplification on curved wall loops.
- **Simplification Tolerance (mm)**: Vertex reduction tolerance ($0.005\text{--}0.05\text{ mm}$).

---

## 8. Speeds & Accelerations
- **Speeds (mm/s)**:
  - Print Speed, Travel Speed, Outer Wall, Inner Wall, Infill, Solid Fill, Bridge Speed, First Layer Speed Limit, and Speed Deadband (%).
- **Accelerations (mm/s²)**:
  - Outer Wall, Inner Wall, Infill, Solid Fill, Bridge, Travel, First Layer, Square Corner Velocity (mm/s), and Acceleration Deadband (%).

---

## 9. Machine & Profiles
- **Bed Dimensions**: Bed X, Bed Y, and Build Height (mm).
- **Tools & Nozzles**: Multi-tool list with nozzle diameter, extrusion multiplier, and per-tool nozzle temperature.
- **Stepper Motor Dynamics**: Zero-speed acceleration ($a_0$), maximum motor velocity ($v_{\text{max}}$), acceleration limit, and speed limit.
- **Custom Gcode Macros**: Start and End G-code macro templates.
- **Save Profile / Load Profile**: Save and restore complete printer + slicer configurations as `.json` files.
