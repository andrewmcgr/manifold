# GUI Settings Panel Reference

The left sidebar in `manifold-gui` organizes slicing and machine parameters into logical collapsible sections.

---

## 1. Layering
- **Layer Height (mm)**: Nominal non-planar layer thickness ($0.05\text{--}1.00\text{ mm}$).
- **First Layer Height (mm)**: Thickness of the initial bed-contact layer ($0.05\text{--}1.00\text{ mm}$).
- **Top Solid Layers**: Number of solid shell layers covering upward-exposed surfaces.
- **Bottom Solid Layers**: Number of solid shell layers covering downward-exposed overhang surfaces.

---

## 2. Extrusion & Walls
- **Wall Line Width (mm)**: Nominal extrusion line width for perimeter walls ($0.05\text{--}1.50\text{ mm}$).
- **Shell Thickness (mm)**: Total thickness of solid perimeter walls ($N_{\text{walls}} = \text{round}(\text{thickness} / \text{width})$).
- **Wall Offset (mm)**: Outward perimeter offset ($0.5 \times d_{\text{nozzle}}$).
- **First Layer Line Width (mm)**: Lateral line width for bed contact squish ($1.3 \times d_{\text{nozzle}}$ default).
- **First Layer Flow Multiplier**: Volumetric extrusion multiplier on layer 0 ($0.5\text{--}2.0$).
- **Filament Diameter (mm)**: Nominal raw filament diameter ($1.75\text{ mm}$ or $2.85\text{ mm}$).
- **Filament Density (g/cm³)**: Used for exact printed mass calculation ($1.24\text{ g/cm}^3$ for PLA, $1.04\text{ g/cm}^3$ for ABS/ASA, $1.27\text{ g/cm}^3$ for PETG).

---

## 3. Temperatures
- **Default Nozzle Temp (°C)**: Base target hotend temperature ($150\text{--}350^\circ\text{C}$).
- **Bed Temperature (°C)**: Target heated build plate temperature ($0\text{--}150^\circ\text{C}$).
- **Chamber Temperature (°C)**: Target heated enclosure temperature ($0\text{--}100^\circ\text{C}$, $0 = \text{unheated}$).

---

## 4. Wave Overhangs
- **Enable Wave Overhangs**: Activates 2D/3D Huygens wave propagation for support-free horizontal overhangs (LaSO).
- **Track Overlap (mm)**: Lateral overlap distance between adjacent wave passes ($0.01\text{--}0.20\text{ mm}$, default $0.05\text{ mm}$).
- **Overhang Speed (mm/s)**: Printing speed for overhang moves ($5\text{--}100\text{ mm/s}$, default $25\text{ mm/s}$).
- **Flow Multiplier**: Teardrop bead extrusion multiplier (default $1.05$).
- **Overhang Fan Speed (%)**: Dedicated part cooling fan speed ($0\text{--}100\%$, default $100\%$).

---

## 5. Retraction & Seams
- **Use Dynamic Fluid Model**: Toggles the 2-point power-law non-Newtonian pressure advance, extrudate swell, and adaptive fluid retraction engine.
  - *Fluid Dynamics Parameters*:
    - **Low-Flow PA & Q**: Calibration point $(C_{\text{PA,low}}, Q_{\text{low}})$.
    - **High-Flow PA & Q**: Calibration point $(C_{\text{PA,high}}, Q_{\text{high}})$.
    - **Static Break Distance (mm)**: Mechanical pull distance ($0.05\text{--}1.0\text{ mm}$) to snap the capillary droplet bridge.
    - **Max Fan Temp Drop (°C)**: Maximum tip cooling drop under 100% fan ($0\text{--}25^\circ\text{C}$).
    - **Ooze Time Constant $\tau$ (s)**: Asymptotic ooze relaxation rate ($0.05\text{--}10.0\text{ s}$).
    - **Max Ooze Prime (mm)**: Maximum extra filament re-primed after long travels ($0.0\text{--}2.0\text{ mm}$).
    - **Swell Ratio @ Low Flow ($B_{\text{low}}$)**: Extrudate die swell expansion ratio under low shear ($1.0\text{--}1.5$).
    - **Swell Ratio @ High Flow ($B_{\text{high}}$)**: Extrudate die swell expansion ratio under high shear ($1.0\text{--}1.5$).
- *Standard Retraction Parameters* (when fluid dynamics is disabled):
  - **Retraction Distance & Speed**: Traditional retraction parameters ($0\text{--}10\text{ mm}$, $10\text{--}150\text{ mm/s}$).
  - **Unretract Extra Length (mm)**: Extra prime distance on unretract.
  - **Pressure Advance (s)**: Static linear pressure advance constant.
  - **Pre-Retract Taper Distance (mm)**: Bleeds melt-zone pressure before travel stops.
  - **Use Firmware Retraction (G10/G11)**: Emits firmware retraction macros.
- **Scarf Joint Seams**: Eliminates vertical seam lines with a ramping overlap lead-in and tail ($1.0\text{--}10.0\text{ mm}$, default $3.0\text{ mm}$).
- **Wipe on Retract**: Wipes nozzle tip inward before travel lifts ($0.1\text{--}10.0\text{ mm}$).

---

## 6. Travel & Simplification
- **Z-hop on Travel**: Lifts nozzle vertically during travel jumps to prevent dragging across printed beads.
- **Z-hop Height (mm)**: Lift clearance ($0.0\text{--}2.0\text{ mm}$).
- **Travel Collision Avoidance**: Detects travel chords crossing solid material against the mesh SDF and computes collision-free 3D detour paths in open air using parallel A* search.
- **Travel Order Optimization**: Reorders paths within each layer using a greedy nearest-neighbor solver to minimize inter-path travel time.
- **Z Travel Penalty**: Cost weighting factor ($1.0\text{--}50.0$) applied to vertical movements during travel path optimization and obstacle routing. Dynamically scales with the machine's Z vs. XY axis speed and acceleration ratio.
- **Simplify Wall Toolpaths**: Ramer-Douglas-Peucker simplification on curved wall loops.
- **Simplification Tolerance (mm)**: Vertex reduction tolerance ($0.0\text{--}0.5\text{ mm}$).

---

## 7. Infill
- **Sparse Pattern**: Pattern for interior sparse cavities (`Cubic`, `Gyroid`, `Schwarz Diamond (D)`, `Schwarz Primitive (P)`, `Monotonic`, `Concentric`, `All Walls`, `None`).
- **Solid Pattern**: Pattern for solid exposure layers (`All Walls`, `Concentric`, `Monotonic`, `Cubic`, `Gyroid`, `Schwarz Diamond (D)`, `Schwarz Primitive (P)`, `None`).
- **Infill Line Width (mm)**: Extrusion width for infill scanlines ($0.05\text{--}1.5\text{ mm}$).
- **Infill Angle (deg)**: Base rotation angle for rectilinear infills ($0\text{--}180^\circ$).
- **Infill Density**: Volume density fraction ($0.0\text{--}1.0$).

---

## 8. Order Field
- **Kind**: Slicing isosurface generator (`Height`, `Conical`, `Eikonal`).
- *Conical Options*:
  - **Apex**: 3D coordinate $(X, Y, Z)$ of the cone apex.
  - **Cone Slope**: Slope angle multiplier ($0.0\text{--}2.0$).
- *Eikonal Options*:
  - **Surface Order Weight**: Multiplier ($0.0\text{--}2.0$, default $1.0$) for the geodesic Surface Eikonal lower bound on the model skin, eliminating surface local minima.
  - **Conform to Top Surfaces**: Blends isosurfaces parallel to upward-facing exterior surfaces.
  - **Top Conform Detach Angle (°)**: Angle threshold ($5.0\text{--}75.0^\circ$) beyond which steep top surfaces revert to bulk slicing.
  - **Conform to Bottom Surfaces**: Warps isosurfaces parallel to downward-facing overhang surfaces.
  - **Bottom Conform Detach Angle (°)**: Angle threshold ($5.0\text{--}75.0^\circ$) for bottom surface tracking.
  - **Conformal Skin Depth (mm)**: Subsurface depth ($0.4\text{--}5.0\text{ mm}$) within which conformal warping applies.
  - **Enforce Vertical Monotonicity**: Enforces strictly increasing layer order along vertical columns ($\partial\Phi/\partial z \ge 0.15$) to prevent downward stalls or mid-air floating loops.
  - **Toolhead Clearance Profile (XZ)**: Interactive table of $(X, Z)$ coordinate points defining the physical gantry clearance cone.

---

## 9. Speeds & Accelerations
- **Speeds (mm/s)**:
  - Print Speed, Outer Wall, Inner Wall, Infill, Solid Infill, Bridge, First Layer, Travel Speed, Max Volumetric Speed ($\text{mm}^3/\text{s}$), and Speed Deadband (%).
- **Accelerations (mm/s²)**:
  - Default, Outer Wall, Inner Wall, Infill, Travel, First Layer, Square Corner Velocity (mm/s), and Acceleration Deadband (%).

---

## 10. Cooling & Fan
- **Fan Speed (%)**: General part cooling fan speed ($0\text{--}100\%$).
- **Overhang Fan Speed (%)**: Dedicated fan speed during overhang moves ($0\text{--}100\%$).
- **Fan Disabled Initial Layers**: Number of initial layers where the fan remains off ($0\text{--}10$).

---

## 11. Machine & Kinematics
- **Bed Dimensions**: Bed X, Bed Y, and Build Height (mm).
- **Tools & Nozzles**: Multi-tool list with nozzle diameter, nozzle flat land diameter (for flat-nozzle slope compensation), extrusion multiplier, and per-tool target temperature.
- **Global Stepper Dynamics**:
  - Zero-speed acceleration ($a_0, \text{mm/s}^2$), max available speed ($v_{\text{max}}, \text{mm/s}$), acceleration limit ($\text{mm/s}^2$), and speed limit ($\text{mm/s}$).
- **Per-Axis Kinematics & Stepper Dynamics**:
  - Independent limit overrides for **X Axis**, **Y Axis**, and **Z Axis** (e.g. capping Z travel speed at $40\text{ mm/s}$ and Z acceleration at $1500\text{ mm/s}^2$ while running XY at $500\text{ mm/s}$ and $10,000\text{ mm/s}^2$).
  - Per-axis stepper dynamics models with independent $a_0$ and $v_{\text{max}}$.
- **Custom Gcode Macros**: Start and End G-code macro editors with placeholder autocompletion.
- **Save / Load Profile**: Save and load complete printer + slicer JSON profile files.

---

## 12. Objects
- **Loaded Objects List**: Triangle counts, tool assignment dropdowns, and per-object delete buttons.
- **Clear All Objects**: Resets the workspace.
- **SDF Debug Panel**: Sign method toggle (Pseudonormal / Winding number), iso-level offset slider, 3D marching cubes extractor, and cross-section plane heatmap viewer (XY, XZ, YZ).
