# 3D Viewport Data Views & Visualization

Manifold features a hardware-accelerated 3D viewport rendered via `wgpu` with 4x MSAA antialiasing and 32-bit floating point depth testing (`Depth32Float`).

---

## 3D Rendering Capabilities

- **Screen-Space Ribbon Quads**: All bed grid lines and toolpaths are rendered as 2D screen-space instanced ribbon quads ($1.4\times$ pixel line width), eliminating aliasing and sub-pixel dropout.
- **Translucent Mesh X-Ray**: When toolpath visualization is enabled, the 3D model geometry automatically switches to a translucent alpha-blended rendering ($\alpha = 0.25$) without depth writes, allowing interior infill, walls, and bridges to be inspected through the outer hull.
- **Inter-Path Travel Moves**: Travel jumps between paths and across layers are rendered in translucent slate blue with normal-aligned departure and touchdown arcs.

---

## Toolpath Data Views

The **Data:** dropdown in the top-right overlay allows selecting between 4 visualization modes:

### 1. `Line Type` Mode
Color-codes each extrusion move by its physical feature classification:

| Line Type | Color Badge | Description |
|---|---|---|
| **Outer Wall** | Light Green | Perimeter contour walls forming the visible exterior. |
| **Inner Wall** | Yellow | Structural internal shell perimeter loops. |
| **Infill** | Slate Blue | Interior sparse lattice scanlines. |
| **Bridge** | Hot Orange / Pink | Overhang bridge chords spanning across solid regions. |
| **Overhang** | Crimson Red | Unsupported overhang contours and wave passes (LaSO). |
| **Top Surface** | Cyan | Solid top skin exposure fills. |
| **Travel** | Translucent Blue | Non-extruding travel and repositioning moves. |

---

### 2. `Speed` Mode
Maps the planned physical nozzle feedrate (in $\text{mm/s}$) across a continuous 5-stop color gradient:

$$\text{Blue (Slow)} \longrightarrow \text{Cyan} \longrightarrow \text{Green} \longrightarrow \text{Yellow} \longrightarrow \text{Red (Fast)}$$

- The top-right legend card displays an interactive horizontal gradient bar with exact minimum and maximum speed bounds (e.g. `20.0 mm/s` to `200.0 mm/s`).

---

### 3. `Flow Rate` Mode
Maps the volumetric polymer extrusion flow rate (in $\text{mm}^3/\text{s}$):

$$Q = A_{\text{bead}} \times v_{\text{toolhead}}$$

- Visually highlights volumetric flow throttling on thick beads, first layer squish, and corner decelerations.

---

### 4. `Acceleration` Mode
Maps the kinematic acceleration limit (in $\text{mm/s}^2$) evaluating stepper motor available torque and per-feature bounds:

- Shows high torque acceleration capability ($15,000\text{--}20,000\text{ mm/s}^2$) at low cornering speeds transitioning to cruise acceleration on long travel straights.

---

## Interactive Order Scrubber

The horizontal slider at the bottom of the viewport allows interactive layer-by-layer scrubbing:
- Dragging the slider trims the toolpaths to display only moves sliced up to order $T \le \text{cutoff}$.
- Allows inspecting layer progression, first-layer footprints, internal infill connectivity, and non-planar curvature.
