# 3D Viewport Data Views & Visualization

Manifold features a hardware-accelerated 3D viewport rendered via `wgpu` with 4x MSAA antialiasing and 32-bit floating point depth testing (`Depth32Float`).

---

## 3D Rendering Capabilities

- **Screen-Space Ribbon Quads**: All bed grid lines and toolpaths are rendered as 2D screen-space instanced ribbon quads ($1.4\times$ pixel line width), eliminating aliasing and sub-pixel dropout.
- **Translucent Mesh X-Ray**: When toolpath visualization is enabled, the 3D model geometry automatically switches to a translucent alpha-blended rendering ($\alpha = 0.25$) without depth writes, allowing interior infill, walls, and bridges to be inspected through the outer hull.
- **Inter-Path Travel Moves**: Travel jumps between paths and across layers are rendered in translucent slate blue with normal-aligned departure and touchdown arcs.

---

## Mesh Overlay Modes

The **Mesh Overlay** dropdown in the top toolbar controls how the 3D model hull is shaded before or during toolpath inspection:

```
[ Mesh Overlay: None (Shaded) | Conformal & Seed Regions | Surface Order Gradient ]
```

### 1. `None (Shaded)`
Standard flat-shaded lighting model with ambient occlusion, switching to translucent X-ray mode when toolpaths are displayed.

### 2. `Conformal & Seed Regions`
Visualizes the Eikonal surface classification and seed boundaries:
- **Green**: Bed contact seed faces identified within contact tolerance ($z \le z_{\text{min}} + \epsilon$).
- **Cyan / Blue**: Top-surface conformal tracking regions whose surface inclination is within the configured detach angle ($\theta \le \theta_{\text{top}}$).
- **Magenta / Red**: Bottom-surface (overhang) conformal tracking regions within the bottom detach angle ($\theta \le \theta_{\text{bottom}}$).
- **Gray**: Unconstrained bulk slicing regions where standard Eikonal geodesic marching applies.
- *Real-time Update*: Re-evaluates and re-shades the mesh dynamically when conformal angle or skin depth sliders are adjusted in the settings panel.

### 3. `Surface Order Gradient`
Renders a continuous multi-stop color ramp over the 3D surface representing the geodesic Surface Eikonal arrival order $\Phi_{\text{surface}}(x, y, z)$:
- Traces how the fast-marching wavefront sweeps across the mesh from bed seeds to peaks.
- Allows visual verification of surface order smoothness and detection of topological saddle points.

---

## Toolpath Data Views

When **Show toolpaths** is enabled, the **Data:** dropdown in the floating top-right viewport legend allows selecting between 7 scalar and categorical visualization modes:

### 1. `Line Type` Mode
Color-codes each extrusion move by its physical feature classification:

| Line Type | Color Badge | Description |
|---|---|---|
| **Outer Wall** | Light Green | Perimeter contour walls forming the visible exterior. |
| **Inner Wall** | Yellow | Structural internal shell perimeter loops. |
| **Infill** | Slate Blue | Interior sparse lattice scanlines (Cubic, Gyroid, TPMS, Concentric, etc.). |
| **Bridge** | Hot Orange / Pink | Overhang bridge chords spanning across solid regions. |
| **Overhang** | Crimson Red | Unsupported overhang contours and wave passes (LaSO). |
| **Top Surface** | Cyan | Solid top skin exposure fills. |
| **Travel** | Translucent Blue | Non-extruding travel and repositioning moves. |

---

### 2. `Speed` Mode (Commanded Speed)
Maps the commanded nominal feedrate (in $\text{mm/s}$) across a 5-stop color gradient:

$$\text{Blue (Slow)} \longrightarrow \text{Cyan} \longrightarrow \text{Green} \longrightarrow \text{Yellow} \longrightarrow \text{Red (Fast)}$$

- The top-right legend card displays the exact minimum and maximum speed bounds across all extruded paths.

---

### 3. `Actual Speed` Mode (Kinematic Resolution)
Maps the kinematically achievable feedrate (in $\text{mm/s}$) after evaluating:
- Per-axis speed ceilings ($\text{AxisLimits}$)
- Stepper motor torque roll-off ODEs ($v_{\text{max}}$ limits)
- Klipper Square Corner Velocity (SCV) cornering limits
- Hotend maximum volumetric flow rate caps

---

### 4. `Flow Rate` Mode
Maps the volumetric polymer extrusion flow rate (in $\text{mm}^3/\text{s}$):

$$Q = A_{\text{bead}} \times v_{\text{toolhead}}$$

- Visually highlights volumetric flow throttling on thick beads, first layer squish, corner decelerations, and slope cosine compensations.

---

### 5. `Acceleration` Mode (Commanded Acceleration)
Maps the configured feature acceleration limits (in $\text{mm/s}^2$) assigned to perimeters, infill, bridges, and travel moves.

---

### 6. `Actual Acceleration` Mode (Motor Torque Limit)
Maps the physically available motor acceleration (in $\text{mm/s}^2$) at current toolhead velocity, integrating the linear torque reduction model:

$$a(v) = a_0 \cdot \max\left(0, 1 - \frac{v}{v_{\text{max}}}\right)$$

---

### 7. `Travel Durations` Mode
Maps the calculated transit duration for each path move (formatted in milliseconds or seconds), allowing immediate identification of long travel jumps or slow non-planar moves.

---

## Hover Inspection Tooltip

Hovering the mouse cursor over any toolpath segment in the 3D viewport displays an interactive HUD card with real-time segment analytics:

```
┌────────────────────────────────────────┐
│ kind: OuterWall                        │
│ speed (cmd): 110.0 mm/s                │
│ speed (actual): 86.4 mm/s              │
│ duration: 18.2 ms                      │
│ flow rate: 8.80 mm³/s                  │
│ accel (cmd): 2500 mm/s²                │
│ accel (actual): 2140 mm/s²             │
│ extrusion_rate: 1.000                  │
│ order: 12.450                          │
└────────────────────────────────────────┘
```

- **kind**: Move feature classification.
- **speed (cmd) / speed (actual)**: Commanded feedrate vs. motor-limited velocity.
- **duration**: Exact move transit duration.
- **flow rate**: Volumetric polymer throughput in $\text{mm}^3/\text{s}$.
- **accel (cmd) / accel (actual)**: Commanded acceleration vs. instantaneous motor torque limit.
- **extrusion_rate**: Volumetric adjustment multiplier.
- **order**: Slicing order field scalar timestamp at this toolpath vertex.

---

## Interactive Order Scrubber

The horizontal slider in the top toolbar allows interactive layer-by-layer scrubbing:
- Dragging **Scrub order** trims the toolpaths to display only moves sliced up to order $T \le \text{cutoff}$.
- Allows inspecting layer progression, first-layer footprints, internal infill connectivity, and non-planar curvature.
