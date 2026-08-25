# Layering & Extrusion Parameters

This section describes all slicing parameters governing layer heights, perimeter shells, bead cross-section math, and first-layer squish.

---

## Layering Parameters

| Key in JSON | Type | Default | Description |
|---|---|---|---|
| `layer_height` | `f64` | `0.20` | Nominal non-planar layer thickness in mm. |
| `first_layer_height` | `Option<f64>` | `layer_height` | Dedicated thickness for the bed contact layer (e.g. `0.25 mm`). Slicing starts at `order_min + first_layer_height`. |
| `solid_layers_top` | `usize` | `3` | Count of solid shell layers covering upward-exposed regions. |
| `solid_layers_bottom` | `usize` | `3` | Count of solid shell layers covering downward-exposed overhangs. |

---

## Wall & Shell Parameters

| Key in JSON | Type | Default | Description |
|---|---|---|---|
| `nozzle_diameter` | `f64` | `0.40` | Default nozzle orifice diameter in mm. |
| `wall_line_width` | `f64` | `0.40` | Nominal extrusion line width for perimeter walls. |
| `shell_thickness` | `f64` | `0.80` | Total perimeter shell thickness ($N_{\text{walls}} = \text{round}(\text{thickness} / \text{width})$). |
| `wall_offset` | `f64` | `0.20` | Distance from geometric surface to outer perimeter centerline ($0.5 \times d_{\text{nozzle}}$). |
| `nozzle_flat_diameter` | `Option<f64>` | `2.0 * nozzle` | Physical flat diameter of nozzle tip used for tangent flat compensation. |

---

## First Layer Squish & Bed Adhesion

| Key in JSON | Type | Default | Description |
|---|---|---|---|
| `first_layer_line_width` | `Option<f64>` | `1.3 * nozzle` | Wider bead line width ($0.52\text{ mm}$ on $0.4\text{ mm}$ nozzle) to provide lateral squish and maximize bed adhesion contact area. |
| `first_layer_print_speed` | `Option<f64>` | `0.4 * speed` | Maximum print speed ceiling for layer 0 extrusions. |
| `first_layer_extrusion_multiplier` | `Option<f64>` | `1.0` | Volumetric extrusion multiplier applied specifically on layer 0. |

---

## Filament Properties

| Key in JSON | Type | Default | Description |
|---|---|---|---|
| `filament_diameter` | `f64` | `1.75` | Diameter of the raw filament spool (mm). |
| `filament_density_g_cm3` | `Option<f64>` | `1.24` | Material density in $\text{g/cm}^3$ for calculating total extruded mass. |

---

## Bead Cross-Section Math

Manifold computes extrusion lengths using a stadium cross-section model:

$$A_{\text{bead}} = (w - h) \cdot h + \pi \left(\frac{h}{2}\right)^2$$

where $w$ is line width and $h$ is local layer thickness. For a segment of length $L$, the volume is $V = A_{\text{bead}} \cdot L$, requiring raw filament extrusion length:

$$\Delta E = \frac{A_{\text{bead}} \cdot L}{A_{\text{filament}}} = \frac{A_{\text{bead}} \cdot L}{\pi (d_{\text{filament}} / 2)^2}$$
