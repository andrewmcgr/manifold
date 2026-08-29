# Wave Overhangs (Huygens LaSO Engine)

Manifold implements the **Wave Overhang** path planning strategy for support-free horizontal and non-planar overhangs (Andersons, Sanchez, Vaneker 2024 / SSRN-6640458).

---

## The Physical Mechanics of Lateral Support (LaSO)

When depositing molten plastic over empty air, an overhanging track can remain mechanically stable without support scaffolding if it is deposited laterally in contact with, and slightly overlapping, an already-solidified adjacent track.

```
       Prior Solid Track                New Overhang Track
     ┌───────────────────┐             ┌───────────────────┐
     │                   │  Overlap    │                   │
     │                   ├───┐     ┌───┤     (Teardrop)    │
     │                   │   │◄───►│   │                   │
     └───────────────────┴───┴─────┴───┴───────────────────┘
               ▲                           ▲
               │                           │
         [Solid Ground]               [Empty Air]
```

- The lateral overlap forms a **teardrop bead cross-section** with a large lateral contact bond area.
- Cohesive and surface tension forces support the track until forced cooling solidifies it below its glass transition temperature ($T_g$).

---

## 3D Shell-Conformal Wave Planning & Lateral Seed Anchoring

- **3D Shell-Conformal Detection**: Unsupported overhang regions are detected across 3D non-planar order field isosurfaces by probing the geometric intersection with the preceding layer's solid shell.
- **Lateral Seed Anchoring**: When an overhang region emerges over an internal void without direct layer-below overlap, contact seeds automatically anchor laterally into adjacent supporting wall perimeter loops, preventing center-outward expansion in empty air.
- **Footprint Masking**: Wall and infill paths are dynamically masked against the wave overhang footprint to eliminate redundant double-extrusion over bridged non-planar spans.
- **Support-Aware Path Emission Deferral**: Within each layer, extrusions are topologically sorted so that supporting wall loops and anchored bridges are printed first before dependent overhanging waves are emitted.

---

## Huygens Wavefront Propagation

Wavefront distances propagate outward from solid contact seeds according to **Huygens principle**:

$$\text{Wavefront}_{i+1} = \left( \bigcup_{p \in W_i} \text{Disk}(p, \lambda) \cap \text{Boundary} \right) \setminus \text{Visited}$$

- Continuous wave diffraction bends smoothly around corners in long, connected passes, giving each track time to cool before the adjacent pass arrives.

---

## Configuration Keys

```json
{
  "wave_overhangs_enabled": true,
  "wave_overhang_overlap": 0.05,
  "wave_overhang_speed": 1500.0,
  "wave_overhang_flow": 1.05,
  "overhang_fan_speed_percent": 100.0
}
```

| Key in JSON | Default | Description |
|---|---|---|
| `wave_overhangs_enabled` | `true` | Enables 2D/3D Huygens wave overhang generation and tags unsupported outer perimeter segments as `MoveKind::Overhang`. |
| `wave_overhang_overlap` | `0.05 mm` | Lateral overlap distance $\delta$ between adjacent wave tracks. Wavelength spacing is $\lambda = d_{\text{nozzle}} - \delta$. |
| `wave_overhang_speed` | `1500.0 mm/min` | Printing speed for overhang moves ($25.0\text{ mm/s}$). |
| `wave_overhang_flow` | `1.05` | Teardrop bead extrusion flow multiplier ($A_{\text{bead}} = \lambda \cdot h \cdot k_{\text{flow}}$). |
| `overhang_fan_speed_percent` | `100.0%` | Dedicated part cooling fan speed percentage during overhang moves. |
