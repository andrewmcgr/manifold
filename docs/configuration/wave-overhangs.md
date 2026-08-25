# Wave Overhangs (Huygens LaSO Engine)

Manifold implements the **Wave Overhang** path planning strategy for support-free horizontal overhangs (Andersons, Sanchez, Vaneker 2024 / SSRN-6640458).

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

## Wavefront Diffraction vs. Arc Overhangs

- **The Problem with Arc Overhangs**: Turning concave corners forces arc centers into sharp singularities ($< 2\text{ mm}$ radius), causing intense local heat accumulation where molten plastic sags before cooling.
- **The Wave Solution**: Propagates distance wavefronts using **Huygens principle**:
  $$\text{Wavefront}_{i+1} = \left( \bigcup_{p \in W_i} \text{Disk}(p, \lambda) \cap \text{Boundary} \right) \setminus \text{Visited}$$
- Wavefronts naturally exhibit **wave diffraction**, bending smoothly around corners in long, continuous paths that give each track ample time to cool before the next adjacent pass arrives.

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
| `wave_overhangs_enabled` | `true` | Enables 2D Huygens wave overhang generation and tags unsupported outer perimeter segments as `MoveKind::Overhang`. |
| `wave_overhang_overlap` | `0.05 mm` | Lateral overlap distance $\delta$ between adjacent wave tracks. Wavelength spacing is $\lambda = d_{\text{nozzle}} - \delta$. |
| `wave_overhang_speed` | `1500.0 mm/min` | Printing speed for overhang moves ($25.0\text{ mm/s}$). |
| `wave_overhang_flow` | `1.05` | Teardrop bead extrusion flow multiplier ($A_{\text{bead}} = \lambda \cdot h \cdot k_{\text{flow}}$). |
| `overhang_fan_speed_percent` | `100.0%` | Dedicated part cooling fan speed percentage during overhang moves. |
