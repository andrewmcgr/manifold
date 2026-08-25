# Machine Envelopes, Tools & Temperatures

This section documents printer hardware definitions, multi-tool setups, and thermal targets.

---

## Machine Build Volume (`machine.build_volume`)

```json
{
  "machine": {
    "build_volume": {
      "Aabb": {
        "min": [0.0, 0.0, 0.0],
        "max": [300.0, 300.0, 300.0]
      }
    },
    "axis_count": 3
  }
}
```

- **AABB Envelope**: Bounding box in physical millimeters `[Bed X, Bed Y, Build Height]`.
- **Substrate Transform**: Full 3D transformation matrix of the print bed ($4 \times 4$ column-major float array).

---

## Multi-Tool Definitions (`machine.tools`)

```json
{
  "machine": {
    "tools": [
      {
        "id": 0,
        "nozzle_diameter": 0.4,
        "mount": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "collision_envelope": {
          "Sphere": { "center": [0.0, 0.0, 0.0], "radius": 0.0 }
        },
        "extrusion_multiplier": 1.0,
        "nozzle_temperature": 240.0
      },
      {
        "id": 1,
        "nozzle_diameter": 0.6,
        "mount": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "collision_envelope": {
          "Sphere": { "center": [0.0, 0.0, 0.0], "radius": 0.0 }
        },
        "extrusion_multiplier": 1.02,
        "nozzle_temperature": 255.0
      }
    ]
  }
}
```

| Key in Tool | Type | Default | Description |
|---|---|---|---|
| `id` | Integer | `0` | Sequential Tool ID (`0`, `1`, `2`...). |
| `nozzle_diameter` | `f64` | `0.40` | Orifice diameter in millimeters. |
| `extrusion_multiplier` | `f64` | `1.0` | Material-specific flow scaling factor. |
| `nozzle_temperature` | `Option<f64>` | `240.0` | Target hotend temperature (°C) for this tool. |

---

## Temperatures in `SlicerConfig`

```json
{
  "config": {
    "default_nozzle_temperature": 240.0,
    "bed_temperature": 105.0,
    "chamber_temperature": 50.0
  }
}
```

- **`default_nozzle_temperature`**: Fallback hotend temperature (°C) when per-tool temperature is unset.
- **`bed_temperature`**: Target heated build plate temperature (°C).
- **`chamber_temperature`**: Target heated chamber temperature (°C, 0 = unheated).
