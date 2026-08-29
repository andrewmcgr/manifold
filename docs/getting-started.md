# Getting Started with Manifold

This guide covers system requirements, building Manifold from source, and running your first slice using both the CLI and GUI.

---

## System Requirements

- **Rust**: Rust toolchain (version 1.80+ recommended, 2021 edition). Install via [rustup](https://rustup.rs/):
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Operating Systems**: macOS (Apple Silicon & Intel), Linux (x86_64 / ARM64 with Vulkan/OpenGL), Windows (x86_64 with Direct3D12/Vulkan).
- **GPU Requirements (GUI only)**: A graphics card supporting WGPU (Metal, Vulkan, DX12) for viewport rendering and GPU-accelerated Eikonal/TPMS compute pipelines.

---

## Building from Source

Clone the workspace repository:
```sh
git clone https://github.com/amcgregor/Manifold.git
cd Manifold
```

### Debug Build (Development)
```sh
cargo build --workspace
```

### Release Build (Recommended for Slicing)
Slicing complex 3D meshes involves CPU-intensive geometric operations (Eikonal Fast Marching, 3D marching cubes, polygon booleans, and A* obstacle routing). Always compile in **release mode** for production slicing performance:
```sh
cargo build --workspace --release
```

Binaries will be placed in `target/release/`:
- `target/release/manifold` (CLI driver)
- `target/release/manifold-gui` (Interactive desktop application)

---

## Quickstart: Command Line

Slice an STL or 3MF model into Klipper-ready G-code using default profile settings:

```sh
cargo run --release -p manifold-cli -- path/to/model.stl -o output.gcode
```

### Common CLI Examples

1. **Conformal Eikonal Slicing with Toolhead Clearance**:
   ```sh
   manifold model.stl --order-field eikonal \
     --eikonal-slope-profile "0:45,5:15,20:5" \
     --eikonal-conform-top-surfaces \
     --eikonal-conform-bottom-surfaces \
     --sparse-infill-pattern gyroid \
     --layer-height 0.20 \
     -o output.gcode
   ```

2. **Multi-Material / Multi-Tool Slicing**:
   ```sh
   manifold body.stl:0 trim.stl:1 \
     --nozzle-temp 240 \
     --bed-temp 105 \
     -o dual_color.gcode
   ```

3. **Enable Thermodynamic Fluid Model & Wave Overhangs**:
   ```sh
   manifold model.stl \
     --fluid-dynamics \
     --wave-overhangs \
     -o optimized.gcode
   ```

For all command line options, see the [CLI Reference](cli.md).

---

## Quickstart: Graphical Interface

Launch the interactive desktop interface:

```sh
cargo run --release -p manifold-gui
```

### Basic Workflow in GUI

1. **Load a Model**: Drag and drop an `.stl` or `.3mf` file into the 3D viewport, or click **Import Objects…** in the left sidebar.
2. **Adjust Settings**: Select layer heights, wall counts, infill patterns (e.g. Gyroid, Schwarz Diamond, Cubic), or machine speeds in the collapsible settings sidebar.
3. **Inspect Overlays**: Use the **Mesh Overlay** dropdown in the top toolbar to preview conformal seed regions or the geodesic surface order gradient.
4. **Slice**: Click the **Slice** button in the top toolbar. Watch real-time progress across order field derivation, outer wall extraction, and layer toolpath generation.
5. **Inspect Toolpaths**: Toggle **Show toolpaths** to view internal loops, infill, bridges, and travels. Switch between `Line Type`, `Speed`, `Actual Speed`, `Flow Rate`, `Acceleration`, `Actual Acceleration`, and `Travel Durations` data views. Hover over any path to view detailed segment metrics.
6. **Export G-code**: Click **Export G-code…** to save the generated program to disk.

For in-depth details on the visual interface, see the [GUI User Guide](gui/index.md).
