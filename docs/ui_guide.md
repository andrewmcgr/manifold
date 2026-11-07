# Manifold Slicer — User Interface Guide

This guide walks through the interactive 3D graphical user interface of **Manifold Slicer**, covering workspace layout, multi-plate project management, interactive 3D viewport tools, orientation hint authoring, volumetric field inspection, and non-planar toolpath previewing.

---

## 1. Main Window Overview

The **Manifold Slicer** window is divided into five coordinated regions around a real-time 3D viewport:

```
+-----------------------------------------------------------------------------------+
| File  Import (.step/.stl/.3mf)  Open/Save Project (.3mf)  Auto-Arrange  Export... |
| Presets: [Printer v] [Save]  [Tool v] [Save]  [Filament v] [Save]  [Process v]    |
+--------------------+-----------------------------------------+--------------------+
| Left Panel         |                                         | Right Panel        |
|                    |                                         |                    |
| • Build Plates     |                                         | • Slicing Action   |
|   - Plate 1 (2)    |                                         |   [Generate ...]   |
|   - Plate 2 (1)    |           3D Interactive                |                    |
|   [+ New] [Dup]    |              Viewport                   | • View &           |
|                    |                                         |   Inspection       |
| • Viewport Tool    |     (Build Plate, G-code Origin,        |   - Layer Sliders  |
|   [Select & Move]  |      Objects, Exclude-Object Outlines,  |   - Feature Toggles|
|   [Place on Face]  |      Surface Patches, VDB Slices,       |   - VDB Fields     |
|   [Normal ⊥ Surf]  |      Conformal Isosurfaces,             |                    |
|   [Normal ∥ Surf]  |      & 3D Color-Coded Toolpaths)        | • Configuration    |
|   [3D Point Hint]  |                                         |   - Printer & Bed  |
|                    |                                         |   - Tools & 5/6Ax  |
| • Objects on Plate |                                         |   - Filament & PA  |
|   - Object List    |                                         |   - Process/Slicing|
|   - Clone / Move   |                                         |   - Eikonal & Hints|
|   - Transform XYZ  |                                         |                    |
+--------------------+-----------------------------------------+--------------------+
| Status Bar: Active Tool Mode | Object Count | Slicing / Export Status Messages    |
+-----------------------------------------------------------------------------------+
```

### 1.1 Top Menu & Action Bar
Located at the very top of the window, the menu bar provides one-click access to core file and plate operations:
- **Import Geometry (`STEP / STL / 3MF`)**: Loads `.step`/`.stp` (via OpenCASCADE exact B-Rep triangulation and analytical face patch extraction), `.stl` (ASCII or binary with automatic coplanar patch clustering), or `.3mf` meshes onto the active build plate.
- **Open / Save Project (`.3mf`)**: Loads or saves a complete Manifold project archive—including all build plates, object transforms, surface/point orientation hints, and the full `SlicerConfig`—in a single standard `.3mf` file. When you open an existing Manifold `.3mf` project, Manifold Slicer remembers its filename and uses it as the default when saving.
- **Auto-Arrange Plate**: Runs the Archimedean spiral collision-free packer on all objects on the active build plate around the bed center `(0, 0)`.
- **Generate Fields & Slice**: Triggers the C++ OpenVDB volumetric solver (`sdf`, `overhang`, `eikonal`), extracts non-planar conformal layers, generates 3D shell/TPMS/support toolpaths, and compiles Klipper G-code.
- **Export G-code (`.gcode`)**: Writes the generated G-code (including Moonraker thumbnails, `[exclude_object]` polygons, and dynamic PA commands) to disk.
- **`Calibration ▾`**: Opens the floating **Polymer Rheology & Flow Calibration Utilities** window for generating and previewing Extrusion Multiplier (Pass 1/2/3), Dual-Band Ellis $90^\circ$ Chevron Pressure Advance, and Transient Melt-Zone Pressure ($\gamma$ & $\tau_{\text{melt}}$) calibration prints for any configured toolhead (`T0`, `T1`, …). See the **[Polymer Rheology & Flow Calibration Guide (`docs/calibration_guide.md`)](calibration_guide.md)** for full instructions and visual reading examples.

### 1.2 Top Presets Bar
Directly beneath the main menu bar sits the **Four-Category Presets Bar**:
- **Printer Preset**: Build volume dimensions, bed shape, 5-axis bed tilt, acceleration limits, Klipper SCV/cruise ratio, and start/end G-code templates.
- **Tool Preset**: Active nozzle diameter, 5/6-axis rotational tilt limits, XZ nozzle clearance profile, and flat-nozzle slope compensation.
- **Filament Preset**: Hotend and bed temperatures, extrusion multipliers, cooling fan speeds, retractions, shear-thinning Pressure Advance ($k_{\text{low}}, k_{\text{high}}$), and viscoelastic die-swell multipliers.
- **Process Preset**: Non-planar layer heights, 3D shell thickness, Arachne variable line widths, TPMS sparse infill, overhang/bridging/support thresholds, and feature print speeds.

Each category includes a dropdown selector (listing built-in factory presets and your user-saved presets), a **Save…** button to store your current tuning as a reusable preset in `~/.config/manifold2/presets/`, and a **Delete** button for custom presets.

---

## 2. Left Panel: Multi-Plate Projects, Viewport Tools & Objects

### 2.1 Build Plates Manager
Manifold Slicer supports multi-plate workflows natively and imports multi-plate `.3mf` projects from **Manifold Slicer**, **OrcaSlicer**, **Bambu Studio**, and **PrusaSlicer**:
- **Plate List**: Click any plate (e.g., `Plate 1 (2 objs)`) to switch the active build plate in the 3D viewport. Switching plates updates the visible objects and clears or refreshes the active plate's preview.
- **`+ New Plate`**: Appends a fresh empty build plate to the project.
- **`Duplicate Plate`**: Clones the active build plate along with all of its objects, transforms, and orientation hints.
- **`Remove Plate`**: Deletes the active plate (when more than one plate exists).
- **Plate Name Editor**: Rename the active build plate; its name is preserved inside `.3mf` project files and used when exporting multi-plate G-code.

### 2.2 Interactive Viewport Tools
Six modal tools control how mouse clicks interact with geometry in the 3D viewport:

| Tool Button | Purpose & Interaction |
| :--- | :--- |
| **Select & Move** | Default manipulation mode. Left-click an object to select it; drag across the build plate (`Z = 0`) to reposition it in $XY$, or use the transform sliders in the Left Panel. |
| **Place on Face** | Displays all candidate planar support facets of the selected object's 3D convex hull. Hover over any hull facet to preview it in bright highlight, and **left-click** to rotate the object so that facet lies flush on the build plate (`z = 0`). |
| **Surface Normal ⊥ (Conformal)** | Highlights selectable CAD B-Rep faces (or contiguous planar mesh patches) under the cursor. **Left-click** a face to attach a soft orientation hint requesting that slicing layers run **conformal/tangent** to that face ($\nabla\tau \parallel \pm\hat{\mathbf{n}}_{\text{face}}$). **Right-click** an existing hint face to remove it. |
| **Surface Normal ∥ (⊥ Layers)** | Highlights selectable surface patches under the cursor. **Left-click** a face to attach a soft orientation hint requesting that slicing layers meet that face **orthogonally at $90^\circ$** ($\nabla\tau \perp \hat{\mathbf{n}}_{\text{face}}$)—ideal for vertical mating walls or overhang faces. |
| **Conformal Skin (`Skin`)** | Highlights selectable surface patches under the cursor. **Left-click** a sloped roof, deck, or dome patch to attach a **`ConformalSurfaceSkin`** hint that erodes the underlying core SDF by $N_p \cdot h_{\text{top}}$ and slices $N_p$ constant-thickness conformal SDF skin layers parallel to $\phi_{\text{sdf}} = 0$ (with automatic 3D tool-clearance stand-off around rising protrusions and printed walls). Setting `Layers = 0` on a patch explicitly excludes it from `Auto Conformal SDF Top Skin`. |
| **3D Point Hint** | Places a localized 3D vector orientation hint in space with a spherical Wendland $C^2$ radial influence zone, allowing fine steering of the Eikonal layer normal inside bulky interiors. |

### 2.3 Objects on Active Plate & Transform Controls
For the currently selected plate, the **Objects** list shows each part along with quick actions:
- **Clone (`+ Clone`)**: Spawns an additional independent copy of the selected object (sharing the underlying CAD/mesh topology while maintaining its own transform and hints) and automatically spiral-arranges the plate so parts do not overlap.
- **Move to Plate**: Reassigns the selected object to any other build plate in the project.
- **Delete (`Remove`)**: Removes the selected object and cleans up its associated orientation hints.
- **Transform Inspector**:
  - **Position (`X`, `Y`, `Z` mm)**: Fine-tune bed coordinates, plus a **Center on Bed** (`X = 0, Y = 0`) button and a **Drop to Bed** (`min Z = 0`) button.
  - **Rotation (`Rx`, `Ry`, `Rz` deg)**: Adjust Euler rotation angles or click quick $\pm 90^\circ$ axis rotation buttons.
  - **Scale (`Uniform` or `Sx, Sy, Sz`)**: Scale part geometry prior to voxelization.

---

## 3. Central 3D Viewport & Build-Plate Visualizations

### 3.1 Camera Navigation
- **Orbit**: Hold **Right Mouse Button** (or **Middle Mouse Button**) and drag to orbit around the target point.
- **Pan**: Hold **Shift + Right/Middle Mouse Button** and drag to pan across the build plate.
- **Zoom**: Scroll the **Mouse Wheel** (or two-finger trackpad scroll) to zoom in and out.

### 3.2 Coordinate Origin vs. Bed Center
- **Bed Center `(0, 0, 0)`**: The center of the build plate is the default spawn point for newly imported parts and the origin of the Archimedean spiral auto-arranger.
- **Printer G-code Origin Triad (`X` Red, `Y` Green, `Z` Blue)**: Drawn at the printer's physical G-code `(0, 0, 0)` home point:
  - At the **front-left corner** of the bed for rectangular printers when `G-code Origin at Bed Center` is unchecked.
  - At the **bed center** for circular (Delta) beds or when `G-code Origin at Bed Center` is enabled.

### 3.3 Exclude-Object Footprint Outlines & Labels
- Every object on the active build plate displays a **cyan 2D convex-hull footprint polygon** directly on the build plate (`z = 0`). This is the exact polygon emitted in Klipper's `EXCLUDE_OBJECT_DEFINE NAME=... POLYGON=[...]` macro.
- Floating above each footprint is a **3D billboard label** displaying the sanitized Klipper object identifier (e.g., `part_1`) and its $(X, Y)$ center in printer G-code coordinates.
- Both the footprint outlines and the floating name labels can be toggled on or off independently in the **View & Inspection** panel.

### 3.4 Multi-Axis Nozzle Clearance & Gravity Gizmo
Enabling **Show Nozzle Clearance & Gravity Gizmo** renders:
- The active tool's local orthonormal `NozzleFrame` basis $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$.
- The revolved 3D wireframe clearance cone defined by `clearance_profile_xz` along $\hat{\mathbf{z}}_{\text{nozzle}}$.
- The true bed-coordinate gravity direction vector $\hat{\mathbf{g}}_{\text{bed}}$ (updated live as you tilt the build plate via `Rx` / `Ry`).

---

## 4. Right Panel: Slicing, Inspection & Configuration

Every parameter row in the Right Panel includes a compact **`?` help button**. Clicking `?` opens a rich popover displaying the parameter's canonical dotted key (for CLI/JSON use), physical units, default value, summary, and detailed tuning guidance.

### 4.1 View & Inspection Panel
Once fields or toolpaths have been generated, the **View & Inspection** section provides three inspection modes:

1. **Toolpath Preview & Combined Legend / Toggles**:
   - **Interactive Layer Range Sliders (`Min Layer .. Max Layer`)**: Scrub through individual non-planar layers or inspect a range of layers from build plate to top skin.
   - **Wrapping Color-Swatch Feature Toggles**: A responsive wrapping flexbox of pill buttons that combines the **color legend** and **visibility toggles** in a single control so the right panel stays compact:
     - **Outer Wall** (Orange-Red)
     - **Inner Wall** (Gold/Amber)
     - **Solid Infill** (Purple)
     - **Sparse Infill** (Red/Crimson)
     - **Gap Fill** (White/Silver)
     - **Bridge** (Sky Blue)
     - **Wave Overhang** (Deep Royal Blue)
     - **Support Base** (Emerald Green)
     - **Support Interface** (Mint/Cyan-Green)
     - **Travel Moves** (Teal, off by default)
     - **3D Scarf Seams** (Magenta/Cyan ramp markers)
   - Quick **`All On`** and **`Walls Only`** filter buttons let you isolate perimeter quality or inspect internal TPMS lattices in one click.

2. **Conformal Isosurface Mesh Preview**:
   - Toggle visibility of the extracted non-planar triangle meshes ($\tau(\mathbf{x}) = \tau_k$) before or alongside the 3D toolpaths to inspect how layers curve over domes, cantilevers, and multi-branch saddles.

3. **OpenVDB Volumetric Field Cross-Sections & 3D Vector Arrows**:
   - Select any OpenVDB grid (`sdf`, `overhang`, `eikonal`) to view a color-mapped cross-section slice plane (`X`, `Y`, or `Z` axis with interactive position slider).
   - Enable **3D Field Vector Arrows** to inspect the SDF surface normal $\nabla\phi_{\text{sdf}}$, the constrained Eikonal layer normal $\nabla\tau / \|\nabla\tau\|$, or the 5/6-axis tool orientation vector $\hat{\mathbf{z}}_{\text{nozzle}}(\mathbf{x})$ across space.

### 4.2 Collapsible Configuration Sections
Beneath **View & Inspection**, all profile variables are organized into collapsible sections matching the preset hierarchy:
- **Print Bed & Auto-Arrangement**: Bed shape, $X/Y/Z$ dimensions, grid spacing, G-code origin placement, part clearance, and bed edge margin.
- **Machine Tools & 5/6-Axis Kinematics**: Multi-tool selector (`Add Tool`, `Remove Tool`), nozzle diameter, hotend temperature, rotational axis toggle, maximum tilt angle, 3D tool clearance envelope mode (`Piecewise XZ`, `Circular Cone`, or `Rectilinear Cone (XY)`), and 5-axis bed tilt (`Rx`, `Ry`).
- **Volumetric Fields & Eikonal Solver**: Order field mode (`Constrained Non-Planar` vs. `Height-Only Planar`), auto fine voxel size toggle, Dual-Grid multi-resolution settings, solver iterations, curvature regularization, automatic overhang conformal/orthogonal surface bias, and a live list of all placed surface/point orientation hints (`⊥`, `∥`, and `Skin`).
- **Non-Planar Layer Slicing & Shells**: Variable layer height (`Target`, `Min`, `Max`, `First Layer`, `Top Layer`), max non-planar slope, **Auto Conformal SDF Top Skin** (`Skin Layers`, `Min Oblique`, `Max Slope`), 3D shell thickness, inner wall count, Arachne variable line widths (`Target`, `Min`, `Max`), solid/sparse infill patterns (`Gyroid`, `Schwarz-D`, `Schwarz-P`, `Cubic`, `Monotonic`, `Concentric`, `All Walls`) with automatic rib-anchored solid-over-sparse infill, circular bridging, wave overhangs, and causal supports (`Columnar Grid` vs. obstacle-avoiding `Non-Planar Tree` supports and dedicated `Support Tool` selection).
- **G-code, Speeds, Seams & Rheology**: Wall print order (`In-Out-In` vs. `Outside-In`), Klipper `EXCLUDE_OBJECT` and Moonraker thumbnail toggles, feature speeds and accelerations, Klipper SCV/cruise ratio, stepper back-EMF dynamics, 3D scarf joints, retractions/wipes, thermal ooze relaxation, flat-nozzle slope clearance, shear-thinning Pressure Advance ($k_{\text{low}}, k_{\text{high}}$), viscoelastic die-swell multipliers, corner flow compensation, transient melt-chamber pressure lag, cooling fan speeds, and Start/End G-code templates.
