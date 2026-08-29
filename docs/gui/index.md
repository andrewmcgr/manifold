# Graphical User Interface (GUI) Guide

`manifold-gui` provides an interactive 3D desktop interface powered by `egui` and `wgpu`.

---

## Interface Overview

The GUI is arranged into three primary visual zones:

```
┌───────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Top Toolbar: [Import] [Slice] [Show toolpaths]    │ [Overlay: Surface Order] [Data: Speed] [Export] │
├───────────────────────────────────────────────────┴─────────────────────────────────────────────────┤
│                                                   │                                                 │
│  Settings Sidebar (Collapsible)                   │  3D Hardware Accelerated Viewport               │
│  • Objects & Workspace                            │  • Orbit / Pan / Zoom 3D Camera                 │
│  • Layering & Extrusions                          │  • 32-bit Depth Buffer + 4x MSAA                │
│  • Infill (TPMS Gyroid / Schwarz / Cubic)         │  • Screen-Space Ribbon Quads                    │
│  • Order Field (Eikonal / Conformal / Clearances) │  • Translucent Mesh X-Ray                       │
│  • Wave Overhangs (LaSO)                          │  • Top-Right Legend & Gradient Bar              │
│  • Retraction & Fluid Dynamics                    │  • Interactive Hover Toolpath Inspector         │
│  • Speeds, Accelerations & Per-Axis Kinematics    │  • Order Scrubber Slider                        │
│  • Machine, Tools & Profiles                      │                                                 │
│                                                   │                                                 │
└───────────────────────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

## Viewport Controls

| Action | Mouse / Keyboard Gesture | Description |
|---|---|---|
| **Orbit Camera** | Left-click + Drag on 3D canvas | Rotates view around the target center. |
| **Pan Camera** | Right-click + Drag (or Middle-click + Drag) | Translates camera across the build plane. |
| **Zoom** | Mouse Scroll Wheel (or Pinch gesture) | Zooms smoothly in/out toward the cursor. |
| **Frame All Objects** | `F` key (or click "Frame All") | Resets camera to encapsulate all loaded geometry. |
| **Delete Selected Object** | `Delete` or `Backspace` key | Removes the currently selected part from the workspace. |
| **Inspect Segment** | Hover cursor over any toolpath | Opens the HUD card with instantaneous velocity, flow, duration, and order. |

---

## Object Management & Scene Setup

- **Import Models**: Click **Import Objects…** or drag `.stl`/`.3mf` files directly into the window.
- **Select Objects**: Click an object row in the sidebar or click directly on the mesh in the viewport to activate the 3D translation/rotation transform gizmo.
- **Per-Object Tool Assignment**: Change the assigned tool ID (`0`, `1`, `2`...) per part.
- **Auto-center on bed**: Positions all loaded parts centered on the build plate.
- **Mesh Overlay Visualizers**: Switch the top bar **Mesh Overlay** to preview conformal seed boundaries or geodesic surface arrival order gradients before slicing.
- **Remove Objects**: Click the inline **Remove** button on an object row or use the **Clear all objects** button.

---

## Detailed GUI Guides

- [Settings Panel Reference](settings-panel.md) — Comprehensive guide to every collapsible settings group.
- [3D Viewport Data Views & Overlays](data-views.md) — Line type badges, 7 continuous data views, hover inspection card, and mesh overlays.
- [Custom G-code Macros](custom-gcode.md) — Template syntax and variable substitutions for Klipper start and end macros.
