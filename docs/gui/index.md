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
| **Drag View / Pan** | Left-click + Drag (on bed/skybox with no object selected), or Right-click + Drag | Translates camera across the build plane (1:1 motion tracking). |
| **Select & Drag Object (XY)** | Left-click on object + Drag, or Drag with object selected | Translates the object across the horizontal build plane ($Z$ locked). |
| **Deselect Object** | Left-click (single click) away from gizmo on bed/skybox | Deselects the current object. |
| **Transform Gizmo** | Left-click + Drag on gizmo handles | Moves along axes, rotates, or scales using the 3D gizmo. |
| **Orbit Camera** | Middle-click + Drag (or Shift + Left-click + Drag) | Rotates view around the hovered pivot or in place. |
| **Zoom** | Mouse Scroll Wheel (or Pinch gesture) | Zooms smoothly in/out toward the cursor. |
| **Delete Selected Object** | `Delete` or `Backspace` key | Removes the currently selected part from the workspace. |
| **Inspect Segment** | Hover cursor over any toolpath | Opens the HUD card with instantaneous velocity, flow, duration, and order. |

---

## Object Management & Scene Setup

- **Import Models**: Click **Import Objects…** or drag `.stl`/`.3mf` files directly into the window.
- **Select Objects**: Click an object row in the sidebar or click directly on the mesh in the viewport to activate the 3D translation/rotation transform gizmo.
- **Per-Object Tool Assignment**: Change the assigned tool ID (`0`, `1`, `2`...) per part.
- **Auto-center on bed**: Positions all loaded parts centered on the build plate.
- **Drop to Bed**: Drops the selected object along $Z$ so its lowest point rests flush on the build plate at $Z = \text{bed\_min.z}$, preserving current orientation.
- **Lay on Face**: Visualizes a simplified convex hull ($\le 26$ facets, each with $\ge 3$ contact points for tripod stability). Hovering highlights a facet; clicking rotates the object so that facet faces down and drops it flush to the bed. Press `Escape` or click "Done Lay on Face" to exit.
- **Mesh Overlay Visualizers**: Switch the top bar **Mesh Overlay** to preview conformal seed boundaries or geodesic surface arrival order gradients before slicing.
- **Remove Objects**: Click the inline **Remove** button on an object row or use the **Clear all objects** button.

---

## Detailed GUI Guides

- [Settings Panel Reference](settings-panel.md) — Comprehensive guide to every collapsible settings group.
- [3D Viewport Data Views & Overlays](data-views.md) — Line type badges, 7 continuous data views, hover inspection card, and mesh overlays.
- [Custom G-code Macros](custom-gcode.md) — Template syntax and variable substitutions for Klipper start and end macros.
