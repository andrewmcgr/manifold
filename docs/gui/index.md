# Graphical User Interface (GUI) Guide

`manifold-gui` provides an interactive 3D desktop interface powered by `egui` and `wgpu`.

---

## Interface Overview

The GUI is arranged into three primary visual zones:

```
┌────────────────────────────────────────┬──────────────────────────────────────────┐
│ Top Toolbar: [Import] [Remove] [Slice] │ [Toolpath Preview] [Data: Speed] [Export]│
├────────────────────────────────────────┴──────────────────────────────────────────┤
│                                        │                                          │
│  Settings Sidebar (Collapsible)        │  3D Hardware Accelerated Viewport        │
│  • Objects & Selection                 │  • Orbit / Pan / Zoom 3D Camera          │
│  • Layering                            │  • 32-bit Depth Buffer + 4x MSAA         │
│  • Extrusion & Walls                   │  • Screen-Space Ribbon Lines             │
│  • Temperatures                        │  • Semi-Transparent Model X-Ray          │
│  • Wave Overhangs                      │  • Top-Right Legend & Gradient Bar       │
│  • Retraction & Seams                  │  • Order Scrubber Slider                 │
│  • Speeds & Accelerations              │                                          │
│  • Machine & Profiles                  │                                          │
│                                        │                                          │
└────────────────────────────────────────┴──────────────────────────────────────────┘
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

---

## Object Management

- **Import Models**: Click **Import Objects…** or drag `.stl`/`.3mf` files directly into the window.
- **Select Objects**: Click an object row in the sidebar to highlight it.
- **Per-Object Tool Assignment**: Change the assigned tool ID (`0`, `1`, `2`...) per part.
- **Positioning & Bed Alignment**: Click **Auto-center on bed** to center all objects on the configured build plate.
- **Remove Objects**: Click the inline **Remove** button on an object row or use the **Clear all objects** button.

---

## Detailed GUI Guides

- [Settings Panel Breakdown](settings-panel.md) — Comprehensive guide to every collapsible settings group.
- [3D Viewport Data Views](data-views.md) — Line type badges, continuous speed/flow/acceleration gradients, and the layer scrubber.
- [Custom G-code Macros](custom-gcode.md) — Template syntax and variable substitutions for Klipper start and end macros.
