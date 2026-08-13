# ROADMAP.md

Planning artifact for the next major expansion of Manifold: multi-object,
multi-tool/material slicing; STL + 3MF mesh loading; and a full slicer GUI
(left settings pane, large 3D viewport with an in-panel action toolbar,
in-scene visualization of the coordinate origin and machine components).

This is a plan, not an implementation — phases are dependency-ordered so
work can be picked up incrementally without large refactors later. Anchor
`// TODO(roadmap): Phase N — ...` comments pointing back here are scattered
at the relevant stub sites in source.

## Research notes / crate decisions

- **STL loading**: [`stl_io`](https://crates.io/crates/stl_io) — mature
  (4.4M downloads), reads/writes binary + ASCII STL.
- **3MF loading**: [`lib3mf`](https://crates.io/crates/lib3mf)
  (`telecos/lib3mf_rust`), **recommended over**
  [`threemf`](https://crates.io/crates/threemf) (`hannobraun/3mf-rs`).
  `lib3mf` passes 100% of the official 3MF Consortium's 1,719 positive
  conformance test cases, and explicitly supports multiple objects via
  build items with transforms, the Materials extension (color groups,
  composite materials, multi-properties/layered materials, textures), and
  optional `mesh-ops` (AABB / transformed-AABB via `parry3d`) — directly
  useful for the Tool collision-envelope field below. `threemf`'s own
  README says "functionality is limited." **Caveat**: there is a
  *second*, unrelated crate family with an overlapping name
  (`lib3mf-core`/`lib3mf-cli`/`lib3mf-converters`/`lib3mf-async` from
  `sscargal/lib3mf-rs`) — Phase 1 must explicitly disambiguate these two
  before locking in the dependency, not assume they're the same project.
- **3D transform gizmos**: [`transform-gizmo-egui`](https://crates.io/crates/transform-gizmo-egui)
  — actively maintained egui integration for move/rotate/scale gizmos.
  **Version check required**: this workspace currently pins
  `eframe = "0.29"` (and matching `egui = "0.29"`); confirm
  `transform-gizmo-egui`'s required egui version is compatible, or plan an
  `eframe`/`egui` upgrade, before adding it as a dependency.
- **Custom 3D rendering inside egui**: `egui_wgpu::Callback` /
  `CallbackTrait` (ships with eframe's existing `wgpu` feature — no new
  dependency needed). Standard pattern: paint the 3D scene via a wgpu
  callback, with egui widgets (toolbar, panels, gizmo overlay) layered on
  top in the same frame.
- **File-open dialogs**: [`rfd`](https://crates.io/crates/rfd) — not yet a
  dependency; needed for GUI mesh import.

## Phase 0 — Domain model (prerequisite for everything below)

`manifold-core` is the only crate allowed to hold this domain logic (see
`CODE_STYLE.md`) — none of it belongs in the GUI/CLI layers. Today the
engine models exactly one `Mesh` sliced by one `SlicerConfig` into one
Gcode stream; this phase replaces that with:

- **`Tool`** — id, nozzle diameter, offset. Also add now, even though
  unused until the deferred collision-avoidance work:
  `collision_envelope: BoundingVolume` (a bounding volume around the
  nozzle/toolhead), and represent the tool's mounting **as a full
  transform (position + orientation)**, not a Z-offset scalar — needed so
  the deferred multi-axis (tool-tilting) work doesn't force a schema
  refactor.
- **`Material`** — id, name, and whatever properties later phases need
  (extrusion temp, etc.) — start minimal, extend via `SlicerConfig`-style
  fields, not loose function args (per `CODE_STYLE.md`).
- **`Object`** — mesh + full transform (position/rotation/scale) +
  assigned `Tool`.
- **`Machine`** — bed/build-volume geometry, tool list/offsets. Model the
  print substrate with **a full transform, not a fixed Z=0 plane**, and
  add an explicit `axis_count`/kinematics-style field describing available
  axes — even though only 3-axis (Z-only, non-tilting) kinematics is
  implemented now. This is the up-front constraint that avoids a large
  refactor when multi-axis substrate/tool orientation is tackled later.
- **`Workspace`** — collection of `Object`s + `Machine` + `SlicerConfig`.
  Becomes the new input to `slice_to_gcode`, replacing the single-`Mesh`
  signature. **This is a breaking API change** both front-ends must
  migrate to.

## Phase 1 — Mesh format loading (needs Phase 0 for multi-object targets)

- STL loader (`stl_io`) → `Mesh`.
- 3MF loader (`lib3mf`, pending the disambiguation above) → populates
  `Object`s directly (3MF natively models build items/transforms/
  materials, which is exactly the `Object`/`Tool`/`Material` shape from
  Phase 0).

## Phase 2 — Multi-object / multi-tool slicing pipeline (needs 0 + 1)

- `slicing::slice_mesh` → operate per-`Object` (apply transform, slice in
  world space), tagging output layers by object id.
- Merge per-object layers sharing a Z-height into a combined layer set.
- `toolpath::plan` → tool-change-aware planning: per-tool paths, tool
  switch points.
- `gcode::emit` → tool-change Gcode (tool-select + prime/purge).
- **Explicitly out of scope for this phase**: collision-aware object
  ordering / toolhead-vs-neighbor clearance checking during simultaneous
  multi-object printing — see Deferred Work below. v1 needs an open
  decision (sequential single-object-at-a-time vs. naive simultaneous
  printing) recorded before implementation.

## Phase 3 — Machine/printer definition (needs 0)

- Flesh out `Machine`: bed size/shape, build volume height, toolhead
  count/geometry/offsets — feeds both slicing bounds-checks and GUI scene
  visualization (Phase 6).

## Phase 4 — GUI shell/layout (needs 0)

- `egui::SidePanel::left` — settings (global `SlicerConfig` + per-object
  tool/material assignment).
- `egui::CentralPanel` — 3D viewport, with an in-panel top toolbar (import,
  select/move/rotate/scale modes, slice, export).
- Wire `rfd` for mesh import (STL/3MF via Phase 1 loaders).

## Phase 5 — 3D rendering pipeline (needs Phase 4)

- Minimal wgpu triangle-mesh pipeline (vertex/index buffers from `Mesh`,
  basic normal-lit shader), embedded via `egui_wgpu::Callback`.
- Orbit camera (pan/zoom/rotate) driven by pointer input in the viewport.

## Phase 6 — Scene content: origin, bed/substrate, toolhead (needs 3 + 5)

- Origin gizmo: fixed-size axis triad at world origin.
- Print bed/substrate mesh + grid, sized from `Machine` (Phase 3).
- Toolhead placeholder geometry, positioned per `Tool` offset; multiple
  toolheads for multi-tool machines.

## Phase 7 — Interactive transform gizmos (needs 4 + 5)

- Integrate `transform-gizmo-egui` for per-object move/rotate/scale in the
  3D viewport, wired to selection state and `Object.transform`.

## Phase 8 — End-to-end wiring (needs all of the above)

- GUI: import → arrange/assign tool in 3D view → configure settings →
  Slice action in toolbar → `slice_to_gcode(&Workspace)` → preview/export.
- CLI: extend `manifold-cli` to accept multiple input files (+ per-file
  tool assignment flag) building a `Workspace`, instead of a single
  `Mesh::default()`.

## Deferred / future work (data model must not preclude these, but they are not being built now)

- **Multi-object collision avoidance**: toolhead-vs-already-printed-object
  clearance checking to safely order/interleave simultaneous multi-object
  printing. Depends on `Tool.collision_envelope` (added in Phase 0) and
  affects Phase 2's toolpath ordering once tackled.
- **Multi-axis (>3 axis) kinematics**: reorienting the print substrate
  and/or tilting tools mid-print. Depends on the full-transform modeling
  of `Machine`'s substrate and `Tool`'s mounting (added in Phase 0), plus
  new toolpath planning and Gcode emission logic once tackled.

## Open decisions to resolve during implementation

1. Disambiguate `lib3mf` (telecos) vs. the `lib3mf-core` family
   (sscargal) before adding either as a dependency (Phase 1).
2. Sequential vs. naive-simultaneous multi-object print ordering for v1,
   given collision avoidance is deferred (Phase 2).
3. Stay on `eframe`/`egui` 0.29 or upgrade workspace-wide for
   `transform-gizmo-egui` compatibility (Phase 4/7).
