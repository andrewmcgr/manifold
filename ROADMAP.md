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
  **Resolved**: the `lib3mf` dependency is `telecos/lib3mf_rust` v0.1.6+,
  added with `default-features = false` (skipping its `mesh-ops`/
  `polygon-ops` extras — parry3d/nalgebra/clipper2/earcutr — which
  `manifold-core` doesn't need for basic parsing).
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
- **File-open dialogs**: [`rfd`](https://crates.io/crates/rfd) v0.15 —
  native file-open dialog, added to `manifold-gui` for mesh import
  (Phase 4).

## Phase 0 — Domain model (prerequisite for everything below) — ✅ done

`manifold-core` is the only crate allowed to hold this domain logic (see
`CODE_STYLE.md`) — none of it belongs in the GUI/CLI layers. Landed as
`ids`, `bounds`, `transform`, `tool`, `material`, `object`, `machine`, and
`workspace` modules in `manifold-core/src/`:

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

- STL loader (`stl_io`) → `Mesh`. **✅ done** —
  `manifold_core::stl::load_stl`, wired into `manifold-cli` for `.stl`
  input (wrapped into a single `Object` with identity transform, tool 0,
  since STL has no build-item/transform/material concept of its own).
- 3MF loader (`lib3mf`) → populates `Object`s directly (3MF natively
  models build items/transforms/materials, which is exactly the
  `Object`/`Tool`/`Material` shape from Phase 0). **✅ done** —
  `manifold_core::threemf::load_3mf`, wired into `manifold-cli` for
  `.3mf` input. Follow-ups not yet covered: flattening `Object`
  assemblies/`components` into world-space geometry, and extracting the
  Materials extension into `Material`/`MaterialId`.

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

## Phase 4 — GUI shell/layout (needs 0) — ✅ done

- `egui::SidePanel::left` — settings (global `SlicerConfig`) + object list.
  Per-object tool/material assignment UI not yet added (needs Phase 8
  end-to-end wiring to be meaningful).
- `egui::CentralPanel` — 3D viewport, with an in-panel top toolbar (import
  only so far; select/move/rotate/scale modes, slice, export land with
  Phase 7/8).
- Wired `rfd` for mesh import (STL/3MF via Phase 1 loaders), mirroring
  `manifold-cli`'s extension-dispatch `load_objects`.

## Phase 5 — 3D rendering pipeline (needs Phase 4) — ✅ done

- Minimal wgpu triangle-mesh pipeline (`manifold-gui/src/render.rs`,
  `mesh_shader.wgsl`): flat per-triangle-face-normal lit shading, embedded
  via `egui_wgpu::Callback`/`CallbackTrait` per the verified
  `egui-wgpu 0.29.1` pattern (resources stored in
  `Renderer::callback_resources`, camera uniform updated in `prepare`,
  draw issued in `paint`).
- Orbit camera (`manifold-gui/src/camera.rs`): drag-to-rotate, secondary-
  drag-to-pan, scroll-to-zoom.
- **Note**: `Mesh` has no per-vertex normal data, so normals are computed
  as flat per-triangle face normals at upload time, expanding to a
  non-indexed vertex buffer (each triangle's 3 vertices duplicated with
  that face's normal) rather than reusing `Mesh.indices` directly.
- **Note**: object transforms are baked into vertex positions at import
  time (CPU-side), not applied as a per-draw model-matrix uniform.
  Interactive per-object transform editing is Phase 7's job (gizmos); this
  keeps this phase's scope contained. Revisit if/when Phase 7 needs
  live-updating transforms without a full re-upload.

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
  tool assignment flag) building a `Workspace`. **Partially done**: 3MF
  input now builds a multi-object `Workspace` via one file; still needs
  multiple input files and per-file tool assignment.

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

1. ~~Disambiguate `lib3mf` (telecos) vs. the `lib3mf-core` family
   (sscargal) before adding either as a dependency (Phase 1).~~ **Resolved**:
   `lib3mf` (telecos/lib3mf_rust) is the dependency in use; see Phase 1.
2. Sequential vs. naive-simultaneous multi-object print ordering for v1,
   given collision avoidance is deferred (Phase 2).
3. Stay on `eframe`/`egui` 0.29 or upgrade workspace-wide for
   `transform-gizmo-egui` compatibility (Phase 4/7).
