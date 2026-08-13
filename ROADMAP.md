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

## Phase 3 — Machine/printer definition (needs 0) — ✅ done

- Flesh out `Machine`: bed size/shape, build volume height, toolhead
  count/geometry/offsets — feeds both slicing bounds-checks and GUI scene
  visualization (Phase 6).

**Implementation notes**: the `Machine`/`BoundingVolume`/`Tool` domain
model already existed from Phase 0; Phase 3 replaces the hardcoded
`default_machine()` placeholder's static role with a live, editable
`Machine` field on `ManifoldApp` (`app.rs`). The settings panel gained a
"Machine" section: bed X/Y/Z (`build_volume`) sliders and the tool 0
nozzle diameter, which rebuild the Phase 6 scene dressing
(`Self::build_scene`) on change. Multi-tool count/geometry editing is
still Phase 8 territory (needs per-object tool assignment UI to be
meaningful) and machine persistence (save/load a project file) is not yet
implemented — the edited `Machine` only lives for the app session.

Also landed alongside this (not itself part of the roadmap, but needed a
real bed size to be meaningful): newly-imported objects are now
auto-centered on the bed instead of sitting at the origin —
`manifold_core::object::center_on_bed` computes the combined world-space
bounding box of a freshly-loaded group (preserving relative placement,
e.g. a 3MF assembly's build-item transforms) and translates it to be
XY-centered on `Machine::build_volume` and resting on its floor (minimum
Z). Wired into `ManifoldApp::import` in the GUI; `manifold-cli`'s
single-shot batch import is unchanged (out of scope — no interactive
viewport to center for).

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

## Phase 9 — MCP automation server for GUI testability (needs 4 + 7, dev/test-only) — ✅ done

Expose an MCP (Model Context Protocol) server from inside `manifold-gui`
so an agent/test harness can drive and inspect the app programmatically
(select objects, set transforms, import files, read state, screenshot),
without synthesizing real pointer/keyboard input. This repurposes MCP's
tool-call transport as an automation RPC layer — legitimate for this use
case because the calling agent already speaks MCP, not because this is
MCP's intended end-user tool-calling purpose. Gate the whole thing behind
a Cargo feature (e.g. `mcp-server`, off by default) — it must not be
present in a release binary (a standing debug automation port is not
something to ship).

- **Crate/dependency**: `rmcp` (the official Rust MCP SDK,
  `modelcontextprotocol/rust-sdk`) — actively maintained, supports
  stdio, HTTP/SSE, and generic async read/write transports. Use HTTP/SSE
  or plain TCP on `127.0.0.1:<port>`, **not stdio** — `eframe` already
  owns the process's stdio (logging via `tracing_subscriber::fmt::init()`
  per `CODE_STYLE.md`), so only one thing can own it.
- **Threading model**: spawn the MCP server on its own background thread
  with a small tokio runtime, using `rmcp`'s worker-transport pattern
  (`serve_server_with_ct` + `LocalSessionHandle`/`LocalSessionWorker`) —
  designed exactly for running the server's message loop independently of
  a host application's own event loop (here, `eframe::run_native`'s
  blocking native loop on the main thread).
- **Crossing the thread boundary**: no cross-thread `egui::Context`/UI
  mutation. Each MCP tool call sends a `Command` enum (e.g.
  `SelectObject(usize)`, `SetTransform{index, transform}`,
  `ImportFile(PathBuf)`, `Screenshot`) over an `mpsc::Sender<Command>`
  owned by `ManifoldApp`, then calls `ctx.request_repaint()` (documented
  as safe from any thread — this is how `eframe` already wakes its loop
  for background events). `ManifoldApp::update()` drains the channel at
  the top of each frame and applies commands on the UI thread, same as
  all existing state mutation. Query-style tools (`get_objects`,
  `screenshot`) pair the command with a `oneshot::Sender<Response>` so the
  MCP handler can await the result after the next frame processes it.
- **In scope**: object list state (ids/triangle counts/selection),
  programmatic `select_object`/`set_transform` (bypassing gizmo drag math
  entirely — actually more precise for testing than synthesizing pointer
  drags), headless `import_file` (STL/3MF), and eventually a screenshot
  tool (read back the wgpu surface texture via a copy-to-buffer +
  map-read — more plumbing than everything else combined, lowest
  priority).
- **Out of scope**: injecting synthetic pointer drags to test
  `transform-gizmo-egui`'s drag interaction pixel-accurately. egui's input
  model wants a `RawInput` fed into `Context::run_ui` per frame — normally
  `eframe`'s job, not something to fight from a side channel. Gizmo drag
  interaction itself is better tested via a separate headless
  `egui::Context::run` integration-test harness (no window, no MCP
  needed) — a distinct, more contained effort from this phase.

**Implementation notes**: landed as `manifold-gui/src/mcp.rs`, gated
behind the `mcp-server` Cargo feature (off by default — build/run with
`--features mcp-server` to enable it). Uses `rmcp` 0.9 + `schemars` 1 +
`tokio` (private `Builder::new_multi_thread()` runtime inside a spawned
`std::thread`, raw TCP via `transport-async-rw`, no HTTP framework). Fixed
port `127.0.0.1:8931`. Tools landed: `list_objects`, `get_selected`,
`select_object`, `set_transform` (translation only — rotation/scale
preserved from the object's current transform), `import_file`. The
screenshot tool (lowest priority per the note above) is not yet
implemented.

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
3. ~~Stay on `eframe`/`egui` 0.29 or upgrade workspace-wide for
   `transform-gizmo-egui` compatibility (Phase 4/7).~~ **Resolved**: stayed
   on `eframe`/`egui` 0.29; `transform-gizmo-egui = "0.4.0"` is compatible
   as-is (see Phase 4/5/7, all landed against 0.29).
