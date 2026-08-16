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

## Phase 2 — Multi-object / multi-tool slicing pipeline (needs 0 + 1) — ✅ done

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

**Implementation notes**: print ordering is v1-sequential (whole object
at a time) but built **pluggable** from the start, landed as
`manifold-core/src/ordering.rs`: an `ObjectOrderStrategy` trait
(`fn order(&self, objects: &[Object]) -> Result<Vec<ObjectId>>`), one
implementation (`SequentialOrder`, declaration order), and a
config-selectable `ObjectOrderingKind` enum (`Serialize`/`Deserialize`/
`Default`, currently just `Sequential`) resolved to a strategy via
`strategy_for`. `SlicerConfig` gained `object_ordering: ObjectOrderingKind`
so the choice persists like any other slicing parameter. Adding a future
algorithm (naive-simultaneous, Z-interleaved, eventually
collision-aware) is an additive change: new enum variant + new struct
implementing the trait + one match arm in `strategy_for` — no changes to
the rest of the pipeline.

`slicing::Layer` gained an `object: ObjectId` tag. `slicing::slice_object`
bakes an `Object`'s `transform` into world-space vertices (same
transform-at-upload-time pattern the GUI renderer already uses) before
slicing; `slicing::slice_workspace(objects, order, config)` slices every
object in the strategy's chosen order and concatenates their layer stacks
back-to-back — that concatenation *is* what makes ordering "sequential";
a future Z-interleaving strategy would replace it with a per-Z merge
across objects instead of a straight `extend`.

`toolpath::Path` gained a `tool: ToolId` field; `toolpath::plan` now takes
`&[Object]` alongside `&[Layer]` to look up each layer's source object's
assigned tool and tag the resulting path with it. `gcode::emit` tracks the
last-emitted tool and inserts a `T{n}` tool-select line whenever
consecutive paths differ in `tool`. Prime/purge Gcode around a tool change
is **not yet implemented** — real toolpath/Gcode content is still a
placeholder pipeline (per Phase 0/2 scope), so this is a follow-up once
actual path planning lands, not a gap introduced by this phase.

`slice_to_gcode` (`lib.rs`) now rejects an empty workspace up front, then
wires `strategy_for(config.object_ordering).order(&objects)` →
`slice_workspace` → `toolpath::plan` (passed `objects`) → `gcode::emit` —
replacing the old "slices only the first object" placeholder.

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

## Phase 6 — Scene content: origin, bed/substrate, toolhead (needs 3 + 5) — ✅ done

- Origin gizmo: fixed-size axis triad at world origin.
- Print bed/substrate mesh + grid, sized from `Machine` (Phase 3).
- Toolhead placeholder geometry, positioned per `Tool` offset; multiple
  toolheads for multi-tool machines.

**Implementation notes**: landed as `manifold-gui/src/scene.rs` (pure
geometry builders operating on `manifold_core::machine::Machine` — no
wgpu types, kept separate from `render.rs`'s GPU concerns):
`build_origin_axes` (fixed-length RGB line triad), `build_grid` +
`build_bed_quad` (line-list grid and translucent triangle-list quad, both
sized from `Machine::build_volume`'s bounding box), and
`build_toolhead_markers` (a small pyramid per `Tool`, positioned at
`tool.mount`'s translation — already iterates all of `machine.tools`, so
multi-tool machines get one marker each with no further work needed).
Uploaded via `ManifoldApp::build_scene` into a `render::UploadedScene`
(separate line-list and triangle-list vertex buffers) and painted each
frame via `render::ScenePaintCallback` alongside the mesh callback.
Rebuilt whenever the settings panel edits `machine.build_volume` (wired
as part of Phase 3), so bed/toolhead dressing stays live without an app
restart.

## Phase 7 — Interactive transform gizmos (needs 4 + 5) — ✅ done

- Integrate `transform-gizmo-egui` for per-object move/rotate/scale in the
  3D viewport, wired to selection state and `Object.transform`.

**Implementation notes**: landed in `manifold-gui/src/app.rs`. Selecting
an object in the Phase 4 settings-panel object list (or via the Phase 9
MCP `select_object` tool) sets `ManifoldApp::selected`; `viewport()` then
drives a single reused `Gizmo` (`transform_gizmo_egui::prelude::Gizmo`)
configured with the camera's view/projection matrices, `GizmoMode::all()`
(move + rotate + scale together), and `GizmoOrientation::Local`. The
gizmo is fed the selected object's current transform (decomposed via
`glam`'s `to_scale_rotation_translation`), and `Gizmo::interact()`'s
result is written back into `Object.transform` and triggers a mesh
reupload so the dragged object updates live. Deliberately painted via
plain egui geometry *after* the Phase 5/6 wgpu paint callbacks in the
same `Ui` so it composites on top of the 3D scene rather than being
z-tested against it. Multi-select (dragging several objects at once) is
out of scope — `self.selected` is a single `Option<usize>`.

## Phase 8 — End-to-end wiring (needs all of the above) — ✅ done

- GUI: import → arrange/assign tool in 3D view → configure settings →
  Slice action in toolbar → `slice_to_gcode(&Workspace)` → preview/export.
- CLI: extend `manifold-cli` to accept multiple input files (+ per-file
  tool assignment flag) building a `Workspace`. **Partially done**: 3MF
  input now builds a multi-object `Workspace` via one file; still needs
  multiple input files and per-file tool assignment.

**Implementation notes**: `manifold-cli`'s `inputs` is now `Vec<String>`
(`num_args = 1..`); each entry is `path` or `path:tool` (e.g.
`part.stl:1`, defaulting to tool `0`) — `parse_input_entry` splits the
suffix, `load_objects` allocates sequential `ObjectId`s across every file
and assigns the parsed `ToolId`, and `tools_for` builds one `Tool` per
distinct tool id referenced (sorted/deduped), all sharing
`--nozzle-diameter` today (per-tool nozzle diameter flags are a further
follow-up, not blocking this phase).

`manifold-gui`'s `app.rs` gained: an "Add tool" button in the Machine
section (`Tool::new` with an incrementing `next_tool_id`, starting after
the one default tool); a per-object tool-assignment `egui::ComboBox` in
the Objects list (writes `object.tool` directly, listing every
`machine.tools` entry by id); a "Slice"/"Export…" pair in the viewport
toolbar ("Slice" enabled once objects are loaded, builds a `Workspace`
from current `objects`/`machine`/`config` and calls
`manifold_core::slice_to_gcode`, storing the result in `self.gcode` or
`self.slice_error`; "Export…" enabled once `gcode` is `Some`, opens
`rfd::FileDialog::save_file` and writes it out); and a Gcode preview
section in the settings panel (line count + a scrolling `ui.monospace`
dump) shown whenever `self.gcode` is populated.

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

## Phase 10 — Settings profiles (needs 3 + 8, no new manifold-core domain logic) — ✅ done

Save/load named presets for `Machine` + `SlicerConfig` so a user doesn't
have to re-enter bed size, tool layout, layer height, etc. each session.

**Implementation notes**: landed as `manifold-gui/src/profile.rs` —
a `Profile { machine: Machine, config: SlicerConfig }` struct (JSON via
`serde_json`, pretty-printed) with `Profile::save`/`Profile::load`
helpers. `serde`/`serde_json` moved from optional (`mcp-server`-gated) to
required deps of `manifold-gui`. "Save Profile…"/"Load Profile…"
buttons sit in the settings panel's Machine section, using the same
`rfd` dialog pattern as Import/Export (filtered to `*.json`). Loading a
profile replaces `self.machine`/`self.config`, recomputes `next_tool_id`
from the loaded tools' max id, and rebuilds the scene (bed geometry may
have changed). Deliberately excludes `objects`/`selected` — a profile
captures printer + slicer setup, not an in-progress project. No
profiles directory/registry UI yet; users pick a file location each
time via the save/open dialog, same as Gcode export.

## Phase 11 — Order-field-driven slicing MVP (needs 0, promotes `NON_PLANAR_SLICING.md`'s spike work) — ✅ done

Prove out the "SDF -> order field -> isosurface walk -> toolpath" pipeline
shape described in `NON_PLANAR_SLICING.md`, using the simplest possible
`order` field (a plain height field along the build/gravity direction,
whose isosurfaces are flat planes — i.e. this deliberately reduces to
conventional planar slicing as a first, zero-geometric-risk validation
step), replacing the previous empty-placeholder `slicing`/`toolpath`
pipeline in `manifold-core` with a real (if minimal) one.

- New `manifold-fidget::order` module: an `OrderField`-shaped abstraction
  (reusing the existing `ScalarField` trait shape) with one concrete
  implementation, `HeightOrderField`, whose isosurfaces are flat planes
  along a chosen direction.
- New planar contour extraction in `manifold-fidget` (marching squares
  over a sampled plane grid), producing closed polylines at a given
  `order` value — the first isoline/contour extractor in the crate,
  alongside the existing 3D `marching_cubes` isosurface extractor.
- `manifold-core::slicing::slice_mesh` becomes real: builds a `MeshSdf`,
  steps `HeightOrderField` values at `config.layer_height` intervals, and
  extracts contour loops per step into `Layer`.
- `manifold-core::toolpath::plan` becomes real (minimally): one `Path`
  per contour loop, tagged with its object's tool via the existing
  (unchanged) tool-assignment logic.
- **New dependency**: `manifold-core` now depends on `manifold-fidget`
  (previously used only by `manifold-gui` for visualization) — the first
  time the slicing pipeline itself consumes fidget's field/SDF machinery.

**Implementation notes**: this is the first phase promoted out of
`NON_PLANAR_SLICING.md`'s "research spike, not committed work" status
into committed `ROADMAP.md` work, per that document's own gating rule
(no phase there is committed until promoted here). It deliberately stays
at the degenerate flat-height-field case — proving the pipeline shape
(SDF -> order field -> contour walk -> toolpath) with minimal risk, not
the actual non-planar capability. The angle-driven Eikonal `speed(p)`
field / "Alternative `order` construction" idea from
`NON_PLANAR_SLICING.md` is explicitly **not** part of this phase; it
remains future work once this pipeline shape has proven out. Still
deferred here, same as the placeholder pipeline it replaces: multiple
perimeters/shells, infill, travel-move ordering/optimization between
loops, and any actual non-planar toolpath deformation (this phase's
toolpaths are still flat, planar loops — only the pipeline *shape* is
new).

## Phase 12 — Motion metadata in manifold-core::toolpath (needs 11) — ✅ done

Extend the toolpath data model so each *segment* (point-to-point move within
a contour loop `Path`) carries classification + kinematic metadata, without
adding any real detection logic yet — preparatory vocabulary/plumbing for
the Phase 13 GUI toolpath-preview follow-on.

- New `manifold-core::toolpath::MoveKind` enum: `WallOuter`, `WallInner`,
  `Infill`, `Bridge`, `Overhang`, `Travel`.
- `Path` reshaped to carry per-segment metadata via a parallel `segments:
  Vec<Segment>` (kept alongside `points: Vec<DVec3>` rather than paired in a
  single `Vec<(DVec3, Segment)>`, so geometry-only callers can read `points`
  without touching `segments`). `Segment` holds `kind: MoveKind`, `speed:
  f64`, `extrusion_rate: f64`, `support_fraction: f64`, and `order: f64`.
- `Layer` threads through the `order`-field value (from
  `manifold_fidget::order`/`HeightOrderField`) its contours were extracted
  at, so `toolpath::plan` can stamp it onto every `Segment` per-segment
  (not per-`Path`/per-`Layer`), ready for non-planar order fields later.
- `toolpath::plan` populates every `Segment` with `MoveKind::WallOuter`
  (the closest existing case), fixed placeholder `speed`/`extrusion_rate`
  defaults, and `support_fraction: 0.0`.
- `gcode::emit` updated to walk the new `Path`/`Segment` shape, deriving
  `G0`-vs-`G1` from `segment.kind != MoveKind::Travel` instead of a
  separate `extruding: bool` field (removed from `Path`).

**Explicitly deferred**: real wall/inner-wall/infill/support/bridge/
overhang *classification* — `plan()` still tags everything
`MoveKind::WallOuter` as a placeholder; actual detection logic is future
work. Phase 13 (GUI toolpath preview: rendering, scrubbing-by-order, hover
tooltips) is the follow-on consumer of this metadata and is out of scope
here.

**Addendum (multi-wall slicing)**: real wall classification has since
landed (see `wall_line_width`/`shell_thickness`/`wall_offset` below,
not part of the original Phase 12 scope) — `slicing::slice_mesh` now
extracts one contour loop per wall pass at successively deeper SDF iso-
values (`manifold_fidget::mesh_sdf::MeshSdf` is positive outside/negative
inside, so inward walls sample at negative iso), tagged with a
`WallLoop::wall_index`; `toolpath::plan` derives `MoveKind::WallOuter`
(index 0) vs. `MoveKind::WallInner` (index > 0) from that index instead
of the old always-`WallOuter` placeholder. Infill/support/bridge/overhang
classification remains future work.

## Phase 13 — GUI toolpath preview: rendering, order scrub, hover tooltips (needs 12) — ✅ done

Consumes Phase 12's per-`Segment` metadata to render an interactive
toolpath preview in `manifold-gui`, without adding any new manifold-core
domain logic.

- `manifold-gui`: new toolpath line-rendering pipeline (`render.rs`) plus
  a `toolpath_shader.wgsl` line shader with a per-vertex `order: f32`
  attribute (reserved for a possible future shader-side discard), driven
  by a `ManifoldApp.toolpaths: Option<Vec<manifold_core::toolpath::Path>>`
  populated from `slice()`/`reupload_toolpaths()`.
- "Show toolpaths" checkbox toggles the pipeline on/off.
- Order-based scrub slider (`egui::Slider`, "up to and including"
  semantics: segments with `order <= scrub_order` are drawn) implemented
  via CPU-side rebuild-on-change filtering (`toolpath_view.rs`) rather
  than a shader-side discard — simpler given the existing bind-group
  setup, accepted tradeoff being a full CPU rebuild + GPU re-upload per
  slider-drag frame at this MVP single-object scale.
- Hover tooltip: reuses the existing `world_to_screen` projection helper
  to do a CPU-side O(n) nearest-segment scan (screen-space point-to-
  segment distance, ~8px threshold) over the current scrub-filtered
  segment set, showing `kind`, `speed`, `extrusion_rate`,
  `support_fraction`, and `order` in an egui tooltip when hovering near a
  rendered segment.

**Explicitly deferred**: real classification-driven styling (e.g. color-
coding by `MoveKind` once Phase 12's placeholder classification is
replaced with real detection), and any shader-side discard for the scrub
slider (the `order` vertex attribute is wired but unused for this).

## Phase 14 — Multi-wall (shell) slicing (needs 11 + 12) — ✅ done

Adds configurable shell thickness so the slicer produces more than the
original single outer-wall contour, tracing additional inner-wall passes
inward from the surface.

- New `SlicerConfig` fields: `wall_line_width` (nozzle-center line width
  per wall pass, also the spacing between walls; defaults to
  `nozzle_diameter`), `shell_thickness` (total desired wall depth, mm;
  defaults to `wall_line_width`, i.e. one wall), and `wall_offset` (inset
  of the outermost wall's nozzle-center path from the true surface;
  defaults to `nozzle_diameter / 2.0`).
- `SlicerConfig::wall_count()` derives the actual pass count as
  `round(shell_thickness / wall_line_width)`, clamped to a minimum of 1.
- **Deliberately kept separate from the order field**: the order field
  (`manifold_fidget::order::OrderField`/`HeightOrderField`) only decides
  *which plane* to slice at; wall inset is a different axis entirely,
  expressed directly as an `MeshSdf` iso-value offset within a plane
  (negative = inward, since the SDF is positive outside/negative inside).
  Multiplexing wall inset through the order field's sign was considered
  and rejected — it would tie inset distances to the order field's
  gradient magnitude (only trivially 1.0 for `HeightOrderField`; not
  necessarily 1.0 for a future curved/non-planar field), silently
  corrupting wall spacing once Phase 11's deferred non-planar order
  fields land.
- `slicing::Layer.loops` changed shape from `Vec<Vec<DVec3>>` to
  `Vec<WallLoop>` (`WallLoop { wall_index: usize, points: Vec<DVec3> }`)
  so `toolpath::plan` can classify `MoveKind::WallOuter` (index 0) vs.
  `MoveKind::WallInner` (index > 0) — the first real (if still simple)
  move classification, closing out part of Phase 12's placeholder.
- GUI: settings panel gained "Wall line width", "Shell thickness", and
  "Wall offset" sliders alongside the existing layer-height/nozzle-
  diameter controls.

**Explicitly out of scope**: inner-wall-specific speed/extrusion-rate
tuning (still fixed placeholder defaults per existing `Segment` fields),
infill, and any change to inner-wall travel-move ordering.

## Phase 15 — Pluggable order fields + curved contour extraction (needs 11 + 14, promotes `ConicalOrderField`/`NON_PLANAR_SLICING.md` follow-on) — ✅ done

Generalizes `manifold-core::slicing::slice_mesh` to slice along an
arbitrary `manifold_fidget::order::OrderField` instead of the hardcoded
`BUILD_DIRECTION`/`HeightOrderField` (see `crates/manifold-fidget/src/
order.rs`'s `ConicalOrderField`, added ahead of this phase as a concrete
curved field to design against), and generalizes contour extraction so a
curved order field's isosurfaces produce real curved walls/infill instead
of silently falling back to flat slicing.

- `SlicerConfig` gains a config-selectable order-field choice (mirroring
  Phase 2's `ObjectOrderingKind` pattern: a serializable enum + a
  `strategy_for`-style constructor), so `HeightOrderField` remains the
  default/zero-risk case and `ConicalOrderField` (or later fields) are
  opt-in, not a breaking change to existing profiles/configs.
- `slice_mesh`/`slice_mesh_with_progress` take the resolved `OrderField`
  instead of reading `BUILD_DIRECTION` directly; `order_min`/`order_max`
  bounds must be derived generically (sampling the field's range over the
  mesh's bounding box) rather than via `min.dot(direction)`/
  `max.dot(direction)`, which is `HeightOrderField`-specific.
- **Contour extraction generalization, resolved design ("contour-on-mesh")**:
  a layer's wall loop is the *intersection of two implicit surfaces* —
  the wall's `MeshSdf` isosurface (`sdf(p) == wall_iso`) and the order
  field's isosurface (`order(p) == c`). Today's `extract_contours` only
  works because that intersection happens to be planar for
  `HeightOrderField`; for a curved field (e.g. `ConicalOrderField`) it is
  a genuine 3D space curve. Rather than inventing a bespoke dual-field
  marching-cubes variant, reuse existing pieces:
  1. Extract the wall surface **once per wall pass** (not once per
     layer — an efficiency win over today's per-layer plane sampling) as
     a triangle mesh via the already-generic
     `manifold_fidget::marching_cubes::extract_isosurface::<MeshSdf>` at
     the wall's SDF iso-value. No changes needed to that module; it
     already works over any `ScalarField`.
  2. Evaluate `order(p)` at every extracted vertex (one field evaluation
     each, done once after step 1 — kept as a separate pass so
     `marching_cubes` itself stays decoupled from `OrderField`).
  3. For each target layer value `c`, walk the triangle mesh's edges for
     `order`-crossings: per triangle, find edges whose endpoints straddle
     `c` and linearly interpolate the crossing point (the same
     `lerp_crossing` idea `contour.rs` already uses for marching
     squares), producing loose line segments.
  4. Stitch those segments into closed loops by generalizing
     `contour.rs`'s existing `stitch_loops`/`point_key`/
     `canonicalize_orientation` helpers (dedup-by-position-key,
     walk-shared-endpoints, canonicalize winding) — that logic doesn't
     appear to assume planarity anywhere, so it should adapt with
     little change from the marching-squares segment-soup case to the
     triangle-mesh-edge-crossing segment-soup case.
  **Open risk to validate before implementing**: triangle-soup
  degeneracies (T-junctions, near-tangent order-field crossings at edges
  shared by non-adjacent-looking triangles, since `extract_isosurface`'s
  output has no shared-vertex indexing) could produce gaps the same way
  the existing marching-squares stitcher already has to guard against —
  worth a stress test specifically against `ConicalOrderField`, whose
  isosurfaces are provably non-planar, before trusting this on real
  meshes.
- `Layer.loops`/`infill_boundary`/`solid_fill_boundary` stay `Vec<DVec3>`-
  based polylines (no data-model change) — only *how* those points are
  generated changes, not what downstream (`toolpath::plan`, `gcode::emit`,
  GUI preview) consumes.
- Non-planar wall/infill support in the GUI preview (Phase 13's toolpath
  view) should need no changes if the above stays within the existing
  `Path`/`Segment` data model — worth a smoke test once this phase lands,
  not a design requirement here.

**Explicitly out of scope**: any new order-field *shape* beyond
`ConicalOrderField` (e.g. the Eikonal/front-propagation `speed(p)`
construction from `NON_PLANAR_SLICING.md`'s "Alternative order
construction" section remains its own unpromoted spike); real toolpath-
level adaptations for non-planar printing itself (retraction/Z-hop
behavior, nozzle collision with already-curved geometry) — this phase is
about the slicing *pipeline* producing correct curved geometry, not the
full physical printability story.

## Phase 16 — Top/bottom solid infill layers (needs 11 + 14) — ✅ done

Adds solid (fully dense) infill near the top and bottom of a print — the
standard "N top layers / N bottom layers" shell behavior — and, along the
way, fixes a reported bug where `Layer.infill_boundary` could come back
empty even when wall geometry existed (e.g. near thin surface detail),
by replacing the old 3D SDF probe for `infill_boundary` with a 2D offset
of the innermost wall loop.

- New `crates/manifold-core/src/polygon2d.rs` module wraps `i_overlay`
  (pure-Rust polygon boolean ops/offsetting, no C/C++ FFI) behind
  `to_2d`/`from_2d` (using the same plane-basis convention as
  `manifold_fidget::contour::plane_basis`), `inward_offset`,
  `difference`, `union`, `intersection` — all `DVec3`-loop-in,
  `DVec3`-loop-out at the module boundary. Repeated per-layer overlay
  calls reuse `FloatOverlay`/`reinit_with_subj_and_clip` rather than
  `SingleFloatOverlay` to avoid reallocation churn.
- `slice_mesh_with_progress`'s `infill_boundary` computation dropped its
  3D `boundary_iso` SDF probe entirely in favor of
  `polygon2d::inward_offset` of the innermost wall loop (the highest
  `wall_index` in `Layer.loops`) by `wall_line_width` — `infill_boundary`
  is now non-empty whenever an innermost wall loop exists, regardless of
  3D depth, directly fixing the empty-`infill_boundary`-despite-walls bug.
- New `SlicerConfig` fields `top_layers`/`bottom_layers` (`usize`, default
  3 each), matching Phase 14's `wall_line_width`/`shell_thickness` style.
- New `Layer.solid_fill_boundary: Vec<Vec<DVec3>>` field and
  `compute_solid_fill_boundaries(layers: &mut [Layer], config: &SlicerConfig)`
  post-pass (run once per object, after the per-layer parallel slice loop,
  so one object's solid-layer detection never leaks into a neighboring
  object's layer stack): for each layer, the region exposed to open air
  above/below (`infill_boundary(i)` minus the neighboring layer's
  `infill_boundary`, treating a missing neighbor as fully exposed) is
  unioned across `top_layers`/`bottom_layers` neighbors in each direction
  and intersected back with that layer's own `infill_boundary` — all via
  `polygon2d` boolean ops, no SDF/3D queries involved.
- `infill::InfillRegion`'s sparse fillable region is now
  `infill_boundary \ solid_fill_boundary` (via `polygon2d::difference`)
  instead of all of `infill_boundary`; `toolpath::plan` generates an
  additional infill pass over `solid_fill_boundary` using the same
  `generator_for(config.infill_pattern)` as sparse infill, appended
  alongside the existing sparse-infill paths under the existing
  `MoveKind::Infill` — no new solid-fill-specific pattern, generator, or
  `MoveKind` was introduced.

**Explicitly out of scope**: a distinct dense infill *pattern* for solid
layers (e.g. always-rectilinear regardless of `config.infill_pattern`),
per-region extrusion-rate tuning for solid vs. sparse infill, and any
change to wall generation itself.

**Open decision deferred**: this phase's exposed-region detection compares
each layer's `infill_boundary` against its immediate Z-neighbor's
`infill_boundary` as flat 2D polygons — correct for today's planar
slicing, but a print surface that is sloped or curved (Phase 15's
pluggable/curved order fields, once landed) will need exposed-region
detection generalized beyond simple layer-to-layer polygon comparison
(e.g. accounting for a slanted top surface spanning several layers'
worth of Z at a shallow angle). That generalization is deferred to
whichever future phase promotes non-planar toolpath printing itself,
not tackled here.

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
2. ~~Sequential vs. naive-simultaneous multi-object print ordering for
   v1, given collision avoidance is deferred (Phase 2).~~ **Resolved**:
   sequential (whole-object-at-a-time) for v1, but implemented behind a
   pluggable `ObjectOrderStrategy` trait + config-selectable
   `ObjectOrderingKind` so alternative algorithms (naive-simultaneous,
   Z-interleaved, eventually collision-aware) can be added later without
   touching the slicing/toolpath/gcode pipeline — see Phase 2.
3. ~~Stay on `eframe`/`egui` 0.29 or upgrade workspace-wide for
   `transform-gizmo-egui` compatibility (Phase 4/7).~~ **Resolved**: stayed
   on `eframe`/`egui` 0.29; `transform-gizmo-egui = "0.4.0"` is compatible
   as-is (see Phase 4/5/7, all landed against 0.29).
