# Toolpath Generation with AnisotropicFSM Slicing

Design document for the toolpath-planning stage of Manifold's slicing
pipeline: how a set of sliced layers becomes an ordered list of
extrusion moves (`toolpath::Path`/`Segment`) ready for G-code emission.

**Assumption:** the upstream slicing stage produced its `Layer[]` from
the **Anisotropic Fast Sweeping Method (FSM)** order field
(`OrderFieldKind::AnisotropicFsm`,
`manifold_fidget::fsm::AnisotropicFsmOrderField`) — i.e. each layer is
the level set of a scalar deposition-order function
$\phi(\mathbf{x})$ solved from the anisotropic Eikonal equation
$\nabla\phi^T \mathbf{D}(\mathbf{x}) \nabla\phi = 1$ on a 3D Cartesian
grid, *not* a flat horizontal cut at a fixed Z height. The implications
of that assumption are stated explicitly throughout; where a later pass
could also run on a flat-height or FMM order field, the difference is
called out.

Scope: everything between `slicing::Layer[]` and `gcode::emit` — the
`manifold_core::toolpath` module and the sibling planners it invokes
(`infill`, `wave_overhang`, `bridge`, `tangent_surface`, `gap_fill`,
`corner_flow`, `transient_pressure`, `kinematics`, `extrusion`).
Slicing itself (order-field solve, contour extraction, wall-gap
stitching) is documented in `NON_PLANAR_SLICING.md` and
`ORDER_FIELD_SOLVERS.md`; G-code emission is covered only as the
handoff target (section 13).

## Overview

### Position in the pipeline

```text
Workspace (objects, machine, SlicerConfig)
   │
   ▼  slicing::slice_mesh_with_progress
   │    (AnisotropicFSM order-field solve + contour extraction)
   │
Layer[] ─────────────────────────────────────────────┐
   │                                                  │
   ▼  toolpath::plan / plan_with_progress             │
   │    (§2 orchestration)                            │
   │      pre-planners: wave overhangs, bridges,      │
   │                    tangent surfaces, gap fill    │ (§5)
   │      per layer (rayon, in parallel):             │
   │        walls (§3) → infill (§4) → surface paths  │ (§5)
   │        → containment & correction (§6)           │
   │        → simplification (§7)                     │
   │        → travel ordering & routing (§8)          │
   │        → Z-hops & traverse subdivision (§9)      │
   │        → extrusion/flow per segment (§10)        │
   │        → seam/wipe/corner post-passes (§11)      │
   │      global: support-aware emission deferral     │ (§12)
   │                                                  │
   ▼                                                  ▼
Vec<Path>  ───────────────────────────────►  gcode::emit (§13)
```

The public entry points are `toolpath::plan` (a convenience wrapper)
and `toolpath::plan_with_progress` (the real orchestrator, which adds
machine/slope-profile inputs and a progress callback). `slice_to_gcode`
/ `plan_toolpaths` in `lib.rs` call into these.

### What AnisotropicFSM slicing changes relative to a flat layer

Three properties of the FSM layer set drive the design of every later
pass:

1. **`Layer::order` is a field value, not a height.** The layer's
   scalar order $\phi$ determines *when* the layer prints; its
   geometry may be non-planar (the contour of $\phi = c$ can bend in
   Z). Nothing downstream may assume all points of a layer share a Z,
   or that consecutive layers are parallel. Per-segment `Segment::order`
   is stamped so G-code/GUI can report the true order value even as
   layers curve.
2. **The order field is carried on the layer.** Each `Layer` caches the
   resolved `OrderField` (here: `AnisotropicFsmOrderField`) plus a
   `MeshSdf` ground-truth containment query. Downstream passes probe
   `order(p)` and the SDF directly: support fractions, floating-material
   deferral (§12), travel routing (§8), and local layer geometry
   (§10) all sample these fields.
3. **"Below a layer" is defined by the order field, not by Z.**
   "Already printed" means *lower order value*; "supported" means a
   probe stepping against $\nabla\phi$ (or along gravity, where
   physical support matters) lands in solid. Several passes therefore
   run two complementary probes — an order-field probe and a
   gravity/SDF probe — and reconcile them (see §3, §10, §12).

Because the FSM solve is a PDE over anisotropic velocity tensors, the
layer stack can also carry *seeded* order values from patch surfaces
(`fsm_seed_surfaces_enabled`); `seed_proximity` then reports whether a
point is closer to the bed or to an upward-facing patch, which the
slicing stage uses to seed wavefronts but which toolpath planning
consumes only indirectly through the layer's `order` values.

### Data model

Toolpath planning's output is `Vec<Path>`, where a `Path` is a
continuous run of printer moves:

- `Path { points: Vec<DVec3>, segments: Vec<Segment>, tool: ToolId,
  object: ObjectId }` — closed loops (wall perimeters, rings) have
  `segments.len() == points.len()` with segment `i` describing the move
  `points[i] → points[(i+1) % N]` (the closing edge included); open
  polylines (infill scanlines) have `segments.len() == points.len() - 1`.
  The `points`/`segments` parallel-vector convention is used
  everywhere in this pipeline.
- `Segment { kind, speed, extrusion_rate, support_fraction, order,
  extrusion_length, line_width, is_scarf, id, island, channel_width,
  flow_breakdown }` — per-edge metadata. `kind` is a `MoveKind`:
  `WallOuter`, `WallInner`, `Infill`, `Bridge`, `Overhang`,
  `TopSurface`, `Travel`, `Wipe`, `DebugExcluded`.
- `FlowBreakdown { slope_cosine, first_layer_mult, directional_flow_mult,
  swell_mult, corner_flow_mult, transient_pressure_mult }` — the
  individual multiplicative flow adjustments that combine into
  `extrusion_length`, recorded *as actually applied* (not re-derived)
  for the GUI's tuning data views.

`Path::tool`/`Path::object` let the G-code emitter insert tool changes
and `EXCLUDE_OBJECT` markers without re-deriving ownership.

### Conventions used throughout

- All geometry is `glam::DVec3` (f64) in world space; the build
  direction is `slicing::BUILD_DIRECTION` (+Z) and the nozzle axis is
  `slicing::NOZZLE_DIRECTION` (flat, horizontal tip, i.e. the nozzle
  sweeps like a cylinder's axis — this drives the flat-nozzle
  compensation, §6).
- "The layer plane" for a given layer is spanned by
  `contour::plane_basis(axis)` around the layer's apex
  (`order_field::resolve_axis_apex_slope`); 2D polygon ops
  (`polygon2d`) always happen in that basis, then are projected back
  onto the order field (`order_field::reconstruct_on_order_field_near`).
- Safety nets prefer *dropping* a bogus path over clipping it, and
  grade containment checks rather than enforcing them exactly (§6).
- Layers are planned in parallel; the global passes (§8 global routing,
  §12 deferral, §13 end-of-print) run afterwards on the flattened
  sequence.

## Section map

1. **Input contract: layers from AnisotropicFSM slicing** — what
   `Layer`/`WallLoop` carry, the cached order field and `MeshSdf`,
   order-value semantics, bed/first-layer conventions.
2. **Top-level orchestration (`plan` / `plan_with_progress`)** — the
   pre-planners, per-layer parallel planning, progress reporting, and
   the post-per-layer global passes.
3. **Wall path construction & segment classification** — print order
   (inner-outer-inner), `MoveKind` assignment, debug loops,
   per-point line widths and channel widths.
4. **Infill & solid-fill planning** — `InfillRegion`, sparse vs. solid
   passes, generator selection, narrow-solid splitting, footprint
   masking against the surface passes.
5. **Dedicated surface passes** — wave overhangs (LaSO), bridges,
   tangent surfaces, and gap fill between inner walls.
6. **Containment, safety nets & geometric corrections** —
   mesh-SDF containment checks, void reclassification, bed-floor
   clamping, micro-path filtering, flat-nozzle slope clearance,
   outer-wall centerline pinning.
7. **Path simplification** — Ramer–Douglas–Peucker for closed and open
   paths.
8. **Travel-move ordering & collision-avoidance routing** —
   order-aware move sorting, A* detour routing around already-printed
   solid, Z-travel penalty, tangent-endpoint detours, waypoint
   shortcutting.
9. **Z-hops & long-traverse subdivision** — hop insertion on travels,
   subdividing long extrusion traverses for order-field fidelity.
10. **Per-segment extrusion & flow computation** — support fractions,
    SDF-based surface classification, local layer geometry, slope
    cosines, bead cross-section, speeds, volumetric clamps, first-layer
    and directional slope compensation, fluid-dynamics swell.
11. **Seam, wipe & corner post-passes** — corner-flow and transient
    pressure compensation, scarf joints, seam gaps, pre-retract taper,
    perimeter wipes.
12. **Support-aware emission deferral** — moving floating paths to
    after the layer group that supports them.
13. **End-of-print wipe/clearance, bounds validation & G-code handoff**
    — the terminal wipe/clearance paths, build-volume validation, and
    what `gcode::emit` consumes.

---

## 1. Input contract: layers from AnisotropicFSM slicing

### 1.1 What the slicing stage produces

`slicing::slice_mesh_with_progress` walks the configured order field's value range in layer-height steps (the step is calibrated per layer by `StepCalibration` against the wall-0 isosurface, so the *order-space* spacing is `layer_height` even when the local `||∇φ||` compresses or stretches the level sets) and, for each order value, extracts contour loops and builds a `Layer`. For the FSM kind the walk is over the PDE solution's range: `order_value` starts at `order_min + first_layer_height` and steps up to `order_max`, capped at `MAX_ORDER_STEPS` layers and, for non-planar fields, closed by an extra final layer at `order_max` (the summit layer).

Each `slicing::Layer` carries:

- `index: usize` — zero-based position in the print sequence (layers are walked in increasing order value).
- `object: ObjectId` — the source mesh; toolpath paths inherit it.
- `order: f64` — the scalar order value whose isosurface produced this layer. **Not a Z height** for the FSM kind; it is the deposition-time coordinate.
- `loops: Vec<WallLoop>` — the wall passes at this order value, outermost first; see §1.2.
- `infill_boundary: Vec<Vec<DVec3>>` — closed loops one `wall_line_width` further inward than the innermost *printed* wall (i.e. where wall pass `wall_count` would sit). This is the fillable area, kept separate from `loops` so it is never mistaken for a printable perimeter. Empty when the layer has no contour.
- `solid_fill_boundary: Vec<Vec<DVec3>>` — the subset of `infill_boundary` that must print *solid* (within `top_layers` of a facing-up exterior surface or `bottom_layers` of a facing-down one). Filled by the post-pass `slicing::compute_solid_fill_boundaries` once every layer of the object exists; always a subset of `infill_boundary`.
- `mesh_sdf: Option<Arc<MeshSdf>>` — ground-truth mesh containment/distance query, built alongside the SDF used for contour extraction in `slice_mesh_with_progress` (the only site with mesh access). `None` for synthetic test layers; toolpath planning then treats containment as *unknown* rather than failing. This is the SDF that the safety nets (§6), surface classification (§10), travel routing (§8), and bridge/overhang inclusion checks (§5) all probe.
- `order_field: Arc<dyn OrderField>` — the *resolved* field cached at construction. For this document that is an `AnisotropicFsmOrderField` (possibly with patch seed metadata attached); `order_field_for_with_sdf` additionally wraps every kind in a `TopSurfaceAwareOrderField` decorator that overrides `seed_proximity` by marching along the local climb direction against a bed-contact-excluded SDF — unless `fsm_seed_surfaces_enabled` is set, in which case the FSM field's own solve-consistent `seed_proximity` is used unwrapped. Caching matters: the FSM grid solve cannot be re-derived from `config` alone at downstream call sites with no mesh in scope.

### 1.2 `WallLoop` — one wall pass at one order value

```rust
pub struct WallLoop {
    pub wall_index: usize,     // 0 = outermost, increasing inward
    pub island: usize,         // groups loops of the same disconnected island
    pub is_open: bool,         // open polylines don't close back to points[0]
    pub points: Vec<DVec3>,    // polyline in world space
    pub unsupported: Vec<bool>,// per-point: stitched across an inter-layer gap
    pub arc_fraction: Vec<f64>,// per-point normalized arc length in [0,1)
    pub top_surface: Vec<bool>,// per-point: roof material before open air
    pub line_widths: Vec<f64>, // per-point dynamic line width (mm)
    pub channel_width: Vec<f64>,// per-point local channel width (mm)
}
```

The parallel-vector convention (`points` + one sibling `Vec` per per-point property, all the same length) is the pipeline-wide pattern: geometry-only consumers read `points` without touching metadata.

- **`wall_index`** is the wall pass. `0` prints as `MoveKind::WallOuter`, `1..` as `MoveKind::WallInner` (§3). `wall_index >= 990` marks *debug* polylines (e.g. loops excluded by mesh containment checks, unclosed contour fragments): they become `MoveKind::DebugExcluded` open paths kept for visualization only and never emitted to G-code.
- **`island`** groups all wall passes of one disconnected island (outer boundary plus its enclosed holes) so planning can order each island's walls independently without interleaving unrelated islands. Copied to `Segment::island`.
- **`unsupported[i]`** is set by the inter-layer *wall-gap stitching* pass (`stitch_wall_gaps`) for points that were not derived from this layer's own isosurface but inserted to bridge a gap between this layer's wall-0 loop and the previous layer's. A segment whose *destination* point is `unsupported` is classified `MoveKind::Overhang` (§3): it will be extruded across material that the order-field isosurface says is not there yet.
- **`arc_fraction[i]`** is the normalized cumulative arc length around the loop (`arc_fraction[0] == 0`, monotonically increasing, wrapping toward 1). `stitch_wall_gaps` consumes it to build an arc-length correspondence between successive layers' wall-0 loops instead of raw nearest-point matching; it is retained (not recomputed-and-discarded) as the per-loop parameterization a future seam-placement feature needs.
- **`top_surface[i]`** marks wall-0 points that have solid mesh material directly beneath (real support) but nothing solid one nozzle diameter along `-BUILD_DIRECTION` — the last material before open air, i.e. the roof of the part there. A point can be `top_surface` *and* supported; when both `top_surface` and `unsupported` would classify the same segment, the genuine overhang classification wins (§3).
- **`line_widths[i]`** is a per-point dynamic line width in mm; when the `Vec` is empty, downstream planning falls back to the nominal configured width. For inner walls (§3) the gap-fill pass (§5.4) can override these with a per-point gap-fit width.
- **`channel_width[i]`** is the local 2D channel width in mm — `2 ×` the minimum distance from that point to the nearest opposing 2D boundary on the same wall pass (`polygon2d::channel_widths_3d`) — or `f64::INFINITY` when nothing nearby narrows the bead. Copied to `Segment::channel_width` and consumed by the bead-clearance clamp in §10 when `config.bead_clearance_compensation_enabled()`.

### 1.3 Order-value semantics (the AnisotropicFSM assumptions in use)

1. **Bed = order 0.** The FSM solve seeds its Dirichlet boundary at `order = 0` on the mesh's bed-contact region (`p.z <= min.z + seed_tolerance`) — the "rests on floor" convention shared with `object::center_on_bed` and the Eikonal seeding: the build plate is at the part's minimum Z, not necessarily world z=0. Toolpath planning mirrors this with `bed_z`, the minimum Z across *all* wall-loop points of the print, used to detect beads squished against the plate (§10).
2. **First-layer detection.** A layer is "layer 0" when `layer.index == 0` or `|layer.order − order_min| < 1e−6` (where `order_min` is the minimum layer order in the set). First-layer special handling (height, width, flow multiplier, no SDF surface classification) keys off this, not off world Z.
3. **"Already printed" = lower order value.** Every support-related probe in toolpath planning steps *against* the local order-field gradient (or along gravity for physical support) and compares landing-point order to the segment's own order — `probe order ≤ own order − eps` means printed/bed, `> own order + eps` means future solid (§8, §10, §12). This is what makes the pipeline work on non-planar FSM layers where "one layer down" in Z is not "one layer back" in deposition order.
4. **No global axis/apex.** `resolve_axis_apex_slope(AnisotropicFsm, config)` returns the *degenerate cone* `(BUILD_DIRECTION, DVec3::ZERO, 0.0)` — same as `Height`. That triple is only the *projection-plane* choice for 2D polygon ops (`contour::plane_basis(axis)` gives the basis, the apex the origin of in-plane coordinates); it must match what `slice_mesh_with_progress` used to produce `layer.order`, not any property of the FSM solve. Correct non-planar geometry comes from `reconstruct_on_order_field(_near)`, which *numerically* solves `field.order(apex + basis1·u + basis2·v + axis·along) == target_order` for `along` by bracket-expansion + bisection on the actual cached field (bounded by `max_along`, `layer_height × MAX_ALONG_LAYER_HEIGHTS`), so a 2D boolean op's result lands back on the true isosurface rather than on a flat plane.

### 1.4 `MeshSdf` ground truth

`manifold_fidget::mesh_sdf::MeshSdf` is a BVH-accelerated signed-distance query over the mesh's actual triangles: `sample(p) -> { value, gradient }`, `value ≤ 0` inside (the `toolpath` passes use a small positive slack, `CONTAINMENT_POINT_SLACK = 0.35` mm ≈ half a default nozzle, so points wandering a couple tenths off the surface near topology changes don't count as violations). It is the single source of truth every "is this point in the real solid?" decision uses: containment safety net (§6), bridge/wave/tangent path inclusion (§5), overhang/bridge/top-surface reclassification (§10), support fractions (§10), and travel-obstruction classification (§8). Where a layer has no `mesh_sdf` (test fixtures), every one of those probes degrades to "unknown/optimistic" rather than erroring.

---

## 2. Top-level orchestration (`plan` / `plan_with_progress`)

### 2.1 Entry points and inputs

`toolpath::plan(layers, objects, tools, config)` is a thin wrapper around `plan_with_progress(layers, objects, tools, config, machine, slope_profile, on_progress)` with `machine = None` and an empty `SlopeProfile`. In production, `lib.rs::plan_toolpaths_with_progress` is the caller: it runs `slice_workspace_with_progress` first (progress `0 → 0.5`), then `plan_with_progress` with `Some(&workspace.machine)` and the machine's slope profile (progress `0.5 → 1.0`), then two post-passes that live *outside* the toolpath module's per-path pipeline — transient-pressure compensation (`transient_pressure::apply_transient_flow_compensation`, when enabled; pressure-advance subdivision itself is deliberately left to `gcode::emit_with_machine`, the sole G-code text consumer, to avoid double-applying the extrusion boost) and the terminal wipe/clearance + bounds validation (§13).

`plan_with_progress` fails with `Error::InvalidMesh` if any layer references an object id absent from `objects`; everything else is best-effort with graded safety nets.

### 2.2 Phase 1 — setup and pre-planners

Before any layer work, the orchestrator:

1. Builds the two infill generators up front: `infill::generator_for(config.sparse_infill_pattern())` and `infill::generator_for(config.solid_infill_pattern())` — sparse and solid fill may use *different* patterns (§4).
2. Scans all wall-loop points for `bed_z` (the minimum Z in the whole print — the build plate, §1.3) and `order_min` (the minimum layer order).
3. Resolves the machine's Z-travel penalty (`config.resolved_z_travel_penalty(machine)`), used by travel ordering/routing to weigh vertical moves (§8).
4. Runs the three *surface pre-planners* concurrently via `rayon::join`, so they overlap with each other and finish before per-layer planning needs their results:

   - `wave_overhang::plan_wave_overhangs` (only when `config.wave_overhangs_enabled()`) — the LaSO Huygens wavefront planner; produces per-layer overhang *paths* **and** per-layer overhang *footprints* (2D loops), plus per-wall/per-point overhang tags (§5.1).
   - `bridge::plan_bridges` — per-layer bridge *paths* and bridge *footprints* (§5.2).
   - `tangent_surface::plan_tangent_surfaces` — per-layer tangent-surface *paths* and footprints, split into total footprints (masking) and *downward*-only footprints (unsupported-material masking) (§5.3).

Each planner's result is a small plan struct indexed by `layer.index`; per-layer planning later pulls out its slice. Note the asymmetry: tangent-surface footprints are split by orientation because only *downward-facing* tangent surfaces create unsupported voids, while all tangent surfaces mask infill.

### 2.3 Phase 2 — per-layer planning (in parallel)

`layers.par_iter().map(...)` plans each layer independently on its own rayon worker: every layer only reads the shared immutable `layers`/`objects` slices plus the pre-planner outputs, and produces its own `Vec<Path>`. The indexed `map` preserves input layer order in the output regardless of completion order. Inside the closure, one layer goes through these sub-stages:

1. **Look up** the layer's `Object` (error if missing); decide `is_layer_0`; resolve the (axis, apex) plane-basis triple and the in-plane basis (§1.3).
2. **Assemble 2D footprints** for the layer: the tangent-surface total footprint, and the *unsupported* void footprint = wave-overhang footprints ∪ bridge footprints ∪ downward tangent footprints. Both are canonicalized (`polygon2d::canonicalize`) into single multi-loops for point-in-polygon tests.
3. **Build wall paths** in `wall_print_order` (§3): for each wall loop in print order, create the `Path` (debug loops become open `DebugExcluded` polylines), classify every segment's `MoveKind` (§3), assign per-point line widths (gap-fit widths for inner walls, §5.4) and channel widths, and append that wall's gap-fill paths.
4. **Plan infill and solid fill** over `InfillRegion::from_layer`, with the region loops gated to the mesh SDF (§4.1) and the void footprints subtracted (§4).
5. **Append dedicated surface paths** — bridges, wave overhangs, tangent surfaces — each kept only if all its points are contained in the mesh SDF (§5).
6. **Drop too-short open extrusion paths** (total length < `2 × nozzle_diameter`), adjusting the wall-path count used later for travel ordering.
7. **Safety nets and corrections**: `retain_contained_paths` (§6), `compensate_flat_nozzle` (§6), `simplify_paths` (§7), and void reclassification of wall segments that landed inside an unsupported footprint (§6).
8. **Travel moves**: `optimize_travel_order` (order-aware sorting over the wall-prefix boundary), then `route_travel_moves` (collision-avoidance A* routing, layer's `order_field` + `mesh_sdf`, layer max-Z bound, slope profile, Z penalty) — both in §8.
9. **`insert_z_hops`** on travels (§9) and **`subdivide_long_traverses`** on long extrusion moves using the layer's order field (§9).
10. **Per-segment extrusion/flow finalization** (the big closure: support fractions, SDF surface classification, local layer geometry, slope cosines, bead areas, speeds, volumetric clamps, fluid swell — §10), plus `pin_outer_wall_centerline` on each path (§6).
11. **Post-passes**: corner-flow compensation, scarf joints, seam gaps, pre-retract tapers, perimeter wipes (§11); then the SDF backstop re-tags fill segments that end outside the solid as travel (§4.5); then drop negligible micro-paths and clamp all points up to the build-bed floor along `BUILD_DIRECTION`.
12. Report progress: `(completed_layers / total_layers) × 0.9` through a `Mutex`-guarded callback (the final 0.9 → 1.0 is reported after the global passes).

### 2.4 Phase 3 — global post-passes

After all layers have planned, the flattened `Vec<Vec<Path>>` goes through sequence-level passes that need the *whole* print:

1. **`defer_unsupported_paths`** — support-aware emission reordering of floating infill/top-surface paths (§12). This is the one place the layer grouping is flattened and re-sequenced.
2. **Global `route_travel_moves`** — a second routing pass over the entire sequence, using the first available layer's `order_field`/`mesh_sdf` (the fields are object-specific but shared by every layer of the same object; multi-object prints reuse the first layer's field, which is a known approximation) and no layer max-Z bound.
3. **Global `insert_z_hops`** — hop insertion that also spans travels *between* layer groups.
4. **Bed-floor clamp** of every point (again, belt-and-suspenders after the per-layer clamp).
5. **Sequential segment ids** — `Segment::id` assigned `1..N` in final emission order across the whole print (stable identifiers for the GUI's data views).
6. Report progress `1.0`.

### 2.5 Parallelism, determinism, and a stale-doc note

- Layers plan concurrently; the expensive per-layer work (polygon boolean ops in `InfillRegion::from_layer`, the infill generators) has no cross-layer dependency. Pre-planners run before per-layer work; global passes run after.
- Output order is deterministic: `rayon`'s indexed `map` preserves layer order, and `defer_unsupported_paths` is a *stable* sort on emission order, so non-deferred paths keep their exact per-layer sequence (including per-layer travel optimization).
- Progress callbacks are serialized behind a `Mutex` because completions arrive in arbitrary worker order; `plan_with_progress` documents that the fraction is per-completed-layer, in completion order.
- **Stale doc warning:** `plan`'s doc comment claims "travel *routing*/collision avoidance around already-printed geometry is not yet implemented (travel moves are still straight lines between their endpoints)". That predates `route_travel_moves` (and its global invocation); the code in this section is authoritative — travels *are* routed.

---

## 3. Wall path construction & segment classification

### 3.1 Print order: `wall_print_order`

`wall_print_order(&layer.loops, config.wall_order())` returns a permutation of the loop indices in print order:

1. **Debug loops out.** Any loop with `wall_index >= 990` is pulled out of island grouping entirely and appended at the very end — excluded fragments print last (they're visualization-only anyway, §6).
2. **Group by island.** Remaining loops are bucketed by `WallLoop::island`, preserving first-appearance order of islands within the layer (islands never interleave with each other).
3. **Order within each island** per `SlicerConfig::wall_order` (`WallOrder`):
   - `InnerOuterInner` (default) — via `inner_outer_inner_rank_table`: for `wall_count` depths, print `[n−1, n−2, …, 2, 0, 1]` (for n ≥ 3; n = 2 is just `[1, 0]`; n = 1 is `[0]`). The outer wall (0) is printed *near the end*, and the backing wall (1) *last*: the outer bead gets a moment to firm up before the second wall's heat and pressure act behind it, reducing bulging and witness lines on visible surfaces.
   - `OutsideIn` — plain ascending `wall_index` (0, 1, 2, … n−1).

The rank table is a `Vec<usize>` permutation (`rank[wall_index] = print position`) so the sort is a stable key-based `sort_by_key`.

### 3.2 One `Path` per wall loop

Each loop becomes a `Path` with `points = wall_loop.points.clone()` (world space) and `segments` of length `points.len()` for closed loops (segment `i` = move `points[i] → points[(i+1) % N]`) or `points.len() − 1` for open polylines. `Path::tool`/`Path::object` come from the layer's object. Debug loops are open `DebugExcluded` paths with `extrusion_rate = 0.0` and zero extrusion length.

### 3.3 Segment classification (destination-point semantics)

Classification is per segment but reads the *destination* point's flags — the segment's kind describes the material being laid down as it arrives at `points[dest]` (open paths: `dest = i + 1`; closed: `dest = (i + 1) % N`). The decision cascade, in priority order:

1. **Debug loop** → `MoveKind::DebugExcluded` (whole path).
2. **Unsupported** → `MoveKind::Overhang` if *any* of:
   - the segment's 2D midpoint (projected to the layer plane basis, §1.3) lies inside the canonicalized *unsupported void footprint* (wave-overhang ∪ bridge ∪ downward-tangent footprints, §2.2) **and** the SDF probe one `layer_height` below the 3D midpoint is *not* in solid (`value > 0`);
   - `wall_loop.unsupported[dest]` — a point inserted by inter-layer wall-gap stitching (§1.2);
   - the wave-overhang planner tagged this exact (layer, wall, point) as overhang (`wave_overhang_plan.wall_overhang_tags_by_layer`).
3. **`wall_loop.top_surface[dest]`** → `MoveKind::TopSurface` (roof material before open air; §1.2).
4. **Fallback** → `MoveKind::WallOuter` for `wall_index == 0`, else `MoveKind::WallInner`.

So a genuine overhang always beats the top-surface label, and both beat the plain wall kind. (This first pass can be *overruled later*: §6 reclassifies wall segments that land inside a void footprint after simplification, and §10's SDF geometric classification can upgrade remaining segments to `TopSurface`/`Bridge`/`Overhang`.)

### 3.4 Per-segment metadata at construction

At this stage each segment carries:

- `speed = speed_for_kind(kind, config)` — `config.motion_model().max_feedrate(kind, false)`; a nominal value that §10 refines per segment (direction-dependent feedrates, volumetric clamps, overhang/bridge speeds).
- `extrusion_rate = 1.0`, `support_fraction = 0.0` (both finalized in §10), `order = layer.order` (the FSM order value that produced this layer — the per-segment stamp §1.3 keeps meaningful on non-planar layers).
- `extrusion_length = 0.0`, `is_scarf = false`, `id = 0` (ids assigned globally in §2.4).
- `line_width` — for *inner* walls (`wall_index > 0`) that are neither unsupported nor debug, the gap-fit width from the gap-fill pass (§5.4) at the destination point when available; otherwise the loop's `line_widths[dest]` (dynamic width from slicing, §1.2); otherwise `config.wall_line_width`.
- `channel_width` — copied from `wall_loop.channel_width[dest]` (§1.2) for the §10 bead-clearance clamp.
- `island` — copied from the loop.
- `flow_breakdown = None` — populated only by the §10 finalization pass.

After the wall path is pushed, that wall's gap-fill paths are appended to the layer's path list (§5.4), keeping the `wall_path_count` (the index boundary of wall paths within the layer's `Vec<Path>`) accurate for `optimize_travel_order`'s wall-prefix handling (§8.1).


---

## 4. Infill & solid-fill planning

Infill is **two generator passes** over two regions of the layer, and both are tagged `MoveKind::Infill` — there is no separate "solid fill" kind. (Sparse fill uses `config.sparse_infill_pattern`, solid fill uses `config.solid_infill_pattern`; they may be different patterns.)

### 4.1 The two regions

- **Sparse region** — `InfillRegion::from_layer(layer, config)`: the layer's `infill_boundary` minus `solid_fill_boundary`. Computed by 2D boolean difference in the layer-plane basis (canonicalized, §1.3), with components smaller than `0.25 × nozzle_diameter²` filtered out (`filter_min_area`), then re-projected onto the true isosurface via `reconstruct_on_order_field_near` (reference loops = the original infill + solid boundary loops; `max_along = order_field::max_along_for(config)` = `layer_height × 50`). Empty when the layer's fillable area is entirely solid.
- **Solid region** — `all_solid_loops = layer.solid_fill_boundary ∪ narrow_solid_loops`, where `narrow_solid_loops` are the sparse region's loops whose 3D bounding-box diagonal (`(max − min).length()` over the loop's points) is **< `15 × nozzle_diameter`**: regions too small to earn a sparse pattern get printed solid instead (a scanline pattern across a few nozzles of material is just a couple of lines with air gaps; solid guarantees a dense floor).

The sparse region is generated at `config.infill_density`; the solid region at `density = 1.0` (the generator's density argument is exactly what distinguishes them — `InfillGenerator::generate(region, config, layer, object_transform, density)`).

Both region sets pass through the **SDF containment gate** before the generators consume them (`infill::gate_sparse_loops` / `infill::gate_skin_loops`): every region-boundary point must satisfy `SDF ≤ −(wall_offset + wall_line_width) + 0.1` for the sparse loops — at least one full wall line inside the outer surface, the same depth relationship the walls themselves have — and `SDF ≤ +0.1` for the solid-skin loops (no inset: skin material belongs on the surface). The gate matters because the region pipeline is 2D on a 3D surface: flattening the layer's order-field isosurface onto the layer plane folds where the isosurface is non-monotonic (curved tops, voids — the same XY column crossing the isosurface at several heights), the 2D boolean then leaks points outside the subject polygon, and the re-lift (`reconstruct_on_order_field_near`) seeds them from the nearest XY reference, which can be a different branch or a point in air. The clipping (`infill::clip_loops_to_sdf`) removes failing points and re-closes the loop locally: for a maximal removed run between retained neighbours `a` and `b`, the chord midpoint `m = (a+b)/2` is inserted iff `SDF(m)` passes, else `a→b` is taken directly; a loop whose whole boundary fails — or which shrinks to fewer than 3 points — is dropped, and a non-finite SDF sample counts as failing. It is a *monotone shrink* (the region only loses area, toward the safe side), is re-applied after the §4.2 footprint-mask reconstruction (which re-runs the re-lift and can reintroduce off-surface points), and is a no-op when `layer.mesh_sdf` is `None`.

### 4.2 Footprint masking before generation

Before either pass generates paths, the *unsupported void footprint* (wave-overhang ∪ bridge ∪ downward-tangent, §2.2) is subtracted:

- **Sparse loops**: 2D difference against the canonicalized footprint → `densify_loops` at `nozzle_diameter` (so boolean edges get enough vertices) → `reconstruct_on_order_field_near` back onto the layer's isosurface using the *original* loops as references.
- **Solid loops**: the same subtraction, **except layers with `index < bottom_layers`** — the bottom solid layers are the part's base floors and must stay 100% solid; they are never subtracted by overhang footprints. (The bottom-band solid fill is computed in slicing via `compute_solid_fill_boundaries`.)

### 4.3 The generators

`infill::generator_for(kind)` resolves an `InfillPatternKind` to a `Box<dyn InfillGenerator>` — new patterns add a variant + an impl, and `toolpath::plan` never changes. The built-ins:

- **`Monotonic`** (default) — boustrophedon scanlines: scan direction in a rotated (u, v) frame at `object in-plane rotation ± infill_angle_deg` (sign alternates by `layer.index` parity, so the angle tracks the object's orientation, not world space). Scan-line spacing is `infill_line_width / density` (density 1.0 packs at bead width). Crossings of each scan line against the region loops use the even-odd rule (holes/islands need no pre-classification), and — the key non-planar detail — **each crossing's world position is re-solved against the layer's order field** (`refine_point_onto_order_field`, accepting moves within `|residual|·4 + 2·layer_height` of the linearly-interpolated seed, else falling back to the seed). Linear interpolation is only exact for `HeightOrderField`; for the FSM field the true position along a short edge near curved/threaded geometry can differ sharply. Spans on consecutive scan lines are union-find-grouped when they overlap within a generous margin (`max(8 × spacing, 8 × infill_line_width)`), and each group becomes one open `Path`: spans are ordered by scan index, traversal direction alternates, adjacent turnarounds (endpoints within `max(2.5 × spacing, 2.5 × infill_line_width)`) are joined by an *extruding* `Infill` segment (the serpentine turnaround bridge), and far jumps split the path so the travel optimizer (§8) can sequence them instead of printing across voids.
- **`Concentric`** — successive inward offsets of the region boundary, each offset printed as its own closed `Path` ring, spaced `infill_line_width` apart (widened by `density` the same way scan spacing is). No travel between rings — inter-ring travel is emitted by the G-code stage. A `MAX_OFFSET_RINGS` (10 000) cap guards against near-zero spacing.
- **`AllWalls`** — like `Concentric` but spacing is *always* exactly `infill_line_width`; `density` is ignored, so the region is always fully filled. This is the "fill everything with perimeter-style loops" option.
- **`Cubic`** — 3D periodic cubic truss: a self-supporting cubic grid rotated so its space diagonals align with the Cartesian axes (45° to the build plate), an isotropic 3D lattice rather than per-layer 2D lines.
- **`Gyroid` / `SchwarzD` / `SchwarzP`** — TPMS minimal surfaces (Gyroid, Schwarz Diamond, Schwarz Primitive). A `TpmsField` with wavelength derived from `infill_line_width` and `density` (`wavelength_for_density`); the surface is clipped per island (outers vs holes via signed area) so paths never jump across voids or between islands, and is evaluated along the non-planar order-field surface — the TPMS lattice follows the layer isosurface rather than a flat plane.
- **`None`** — no sparse infill at all; walls and solid-fill regions still print.

All generators emit open `Path`s (Monotonic) or closed ring `Path`s (Concentric/AllWalls/TPMS-derived) with segments tagged `MoveKind::Infill`, nominal `speed_for_kind(Infill, config)`, `line_width = config.infill_line_width`, `channel_width = ∞`, and `order = layer.order`. `plan` stamps `path.tool = object.tool` on every generated path before pushing it into the layer's list.

### 4.4 Post-generation filter

After both passes, *open extrusion paths* whose total geometric length is under `2 × nozzle_diameter` are dropped from the layer (they can't lay a bead). This filter tracks which dropped paths were wall paths (by index vs `wall_path_count`) so the wall-prefix boundary for `optimize_travel_order` stays exact (§8.1).

### 4.5 SDF backstop for fill extrusion

The region gate (§4.1) constrains the region *boundaries*; the generators still re-lift interior pattern points onto the layer's isosurface (§4.3), and on non-monotonic order fields that re-lift can leave dangling chords in air (verified on TestObj1: up to +2.86 mm on fully gated layers). As a final per-layer pass — after the wipe, before the micro-path filter — every `Infill` or `TopSurface` segment whose destination point has `SDF > 0.5 × nozzle_diameter` is re-tagged `MoveKind::Travel` with zero extrusion length: the path topology is preserved, the dangling chord simply travels without extruding, and the micro-path filter drops any stub that results. Bridge/Overhang segments are exempt — a bridge spans a void by construction, so positive SDF at its midpoints is expected. This is the §6.1 "never print infill in open air" policy at segment granularity: §6.1 drops a whole path on gross violation, the backstop keeps the valid part of a mostly-good path instead of dropping it.


---

## 5. Dedicated surface passes

Four specialized planners handle geometry that the wall/infill pipeline can't express: unsupported overhang spans, straight bridges, tangent (near-parallel-to-build-direction) surfaces, and the gaps between inner walls that open up when the layer isosurface is non-planar. All run as pre-planners (§2.2), before per-layer planning.

### 5.1 Wave overhangs (LaSO) — `wave_overhang`

The support-free overhang engine. Two regimes in `plan_wave_overhangs`:

- **Surface-guided (objects available — the production path).** Solves a **geodesic arrival-time field `T(v)` on the mesh's own 2-manifold surface** via Kimmel–Sethian Fast Marching (`manifold_fidget::surface_eikonal::solve_surface_eikonal`), seeded from bed-contact vertices (`p.z ≤ seed_tol`, `seed_tol = max(min.z + 0.20, 0.1)` — near-bed within 0.20 mm for a part sitting on a z=0 plate). Downward-facing triangles (`normal.z / |normal| < −0.15`, excluding triangles whose vertices are all bed-contact) are the overhang region; wavefront isocontours are extracted **directly on the overhang faces** at increments of `wavelength = (nozzle_diameter − wave_overhang_overlap).max(0.10)` using 3D triangle-edge interpolation (`contour::extract_order_contours_on_mesh_with_debug`). Every wave bead is therefore physically adjacent to its supporting predecessor and follows the true mesh surface — no air-gap jumps, no disconnected mid-air loops. Each contour is assigned to the layer whose `order` is closest to the polygon's mean order (sampled with the first layer's cached order field), its direction alternates per contour, and it becomes an open `MoveKind::Overhang` path with `line_width = nozzle_diameter` and `speed = wave_overhang_speed()`. Polylines shorter than `0.75 × nozzle_diameter` are dropped.
- **2D fallback (no objects).** Per layer, the unsupported region is the 2D difference `outer(k) \ outer(k−1)` (minimum area `0.25 d²`), wave-filled by `generate_wave_overhang_paths_2d` — a 2D Fast Marching solve on a grid of step `wavelength/4` (clamped `[0.04, 0.15]` mm, capped at 2000×2000 cells) seeded from the contact segments with the previous layer's boundary, with the standard anisotropic FMM update (the `√(2h² − (uₓ−u_y)²)` correction term for corner propagation) and wavefront isocontours at `k × wavelength` extracted by marching squares and stitched into polylines.

Both regimes also produce **wall overhang tags** — a per-layer/per-wall/per-point `Vec<bool>` marking points whose support probe fails (SDF value > 0 one `layer_height` below the point; in the 2D fallback, 2D containment against the previous layer's boundary within `0.6 × nozzle_diameter` with a 3D distance fallback). These tags feed the §3.3 classification cascade directly.

`WaveOverhangPlan { paths_by_layer, wall_overhang_tags_by_layer, overhang_footprints_by_layer }` — the footprints (2D overhang shapes in the layer-plane basis) feed the void-footprint masking (§3.3, §4.2). In the surface-guided regime the footprints are empty: the surface wave paths already cover the geometry, and wall tagging is SDF-based rather than footprint-based.

### 5.2 Bridges — `bridge`

A bridge is an extrusion **across empty space that contacts preexisting material at both ends**; it must be a *straight line* (no turning or curling mid-air), with dedicated bridge feedrate and line width. `plan_bridges` works per layer (in parallel): the unsupported region is `outer(k) \ outer(prev_k)` in the 2D basis (minimum area `0.25 d²`); each resulting shape's boundary is scanned for **contact runs** — maximal runs of edges whose midpoints lie within `search_dist = max(1.25 × nozzle_diameter, 0.4)` of the *previous* layer's outer boundary. `generate_straight_bridge_paths_2d` then emits straight paths between the anchor points of each contact run (one `MoveKind::Bridge` path per run), with `speed = bridge_speed()` and `line_width = max(infill_line_width, 0.1)`. The 2D shapes also become `bridge_footprints_by_layer`, which join the void footprint for masking (§4.2) and segment classification (§3.3). The adjacency direction (`k−1` vs `k+1`) follows `slicing::layer_z_increases(layers)` — which physical neighbor is "beneath" depends on the order-field walk direction.

### 5.3 Tangent surfaces — `tangent_surface`

Tangent surfaces are regions where an inner wall or infill of an *adjacent* isosurface would be exposed to air — the classic near-vertical/steep surfaces that the FSM's `fsm_top_tangency_aspect` deliberately steers the order field into near-tangency with. `plan_tangent_surfaces` categorizes them by orientation relative to build progression:

- **`Downward`** — air *below* (the layer beneath stepped inward or doesn't exist; region = `outer(k) \ outer(prev_k)`): printed with wave-overhang settings (`MoveKind::Overhang`, `wave_overhang_speed`); its footprints enter the *unsupported* void footprint (§2.2, §4.2).
- **`Upward`** — air *above* (the layer above steps inward; region = `outer(k) \ outer(next_k)`): printed with outer-wall settings (`MoveKind::WallOuter`); its footprints mask infill but do *not* create unsupported voids.

Each region (minimum area `0.25 d²`) is wave-filled with `generate_wave_overhang_paths_2d` (same 2D FMM as the wave-overhang fallback, seeded from the contact with the adjacent layer's boundary) — "complete wave fill with longest-segment midpoint seeding." `TangentSurfacePlan` carries both the total footprints (`footprints_by_layer`, used for gap-fill suppression and infill masking) and the split `downward_`/`upward_footprints_by_layer`. Tangent-surface points are excluded from gap-fill (§5.4) because the wave fill already occupies that material.

### 5.4 Gap fill — `gap_fill`

When the layer isosurface is non-planar, consecutive wall isosurfaces (and the innermost wall vs the infill boundary) can diverge by more than the wall bead's width — a *gap* of air opens between walls where a flat slice would have packed them together. `plan_gap_fill_for_wall(wall_loop, layer, ctx)` closes these per point:

1. **Measure the gap.** For each wall point (skipping points inside the canonicalized tangent footprint — those already have wave fill), compute the CAD surface normal `n_cad` (finite-difference SDF gradient, `eps = 0.02`) and the order-field gradient `n_order` from `layer.order_field`. The isosurface separation that a bead of `wall_line_width` needs to span is `Δs = wall_line_width / |n_cad × n_order|`. Note the FSM-specific degeneracies: near-tangency (`|cross| → 0`, the regime `fsm_top_tangency_aspect` produces) makes `Δs → ∞` — exactly where gap fill is needed most — so it clamps to `max_w + 1` to force the gap-fill branch; *antiparallel* normals (`n_cad·n_order < −0.9`, e.g. every bottom-layer point on a flat-bottomed model) are ordinary bed contact, not tangency, and skip gap fill with the nominal width.
2. **Decide.** If `Δs ≤ max_bead_width`, the wall's own bead spans the gap; the point's wall width is simply updated to `Δs` clamped to `[min_bead_width, max_bead_width]`. If `Δs > max_bead_width`, a separate gap-fill pass is needed: *outward* between this wall and wall `w−1` (for `wall_index > 0`), and *inward* between the innermost wall and the infill boundary (for the island's innermost wall).
3. **Locate the gap bead.** `compute_gap_point` steps from the wall point by `0.5 · Δs` along `u = n_cad − (n_cad·n_order)·n_order` (the component of the CAD normal orthogonal to the order gradient — the direction that stays on the surface band between the two isosurfaces), capped at `1.5 × wall_line_width`, then refines the point onto the layer's order-field isosurface (`refine_point_onto_order_field`) and verifies both the point *and* the chord to the wall point are inside or on the solid (SDF ≤ 0.05). Its line width is `0.5 · Δs` clamped to `[min_bead_width, max_bead_width]`.
4. **Chain into paths.** `extract_gap_chains` groups consecutive qualifying points into chains (closed when the whole loop qualifies); chains with any edge longer than `3 × wall_line_width` (a void jump) or shorter than `0.75 × nozzle_diameter` in total are discarded. Chains become `MoveKind::WallInner` paths with per-point gap line widths.

The return value `(updated_line_widths, gap_fill_paths)` is consumed in §3: the updated widths replace the inner wall's per-point `line_widths` (and halve to `0.5·Δs` where a gap bead also prints there), and the gap paths are appended right after their wall's path.

### 5.5 Integration

All four passes feed per-layer planning through small per-layer slices of their plan structs: dedicated paths are appended to the layer's path list **only if every point is contained in the mesh SDF** (`sample(p).value ≤ CONTAINMENT_POINT_SLACK`, §1.4) — the same graded-containment philosophy as §6; their footprints enter the void-footprint union (§2.2) for masking and classification; and the wall overhang tags enter the §3.3 cascade.


---

## 6. Containment, safety nets & geometric corrections

Between generation (§3–§5) and travel/extrusion finalization (§8–§10), each layer's path set is checked against the real solid and geometrically corrected. These passes run per layer, in this order: `retain_contained_paths` → `compensate_flat_nozzle` → `simplify_paths` (§7) → void reclassification → SDF backstop for fill extrusion (§4.5) → (later) micro-path filter and bed-floor clamp (§2.3).

### 6.1 Mesh-SDF containment safety net — `retain_contained_paths`

Wall/infill loops derive from contour extraction and 2D polygon booleans on `infill_boundary`/`solid_fill_boundary`, which have (rarely) produced loops that don't correspond to real material — infill inside a hole that isn't part of the object, or small fragment loops shattered off near level-set topology changes (a side hole meeting a bore). Rather than prevent every such source, `plan` re-validates every path against `layer.mesh_sdf` and grades the violation:

- Per point, sample the SDF and tally: points with `d > gross_tolerance` (`nozzle_diameter` — more than a whole bead hanging in air) and points with `d > CONTAINMENT_POINT_SLACK` (`0.35` mm ≈ half a default nozzle; wall points near topology changes legitimately wander a couple tenths of a millimetre off the exact surface, and inter-layer stitch points are deliberately allowed up to a bead radius outside).
- A path is **contained** when `gross_outside_fraction ≤ 0.10` *and* `outside_fraction ≤ CONTAINMENT_OUTSIDE_FRACTION` (`0.25`). A genuine wall loop has thousands of points with a handful of outliers; a spurious fragment is small and mostly outside. The threshold is deliberately low — a fragment loop anchored at both ends to real surface can otherwise float mostly in open air near an arch while still averaging under a lenient fraction.
- **Contained**: kept. **Not contained, pure `Infill` path**: dropped entirely — "never print infill in open air". **Not contained, any other kind**: the path is kept but every segment is retagged `MoveKind::DebugExcluded` (still rendered for inspection, never emitted to G-code), with a `tracing::warn!` summary. A partially-valid path is dropped/retagged *wholesale* rather than clipped: splitting would risk a spurious partial loop that's arguably worse than omitting the already-wrong path.
- No-op when `mesh_sdf` is `None` (synthetic/test layers): containment is treated as *unknown*, not as failure.
- The §4.5 SDF backstop complements this at segment granularity: for `Infill`/`TopSurface` segments whose destination point sits more than `0.5 × nozzle_diameter` outside the solid, the segment is re-tagged `MoveKind::Travel` (no extrusion) rather than dropping the whole path — the dangling-chord counterpart of this safety net, needed because the region pipeline's 2D re-lift (§4.1) can place interior chords in air even on gated boundaries.

### 6.2 Flat-nozzle slope clearance — `compensate_flat_nozzle`

Skipped entirely in `SlopeCompensationMode::VolumetricModulation` (flow-only mode; §10.4). In `GeometricOffset` mode, **outer-wall loops only** (`WallOuter` first segment, ≥ 3 points) get their points lifted along the nozzle axis to clear a sloped surface. A flat nozzle tip of land radius `flat_radius` (per tool, or `config.nozzle_flat_diameter()/2`) tilts by angle `α` relative to the local surface normal on a slope; instead of shifting the centerline laterally in X/Y (which introduces dimensional asymmetry and wall bulging), each point is elevated by:

- `z_slope_clearance = flat_radius · sin α · (1 − cos α)` — the drop of the tip's lowest outer edge when the land tilts by `α` (normal from the numeric order-field gradient, `α` against `NOZZLE_DIRECTION`); plus
- `z_concave_clearance` — a transverse concave (V-groove/valley) term: sample normals at `p ± u_perp·flat_radius` (where `u_perp = tangent × normal`), the difference gives the transverse flank rise `flank_rise = flat_radius · sin(β)` (β = transverse inclination), adding `flank_rise · (1 − cos β)` capped at `0.60 × layer_height`.

The lifted point is re-projected so it can't fall below `min_extrusion_z = 0.5 × first_layer_height` (or below the build bed at 0) along `BUILD_DIRECTION`.

### 6.3 Void reclassification after simplification

Simplification (§7) can move points off their original isosurface samples, so *after* `simplify_paths` the wall segments are re-tested: for `WallOuter`/`WallInner` segments whose 2D midpoint (or either endpoint) lies inside the canonicalized unsupported void footprint and whose SDF probe one `layer_height` below is not in solid, the segment is reclassified `MoveKind::Overhang` with `speed_for_kind(Overhang, config)`. This keeps the §3.3 classification honest after the geometry has changed.

### 6.4 Micro-path filter and bed-floor clamp

- **Micro paths**: after the seam/wipe post-passes (§11), any path whose total extruding length is < `0.5 × nozzle_diameter` **or** whose total extruded filament < `0.0005` mm is dropped (paths with < 2 points as well). These can't lay a bead and only add start/stop pressure artifacts.
- **Bed floor**: every point with `p·BUILD_DIRECTION < 0` is shifted up to 0 — nothing may dip below the build bed. Applied per layer *and* globally after the sequence-level passes (§2.4).

### 6.5 Outer-wall centerline pinning — `pin_outer_wall_centerline`

Variable line widths (gap-fit widths §5.4, dynamic slicing widths §1.2) change the extruded bead's width along a wall. To keep the bead's **exterior boundary pinned to the CAD surface** across that variation, outer-wall paths get their centerline displaced by

`p_pinned = p − û · (w_eff − w_nominal) / 2`

where `û` is the unit CAD surface normal (`mesh_sdf` gradient) *projected onto the layer order surface* (`û = normalize(n_cad − (n_cad·n_order) n_order)` — the in-surface component), and the shift is clamped to `±0.5 × wall_line_width`. Points with nominal width are untouched. This runs per path just before the §10 extrusion finalization loop.


---

## 7. Path simplification

`simplify_paths(paths, config)` applies Ramer–Douglas–Peucker to **wall loops only** (a path qualifies when its first segment is `WallOuter`/`WallInner`; infill, bridge, overhang, and other paths pass through untouched), gated by `config.path_simplify_enabled` with tolerance `config.path_simplify_tolerance` (mm). Degenerate inputs (< 3 points, or `tolerance ≤ 0`) are returned unchanged.

The classic RDP algorithm is defined on open polylines, so the two variants differ only in how the path is fed to it:

- **Open paths** (`segments.len() == points.len() − 1`): RDP over the single chain `0..n`, always keeping the first and last point.
- **Closed loops** (`segments.len() == points.len()`): the loop is split into two open chains at its **two most mutually distant points** (`farthest_pair`, a plain O(n²) scan — noted in code as an optimization candidate for extremely dense loops). Each chain is RDP'd independently against a shared `keep` mask (both split points always kept), and the surviving points are rejoined into one closed loop, rotated so the lowest surviving index leads.

Implementation notes:

- `rdp_mark` is the standard recursive formulation: find the interior point of maximum perpendicular distance from the chord through the chain endpoints; if it exceeds `tolerance`, keep it and recurse on both halves, else drop the whole interior. The distance used (`perpendicular_distance`) projects onto the segment with `t` clamped to `[0, 1]` (a point-to-*segment* distance; the doc comment mentions the classic line variant, but the clamped implementation is what runs), falling back to plain point distance when the chord is (near-)zero.
- **Kept segments are preserved verbatim** — a kept point keeps its own original outgoing `Segment` (kind, speed, line width, …) with no interpolation or averaging across dropped points. Since §10's extrusion finalization runs *after* simplification on the final geometry, per-segment flow values always match the emitted polyline.
- The `points`/`segments` parallel-array invariant is preserved exactly: open paths end with `segments.len() == points.len() − 1`, closed loops with equal lengths, and the closing edge of a closed loop is `segments[i]` describing `points[i] → points[0]`.

Simplification runs *before* travel optimization/routing (§8) and Z-hops (§9), so those passes operate on the simplified geometry; and it runs *after* `compensate_flat_nozzle` (§6.2), so lifted wall points are the ones that get thinned. The post-simplification void reclassification (§6.3) exists precisely because this pass can move points off their original isosurface samples.


---

## 8. Travel-move ordering & collision-avoidance routing

Travel moves (no extrusion between paths) are handled by two passes, both per layer (§2.3) and both re-run globally on the flattened sequence (§2.4): `optimize_travel_order` decides *which path to print next* and *which end to enter it from*; `route_travel_moves` decides *how to fly there* without clipping already-printed material.

### 8.1 `optimize_travel_order` — order-aware move sorting

Gated by `config.travel_order_optimization_enabled` (no-op when disabled). The first `fixed_prefix_len` paths — `plan` passes `wall_path_count`, so **at least 1 and all wall paths** — are frozen: neither reordered nor reversed, because wall print order (§3.1) is a deliberate print-quality choice that a greedy search would happily undo chasing shorter travels. Every remaining path is then appended by a **greedy nearest-neighbor** scan (O(n²) per step — fine for the tens-to-hundreds of paths in a layer; a TSP solve would be overkill):

- The current position starts at the last point of the last wall path. Each step picks the remaining path whose *entry point* is closest, using a **kinematic cost**: 3D Euclidean distance with the ΔZ term scaled by `z_scale = max(z_travel_penalty, 1.0)` — the machine's Z-travel penalty (`config.resolved_z_travel_penalty(machine)`), which encodes how expensive vertical moves are for the stepper dynamics.
- **Open paths may be entered reversed** (from their last point, exiting at their first) when that orientation is closer: `reverse_open_path` reverses `points` and `segments`, which is exactly self-inverse for the parallel-array convention (segment `i` becomes the same edge walked backward, at index `len−2−i`; every segment's kind/speed/extrusion metadata is preserved). **Closed loops are never reversed or start-rotated**: `points[0]` is meaningful (the inter-layer wall-gap stitching / arc-length-correspondence passes index into it).

### 8.2 `route_travel_moves` — A* routing around printed solid

Gated by `config.travel_collision_avoidance_enabled`; requires ≥ 2 paths and a `mesh_sdf`. Derived constants: `clearance = 2 × max(wall_line_width, nozzle_diameter)`; `endpoint_clearance = config.resolved_z_hop_height()`; `cell_size = min(layer_height, nozzle_diameter)/2`.

For every consecutive path pair (last point of path `i` → first point of path `i+1`, parallel across pairs), the pass captures each path's exit/entry direction (the last/first edge, for tangent departure) and its `order`, then asks whether the straight chord is blocked (§8.3). If so, `route_around_obstruction` (§8.4) computes waypoints; a **detour is inserted as its own `Path`** — a pure `MoveKind::Travel` path (speed `speed_for_kind(Travel, config)`, zero extrusion) inheriting tool/object/order — spliced between the two extrusion paths. The per-layer call bounds the search with `max_layer_z` (the layer's max Z, falling back to `layer.order`); the global call passes `None`.

### 8.3 `travel_chord_is_blocked` — temporal already-printed-solid classification

The "what is in the way" oracle. Fast-exit: if both endpoints sit above `z_ceiling = max(max_layer_z, a.z, b.z)` the chord can't hit anything printed (nothing printed ever exceeds the layer's physical ceiling). Otherwise the chord is sampled at 4–64 points (step `max(clearance/2, 0.1)`) and a sample blocks the chord if **any** of three checks fires — all gated by the *temporal* classification that makes this FSM-aware: a point counts as **printed** when `order_field.order(p) ≤ current_order + 1e-4` (lower order value = already deposited, §1.3); with no order field, every sample is treated as potentially printed:

1. **Non-planar convexity check.** If the surface crowns upward between `a` and `b`, the chord cuts under the crown into earlier material: blocked when the sample is inside solid (`SDF < −1e-4`) *and* `order(p) < lerp(order_a, order_b, t) − 0.05` (a 50-µm convex-depth tolerance).
2. **Air-gap re-entry.** If the chord has already passed through open air (`SDF > clearance/2`) and now re-enters solid (`SDF < −1e-4`), it is crossing a void into a separate feature/island: blocked.
3. **Lateral clearance.** Away from the chord's endpoints the nozzle needs full clearance from previously printed walls: blocked when `SDF < required_clearance − 1e-4`, where `required_clearance = min(clearance, dist_from_start, dist_from_end)` — *tapered near the endpoints*, because within `clearance` of a start/end the nozzle is moving along the current layer's surface (e.g. wall transitions), not into free space. This raw-SDF check is skipped for a sample when the order field already vouches for it as legitimately free space at or above the chord's interpolated order (so a chord passing near an *unrelated* side feature doesn't override the order-aware verdict), and when the local surface points mostly upward (`normal.z > 0.70`) at the layer ceiling — traveling along the top surface is not a side-wall collision.

### 8.4 `route_around_obstruction` — three tiers

1. **Tangent endpoint waypoints.** `tangent_endpoint_waypoint` departs/arrives *shallowly* along the local isosurface tangent plane instead of lifting straight up along the surface normal: the departure step is `safe_dir · (2·clear_dist) + n_cad · clear_dist` (≈26° inclination from the tangent plane over `2×` the clearance distance, reaching `clear_dist` of open air), where `safe_dir` is the exit direction projected off the isosurface normal (and off the CAD normal when it points outward from a wall). The idea: a steep normal lift puts the molten meniscus in tension; a shallow tangent departure shears it cleanly in shear. The candidate is validated (SDF must clear by `0.5·clear_dist`, falling back to a direct outward step), and Z is clamped to `min_travel_z = max(min(a.z, b.z), 0.5·first_layer_height)`. If the chord between the two tangent waypoints is unblocked, that 4-point path (start → start_clear → end_clear → end) is the whole route.
2. **Tier 1 — planar XY A* (`route_planar_xy_detour`).** A 2D A* search on a horizontal grid at a single fixed Z (`max(start.z, end.z, min_travel_z)` — *zero vertical excursion*): the grid spans the chord's bounding box plus margin (`max(0.75×chord, 6×clearance, 10 mm)`), with cell size `max(clearance/2, 0.4)` grown by 1.25× until the node count fits `MAX_TRAVEL_GRID_NODES = 8000`. 8-connected; a cell is passable when its SDF ≥ clearance, it sits above the Z ceiling, or — the key non-planar clause — the **order field says it is future (unprinted) space** (`order > current_order`): unprinted material is not an obstacle yet. Clearances are memoized per cell; the heuristic is planar Euclidean distance; the reconstructed path is deduplicated and run through `shortcut_waypoints` (line-of-sight collapsing: from the current waypoint, jump to the *furthest* waypoint whose chord is unblocked — collapsing grid staircase steps into clean straights around obstacle corners). A detour is rejected if its total length exceeds `max(3×chord, 40 mm)`.
3. **Tier 2 — trapezoidal flyover (`route_single_flyover`).** When no planar detour exists (or is too long): sample the obstacle's height profile along the direct chord (8–64 samples at `clearance/2` spacing); at each sample that is solid *and* printed (`SDF < clearance`, `order ≤ current`), march straight up the SDF (step `clamp(−SDF, 0.2, 1.0)`, capped at `p.z + 5` and the Z ceiling) to find the minimum clearance Z, and record `(t, z)`. The **upper convex hull** of that profile (Andrew's monotone chain) becomes the flyover polyline — simultaneous XY+Z ramping moves, no stationary vertical stalls. Every hull chord is re-verified with `travel_chord_is_blocked`; up to three repair rounds lift the intermediate points by `0.5×clearance`, and as a last resort the router tries a single elevated horizontal cruise at `fly_z = z_ceiling + clearance` (lift → cruise → drop). If everything fails it returns `None` and the original straight chord is kept.

### 8.5 What routing does *not* do

Routing never resequences paths (that's §8.1) and never modifies extrusion geometry. It only inserts pure-Travel detour paths between existing paths; a failed route leaves the chord straight. The global re-run (§2.4) uses the first layer's `order_field`/`mesh_sdf` for every object's travels — a known approximation for multi-object prints with different fields.


---

## 9. Z-hops & long-traverse subdivision

### 9.1 `insert_z_hops` — lift/travel/lower on travels

Enabled by `config.z_hop_enabled` (no-op, not even reallocated, when the resolved hop height is ≤ 0 — the default). Runs per layer and again globally (§2.4), on every path, in parallel. For each **maximal run of consecutive `MoveKind::Travel` segments** inside a path it inserts:

- a **lift point** immediately after the run's departure point (same XY, `Z + hop_height`);
- every original travel point *strictly inside* the run, raised by the same `Z + hop_height` — so lateral travel happens entirely at hop height, not just at the endpoints;
- a **drop point** immediately before the run's arrival point (arrival's XY, still at hop height), followed by the unmodified arrival point (its real Z) that lowers back down.

All inserted points/segments are tagged `Travel` with `extrusion_length: 0.0`, inheriting the remaining segment metadata from the run's segments — `gcode::emit` needs no special-casing. Two runs are deliberately **not** hopped:

- runs **entirely surrounded by `Infill` segments** (the arriving and leaving moves are both infill) — that's an internal jump within the same infill patch, where lifting would add a stall inside material the next scanline lands on; runs at a path's end (no bounding segment on one side) are conservatively treated as hop-worthy.
- runs whose total travel distance is ≤ `min_travel_for_hop = config.effective_min_travel_for_retract()` — too short to clear anything.

Paths consisting *entirely* of Travel segments — the detour paths inserted by `route_travel_moves` (§8.2) — are skipped whole: their tangent-departure/arrival waypoints already carry the required clearance height, and stacking a pure vertical hop on top would be redundant. The closing edge of closed loops is never modified, preserving whichever parallel-array shape the path had.

### 9.2 `subdivide_long_traverses` — order-field fidelity on long chords

Long extrusion traverses (`MoveKind::Infill | TopSurface | Bridge` segments > `MAX_SEG_LEN = 2.5 mm` — the threshold that matters across arched structures and folds) are subdivided when the **local layer geometry changes along the chord**: `extrusion::local_layer_geometry(field, p, h_nominal)` is evaluated at both endpoints, and the segment is split when `|h_start − h_end| > 0.02` mm, when the surface normals' vertical components differ by > `0.05`, or when the chord is > `6 mm` unconditionally. It is split into `ceil(dist / 2.5)` sub-segments clamped to `[2, 16]`, with points placed at linear interpolations along the original chord and the segment's metadata duplicated onto each sub-segment (the §10 finalization pass then computes a *separate* extrusion length for each).

The purpose is exactly the non-planar one: a long chord over an arch or a fold must sample the local layer gap (the ray-marched physical gap of §10.3 — *not* the `h/||∇φ||` gradient-inversion shortcut, which is uncalibrated in the FSM's boundary-metric regions) and surface normal per sub-segment rather than evaluating extrusion from the two distant endpoints alone. Under a flat `Height` field the gap and the normal are nearly constant, so this pass fires almost never; under the FSM field it is what keeps infill flow correct across steep, order-compressed regions.


---

## 10. Per-segment extrusion & flow computation

After travel routing, Z-hops, and traverse subdivision, each layer's paths get their per-segment metadata finalized in one loop (the "big closure" of §2.3 step 10). `pin_outer_wall_centerline` (§6.5) runs per path first; then, per segment: `Travel` segments get `extrusion_length = 0` and skip the rest; every extruding segment goes through the pipeline below. All of it is *local to the segment's destination geometry* — that is the point: on a non-planar layer, each segment of a loop can have a different gap, slope, and support.

### 10.1 Support fractions

`support_fractions_at(mid, segment.order, order_field, mesh_sdf, bed_z, config)` returns `(support_fraction, bed_fraction)`:

- **Order-field probe.** Step from the segment midpoint against the numeric order-field gradient by `step = (layer_height / ||∇φ||).clamp(layer_height, 4·layer_height)` (i.e. "one nominal layer down" in *order units*, adapted to the local gradient magnitude). `bed_fraction = ((bed_z − probe.z) / layer_height).clamp(0, 1)` measures how much of that step falls to the build plate.
- **SDF support with an order gate.** The probe's SDF distance gives a raw fraction `(1 − dist / nozzle_radius).clamp(0, 1)` — but mesh-solid material only *supports* this bead if the order field schedules it *earlier*: the fraction counts only when `order(probe) ≤ bead_order − 0.5·layer_height` (solid-but-later material is air at deposition time — the order field can invert deposition order relative to the mesh). No SDF → 1.0 (unknown).
- **Gravity check.** Because in a conformal/non-planar field the order gradient can point horizontally along an arch and falsely report support from the pillar *behind* the bead, the per-segment loop additionally probes the SDF one `layer_height` straight down (`mid − ẑ·layer_height`): `vertical_support = (1 − dist / nozzle_radius).clamp(0, 1)`. The segment's `support_fraction = min(grad_support_fraction, vertical_support).max(bed_fraction)`.

### 10.2 SDF geometric surface classification

For non-first-layer segments with a `mesh_sdf` (and not `DebugExcluded`), the SDF re-classifies the segment's kind from the *surface geometry* at its midpoint — `d_surface = −SDF(mid)` (positive inside solid), surface normal from the SDF gradient, skin thickness `1.4 × nozzle_diameter`, air probes one `layer_height` above/below:

- **`TopSurface`** — upward-facing normal within `10°` of horizontal (`angle_from_horiz = atan2(√(nₓ²+nᵧ²), |n_z|) ≤ top_max_angle = 10°`), air above (`SDF(mid + ẑ·h) > 0.02`), and within skin thickness of the surface.
- **`Bridge`** — downward-facing normal within `bottom_max_angle = 10°` of horizontal (a near-horizontal arch underside/ceiling), air below, within skin; speed set to `config.bridge_speed()`.
- **`Overhang`** — downward-facing normal between `10°` and `45°` from horizontal, within skin; speed set to `speed_for_kind(Overhang, config)`.

Steeper or deeper-than-skin geometry keeps its §3/§6 kind (Interior/vertical stays `WallInner`/`WallOuter`). This geometric pass complements — and can override — the footprint/stitch-based classification of §3.3 and §6.3.

### 10.3 Local layer geometry & slope cosines

The gap between this layer's isosurface and the previous one is measured by `extrusion::local_layer_geometry(field, mid, layer_height)` — and **this is where the AnisotropicFSM assumption is load-bearing**: it does *not* use the one-line gradient-inversion shortcut `h_local = h_nominal / ||∇φ||`. That shortcut is only correct when the field is arc-length calibrated (one order unit = one mm along the build direction — true for `HeightOrderField` and for the *isotropic* regions of the FSM solve), but `fsm_field_for`'s boundary-metric blending (`fsm_top_tangency_aspect`, `fsm_wall_ortho_aspect`) deliberately distorts front-propagation speed near walls for sequencing quality, *not* distance calibration; inverting that distorted magnitude produced wildly wrong bead heights in exactly the near-wall region the feature targets. Instead, `local_layer_geometry` **ray-marches from `p` along `−normal`** (geometrically growing steps, ×1.6 from `0.05·h_nom` up to `32·h_nom`) to bracket where the field crosses `order(p) − h_nominal`, then bisects to refine — measuring the *real* physical gap, exact regardless of local calibration. The result is clamped to `[0.1, 3.0] × h_nominal` (a physical safety bound, wide enough to capture the several-times-nominal real gaps a heavily metric-distorted FSM region produces; the gradient-inversion value is the fallback when no crossing is found).

Two cosines then shrink the *effective path length* used for volume (not `extrusion_length` directly):

- `surface_cos = surface_inclination_flow_factor(normal) = |n·ẑ|.clamp(0.15, 1.0)` — a flat horizontal nozzle tip over a surface inclined by `θ` sweeps a cross-section contracted by `cos θ` (clamped at 0.15 to avoid starving near-vertical walls);
- `trajectory_cos = √(1 − (dir·NOZZLE_DIRECTION)²)` — a move climbing along the nozzle axis sweeps a shrunken orthogonal cross-section;
- `slope_cosine = min(surface_cos, trajectory_cos).clamp(0.15, 1.0)`; `effective_distance = distance × slope_cosine`.

### 10.4 Bead cross-section

First-layer segments (`bed_fraction > 0` or `|layer.order − order_min| < 1e-6`) use `first_layer_height()`, `first_layer_line_width()`, the `+Z` normal, and carry `first_layer_mult = config.first_layer_extrusion_multiplier()`; all others use §10.3's values and `first_layer_mult = 1.0`.

Line width: the segment's `line_width` when set, else `extrusion::line_width_for_kind(kind, config)` — walls/top-surface use `wall_line_width`, infill/bridge use `infill_line_width`, and **`Overhang` is clamped to `min(wall_line_width, nozzle_diameter)`** (an unsupported line must never be wider than the nozzle bore — there's no supporting surface to squish/spread a wider bead).

The raw bead area is kind-specific:

- **`Overhang`** — a support-free track of width `nozzle_diameter − wave_overhang_overlap()`: `track_w × effective_layer_height × wave_overhang_flow()`.
- **`Bridge`** — `0.25·π·d² × 0.90` (a near-circular bore with a 10% squish allowance).
- **Everything else** — `blended_bead_cross_section_area(line_width, layer_height, nozzle_diameter, support_fraction, bed_fraction)`: a three-way blend of the physical bead shapes by support — the **stadium** (rounded rectangle, `bead_cross_section_area`, fully supported, the Slic3r model), the **circle** (`π·(d/2)²`, free air — *more* volume per mm than the stadium at typical width/height ratios; unsupported lines fed at stadium flow come out as thin saggy strands), and the **rectangle** (`width × height`, squished against the rigid plate — under-feeding it with the stadium volume is the classic ~12% first-layer underextrusion). `bed_fraction` takes precedence, then linearly blends stadium↔circle by `support_fraction`.

When `config.bead_clearance_compensation_enabled()` (and the segment is neither overhang nor bridge), the bead area is computed from the *channel-clamped* width via `clamped_bead_cross_section_area` — `line_width` shrunk to the segment's `channel_width` (§1.2) when finite — so a bead squeezed into a narrow feature isn't fed as if it had the full nominal width (only ever shrinks, never widens).

### 10.5 Speeds and volumetric clamping

Nominal speed by kind: `Overhang` → `config.wave_overhang_speed()`, `Bridge` → `config.bridge_speed()`, everything else → `motion_model.max_directional_feedrate(kind, is_first_layer, unit_dir)` — the machine's directional kinematic limit. It is then clamped by the machine's volumetric feed limit: `kinematics::clamp_feedrate_by_volumetric_limit(nominal_speed, bead_area, config.max_volumetric_speed)` (a bigger bead means the extruder can push less filament per second, so the feedrate drops). The result is stored in `segment.speed`.

### 10.6 Directional slope flow compensation

`config.slope_compensation_mode()` picks the compensation philosophy for climbing/descending non-planar segments (`climb_slope = unit_dir · BUILD_DIRECTION`):

- **`GeometricOffset`** (default) — the geometry is corrected (§6.2 lifts the nozzle instead), so flow only corrects *descents*: `1 − 0.12 · (−climb_slope)` clamped to [0,1]; climbs are untouched.
- **`VolumetricModulation`** — no geometric lift (§6.2 is skipped); flow modulates by the bead's squeeze ratio `squeeze_ratio = clamp(nozzle_flat_diameter / (2·bead_width), 0.5, 2.5)`: climbs get `1 + 0.03·climb` (more material to fill the climbing bead), descents get `(1 − 0.15·squeeze_ratio·descent_sin)` with a floor of `0.60`.

### 10.7 Fluid-dynamics swell

When a `fluid_dynamics::FluidDynamicsEngine` is configured for the active tool's nozzle temperature, the segment's flow `q = (clamped_speed / 60) × bead_area` (mm³/s, floor `0.01`) is fed to `engine.swell_volume_multiplier(q, 0.0)` — the viscoelastic die-swell correction for the flow rate; `1.0` when no engine is configured.

### 10.8 Final extrusion length & `FlowBreakdown`

Volume conservation between deposited bead and pushed filament: `segment_extrusion_length(effective_distance, bead_area, filament_area) = effective_distance × bead_area / filament_area` (filament is a circle of `filament_diameter`). The final value is

```text
extrusion_length = segment_extrusion_length × extrusion_rate
                × tool.extrusion_multiplier        (by the layer's object's ToolId)
                × first_layer_mult
                × directional_flow_mult            (§10.6)
                × swell_mult                       (§10.7)
```

and `segment.flow_breakdown` records the *actual* multipliers applied (`slope_cosine`, `first_layer_mult`, `directional_flow_mult`, `swell_mult`), defaulting the rest to neutral `1.0` — including `corner_flow_mult` and `transient_pressure_mult`, which later passes (§11) update **in place** on the same `FlowBreakdown` rather than overwriting it, so the breakdown always reflects what the printer actually receives.


---

## 11. Seam, wipe & corner post-passes

Five post-passes refine the paths **per layer**, at the end of the §2.3 layer loop, in this order (each gated on its own config flag): corner-flow (§11.1), scarf joints (§11.6), seam gaps (§11.4), pre-retract tapers (§11.3), and perimeter wipes (§11.5) — none resequences paths. A sixth, **transient-pressure compensation (§11.2)**, is different: it is a *global* post-pass that `plan_toolpaths_with_progress` (§2.1) runs on the flattened whole-print sequence after `plan_with_progress` returns, because the hotend's melt pressure carries across layer boundaries. All of them touch `extrusion_length`/`extrusion_rate` *in place* and record their factor in the segment's `FlowBreakdown` (`corner_flow_mult` / `transient_pressure_mult`), the ones §10.8 reserved.

### 11.1 Corner flow compensation — `corner_flow::apply_corner_flow_compensation`

Gated on `config.enable_corner_flow_compensation` with `corner_flow_compensation_ratio() > 1e-4`. At every junction between two extruding segments (closed loops wrap around: junction `j` uses `points[(j−1) % n, j, (j+1) % n]`), `calculate_corner_excess` measures the volume the junction deposits more than a straight run would:

- **Geometry.** In/out directions are projected onto the *local layer tangent plane* — surface normal from the order field's numeric gradient at the corner (fallback `+Z`, then pure-XY projection) — and the turning angle `α` is taken from the projected directions. Juctions with `cos α ≥ 0.9998` (α < ~1.15°) are skipped.
- **Geometric inner-corner overlap** — `v_geom = (w²·h / 4) · tan(α/2)` (with `tan(α/2)` ceilinged at 4.0 to tame 180° reversals), where `w` is the mean line width and `h` the layer (or first-layer) height.
- **Kinematic SCV shortening** — when the machine's square-corner velocity `v_corner = klipper_corner_velocity(dir_in, dir_out, scv, accel)` is > `0.05` and the directional acceleration > `1.0`, the rounded arc of radius `r_eff` (unconstrained `v_corner² / (accel·sin(α/2))`, capped so each leg's tangent distance stays under 40% of the shorter segment) shortens the path by `ΔL = r_eff·(2·tan(α/2) − α)`; `v_kinematic = ΔL·w·h`.

The excess `v_total = v_geom + v_kinematic`, scaled by `corner_flow_compensation_ratio()`, is deducted **symmetrically — half from the incoming segment, half from the outgoing one** — each deduction clamped to 35% of that segment's nominal volume (`length × line_width × layer_h`): `extrusion_length` drops by `deduction / filament_area`, `extrusion_rate` is rescaled by `new/old`, and `flow_breakdown.corner_flow_mult` records the factor in place.

### 11.2 Transient-pressure compensation — `transient_pressure::apply_transient_flow_compensation`

Gated on `config.enable_transient_pressure_compensation` — and called only from `plan_toolpaths_with_progress` (§2.1), after `plan_with_progress` has returned the complete print sequence, so the tracker walks the flattened paths **in final print order across layer boundaries**. It models the hotend's melt pressure as a first-order system with time constant `K_PA` (static `config.pressure_advance`, or `engine.dynamic_pressure_advance(q, 0.0)` from the fluid-dynamics engine when one is configured):

- **Per extruding segment** — `v_nominal = extrusion_length × filament_area`; with `x = t_move / K_PA`, the average pressure during the move is `P_avg = Q_target + (P_start − Q_target)·(1 − e^(−x))/x`, and the multiplier is `M = (Q_target / P_avg)^β` (clamped to `[transient_pressure_min_multiplier(), 1.0]`) when the move starts pre-pressurized, else 1.0. `extrusion_length = v_compensated / filament_area`, `extrusion_rate *= M`, `flow_breakdown.transient_pressure_mult = M` in place. Segments whose `extrusion_rate < 0.90` (i.e. already tapered by §11.3) hold the tracker's pressure at 50% of the pre-taper steady flow — viscous resistance prevents real hotends cratering to zero, and the next move must not assume an empty nozzle.
- **Short-move handling** — moves shorter than `0.25·K_PA` degenerate continuous integration; an adaptive proxy instead ramps `M` toward `M_min` over four consecutive short moves (high-frequency direction reversals build backpressure that the PDE form can't see), taking the min with the continuous multiplier.
- **Travels & retractions** — `process_travel` decays pressure exponentially over the travel duration; retractions charge `−retraction_length × filament_area` into the tracker and unretractions charge the positive volume, exactly mirroring `gcode::emit`'s retract-before-travel / unretract-before-resuming logic (per-tool `retracted` state, `effective_min_travel_for_retract()` threshold, tool-change reset) so the compensation tracks what the Gcode actually does.

Segment durations come from `plan_path_velocities` (the same SCV / minimum-cruise kinematic profile the machine will execute), which is why this pass must run after §10 has set every segment's speed.

### 11.3 Pre-retract taper — `kinematics::apply_pre_retract_taper`

For paths whose extruding run is ≥ 1.5× the configured taper distance: across the final `taper_distance_mm` of the last extruding run, each segment's `extrusion_rate` and `extrusion_length` taper linearly from 1.0 down to `min_rate` (default `0.20`) by distance-to-end — bleeding melt-zone pressure *before* the retraction so the nozzle doesn't blob at the retract point. If the final segment alone exceeds the taper distance by > 0.1 mm, it is split into an untapered lead plus a tapered tail (the tail getting the average rate `(1 + min_rate)/2` instead of a point-wise ramp).

### 11.4 Seam gap — `kinematics::apply_seam_gap`

For closed **wall loops** (first segment `WallOuter`/`WallInner`): the last `seam_gap_mm` of travel before the loop closes is unextruded — the final extruding segment is split into an extruding lead plus an unextruded coasting tail (`extrusion_length = extrusion_rate = 0.0`, geometry unchanged), or consecutive tail segments are zeroed until the gap is filled (splitting the boundary segment when it straddles). The residual nozzle pressure bleeds into the gap instead of landing on the closure point — the seam blob/zit is eliminated rather than moved.

### 11.5 Wipe moves — `kinematics::apply_wipe_moves`

For paths that both start and end with extruding segments, a `MoveKind::Travel` wipe segment of up to `wipe_distance_mm` (clamped to the first segment's length) is appended along the direction of the loop's first segment, anchored at the loop start. It is unextruded (`extrusion_length = 0.0`, `line_width = 0.0`), copies the last segment's speed/order/support metadata, and drags the nozzle tip along the just-deposited wall so a travel lift or retraction doesn't leave a drool mark at the seam.

### 11.6 Scarf joints — `kinematics::apply_scarf_joint`

The non-planar perimeter-seam treatment, for closed wall loops (≥ 3 points, total length ≥ 2.5× the scarf length, **no unsupported segment**: any segment with `support_fraction < 0.8`, kind `Overhang`, or `Bridge` disqualifies the loop). It rebuilds the loop with two complementary flow ramps around the seam point:

- **Lead-in wedge** — over the first `scarf_length_mm` (effective, capped at 40% of the loop and subdivided into `steps` of ≥ 0.2 mm each), the flow ramps from `start_height_fraction` (clamped to `[0, 0.95]`) to 100%. In `GeometricOffset` mode the ramp points are offset *against the local slice normal* by `−(1 − h_frac)·layer_height` — the slice normal being the order field's numeric gradient at the point (flipped upward, `+Z` fallback) — so the wedge is normal to the *surface*, not to Z; the first bead sits at 10% of the layer height and grows as it climbs out of the joint.
- **Lead-out wedge** — the loop continues past its start point for the same distance at nominal height, with the flow ramping from `1 − start_height_fraction` to 0%, so the overlap of the two beads sums to exactly one nominal bead everywhere across the joint: no vertical seam line, no localized overextrusion.

Ramp segments get `extrusion_rate *= flow_frac × scarf_flow_ratio` (ratio clamped to [0.10, 2.0]) and a fully recomputed `extrusion_length` — per-mm flow × ramp distance × the flow multiplier × a fluid-engine swell correction for the ramped flow rate (clamped [0.60, 1.0]) × a downhill correction (ramp segments climbing the wedge against the build direction get up to 15% less flow, floor 0.70) — then their `FlowBreakdown` is **nulled** (the recomputation mixes factors with no home in the breakdown schema) and `is_scarf = true` is set. The main loop body between the two wedges is left at nominal height and flow.


---

## 12. Support-aware emission deferral (`defer_unsupported_paths`)

The first pass of §2.4's global phase. It exists for one specific FSM failure mode: a path planned at layer order `O` that rests mostly on mesh-solid whose order-field value is *later* than `O` — for example infill over a fast-march tunnel interior that the front reaches from the far side, so the material underneath it is deposited **after** it. Emitted in place, that path would print into free air. The pass reorders the flattened path list so such a path is emitted only *after* the layer group that prints its supporting material.

### 12.1 Detection (order-aware floating probe)

Only paths with at least one `Infill` or `TopSurface` segment are candidates; everything else (walls, bridges, overhangs) keeps its position. For each candidate, every extruding segment midpoint is stepped one layer-height against the local order-field gradient — `step = (layer_height / ||∇φ||).clamp(layer_height, 4·layer_height)` in `−gradient` direction, exactly the §10.1/`support_fractions_at` convention — and the landing probe is classified against the mesh SDF (`sdf_tolerance = 0.5 × nozzle_diameter`) and the order field (`order_epsilon = 0.5 × layer_height`):

- **bed** — probe at or below `bed_z + 0.25·layer_height`: resting on the plate, counted as supported (skipped);
- **already-printed solid** — probe inside SDF and `order(probe) ≤ segment.order − order_epsilon`: supported by material that is printed first (skipped);
- **future solid** — probe inside SDF but at the *same or later* order (`> segment.order − order_epsilon`, whether same-band or genuinely later): this length counts as `future` and updates `required_order = max(required_order, order(probe))`;
- **open air** — probe not inside SDF: counted as `unsupported` but *not* `future` — a genuine overhang with nothing beneath anywhere in the model is a bridging/overhang problem, not a deferral problem.

Every other length counts as `unsupported` (the SDF-inside-but-same/later-order case adds to it as well).

### 12.2 The deferral decision

A path is deferred **iff all of** hold:

- `unsupported / total ≥ DEFER_MIN_UNSUPPORTED_FRACTION = 0.7` — at least 70% of its extruded length is not on bed or earlier-order solid;
- `future / unsupported ≥ DEFER_MIN_FUTURE_FRACTION = 0.5` — at least half of that unsupported length lands on *future* solid;
- `required_order − layer.order ≤ DEFER_MAX_ORDER_SPAN × layer_height` (span `40.0` layer heights) — deferring further risks nozzle collisions with much-taller surrounding geometry, so paths whose support lies more than 40 layers out are left in place.

When deferred, the raw field-sample order `required_order` is **snapped up to the discrete layer order** that actually prints the supporting material — the first layer order `o ≥ required_order − 1e−9` in the sorted `layer_orders` — and the path's emission order becomes `max(group_order, layer.order) + 1e−9`. The `+1e−9` nudge makes ties resolve to "after": the path lands *just past* the group it now rests on. Non-deferred paths keep emission order `layer.order` exactly.

### 12.3 The reorder

Paths are re-keyed as `(emission_order, flat_index)` and the list is **stably sorted** by that key — so non-deferred paths (emission order = their layer order) keep their exact original sequence, including per-layer travel optimization, and each deferred path lands immediately after the supporting group. Notably, the segments' `order` stamps are **left untouched**: they describe the layer the geometry belongs to, not when it is emitted — downstream metadata (GUI views, per-segment support math) keeps referring to the right layer while the G-code emits the path later. This is the only place the whole print is re-sequenced; every subsequent global pass (§2.4 steps 2–5, §11.2, §13) operates on this final order.


---

## 13. End-of-print wipe/clearance, bounds validation & G-code handoff

The last three steps of `plan_toolpaths_with_progress` (§2.1) wrap up the print, then the paths cross into G-code.

### 13.1 `append_end_of_print_wipe_and_clearance` — terminal wipe & parking

Gated on `config.end_of_print_wipe_enabled()`. It locates the **last path with any extruding segment** in the whole print and rewrites its tail into a three-move exit sequence (all unextruded, all inheriting the last extruding segment's `order`/`support_fraction`/`island`; their `Segment::id` stays `0` — the global id assignment of §2.4 has already run):

1. **Wipe** — first, any *trailing* non-extruding segments after the last extruding one are truncated so the path ends cleanly at that segment's destination `p_end` (the `§11.4` seam-gap coasts and `§11.5` wipe segments are discarded — they belong to the loop, not the print). Then a `MoveKind::Wipe` segment travels *backwards* from `p_end` along the bead by `calculate_end_of_print_wipe_distance`: the manual override `config.end_of_print_wipe_distance` clamped to `[0, segment_len]`, or — when unset — `c_pa·v_terminal + 1.0·nozzle_diameter` (dynamic pressure advance from the fluid engine at the terminal flow, or static `config.pressure_advance`), clamped to `[0.5·d, min(5.0, segment_len)]`. Wiping backwards drags the residual filament in the nozzle into the bead just printed instead of leaving a drool blob at the endpoint.
2. **Shear step** — a `Travel` to `p_shear = p_wipe + exit_dir·(1.5·nozzle_diameter)`, where `exit_dir` is the normalized blend `1.5·n_cad + 0.5·n_iso` (CAD normal from the SDF gradient + isosurface normal from the order-field gradient, both at `p_wipe`; the isosurface normal flipped upward, `+Z` fallback; pure XY of the reversed direction when no SDF). Both the exit target and everything below stay inside the **safe box**: the build volume inset by 0.5 mm on every axis (sphere volumes use their bounding box).
3. **Clearance move** — a second `Travel` to `p_clear`: the shear point shifted outward in XY by `0.5 × machine.arrangement_clearance()` (min 1.0 mm, direction normalized; `+X` fallback) and lifted by `config.end_of_print_clearance_z_lift()`, clamped to the safe box. Before emitting, if `travel_chord_is_blocked` (§8.3) says the shear→clearance chord would clip already-printed solid, the clearance point is raised to `max_z + nozzle_diameter` (capped at `safe_max.z`) so the print ends above its own geometry rather than through it.

Note this pass rebuilds its own world-space SDF and order field for the final object (the object's vertices transformed by its placement, `order_field_for(config.order_field, …)` — a *fresh* solve scoped to that object's mesh) and a fluid engine for the final tool's temperature; it does not reuse the slice's cached fields.

### 13.2 `validate_within_bounds` — machine-envelope check

The last-resort safety net, checked once after planning: every point of every path — **including pure travel moves** — must lie inside `machine.build_volume`, or planning fails fast with `Error::MoveOutOfBounds { point }` naming the first offending point in `paths` order. This is a machine-envelope check, not a solid-containment check: travel moves are deliberately *not* constrained to the mesh (§6), and a travel point outside the build envelope can only be discovered by the printer firmware refusing the move at print time. Failing here, before any G-code is written, is strictly preferable.

### 13.3 Handoff to G-code — `gcode::emit` / `emit_with_machine`

`gcode::emit(paths, config)` wraps `emit_with_machine(paths, config, machine: Option<&Machine>)` (the production consumer goes through `slice_to_gcode(_with_progress)` or the GUI's `finish_slice`, always with a machine). The emitter is the only thing that turns the planned `Vec<Path>` into G-code text; everything upstream (§2–§12) is machine- and format-agnostic.

**Structure.** The program is wrapped in OrcaSlicer/PrusaSlicer-style `; HEADER_BLOCK_START`/`_END`, `; EXECUTABLE_BLOCK_START`/`_END`, and `; CONFIG_BLOCK_START`/`_END` markers. The whole `SlicerConfig` is echoed as sorted `; config.<field> = <value>` comment lines in the `CONFIG_BLOCK` (serialized via `SlicerConfig`'s serde impl — new fields appear automatically) alongside a narrower Moonraker-parseable footer subset. `config.start_gcode` is prepended and `config.end_gcode` appended with `{print_min_x}`/`{print_max_x}`/… placeholders substituted from the **first layer's XY bounding box**, where "first layer" is identified by the *minimum `Segment::order` value across all paths* — robust to non-planar order fields, where Z alone doesn't identify layer 1 (a curved layer's points don't share one Z, but its segments all share the source layer's order).

**Extrusion model.** `M83` (relative extrusion) is emitted explicitly immediately *after* `start_gcode` — the start template may itself change extrusion mode, so the emitter can't assume the machine is left in relative mode — and every extruding move's `E` value is `Segment::extrusion_length` as an incremental per-move delta: the entire §10–§12 flow model (slope cosines, bead blending, first-layer multiplier, corner/transient deductions, taper, scarf ramps) arrives pre-baked in that one number.

**Klipper object exclusion.** Each object's paths are wrapped in `EXCLUDE_OBJECT_START/END NAME=<sanitized name>` blocks, preceded up front by `EXCLUDE_OBJECT_DEFINE NAME=… CENTER=… POLYGON=[…]` (center + convex XY footprint polygon, spaces → underscores in names); grouping uses `Path::object`.

**Tool changes.** A `T{n}` line is emitted whenever consecutive paths carry different `Path::tool`, followed by `G92 E0` (reset that tool's filament position) and `M83` again, and the per-tool `retracted` flag resets to `true` (a fresh tool has no primed filament). Prime/purge G-code around tool changes is a documented follow-up (ROADMAP.md Phase 2).

**Retractions & pressure advance.** `G10`/`G11` (when `use_firmware_retraction` and no fluid dynamics) are emitted only at travel-run/extrusion-run *transitions*, tracked by the per-tool `retracted` flag (initial `true`: `start_gcode` is assumed to leave the printer retracted). A retract fires only when the *upcoming contiguous travel run's total distance* exceeds `effective_min_travel_for_retract()` (including the inter-path move when entering a path's first travel run), or **immediately when the previous move was a `Wipe`** — a wipe has already left the nozzle at the part edge, and there's no reason to skip the retract. Otherwise (firmware retraction off, or a fluid-dynamics engine present) retraction is an E-only move: `G1 E−<length> F<retraction_speed>`, with the length from the fluid engine (`engine.retraction_length(pa, filament_velocity)`) when configured and an unretract `G1 E<length>` on resuming (the engine's `unretract_length(last_retraction_len, accumulated_travel_time_s, fan_fraction)` plus `unretract_extra_length()`). When a fluid-dynamics engine is active, `SET_PRESSURE_ADVANCE ADVANCE=<q>` is emitted — deadband-gated — just before the travel move that leads into the next extruding segment, so the firmware can process the PA change during transit; when **slicer-side pressure advance** (`enable_slicer_pressure_advance`) is on, firmware PA stays 0 and each path is instead run through `subdivide_pa::subdivide_path_for_pressure_advance` — error-bounded adaptive subdivision of the acceleration/deceleration phases into piecewise-linear chords that track the non-Newtonian constitutive fluid advance `E*(s)` (cruise segments untouched; bounded by `slicer_pa_tolerance_mm`, minimum printable segment length, and a maximum command frequency to avoid starving Klipper's serial buffer) — followed by a fresh `plan_path_velocities` profile for the subdivided points. This is the *sole* site of pressure-advance baking, deliberately: `plan_toolpaths_with_progress` leaves it out so it isn't applied twice.

**Checkpoints.** When `config.enable_slicer_checkpoints` is set, the emitter models its own elapsed time (per-path `plan_chained_path_velocities` profiles plus retraction/unretract durations — mirroring §11.2's exact retraction/unretract timing, per-tool `retracted` flag and all) and, once the modeled clock crosses each `slicer_checkpoint_interval_seconds` threshold, emits a `RESPOND TYPE=command MSG="action:slicer_checkpoint {\"num\": N, \"rem\": R}"` line at the next non-extruding move — the Moonraker `slicer_checkpoint` action with the remaining time, safe to pause on because it lands between extrusion moves. The modeled total (used for `rem`) is computed by `checkpoint_total_time_seconds` over the whole print with the same retraction logic and, when slicer-side pressure advance is on, the same `subdivide_path_for_pressure_advance` + re-profile pass the emit loop applies, so checkpoint timings track what the machine actually does.



