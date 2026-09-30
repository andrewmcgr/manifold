# SDF-Inset Containment Gate for Sparse Infill Regions — Design

## Problem

Slicing `TestObj1.stl` with `examples/profile.json` (AnisotropicFsm order
field, `layer_height` 0.2, `shell_thickness` 1.2, `wall_line_width` 0.4,
`wall_offset` 0.2 → `wall_count() == 3`) places sparse infill **outside the
solid**: 120 Infill paths, 2 118 points with SDF > 0, worst 2.2 mm away.
Instances are reported on the part's midline at order ≈ 11.1, and the
worst cluster is at order ≈ 11.847 (slice layer 66).

## Root cause (verified with scratch probes)

Stage-by-stage probing of layer 66 (`probe_layer_diag`):

| Stage | Result |
|---|---|
| Wall loops (16) | all inside (SDF ≤ −0.19) |
| `infill_boundary` (338 pts) | all inside; SDF range −1.40 … −0.33 (most ≈ −1.4, i.e. on the infill-boundary SDF isosurface; a tail up to −0.33) |
| `solid_fill_boundary` (125 pts) | **117 pts outside the solid** (max +0.35) |
| 2D sparse difference, pre-reconstruction (131 pts) | all inside *at seed height* — but **16/131 pts are outside the 2D `infill_boundary` polygon** (`polygon2d::difference` leaked) and 54 sit inside the 2D `solid_fill` polygon (not clipped out) |
| Post-`reconstruct_on_order_field_near` (131 pts) | **110 pts outside the solid** (max +0.35); worst final points sit at z = 4.3–8.3 although the layer order is 11.85 (≈ z 12.3) |
| Final Infill/TopSurface paths at this layer | 892 outside points, worst SDF +2.22 |

Column analysis at the worst point (172.213, 178.875): SDF is positive at
*every* z in 3.0–14.0 (min +0.375 — no material in the column at all) and
the order field is undefined (`+inf`) along it; its nearest XY reference
(the layer's infill-boundary point at z = 12.297) has SDF +0.123. The same
structure, milder (max +0.12), appears at the user-reported order ≈ 11.1
layer (217/424 region points outside the 2D infill polygon).

Chain of causes:

1. The layer's wall/infill-boundary contours are **3D curves on the
   order-field isosurface**. `InfillRegion::from_layer` flattens them into
   the layer plane (`polygon2d::to_2d`), runs a 2D inset/difference, then
   re-lifts the result onto the isosurface via
   `order_field::reconstruct_on_order_field_near`.
2. Where the isosurface is non-monotonic along the slice axis (curved
   tops, voids/bored sections — same XY column crosses the isosurface at
   several heights), the flattening folds: the 2D polygons become
   self-overlapping, and the 2D boolean leaks points outside the subject
   polygon (observed: 16 and 217 leaked points on the two layers above).
3. `reconstruct_on_order_field_near` seeds each (u, v) from the nearest
   reference in *XY only* — which can be a different branch, or a point in
   air — and when no valid bracket is found its fallback leaves the seed
   unadjusted (`None => planar`). Points then sit where no material
   exists.
4. `solid_fill_boundary` (the top/bottom skin region) leaks identically,
   which feeds the re-tagged TopSurface paths.
5. The final path-level containment filter
   (`CONTAINMENT_POINT_SLACK` 0.35 mm / `CONTAINMENT_OUTSIDE_FRACTION`
   25 %) is too permissive for this defect class, so the leaking paths
   survive.

## Design: SDF containment gate on region loops

Enforce, as a hard invariant on the region loops *before* any infill
generator runs:

- **Sparse region loops** (including narrow-solid slivers derived from
  the sparse region): every boundary point must satisfy
  `SDF(p) ≤ −(wall_offset + wall_line_width) + slack` — i.e. at least one
  full wall line inside the outer surface, the same depth relationship
  the walls themselves have (`slicing.rs` builds wall w at
  `SDF = −(wall_offset + w × wall_line_width)`). This is the user-requested
  "inset the SDF and intersect with the order isosurface", applied to the
  region boundary: the boundary is the layer's isosurface curve (from the
  existing reconstruction) clipped to the inset-SDF region. Interior
  chords of generated paths are covered by the segment-level backstop
  below — the §6 path-containment filter alone was verified
  insufficient: it keeps mostly-valid paths, and re-lifted chords can
  dangle in air even on fully gated boundaries.
- **Solid-skin loops** (`layer.solid_fill_boundary`): every boundary
  point must satisfy `SDF(p) ≤ +slack` (inside the solid; no inset —
  skin material belongs on the surface).

`slack = 0.1 mm` (`SDF_REGION_GATE_SLACK`), a fixed constant: SDF
sampling error on the test meshes is well under this, and it is
independent of layer height.

Points failing the test are removed and the loop is re-closed locally:
for a maximal run of removed points between retained neighbours `a` and
`b`, insert the chord midpoint `m = (a+b)/2` iff `SDF(m) ≤ max_sdf`;
otherwise insert nothing (direct chord `a→b`). A loop whose entire
boundary fails is dropped; a loop reduced to fewer than 3 points is
dropped. Non-finite SDF samples count as failing (undefined field ⇒
unknown containment ⇒ drop). This is a *monotone shrink*: the region only
loses area, always toward the safe (deeper) side.

Where the gate runs: `toolpath.rs`, in the sparse/solid infill branch of
the per-layer planning loop — after the sparse/narrow partition and again
after the wave/bridge/tangent footprint-mask reconstruction (which
re-runs `reconstruct_on_order_field_near` and can reintroduce off-surface
points). When `layer.mesh_sdf` is `None` (synthetic test layers) the gate
is a no-op, matching the existing behaviour of every other SDF-based
pass.

### Segment-level backstop

The gate controls region *boundaries*, but the generators re-lift
interior pattern points onto the order isosurface (the same
`reconstruct_on_order_field_near`), and interior chords can still
cross air where the isosurface folds (verified on TestObj1: up to
+2.86 mm on gated layers). As a final per-layer pass — after wipe,
before the micro-path drop — every Infill or TopSurface segment whose
destination point has `SDF > 0.5 × nozzle_diameter` is re-tagged
`MoveKind::Travel` with zero extrusion length: path topology is
preserved, the dangling chord simply travels without extruding, and the
micro-path drop removes any resulting stub. Bridge/Overhang segments
are exempt: a bridge spans a void by construction, so positive SDF at
its midpoints is expected. This is the codebase's existing
"never print infill in open air" policy (`retain_contained_paths`
drops whole infill paths on gross violation) applied at segment
granularity.
## New API

In `crates/manifold-core/src/infill.rs`:

```rust
const SDF_REGION_GATE_SLACK: f64; // 0.1

pub fn clip_loops_to_sdf(
    loops: Vec<Vec<DVec3>>,
    sdf: &manifold_fidget::mesh_sdf::MeshSdf,
    max_sdf: f64,
) -> Vec<Vec<DVec3>>;

pub fn gate_sparse_loops(
    layer: &crate::slicing::Layer,
    config: &SlicerConfig,
    loops: Vec<Vec<DVec3>>,
) -> Vec<Vec<DVec3>>;
// applies clip_loops_to_sdf with
// max_sdf = -(config.wall_offset + config.wall_line_width) + SDF_REGION_GATE_SLACK;
// no-op when layer.mesh_sdf is None.

pub fn gate_skin_loops(
    layer: &crate::slicing::Layer,
    config: &SlicerConfig,
    loops: Vec<Vec<DVec3>>,
) -> Vec<Vec<DVec3>>;
// applies clip_loops_to_sdf with max_sdf = +SDF_REGION_GATE_SLACK;
// no-op when layer.mesh_sdf is None.
```

Call sites in `crates/manifold-core/src/toolpath.rs` (per-layer infill
branch, ~line 3102–3250):

```rust
let region = InfillRegion::from_layer(layer, config);
let (mut sparse_loops, mut narrow_solid_loops) = /* existing partition */;
sparse_loops = infill::gate_sparse_loops(layer, config, sparse_loops);
narrow_solid_loops = infill::gate_sparse_loops(layer, config, narrow_solid_loops);
let mut all_solid_loops = infill::gate_skin_loops(layer, config, layer.solid_fill_boundary.clone());
all_solid_loops.extend(narrow_solid_loops);
// ... footprint mask block (existing) ...
// after the mask's reconstructions, re-gate both sets (idempotent):
sparse_loops = infill::gate_sparse_loops(layer, config, sparse_loops);
all_solid_loops = infill::gate_skin_loops(layer, config, all_solid_loops);
```

(The second re-gate of `all_solid_loops` with the looser skin threshold
is a no-op on the already-gated narrow slivers: anything passing the
stricter sparse threshold passes the looser one.)

## Testing

1. **Unit tests** for `clip_loops_to_sdf` against a synthetic
   `MeshSdf` of the box `[−5, 5]³` (SDF at an interior point is exactly
   `−min(5−|x|, 5−|y|, 5−|z|)`): fully-inside loop unchanged; fully-outside
   loop dropped; local outside run bridged with the passing midpoint
   (exact expected output); single-retained-point loop dropped; outside
   run wrapping loop index 0 handled.
2. **Regression test** (failing before the gate):
   `plan_toolpaths` on the tracked `TestObj1.stl` fixture with a config
   mirroring `examples/profile.json` (AnisotropicFsm, `shell_thickness`
   1.2, `wall_line_width` 0.4, `wall_offset` 0.2, `layer_height` 0.2).
   Assert that no Infill or TopSurface *segment* ends more than
   `0.5 × nozzle_diameter` (0.2 mm) outside the solid — pre-gate worst
   case was +2.22 mm; the residual ≤ 0.2 mm points are the
   bottom-layer isosurface band, a slicing-side effect within the
   codebase's documented 0.35 mm containment slack — with a
   non-vacuity bound (≥ 1 000 Infill segments). The pre-gate probe run
   already establishes the test fails today.
3. **Verification**: `probe_infill_outside` on TestObj1 must report no
   Infill/TopSurface extrusion point more than 0.4 mm (one nozzle
   diameter) outside the solid (pre-gate: 2 118 points > 0.05 mm,
   worst +2.22; post-fix: worst +0.199, zero points beyond 0.4 mm —
   the residual 0.05–0.2 mm points are the bottom-layer band and
   top-surface rounding, within documented slack); planning wall-time
   overhead on TestObj1 ≤ ~10 % (gate cost is one SDF sample per
   region-boundary point plus one per fill segment — the same cost
   class the §10/§11 passes already pay).

## Out of scope / follow-ups

- **Explicit 3D intersection curve** `{SDF = −inset} ∩ {order = c}`
  extracted by dual-field marching, used as the region boundary source.
  Mathematically the cleanest version of this design; not needed for the
  observed defects because the gate enforces the same containment
  invariant on the boundary the generators actually consume. Revisit only
  if the gate leaves residual artifacts (e.g. interior chords cutting
  air on severely non-monotonic geometry that the per-segment checks
  also miss).
- Tightening `CONTAINMENT_POINT_SLACK` / `CONTAINMENT_OUTSIDE_FRACTION`
  (§6). The gate removes the defect at the source; §6 stays as
  defense-in-depth at its current values.
- The separate summit-layer contour defect (stray WallOuter at order
  ≈ 14.3 floating ~2.4 mm above the part): a slicing-side issue, not
  addressed here.
