# Perpendicular Top-Surface Distance & Wall/Solid-Fill Overlap Elimination — Design

## Status: Approved design, ready for implementation planning

## Background

A recent plan (`docs/superpowers/plans/2026-09-16-directional-top-surface-distance.md`,
merged at `9a6df76`) replaced angle-thresholded top-surface patch clustering
with `TopSurfaceAwareOrderField`, a directional ray-march that classifies
any point as near a "top surface" (`SeedKind::Patch`) or not
(`SeedKind::Bed`), for any geometry — flat, tapered, domed. That plan's
final review parked two known issues rather than fixing them:

1. Under `SlicerConfig::default()`'s 0.4mm nozzle stack, a steeply
   tapering top still gets no `solid_fill_boundary` near its apex, because
   the march reports raw vertical (climb-direction) travel distance, and
   points sampled for the solid-fill decision are already inset from the
   true surface by the wall stack — a taper's inset translates into a
   larger vertical march distance than the same lateral inset would on a
   flat top.
2. The march measures climb-direction distance in general, not true
   (perpendicular) proximity to the top surface — a broader architectural
   limitation of which (1) is one symptom.

A separate, independently observed bug: near a taper's tip, where the
local cross-section becomes narrower than the full wall stack
(`wall_count * wall_line_width`), inner wall rings and solid infill can
end up printing over the same physical space — duplicate extrusion.

This design fixes both, because they share a root cause and a mechanism.

## Root Cause

`seed_proximity`'s march travels straight up from a query point through
solid material until it exits, and reports the raw accumulated travel
distance. For a locally planar exit surface with unit outward normal `n`,
and climb direction `d` (unit vector), the *perpendicular* distance from
the query point to that surface is `L * |d · n|`, not `L` itself — the
march currently reports `L`. On a flat top, `n == d` so the two coincide
and the bug is invisible; on a taper, `|d · n| < 1` and the raw distance
overstates true proximity, sometimes enough to miss the `top_threshold`
cutoff entirely (issue 1 above). This is also exactly the general
perpendicular-vs-vertical distinction that issue (2) names.

Separately, inner wall loops (`wall_index 1..wall_count`) and solid
infill are extracted independently and never checked against each other.
Normally they don't overlap (inner walls live in the shell between the
true surface and `infill_boundary`; solid infill lives inside
`infill_boundary`), but where the local solid narrows below the full wall
stack's width, that separation breaks down.

## Design

### 1. `march_to_top` reports perpendicular distance

At the march's interpolated exit point, sample the bed-excluded mesh
SDF's local gradient to get the outward surface normal `n`. Project the
accumulated climb-direction travel distance `L` onto it:
`perpendicular_distance = L * |climb_direction · n|`.

- If `MeshSdf` does not already expose a gradient/normal query, add a
  small central-difference helper alongside it, mirroring the existing
  `numeric_gradient` helper's pattern for `OrderField`.
- Degenerate gradient (near a sharp edge, or a flat numerical spot): fall
  back to the raw climb distance `L` rather than producing NaN or a
  wildly wrong value — same defensive posture as the march's existing
  `len < 1e-9 → return None` check.
- This is a pure quality improvement to what `seed_proximity` already
  returns for `SeedKind::Patch`; `SeedKind::Bed`'s distance
  (`inner.order(p)`, exact vertical height above a flat bed) is
  untouched — the bed is always flat, so no analogous correction applies
  there.

### 2. Factor out a reusable "solid-fill-eligible 2D region" function

`compute_solid_fill_boundaries` already computes, per layer, a 2D region
of points where `seed_proximity`'s margin (`threshold_for_kind -
distance`) is non-negative, via a `SeedMarginField` + `extract_contours`
+ center-sample disambiguation for the uniform-region case. Extract this
into a standalone function:

```rust
fn seed_eligible_region(
    layer: &Layer,
    config: &SlicerConfig,
    axis: DVec3,
    apex: DVec3,
    basis1: DVec3,
    basis2: DVec3,
    extent_2d: (f64, f64, f64, f64), // min_u, min_v, max_u, max_v
) -> Vec<Vec<DVec3>>
```

This function never actually depended on `infill_boundary` existing — it
only used it as a convenient sampling extent. Callers supply their own
extent, so it can run before `infill_boundary` exists.

`compute_solid_fill_boundaries` becomes a thin wrapper: call this
function using `infill_boundary`'s own extent, intersect with
`infill_boundary` as today, filter by `min_solid_area`, reconstruct.
No behavior change beyond the underlying distance now being
perpendicular (item 1).

### 3. Subtract the eligible region from inner wall loops

In the per-layer wall-extraction loop (`slicing.rs`), for each wall loop
with `wall_index >= 1` (never `wall_index == 0` — the outer perimeter
always prints, unconditionally, for surface quality), after the loop is
extracted and validated:

1. Project the loop to 2D in the same frame it was extracted in.
2. Call `seed_eligible_region` using that loop's own local extent.
3. `polygon2d::difference(wall_loop_2d, eligible_2d)`, `filter_min_area`
   to drop degenerate slivers.
4. Reconstruct the remaining sub-path(s) back onto the order field via
   `reconstruct_on_order_field_near`, the same pattern used everywhere
   else in this file.
5. Replace the original whole loop with zero, one, or more resulting
   sub-paths.

This mirrors the existing pattern in `toolpath::plan`, which already
subtracts bridge/wave-overhang 2D footprints from infill loops the same
way — no new geometric machinery, an existing one applied to a new
producer/consumer pair.

A wall loop entirely consumed by the subtraction (fully within solid-fill
range) naturally degrades to "dropped entirely" — the same effect the
simpler "whole-loop bail" alternative would have given for the all-or-
nothing case, but reached correctly here for the partial case too, which
whole-loop bail cannot do (its conservative default — keep the whole loop
when only part qualifies — leaves the qualifying sub-arc double-printed,
which is the exact bug this design exists to eliminate).

## Known Open Question — must be resolved as the plan's first task

`WallLoop`'s structure and the downstream toolpath pipeline (ordering,
gap-fill, gcode emission) may assume wall loops are always closed. Step
3 above can produce an **open arc** when a loop is partially consumed.
Gap-fill paths are already open, so there is precedent for open paths in
this pipeline, but this must be verified — read `WallLoop`'s definition
and every consumer before implementing the subtraction, and if open wall
paths aren't already supported end-to-end, add that support as an
explicit sub-task before wiring in the subtraction itself.

## Known Interaction — deferred to a later plan

Wall loops today are extracted in `plane_basis(BUILD_DIRECTION)` with a
bbox-centered origin; `compute_solid_fill_boundaries` (and this design's
new `seed_eligible_region`) use `plane_basis(axis)` with `origin = apex`.
These coincide for `OrderFieldKind::Height` (`axis == Z ==
BUILD_DIRECTION`) but diverge for `Conical` and other non-Z-axis kinds —
a separate, already-identified frame-mismatch issue tracked for a later
plan (see Backlog below). This design's wall-subtraction step should use
whatever frame the wall loop was already extracted in for its own 2D
projection, and is not expected to be geometrically exact for non-Z-axis
order fields until that later plan lands. Note this explicitly in code
comments at the subtraction site.

## Testing

- Unit test on `march_to_top`: a point on a known-slope cone at a
  non-apex, non-step-aligned location, asserting the reported distance
  matches the perpendicular closed-form value (`L * cos(slope_angle)`),
  not the raw vertical one — same style as the existing distance-pin
  test added in the prior plan's final fix wave, extended to a sloped
  surface.
- Regression test reusing the existing tapering-cone end-to-end test
  (`compute_solid_fill_boundaries_covers_a_steeply_tapering_top_not_just_flat_ones`),
  but at `SlicerConfig::default()`'s actual 0.4mm nozzle stack — the
  config that currently fails — proving solid fill now activates without
  any test-side config tightening.
- New end-to-end test: a mesh with a taper narrow enough that its
  cross-section drops below `wall_count * wall_line_width` near the tip,
  asserting the emitted `Path`s contain no overlapping
  wall/solid-infill segments in that region (e.g. via mutual
  containment or distance checks between emitted extrusion paths) — the
  direct regression test for the duplication bug.

## Backlog (not designed yet, tracked for future plans)

- **Plan B** — narrow-region-promotion heuristic (`extent < 15 *
  nozzle_diameter` forces solid, regardless of the actual solid-fill
  decision) and center-sample disambiguation (a single bbox-center
  sample decides "uniformly solid vs uniformly sparse" for an entire
  layer, which can misclassify concave/multi-island footprints). Smaller,
  independent robustness fixes.
- **Plan C** — the plane-basis frame mismatch (`BUILD_DIRECTION` vs
  `axis`, noted above) and the complete absence of slope correction in
  every infill pattern generator (spacing is computed in the flat
  projected plane and never compensated for surface steepness). Larger,
  more speculative infill-quality work.

## Self-Review

- **Placeholder scan:** No TBDs left unresolved except the explicitly
  flagged "Known Open Question" (WallLoop open-path support), which is
  deliberately scoped as the implementation plan's first investigation
  task rather than a design gap — its answer determines whether a
  sub-task is needed, not whether the design is complete.
- **Internal consistency:** The perpendicular-distance formula is used
  consistently in both the march (item 1) and is not re-derived
  elsewhere; `seed_eligible_region`'s signature is used identically by
  both call sites described (wall subtraction and
  `compute_solid_fill_boundaries`).
- **Scope check:** Focused on one implementation plan. Explicitly
  excludes the frame-mismatch and slope-correction items (Backlog),
  which are real but independent.
- **Ambiguity check:** The partial-vs-whole-loop bail decision is made
  explicit with a documented rationale (partial suppression is the only
  approach that actually guarantees no overlap); the outer-wall-always-
  prints rule is stated unambiguously (`wall_index >= 1` only).
