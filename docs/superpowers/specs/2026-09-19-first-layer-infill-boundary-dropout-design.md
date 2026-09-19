# Design: First-Layer Infill Boundary Dropout

## Background

Reported by the user via the volume-audit fill visualization
(`2026-09-18-volume-audit-fill-visualization`): slicing
`/Users/amcgregor/work/Manifold/TestObj1.stl` with the real saved profile
`/Users/amcgregor/3D/profile.json` (`order_field: AnisotropicFsm`,
`wall_line_width: 0.4`, `shell_thickness: 1.2` → `wall_count() == 3`,
`bottom_layers: 3`, `infill_density: 0.2`) produces a sliced object whose
first few layers are missing roughly half their interior fill. Confirmed
independently, not just visually:

- The volume audit (`crates/manifold-core/src/volume_audit.rs`) at
  `cell_size = 1.2` shows ~750 `Solid`-zone cells at `z < 3mm` reading
  0-9% of expected fill, tightly clustered at world `x ∈ [180, 183.6]`
  across nearly the full `y` range (`154..197`).
- `crates/manifold-cli/examples/probe_first_layer_dropout.rs` (new,
  committed alongside this design) confirmed the mechanism precisely:
  - **Walls are unaffected.** All 3 wall passes (`wall_index` 0, 1, 2)
    are present and correctly planned across the *entire* footprint,
    including `x > 178`, at layers 0-3.
  - **`Layer::infill_boundary` itself is confined to `x < 178`** at
    layers 0, 1, and 2 -- zero polygons touch `x > 178` at all, not
    merely classified sparse-vs-solid. At layer 3 it partially recovers
    (starts including the region), but `Layer::solid_fill_boundary`
    still doesn't.
  - **The mesh SDF confirms real geometry exists there**: sampling 20
    points along `x` from 178.5 at the affected `y`/`z` shows 20/20
    inside the mesh (`crates/manifold-fidget`'s `MeshSdf`, the same
    ground truth `volume_audit` itself uses).

This rules out a `compute_solid_fill_boundaries`
(`crates/manifold-core/src/slicing.rs:3959`) defect directly: that
function early-returns empty when handed an already-empty
`infill_boundary` (`if layer.infill_boundary.is_empty() { return (pos,
Vec::new()); }`), which is exactly what's happening -- it's correctly
propagating an already-wrong input, not itself the bug.

## What This Is Not

- **Not Plan B** (`docs/superpowers/specs/2026-09-17-perpendicular-top-distance-and-wall-overlap-design.md`'s
  backlog): Plan B describes *misclassification* (solid vs. sparse
  decided wrong for an entire layer via a single bbox-center sample).
  This defect is more severe -- the fillable region itself never gets
  computed for roughly half the footprint, so neither solid nor sparse
  infill exists there. Worth re-checking once this is fixed in case Plan
  B's symptom also resolves or needs re-scoping, but treat as a
  distinct, real defect for now.
- **Not a wall/perimeter bug.** Confirmed walls are complete and correct
  in the affected region at every layer checked.
- **Not (confirmed) a mesh geometry problem.** The mesh SDF says real
  solid geometry exists at the missing region -- the input is fine.

## Candidate Root-Cause Locations (not yet confirmed -- Task 1's job)

`infill_boundary` is derived from the wall loops via inward offsetting
in several places in `crates/manifold-core/src/slicing.rs`, all
candidates for where a real per-island region could be silently dropped:

- `polygon2d::inward_offset` calls around `slicing.rs:1463` and
  `slicing.rs:1661` -- the primary derivation of `infill_boundary` from
  the innermost printed wall loop's inward offset.
- The per-island "deepest wall" fallback logic around `slicing.rs:6649-6652`
  (`if let Some(deepest_wall) = partitioned.last() { ... }`) -- explicitly
  documented nearby (`slicing.rs:6612-6616`) as handling multi-island
  footprints where one island's inward offset "collapses to nothing" and
  must NOT silently vanish; this exact codepath is a strong candidate
  given the symptom is per-region, not per-layer.
- `clean_first_layer_geometry` (`slicing.rs:1774` area) -- filters
  first-layer infill-boundary slivers below
  `nozzle_diameter^2 * 2` using `filter_min_area`; scoped to layer 0
  only per its own doc comment, so it can explain layer 0's symptom but
  not layers 1-2 on its own unless something upstream feeds it a
  per-island split that then gets filtered.
- The "narrow region" / island-partitioning machinery referenced near
  `slicing.rs:2722-2803` (`suppress_close_redundant_loops`,
  CAD-groove-narrower-than-offset-distance bifurcation handling) --
  documented as producing "locally disjoint but individually
  order-field-correct sheets"; a bug here could plausibly cause one
  sheet/island to be dropped rather than kept.

None of these are confirmed as *the* cause yet -- Task 1's job is to
actually trace the real repro through these candidates (or find the
true cause elsewhere) and pin it down with evidence, the same
discipline `probe_first_layer_dropout.rs` already used to rule out walls
and `compute_solid_fill_boundaries`.

## Acceptance / Proof

The bar for this plan is a *minimized, committable regression test* that
fails before the fix and passes after -- not just re-running the probe
against the real 100KB STL (which stays as ad hoc evidence, not a
regression test). If a small synthetic mesh reproducing the same defect
class can be constructed, prefer that; if genuinely impractical (e.g.
the defect depends on `TestObj1.stl`'s specific triangulation in a way
that doesn't reduce), document why and add a targeted integration test
against the real file instead, following the precedent already
established by `manifold-cli`'s `diagnose_*`/`probe_*` scratch tools of
using real saved profiles/meshes when synthetic repro isn't practical.

Additionally, once fixed, re-run `probe_volume_audit_shell` against the
real `TestObj1.stl` + `profile.json` at `cell_size = 1.2` and confirm the
near-zero-fill cluster at `x ∈ [180, 183.6]`, `z < 3mm` is gone (or
reduced to what's geometrically expected -- the user separately noted
the object is under 1.2mm thick along its midline, so *some* cells near
there legitimately read below 100% even when correct; this proof step is
about the dense cluster of near-*zero* fill, not perfect 100% everywhere).

## Global Constraints

- Core geometry: `glam::DVec3`/f64 only, no `f32`/`Vec3`.
- Do not touch `compute_solid_fill_boundaries` unless Task 1's
  investigation proves it's actually implicated (current evidence says
  it isn't -- it's correctly propagating an already-empty input).
- Do not touch wall-loop generation/planning -- confirmed unaffected.
- Repo pre-commit gate: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy
  cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test
  cargo nextest run --workspace`.

## Self-Review

- **Placeholder scan:** the root cause is genuinely not yet known --
  stated plainly as Task 1's job, with a concrete, code-grounded
  candidate list rather than a vague "investigate and fix."
- **Internal consistency:** every claim in Background is backed by a
  specific probe run's output or a specific line reference, not
  assumption.
- **Scope check:** explicitly separated from Plan B (different defect
  class) and from wall/mesh-input concerns (both ruled out with
  evidence, not assumption).
