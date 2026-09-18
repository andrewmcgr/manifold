# Design: Infill-Aware Volume Audit (Plan F)

## Background

`crates/manifold-core/src/volume_audit.rs` (shipped by
`2026-09-18-extrusion-volume-audit`) computes, per grid cell, an
independently-derived expected volume from the raw mesh + `SlicerConfig`
via `expected_fill_fraction`, and compares it against actual accumulated
extrusion volume. For the sparse-infill interior, `expected_fill_fraction`
returns a flat `config.infill_density` sampled once at the cell center.

This flat nominal-density model mismatches real infill geometry badly
enough (measured ~1.46x at `cell_size = 2.0`, ~5.94x at `cell_size = 0.8`
on a healthy 20mm box; 3629/14558 non-empty cells already exceed 2.0x at
`cell_size = 0.8` with nothing wrong) that the resulting noise swamps any
real wall-shell duplication signal in the tool's global `assert_no_overfill`
assertion. Measured directly: the global max overfill ratio is identical
to 13 significant figures whether 0, 1, 95, or 193 inner-wall paths are
duplicated on real pipeline output, at every `cell_size` tried — because
the worst cell is always a pure sparse-infill cell carrying zero wall
material, so wall duplication never moves the global signal at all.

Per the backlog entry (Plan F) and follow-up discussion: overextrusion in
sparse infill is a lower priority than overextrusion in the solid
shell/top/bottom zones. The fix should stop sparse-infill noise from
polluting the solid-shell checks, while still giving infill *some*
coverage — just at a coarser, appropriately-scoped resolution.

## Chosen Approach

Stop comparing sparse-infill volume per fine cell against the flat
nominal-density model. Instead:

1. Classify *why* a cell has its expected fraction (not just the number),
   distinguishing `Solid` cells (wall-shell, top-facing, or bottom-facing
   — all currently `1.0`) from `SparseInfill` cells (interior,
   `config.infill_density`) from `Outside` cells (`0.0`).
2. Restrict the existing per-cell ratio queries (`overfilled_cells`,
   `underfilled_cells`, and therefore `assert_no_overfill`/
   `assert_no_underfill`) to `Solid` cells only. Sparse-infill cells never
   appear in these queries, so their periodic line/gap sampling noise can
   no longer swamp a real solid-shell defect.
3. Add one new, deliberately coarse, whole-grid aggregate check for
   sparse infill: total accumulated volume vs. total expected volume
   summed across every `SparseInfill`-zone cell, expressed as a single
   ratio. This catches gross infill under/over-deposition (e.g. "half the
   infill is missing", "infill deposited 3x over") without attempting
   per-cell localization, which the flat density model cannot support
   accurately at fine resolution.

This is a **comparison-granularity** fix, not a model-accuracy fix: the
flat nominal-density number itself is unchanged and still approximate for
infill. It is checked at a resolution where that approximation is valid
(the whole grid, or a large region, averages over many infill periods)
rather than at a resolution where it isn't (a single fine cell).

### Rejected Approaches

- **Reconstruct real infill-line geometry analytically** (compute expected
  fill from the actual line spacing/pattern/angle, independently of the
  pipeline) would fix the model itself and preserve per-cell localization
  for infill defects too. Rejected for this plan as disproportionate scope
  for a lower-priority defect class: it requires per-pattern geometric
  models (lines/grid/gyroid/etc.), z/layer-parity awareness for
  cross-hatch patterns, and carries real risk of drifting back toward
  reading pipeline-computed geometry if not built carefully independent of
  it. Left as a future, larger follow-up if infill accuracy becomes a
  priority.
- **Supersampling within each cell using the existing flat-density model**
  was rejected: averaging multiple samples of a spatially-constant value
  doesn't change anything when `cell_size` is comparable to or smaller
  than real infill line spacing — it doesn't address the actual mismatch.
- **Guidance-only (require `cell_size` large relative to infill line
  spacing)** was rejected as the weakest option: it narrows the regime
  where the existing bug is tolerable rather than fixing it, and real
  infill spacing can be several mm, which may force an impractically
  coarse grid for other purposes (wall-shell localization).

## Components

### `FillZone` (new)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillZone {
    Outside,
    Solid,
    SparseInfill,
}
```

Crate-private (not `pub`) — an internal classification used to route
grid cells to the right check, not a new public API surface by itself.

### `classify_fill_zone` (new, replaces `expected_fill_fraction`'s inline
branching)

```rust
fn classify_fill_zone(
    mesh_sdf: &MeshSdf,
    bed_excluded_sdf: &MeshSdf,
    p: DVec3,
    config: &SlicerConfig,
) -> FillZone
```

Contains exactly the branching `expected_fill_fraction` has today (mesh
membership, wall-shell zone, top-facing march, bottom-facing march), but
returns the zone classification instead of the numeric fraction directly.

`expected_fill_fraction` becomes a thin wrapper:

```rust
fn expected_fill_fraction(..., config: &SlicerConfig) -> f64 {
    match classify_fill_zone(..., config) {
        FillZone::Outside => 0.0,
        FillZone::Solid => 1.0,
        FillZone::SparseInfill => config.infill_density,
    }
}
```

**No behavioral change to `expected_fill_fraction`'s numeric output** —
this is a pure internal refactor. All four of its existing tests
(`expected_fill_fraction_is_zero_outside_the_mesh`,
`expected_fill_fraction_returns_wall_solid_near_a_face`,
`expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box`,
and the wall-shell-zone test that exercises the same branch) must pass
unmodified.

### `VolumeAuditGrid` (extended)

Add one field:

```rust
pub(crate) zone: Vec<FillZone>,
```

Populated in the same parallel pass (`audit_extrusion_volume`'s existing
`expected.par_iter_mut()` loop) that computes `expected`, using
`classify_fill_zone` directly (not derived from the `expected` value
after the fact — deriving it from the number would reintroduce exactly
the `expected == 0.0` ambiguity between `Outside` and
`SparseInfill`-at-zero-density this design exists to remove).

### `overfilled_cells` / `underfilled_cells` (changed filter)

Change the exclusion filter from `expected <= 0.0` to `zone[idx] !=
FillZone::Solid`. This:

- Excludes `SparseInfill` cells entirely (the actual fix).
- Excludes `Outside` cells (unchanged behavior — already excluded before,
  now via zone instead of the raw number).
- Cells with `expected == 1.0 * cell_size^3` (i.e. `Solid`) are the only
  ones considered, matching wall-shell/top-facing/bottom-facing exactly.

Doc comments updated to state the new, narrower scope plainly: these
queries (and their `assert_no_*` wrappers) now only ever report on
solid-shell zones (wall, top-facing, bottom-facing); sparse-infill
coverage is `infill_aggregate_ratio`/`assert_infill_volume_within` (below).

### `extrusion_outside_mesh_cells` (tightened, incidental improvement)

Change `expected[idx] == 0.0` to `zone[idx] == FillZone::Outside`. This
resolves the doc-noted wart from the previous plan (a `config.
infill_density == 0.0` interior cell no longer gets conflated with "outside
the mesh entirely" — they're now distinguishable via `zone` rather than
both collapsing to `expected == 0.0`). Doc comment's "Note that `expected
== 0` also arises..." caveat is removed since it no longer applies.

### `infill_aggregate_ratio` (new)

```rust
/// The ratio of total accumulated volume to total expected volume, summed
/// across every `SparseInfill`-zone cell in the grid. `None` if the grid
/// has no sparse-infill cells at all (e.g. `infill_density == 1.0`,
/// collapsing every interior cell to `Solid`, or a mesh too small to have
/// an interior).
///
/// Deliberately whole-grid and unlocalized: the flat nominal-density
/// expected model this ratio is checked against is only accurate in
/// aggregate, not per fine cell -- see [`VolumeAuditGrid::overfilled_cells`]
/// for why sparse infill isn't checked at that resolution. This catches
/// gross infill under/over-deposition, not a single duplicated infill
/// line.
pub fn infill_aggregate_ratio(&self) -> Option<f64>
```

### `assert_infill_volume_within` (new)

```rust
/// Panics if [`VolumeAuditGrid::infill_aggregate_ratio`] falls outside
/// `[min_fraction, max_ratio]`. No-op if the grid has no sparse-infill
/// cells (see `infill_aggregate_ratio`'s doc comment).
pub fn assert_infill_volume_within(&self, min_fraction: f64, max_ratio: f64)
```

### `assert_no_overfill` (doc comment rewritten)

The existing "Known limitation (measured, not hypothetical)" paragraph
describing the swamping is removed and replaced with a short note that
this assertion now only considers solid-shell cells (wall/top/bottom),
and that sparse-infill coverage is a separate, coarser check
(`infill_aggregate_ratio`/`assert_infill_volume_within`).

## Acceptance / Proof

The bar for this plan is not "tests still pass" — it's a direct
re-measurement proving the fix actually works, using the same
methodology the discovery used:

1. Re-run the real-pipeline wall-duplication test from the prior plan
   (`duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios`'s
   fixture: slice a real box, plan real toolpaths, duplicate every
   `WallInner` path) and confirm `assert_no_overfill` — the whole-grid
   assertion, not just the per-cell query — now actually panics on the
   duplicated case and does not panic on the healthy baseline, at the
   recommended `cell_size`. This is the concrete, falsifiable proof that
   restricting to `Solid` cells removed the swamping. If it does not hold,
   the design is wrong and needs revisiting before merge, not documenting
   around.
2. Confirm the new `infill_aggregate_ratio`/`assert_infill_volume_within`
   genuinely detects a gross infill defect (e.g. all infill paths
   duplicated) while not false-positiving on a healthy real-pipeline
   slice, using the same real-pipeline-fixture-plus-relative-comparison
   discipline established in the prior plan (no absolute-threshold
   fixtures at a `cell_size` this module's own docs say are unreliable).

## Global Constraints (carried forward, unchanged)

- `classify_fill_zone`/`expected_fill_fraction` remain grounded solely in
  `MeshSdf` + `SlicerConfig` — never `Layer`/`WallLoop`/`OrderField`/any
  pipeline output. This design does not touch that boundary; it only adds
  a classification of the *existing* independently-derived value.
- Core geometry: `glam::DVec3`/f64 only.
- Repo pre-commit gate: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy
  cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test
  cargo nextest run --workspace`.

## Self-Review

- **Placeholder scan:** No TBDs. The rejected-approaches section
  documents why A/C/D were not chosen, so a future reader isn't left
  wondering why the "more accurate" option wasn't picked.
- **Internal consistency:** `zone` is computed once, directly from
  `classify_fill_zone`, and consumed identically by
  `overfilled_cells`/`underfilled_cells` (require `Solid`) and
  `extrusion_outside_mesh_cells` (require `Outside`) and
  `infill_aggregate_ratio` (require `SparseInfill`) — every zone value has
  exactly one query family that reads it, no cell is silently excluded
  from all three.
- **Scope check:** Explicitly does not touch the flat nominal-density
  model's accuracy (Rejected Approaches, item A) — only where/how it's
  compared. Does not touch wall-shell/top-facing/bottom-facing detection
  logic at all (unchanged branches, just now returning an enum tag
  alongside their existing `1.0`).
- **Ambiguity check:** `infill_aggregate_ratio`'s `None` case (no
  sparse-infill cells) is stated explicitly, so a caller can't
  misinterpret "no sparse-infill zone exists" as "sparse infill is
  perfectly correct."

## Backlog (not designed yet, tracked for future plans)

- **Plan A' (analytic infill-line reconstruction)** — the rejected
  approach A above, if per-cell infill defect localization becomes a
  priority later. Would need its own design cycle: per-pattern geometric
  models, layer-parity handling for cross-hatch patterns, and careful
  verification it stays independent of pipeline-generated toolpaths.
