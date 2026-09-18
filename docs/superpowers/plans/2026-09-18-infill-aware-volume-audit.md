# Infill-Aware Volume Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the volume audit's known blind spot (Plan F): the flat nominal-density sparse-infill model swamps real wall-shell duplication signal in `assert_no_overfill`'s global assertion. Classify each grid cell's zone (`Outside`/`Solid`/`SparseInfill`), restrict the per-cell ratio checks to `Solid` cells only, and give sparse infill its own separate, deliberately coarse whole-grid aggregate check instead.

**Architecture:** A new `FillZone` classification (crate-private) makes explicit *why* `expected_fill_fraction` returned the value it did, computed once per cell in the same pass that already computes `expected`. `overfilled_cells`/`underfilled_cells` (and their `assert_no_*` wrappers) filter on `zone == Solid` instead of `expected > 0`, structurally excluding the noisy sparse-infill cells from the checks that need precision. A new `infill_aggregate_ratio`/`assert_infill_volume_within` pair gives sparse infill a separate, whole-grid-summed check at the resolution its flat model can actually support.

**Tech Stack:** Rust, `glam::DVec3`, `manifold_fidget::mesh_sdf::MeshSdf`, `manifold_fidget::ScalarField`, `rayon` (already workspace dependencies, no new ones).

**Spec:** `docs/superpowers/specs/2026-09-18-infill-aware-volume-audit-design.md`

## Global Constraints

- Core geometry uses `glam::DVec3` (f64) — no `f32`/`Vec3`.
- `classify_fill_zone`/`expected_fill_fraction` remain grounded solely in `MeshSdf` + `SlicerConfig` — never `Layer`, `WallLoop`, `solid_fill_boundary`, `infill_boundary`, or any `OrderField`. This plan only adds a classification of the *existing* independently-derived value; do not thread any pipeline output through it.
- `FillZone` is crate-private (not `pub`) — an internal routing classification, not new public API surface.
- No GPU/`wgpu` dependency in this plan. No GUI visualization.
- This repo's commit hook requires the subject line ≤72 characters in `<type>(<scope>): <subject>` format.
- After each task: `cargo fmt --all` -> `CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets` -> `CARGO_TARGET_DIR=target/test cargo nextest run --workspace` must all pass (per `AGENTS.md`'s current commands).
- Every re-tuned tolerance or threshold in this plan must be set from a real, observed measurement on this exact fixture (temporarily widen the bound, run the test, read the panic message's own reported worst cell/ratio, then hardcode a value comfortably past what was actually observed) — never guessed. This mirrors the exact technique the prior plan's `audit_extrusion_volume_passes_on_a_healthy_sliced_box` test and `assert_no_overfill_does_not_false_positive_but_panics_on_duplication` test already used; follow it, don't invent a new one.

---

### Task 1: Zone classification and restricting solid-shell checks

**Files:**
- Modify: `crates/manifold-core/src/volume_audit.rs` (this is the only file touched by this entire plan)

**Interfaces:**
- Produces (crate-private): `enum FillZone { Outside, Solid, SparseInfill }` (`#[derive(Debug, Clone, Copy, PartialEq, Eq)]`).
- Produces (private): `fn classify_fill_zone(mesh_sdf: &MeshSdf, bed_excluded_sdf: &MeshSdf, p: DVec3, config: &SlicerConfig) -> FillZone`.
- Produces (private): `fn fraction_for_zone(zone: FillZone, config: &SlicerConfig) -> f64`.
- Modifies: `expected_fill_fraction`'s body (signature unchanged: `fn expected_fill_fraction(mesh_sdf: &MeshSdf, bed_excluded_sdf: &MeshSdf, p: DVec3, config: &SlicerConfig) -> f64`) — becomes a thin wrapper over `classify_fill_zone` + `fraction_for_zone`. **Its numeric output must not change** — this task's own existing tests for `expected_fill_fraction` must pass unmodified.
- Modifies: `VolumeAuditGrid` struct — adds `pub(crate) zone: Vec<FillZone>` field.
- Modifies: `audit_extrusion_volume`'s expected-computation loop and its final struct literal to populate `zone`.
- Modifies: `overfilled_cells`, `underfilled_cells`, `extrusion_outside_mesh_cells` filters (from `expected`-based to `zone`-based) and their doc comments.
- Modifies: `assert_no_overfill`, `assert_no_underfill` doc comments (remove the now-obsolete "Known limitation" paragraph and swamping description).
- Modifies: the module-level (`//!`) doc comment at the top of the file (remove the "Note one measured limitation..." paragraph — no longer true after this task).
- Produces later tasks rely on: `FillZone::SparseInfill` variant and the `zone: Vec<FillZone>` field, both consumed by Task 2's `infill_aggregate_ratio`.

- [ ] **Step 1: Add `FillZone`, `classify_fill_zone`, `fraction_for_zone`, and refactor `expected_fill_fraction`**

In `crates/manifold-core/src/volume_audit.rs`, find the existing `expected_fill_fraction` function (it currently sits directly after `directional_march` and its doc comment). Replace the entire function (doc comment and body) with:

```rust
/// Which physical zone a world-space point falls into, used to route grid
/// cells to the right check in `VolumeAuditGrid`'s query methods. See
/// `expected_fill_fraction`'s doc comment for why this classification --
/// like the fraction it's derived from -- is computed only from `MeshSdf`
/// and `SlicerConfig`, never from pipeline outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FillZone {
    /// Outside the mesh entirely -- no material should ever land here.
    Outside,
    /// Wall-shell, top-facing, or bottom-facing solid material -- expected
    /// fill fraction is always `1.0`.
    Solid,
    /// Sparse interior infill -- expected fill fraction is
    /// `config.infill_density`, a flat nominal value that is only accurate
    /// in aggregate over many infill periods, not per fine cell -- see
    /// `VolumeAuditGrid::overfilled_cells`'s doc comment.
    SparseInfill,
}

/// Classifies which physical zone `p` falls into: `Outside` the mesh
/// entirely, `Solid` (wall-shell, top-facing, or bottom-facing material),
/// or `SparseInfill` (interior). See this function's own logic below --
/// identical to `expected_fill_fraction`'s previous inline branching,
/// factored out so `VolumeAuditGrid` can route cells by zone without
/// re-deriving it from the numeric fraction (which cannot distinguish
/// `Outside` from a `SparseInfill` cell at zero density -- both would
/// read `0.0`).
///
/// **Computed entirely from `mesh_sdf`/`bed_excluded_sdf` and `config` --
/// never from `Layer`, `WallLoop`, `solid_fill_boundary`,
/// `infill_boundary`, or any `OrderField`.** This is the load-bearing
/// design constraint this whole module exists to satisfy: those types
/// are themselves outputs of the layering pipeline this audit is meant
/// to check, so using them as "expected" ground truth would let a
/// layer-to-layer consistency bug in that pipeline corrupt both the
/// actual printed geometry and the expected baseline identically,
/// canceling out. See the design spec's "Central Design Constraint"
/// section. Do not "simplify" this function by threading a `&Layer` or
/// `&dyn OrderField` through it.
fn classify_fill_zone(
    mesh_sdf: &MeshSdf,
    bed_excluded_sdf: &MeshSdf,
    p: DVec3,
    config: &SlicerConfig,
) -> FillZone {
    if mesh_sdf.sample(p).value > 0.0 {
        return FillZone::Outside;
    }

    let wall_threshold = config.wall_offset + config.wall_count() as f64 * config.wall_line_width;
    if wall_shell_zone(mesh_sdf, p, wall_threshold) {
        return FillZone::Solid;
    }

    let step = (config.layer_height.min(config.nozzle_diameter) / 4.0).max(0.01);
    let max_search = (config.layer_height * 20.0).max(5.0);

    let top_threshold = config.top_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(bed_excluded_sdf, p, BUILD_DIRECTION, step, max_search) {
        if d <= top_threshold {
            return FillZone::Solid;
        }
    }

    let bottom_threshold = config.bottom_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(mesh_sdf, p, -BUILD_DIRECTION, step, max_search) {
        if d <= bottom_threshold {
            return FillZone::Solid;
        }
    }

    FillZone::SparseInfill
}

/// The expected fill fraction for a given `FillZone`: `0.0` for `Outside`,
/// `1.0` for `Solid`, `config.infill_density` for `SparseInfill`. The
/// single source of truth for this mapping, shared by
/// `expected_fill_fraction` and `audit_extrusion_volume`'s grid-building
/// loop so the two never drift apart.
fn fraction_for_zone(zone: FillZone, config: &SlicerConfig) -> f64 {
    match zone {
        FillZone::Outside => 0.0,
        FillZone::Solid => 1.0,
        FillZone::SparseInfill => config.infill_density,
    }
}

/// The expected fill fraction (`0.0`..`1.0`) at world-space point `p`:
/// `0.0` outside the mesh entirely, `1.0` within the wall-shell,
/// top-facing, or bottom-facing zones, else `config.infill_density`. A
/// thin wrapper over `classify_fill_zone` + `fraction_for_zone` -- see
/// `classify_fill_zone`'s doc comment for the zone boundaries and the
/// load-bearing SDF-only grounding constraint.
fn expected_fill_fraction(
    mesh_sdf: &MeshSdf,
    bed_excluded_sdf: &MeshSdf,
    p: DVec3,
    config: &SlicerConfig,
) -> f64 {
    fraction_for_zone(
        classify_fill_zone(mesh_sdf, bed_excluded_sdf, p, config),
        config,
    )
}
```

- [ ] **Step 2: Run the existing `expected_fill_fraction` tests to confirm no behavioral change**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::expected_fill_fraction`

Expected: all 3 pass unchanged (`expected_fill_fraction_is_zero_outside_the_mesh`, `expected_fill_fraction_returns_wall_solid_near_a_face`, `expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box`). Also run `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::wall_shell_zone_is_true` and the two `directional_march_matches_closed_form` tests — none of these should be affected by this step, confirming the refactor changed nothing observable yet.

- [ ] **Step 3: Add the `zone` field to `VolumeAuditGrid` and populate it in `audit_extrusion_volume`**

Find the `VolumeAuditGrid` struct definition:

```rust
pub struct VolumeAuditGrid {
    pub(crate) origin: DVec3,
    pub(crate) cell_size: f64,
    pub(crate) dims: [usize; 3],
    pub(crate) accumulated: HashMap<VolumeKindBucket, Vec<f64>>,
    pub(crate) expected: Vec<f64>,
}
```

Add the new field after `expected`:

```rust
pub struct VolumeAuditGrid {
    pub(crate) origin: DVec3,
    pub(crate) cell_size: f64,
    pub(crate) dims: [usize; 3],
    pub(crate) accumulated: HashMap<VolumeKindBucket, Vec<f64>>,
    pub(crate) expected: Vec<f64>,
    pub(crate) zone: Vec<FillZone>,
}
```

Find the expected-computation loop inside `audit_extrusion_volume`:

```rust
    use rayon::prelude::*;
    let mut expected = vec![0.0f64; cell_count];
    expected.par_iter_mut().enumerate().for_each(|(idx, e)| {
        let k = idx / (dims[0] * dims[1]);
        let j = (idx / dims[0]) % dims[1];
        let i = idx % dims[0];
        let p = origin
            + DVec3::new(
                (i as f64 + 0.5) * cell_size,
                (j as f64 + 0.5) * cell_size,
                (k as f64 + 0.5) * cell_size,
            );
        *e = expected_fill_fraction(&mesh_sdf, &bed_excluded_sdf, p, config) * cell_size.powi(3);
    });
```

Replace it with:

```rust
    use rayon::prelude::*;
    let mut expected = vec![0.0f64; cell_count];
    let mut zone = vec![FillZone::Outside; cell_count];
    expected
        .par_iter_mut()
        .zip(zone.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (e, z))| {
            let k = idx / (dims[0] * dims[1]);
            let j = (idx / dims[0]) % dims[1];
            let i = idx % dims[0];
            let p = origin
                + DVec3::new(
                    (i as f64 + 0.5) * cell_size,
                    (j as f64 + 0.5) * cell_size,
                    (k as f64 + 0.5) * cell_size,
                );
            let classification = classify_fill_zone(&mesh_sdf, &bed_excluded_sdf, p, config);
            *z = classification;
            *e = fraction_for_zone(classification, config) * cell_size.powi(3);
        });
```

Find the `VolumeAuditGrid` struct literal at the end of `audit_extrusion_volume`:

```rust
    VolumeAuditGrid {
        origin,
        cell_size,
        dims,
        accumulated,
        expected,
    }
}
```

Replace with:

```rust
    VolumeAuditGrid {
        origin,
        cell_size,
        dims,
        accumulated,
        expected,
        zone,
    }
}
```

- [ ] **Step 4: Restrict `overfilled_cells`/`underfilled_cells` to `Solid`-zone cells and update their doc comments**

Find `overfilled_cells`:

```rust
    /// `(cell index, ratio)` for every cell where total accumulated
    /// volume (summed across all buckets) exceeds `expected * max_ratio`.
    ///
    /// Only cells with `expected > 0` are considered: a ratio against zero
    /// expected volume is not a meaningful multiple, so material deposited
    /// entirely outside the mesh is a separate, binary defect with no
    /// tolerance knob to tune -- see
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`] for that case. The
    /// two queries partition the grid between them: this one (and
    /// [`VolumeAuditGrid::underfilled_cells`]) covers tunable-tolerance
    /// defects within real geometry, that one covers extrusion into open
    /// air.
    pub fn overfilled_cells(&self, max_ratio: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                let expected = self.expected[idx];
                if expected <= 0.0 {
                    return None;
                }
                let ratio = self.total_accumulated(idx) / expected;
                (ratio > max_ratio).then(|| (self.unflatten(idx), ratio))
            })
            .collect()
    }
```

Replace with:

```rust
    /// `(cell index, ratio)` for every `Solid`-zone cell (wall-shell,
    /// top-facing, or bottom-facing material) where total accumulated
    /// volume (summed across all buckets) exceeds `expected * max_ratio`.
    ///
    /// Only `Solid`-zone cells are considered. Two other zones are
    /// deliberately excluded, each for a different reason:
    /// - `Outside`-zone cells (material deposited entirely outside the
    ///   mesh) are a separate, binary defect with no tolerance knob to
    ///   tune -- see [`VolumeAuditGrid::extrusion_outside_mesh_cells`].
    /// - `SparseInfill`-zone cells are excluded because the flat
    ///   nominal-density expected model for infill is only accurate in
    ///   aggregate over many infill periods, not per fine cell: mixing
    ///   them into this per-cell query would swamp genuine solid-shell
    ///   defects in sampling noise (measured, before this exclusion
    ///   existed: a healthy print's worst cell by this metric was always
    ///   a sparse-infill artifact, unrelated to any real defect, which
    ///   made this query and [`VolumeAuditGrid::assert_no_overfill`] blind
    ///   to real wall-shell duplication). See
    ///   [`VolumeAuditGrid::infill_aggregate_ratio`] for infill's own
    ///   coarser, whole-grid check instead.
    pub fn overfilled_cells(&self, max_ratio: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                if self.zone[idx] != FillZone::Solid {
                    return None;
                }
                let ratio = self.total_accumulated(idx) / self.expected[idx];
                (ratio > max_ratio).then(|| (self.unflatten(idx), ratio))
            })
            .collect()
    }
```

Find `underfilled_cells`:

```rust
    /// `(cell index, fraction)` for every cell where total accumulated
    /// volume is below `expected * min_fraction`, among cells with
    /// `expected > 0`.
    pub fn underfilled_cells(&self, min_fraction: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                let expected = self.expected[idx];
                if expected <= 0.0 {
                    return None;
                }
                let fraction = self.total_accumulated(idx) / expected;
                (fraction < min_fraction).then(|| (self.unflatten(idx), fraction))
            })
            .collect()
    }
```

Replace with:

```rust
    /// `(cell index, fraction)` for every `Solid`-zone cell where total
    /// accumulated volume is below `expected * min_fraction`. See
    /// [`VolumeAuditGrid::overfilled_cells`] for why `SparseInfill`- and
    /// `Outside`-zone cells are excluded.
    pub fn underfilled_cells(&self, min_fraction: f64) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter_map(|idx| {
                if self.zone[idx] != FillZone::Solid {
                    return None;
                }
                let fraction = self.total_accumulated(idx) / self.expected[idx];
                (fraction < min_fraction).then(|| (self.unflatten(idx), fraction))
            })
            .collect()
    }
```

- [ ] **Step 5: Tighten `extrusion_outside_mesh_cells` to zone-based, resolving the known conflation wart**

Find:

```rust
    /// Cell indices holding accumulated volume where *no* material is
    /// expected at all (`expected == 0`) -- extrusion into open air.
    ///
    /// Deliberately binary rather than ratio-based: with zero expected
    /// volume there is no meaningful multiple to compare against, so unlike
    /// [`VolumeAuditGrid::overfilled_cells`] this takes no tolerance
    /// argument. Any material here is a defect.
    ///
    /// Note that `expected == 0` also arises for interior cells when
    /// `config.infill_density` is `0.0`, in which case sparse-interior
    /// extrusion is reported here too -- correctly, in the sense that no
    /// material was expected there either, though the cause is a zero
    /// density rather than being outside the mesh.
    pub fn extrusion_outside_mesh_cells(&self) -> Vec<[usize; 3]> {
        (0..self.cell_count())
            .filter(|&idx| self.expected[idx] == 0.0 && self.total_accumulated(idx) > 0.0)
            .map(|idx| self.unflatten(idx))
            .collect()
    }
```

Replace with:

```rust
    /// Cell indices holding accumulated volume in a cell classified
    /// `Outside` the mesh entirely -- extrusion into open air.
    ///
    /// Deliberately binary rather than ratio-based: outside the mesh there
    /// is no meaningful multiple to compare against, so unlike
    /// [`VolumeAuditGrid::overfilled_cells`] this takes no tolerance
    /// argument. Any material here is a defect.
    ///
    /// Zone-based rather than `expected == 0.0`-based: a `SparseInfill`
    /// cell with `config.infill_density == 0.0` also has `expected == 0.0`
    /// but is a different situation entirely (a configured zero-density
    /// interior, not "outside the mesh") -- see
    /// [`VolumeAuditGrid::infill_aggregate_ratio`] for that case instead.
    pub fn extrusion_outside_mesh_cells(&self) -> Vec<[usize; 3]> {
        (0..self.cell_count())
            .filter(|&idx| self.zone[idx] == FillZone::Outside && self.total_accumulated(idx) > 0.0)
            .map(|idx| self.unflatten(idx))
            .collect()
    }
```

- [ ] **Step 6: Update `assert_no_overfill`/`assert_no_underfill` doc comments**

Find `assert_no_overfill`:

```rust
    /// Panics, naming the single worst offending cell (highest ratio),
    /// if any cell's accumulated volume exceeds `expected * max_ratio`.
    ///
    /// **Known limitation (measured, not hypothetical):** this whole-grid
    /// assertion is dominated by the sparse-infill expected-volume model
    /// (nominal `infill_density * cell_size^3`, sampled once at the cell
    /// center), which mismatches real infill geometry badly enough to swamp
    /// genuine defects elsewhere. On a healthy sliced 20mm box the worst
    /// cell is always a pure sparse-infill cell -- 1.46x at `cell_size` 2.0,
    /// 5.94x at 0.8, where 3629 of 14558 non-empty cells already exceed 2.0x
    /// with nothing wrong. Because that worst cell contains no wall
    /// material, duplicating wall paths does not move this assertion's
    /// signal at all: the global max ratio is unchanged to 13 significant
    /// figures whether 0, 1, 95, or all 193 wall paths are duplicated.
    ///
    /// So do not rely on `assert_no_overfill` alone to catch wall-shell
    /// duplication on real prints. To detect that class, compare
    /// [`VolumeAuditGrid::overfilled_cells`] per-cell against a known-good
    /// baseline (see this module's
    /// `duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios`
    /// test), or restrict analysis to non-`Infill` buckets via
    /// [`VolumeAuditGrid::accumulated_volume`].
    pub fn assert_no_overfill(&self, max_ratio: f64) {
```

Replace the doc comment (keep the function body and signature unchanged) with:

```rust
    /// Panics, naming the single worst offending cell (highest ratio),
    /// if any `Solid`-zone cell's (wall-shell, top-facing, or
    /// bottom-facing) accumulated volume exceeds `expected * max_ratio`.
    /// See [`VolumeAuditGrid::overfilled_cells`] for why `SparseInfill`-
    /// and `Outside`-zone cells are excluded, and
    /// [`VolumeAuditGrid::infill_aggregate_ratio`] /
    /// [`VolumeAuditGrid::assert_infill_volume_within`] for sparse
    /// infill's separate, coarser coverage.
    pub fn assert_no_overfill(&self, max_ratio: f64) {
```

Find `assert_no_underfill`:

```rust
    /// Panics, naming the single worst offending cell (lowest fraction),
    /// if any cell's accumulated volume is below `expected * min_fraction`.
    pub fn assert_no_underfill(&self, min_fraction: f64) {
```

Replace with:

```rust
    /// Panics, naming the single worst offending cell (lowest fraction),
    /// if any `Solid`-zone cell's accumulated volume is below `expected *
    /// min_fraction`. See [`VolumeAuditGrid::underfilled_cells`] for why
    /// `SparseInfill`- and `Outside`-zone cells are excluded.
    pub fn assert_no_underfill(&self, min_fraction: f64) {
```

- [ ] **Step 7: Update the module-level doc comment**

Find, at the very top of the file:

```rust
//! Independent extrusion-volume audit: accumulates actual extruded volume
//! from a planned toolpath into a coarse 3D grid, computes an independent
//! expected volume for the same grid directly from the raw input mesh
//! (never from `Layer`/`OrderField` outputs -- see this module's own
//! `expected_fill_fraction` doc comment for why), and flags cells where
//! the two diverge. See
//! `docs/superpowers/specs/2026-09-18-extrusion-volume-audit-design.md`
//! for the full design rationale.
//!
//! Note one measured limitation before relying on the whole-grid
//! assertions: the sparse-infill expected-volume model can mismatch real
//! geometry by ~1.5x-6x depending on `cell_size`, which is enough to swamp
//! a real wall-duplication defect in the global max-ratio signal. See
//! [`VolumeAuditGrid::assert_no_overfill`] for the measured numbers and
//! what to use instead.
```

Replace with:

```rust
//! Independent extrusion-volume audit: accumulates actual extruded volume
//! from a planned toolpath into a coarse 3D grid, computes an independent
//! expected volume for the same grid directly from the raw input mesh
//! (never from `Layer`/`OrderField` outputs -- see this module's own
//! `expected_fill_fraction` doc comment for why), and flags cells where
//! the two diverge. See
//! `docs/superpowers/specs/2026-09-18-extrusion-volume-audit-design.md`
//! for the full design rationale.
//!
//! Solid-shell material (wall-shell, top-facing, bottom-facing) and
//! sparse interior infill are checked separately, at different
//! resolutions: [`VolumeAuditGrid::overfilled_cells`]/
//! [`VolumeAuditGrid::underfilled_cells`] (and their `assert_no_*`
//! wrappers) are precise per-cell checks restricted to solid-shell
//! material; [`VolumeAuditGrid::infill_aggregate_ratio`]/
//! [`VolumeAuditGrid::assert_infill_volume_within`] give sparse infill a
//! deliberately coarser, whole-grid check instead, since its flat
//! nominal-density expected model is only accurate in aggregate. See
//! `docs/superpowers/specs/2026-09-18-infill-aware-volume-audit-design.md`
//! for why.
```

(`infill_aggregate_ratio` and `assert_infill_volume_within` don't exist until Task 2 — this doc comment will reference them before they're defined, which is fine, Rust doc comments don't need forward-declared targets to compile, but note this for your own orientation: Task 2 adds those two functions.)

- [ ] **Step 8: Run the full test suite, observe expected failures**

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit`

Expected: two tests now fail (this is the direct, expected consequence of Steps 4-5, not a bug in this step):
- `audit_extrusion_volume_passes_on_a_healthy_sliced_box` — its tolerances (`2.0`/`0.25`) were tuned to a worst-case ratio that came from a sparse-infill cell, which no longer appears in `overfilled_cells`/`underfilled_cells` at all. The new worst case (now necessarily a `Solid`-zone cell) needs fresh measurement.
- `duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios` — its final assertion block explicitly asserts that `assert_no_overfill`'s global signal is *blind* to wall duplication (`(defective_max - healthy_max).abs() < 1e-9`). That assertion should now fail, because the fix in Steps 4-5 removed the sparse-infill cell that was causing the blindness. **A failure here is the fix working as designed** — Step 10 below rewrites this test's assertions accordingly.

Everything else in the module (all `expected_fill_fraction`/`classify_fill_zone`-adjacent tests, the wall/infill bucket-separation test, both hand-built duplication-detection tests, both outside-the-mesh tests, and the cell-decoding test) should still pass unchanged — none of their fixtures touch a `SparseInfill`-zone cell.

- [ ] **Step 9: Re-measure and re-tune `audit_extrusion_volume_passes_on_a_healthy_sliced_box`**

Find the test's assertions:

```rust
        let grid = audit_extrusion_volume(&mesh, &paths, &config, 2.0);

        // Tolerances tuned from real observed output on this exact
        // fixture (measured directly by temporarily setting both bounds to
        // 1.0 and reading the panic message's own reported worst cell):
        // worst observed overfill ratio 1.46x, worst observed underfill
        // fraction 0.32. Both bounds below are set comfortably past those
        // measured values, not guessed -- `assert_no_overfill`/
        // `assert_no_underfill` name the exact offending cell and its
        // exact ratio/fraction in their own panic message if either ever
        // regresses past these margins.
        grid.assert_no_overfill(2.0);
        grid.assert_no_underfill(0.25);
```

Temporarily change the two calls to `grid.assert_no_overfill(1.0)` and `grid.assert_no_underfill(1.0)`, run `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::audit_extrusion_volume_passes_on_a_healthy_sliced_box --no-capture`, and read the panic message (it names the exact worst cell and its exact ratio/fraction, same format as before). Record both observed numbers.

Restore the two calls to real, non-`1.0` values set comfortably past what you observed (follow the same margin discipline as the original comment: the prior measurement used a >35% margin above the observed overfill ratio and a >20% margin below the observed underfill fraction — use your own judgment for a comparable margin against your new numbers, but do not set either bound so tight it would flag on ordinary floating-point noise between runs). Update the comment to record your new measured numbers in the same style as the one being replaced (do not leave the old 1.46x/0.32 numbers in the comment — they no longer apply to a `Solid`-zone-only worst case).

- [ ] **Step 10: Rewrite `duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios` and add a companion test proving `assert_no_overfill` now catches the defect**

Find the entire existing test (from its `#[test]` attribute through its closing `}`) — it currently ends with:

```rust
        // Pin the limitation itself, so it cannot regress silently in either
        // direction. If the sparse-infill expected model is ever corrected
        // (tracked as a follow-up), this assertion is expected to start
        // failing -- at which point the global `assert_no_overfill` may
        // finally be able to catch this defect class and this test, plus
        // `assert_no_overfill`'s own limitation note, should be revisited.
        let global_max = |g: &VolumeAuditGrid| {
            g.overfilled_cells(0.0)
                .into_iter()
                .map(|(_, r)| r)
                .fold(0.0f64, f64::max)
        };
        let healthy_max = global_max(&baseline);
        let defective_max = global_max(&defective);
        assert!(
            (defective_max - healthy_max).abs() < 1e-9,
            "the global max ratio is expected to be blind to wall duplication (both ~1.4610, \
             set by a pure sparse-infill cell): healthy {healthy_max}, defective \
             {defective_max}. If these now differ, the sparse-infill expected-volume model may \
             have been fixed -- revisit assert_no_overfill's documented limitation."
        );
    }
```

This is exactly the "tracked as a follow-up" moment the old comment anticipated. Delete this trailing block (from `// Pin the limitation itself` through the `assert!` that closes it, but keep the test's closing `}`), and also update the test's own header doc comment, which currently reads:

```rust
        // Every other duplication test on this module uses hand-built
        // `Path`s at a `cell_size` below the recommended lower bound. This
        // one injects the defect into REAL pipeline output -- slice a box,
        // plan its toolpaths, then duplicate every planned inner-wall path --
        // and audits at `cell_size = 2.0`, the same recommended setting the
        // golden-path test uses.
        //
        // It deliberately asserts on the PER-CELL query, not on
        // `assert_no_overfill`, because the global assertion provably cannot
        // see this defect. Measured on exactly this fixture: the healthy
        // print's worst cell is a pure sparse-infill cell (cell [6,9,5],
        // world (11,17,9), expected 1.6mm^3 from nominal density, actual
        // infill 2.34mm^3, wall 0.0), so it carries no wall material and
        // duplicating walls leaves it untouched -- the global max ratio stays
        // 1.4609534807622 whether 0, 1, 95, or all 193 wall paths are
        // duplicated. That is a limitation of the sparse-infill expected
        // model, documented on `assert_no_overfill`, not of the accumulation
        // pass: the per-cell signal below is clean and strong.
```

Replace it with:

```rust
        // Every other duplication test on this module uses hand-built
        // `Path`s at a `cell_size` below the recommended lower bound. This
        // one injects the defect into REAL pipeline output -- slice a box,
        // plan its toolpaths, then duplicate every planned inner-wall path --
        // and audits at `cell_size = 2.0`, the same recommended setting the
        // golden-path test uses.
        //
        // Asserts on the PER-CELL query directly (rather than the global
        // `assert_no_overfill`) because comparing each cell against its own
        // healthy baseline cancels sampling artifacts both runs share,
        // leaving only the injected duplication -- a cleaner signal than a
        // single global max. See the companion test below for the
        // equivalent proof via `assert_no_overfill` itself.
```

Keep everything else in the test body (mesh/config/slicing/planning setup, the `baseline`/`baseline_ratios` computation, the duplication injection, and the `worst_increase >= 1.3` assertion) exactly as-is.

Now add a new test directly after it, in the same file, using the same fixture-building pattern:

```rust
    #[test]
    fn assert_no_overfill_catches_wall_duplication_on_real_pipeline_output() {
        // Companion to `duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios`
        // above, proving the same defect is now caught by the public,
        // panic-based `assert_no_overfill` API directly -- not just the
        // per-cell query -- now that `Solid`-zone restriction (this task)
        // keeps sparse-infill sampling noise from swamping the signal.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let config = SlicerConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            shell_thickness: 0.8,
            top_layers: 3,
            bottom_layers: 3,
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let tool_id = crate::ids::ToolId(0);
        let object = crate::object::Object::new(crate::ids::ObjectId(0), mesh.clone(), tool_id);
        let tool = crate::tool::Tool::new(tool_id, config.nozzle_diameter);
        let layers = crate::slicing::slice_object(&object, &config)
            .expect("slicing a plain box must succeed");
        let paths = crate::toolpath::plan(
            &layers,
            std::slice::from_ref(&object),
            std::slice::from_ref(&tool),
            &config,
        )
        .expect("planning toolpaths for a plain box must succeed");

        let cell_size = 2.0;
        let mut injected = paths.clone();
        let duplicated: Vec<_> = paths
            .iter()
            .filter(|p| p.segments.iter().any(|s| s.kind == MoveKind::WallInner))
            .cloned()
            .collect();
        assert!(
            !duplicated.is_empty(),
            "the planner should emit inner wall paths for a 20mm box -- nothing to duplicate"
        );
        injected.extend(duplicated);

        // TODO(implementer): measure the actual healthy/defective global max
        // overfill ratios on this fixture (see this file's
        // `assert_no_overfill_does_not_false_positive_but_panics_on_duplication`
        // test for the exact technique: print both via the `global_max`
        // closure pattern, or temporarily call `assert_no_overfill` with a
        // very large bound on each grid and read nothing -- then narrow it
        // until you observe where each one starts panicking). Pick a
        // `max_ratio` strictly between the two observed numbers, following
        // the same negative/positive control pairing as
        // `assert_no_overfill_does_not_false_positive_but_panics_on_duplication`:
        let max_ratio = /* fill in from measurement */;

        let baseline = audit_extrusion_volume(&mesh, &paths, &config, cell_size);
        // Negative control: the healthy baseline must NOT trip the
        // assertion. If this panics, the test fails here.
        baseline.assert_no_overfill(max_ratio);

        // Positive control: the same box with every inner wall path
        // duplicated MUST trip it.
        let defective = audit_extrusion_volume(&mesh, &injected, &config, cell_size);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            defective.assert_no_overfill(max_ratio);
        }));
        let payload = panicked.expect_err(
            "assert_no_overfill must now catch wall duplication on real pipeline output -- if \
             this doesn't panic, the Solid-zone restriction did not fix the swamping",
        );
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("<non-string panic payload>");
        assert!(
            message.contains("extrusion volume audit"),
            "panic message should identify itself as an extrusion volume audit failure, got: {message}"
        );
    }
```

Replace the `TODO`/placeholder `max_ratio` line with a real measured value before this step is done — this is a required measurement, not a step you may skip or leave as a literal `TODO` in committed code. Use the same technique as Step 9: temporarily set `max_ratio` to something very large (e.g. `100.0`) so neither call panics, add a `println!("baseline: {:?}", baseline.overfilled_cells(0.0).into_iter().map(|(_, r)| r).fold(0.0f64, f64::max));` (and the same for `defective`) right before the assertions, run with `--no-capture`, read both printed maxima, then delete the `println!`s and hardcode `max_ratio` strictly between them (same margin discipline as Step 9 and as `assert_no_overfill_does_not_false_positive_but_panics_on_duplication`'s existing `max_ratio = 3.0`). Update the comment above `max_ratio` to record what you measured, in the same style as that existing test's comment.

- [ ] **Step 11: Run the full pre-commit gate and commit**

Run, in order:
```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All three must be clean (fmt: no diff; clippy: zero warnings/errors from workspace crates, the pre-existing unrelated `block v0.1.6` future-incompat notice is fine; nextest: all tests pass, including the two you fixed in Steps 9-10 and the new one from Step 10).

Commit:
```bash
git add crates/manifold-core/src/volume_audit.rs
git commit -m "feat(core): classify fill zones, stop infill swamping wall checks"
```

---

### Task 2: Coarse whole-grid infill aggregate check

**Files:**
- Modify: `crates/manifold-core/src/volume_audit.rs` (same single file)

**Interfaces:**
- Consumes: `FillZone::SparseInfill` and the `zone: Vec<FillZone>` field from Task 1 (both already exist on `main`/this branch by the time this task starts).
- Produces: `pub fn infill_aggregate_ratio(&self) -> Option<f64>` on `impl VolumeAuditGrid`.
- Produces: `pub fn assert_infill_volume_within(&self, min_fraction: f64, max_ratio: f64)` on `impl VolumeAuditGrid`.

- [ ] **Step 1: Add `infill_aggregate_ratio` and `assert_infill_volume_within`**

In `crates/manifold-core/src/volume_audit.rs`, inside `impl VolumeAuditGrid`, find `assert_no_extrusion_outside_mesh` (the last method in the `impl` block, just before the block's closing `}`). Add the two new methods directly after it, before the `impl` block's closing `}`:

```rust

    /// The ratio of total accumulated volume to total expected volume,
    /// summed across every `SparseInfill`-zone cell in the grid. `None` if
    /// the grid has no `SparseInfill`-zone cells with nonzero expected
    /// volume at all (e.g. `infill_density == 1.0`, collapsing every
    /// interior cell to `Solid`; a mesh too small to have an interior; or
    /// `infill_density == 0.0`, where every `SparseInfill` cell's own
    /// expected volume is itself `0.0` and this ratio is undefined -- see
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`]'s doc comment for
    /// why that specific zero-density case is a known, separate gap this
    /// method does not cover).
    ///
    /// Deliberately whole-grid and unlocalized: the flat nominal-density
    /// expected model this ratio is checked against is only accurate in
    /// aggregate, not per fine cell -- see
    /// [`VolumeAuditGrid::overfilled_cells`] for why sparse infill isn't
    /// checked at that resolution. This catches gross infill
    /// under/over-deposition, not a single duplicated infill line.
    pub fn infill_aggregate_ratio(&self) -> Option<f64> {
        let (accumulated, expected) = (0..self.cell_count())
            .filter(|&idx| self.zone[idx] == FillZone::SparseInfill)
            .fold((0.0f64, 0.0f64), |(acc, exp), idx| {
                (acc + self.total_accumulated(idx), exp + self.expected[idx])
            });
        if expected <= 0.0 {
            return None;
        }
        Some(accumulated / expected)
    }

    /// Panics if [`VolumeAuditGrid::infill_aggregate_ratio`] falls outside
    /// `[min_fraction, max_ratio]`. No-op (does not panic) if
    /// `infill_aggregate_ratio` returns `None` -- see its doc comment for
    /// when that happens.
    pub fn assert_infill_volume_within(&self, min_fraction: f64, max_ratio: f64) {
        let Some(ratio) = self.infill_aggregate_ratio() else {
            return;
        };
        if ratio < min_fraction || ratio > max_ratio {
            panic!(
                "extrusion volume audit: aggregate sparse-infill volume is {ratio:.2}x its \
                 expected total (allowed range {min_fraction:.2}x-{max_ratio:.2}x)"
            );
        }
    }
```

- [ ] **Step 2: Add a hand-built unit test proving `infill_aggregate_ratio`'s arithmetic directly**

Add this test in the `mod tests` block, after `assert_no_extrusion_outside_mesh_passes_when_every_bead_is_inside` (or any convenient location alongside the other hand-built-fixture tests):

```rust
    #[test]
    fn infill_aggregate_ratio_sums_across_the_whole_grid_not_per_cell() {
        // Tall enough that a genuine interior sparse-infill zone exists,
        // same shape as `expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box`.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig {
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let cell_size = 2.0;
        let bead_area = config.infill_line_width * config.layer_height;
        // A short infill bead centered deep in the interior (20,20,20 is
        // >15mm from every face of this 40mm box, well beyond
        // wall_shell/top_layers/bottom_layers -- squarely SparseInfill).
        let infill_path = straight_extruding_path(
            DVec3::new(19.0, 20.0, 20.0),
            DVec3::new(21.0, 20.0, 20.0),
            MoveKind::Infill,
            bead_area,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&infill_path), &config, cell_size);

        let ratio = grid
            .infill_aggregate_ratio()
            .expect("a 40mm box's interior must contain SparseInfill-zone cells");

        // Manually compute the same ratio a different way (summing
        // `accumulated_volume`/`expected` per reported cell rather than
        // relying on the grid's own internal fold) to catch a
        // sign/indexing bug in `infill_aggregate_ratio`'s implementation
        // that summing over the exact same cells the same way could not.
        let bead_volume = infill_path.segments[0].extrusion_length
            * crate::extrusion::filament_cross_section_area(config.filament_diameter);
        // No wall/top/bottom material is present anywhere in this fixture,
        // so the reported total accumulated volume across ALL SparseInfill
        // cells must equal the single bead's own volume exactly (within
        // floating-point tolerance), since accumulation conserves volume
        // exactly by construction.
        let total_infill_accumulated: f64 = (0..grid.dims[0] * grid.dims[1] * grid.dims[2])
            .filter(|&idx| grid.zone[idx] == FillZone::SparseInfill)
            .map(|idx| grid.accumulated[&VolumeKindBucket::Infill][idx])
            .sum();
        assert!(
            (total_infill_accumulated - bead_volume).abs() < bead_volume * 0.01,
            "total accumulated infill volume across SparseInfill cells {total_infill_accumulated} \
             should match the bead's own volume {bead_volume} within 1%"
        );
        assert!(
            ratio > 0.0,
            "a real bead's worth of infill volume should register a nonzero ratio, got {ratio}"
        );
    }

    #[test]
    fn infill_aggregate_ratio_is_none_when_no_sparse_infill_cells_exist() {
        // A box small enough that every interior point falls within the
        // wall-shell/top/bottom thresholds -- no SparseInfill zone exists
        // at all, so the ratio must be `None`, not a division-by-zero
        // artifact or a silently-wrong `0.0`/`1.0`.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0));
        let config = SlicerConfig::default();
        let grid = audit_extrusion_volume(&mesh, &[], &config, 0.5);
        assert_eq!(
            grid.infill_aggregate_ratio(),
            None,
            "a box entirely covered by Solid zones should have no SparseInfill cells at all"
        );
        // Must not panic either -- a `None` ratio is a no-op, not a defect.
        grid.assert_infill_volume_within(0.5, 1.5);
    }
```

Run: `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::infill_aggregate_ratio`

Expected: both pass. If `infill_aggregate_ratio_is_none_when_no_sparse_infill_cells_exist` fails because the 1mm box's config still leaves a `SparseInfill` cell somewhere (e.g. `SlicerConfig::default()`'s wall/top/bottom thresholds don't cover the whole 1mm interior at whatever `cell_size` the grid uses), pick a smaller mesh, larger `wall_offset`/`wall_count`, or fewer `top_layers`/`bottom_layers` in that test's `config` until the assertion genuinely holds — don't weaken the assertion itself.

- [ ] **Step 3: Add a real-pipeline test proving the aggregate check detects gross infill duplication without false-positiving on a healthy print**

Add this test after the two from Step 2, following the same real-pipeline pattern as `duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios`/`assert_no_overfill_catches_wall_duplication_on_real_pipeline_output` from Task 1, but duplicating infill paths instead of wall paths, and asserting via `infill_aggregate_ratio`/`assert_infill_volume_within` instead:

```rust
    #[test]
    fn assert_infill_volume_within_catches_duplicated_infill_on_real_pipeline_output() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let config = SlicerConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            shell_thickness: 0.8,
            top_layers: 3,
            bottom_layers: 3,
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let tool_id = crate::ids::ToolId(0);
        let object = crate::object::Object::new(crate::ids::ObjectId(0), mesh.clone(), tool_id);
        let tool = crate::tool::Tool::new(tool_id, config.nozzle_diameter);
        let layers = crate::slicing::slice_object(&object, &config)
            .expect("slicing a plain box must succeed");
        let paths = crate::toolpath::plan(
            &layers,
            std::slice::from_ref(&object),
            std::slice::from_ref(&tool),
            &config,
        )
        .expect("planning toolpaths for a plain box must succeed");

        let cell_size = 2.0;
        let baseline = audit_extrusion_volume(&mesh, &paths, &config, cell_size);
        let baseline_ratio = baseline
            .infill_aggregate_ratio()
            .expect("a 20mm box with infill_density 0.2 must have SparseInfill cells");

        let mut injected = paths.clone();
        let duplicated: Vec<_> = paths
            .iter()
            .filter(|p| p.segments.iter().any(|s| s.kind == MoveKind::Infill))
            .cloned()
            .collect();
        assert!(
            !duplicated.is_empty(),
            "the planner should emit sparse infill paths for a 20mm box -- nothing to duplicate"
        );
        injected.extend(duplicated);
        let defective = audit_extrusion_volume(&mesh, &injected, &config, cell_size);
        let defective_ratio = defective
            .infill_aggregate_ratio()
            .expect("the defective grid must also have SparseInfill cells");

        // TODO(implementer): measure `baseline_ratio` and `defective_ratio`
        // directly (temporarily `println!` both, run with --no-capture,
        // then delete the println!s), and use those real numbers below --
        // do not guess. Expect `defective_ratio` to be roughly double
        // `baseline_ratio`, mirroring the wall-duplication measurements
        // from Task 1.
        assert!(
            defective_ratio >= baseline_ratio * 1.3,
            "duplicating every sparse infill path should raise the aggregate ratio well above \
             the healthy baseline: baseline {baseline_ratio}, defective {defective_ratio}"
        );

        // The healthy baseline must not itself trip a reasonable tolerance
        // -- fill in real measured margins here, following the same
        // discipline as `audit_extrusion_volume_passes_on_a_healthy_sliced_box`.
        baseline.assert_infill_volume_within(/* min_fraction */ 0.0, /* max_ratio */ 0.0);
        // The duplicated case, at the SAME tolerance, must panic.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            defective.assert_infill_volume_within(/* min_fraction */ 0.0, /* max_ratio */ 0.0);
        }));
        panicked.expect_err(
            "assert_infill_volume_within must panic when infill paths are duplicated \
             wholesale on real pipeline output",
        );
    }
```

Replace the two `/* min_fraction */ 0.0, /* max_ratio */ 0.0` placeholder pairs with real measured tolerances before this step is done — these are required measurements, not literal placeholders to leave in committed code. Measure `baseline_ratio` directly (temporarily add `println!("baseline: {baseline_ratio}, defective: {defective_ratio}");` right after both are computed, run `CARGO_TARGET_DIR=target/test cargo nextest run -p manifold-core volume_audit::tests::assert_infill_volume_within_catches_duplicated_infill_on_real_pipeline_output --no-capture`, read the printed values, then delete the `println!`). Set `min_fraction` comfortably below the observed `baseline_ratio` and `max_ratio` strictly between `baseline_ratio` and `defective_ratio` (same margin discipline as every other re-tuned bound in this plan), so the first `assert_infill_volume_within` call (on `baseline`) does not panic and the second (on `defective`) does. Update the comment above the two calls to record what you measured.

- [ ] **Step 4: Finalize the module-level doc comment cross-reference**

The module-level doc comment (updated in Task 1, Step 7) already references `infill_aggregate_ratio`/`assert_infill_volume_within` by name. Confirm it reads correctly now that both exist (run `cargo doc -p manifold-core --no-deps 2>&1 | grep -i "unresolved link"` — expect no output referencing this module; if there is output, fix the broken intra-doc link, most likely a typo in the `[\`...\`]` syntax).

- [ ] **Step 5: Run the full pre-commit gate and commit**

Run, in order:
```bash
cargo fmt --all
CARGO_TARGET_DIR=target/clippy cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=target/test cargo nextest run --workspace
```

All three must be clean.

Commit:
```bash
git add crates/manifold-core/src/volume_audit.rs
git commit -m "feat(core): add coarse whole-grid infill volume check"
```
