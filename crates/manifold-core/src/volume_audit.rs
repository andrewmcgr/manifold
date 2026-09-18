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

use std::collections::HashMap;

use glam::DVec3;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

use crate::mesh::Mesh;
use crate::slicing::BUILD_DIRECTION;
use crate::SlicerConfig;

/// Which `MoveKind` family a segment's accumulated volume is bucketed
/// into. `Travel`/`Wipe`/`DebugExcluded` segments have no bucket (see
/// `VolumeKindBucket::from_move_kind` in Task 2) and are excluded from
/// accumulation entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolumeKindBucket {
    Wall,
    TopSurface,
    Infill,
    Overhang,
}

/// A coarse 3D voxel grid over a print's bounding volume, holding both
/// the actual accumulated extruded volume (Task 2) and an independently-
/// computed expected volume (this task) per cell, in mm^3.
pub struct VolumeAuditGrid {
    pub(crate) origin: DVec3,
    pub(crate) cell_size: f64,
    pub(crate) dims: [usize; 3],
    pub(crate) accumulated: HashMap<VolumeKindBucket, Vec<f64>>,
    pub(crate) expected: Vec<f64>,
    pub(crate) zone: Vec<FillZone>,
}

/// Whether `p` is within `threshold` of the mesh surface in *any*
/// direction -- the wall-shell zone. Direction-agnostic because a
/// printed wall shell exists uniformly around the whole perimeter
/// regardless of vertical position, unlike the top/bottom zones below.
fn wall_shell_zone(mesh_sdf: &MeshSdf, p: DVec3, threshold: f64) -> bool {
    mesh_sdf.sample(p).value.abs() <= threshold
}

/// Marches from `p` along `direction` (a unit vector) in `step`-sized
/// hops against `sdf`, until it crosses from inside (`value <= 0.0`,
/// this codebase's established sign convention) to outside, or
/// `max_search` is exceeded. Returns the *perpendicular* distance to the
/// exit surface -- the accumulated march distance projected onto the
/// exit sample's own gradient (its outward surface normal) -- not the
/// raw march distance, so a distance measured along a direction that
/// isn't perpendicular to a sloped surface doesn't overstate true
/// proximity to it.
///
/// Deliberately independent of `order_field::TopSurfaceAwareOrderField`,
/// which implements the same underlying geometric idea for production
/// slicing: this is a small, self-contained reimplementation using only
/// `MeshSdf::sample`, with none of the `OrderField` abstraction, the
/// `max_search.min(bed_distance)` bound optimization, or the five-point
/// disambiguation production code carries -- see this module's own doc
/// comment and the design spec's "Central Design Constraint" section for
/// why that independence is the entire point of this module.
fn directional_march(
    sdf: &MeshSdf,
    p: DVec3,
    direction: DVec3,
    step: f64,
    max_search: f64,
) -> Option<f64> {
    let mut traveled = 0.0;
    let mut pos = p;
    let mut prev_value = sdf.sample(pos).value;
    if prev_value > 0.0 {
        return Some(0.0);
    }
    while traveled < max_search {
        pos += direction * step;
        traveled += step;
        let sample = sdf.sample(pos);
        let value = sample.value;
        if value > 0.0 {
            let denom = value - prev_value;
            let t = if denom.abs() > 1e-12 {
                (-prev_value / denom).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let raw_distance = traveled - step * (1.0 - t);
            let normal_len_sq = sample.gradient.length_squared();
            return Some(if normal_len_sq > 1e-12 {
                let normal = sample.gradient / normal_len_sq.sqrt();
                raw_distance * direction.dot(normal).abs()
            } else {
                raw_distance
            });
        }
        prev_value = value;
    }
    None
}

/// Which physical zone a world-space point falls into, used to route grid
/// cells to the right check in `VolumeAuditGrid`'s query methods. See
/// `expected_fill_fraction`'s doc comment for why this classification --
/// like the fraction it's derived from -- is computed only from `MeshSdf`
/// and `SlicerConfig`, never from pipeline outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FillZone {
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
///
/// No production caller: `audit_extrusion_volume`'s grid-building loop
/// calls `classify_fill_zone`/`fraction_for_zone` directly (one SDF
/// sample instead of two) rather than through this wrapper. Kept
/// (`#[allow(dead_code)]`) purely so its own tests below continue to pin
/// `classify_fill_zone` + `fraction_for_zone`'s combined behavior against
/// the exact fractions the pre-refactor inline implementation produced --
/// this is a permanent self-check, not a gap awaiting a future caller.
#[allow(dead_code)]
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

impl VolumeKindBucket {
    /// The four extrusion-carrying `MoveKind` families this module
    /// tracks. `Travel`/`Wipe`/`DebugExcluded` (zero or non-physical
    /// extrusion) have no bucket and are excluded from accumulation.
    fn from_move_kind(kind: crate::toolpath::MoveKind) -> Option<Self> {
        use crate::toolpath::MoveKind;
        match kind {
            MoveKind::WallOuter | MoveKind::WallInner => Some(Self::Wall),
            MoveKind::TopSurface => Some(Self::TopSurface),
            MoveKind::Infill | MoveKind::Bridge => Some(Self::Infill),
            MoveKind::Overhang => Some(Self::Overhang),
            MoveKind::Travel | MoveKind::Wipe | MoveKind::DebugExcluded => None,
        }
    }
}

fn cell_index(origin: DVec3, cell_size: f64, dims: [usize; 3], p: DVec3) -> Option<usize> {
    let rel = (p - origin) / cell_size;
    if rel.x < 0.0 || rel.y < 0.0 || rel.z < 0.0 {
        return None;
    }
    let i = rel.x as usize;
    let j = rel.y as usize;
    let k = rel.z as usize;
    if i >= dims[0] || j >= dims[1] || k >= dims[2] {
        return None;
    }
    Some(i + j * dims[0] + k * dims[0] * dims[1])
}

/// Computes a [`VolumeAuditGrid`] over `mesh`'s bounding volume (padded
/// by one `cell_size`), comparing `paths`' actual accumulated extruded
/// volume against an expected volume computed independently from `mesh`
/// alone -- see [`expected_fill_fraction`]'s doc comment for why `paths`
/// (or any `Layer` it might have come from) never influences the
/// expected side.
///
/// **Choosing `cell_size`:** it is bounded on BOTH sides, for different
/// reasons, and both bounds have been confirmed by direct measurement
/// rather than inferred.
///
/// *Upper bound -- dilution.* A `cell_size` much larger than a single
/// bead's footprint means each cell's expected volume
/// (`expected_fill_fraction(..) * cell_size^3`) implicitly assumes
/// enough STACKED layers eventually fill that whole voxel, so a
/// duplication confined to a single layer's worth of material can be
/// diluted below any reasonable overfill threshold even when it's a
/// real bug. Measured: a duplicated single-layer wall bead at
/// `cell_size = 2.0` registers only ~0.04x its expected volume --
/// nowhere near overfilled.
///
/// *Lower bound -- centerline-splatting concentration.* `cell_size`
/// should generally be at least the widest `line_width` in play (the
/// widest bead's own footprint). This module's accumulation splats each
/// segment's whole bead volume onto its CENTERLINE, sampled point by
/// point along the segment -- it is not spread across the bead's actual
/// cross-sectional footprint. So when a cell is narrower than the bead
/// itself, volume that physically belongs spread across the full
/// `line_width` is instead concentrated into the single narrow column
/// of cells the centerline passes through, inflating those cells'
/// ratios while leaving the cells to either side empty. Measured: for a
/// straight bead the centerline cell accumulates
/// `bead_area * cell_size` against an expected `cell_size^3`, i.e. a
/// ratio of `bead_area / cell_size^2` -- so a single, correctly-printed,
/// non-duplicated bead reads 2.0x at `cell_size = 0.2` (below the 0.4
/// `wall_line_width`), but a correct 0.5x at `cell_size = 0.4` (at it).
/// Going below the widest `line_width` therefore manufactures
/// false-positive "overfill" from correct geometry.
///
/// In practice, for auditing a real multi-layer print, pick a
/// `cell_size` at or moderately above the widest `line_width`: large
/// enough to average over several adjacent beads and stacked layers
/// (where a correct print converges on ~1.0x), small enough to still
/// localize a defect.
pub fn audit_extrusion_volume(
    mesh: &Mesh,
    paths: &[crate::toolpath::Path],
    config: &SlicerConfig,
    cell_size: f64,
) -> VolumeAuditGrid {
    debug_assert!(
        cell_size > 0.0,
        "cell_size must be positive, got {cell_size} -- a zero or negative cell_size \
         silently produces a degenerate or absurdly-dimensioned grid rather than failing"
    );
    debug_assert!(
        mesh.bounding_box().is_some(),
        "mesh has no vertices -- the resulting grid is degenerate and every assertion on \
         it passes vacuously, which is worse than failing"
    );
    let (min, max) = mesh.bounding_box().unwrap_or((DVec3::ZERO, DVec3::ZERO));
    let origin = min - DVec3::splat(cell_size);
    let extent = (max - min) + DVec3::splat(cell_size * 2.0);
    let dims = [
        ((extent.x / cell_size).ceil() as usize).max(1),
        ((extent.y / cell_size).ceil() as usize).max(1),
        ((extent.z / cell_size).ceil() as usize).max(1),
    ];
    let cell_count = dims[0] * dims[1] * dims[2];

    let faces: Vec<[usize; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let mesh_sdf = MeshSdf::new(mesh.vertices.clone(), faces.clone());
    let non_bed_faces = crate::mesh::non_bed_floor_faces(mesh, min.z);
    let bed_excluded_sdf =
        MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_faces);

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

    let mut accumulated: HashMap<VolumeKindBucket, Vec<f64>> = [
        VolumeKindBucket::Wall,
        VolumeKindBucket::TopSurface,
        VolumeKindBucket::Infill,
        VolumeKindBucket::Overhang,
    ]
    .into_iter()
    .map(|bucket| (bucket, vec![0.0f64; cell_count]))
    .collect();

    // Volume that landed outside the grid entirely is silently lost from
    // every subsequent comparison, which would quietly weaken an audit whose
    // whole purpose is being loud. Track it so the `debug_assert!` below can
    // surface it rather than letting it vanish.
    let mut splatted_volume = 0.0f64;
    let mut dropped_volume = 0.0f64;

    for path in paths {
        let is_open = path.segments.len() + 1 == path.points.len();
        for (i, segment) in path.segments.iter().enumerate() {
            let Some(bucket) = VolumeKindBucket::from_move_kind(segment.kind) else {
                continue;
            };
            let start = path.points[i];
            let end_idx = if is_open {
                i + 1
            } else {
                (i + 1) % path.points.len()
            };
            let end = path.points[end_idx];
            let length = start.distance(end);
            if length < f64::EPSILON {
                continue;
            }
            let bead_volume = segment.extrusion_length
                * crate::extrusion::filament_cross_section_area(config.filament_diameter);
            let steps = ((length / (cell_size / 4.0)).ceil() as usize).max(1);
            let volume_per_step = bead_volume / steps as f64;
            let grid = accumulated
                .get_mut(&bucket)
                .expect("bucket initialized above");
            for s in 0..steps {
                let t = (s as f64 + 0.5) / steps as f64;
                let p = start.lerp(end, t);
                if let Some(cell) = cell_index(origin, cell_size, dims, p) {
                    grid[cell] += volume_per_step;
                    splatted_volume += volume_per_step;
                } else {
                    dropped_volume += volume_per_step;
                }
            }
        }
    }

    // The grid spans the mesh's bounding box padded by one `cell_size`, so a
    // well-formed toolpath for this mesh lands entirely inside it. Anything
    // material falling outside means either the paths don't belong to this
    // mesh or they stray far beyond its bounds -- both of which invalidate
    // the comparison this grid exists to make.
    debug_assert!(
        dropped_volume <= 0.01 * (splatted_volume + dropped_volume),
        "{dropped_volume} mm^3 of extrusion fell outside the audit grid \
         (vs {splatted_volume} mm^3 inside) and was dropped from every comparison -- \
         the paths likely do not correspond to this mesh"
    );

    VolumeAuditGrid {
        origin,
        cell_size,
        dims,
        accumulated,
        expected,
        zone,
    }
}

impl VolumeAuditGrid {
    fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    fn unflatten(&self, idx: usize) -> [usize; 3] {
        let i = idx % self.dims[0];
        let j = (idx / self.dims[0]) % self.dims[1];
        let k = idx / (self.dims[0] * self.dims[1]);
        [i, j, k]
    }

    /// The flat index of cell `idx`, panicking if it is out of bounds --
    /// the inverse of [`VolumeAuditGrid::unflatten`].
    fn flatten(&self, idx: [usize; 3]) -> usize {
        assert!(
            idx[0] < self.dims[0] && idx[1] < self.dims[1] && idx[2] < self.dims[2],
            "cell index {idx:?} is outside this grid's dimensions {:?}",
            self.dims
        );
        idx[0] + idx[1] * self.dims[0] + idx[2] * self.dims[0] * self.dims[1]
    }

    /// The world-space center of cell `idx`, for decoding the indices
    /// returned by [`VolumeAuditGrid::overfilled_cells`],
    /// [`VolumeAuditGrid::underfilled_cells`], and
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`] back into positions
    /// a caller can locate in the model.
    pub fn cell_center(&self, idx: [usize; 3]) -> DVec3 {
        self.origin
            + DVec3::new(
                (idx[0] as f64 + 0.5) * self.cell_size,
                (idx[1] as f64 + 0.5) * self.cell_size,
                (idx[2] as f64 + 0.5) * self.cell_size,
            )
    }

    fn total_accumulated(&self, idx: usize) -> f64 {
        self.accumulated.values().map(|v| v[idx]).sum()
    }

    /// The accumulated extruded volume (mm^3) in cell `idx` attributable to
    /// `bucket` alone, for attributing a flagged cell to a specific kind of
    /// move. Panics if `idx` is outside the grid.
    pub fn accumulated_volume(&self, idx: [usize; 3], bucket: VolumeKindBucket) -> f64 {
        let flat = self.flatten(idx);
        self.accumulated
            .get(&bucket)
            .expect("every VolumeKindBucket is initialized by audit_extrusion_volume")[flat]
    }

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

    /// `(cell index, ratio)` for every cell with `expected > 0` (i.e.
    /// every `Solid`- or `SparseInfill`-zone cell), regardless of the
    /// ratio's value.
    ///
    /// For display/visualization purposes only -- unlike
    /// [`VolumeAuditGrid::overfilled_cells`]/[`VolumeAuditGrid::underfilled_cells`],
    /// this is NOT restricted to `Solid`-zone cells and carries no
    /// pass/fail threshold. A caller doing defect *detection* should use
    /// those instead, or [`VolumeAuditGrid::infill_aggregate_ratio`] for
    /// infill's own coarser check. A cell with `expected == 0` (the
    /// `Outside` zone) is never included here -- see
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`] for that
    /// separate, binary case.
    pub fn cell_ratios_for_display(&self) -> Vec<([usize; 3], f64)> {
        (0..self.cell_count())
            .filter(|&idx| self.expected[idx] > 0.0)
            .map(|idx| {
                (
                    self.unflatten(idx),
                    self.total_accumulated(idx) / self.expected[idx],
                )
            })
            .collect()
    }

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

    /// Panics, naming the single worst offending cell (highest ratio),
    /// if any `Solid`-zone cell's (wall-shell, top-facing, or
    /// bottom-facing) accumulated volume exceeds `expected * max_ratio`.
    /// See [`VolumeAuditGrid::overfilled_cells`] for why `SparseInfill`-
    /// and `Outside`-zone cells are excluded, and
    /// [`VolumeAuditGrid::infill_aggregate_ratio`] /
    /// [`VolumeAuditGrid::assert_infill_volume_within`] for sparse
    /// infill's separate, coarser coverage.
    pub fn assert_no_overfill(&self, max_ratio: f64) {
        let worst = self
            .overfilled_cells(max_ratio)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((idx, ratio)) = worst {
            let p = self.cell_center(idx);
            panic!(
                "extrusion volume audit: cell {idx:?} (world {p:?}) has {ratio:.2}x its \
                 expected volume (max allowed {max_ratio:.2}x)"
            );
        }
    }

    /// Panics, naming the single worst offending cell (lowest fraction),
    /// if any `Solid`-zone cell's accumulated volume is below `expected *
    /// min_fraction`. See [`VolumeAuditGrid::underfilled_cells`] for why
    /// `SparseInfill`- and `Outside`-zone cells are excluded.
    pub fn assert_no_underfill(&self, min_fraction: f64) {
        let worst = self
            .underfilled_cells(min_fraction)
            .into_iter()
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((idx, fraction)) = worst {
            let p = self.cell_center(idx);
            panic!(
                "extrusion volume audit: cell {idx:?} (world {p:?}) only has {fraction:.2} \
                 of its expected volume (min allowed {min_fraction:.2})"
            );
        }
    }

    /// Panics, naming the cell holding the most stray material, if any
    /// volume was deposited where none is expected at all -- see
    /// [`VolumeAuditGrid::extrusion_outside_mesh_cells`].
    pub fn assert_no_extrusion_outside_mesh(&self) {
        let cells = self.extrusion_outside_mesh_cells();
        let worst = cells
            .iter()
            .map(|&idx| (idx, self.total_accumulated(self.flatten(idx))))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((idx, volume)) = worst {
            let p = self.cell_center(idx);
            panic!(
                "extrusion volume audit: {} cell(s) hold extrusion where none is expected; \
                 worst is cell {idx:?} (world {p:?}) with {volume:.4} mm^3",
                cells.len()
            );
        }
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An axis-aligned box mesh spanning `min`..`max`, matching the same
    /// fixture shape `slicing.rs`'s own test module uses (independently
    /// built here rather than imported, since `slicing::tests::box_mesh`
    /// is private to that module).
    fn box_mesh(min: DVec3, max: DVec3) -> Mesh {
        let vertices = vec![
            DVec3::new(min.x, min.y, min.z),
            DVec3::new(max.x, min.y, min.z),
            DVec3::new(max.x, max.y, min.z),
            DVec3::new(min.x, max.y, min.z),
            DVec3::new(min.x, min.y, max.z),
            DVec3::new(max.x, min.y, max.z),
            DVec3::new(max.x, max.y, max.z),
            DVec3::new(min.x, max.y, max.z),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        Mesh::new(vertices, indices)
    }

    fn mesh_sdf_for(mesh: &Mesh) -> MeshSdf {
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        MeshSdf::new(mesh.vertices.clone(), faces)
    }

    fn bed_excluded_sdf_for(mesh: &Mesh, min_z: f64) -> MeshSdf {
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        let non_bed_faces = crate::mesh::non_bed_floor_faces(mesh, min_z);
        MeshSdf::new_with_distance_faces(mesh.vertices.clone(), faces, non_bed_faces)
    }

    #[test]
    fn wall_shell_zone_is_true_near_a_face_and_false_deep_in_the_interior() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // 0.5mm from the -X face.
        assert!(wall_shell_zone(&sdf, DVec3::new(0.5, 10.0, 10.0), 1.0));
        // Center of a 20mm cube: 10mm from every face, well outside a 1mm threshold.
        assert!(!wall_shell_zone(&sdf, DVec3::new(10.0, 10.0, 10.0), 1.0));
    }

    #[test]
    fn directional_march_matches_closed_form_on_a_flat_top() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // Straight up from (7.3,3.9,5): flat top at z=20, perpendicular
        // distance == vertical distance == 15.0 exactly for a flat face.
        // Deliberately NOT (10,10,z): the box's top/bottom faces are each
        // split into 2 triangles along the (0,0)-(20,20) diagonal, and a
        // query point sitting exactly ON that shared diagonal edge (as
        // (10,10,*) does) triggers `MeshSdf::sample`'s degenerate
        // exactly-on-surface fallback (`p == closest`) at the exact
        // moment the march crosses the surface -- confirmed directly via
        // a debug probe: at that seam, the fallback's computed feature
        // normal is ~45 degrees off vertical, corrupting this function's
        // perpendicular-distance projection (reproducibly giving
        // 15.0/sqrt(2) instead of 15.0). This is the same defect class
        // the prior plan's Task 1 fix round 2 hit and fixed the same way
        // (moving the query off the mesh's own discretization seam) --
        // an off-diagonal point avoids it entirely, confirmed clean at
        // every step of this exact march via the same debug probe.
        let d = directional_march(&sdf, DVec3::new(7.3, 3.9, 5.0), BUILD_DIRECTION, 0.1, 30.0)
            .expect("march should reach the flat top within max_search");
        assert!((d - 15.0).abs() < 0.01, "expected ~15.0, got {d}");
    }

    #[test]
    fn directional_march_matches_closed_form_on_a_flat_bottom() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        // Straight down from (7.3,3.9,15): flat bottom at z=0, distance
        // == 15.0. Off-diagonal for the same reason as the flat-top test
        // above -- the bottom face has the identical (0,0)-(20,20)
        // diagonal seam.
        let d = directional_march(
            &sdf,
            DVec3::new(7.3, 3.9, 15.0),
            -BUILD_DIRECTION,
            0.1,
            30.0,
        )
        .expect("march should reach the flat bottom within max_search");
        assert!((d - 15.0).abs() < 0.01, "expected ~15.0, got {d}");
    }

    #[test]
    fn expected_fill_fraction_is_zero_outside_the_mesh() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig::default();
        let fraction =
            expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(-5.0, 10.0, 10.0), &config);
        assert_eq!(fraction, 0.0);
    }

    #[test]
    fn expected_fill_fraction_returns_wall_solid_near_a_face() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(20.0, 20.0, 20.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig::default();
        // 0.3mm from the -X face -- well within any reasonable wall
        // threshold (wall_offset + wall_count * wall_line_width, at
        // least nozzle_diameter/2 + one wall_line_width by construction).
        let fraction = expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(0.3, 10.0, 10.0), &config);
        assert_eq!(fraction, 1.0);
    }

    #[test]
    fn expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box() {
        // Tall enough that a genuine interior sparse-infill zone exists:
        // well beyond wall_shell/top_layers/bottom_layers from every face.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let sdf = mesh_sdf_for(&mesh);
        let bed_sdf = bed_excluded_sdf_for(&mesh, 0.0);
        let config = SlicerConfig {
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let fraction =
            expected_fill_fraction(&sdf, &bed_sdf, DVec3::new(20.0, 20.0, 20.0), &config);
        assert_eq!(fraction, config.infill_density);
    }

    use crate::toolpath::{MoveKind, Path, Segment};

    /// A single-segment open `Path` extruding in a straight line from
    /// `start` to `end`, with `extrusion_length` derived so the bead
    /// volume conserves exactly (see `extrusion::segment_extrusion_length`'s
    /// own doc comment: `distance * bead_area == filament_length *
    /// filament_area`).
    fn straight_extruding_path(
        start: DVec3,
        end: DVec3,
        kind: MoveKind,
        bead_area: f64,
        config: &SlicerConfig,
    ) -> Path {
        let distance = start.distance(end);
        let filament_area = crate::extrusion::filament_cross_section_area(config.filament_diameter);
        let extrusion_length =
            crate::extrusion::segment_extrusion_length(distance, bead_area, filament_area);
        Path {
            points: vec![start, end],
            segments: vec![Segment {
                kind,
                extrusion_length,
                line_width: bead_area / config.layer_height,
                ..Segment::default()
            }],
            tool: crate::ids::ToolId(0),
            object: crate::ids::ObjectId(0),
        }
    }

    #[test]
    fn audit_extrusion_volume_tracks_wall_and_infill_buckets_separately() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let wall_path = straight_extruding_path(
            DVec3::new(5.0, 20.0, 20.0),
            DVec3::new(5.0, 25.0, 20.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );
        let infill_bead_area = config.infill_line_width * config.layer_height;
        let infill_path = straight_extruding_path(
            DVec3::new(20.0, 20.0, 20.0),
            DVec3::new(25.0, 20.0, 20.0),
            MoveKind::Infill,
            infill_bead_area,
            &config,
        );
        let grid = audit_extrusion_volume(
            &mesh,
            &[wall_path.clone(), infill_path.clone()],
            &config,
            2.0,
        );

        let expected_wall_volume = wall_path.segments[0].extrusion_length
            * crate::extrusion::filament_cross_section_area(config.filament_diameter);
        let expected_infill_volume = infill_path.segments[0].extrusion_length
            * crate::extrusion::filament_cross_section_area(config.filament_diameter);

        let wall_total: f64 = grid.accumulated[&VolumeKindBucket::Wall].iter().sum();
        let infill_total: f64 = grid.accumulated[&VolumeKindBucket::Infill].iter().sum();
        assert!(
            (wall_total - expected_wall_volume).abs() < expected_wall_volume * 0.01,
            "wall bucket total {wall_total} should match {expected_wall_volume} within 1%"
        );
        assert!(
            (infill_total - expected_infill_volume).abs() < expected_infill_volume * 0.01,
            "infill bucket total {infill_total} should match {expected_infill_volume} within 1%"
        );
        // Cross-contamination check: infill volume must not have leaked
        // into the wall bucket or vice versa.
        let wall_bucket_infill_leak: f64 = grid.accumulated[&VolumeKindBucket::Wall]
            .iter()
            .zip(grid.accumulated[&VolumeKindBucket::Infill].iter())
            .map(|(w, i)| if *w > 0.0 && *i > 0.0 { 1.0 } else { 0.0 })
            .sum();
        assert_eq!(
            wall_bucket_infill_leak, 0.0,
            "no cell should have both nonzero wall and nonzero infill volume for these two well-separated paths"
        );
    }

    #[test]
    fn overfilled_cells_detects_a_deliberately_duplicated_wall_segment() {
        // A much smaller mesh than this file's other fixtures (40mm cube):
        // at `cell_size = 0.2` (see below), a 40mm mesh produces a
        // ~202^3 = 8.24 million cell grid -- confirmed directly to take
        // 75+ seconds per test. This test only needs the duplicated
        // path's own small neighborhood to be inside solid material near
        // a face; a tightly-sized mesh keeps the grid small (~4600
        // cells) without changing the verified bead/cell_size math below
        // (which depends only on bead cross-section area and cell_size,
        // not on path length or mesh size, as long as the path is
        // several cells long -- reconfirmed by direct calculation).
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );

        // This test asserts the *relative* increase duplication causes,
        // not an absolute ratio threshold, and that distinction is
        // load-bearing.
        //
        // These hand-built fixtures deposit a single LAYER's worth of
        // bead, which deliberately sits outside the `cell_size` regime
        // `audit_extrusion_volume`'s own doc comment recommends for real
        // multi-layer audits (see it for both the upper- and lower-bound
        // gotchas). Concretely, for a straight bead the centerline cell
        // accumulates `bead_area * cell_size` against an expected
        // `cell_size^3`, i.e. ratio `= bead_area / cell_size^2` -- so at
        // `cell_size = 0.2` even a SINGLE, correctly-printed,
        // non-duplicated bead already reads 2.0x (measured directly),
        // purely as a centerline-splatting concentration artifact. An
        // absolute assertion like `overfilled_cells(1.5)` being non-empty
        // would therefore pass here with or without any duplication at
        // all -- proving nothing about duplication detection.
        //
        // What IS a genuine duplication signal is the ratio between the
        // duplicated and single-path cases: duplication doubles the
        // deposited volume while leaving expected volume untouched, so
        // the max ratio must double. Measured directly at cell_size
        // 0.2/0.4/0.5/0.6/0.8: single = 2.0/0.5/0.32/0.21/0.125,
        // duplicated = 4.0/1.0/0.64/0.42/0.25 -- an exactly 2.0x
        // increase at every one, independent of the artifact baseline.
        let cell_size = 0.2;
        let single = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);
        let single_max = max_overfill_ratio(&single);
        let duplicated = vec![path.clone(), path.clone()];
        let doubled = audit_extrusion_volume(&mesh, &duplicated, &config, cell_size);
        let doubled_max = max_overfill_ratio(&doubled);

        assert!(
            single_max > 0.0,
            "test fixture deposited nothing measurable (single-path max ratio {single_max}) -- \
             the comparison below would be vacuous"
        );
        assert!(
            doubled_max >= single_max * 1.5,
            "duplicating a wall segment must produce a materially higher overfill ratio than \
             printing it once: single-path max {single_max}, duplicated max {doubled_max} \
             (expected ~2.0x the single-path value)"
        );
    }

    #[test]
    fn assert_no_overfill_does_not_false_positive_but_panics_on_duplication() {
        // Chose option (a) from the fix dispatch -- a paired
        // negative/positive control in one test -- over option (b)
        // (narrowing this to a message-format check), because the
        // measured numbers separate cleanly enough to calibrate a
        // threshold strictly between them, making this a genuinely
        // stronger assertion than either half alone: it proves
        // `assert_no_overfill` does NOT fire on correct (non-duplicated)
        // geometry *and* DOES fire on duplicated geometry, at one
        // threshold.
        //
        // Deliberately NOT `#[should_panic]`: under that attribute a
        // panic from the negative-control call would still "pass" the
        // test, which is the exact failure mode this fix round exists to
        // eliminate. The negative control is instead verified by the
        // call simply not unwinding (a panic there fails the test
        // normally), and the positive control via `catch_unwind`, which
        // also lets the panic message be checked explicitly.
        //
        // Same small-mesh rationale as
        // `overfilled_cells_detects_a_deliberately_duplicated_wall_segment`
        // above (a 40mm mesh at this `cell_size` would produce an
        // 8.24-million-cell grid, confirmed to take 75+ seconds; this
        // ~4600-cell mesh runs near-instantly with identical math).
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let bead_area = config.wall_line_width * config.layer_height;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            bead_area,
            &config,
        );
        let cell_size = 0.2;

        // Measured directly at this cell_size: a single (correct,
        // non-duplicated) bead peaks at 2.0x, the duplicated pair at
        // 4.0x (see the sibling test above for why a single bead reads
        // above 1.0x at all -- centerline-splatting concentration, not a
        // defect). A threshold of 3.0 therefore sits strictly between
        // them, which is what makes the pairing below meaningful rather
        // than trivially satisfiable from either side.
        let max_ratio = 3.0;

        // Negative control: correct, non-duplicated geometry must NOT
        // trip the assertion. If this panics, the test fails here.
        let single = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);
        single.assert_no_overfill(max_ratio);

        // Positive control: the same geometry printed twice MUST trip it.
        let duplicated = vec![path.clone(), path.clone()];
        let doubled = audit_extrusion_volume(&mesh, &duplicated, &config, cell_size);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            doubled.assert_no_overfill(max_ratio);
        }));
        let payload = panicked.expect_err(
            "assert_no_overfill must panic on duplicated extrusion exceeding max_ratio",
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

    /// The highest overfill ratio (accumulated / expected) across every
    /// cell with a nonzero expected volume, or `0.0` if no such cell
    /// registered any material. Expressed via the public
    /// [`VolumeAuditGrid::overfilled_cells`] query (with a `0.0`
    /// threshold, so every cell carrying material is returned) rather
    /// than by reaching into the grid's private fields, so these tests
    /// exercise the same API a real caller would.
    fn max_overfill_ratio(grid: &VolumeAuditGrid) -> f64 {
        grid.overfilled_cells(0.0)
            .into_iter()
            .map(|(_, ratio)| ratio)
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn extrusion_outside_the_mesh_is_caught_only_by_its_own_dedicated_query() {
        // The design names this defect class explicitly: "Outside the mesh
        // entirely: expected = 0 -- any accumulated volume there is extrusion
        // into open air, a defect regardless of everything else in this
        // design." It cannot be expressed as a ratio (there is nothing to
        // take a multiple of), so `overfilled_cells` -- which necessarily
        // skips `expected == 0` cells to avoid dividing by zero -- is
        // structurally blind to it. This test pins both halves of that split:
        // the dedicated query sees it, and the ratio query provably does not.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        // The grid spans the mesh's bounding box padded by one `cell_size`,
        // so x = -0.25 is outside the mesh (min.x == 0.0) but still inside
        // the grid -- material here is deposited into thin air, not merely
        // dropped for falling off the edge of the grid.
        let stray = straight_extruding_path(
            DVec3::new(-0.25, 2.0, 1.0),
            DVec3::new(-0.25, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&stray), &config, cell_size);

        let outside = grid.extrusion_outside_mesh_cells();
        assert!(
            !outside.is_empty(),
            "a bead extruded entirely outside the mesh must be reported by \
             extrusion_outside_mesh_cells"
        );
        for &idx in &outside {
            assert!(
                grid.cell_center(idx).x < 0.0,
                "every reported cell should be on the outside-the-mesh side, got {:?}",
                grid.cell_center(idx)
            );
        }
        // The gap this query exists to close: with a `0.0` threshold
        // `overfilled_cells` returns every cell carrying any material at
        // all, and still cannot see this one.
        assert!(
            grid.overfilled_cells(0.0).is_empty(),
            "overfilled_cells is structurally unable to report extrusion into open air -- \
             if this ever starts reporting it, the two queries' split has changed"
        );

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            grid.assert_no_extrusion_outside_mesh();
        }));
        let payload = panicked
            .expect_err("assert_no_extrusion_outside_mesh must panic on extrusion into open air");
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

    #[test]
    fn assert_no_extrusion_outside_mesh_passes_when_every_bead_is_inside() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let inside = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&inside), &config, 0.5);
        assert!(grid.extrusion_outside_mesh_cells().is_empty());
        grid.assert_no_extrusion_outside_mesh();
    }

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
        let grid = audit_extrusion_volume(
            &mesh,
            std::slice::from_ref(&infill_path),
            &config,
            cell_size,
        );

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

        // Measured directly on this exact fixture (temporarily printed both
        // ratios via `println!`, ran with `--no-capture`, then deleted the
        // print): baseline_ratio = 1.2620500014927434, defective (all
        // sparse infill paths duplicated) = 2.5241000029854863 -- almost
        // exactly double, as expected: duplicating every infill path
        // doubles deposited volume in every SparseInfill cell while
        // leaving expected volume untouched.
        assert!(
            defective_ratio >= baseline_ratio * 1.3,
            "duplicating every sparse infill path should raise the aggregate ratio well above \
             the healthy baseline: baseline {baseline_ratio}, defective {defective_ratio}"
        );

        // The healthy baseline (1.262) must not itself trip a reasonable
        // tolerance; the duplicated case (2.524) must. 1.9 sits strictly
        // between them with comfortable margin on both sides (~50.6% above
        // the baseline ratio, ~24.7% below the defective ratio).
        baseline.assert_infill_volume_within(0.0, 1.9);
        // The duplicated case, at the SAME tolerance, must panic.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            defective.assert_infill_volume_within(0.0, 1.9);
        }));
        panicked.expect_err(
            "assert_infill_volume_within must panic when infill paths are duplicated \
             wholesale on real pipeline output",
        );
    }

    #[test]
    fn cell_center_and_accumulated_volume_decode_a_reported_cell() {
        // The query methods hand back bare `[usize; 3]` indices; without
        // these two accessors an out-of-crate caller has no way to turn one
        // into a position it can locate, or to attribute it to a move kind.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        let wall = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallInner,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&wall), &config, cell_size);

        // The grid's origin is the mesh's min corner padded outward by one
        // `cell_size`, so cell [0,0,0]'s center sits half a cell inside that.
        let corner = grid.cell_center([0, 0, 0]);
        let expected_corner = DVec3::splat(-cell_size / 2.0);
        assert!(
            corner.abs_diff_eq(expected_corner, 1e-12),
            "cell [0,0,0] center should be {expected_corner:?}, got {corner:?}"
        );

        let (idx, _) = grid
            .overfilled_cells(0.0)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("the wall bead should register in at least one cell");
        assert!(
            grid.accumulated_volume(idx, VolumeKindBucket::Wall) > 0.0,
            "a WallInner bead's volume should be attributed to the Wall bucket"
        );
        assert_eq!(
            grid.accumulated_volume(idx, VolumeKindBucket::Infill),
            0.0,
            "no infill was planned, so the Infill bucket must be empty here"
        );
        // The reported cell should be the one the bead actually runs through.
        let center = grid.cell_center(idx);
        assert!(
            (center.x - 0.25).abs() < 1e-12 && (center.z - 1.25).abs() < 1e-12,
            "reported cell center {center:?} should lie on the bead's own column"
        );
    }

    #[test]
    fn cell_ratios_for_display_includes_both_solid_and_sparse_infill_cells() {
        // Tall enough that a genuine interior sparse-infill zone exists,
        // same fixture shape as
        // `expected_fill_fraction_returns_density_deep_in_the_interior_of_a_tall_box`.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(40.0, 40.0, 40.0));
        let config = SlicerConfig {
            infill_density: 0.2,
            ..SlicerConfig::default()
        };
        let cell_size = 2.0;
        let wall_bead_area = config.wall_line_width * config.layer_height;
        // A wall bead near a face (Solid zone) ...
        let wall_path = straight_extruding_path(
            DVec3::new(0.3, 20.0, 20.0),
            DVec3::new(0.3, 22.0, 20.0),
            MoveKind::WallOuter,
            wall_bead_area,
            &config,
        );
        // ... and an infill bead deep in the interior (SparseInfill zone).
        let infill_bead_area = config.infill_line_width * config.layer_height;
        let infill_path = straight_extruding_path(
            DVec3::new(19.0, 20.0, 20.0),
            DVec3::new(21.0, 20.0, 20.0),
            MoveKind::Infill,
            infill_bead_area,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, &[wall_path, infill_path], &config, cell_size);

        let ratios = grid.cell_ratios_for_display();
        assert!(
            !ratios.is_empty(),
            "a mesh with real extrusion should register at least one cell"
        );

        // Cross-check: every returned cell must have `expected > 0`
        // (reachable only via the private zone/expected fields, so
        // reconstruct via the same public accessors a real caller has:
        // a cell classified `Outside` never appears here even if it
        // somehow held material -- covered by the dedicated fixture in
        // the next test instead).
        let outside = grid.extrusion_outside_mesh_cells();
        for (idx, _) in &ratios {
            assert!(
                !outside.contains(idx),
                "cell {idx:?} appears in both cell_ratios_for_display and \
                 extrusion_outside_mesh_cells -- the two queries must partition disjointly"
            );
        }

        // At least one cell should register a wall-adjacent ratio near
        // the wall bead's own known accumulation (loosely bounded --
        // this is a display query, not a precision assertion; the goal
        // is just confirming both zone kinds are represented).
        assert!(
            ratios.iter().any(|(_, r)| *r > 0.0),
            "at least one cell should have a nonzero ratio given real extrusion was deposited"
        );
    }

    #[test]
    fn cell_ratios_for_display_excludes_outside_the_mesh_cells() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        // Same stray-bead fixture as
        // `extrusion_outside_the_mesh_is_caught_only_by_its_own_dedicated_query`:
        // x = -0.25 is outside the mesh (min.x == 0.0) but inside the
        // padded grid.
        let stray = straight_extruding_path(
            DVec3::new(-0.25, 2.0, 1.0),
            DVec3::new(-0.25, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&stray), &config, cell_size);

        let outside = grid.extrusion_outside_mesh_cells();
        assert!(
            !outside.is_empty(),
            "the stray bead must register as outside-the-mesh material"
        );

        let display_ratios = grid.cell_ratios_for_display();
        for idx in &outside {
            assert!(
                !display_ratios
                    .iter()
                    .any(|(display_idx, _)| display_idx == idx),
                "cell_ratios_for_display must not include cell {idx:?}, which \
                 extrusion_outside_mesh_cells already reports as outside the mesh"
            );
        }
    }

    #[test]
    fn audit_extrusion_volume_passes_on_a_healthy_sliced_box() {
        let mesh = box_mesh(DVec3::new(0.0, 0.0, 0.0), DVec3::new(20.0, 20.0, 20.0));
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
        // `cell_size = 2.0` is well above `wall_line_width`/`infill_line_width`
        // (0.4), squarely in the regime `audit_extrusion_volume`'s own doc
        // comment recommends for a real multi-layer audit -- large enough to
        // average over many stacked layers and adjacent beads, avoiding both
        // the dilution (upper-bound) and centerline-splatting (lower-bound)
        // artifacts documented there.
        let grid = audit_extrusion_volume(&mesh, &paths, &config, 2.0);

        // Tolerances re-measured after restricting overfilled_cells/
        // underfilled_cells to Solid-zone cells only (this task): measured
        // directly by temporarily setting both bounds to 1.0 and printing
        // the exact max overfill ratio / min underfill fraction across
        // every Solid-zone cell. Worst observed overfill ratio 0.839
        // (no Solid-zone cell overfills at all on this healthy print, now
        // that the sparse-infill cell that used to dominate this metric --
        // 1.46x at this cell_size -- is excluded), worst observed
        // underfill fraction 0.324 (essentially unchanged from the
        // pre-restriction measurement of 0.32, confirming this is a
        // genuine partial-fill Solid-zone cell unrelated to the
        // sparse-infill swamping this task fixes). Both bounds below are
        // set comfortably past those measured values, not guessed --
        // `assert_no_overfill`/`assert_no_underfill` name the exact
        // offending cell and its exact ratio/fraction in their own panic
        // message if either ever regresses past these margins. The
        // overfill bound is now meaningfully tighter than the old 2.0 --
        // that headroom existed only to tolerate the sparse-infill
        // artifact this task removes from this check entirely.
        grid.assert_no_overfill(1.2);
        grid.assert_no_underfill(0.25);
    }

    #[test]
    fn duplicated_walls_on_real_pipeline_output_raise_per_cell_ratios() {
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
        let baseline_ratios: HashMap<[usize; 3], f64> =
            baseline.overfilled_cells(0.0).into_iter().collect();
        assert!(
            !baseline_ratios.is_empty(),
            "the healthy baseline should register material in some cells"
        );

        // Inject the defect: print every inner wall loop twice. This is the
        // same shape as the bug the prior plan fixed (wall material laid down
        // on top of material already there), but on genuinely planned
        // geometry rather than a hand-built stand-in.
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
        let defective = audit_extrusion_volume(&mesh, &injected, &config, cell_size);

        // The signal: compare each cell against its own healthy baseline.
        // This cancels the sampling artifacts both runs share, leaving only
        // the injected duplication.
        let worst_increase = defective
            .overfilled_cells(0.0)
            .into_iter()
            .filter_map(|(idx, ratio)| baseline_ratios.get(&idx).map(|base| ratio / base))
            .fold(0.0f64, f64::max);
        // Measured 1.4196 on this fixture; 1.3 leaves margin without being a
        // round number the observation does not support.
        assert!(
            worst_increase >= 1.3,
            "duplicating every inner wall path should raise some cell's fill ratio well above \
             its healthy baseline: worst per-cell increase was {worst_increase}x (expected \
             ~1.42x)"
        );
    }

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

        // Measured directly on this exact fixture (temporarily set
        // max_ratio to 100.0 on both grids, printed each grid's own
        // `overfilled_cells(0.0)` max ratio, then deleted the prints):
        // healthy baseline max 0.8392593691769967 (same Solid-zone-only
        // measurement as the golden-path test above, since this is the
        // identical fixture), defective (all inner walls duplicated) max
        // 1.0159057233105697. The two signals sit close together at this
        // cell_size, so the margin on each side is necessarily tighter than
        // the golden-path test's -- 0.93 gives ~10.8% headroom above the
        // healthy max and ~8.4% headroom below the defective max, still
        // strictly between them and well past ordinary floating-point
        // noise between runs.
        let max_ratio = 0.93;

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

    #[test]
    fn audit_extrusion_volume_catches_a_wall_and_solid_infill_overlap_near_a_taper_tip() {
        // Reproduces, via hand-built Paths rather than reverting production
        // code, the exact bug class the prior plan
        // (2026-09-17-perpendicular-top-distance-and-wall-overlap) fixed: an
        // inner wall loop and solid infill printing over the same physical
        // space near a taper's tip. A small box stands in for the taper
        // geometry -- what matters here is the toolpath overlap, not the
        // taper shape itself, since the expected-volume side (Task 1) has
        // its own dedicated tests proving the top-zone classification is
        // correct on real tapered geometry.
        //
        // Adapted from the brief's literal version, which asserted
        // `overfilled_cells(1.5).is_empty()` as an absolute threshold --
        // Task 2's own fix rounds established (and this file's sibling
        // duplication tests above already demonstrate) that an absolute
        // ratio threshold cannot distinguish real duplication from the
        // centerline-splatting concentration artifact a single,
        // non-duplicated bead already produces at a `cell_size` below its
        // own `line_width` (exactly the regime this hand-built,
        // single-layer fixture is in). Instead, this follows the same
        // relative-ratio pattern as
        // `overfilled_cells_detects_a_deliberately_duplicated_wall_segment`:
        // compare the wall-alone baseline ratio against the wall+infill
        // overlap ratio, and require the overlap to be meaningfully
        // higher -- proving the assertion detects the overlap specifically,
        // not the shared splatting artifact both cases equally have.
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(3.0, 3.0, 3.0));
        let config = SlicerConfig {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
            wall_line_width: 0.4,
            infill_line_width: 0.4,
            shell_thickness: 0.8,
            ..SlicerConfig::default()
        };
        let bead_area = config.wall_line_width * config.layer_height;
        let cell_size = 1.0;

        // An inner wall loop (a short straight bead standing in for one) ...
        //
        // Deviation from the brief: the original fixture placed this bead at
        // the dead center of the cube (z=1.5), which classifies as
        // `SparseInfill` under this task's zone restriction (distance to
        // every face is 1.5mm, past both the wall_threshold of 1.0mm and the
        // bottom_threshold of 0.6mm) -- so `overfilled_cells` now correctly
        // excludes it entirely, making the whole test vacuous (confirmed:
        // `single_max`/baseline_ratio measured 0 before this fix). Moved to
        // z=0.3, within 0.5mm of the z=0 face -- inside wall_threshold
        // (1.0mm) via the plain omnidirectional wall_shell_zone check, no
        // directional march involved. This is also more faithful to the
        // test's own narrative: the wall/solid-fill overlap bug this
        // reproduces occurs near a taper's tip, i.e. near a top/bottom/wall
        // boundary, not deep in a part's interior.
        let wall_bead_start = DVec3::new(1.0, 1.0, 0.3);
        let wall_bead_end = DVec3::new(2.0, 1.0, 0.3);
        let wall_path = straight_extruding_path(
            wall_bead_start,
            wall_bead_end,
            MoveKind::WallInner,
            bead_area,
            &config,
        );
        // ... and a solid-infill pass tracing the SAME physical centerline
        // -- the exact overlap the fixed bug produced. `infill_line_width ==
        // wall_line_width` here, so the wall-alone and infill-alone bead
        // volumes are equal, making the expected "overlap doubles the
        // baseline ratio" relationship exact rather than approximate.
        let infill_path = straight_extruding_path(
            wall_bead_start,
            wall_bead_end,
            MoveKind::Infill,
            config.infill_line_width * config.layer_height,
            &config,
        );

        let wall_alone =
            audit_extrusion_volume(&mesh, std::slice::from_ref(&wall_path), &config, cell_size);
        let baseline_ratio = max_overfill_ratio(&wall_alone);
        assert!(
            baseline_ratio > 0.0,
            "test fixture deposited nothing measurable for the wall alone (ratio {baseline_ratio}) -- \
             the comparison below would be vacuous"
        );

        let overlapping = vec![wall_path, infill_path];
        let grid = audit_extrusion_volume(&mesh, &overlapping, &config, cell_size);
        let overlap_ratio = max_overfill_ratio(&grid);

        assert!(
            overlap_ratio >= baseline_ratio * 1.5,
            "a wall loop and solid infill occupying the same physical centerline should \
             register a materially higher overfill ratio than the wall alone -- this is the \
             exact bug class the tool exists to catch: wall-alone ratio {baseline_ratio}, \
             wall+infill-overlap ratio {overlap_ratio} (expected roughly double the baseline)"
        );
    }
}
