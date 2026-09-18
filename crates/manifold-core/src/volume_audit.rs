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

/// The expected fill fraction (`0.0`..`1.0`) at world-space point `p`:
/// `0.0` outside the mesh entirely, `1.0` within the wall-shell,
/// top-facing, or bottom-facing zones (see this function's body for each
/// threshold), else `config.infill_density`.
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
fn expected_fill_fraction(
    mesh_sdf: &MeshSdf,
    bed_excluded_sdf: &MeshSdf,
    p: DVec3,
    config: &SlicerConfig,
) -> f64 {
    if mesh_sdf.sample(p).value > 0.0 {
        return 0.0;
    }

    let wall_threshold = config.wall_offset + config.wall_count() as f64 * config.wall_line_width;
    if wall_shell_zone(mesh_sdf, p, wall_threshold) {
        return 1.0;
    }

    let step = (config.layer_height.min(config.nozzle_diameter) / 4.0).max(0.01);
    let max_search = (config.layer_height * 20.0).max(5.0);

    let top_threshold = config.top_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(bed_excluded_sdf, p, BUILD_DIRECTION, step, max_search) {
        if d <= top_threshold {
            return 1.0;
        }
    }

    let bottom_threshold = config.bottom_layers as f64 * config.layer_height;
    if let Some(d) = directional_march(mesh_sdf, p, -BUILD_DIRECTION, step, max_search) {
        if d <= bottom_threshold {
            return 1.0;
        }
    }

    config.infill_density
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
    /// if any cell's accumulated volume is below `expected * min_fraction`.
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
        let wall_bead_start = DVec3::new(1.0, 1.0, 1.5);
        let wall_bead_end = DVec3::new(2.0, 1.0, 1.5);
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
