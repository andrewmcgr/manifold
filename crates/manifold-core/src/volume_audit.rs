//! Independent extrusion-volume audit: accumulates actual extruded volume
//! from a planned toolpath into a coarse 3D grid, computes an independent
//! expected volume for the same grid directly from the raw input mesh
//! (never from `Layer`/`OrderField` outputs -- see this module's own
//! `expected_fill_fraction` doc comment for why), and flags cells where
//! the two diverge. See
//! `docs/superpowers/specs/2026-09-18-extrusion-volume-audit-design.md`
//! for the full design rationale.

use std::collections::HashMap;

use glam::DVec3;
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::ScalarField;

use crate::slicing::BUILD_DIRECTION;
use crate::SlicerConfig;

#[cfg(test)]
use crate::mesh::Mesh;

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
///
/// `#[allow(dead_code)]` here is temporary: this task builds the grid and
/// its expected-volume primitives, but nothing in production code reads
/// these fields until Task 2's `audit_extrusion_volume` constructs and
/// populates a real `VolumeAuditGrid` -- only this module's own tests
/// touch them today. Remove this allow when Task 2 lands.
#[allow(dead_code)]
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
///
/// `#[allow(dead_code)]`: only called by `expected_fill_fraction` below
/// and this module's own tests until Task 2's `audit_extrusion_volume`
/// calls `expected_fill_fraction` from production code. Remove this
/// allow when Task 2 lands.
#[allow(dead_code)]
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
///
/// `#[allow(dead_code)]`: only called by `expected_fill_fraction` below
/// and this module's own tests until Task 2's `audit_extrusion_volume`
/// calls `expected_fill_fraction` from production code. Remove this
/// allow when Task 2 lands.
#[allow(dead_code)]
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
///
/// `#[allow(dead_code)]`: only called by this module's own tests until
/// Task 2's `audit_extrusion_volume` calls this from production code.
/// Remove this allow when Task 2 lands.
#[allow(dead_code)]
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
}
