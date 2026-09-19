//! Pure geometry builder for the volume-audit fill visualization: turns a
//! `manifold_core::volume_audit::VolumeAuditGrid`'s per-cell fill ratios
//! into flat-colored cube triangles. No GPU/wgpu types here — mirrors
//! `toolpath_view.rs`'s existing separation from `render.rs`'s GPU
//! upload/pipeline concerns. See
//! `docs/superpowers/specs/2026-09-18-volume-audit-fill-visualization-design.md`
//! for the full design rationale.

use glam::DVec3;
use manifold_core::volume_audit::VolumeAuditGrid;

/// One GPU vertex for a flat-colored cube face: position + face normal +
/// RGBA color, all in world space. Bit-identical layout to `render.rs`'s
/// private `Vertex`, so `UploadedMesh::upload_colored_cells` can build
/// its buffer directly from a slice of these.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeAuditCellVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

/// Fixed color for a cell reported by `VolumeAuditGrid::extrusion_outside_mesh_cells`
/// (material where none is expected at all) -- a defect at any
/// magnitude, so distinct from the ratio-based blue/green/red scale and
/// never hidden by `deviation_threshold`.
const OUTSIDE_MESH_COLOR: [f32; 4] = [0.95, 0.05, 0.85, CUBE_ALPHA];

/// Shared opacity for every volume-audit cube (both the ratio-based scale
/// and [`OUTSIDE_MESH_COLOR`]). Semi-transparent rather than fully opaque
/// so overlapping/nearby cubes, and the now-semi-transparent mesh surface
/// itself (see `render.rs`'s `mesh_transparent_single_sided_pipeline`),
/// remain individually distinguishable instead of a solid colored mass.
const CUBE_ALPHA: f32 = 0.7;

/// The six axis-aligned face normals of a cube, in the same face order
/// `push_cube`'s vertex generation below uses.
const FACE_NORMALS: [DVec3; 6] = [
    DVec3::new(1.0, 0.0, 0.0),
    DVec3::new(-1.0, 0.0, 0.0),
    DVec3::new(0.0, 1.0, 0.0),
    DVec3::new(0.0, -1.0, 0.0),
    DVec3::new(0.0, 0.0, 1.0),
    DVec3::new(0.0, 0.0, -1.0),
];

/// Appends 36 vertices (12 triangles, non-indexed, one flat color for
/// the whole cube) for an axis-aligned cube centered at `center` with
/// half-extent `half_size` in every axis.
fn push_cube(out: &mut Vec<VolumeAuditCellVertex>, center: DVec3, half_size: f64, color: [f32; 4]) {
    let h = half_size;
    // Per-face 4 corners (in a consistent winding), split into 2
    // triangles each. Corner order per face matches `FACE_NORMALS`'s
    // ordering: +X, -X, +Y, -Y, +Z, -Z.
    let faces: [[DVec3; 4]; 6] = [
        // +X
        [
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(h, h, -h),
            center + DVec3::new(h, h, h),
            center + DVec3::new(h, -h, h),
        ],
        // -X
        [
            center + DVec3::new(-h, -h, h),
            center + DVec3::new(-h, h, h),
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(-h, -h, -h),
        ],
        // +Y
        [
            center + DVec3::new(h, h, -h),
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(-h, h, h),
            center + DVec3::new(h, h, h),
        ],
        // -Y
        [
            center + DVec3::new(-h, -h, -h),
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(h, -h, h),
            center + DVec3::new(-h, -h, h),
        ],
        // +Z
        [
            center + DVec3::new(-h, -h, h),
            center + DVec3::new(h, -h, h),
            center + DVec3::new(h, h, h),
            center + DVec3::new(-h, h, h),
        ],
        // -Z
        [
            center + DVec3::new(-h, h, -h),
            center + DVec3::new(h, h, -h),
            center + DVec3::new(h, -h, -h),
            center + DVec3::new(-h, -h, -h),
        ],
    ];

    for (face_idx, corners) in faces.iter().enumerate() {
        let normal = FACE_NORMALS[face_idx].as_vec3().to_array();
        // Two triangles per quad: (0,1,2) and (0,2,3).
        for &(a, b, c) in &[(0usize, 1usize, 2usize), (0, 2, 3)] {
            for &i in &[a, b, c] {
                out.push(VolumeAuditCellVertex {
                    position: corners[i].as_vec3().to_array(),
                    normal,
                    color,
                });
            }
        }
    }
}

/// Maps a fill ratio (`accumulated / expected`) to a color on the blue
/// (under) -> green (healthy at 1.0) -> red (over) scale, auto-normalized
/// against `min_ratio`/`max_ratio` -- the actual minimum and maximum
/// ratio observed across every cell in the current audit run (see
/// [`build_volume_audit_cells`]).
///
/// A fixed absolute scale (e.g. always clamping to `[0, 2]`) made nearly
/// every cell in a real print read the same narrow shade, because the
/// systematic sparse-infill nominal-density mismatch (documented on
/// `VolumeAuditGrid::overfilled_cells`) dominates the range with a
/// roughly constant, non-defect deviation -- observed directly: a real
/// slice's cells clustered around ratio ~1.6-1.8, all rendering as the
/// same shade of orange, leaving no visible distinction between a
/// genuine defect and routine sparse-infill noise. Auto-normalizing
/// spreads whatever variation is actually present across the full
/// color range instead.
///
/// `ratio == 1.0` always maps to the exact green midpoint regardless of
/// `min_ratio`/`max_ratio`, and the two sides are scaled independently
/// (`[min_ratio, 1.0]` stretched to blue..green, `[1.0, max_ratio]`
/// stretched to green..red) so a healthy print's true center is never
/// skewed by an asymmetric spread of over- vs. under-fill.
fn ratio_to_color(ratio: f64, min_ratio: f64, max_ratio: f64) -> [f32; 4] {
    let deviation = if ratio >= 1.0 {
        if max_ratio > 1.0 {
            ((ratio - 1.0) / (max_ratio - 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    } else if min_ratio < 1.0 {
        -((1.0 - ratio) / (1.0 - min_ratio)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let t = 0.5 + 0.5 * deviation;
    let [r, g, b, _] = crate::toolpath_view::scalar_to_color(t);
    [r, g, b, CUBE_ALPHA]
}

/// Builds a non-indexed triangle-list cube (12 triangles, 36 vertices)
/// per qualifying cell in `grid`, colored by
/// `VolumeAuditGrid::cell_ratios_for_display`'s ratio (blue = under,
/// green = healthy at ratio 1.0, red = over) or a fixed magenta for
/// cells reported by `VolumeAuditGrid::extrusion_outside_mesh_cells`
/// (material where none is expected at all -- shown unconditionally,
/// never hidden by `deviation_threshold`).
///
/// `deviation_threshold` (`>= 0.0`) hides every ratio-based cube whose
/// `|ratio - 1.0|` is below it -- sparse-infill cells legitimately never
/// read near 1.0 per-cell (see this module's own doc comment / the
/// design spec for why), so without this filter a healthy sparse-infill
/// interior would visually swamp genuine wall-shell defects.
///
/// Color is auto-normalized against the actual min/max ratio observed
/// across every cell in this run -- see [`ratio_to_color`]'s doc comment
/// for why a fixed absolute scale doesn't work here.
///
/// `shrink_factor` (`(0.0, 1.0]`) draws each cube at
/// `cell_size * shrink_factor` rather than the full cell size, leaving a
/// visible gap between adjacent cells.
pub fn build_volume_audit_cells(
    grid: &VolumeAuditGrid,
    deviation_threshold: f64,
    shrink_factor: f64,
) -> Vec<VolumeAuditCellVertex> {
    // Deviation from the brief: `grid.cell_size()` (method call), not
    // `grid.cell_size` (field access) -- see this module's own report
    // for why (the field is `pub(crate)` in `manifold-core`, invisible
    // from this separate `manifold-gui` crate). A matching public
    // accessor was added to `VolumeAuditGrid` alongside this task.
    let half_size = grid.cell_size() * shrink_factor * 0.5;
    let mut vertices = Vec::new();

    let ratios = grid.cell_ratios_for_display();
    let (min_ratio, max_ratio) = ratios
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, r)| {
            (lo.min(r), hi.max(r))
        });

    for (idx, ratio) in ratios {
        if (ratio - 1.0).abs() < deviation_threshold {
            continue;
        }
        push_cube(
            &mut vertices,
            grid.cell_center(idx),
            half_size,
            ratio_to_color(ratio, min_ratio, max_ratio),
        );
    }

    for idx in grid.extrusion_outside_mesh_cells() {
        push_cube(
            &mut vertices,
            grid.cell_center(idx),
            half_size,
            OUTSIDE_MESH_COLOR,
        );
    }

    vertices
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_core::toolpath::{MoveKind, Path, Segment};
    use manifold_core::{mesh::Mesh, volume_audit::audit_extrusion_volume, SlicerConfig};

    /// Same box-mesh fixture shape as `manifold_core::volume_audit`'s own
    /// tests (independently built here since that helper is private to
    /// that crate's test module).
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

    fn straight_extruding_path(
        start: DVec3,
        end: DVec3,
        kind: MoveKind,
        bead_area: f64,
        config: &SlicerConfig,
    ) -> Path {
        let distance = start.distance(end);
        let filament_area =
            manifold_core::extrusion::filament_cross_section_area(config.filament_diameter);
        let extrusion_length =
            manifold_core::extrusion::segment_extrusion_length(distance, bead_area, filament_area);
        Path {
            points: vec![start, end],
            segments: vec![Segment {
                kind,
                extrusion_length,
                line_width: bead_area / config.layer_height,
                ..Segment::default()
            }],
            tool: manifold_core::ids::ToolId(0),
            object: manifold_core::ids::ObjectId(0),
        }
    }

    #[test]
    fn build_volume_audit_cells_emits_36_vertices_per_qualifying_cell() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);

        // Threshold of 0.0 keeps everything with any deviation at all --
        // a wall bead's own cells should qualify (their ratio is never
        // exactly 1.0 for a hand-built single-segment fixture).
        let vertices = build_volume_audit_cells(&grid, 0.0, 0.9);
        assert!(
            !vertices.is_empty(),
            "a grid with real wall extrusion should produce at least one cube"
        );
        assert_eq!(
            vertices.len() % 36,
            0,
            "every cube must contribute exactly 36 vertices (12 triangles), got {} total",
            vertices.len()
        );
    }

    #[test]
    fn build_volume_audit_cells_respects_the_deviation_threshold() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        let path = straight_extruding_path(
            DVec3::new(0.3, 2.0, 1.0),
            DVec3::new(0.3, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&path), &config, cell_size);

        let permissive = build_volume_audit_cells(&grid, 0.0, 0.9);
        // A threshold far above any real cell's deviation must hide
        // every ratio-based cube.
        let strict = build_volume_audit_cells(&grid, 1e6, 0.9);
        assert!(
            strict.len() < permissive.len(),
            "an extreme deviation_threshold should hide ratio-based cubes: \
             permissive={}, strict={}",
            permissive.len(),
            strict.len()
        );
    }

    #[test]
    fn build_volume_audit_cells_never_hides_outside_the_mesh_defects() {
        let mesh = box_mesh(DVec3::ZERO, DVec3::new(2.0, 6.0, 2.0));
        let config = SlicerConfig::default();
        let cell_size = 0.5;
        // Same stray-bead fixture as `manifold_core::volume_audit`'s own
        // `extrusion_outside_the_mesh_is_caught_only_by_its_own_dedicated_query`.
        let stray = straight_extruding_path(
            DVec3::new(-0.25, 2.0, 1.0),
            DVec3::new(-0.25, 4.0, 1.0),
            MoveKind::WallOuter,
            config.wall_line_width * config.layer_height,
            &config,
        );
        let grid = audit_extrusion_volume(&mesh, std::slice::from_ref(&stray), &config, cell_size);
        assert!(
            !grid.extrusion_outside_mesh_cells().is_empty(),
            "test fixture must produce an outside-the-mesh cell -- otherwise this test is vacuous"
        );

        // Even an extreme deviation_threshold must not hide the
        // outside-mesh defect cube.
        let vertices = build_volume_audit_cells(&grid, 1e6, 0.9);
        assert!(
            vertices.iter().any(|v| v.color == OUTSIDE_MESH_COLOR),
            "outside-the-mesh defect cubes must never be hidden by deviation_threshold"
        );
    }

    #[test]
    fn ratio_to_color_maps_extremes_to_the_scale_endpoints() {
        // Compare RGB against `scalar_to_color`'s reference gradient; alpha
        // is deliberately CUBE_ALPHA (not scalar_to_color's own 1.0), see
        // `ratio_to_color_uses_the_shared_cube_alpha_not_scalar_to_colors_own`.
        let rgb = |c: [f32; 4]| [c[0], c[1], c[2]];

        // Healthy ratio (== 1.0) is always exact green, regardless of spread.
        assert_eq!(
            rgb(ratio_to_color(1.0, 0.2, 3.0)),
            rgb(crate::toolpath_view::scalar_to_color(0.5))
        );
        // The observed maximum maps to the pure red end (t = 1.0).
        assert_eq!(
            rgb(ratio_to_color(3.0, 0.2, 3.0)),
            rgb(crate::toolpath_view::scalar_to_color(1.0))
        );
        // The observed minimum maps to the pure blue end (t = 0.0).
        assert_eq!(
            rgb(ratio_to_color(0.2, 0.2, 3.0)),
            rgb(crate::toolpath_view::scalar_to_color(0.0))
        );
    }

    #[test]
    fn ratio_to_color_uses_the_shared_cube_alpha_not_scalar_to_colors_own() {
        // `scalar_to_color` itself always returns alpha 1.0 (used elsewhere,
        // e.g. toolpath heatmaps, at full opacity); `ratio_to_color` must
        // override that with the shared `CUBE_ALPHA` so audit cubes render
        // semi-transparently.
        assert_eq!(ratio_to_color(1.0, 0.2, 3.0)[3], CUBE_ALPHA);
        assert_ne!(CUBE_ALPHA, 1.0, "test is vacuous if CUBE_ALPHA is opaque");
    }

    #[test]
    fn ratio_to_color_spreads_a_tight_cluster_across_the_full_range() {
        // The bug this normalization fixes: a fixed absolute scale made a
        // tight cluster of ratios (e.g. 1.55-1.65, all landing near one
        // shade of orange) visually indistinguishable from each other,
        // hiding real defects among routine sparse-infill noise. Once
        // normalized against their own min/max, the two ends of even a
        // tight cluster must still be visually distinct.
        let low = ratio_to_color(1.55, 1.55, 1.65);
        let high = ratio_to_color(1.65, 1.55, 1.65);
        assert_ne!(
            low, high,
            "a tight, non-trivial ratio spread must still produce visually distinct colors"
        );
    }

    #[test]
    fn ratio_to_color_flat_population_at_healthy_ratio_is_green() {
        // If every displayed cell happens to have ratio exactly 1.0 (the
        // designed special case, independent of min/max), the result
        // must be exact green.
        let flat_healthy = ratio_to_color(1.0, 1.0, 1.0);
        let [r, g, b, _] = crate::toolpath_view::scalar_to_color(0.5);
        assert_eq!(
            [flat_healthy[0], flat_healthy[1], flat_healthy[2]],
            [r, g, b]
        );
    }

    #[test]
    fn ratio_to_color_flat_population_away_from_healthy_never_panics_or_produces_nan() {
        // If every displayed cell shares the SAME non-1.0 ratio (min ==
        // max == ratio), that single value is trivially both the min
        // and the max of its own population, so it legitimately renders
        // at the corresponding scale extreme rather than green -- what
        // this test actually checks is that the division by
        // `max_ratio - 1.0` (or the mirrored underfill formula) never
        // divides by zero or produces NaN in that case.
        let overfilled_flat = ratio_to_color(1.3, 1.3, 1.3);
        assert!(overfilled_flat.iter().all(|c| c.is_finite()));
        let underfilled_flat = ratio_to_color(0.4, 0.4, 0.4);
        assert!(underfilled_flat.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn ratio_to_color_defensive_branches_never_panic_on_inconsistent_input() {
        // These inputs are internally inconsistent (the ratio falls
        // outside its own claimed min/max) and should never arise from
        // a real `build_volume_audit_cells` call, since min/max are
        // always derived from the exact same population being colored
        // -- but the function must still degrade safely (no panic, no
        // NaN) rather than assume its caller is well-behaved.
        let a = ratio_to_color(0.5, 1.5, 2.0);
        assert!(a.iter().all(|c| c.is_finite()));
        let b = ratio_to_color(2.0, 0.5, 0.8);
        assert!(b.iter().all(|c| c.is_finite()));
    }
}
