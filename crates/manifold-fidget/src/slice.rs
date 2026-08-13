//! 2D grid sampling of a [`ScalarField`] over an arbitrary plane, for slice
//! heatmap visualization (see `MESH_SDF_VISUALIZATION.md`).
//!
//! Pure sampling logic: no GUI/wgpu/egui dependency, independently testable.

use glam::DVec3;

use crate::ScalarField;

/// A 2D grid of scalar-field values sampled over a plane, row-major:
/// `values[row * width + col]` is the sample at grid cell `(col, row)`,
/// where `row` increases along `basis2` and `col` increases along
/// `basis1` (see [`sample_plane`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SliceGrid {
    pub values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

/// Samples `field` over a 2D grid on the plane defined by `origin` and the
/// two basis vectors `basis1`/`basis2`.
///
/// The plane is parameterized by `(u, v)` with `u` ranging over
/// `[-half_width, half_width]` along `basis1` and `v` ranging over
/// `[-half_height, half_height]` along `basis2`, where `half_width =
/// width / 2.0` and `half_height = height / 2.0` (`width`/`height` given in
/// world units). For each grid cell, the world-space sample point is the
/// cell *center*:
///
/// ```text
/// p = origin + u * basis1 + v * basis2
/// ```
///
/// `basis1`/`basis2` are expected to be orthonormal (unit length, mutually
/// perpendicular) — the simple case from the design doc. Passing
/// non-orthonormal (scaled/skewed) vectors still works mechanically (the
/// grid is simply sheared/scaled in world space accordingly) but the
/// `width`/`height` extents then no longer correspond to world-space
/// distances along an orthogonal frame.
///
/// `resolution_u`/`resolution_v` are the number of grid cells along
/// `basis1`/`basis2` respectively (must both be `> 0`, else the returned
/// grid is empty).
///
/// Only `FieldSample::value` is used; gradients aren't needed for a slice
/// heatmap, but callers wanting per-pixel normals for lighting could sample
/// `field.sample(p)` directly at the same `p` computed here.
#[allow(clippy::too_many_arguments)]
pub fn sample_plane<F: ScalarField + ?Sized>(
    field: &F,
    origin: DVec3,
    basis1: DVec3,
    basis2: DVec3,
    width: f64,
    height: f64,
    resolution_u: usize,
    resolution_v: usize,
) -> SliceGrid {
    let mut values = Vec::with_capacity(resolution_u * resolution_v);

    let half_width = width / 2.0;
    let half_height = height / 2.0;

    for row in 0..resolution_v {
        // Cell-center v-coordinate: map row index to [-half_height, half_height].
        let v = if resolution_v > 1 {
            -half_height + (row as f64 + 0.5) * height / resolution_v as f64
        } else {
            0.0
        };
        for col in 0..resolution_u {
            let u = if resolution_u > 1 {
                -half_width + (col as f64 + 0.5) * width / resolution_u as f64
            } else {
                0.0
            };
            let p = origin + u * basis1 + v * basis2;
            let sample = field.sample(p);
            values.push(sample.value as f32);
        }
    }

    SliceGrid {
        values,
        width: resolution_u,
        height: resolution_v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sphere_tree, TreeField};

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn dimensions_match_resolution() {
        let field = TreeField::new(sphere_tree(1.0));
        let grid = sample_plane(&field, DVec3::ZERO, DVec3::X, DVec3::Y, 4.0, 4.0, 8, 6);
        assert_eq!(grid.width, 8);
        assert_eq!(grid.height, 6);
        assert_eq!(grid.values.len(), grid.width * grid.height);
    }

    #[test]
    fn equatorial_slice_center_reads_negative_radius() {
        // XY plane at Z=0 passes through the sphere's center; the grid
        // center cell (odd resolution -> exact center at u=v=0) should
        // read ~ -radius (0 - radius).
        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let grid = sample_plane(&field, DVec3::ZERO, DVec3::X, DVec3::Y, 4.0, 4.0, 5, 5);

        // Center cell is at index (2, 2) for a 5x5 grid (0-indexed).
        let center = grid.values[2 * grid.width + 2];
        assert!(
            approx_eq(center, -radius as f32, 1e-3),
            "expected ~{}, got {}",
            -radius,
            center
        );
    }

    #[test]
    fn equatorial_slice_matches_expected_distance_at_known_offset() {
        // Sample at u = radius + 1 along basis1 (still on the Z=0 plane, so
        // p = (radius + 1, 0, 0)); expected value is |p| - radius = 1.0.
        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        // Grid spans u in [-1, 1] over 2 cells -> cell centers at u = -0.5, 0.5.
        // Use width/resolution chosen so a cell center lands exactly at
        // u = radius + 1 = 2.0: width = 4.0, resolution_u = 2 gives cell
        // centers at u = -1.0 and u = 1.0 (since half_width=2,
        // step=4/2=2, centers = -2 + 0.5*2 = -1, -2 + 1.5*2 = 1).
        let grid = sample_plane(&field, DVec3::ZERO, DVec3::X, DVec3::Y, 4.0, 1.0, 2, 1);
        // col=1 -> u = -2.0 + 1.5 * 2.0 = 1.0, v = 0 (single row).
        let p = DVec3::new(1.0, 0.0, 0.0);
        let expected = (p.length() - radius) as f32;
        let sampled = grid.values[1];
        assert!(
            approx_eq(sampled, expected, 1e-3),
            "expected ~{}, got {}",
            expected,
            sampled
        );
    }

    #[test]
    fn zero_resolution_yields_empty_grid() {
        let field = TreeField::new(sphere_tree(1.0));
        let grid = sample_plane(&field, DVec3::ZERO, DVec3::X, DVec3::Y, 4.0, 4.0, 0, 5);
        assert_eq!(grid.values.len(), 0);
        assert_eq!(grid.width, 0);
    }
}
