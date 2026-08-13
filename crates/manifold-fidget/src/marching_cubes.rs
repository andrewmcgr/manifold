//! Hand-rolled marching cubes isosurface extraction, generic over any
//! [`ScalarField`] (works unmodified for [`TreeField`](crate::TreeField) and
//! the future mesh-derived `MeshSdf`).
//!
//! See `MESH_SDF_VISUALIZATION.md` Phase B for the design this module
//! implements. This is the classic cube-configuration/edge-table algorithm
//! (Lorensen & Cline 1987 / the widely used Paul Bourke lookup tables) —
//! no new crate dependency.
//!
//! ## Grid convention
//!
//! The sample grid spans the axis-aligned box `[min, max]` and is divided
//! into `resolution` cells per axis, i.e. `resolution + 1` sample points
//! per axis (a cell count, not a sample-point count or a fixed cell size).
//!
//! ## Output shape
//!
//! [`extract_isosurface`] returns a triangle soup: a flat `Vec<Vertex>`
//! where every consecutive group of 3 entries forms one triangle. There is
//! no index buffer and no de-duplication of shared vertices — this is the
//! simplest shape for a direct-upload consumer (see `MESH_SDF_VISUALIZATION.md`
//! Phase D, `manifold-gui`'s mesh upload path) and keeps this module free of
//! any topology bookkeeping.

use crate::ScalarField;
use glam::DVec3;

/// One vertex of the extracted isosurface: a position on the surface and
/// the field's (normalized) gradient at that position, used directly as
/// the shading normal — no separate face-normal recomputation pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: DVec3,
    pub normal: DVec3,
}

/// Extracts the isosurface `field(p) == iso` of `field` over the axis-aligned
/// box `[min, max]`, sampled on a grid of `resolution` cells per axis
/// (`resolution + 1` sample points per axis; `resolution` must be >= 1).
///
/// Returns a triangle soup (see module docs): 3 consecutive [`Vertex`]
/// entries form one triangle. Returns an empty `Vec` if `resolution == 0`
/// or the field does not cross `iso` anywhere in the box.
pub fn extract_isosurface<F: ScalarField>(
    field: &F,
    min: DVec3,
    max: DVec3,
    resolution: usize,
    iso: f64,
) -> Vec<Vertex> {
    if resolution == 0 {
        return Vec::new();
    }

    let dims = resolution + 1;
    let cell_size = DVec3::new(
        (max.x - min.x) / resolution as f64,
        (max.y - min.y) / resolution as f64,
        (max.z - min.z) / resolution as f64,
    );

    // Cache samples over the whole grid: (dims)^3 evaluations, one
    // ScalarField::sample per grid point.
    let mut values = vec![0.0_f64; dims * dims * dims];
    let idx = |xi: usize, yi: usize, zi: usize| -> usize { (zi * dims + yi) * dims + xi };
    for zi in 0..dims {
        for yi in 0..dims {
            for xi in 0..dims {
                let p = min
                    + DVec3::new(
                        xi as f64 * cell_size.x,
                        yi as f64 * cell_size.y,
                        zi as f64 * cell_size.z,
                    );
                values[idx(xi, yi, zi)] = field.sample(p).value;
            }
        }
    }

    // Corner offsets in grid-index space, in the standard marching-cubes
    // corner order.
    const CORNER_OFFSETS: [(usize, usize, usize); 8] = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    // Edge -> corner index pairs, matching CORNER_OFFSETS / the standard
    // tables below.
    const EDGE_CORNERS: [(usize, usize); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];

    let mut vertices = Vec::new();

    for zi in 0..resolution {
        for yi in 0..resolution {
            for xi in 0..resolution {
                let corner_pos: [DVec3; 8] = std::array::from_fn(|c| {
                    let (dx, dy, dz) = CORNER_OFFSETS[c];
                    min + DVec3::new(
                        (xi + dx) as f64 * cell_size.x,
                        (yi + dy) as f64 * cell_size.y,
                        (zi + dz) as f64 * cell_size.z,
                    )
                });
                let corner_val: [f64; 8] = std::array::from_fn(|c| {
                    let (dx, dy, dz) = CORNER_OFFSETS[c];
                    values[idx(xi + dx, yi + dy, zi + dz)]
                });

                let mut cube_index = 0usize;
                for (c, &v) in corner_val.iter().enumerate() {
                    if v < iso {
                        cube_index |= 1 << c;
                    }
                }

                let edge_flags = EDGE_TABLE[cube_index];
                if edge_flags == 0 {
                    continue;
                }

                // Interpolated vertex position (and field-gradient normal)
                // for each of the 12 cube edges, computed lazily only for
                // edges this configuration actually crosses.
                let mut edge_vertex: [Option<Vertex>; 12] = [None; 12];
                for (e, &(c0, c1)) in EDGE_CORNERS.iter().enumerate() {
                    if edge_flags & (1 << e) == 0 {
                        continue;
                    }
                    let v0 = corner_val[c0];
                    let v1 = corner_val[c1];
                    let denom = v1 - v0;
                    let t = if denom.abs() <= f64::EPSILON {
                        0.5
                    } else {
                        (iso - v0) / denom
                    };
                    let t = t.clamp(0.0, 1.0);
                    let position = corner_pos[c0].lerp(corner_pos[c1], t);
                    let sample = field.sample(position);
                    let normal = if sample.gradient.length_squared() > f64::EPSILON {
                        sample.gradient.normalize()
                    } else {
                        DVec3::ZERO
                    };
                    edge_vertex[e] = Some(Vertex { position, normal });
                }

                for tri in TRI_TABLE[cube_index].chunks(3) {
                    if tri.len() < 3 || tri[0] < 0 {
                        break;
                    }
                    for &e in tri {
                        vertices.push(
                            edge_vertex[e as usize]
                                .expect("TRI_TABLE only references edges flagged in EDGE_TABLE"),
                        );
                    }
                }
            }
        }
    }

    vertices
}

/// Standard marching-cubes edge table: for each of the 256 cube
/// (inside/outside corner) configurations, a 12-bit mask of which of the 12
/// cube edges the isosurface crosses.
#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

/// Standard marching-cubes triangle table: for each of the 256 cube
/// configurations, up to 5 triangles (15 edge indices) describing how to
/// triangulate the isosurface within that cube, terminated by `-1`.
#[rustfmt::skip]
const TRI_TABLE: [[i8; 16]; 256] = include!("marching_cubes_tri_table.rs.in");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sphere_tree, TreeField};

    #[test]
    fn sphere_vertices_lie_on_surface() {
        let radius = 1.0;
        let field = TreeField::new(sphere_tree(radius));
        let resolution = 24;
        let bound = radius * 1.5;
        let min = DVec3::splat(-bound);
        let max = DVec3::splat(bound);

        let vertices = extract_isosurface(&field, min, max, resolution, 0.0);
        assert!(
            !vertices.is_empty(),
            "sphere isosurface should not be empty"
        );
        assert_eq!(
            vertices.len() % 3,
            0,
            "triangle soup length must be a multiple of 3"
        );

        // Tolerance relative to the grid cell size: a vertex can be off the
        // true surface by up to roughly one cell diagonal's worth of linear
        // interpolation error.
        let cell_size = 2.0 * bound / resolution as f64;
        let tolerance = 1.5 * cell_size;

        // Spot-check a sample of vertices (every 7th, to keep the test
        // fast while still covering many triangles).
        for vertex in vertices.iter().step_by(7) {
            let distance = vertex.position.length();
            assert!(
                (distance - radius).abs() <= tolerance,
                "vertex at {:?} has distance {distance} from origin, expected ~{radius} (tol {tolerance})",
                vertex.position
            );
            assert!(
                (vertex.normal.length() - 1.0).abs() <= 1e-6,
                "vertex normal should be normalized, got length {}",
                vertex.normal.length()
            );
        }
    }

    #[test]
    fn zero_resolution_returns_empty() {
        let field = TreeField::new(sphere_tree(1.0));
        let vertices = extract_isosurface(&field, DVec3::splat(-2.0), DVec3::splat(2.0), 0, 0.0);
        assert!(vertices.is_empty());
    }
}
