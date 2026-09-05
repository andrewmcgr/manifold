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
use rayon::prelude::*;

/// One vertex of the extracted isosurface: a position on the surface and
/// the field's (normalized) gradient at that position, used directly as
/// the shading normal — no separate face-normal recomputation pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: DVec3,
    pub normal: DVec3,
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

/// Extracts the isosurface `field(p) == iso` of `field` over the axis-aligned
/// box `[min, max]`, sampled on a grid of `resolution` cells per axis
/// (`resolution + 1` sample points per axis; `resolution` must be >= 1).
///
/// Returns a triangle soup (see module docs): 3 consecutive [`Vertex`]
/// entries form one triangle. Returns an empty `Vec` if `resolution == 0`
/// or the field does not cross `iso` anywhere in the box.
///
/// Both the grid-sampling pass and the per-cell triangle-generation pass run
/// in parallel across all available cores (via `rayon`): every grid point's
/// `field.sample` call and every cell's triangle emission only read `field`
/// (shared, immutable) and write their own independent output slot, so
/// there's no cross-point/cross-cell dependency to serialize on. Output is a
/// triangle soup with no ordering contract, so scrambled cell-completion
/// order (inherent to parallel iteration) doesn't affect correctness.
/// Extracts only the triangle positions `field(p) == iso` of `field` over
/// `[min, max]`, without computing per-vertex normals.
///
/// Much faster than [`extract_isosurface`] when normals are not needed (e.g.
/// for slicing contour walks in `manifold-core::slicing`), because it only
/// samples `field` at the regular grid points and avoids secondary `sample()`
/// queries on every interpolated surface vertex.
pub fn extract_isosurface_positions<F: ScalarField + Sync>(
    field: &F,
    min: DVec3,
    max: DVec3,
    resolution: usize,
    iso: f64,
) -> Vec<DVec3> {
    if resolution == 0 {
        return Vec::new();
    }

    let dims = resolution + 1;
    let cell_size = DVec3::new(
        (max.x - min.x) / resolution as f64,
        (max.y - min.y) / resolution as f64,
        (max.z - min.z) / resolution as f64,
    );

    let idx = |xi: usize, yi: usize, zi: usize| -> usize { (zi * dims + yi) * dims + xi };
    let values: Vec<f64> = (0..dims * dims * dims)
        .into_par_iter()
        .map(|flat| {
            let xi = flat % dims;
            let yi = (flat / dims) % dims;
            let zi = flat / (dims * dims);
            let p = min
                + DVec3::new(
                    xi as f64 * cell_size.x,
                    yi as f64 * cell_size.y,
                    zi as f64 * cell_size.z,
                );
            field.sample(p).value
        })
        .collect();

    (0..resolution * resolution * resolution)
        .into_par_iter()
        .flat_map_iter(|flat| {
            let xi = flat % resolution;
            let yi = (flat / resolution) % resolution;
            let zi = flat / (resolution * resolution);

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
            let mut cell_positions = Vec::new();
            if edge_flags == 0 {
                return cell_positions;
            }

            let mut edge_vertex: [Option<DVec3>; 12] = [None; 12];
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
                edge_vertex[e] = Some(corner_pos[c0].lerp(corner_pos[c1], t));
            }

            for tri in TRI_TABLE[cube_index].chunks(3) {
                if tri.len() < 3 || tri[0] < 0 {
                    break;
                }
                for &e in tri {
                    cell_positions.push(
                        edge_vertex[e as usize]
                            .expect("TRI_TABLE only references edges flagged in EDGE_TABLE"),
                    );
                }
            }
            cell_positions
        })
        .collect()
}

pub fn extract_isosurface<F: ScalarField + Sync>(
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
    let idx = |xi: usize, yi: usize, zi: usize| -> usize { (zi * dims + yi) * dims + xi };
    let values: Vec<f64> = (0..dims * dims * dims)
        .into_par_iter()
        .map(|flat| {
            let xi = flat % dims;
            let yi = (flat / dims) % dims;
            let zi = flat / (dims * dims);
            let p = min
                + DVec3::new(
                    xi as f64 * cell_size.x,
                    yi as f64 * cell_size.y,
                    zi as f64 * cell_size.z,
                );
            field.sample(p).value
        })
        .collect();

    (0..resolution * resolution * resolution)
        .into_par_iter()
        .flat_map_iter(|flat| {
            let xi = flat % resolution;
            let yi = (flat / resolution) % resolution;
            let zi = flat / (resolution * resolution);

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
            let mut cell_vertices = Vec::new();
            if edge_flags == 0 {
                return cell_vertices;
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
                    cell_vertices.push(
                        edge_vertex[e as usize]
                            .expect("TRI_TABLE only references edges flagged in EDGE_TABLE"),
                    );
                }
            }
            cell_vertices
        })
        .collect()
}

/// Extracts only the triangle positions `field(p) == iso` of `field` over
/// `[min, max]` using a hierarchical block-sparse narrow-band grid with target `cell_size`.
///
/// Unlike [`extract_isosurface_positions`], this does not allocate a dense 3D volume grid
/// ($O(N^3)$), but instead adaptively discovers and evaluates only $8 \times 8 \times 8$
/// leaf blocks within the narrow band of the surface ($O(N^2)$ surface area). This enables
/// sub-0.1mm cell resolution across large meshes without memory exhaustion or thin-feature
/// voxel aliasing voids.
///
/// `extra_margin` widens the super-block/leaf-block culling bound beyond the plain
/// Lipschitz-1 assumption (`|sample(center) - iso| <= radius + cell_diag`). Pass `0.0`
/// for a field known to be a true 1-Lipschitz signed distance function everywhere. Pass a
/// positive value for fields that only satisfy that bound *away* from certain regions —
/// e.g. an SDF built from a reduced subset of a mesh's faces (such as one that excludes
/// bed-contact floor triangles so wall passes can reach the bed cleanly): near the
/// boundary between an excluded face and an included one, the "nearest included face"
/// (and its pseudonormal-derived sign) can change discontinuously as position moves
/// smoothly, producing a real jump in `sample(p).value` larger than physical distance
/// moved. Without a margin covering that worst-case jump, blocks that do contain a true
/// isosurface crossing can be wrongly culled as "provably empty", leaving coverage holes
/// in the extracted mesh near those exclusion boundaries.
pub fn extract_sparse_isosurface_positions<F: ScalarField + Sync>(
    field: &F,
    min: DVec3,
    max: DVec3,
    cell_size: f64,
    iso: f64,
    extra_margin: f64,
) -> Vec<DVec3> {
    if cell_size <= 0.0 || min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return Vec::new();
    }

    const BLOCK_SIZE: usize = 8;
    const SUPER_BLOCK_BLOCKS: usize = 4;
    const SUPER_BLOCK_SIZE: usize = BLOCK_SIZE * SUPER_BLOCK_BLOCKS; // 32 cells

    let extent = max - min;
    let n_cells_x = ((extent.x / cell_size).ceil() as usize).max(1);
    let n_cells_y = ((extent.y / cell_size).ceil() as usize).max(1);
    let n_cells_z = ((extent.z / cell_size).ceil() as usize).max(1);

    let dx = extent.x / n_cells_x as f64;
    let dy = extent.y / n_cells_y as f64;
    let dz = extent.z / n_cells_z as f64;
    let cell_diag = DVec3::new(dx, dy, dz).length();

    let n_blocks_x = n_cells_x.div_ceil(BLOCK_SIZE);
    let n_blocks_y = n_cells_y.div_ceil(BLOCK_SIZE);
    let n_blocks_z = n_cells_z.div_ceil(BLOCK_SIZE);

    let n_super_x = n_blocks_x.div_ceil(SUPER_BLOCK_BLOCKS);
    let n_super_y = n_blocks_y.div_ceil(SUPER_BLOCK_BLOCKS);
    let n_super_z = n_blocks_z.div_ceil(SUPER_BLOCK_BLOCKS);

    let total_super_blocks = n_super_x * n_super_y * n_super_z;

    // Phase 1: Filter super-blocks in parallel
    let active_blocks: Vec<[usize; 3]> = (0..total_super_blocks)
        .into_par_iter()
        .flat_map_iter(|s_flat| {
            let sx = s_flat % n_super_x;
            let sy = (s_flat / n_super_x) % n_super_y;
            let sz = s_flat / (n_super_x * n_super_y);

            let x0 = sx * SUPER_BLOCK_SIZE;
            let x1 = ((sx + 1) * SUPER_BLOCK_SIZE).min(n_cells_x);
            let y0 = sy * SUPER_BLOCK_SIZE;
            let y1 = ((sy + 1) * SUPER_BLOCK_SIZE).min(n_cells_y);
            let z0 = sz * SUPER_BLOCK_SIZE;
            let z1 = ((sz + 1) * SUPER_BLOCK_SIZE).min(n_cells_z);

            let sb_min = min + DVec3::new(x0 as f64 * dx, y0 as f64 * dy, z0 as f64 * dz);
            let sb_max = min + DVec3::new(x1 as f64 * dx, y1 as f64 * dy, z1 as f64 * dz);
            let sb_center = (sb_min + sb_max) * 0.5;
            let sb_radius = (sb_max - sb_min).length() * 0.5;

            // Bounding test: Lipschitz / distance-field bound with margin
            if (field.sample(sb_center).value - iso).abs() > sb_radius + cell_diag + extra_margin {
                return Vec::new();
            }

            // Test leaf blocks within active super-block
            let bx_start = sx * SUPER_BLOCK_BLOCKS;
            let bx_end = ((sx + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_x);
            let by_start = sy * SUPER_BLOCK_BLOCKS;
            let by_end = ((sy + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_y);
            let bz_start = sz * SUPER_BLOCK_BLOCKS;
            let bz_end = ((sz + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_z);

            let mut local_blocks = Vec::new();
            for bz in bz_start..bz_end {
                for by in by_start..by_end {
                    for bx in bx_start..bx_end {
                        let lx0 = bx * BLOCK_SIZE;
                        let lx1 = ((bx + 1) * BLOCK_SIZE).min(n_cells_x);
                        let ly0 = by * BLOCK_SIZE;
                        let ly1 = ((by + 1) * BLOCK_SIZE).min(n_cells_y);
                        let lz0 = bz * BLOCK_SIZE;
                        let lz1 = ((bz + 1) * BLOCK_SIZE).min(n_cells_z);

                        let b_min =
                            min + DVec3::new(lx0 as f64 * dx, ly0 as f64 * dy, lz0 as f64 * dz);
                        let b_max =
                            min + DVec3::new(lx1 as f64 * dx, ly1 as f64 * dy, lz1 as f64 * dz);
                        let b_center = (b_min + b_max) * 0.5;
                        let b_radius = (b_max - b_min).length() * 0.5;

                        if (field.sample(b_center).value - iso).abs()
                            <= b_radius + cell_diag + extra_margin
                        {
                            local_blocks.push([bx, by, bz]);
                        }
                    }
                }
            }
            local_blocks
        })
        .collect();

    // Phase 2: Process all active leaf blocks in parallel
    active_blocks
        .into_par_iter()
        .flat_map_iter(|[bx, by, bz]| {
            let x0 = bx * BLOCK_SIZE;
            let x1 = ((bx + 1) * BLOCK_SIZE).min(n_cells_x);
            let y0 = by * BLOCK_SIZE;
            let y1 = ((by + 1) * BLOCK_SIZE).min(n_cells_y);
            let z0 = bz * BLOCK_SIZE;
            let z1 = ((bz + 1) * BLOCK_SIZE).min(n_cells_z);

            let wx = x1 - x0;
            let wy = y1 - y0;
            let wz = z1 - z0;

            let dims_x = wx + 1;
            let dims_y = wy + 1;
            let dims_z = wz + 1;

            let mut values = vec![0.0f64; dims_x * dims_y * dims_z];
            for zi in 0..dims_z {
                for yi in 0..dims_y {
                    for xi in 0..dims_x {
                        let p = min
                            + DVec3::new(
                                (x0 + xi) as f64 * dx,
                                (y0 + yi) as f64 * dy,
                                (z0 + zi) as f64 * dz,
                            );
                        values[(zi * dims_y + yi) * dims_x + xi] = field.sample(p).value;
                    }
                }
            }

            let mut block_triangles = Vec::new();
            for cz in 0..wz {
                for cy in 0..wy {
                    for cx in 0..wx {
                        let corner_pos: [DVec3; 8] = std::array::from_fn(|c| {
                            let (cdx, cdy, cdz) = CORNER_OFFSETS[c];
                            min + DVec3::new(
                                (x0 + cx + cdx) as f64 * dx,
                                (y0 + cy + cdy) as f64 * dy,
                                (z0 + cz + cdz) as f64 * dz,
                            )
                        });
                        let corner_val: [f64; 8] = std::array::from_fn(|c| {
                            let (cdx, cdy, cdz) = CORNER_OFFSETS[c];
                            values[((cz + cdz) * dims_y + (cy + cdy)) * dims_x + (cx + cdx)]
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

                        let mut edge_vertex: [Option<DVec3>; 12] = [None; 12];
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
                            edge_vertex[e] = Some(corner_pos[c0].lerp(corner_pos[c1], t));
                        }

                        for tri in TRI_TABLE[cube_index].chunks(3) {
                            if tri.len() < 3 || tri[0] < 0 {
                                break;
                            }
                            for &e in tri {
                                block_triangles.push(edge_vertex[e as usize].expect(
                                    "TRI_TABLE only references edges flagged in EDGE_TABLE",
                                ));
                            }
                        }
                    }
                }
            }
            block_triangles
        })
        .collect()
}

/// Extracts vertices (position + normal) at `field(p) == iso` of `field` over
/// `[min, max]` using a hierarchical block-sparse narrow-band grid with target `cell_size`.
pub fn extract_sparse_isosurface<F: ScalarField + Sync>(
    field: &F,
    min: DVec3,
    max: DVec3,
    cell_size: f64,
    iso: f64,
) -> Vec<Vertex> {
    if cell_size <= 0.0 || min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return Vec::new();
    }

    const BLOCK_SIZE: usize = 8;
    const SUPER_BLOCK_BLOCKS: usize = 4;
    const SUPER_BLOCK_SIZE: usize = BLOCK_SIZE * SUPER_BLOCK_BLOCKS;

    let extent = max - min;
    let n_cells_x = ((extent.x / cell_size).ceil() as usize).max(1);
    let n_cells_y = ((extent.y / cell_size).ceil() as usize).max(1);
    let n_cells_z = ((extent.z / cell_size).ceil() as usize).max(1);

    let dx = extent.x / n_cells_x as f64;
    let dy = extent.y / n_cells_y as f64;
    let dz = extent.z / n_cells_z as f64;
    let cell_diag = DVec3::new(dx, dy, dz).length();

    let n_blocks_x = n_cells_x.div_ceil(BLOCK_SIZE);
    let n_blocks_y = n_cells_y.div_ceil(BLOCK_SIZE);
    let n_blocks_z = n_cells_z.div_ceil(BLOCK_SIZE);

    let n_super_x = n_blocks_x.div_ceil(SUPER_BLOCK_BLOCKS);
    let n_super_y = n_blocks_y.div_ceil(SUPER_BLOCK_BLOCKS);
    let n_super_z = n_blocks_z.div_ceil(SUPER_BLOCK_BLOCKS);

    let total_super_blocks = n_super_x * n_super_y * n_super_z;

    let active_blocks: Vec<[usize; 3]> = (0..total_super_blocks)
        .into_par_iter()
        .flat_map_iter(|s_flat| {
            let sx = s_flat % n_super_x;
            let sy = (s_flat / n_super_x) % n_super_y;
            let sz = s_flat / (n_super_x * n_super_y);

            let x0 = sx * SUPER_BLOCK_SIZE;
            let x1 = ((sx + 1) * SUPER_BLOCK_SIZE).min(n_cells_x);
            let y0 = sy * SUPER_BLOCK_SIZE;
            let y1 = ((sy + 1) * SUPER_BLOCK_SIZE).min(n_cells_y);
            let z0 = sz * SUPER_BLOCK_SIZE;
            let z1 = ((sz + 1) * SUPER_BLOCK_SIZE).min(n_cells_z);

            let sb_min = min + DVec3::new(x0 as f64 * dx, y0 as f64 * dy, z0 as f64 * dz);
            let sb_max = min + DVec3::new(x1 as f64 * dx, y1 as f64 * dy, z1 as f64 * dz);
            let sb_center = (sb_min + sb_max) * 0.5;
            let sb_radius = (sb_max - sb_min).length() * 0.5;

            if (field.sample(sb_center).value - iso).abs() > sb_radius + cell_diag {
                return Vec::new();
            }

            let bx_start = sx * SUPER_BLOCK_BLOCKS;
            let bx_end = ((sx + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_x);
            let by_start = sy * SUPER_BLOCK_BLOCKS;
            let by_end = ((sy + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_y);
            let bz_start = sz * SUPER_BLOCK_BLOCKS;
            let bz_end = ((sz + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_z);

            let mut local_blocks = Vec::new();
            for bz in bz_start..bz_end {
                for by in by_start..by_end {
                    for bx in bx_start..bx_end {
                        let lx0 = bx * BLOCK_SIZE;
                        let lx1 = ((bx + 1) * BLOCK_SIZE).min(n_cells_x);
                        let ly0 = by * BLOCK_SIZE;
                        let ly1 = ((by + 1) * BLOCK_SIZE).min(n_cells_y);
                        let lz0 = bz * BLOCK_SIZE;
                        let lz1 = ((bz + 1) * BLOCK_SIZE).min(n_cells_z);

                        let b_min =
                            min + DVec3::new(lx0 as f64 * dx, ly0 as f64 * dy, lz0 as f64 * dz);
                        let b_max =
                            min + DVec3::new(lx1 as f64 * dx, ly1 as f64 * dy, lz1 as f64 * dz);
                        let b_center = (b_min + b_max) * 0.5;
                        let b_radius = (b_max - b_min).length() * 0.5;

                        if (field.sample(b_center).value - iso).abs() <= b_radius + cell_diag {
                            local_blocks.push([bx, by, bz]);
                        }
                    }
                }
            }
            local_blocks
        })
        .collect();

    active_blocks
        .into_par_iter()
        .flat_map_iter(|[bx, by, bz]| {
            let x0 = bx * BLOCK_SIZE;
            let x1 = ((bx + 1) * BLOCK_SIZE).min(n_cells_x);
            let y0 = by * BLOCK_SIZE;
            let y1 = ((by + 1) * BLOCK_SIZE).min(n_cells_y);
            let z0 = bz * BLOCK_SIZE;
            let z1 = ((bz + 1) * BLOCK_SIZE).min(n_cells_z);

            let wx = x1 - x0;
            let wy = y1 - y0;
            let wz = z1 - z0;

            let dims_x = wx + 1;
            let dims_y = wy + 1;
            let dims_z = wz + 1;

            let mut values = vec![0.0f64; dims_x * dims_y * dims_z];
            for zi in 0..dims_z {
                for yi in 0..dims_y {
                    for xi in 0..dims_x {
                        let p = min
                            + DVec3::new(
                                (x0 + xi) as f64 * dx,
                                (y0 + yi) as f64 * dy,
                                (z0 + zi) as f64 * dz,
                            );
                        values[(zi * dims_y + yi) * dims_x + xi] = field.sample(p).value;
                    }
                }
            }

            let mut block_vertices = Vec::new();
            for cz in 0..wz {
                for cy in 0..wy {
                    for cx in 0..wx {
                        let corner_pos: [DVec3; 8] = std::array::from_fn(|c| {
                            let (cdx, cdy, cdz) = CORNER_OFFSETS[c];
                            min + DVec3::new(
                                (x0 + cx + cdx) as f64 * dx,
                                (y0 + cy + cdy) as f64 * dy,
                                (z0 + cz + cdz) as f64 * dz,
                            )
                        });
                        let corner_val: [f64; 8] = std::array::from_fn(|c| {
                            let (cdx, cdy, cdz) = CORNER_OFFSETS[c];
                            values[((cz + cdz) * dims_y + (cy + cdy)) * dims_x + (cx + cdx)]
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
                                block_vertices.push(edge_vertex[e as usize].expect(
                                    "TRI_TABLE only references edges flagged in EDGE_TABLE",
                                ));
                            }
                        }
                    }
                }
            }
            block_vertices
        })
        .collect()
}

/// Standard marching-cubes edge table: for each of the 256 cube
/// (inside/outside corner) configurations, a 12-bit mask of which of the 12
/// cube edges the isosurface crosses.
#[rustfmt::skip]
pub(crate) const EDGE_TABLE: [u16; 256] = [
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
pub(crate) const TRI_TABLE: [[i8; 16]; 256] = include!("marching_cubes_tri_table.rs.in");

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

    #[test]
    fn sparse_marching_cubes_extracts_sphere_surface() {
        let field = TreeField::new(sphere_tree(1.0));
        let positions = extract_sparse_isosurface_positions(
            &field,
            DVec3::splat(-2.0),
            DVec3::splat(2.0),
            0.1,
            0.0,
            0.0,
        );
        assert!(!positions.is_empty());
        assert_eq!(positions.len() % 3, 0);
        // All extracted positions should lie very close to radius 1.0
        for p in &positions {
            let r = p.length();
            assert!((r - 1.0).abs() < 0.05, "radius {} off 1.0", r);
        }
    }

    #[test]
    fn sparse_marching_cubes_with_normals_extracts_sphere_surface() {
        let field = TreeField::new(sphere_tree(1.0));
        let vertices =
            extract_sparse_isosurface(&field, DVec3::splat(-2.0), DVec3::splat(2.0), 0.1, 0.0);
        assert!(!vertices.is_empty());
        assert_eq!(vertices.len() % 3, 0);
        for v in &vertices {
            let r = v.position.length();
            assert!((r - 1.0).abs() < 0.05, "radius {} off 1.0", r);
            let n_len = v.normal.length();
            assert!((n_len - 1.0).abs() < 1e-3, "normal len {} off 1.0", n_len);
            // Normal on a sphere centered at origin should point in position's direction
            let dot = v.normal.dot(v.position.normalize());
            assert!(
                dot > 0.95,
                "normal not aligned with radial direction: dot = {}",
                dot
            );
        }
    }
}
