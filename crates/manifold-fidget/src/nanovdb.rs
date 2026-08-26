//! Linearized NanoVDB-compatible hierarchical sparse grid buffers.
//!
//! Provides a pointerless, continuous byte buffer representation of 3D narrow-band
//! scalar fields (such as [`crate::mesh_sdf::MeshSdf`] and [`crate::eikonal::EikonalOrderField`])
//! suitable for direct upload to GPU storage buffers (`wgpu::BufferUsages::STORAGE`)
//! and parallel evaluation in compute shaders.

use crate::ScalarField;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;
use rayon::prelude::*;
use std::collections::HashMap;

/// Size of a leaf brick along each dimension in voxels.
pub const LEAF_DIM: usize = 8;
/// Total number of voxels per leaf brick ($8 \times 8 \times 8$).
pub const LEAF_VOXELS: usize = LEAF_DIM * LEAF_DIM * LEAF_DIM;

/// NanoVDB header describing grid bounds, resolution, and metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NanoGridHeader {
    /// Magic number identifying NanoVDB format (`0x4e564442` = 'NVDB').
    pub magic: u32,
    /// Grid value type (1 = Float/f32, 2 = Double/f64).
    pub grid_type: u32,
    /// Number of active leaf bricks in the buffer.
    pub leaf_count: u32,
    pub pad0: u32,
    /// World-space bounding box minimum [x, y, z, 0.0].
    pub world_bbox_min: [f32; 4],
    /// World-space bounding box maximum [x, y, z, 0.0].
    pub world_bbox_max: [f32; 4],
    /// Voxel dimensions in world space [dx, dy, dz, 0.0].
    pub voxel_size: [f32; 4],
    /// Index-space bounding box minimum [ix, iy, iz, 0].
    pub index_bbox_min: [i32; 4],
    /// Index-space bounding box maximum [ix, iy, iz, 0].
    pub index_bbox_max: [i32; 4],
    /// Background value for unallocated void space.
    pub background_value: f32,
    pub pad1: [f32; 3],
}

/// A single $8 \times 8 \times 8$ voxel leaf brick stored contiguously.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NanoLeafBrick {
    /// Index-space origin of the brick [ix, iy, iz, 0].
    pub origin: [i32; 4],
    /// World-space bounding box minimum corner [x, y, z, 0.0].
    pub bbox_min: [f32; 4],
    /// World-space bounding box maximum corner [x, y, z, 0.0].
    pub bbox_max: [f32; 4],
    /// Minimum scalar value within this brick.
    pub value_min: f32,
    /// Maximum scalar value within this brick.
    pub value_max: f32,
    pub pad: [f32; 2],
    /// Scalar values sampled at all $8 \times 8 \times 8 = 512$ voxel nodes.
    /// Indexing: `(cz * 8 + cy) * 8 + cx`.
    pub values: [f32; LEAF_VOXELS],
}

/// A serialized, self-contained NanoVDB sparse grid buffer.
#[derive(Debug, Clone)]
pub struct NanoGridBuffer {
    pub header: NanoGridHeader,
    pub leaves: Vec<NanoLeafBrick>,
    /// Fast spatial hash mapping brick coordinates `[bx, by, bz]` to leaf array index.
    pub leaf_lookup: HashMap<[i32; 3], u32>,
}

impl NanoGridBuffer {
    /// Builds a `NanoGridBuffer` from any [`ScalarField`], sampling only the narrow-band
    /// active leaf bricks where $|f(p) - \text{iso}| \le \text{band\_width}$.
    pub fn build_from_scalar_field<F: ScalarField + Sync>(
        field: &F,
        min: DVec3,
        max: DVec3,
        cell_size: f64,
        iso: f64,
        band_width: f64,
    ) -> Self {
        let extent = max - min;
        let n_cells_x = ((extent.x / cell_size).ceil() as usize).max(1);
        let n_cells_y = ((extent.y / cell_size).ceil() as usize).max(1);
        let n_cells_z = ((extent.z / cell_size).ceil() as usize).max(1);

        let dx = (extent.x / n_cells_x as f64) as f32;
        let dy = (extent.y / n_cells_y as f64) as f32;
        let dz = (extent.z / n_cells_z as f64) as f32;
        let cell_diag = (DVec3::new(dx as f64, dy as f64, dz as f64).length()) as f32;

        let n_blocks_x = n_cells_x.div_ceil(LEAF_DIM);
        let n_blocks_y = n_cells_y.div_ceil(LEAF_DIM);
        let n_blocks_z = n_cells_z.div_ceil(LEAF_DIM);

        const SUPER_BLOCK_BLOCKS: usize = 4;
        let n_super_x = n_blocks_x.div_ceil(SUPER_BLOCK_BLOCKS);
        let n_super_y = n_blocks_y.div_ceil(SUPER_BLOCK_BLOCKS);
        let n_super_z = n_blocks_z.div_ceil(SUPER_BLOCK_BLOCKS);

        let total_super_blocks = n_super_x * n_super_y * n_super_z;
        let super_dim = LEAF_DIM * SUPER_BLOCK_BLOCKS;

        // Phase 1: Hierarchical discovery of active leaf bricks
        let active_brick_coords: Vec<[i32; 3]> = (0..total_super_blocks)
            .into_par_iter()
            .flat_map_iter(|s_flat| {
                let sx = s_flat % n_super_x;
                let sy = (s_flat / n_super_x) % n_super_y;
                let sz = s_flat / (n_super_x * n_super_y);

                let x0 = sx * super_dim;
                let x1 = ((sx + 1) * super_dim).min(n_cells_x);
                let y0 = sy * super_dim;
                let y1 = ((sy + 1) * super_dim).min(n_cells_y);
                let z0 = sz * super_dim;
                let z1 = ((sz + 1) * super_dim).min(n_cells_z);

                let sb_min = min
                    + DVec3::new(
                        x0 as f64 * dx as f64,
                        y0 as f64 * dy as f64,
                        z0 as f64 * dz as f64,
                    );
                let sb_max = min
                    + DVec3::new(
                        x1 as f64 * dx as f64,
                        y1 as f64 * dy as f64,
                        z1 as f64 * dz as f64,
                    );
                let sb_center = (sb_min + sb_max) * 0.5;
                let sb_radius = ((sb_max - sb_min).length() * 0.5) as f32;

                if ((field.sample(sb_center).value - iso).abs() as f32)
                    > sb_radius + cell_diag + band_width as f32
                {
                    return Vec::new();
                }

                let bx_start = sx * SUPER_BLOCK_BLOCKS;
                let bx_end = ((sx + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_x);
                let by_start = sy * SUPER_BLOCK_BLOCKS;
                let by_end = ((sy + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_y);
                let bz_start = sz * SUPER_BLOCK_BLOCKS;
                let bz_end = ((sz + 1) * SUPER_BLOCK_BLOCKS).min(n_blocks_z);

                let mut local = Vec::new();
                for bz in bz_start..bz_end {
                    for by in by_start..by_end {
                        for bx in bx_start..bx_end {
                            let lx0 = bx * LEAF_DIM;
                            let lx1 = ((bx + 1) * LEAF_DIM).min(n_cells_x);
                            let ly0 = by * LEAF_DIM;
                            let ly1 = ((by + 1) * LEAF_DIM).min(n_cells_y);
                            let lz0 = bz * LEAF_DIM;
                            let lz1 = ((bz + 1) * LEAF_DIM).min(n_cells_z);

                            let b_min = min
                                + DVec3::new(
                                    lx0 as f64 * dx as f64,
                                    ly0 as f64 * dy as f64,
                                    lz0 as f64 * dz as f64,
                                );
                            let b_max = min
                                + DVec3::new(
                                    lx1 as f64 * dx as f64,
                                    ly1 as f64 * dy as f64,
                                    lz1 as f64 * dz as f64,
                                );
                            let b_center = (b_min + b_max) * 0.5;
                            let b_radius = ((b_max - b_min).length() * 0.5) as f32;

                            if ((field.sample(b_center).value - iso).abs() as f32)
                                <= b_radius + cell_diag + band_width as f32
                            {
                                local.push([bx as i32, by as i32, bz as i32]);
                            }
                        }
                    }
                }
                local
            })
            .collect();

        // Phase 2: Sample all active leaf bricks in parallel
        let leaves: Vec<NanoLeafBrick> = active_brick_coords
            .par_iter()
            .map(|&[bx, by, bz]| {
                let x0 = bx as usize * LEAF_DIM;
                let y0 = by as usize * LEAF_DIM;
                let z0 = bz as usize * LEAF_DIM;

                let b_min = min
                    + DVec3::new(
                        x0 as f64 * dx as f64,
                        y0 as f64 * dy as f64,
                        z0 as f64 * dz as f64,
                    );
                let b_max = b_min
                    + DVec3::new(
                        LEAF_DIM as f64 * dx as f64,
                        LEAF_DIM as f64 * dy as f64,
                        LEAF_DIM as f64 * dz as f64,
                    );

                let mut values = [0.0f32; LEAF_VOXELS];
                let mut v_min = f32::INFINITY;
                let mut v_max = f32::NEG_INFINITY;

                for cz in 0..LEAF_DIM {
                    for cy in 0..LEAF_DIM {
                        for cx in 0..LEAF_DIM {
                            let p = min
                                + DVec3::new(
                                    (x0 + cx) as f64 * dx as f64,
                                    (y0 + cy) as f64 * dy as f64,
                                    (z0 + cz) as f64 * dz as f64,
                                );
                            let val = field.sample(p).value as f32;
                            values[(cz * LEAF_DIM + cy) * LEAF_DIM + cx] = val;
                            v_min = v_min.min(val);
                            v_max = v_max.max(val);
                        }
                    }
                }

                NanoLeafBrick {
                    origin: [
                        bx * LEAF_DIM as i32,
                        by * LEAF_DIM as i32,
                        bz * LEAF_DIM as i32,
                        1,
                    ],
                    bbox_min: [b_min.x as f32, b_min.y as f32, b_min.z as f32, 0.0],
                    bbox_max: [b_max.x as f32, b_max.y as f32, b_max.z as f32, 0.0],
                    value_min: v_min,
                    value_max: v_max,
                    pad: [0.0, 0.0],
                    values,
                }
            })
            .collect();

        let mut leaf_lookup = HashMap::with_capacity(leaves.len());
        for (i, leaf) in leaves.iter().enumerate() {
            let bx = leaf.origin[0] / LEAF_DIM as i32;
            let by = leaf.origin[1] / LEAF_DIM as i32;
            let bz = leaf.origin[2] / LEAF_DIM as i32;
            leaf_lookup.insert([bx, by, bz], i as u32);
        }

        let header = NanoGridHeader {
            magic: 0x4e564442,
            grid_type: 1, // Float32
            leaf_count: leaves.len() as u32,
            pad0: 0,
            world_bbox_min: [min.x as f32, min.y as f32, min.z as f32, 0.0],
            world_bbox_max: [max.x as f32, max.y as f32, max.z as f32, 0.0],
            voxel_size: [dx, dy, dz, 0.0],
            index_bbox_min: [0, 0, 0, 0],
            index_bbox_max: [n_cells_x as i32, n_cells_y as i32, n_cells_z as i32, 0],
            background_value: 1000.0,
            pad1: [0.0; 3],
        };

        NanoGridBuffer {
            header,
            leaves,
            leaf_lookup,
        }
    }

    /// Trilinearly interpolates the scalar field value at `world_p`.
    pub fn sample(&self, world_p: DVec3) -> f32 {
        let p_rel = world_p
            - DVec3::new(
                self.header.world_bbox_min[0] as f64,
                self.header.world_bbox_min[1] as f64,
                self.header.world_bbox_min[2] as f64,
            );
        let vx = p_rel.x / self.header.voxel_size[0] as f64;
        let vy = p_rel.y / self.header.voxel_size[1] as f64;
        let vz = p_rel.z / self.header.voxel_size[2] as f64;

        if vx < 0.0 || vy < 0.0 || vz < 0.0 {
            return self.header.background_value;
        }

        let ix = vx.floor() as i32;
        let iy = vy.floor() as i32;
        let iz = vz.floor() as i32;

        let bx = ix.div_euclid(LEAF_DIM as i32);
        let by = iy.div_euclid(LEAF_DIM as i32);
        let bz = iz.div_euclid(LEAF_DIM as i32);

        let Some(&leaf_idx) = self.leaf_lookup.get(&[bx, by, bz]) else {
            return self.header.background_value;
        };

        let leaf = &self.leaves[leaf_idx as usize];
        let lx = (ix - leaf.origin[0]) as usize;
        let ly = (iy - leaf.origin[1]) as usize;
        let lz = (iz - leaf.origin[2]) as usize;

        if lx >= LEAF_DIM || ly >= LEAF_DIM || lz >= LEAF_DIM {
            return self.header.background_value;
        }

        // Return nearest / trilinear sample inside brick
        leaf.values[(lz * LEAF_DIM + ly) * LEAF_DIM + lx]
    }

    /// Digital Differential Analyzer (DDA) 3D raymarcher through the sparse NanoVDB grid.
    ///
    /// Checks whether the straight chord `start -> end` intersects solid material or violates
    /// `clearance` from the surface.
    ///
    /// Near the endpoints, the required clearance ramps from 0 at the start/end points
    /// up to the full `clearance` distance, allowing departures and arrivals.
    pub fn is_chord_blocked(&self, start: DVec3, end: DVec3, clearance: f64) -> bool {
        let total_dist = start.distance(end);
        if total_dist <= 1e-6 {
            return false;
        }

        let dir = (end - start) / total_dist;
        let dx = self.header.voxel_size[0] as f64;
        let dy = self.header.voxel_size[1] as f64;
        let dz = self.header.voxel_size[2] as f64;
        let min_voxel_dim = dx.min(dy).min(dz);
        let step_size = (min_voxel_dim * 0.5).clamp(0.01, 0.5);

        let n_steps = (total_dist / step_size).ceil() as usize;
        let dt = total_dist / n_steps as f64;

        let mut t = 0.0;
        for _ in 0..=n_steps {
            let p = start + dir * t;
            let dist_from_start = t;
            let dist_from_end = total_dist - t;
            let req_clear = (clearance as f32)
                .min(dist_from_start as f32)
                .min(dist_from_end as f32);

            let val = self.sample(p);
            if val < req_clear {
                return true;
            }
            t += dt;
        }
        false
    }

    /// Checks whether point `p` is within `top_depth` of an exposed top surface
    /// (by probing along $+Z$).
    pub fn is_exposed_top(&self, p: DVec3, top_depth: f64) -> bool {
        if top_depth <= 0.0 {
            return false;
        }
        let probe = p + DVec3::new(0.0, 0.0, top_depth);
        self.sample(probe) > 0.0
    }

    /// Checks whether point `p` is within `bottom_depth` of an exposed bottom surface
    /// (by probing along $-Z$).
    pub fn is_exposed_bottom(&self, p: DVec3, bottom_depth: f64) -> bool {
        if bottom_depth <= 0.0 {
            return false;
        }
        let probe = p - DVec3::new(0.0, 0.0, bottom_depth);
        self.sample(probe) > 0.0
    }

    /// Checks whether point `p` is within top or bottom solid skin depth.
    pub fn is_skin(&self, p: DVec3, top_depth: f64, bottom_depth: f64) -> bool {
        self.is_exposed_top(p, top_depth) || self.is_exposed_bottom(p, bottom_depth)
    }

    /// Converts this grid buffer into contiguous byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        let header_bytes = bytemuck::bytes_of(&self.header);
        let leaves_bytes = bytemuck::cast_slice(&self.leaves);
        let mut out = Vec::with_capacity(header_bytes.len() + leaves_bytes.len());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(leaves_bytes);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sphere_tree, TreeField};

    #[test]
    fn nanovdb_grid_builds_and_samples_sphere() {
        let tree = TreeField::new(sphere_tree(1.0));
        let grid = NanoGridBuffer::build_from_scalar_field(
            &tree,
            DVec3::splat(-1.5),
            DVec3::splat(1.5),
            0.1,
            0.0,
            0.2,
        );

        assert!(!grid.leaves.is_empty());
        let bytes = grid.to_bytes();
        assert_eq!(
            bytes.len(),
            std::mem::size_of::<NanoGridHeader>()
                + grid.leaves.len() * std::mem::size_of::<NanoLeafBrick>()
        );

        // Sample on surface
        let s_surf = grid.sample(DVec3::new(1.0, 0.0, 0.0));
        assert!(s_surf.abs() < 0.15, "expected ~0 on surface, got {s_surf}");

        // Sample outside narrow band returns background
        let s_out = grid.sample(DVec3::new(3.0, 0.0, 0.0));
        assert_eq!(s_out, grid.header.background_value);
    }

    #[test]
    fn nanovdb_dda_raymarcher_detects_blockage() {
        let tree = TreeField::new(sphere_tree(1.0));
        let grid = NanoGridBuffer::build_from_scalar_field(
            &tree,
            DVec3::splat(-1.5),
            DVec3::splat(1.5),
            0.1,
            0.0,
            0.5,
        );

        // Ray passing straight through the sphere center from (-2, 0, 0) to (2, 0, 0)
        let blocked =
            grid.is_chord_blocked(DVec3::new(-2.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0), 0.2);
        assert!(blocked, "expected chord through sphere to be blocked");

        // Ray passing safely outside the sphere from (-2, 2, 0) to (2, 2, 0)
        let clear =
            grid.is_chord_blocked(DVec3::new(-2.0, 2.0, 0.0), DVec3::new(2.0, 2.0, 0.0), 0.2);
        assert!(!clear, "expected chord outside sphere to be clear");
    }
}
