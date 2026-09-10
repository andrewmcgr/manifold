//! 3D Anisotropic Fast Sweeping Method (FSM) Order Field.
//!
//! Evaluates the anisotropic Eikonal equation:
//!
//! $$\nabla \phi^T \mathbf{D}(\mathbf{x}) \nabla \phi = 1$$
//!
//! on a 3D Cartesian grid using multi-directional Gauss-Seidel sweeping
//! with lock-free wavefront hyperplane parallelization via Rayon.

use crate::fsm_simplex::solve_anisotropic_octant_simplex;
use crate::fsm_tensor::{MetricTensor3, TensorGrid};
use crate::height_along::HeightAlong;
use crate::order::OrderField;
use crate::slope_profile::SlopeProfile;
use glam::DVec3;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Min-heap entry for label-correcting Lipschitz slope relaxation.
#[derive(Copy, Clone, PartialEq)]
struct HeapEntry {
    value: f64,
    x: usize,
    y: usize,
    z: usize,
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .partial_cmp(&self.value)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 8 canonical coordinate sweeping directions: $(s_x, s_y, s_z) \in \{-1.0, +1.0\}^3$.
const SWEEP_DIRECTIONS: [(f64, f64, f64); 8] = [
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (1.0, -1.0, -1.0),
    (-1.0, -1.0, -1.0),
];

/// An independent [`OrderField`] implementation backed by an anisotropic
/// Fast Sweeping Method (FSM) solve over a 3D Cartesian grid.
pub struct AnisotropicFsmOrderField {
    min_corner: DVec3,
    dims: [usize; 3],
    h: f64,
    distances: Vec<f64>,
    gradients: Vec<DVec3>,
}

impl AnisotropicFsmOrderField {
    /// Builds a new [`AnisotropicFsmOrderField`] with a uniform isotropic metric tensor ($D = \mathbf{I}$).
    pub fn new_isotropic(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
        is_seed: &(dyn Fn(DVec3) -> bool + Sync),
    ) -> Self {
        let (dims, h, actual_min) =
            Self::compute_grid_dims(min_corner, max_corner, requested_cell_size);
        let tensor_grid = TensorGrid::new_isotropic(actual_min, dims, h);
        Self::solve_with_tensor_grid(
            actual_min,
            dims,
            h,
            &tensor_grid,
            is_solid,
            is_seed,
            8,
            None,
            None,
        )
    }

    /// Solves the anisotropic order field using the specified [`TensorGrid`].
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_tensor_grid(
        min_corner: DVec3,
        dims: [usize; 3],
        h: f64,
        tensor_grid: &TensorGrid,
        is_solid: &(dyn Fn(DVec3) -> bool + Sync),
        is_seed: &(dyn Fn(DVec3) -> bool + Sync),
        max_sweeps: usize,
        slope_profile: Option<&SlopeProfile>,
        height_along: Option<&dyn HeightAlong>,
    ) -> Self {
        let [nx, ny, nz] = dims;
        let total = nx * ny * nz;

        // Classify occupancy and Dirichlet seed status
        let mut occupied = vec![false; total];
        let mut is_fixed_seed = vec![false; total];
        let mut distances = vec![f64::INFINITY; total];

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + y * nx + z * nx * ny;
                    let p = DVec3::new(
                        min_corner.x + x as f64 * h,
                        min_corner.y + y as f64 * h,
                        min_corner.z + z as f64 * h,
                    );
                    if is_solid(p) {
                        occupied[idx] = true;
                        if is_seed(p) {
                            is_fixed_seed[idx] = true;
                            distances[idx] = 0.0;
                        }
                    }
                }
            }
        }

        // Precompute velocity tensors D = M^-1
        let mut velocity_tensors = Vec::with_capacity(total);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let m = tensor_grid.get(x, y, z);
                    let d = m.inverse().unwrap_or_else(MetricTensor3::identity);
                    velocity_tensors.push(d);
                }
            }
        }

        // Precompute diagonal hyperplane wavefront sets for lock-free parallel sweeping
        let max_s = (nx.saturating_sub(1)) + (ny.saturating_sub(1)) + (nz.saturating_sub(1));
        let mut planes: Vec<Vec<[usize; 3]>> = vec![Vec::new(); max_s + 1];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    planes[x + y + z].push([x, y, z]);
                }
            }
        }

        // Execute sweeps
        let num_full_cycles = max_sweeps.max(1).div_ceil(8);
        for _cycle in 0..num_full_cycles {
            let mut max_diff: f64 = 0.0;

            for &(sx, sy, sz) in &SWEEP_DIRECTIONS {
                let diff = Self::execute_hyperplane_sweep(
                    dims,
                    h,
                    sx,
                    sy,
                    sz,
                    &planes,
                    &occupied,
                    &is_fixed_seed,
                    &velocity_tensors,
                    &mut distances,
                );
                max_diff = max_diff.max(diff);
            }

            if max_diff < 1e-4 {
                break;
            }
        }

        let mut field = Self {
            min_corner,
            dims,
            h,
            distances,
            gradients: Vec::new(),
        };

        if let Some(profile) = slope_profile {
            let default_height =
                crate::height_along::ConstantAxisHeight::new(glam::DVec3::Z, min_corner);
            let ha: &dyn HeightAlong = height_along.unwrap_or(&default_height);
            field.relax_with_slope_limit(profile, ha, &occupied);
        }

        // Compute Hermite gradients
        field.gradients = Self::compute_gradients(dims, h, &field.distances, &occupied);
        field
    }

    /// Enforces the machine's slope limit profile (Lipschitz extension) post-sweep:
    /// `|phi(p) - phi(q)| <= tan(max_angle) * h` for every pair of horizontally
    /// adjacent grid nodes `(p, q)`.
    fn relax_with_slope_limit(
        &mut self,
        profile: &SlopeProfile,
        height_along: &dyn HeightAlong,
        occupied: &[bool],
    ) {
        let [nx, ny, nz] = self.dims;
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + y * nx + z * nx * ny;
                    if occupied[idx] && self.distances[idx].is_finite() {
                        heap.push(HeapEntry {
                            value: self.distances[idx],
                            x,
                            y,
                            z,
                        });
                    }
                }
            }
        }

        const NEAR_VERTICAL_ANGLE_DEG: f64 = 89.999;
        const NEIGHBOR_OFFSETS: [(isize, isize, isize); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];

        while let Some(HeapEntry { value, x, y, z }) = heap.pop() {
            let idx = x + y * nx + z * nx * ny;
            if self.distances[idx] < value {
                continue;
            }
            let t_p = self.distances[idx];

            let p = DVec3::new(
                self.min_corner.x + x as f64 * self.h,
                self.min_corner.y + y as f64 * self.h,
                self.min_corner.z + z as f64 * self.h,
            );
            let height = height_along.height(p);
            if height.is_nan() {
                continue;
            }
            let max_angle = profile.max_slope_at(height);
            if !max_angle.is_finite() || max_angle >= NEAR_VERTICAL_ANGLE_DEG {
                continue;
            }
            let slope_multiplier = max_angle.to_radians().tan();
            if !slope_multiplier.is_finite() {
                continue;
            }

            for (dx, dy, dz) in NEIGHBOR_OFFSETS {
                if dx == 0 && dy == 0 {
                    // Pure vertical neighbor: leave unconstrained so vertical progression is not throttled
                    continue;
                }
                let nxp = x as isize + dx;
                let nyp = y as isize + dy;
                let nzp = z as isize + dz;
                if nxp < 0
                    || nyp < 0
                    || nzp < 0
                    || nxp as usize >= nx
                    || nyp as usize >= ny
                    || nzp as usize >= nz
                {
                    continue;
                }
                let (nxu, nyu, nzu) = (nxp as usize, nyp as usize, nzp as usize);
                let nidx = nxu + nyu * nx + nzu * nx * ny;
                if !occupied[nidx] {
                    continue;
                }

                let candidate = t_p + slope_multiplier * self.h;
                if candidate < self.distances[nidx] {
                    self.distances[nidx] = candidate;
                    heap.push(HeapEntry {
                        value: candidate,
                        x: nxu,
                        y: nyu,
                        z: nzu,
                    });
                }
            }
        }
    }

    /// Executes one serial sweep along direction `(sx, sy, sz)`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_serial_sweep(
        dims: [usize; 3],
        h: f64,
        sx: f64,
        sy: f64,
        sz: f64,
        occupied: &[bool],
        is_fixed_seed: &[bool],
        velocity_tensors: &[MetricTensor3],
        distances: &mut [f64],
    ) -> f64 {
        let [nx, ny, nz] = dims;
        let mut max_diff = 0.0f64;

        let x_range: Vec<usize> = if sx > 0.0 {
            (0..nx).collect()
        } else {
            (0..nx).rev().collect()
        };
        let y_range: Vec<usize> = if sy > 0.0 {
            (0..ny).collect()
        } else {
            (0..ny).rev().collect()
        };
        let z_range: Vec<usize> = if sz > 0.0 {
            (0..nz).collect()
        } else {
            (0..nz).rev().collect()
        };

        for &z in &z_range {
            for &y in &y_range {
                for &x in &x_range {
                    let idx = x + y * nx + z * nx * ny;
                    if !occupied[idx] || is_fixed_seed[idx] {
                        continue;
                    }

                    let phi_x = if sx > 0.0 {
                        if x > 0 {
                            distances[idx - 1]
                        } else {
                            f64::INFINITY
                        }
                    } else if x + 1 < nx {
                        distances[idx + 1]
                    } else {
                        f64::INFINITY
                    };

                    let phi_y = if sy > 0.0 {
                        if y > 0 {
                            distances[idx - nx]
                        } else {
                            f64::INFINITY
                        }
                    } else if y + 1 < ny {
                        distances[idx + nx]
                    } else {
                        f64::INFINITY
                    };

                    let phi_z = if sz > 0.0 {
                        if z > 0 {
                            distances[idx - nx * ny]
                        } else {
                            f64::INFINITY
                        }
                    } else if z + 1 < nz {
                        distances[idx + nx * ny]
                    } else {
                        f64::INFINITY
                    };

                    let d = &velocity_tensors[idx];
                    let u = solve_anisotropic_octant_simplex(phi_x, phi_y, phi_z, sx, sy, sz, d, h);

                    let current = distances[idx];
                    if u < current {
                        distances[idx] = u;
                        max_diff = max_diff.max(current - u);
                    }
                }
            }
        }

        max_diff
    }

    /// Executes one parallel wavefront hyperplane sweep along direction `(sx, sy, sz)`.
    #[allow(clippy::too_many_arguments)]
    fn execute_hyperplane_sweep(
        dims: [usize; 3],
        h: f64,
        sx: f64,
        sy: f64,
        sz: f64,
        planes: &[Vec<[usize; 3]>],
        occupied: &[bool],
        is_fixed_seed: &[bool],
        velocity_tensors: &[MetricTensor3],
        distances: &mut [f64],
    ) -> f64 {
        let [nx, ny, nz] = dims;

        #[derive(Clone, Copy)]
        struct UnsafeDistances(usize);
        unsafe impl Send for UnsafeDistances {}
        unsafe impl Sync for UnsafeDistances {}

        impl UnsafeDistances {
            #[inline(always)]
            unsafe fn get(&self, idx: usize) -> f64 {
                *(self.0 as *const f64).add(idx)
            }

            #[inline(always)]
            unsafe fn set(&self, idx: usize, val: f64) {
                *(self.0 as *mut f64).add(idx) = val;
            }
        }

        let dist_wrapper = UnsafeDistances(distances.as_mut_ptr() as usize);

        let mut max_sweep_diff = 0.0f64;

        for plane in planes {
            let plane_max_diff = plane
                .par_iter()
                .map(|&[hx, hy, hz]| {
                    let x = if sx > 0.0 { hx } else { nx - 1 - hx };
                    let y = if sy > 0.0 { hy } else { ny - 1 - hy };
                    let z = if sz > 0.0 { hz } else { nz - 1 - hz };
                    let idx = x + y * nx + z * nx * ny;

                    if !occupied[idx] || is_fixed_seed[idx] {
                        return 0.0;
                    }

                    // Look up upwind neighbor values
                    let phi_x = if sx > 0.0 {
                        if x > 0 {
                            unsafe { dist_wrapper.get(idx - 1) }
                        } else {
                            f64::INFINITY
                        }
                    } else if x + 1 < nx {
                        unsafe { dist_wrapper.get(idx + 1) }
                    } else {
                        f64::INFINITY
                    };

                    let phi_y = if sy > 0.0 {
                        if y > 0 {
                            unsafe { dist_wrapper.get(idx - nx) }
                        } else {
                            f64::INFINITY
                        }
                    } else if y + 1 < ny {
                        unsafe { dist_wrapper.get(idx + nx) }
                    } else {
                        f64::INFINITY
                    };

                    let phi_z = if sz > 0.0 {
                        if z > 0 {
                            unsafe { dist_wrapper.get(idx - nx * ny) }
                        } else {
                            f64::INFINITY
                        }
                    } else if z + 1 < nz {
                        unsafe { dist_wrapper.get(idx + nx * ny) }
                    } else {
                        f64::INFINITY
                    };

                    let d = &velocity_tensors[idx];
                    let u = solve_anisotropic_octant_simplex(phi_x, phi_y, phi_z, sx, sy, sz, d, h);

                    let current = unsafe { dist_wrapper.get(idx) };
                    if u < current {
                        unsafe {
                            dist_wrapper.set(idx, u);
                        }
                        current - u
                    } else {
                        0.0
                    }
                })
                .reduce(|| 0.0f64, f64::max);

            max_sweep_diff = max_sweep_diff.max(plane_max_diff);
        }

        max_sweep_diff
    }

    /// Computes centered or one-sided gradients $\nabla \phi$ for Hermite interpolation.
    fn compute_gradients(
        dims: [usize; 3],
        h: f64,
        distances: &[f64],
        occupied: &[bool],
    ) -> Vec<DVec3> {
        let [nx, ny, nz] = dims;
        let total = nx * ny * nz;
        let mut gradients = vec![DVec3::ZERO; total];

        let inv_2h = 1.0 / (2.0 * h);
        let inv_h = 1.0 / h;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = x + y * nx + z * nx * ny;
                    if !occupied[idx] || !distances[idx].is_finite() {
                        continue;
                    }

                    let val = distances[idx];

                    // X gradient
                    let gx = if x > 0 && x + 1 < nx {
                        let vl = distances[idx - 1];
                        let vr = distances[idx + 1];
                        if vl.is_finite() && vr.is_finite() {
                            (vr - vl) * inv_2h
                        } else if vr.is_finite() {
                            (vr - val) * inv_h
                        } else if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else if x + 1 < nx {
                        let vr = distances[idx + 1];
                        if vr.is_finite() {
                            (vr - val) * inv_h
                        } else {
                            0.0
                        }
                    } else if x > 0 {
                        let vl = distances[idx - 1];
                        if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Y gradient
                    let gy = if y > 0 && y + 1 < ny {
                        let vl = distances[idx - nx];
                        let vr = distances[idx + nx];
                        if vl.is_finite() && vr.is_finite() {
                            (vr - vl) * inv_2h
                        } else if vr.is_finite() {
                            (vr - val) * inv_h
                        } else if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else if y + 1 < ny {
                        let vr = distances[idx + nx];
                        if vr.is_finite() {
                            (vr - val) * inv_h
                        } else {
                            0.0
                        }
                    } else if y > 0 {
                        let vl = distances[idx - nx];
                        if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Z gradient
                    let gz = if z > 0 && z + 1 < nz {
                        let vl = distances[idx - nx * ny];
                        let vr = distances[idx + nx * ny];
                        if vl.is_finite() && vr.is_finite() {
                            (vr - vl) * inv_2h
                        } else if vr.is_finite() {
                            (vr - val) * inv_h
                        } else if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else if z + 1 < nz {
                        let vr = distances[idx + nx * ny];
                        if vr.is_finite() {
                            (vr - val) * inv_h
                        } else {
                            0.0
                        }
                    } else if z > 0 {
                        let vl = distances[idx - nx * ny];
                        if vl.is_finite() {
                            (val - vl) * inv_h
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    gradients[idx] = DVec3::new(gx, gy, gz);
                }
            }
        }

        gradients
    }

    /// Helper to compute grid dimensions and spacing from bounding box.
    pub fn compute_grid_dims(
        min_corner: DVec3,
        max_corner: DVec3,
        requested_cell_size: f64,
    ) -> ([usize; 3], f64, DVec3) {
        let actual_min = min_corner.min(max_corner);
        let actual_max = min_corner.max(max_corner);
        let extent = actual_max - actual_min;

        let h = if requested_cell_size.is_finite() && requested_cell_size > 1e-6 {
            requested_cell_size
        } else {
            (extent.max_element() / 20.0).max(1.0)
        };

        let nx = ((extent.x / h).ceil() as usize).max(1) + 1;
        let ny = ((extent.y / h).ceil() as usize).max(1) + 1;
        let nz = ((extent.z / h).ceil() as usize).max(1) + 1;

        ([nx, ny, nz], h, actual_min)
    }

    #[inline]
    #[must_use]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }
}

impl OrderField for AnisotropicFsmOrderField {
    fn order(&self, p: DVec3) -> f64 {
        let [nx, ny, nz] = self.dims;
        let h = self.h;

        let fx = ((p.x - self.min_corner.x) / h).clamp(0.0, (nx - 1) as f64);
        let fy = ((p.y - self.min_corner.y) / h).clamp(0.0, (ny - 1) as f64);
        let fz = ((p.z - self.min_corner.z) / h).clamp(0.0, (nz - 1) as f64);

        let x0 = (fx.floor() as usize).min(nx - 2);
        let y0 = (fy.floor() as usize).min(ny - 2);
        let z0 = (fz.floor() as usize).min(nz - 2);

        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let tz = fz - z0 as f64;

        let corners = [
            (x0, y0, z0),
            (x0 + 1, y0, z0),
            (x0, y0 + 1, z0),
            (x0 + 1, y0 + 1, z0),
            (x0, y0, z0 + 1),
            (x0 + 1, y0, z0 + 1),
            (x0, y0 + 1, z0 + 1),
            (x0 + 1, y0 + 1, z0 + 1),
        ];

        let mut vals = [0.0; 8];
        let mut grads = [DVec3::ZERO; 8];

        for (i, &(cx, cy, cz)) in corners.iter().enumerate() {
            let idx = self.idx(cx, cy, cz);
            vals[i] = self.distances[idx];
            grads[i] = self.gradients[idx];
        }

        // If any corner is infinity, fall back to inverse-distance weighted finite average
        if vals.iter().any(|v| !v.is_finite()) {
            let mut sum_v = 0.0;
            let mut sum_w = 0.0;
            for (i, &v) in vals.iter().enumerate() {
                if v.is_finite() {
                    let cx = (i & 1) as f64;
                    let cy = ((i >> 1) & 1) as f64;
                    let cz = ((i >> 2) & 1) as f64;
                    let dist =
                        ((tx - cx).powi(2) + (ty - cy).powi(2) + (tz - cz).powi(2)).sqrt() + 1e-6;
                    let w = 1.0 / dist;
                    sum_v += v * w;
                    sum_w += w;
                }
            }
            if sum_w > 0.0 {
                return sum_v / sum_w;
            }
            return f64::INFINITY;
        }

        // Standard trilinear interpolation with Hermite derivative smoothing
        let h00 = |t: f64| 2.0 * t * t * t - 3.0 * t * t + 1.0;
        let h10 = |t: f64| t * t * t - 2.0 * t * t + t;
        let h01 = |t: f64| -2.0 * t * t * t + 3.0 * t * t;
        let h11 = |t: f64| t * t * t - t * t;

        let hermite_blend = |v0: f64, g0: f64, v1: f64, g1: f64, t: f64| {
            h00(t) * v0 + h10(t) * h * g0 + h01(t) * v1 + h11(t) * h * g1
        };

        let v_z0 = hermite_blend(vals[0], grads[0].z, vals[4], grads[4].z, tz);
        let v_z1 = hermite_blend(vals[1], grads[1].z, vals[5], grads[5].z, tz);
        let v_z2 = hermite_blend(vals[2], grads[2].z, vals[6], grads[6].z, tz);
        let v_z3 = hermite_blend(vals[3], grads[3].z, vals[7], grads[7].z, tz);

        let v_y0 = (1.0 - ty) * v_z0 + ty * v_z2;
        let v_y1 = (1.0 - ty) * v_z1 + ty * v_z3;

        (1.0 - tx) * v_y0 + tx * v_y1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsm_solves_isotropic_distance_from_point_seed() {
        let min = DVec3::new(0.0, 0.0, 0.0);
        let max = DVec3::new(10.0, 10.0, 10.0);
        let h = 1.0;

        let is_solid = |_p: DVec3| true;
        let is_seed = |p: DVec3| p.distance(DVec3::new(0.0, 0.0, 0.0)) < 0.5;

        let field = AnisotropicFsmOrderField::new_isotropic(min, max, h, &is_solid, &is_seed);

        // Test arrival times along axes
        let d_x5 = field.order(DVec3::new(5.0, 0.0, 0.0));
        assert!((d_x5 - 5.0).abs() < 0.15, "Expected ~5.0, got {}", d_x5);

        let d_diag = field.order(DVec3::new(3.0, 4.0, 0.0));
        assert!((d_diag - 5.0).abs() < 0.6, "Expected ~5.0, got {}", d_diag);
    }

    #[test]
    fn fsm_respects_anisotropic_tensor_speed() {
        let min = DVec3::new(0.0, 0.0, 0.0);
        let _max = DVec3::new(10.0, 10.0, 10.0);
        let h = 1.0;
        let dims = [11, 11, 11];

        // Velocity tensor: speed in X is 2.0 (D_xx = 4.0), speed in Y is 1.0 (D_yy = 1.0)
        // Metric tensor M = D^-1 = diag(0.25, 1.0, 1.0)
        let mut tensor_grid = TensorGrid::new_isotropic(min, dims, h);
        let m = MetricTensor3::from_diagonal(0.25, 1.0, 1.0);
        for t in &mut tensor_grid.tensors {
            *t = m;
        }

        let is_solid = |_p: DVec3| true;
        let is_seed = |p: DVec3| p.distance(DVec3::ZERO) < 0.5;

        let field = AnisotropicFsmOrderField::solve_with_tensor_grid(
            min,
            dims,
            h,
            &tensor_grid,
            &is_solid,
            &is_seed,
            8,
            None,
            None,
        );

        // Distance = 4.0 along X: speed is 2.0 => arrival time ~ 2.0
        let t_x = field.order(DVec3::new(4.0, 0.0, 0.0));
        assert!(
            (t_x - 2.0).abs() < 0.15,
            "Expected ~2.0 along X, got {}",
            t_x
        );

        // Distance = 4.0 along Y: speed is 1.0 => arrival time ~ 4.0
        let t_y = field.order(DVec3::new(0.0, 4.0, 0.0));
        assert!(
            (t_y - 4.0).abs() < 0.15,
            "Expected ~4.0 along Y, got {}",
            t_y
        );
    }

    #[test]
    fn fsm_respects_slope_limit_profile() {
        let min = DVec3::new(0.0, 0.0, 0.0);
        let h = 0.5;
        let dims = [21, 21, 21];

        let tensor_grid = TensorGrid::new_isotropic(min, dims, h);
        let is_solid = |_p: DVec3| true;
        let is_seed = |p: DVec3| p.distance(DVec3::ZERO) < 0.25;

        let max_angle_deg = 15.0;
        let profile = SlopeProfile::from_angle(max_angle_deg);
        let height_along = crate::height_along::ConstantAxisHeight::new(glam::DVec3::Z, min);

        let field = AnisotropicFsmOrderField::solve_with_tensor_grid(
            min,
            dims,
            h,
            &tensor_grid,
            &is_solid,
            &is_seed,
            8,
            Some(&profile),
            Some(&height_along),
        );

        let [nx, ny, nz] = dims;
        let tan_bound = max_angle_deg.to_radians().tan();
        let max_allowed_h_step = tan_bound * h + 1e-4;

        for z in 1..nz - 1 {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let idx = x + y * nx + z * nx * ny;
                    let val = field.distances[idx];
                    if !val.is_finite() {
                        continue;
                    }
                    let right = field.distances[idx + 1];
                    let up = field.distances[idx + nx];
                    if right.is_finite() {
                        let diff = (val - right).abs();
                        assert!(
                            diff <= max_allowed_h_step,
                            "Horizontal step X exceeded slope bound: diff={}, max={}",
                            diff,
                            max_allowed_h_step
                        );
                    }
                    if up.is_finite() {
                        let diff = (val - up).abs();
                        assert!(
                            diff <= max_allowed_h_step,
                            "Horizontal step Y exceeded slope bound: diff={}, max={}",
                            diff,
                            max_allowed_h_step
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parallel_sweeping_matches_serial_sweeping() {
        let min = DVec3::new(0.0, 0.0, 0.0);
        let dims = [8, 8, 8];
        let h = 1.0;
        let total = 8 * 8 * 8;

        let _tensor_grid = TensorGrid::new_isotropic(min, dims, h);
        let occupied = vec![true; total];
        let mut is_fixed_seed = vec![false; total];
        is_fixed_seed[0] = true;

        let velocity_tensors = vec![MetricTensor3::identity(); total];

        let mut dist_serial = vec![f64::INFINITY; total];
        dist_serial[0] = 0.0;
        let mut dist_parallel = dist_serial.clone();

        let max_s = (dims[0] - 1) + (dims[1] - 1) + (dims[2] - 1);
        let mut planes: Vec<Vec<[usize; 3]>> = vec![Vec::new(); max_s + 1];
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    planes[x + y + z].push([x, y, z]);
                }
            }
        }

        // Run 8 sweeps on both
        for &(sx, sy, sz) in &SWEEP_DIRECTIONS {
            AnisotropicFsmOrderField::execute_serial_sweep(
                dims,
                h,
                sx,
                sy,
                sz,
                &occupied,
                &is_fixed_seed,
                &velocity_tensors,
                &mut dist_serial,
            );
            AnisotropicFsmOrderField::execute_hyperplane_sweep(
                dims,
                h,
                sx,
                sy,
                sz,
                &planes,
                &occupied,
                &is_fixed_seed,
                &velocity_tensors,
                &mut dist_parallel,
            );
        }

        // Verify identical results
        for idx in 0..total {
            let diff = (dist_serial[idx] - dist_parallel[idx]).abs();
            assert!(
                diff < 1e-12,
                "Node {} mismatch: serial {} vs parallel {}",
                idx,
                dist_serial[idx],
                dist_parallel[idx]
            );
        }
    }
}
