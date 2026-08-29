//! Fast Marching Method for the Eikonal equation on 2-manifold triangle meshes.
//!
//! Solves `\|\nabla_{\mathcal{M}} T\| = \text{cost}` on the mesh surface $\mathcal{M}$, with
//! boundary condition `T(v) = 0` at bed contact vertices ($v_z \le \min.z + \text{seed\_tolerance}$).
//!
//! The metric cost along each triangle facet scales with its vertical inclination
//! `cost = \max(\|\vec{n}_{xy}\|, \text{MIN\_GROWTH})` so surface arrival times accurately
//! track vertical build height progression without causing horizontal overhangs to out-pace
//! or lag the solid bulk.

use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const MIN_GROWTH: f64 = 0.15;

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    cost: f64,
    vertex: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.vertex == other.vertex
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller costs have higher priority
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

/// Computes the geodesic arrival time field $T(v)$ across all mesh vertices
/// using the Kimmel-Sethian Fast Marching Method on triangle meshes.
pub fn solve_surface_eikonal(
    vertices: &[DVec3],
    faces: &[[usize; 3]],
    is_seed: impl Fn(DVec3) -> bool,
) -> Vec<f64> {
    let num_verts = vertices.len();
    if num_verts == 0 {
        return Vec::new();
    }

    // Precompute face normals and local metric cost scales
    let mut face_costs = Vec::with_capacity(faces.len());
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); num_verts];
    for (f_idx, &[i0, i1, i2]) in faces.iter().enumerate() {
        if i0 < num_verts && i1 < num_verts && i2 < num_verts {
            vertex_faces[i0].push(f_idx);
            vertex_faces[i1].push(f_idx);
            vertex_faces[i2].push(f_idx);

            let v0 = vertices[i0];
            let v1 = vertices[i1];
            let v2 = vertices[i2];
            let normal = (v1 - v0).cross(v2 - v0);
            let len = normal.length();
            let cost = if len > 1e-9 {
                let n_z = (normal.z / len).clamp(-1.0, 1.0);
                (1.0 - n_z * n_z).max(0.0).sqrt().max(MIN_GROWTH)
            } else {
                1.0
            };
            face_costs.push(cost);
        } else {
            face_costs.push(1.0);
        }
    }

    let mut times = vec![f64::INFINITY; num_verts];
    let mut frozen = vec![false; num_verts];
    let mut heap = BinaryHeap::new();

    // Initialize seed vertices (e.g. bed contact)
    for (v_idx, &p) in vertices.iter().enumerate() {
        if is_seed(p) {
            times[v_idx] = 0.0;
            heap.push(HeapEntry {
                cost: 0.0,
                vertex: v_idx,
            });
        }
    }

    // Fast Marching propagation loop
    while let Some(HeapEntry { cost, vertex: v_a }) = heap.pop() {
        if frozen[v_a] {
            continue;
        }
        if cost > times[v_a] {
            continue;
        }
        frozen[v_a] = true;
        let t_a = times[v_a];
        let p_a = vertices[v_a];

        // Iterate over all triangles incident to v_a
        for &f_idx in &vertex_faces[v_a] {
            let [i0, i1, i2] = faces[f_idx];
            let f_cost = face_costs[f_idx];

            // Identify the other two vertices in the face
            let (v_b, v_c) = if i0 == v_a {
                (i1, i2)
            } else if i1 == v_a {
                (i0, i2)
            } else {
                (i0, i1)
            };

            // 1. Direct edge updates (1D Dijkstra fallback)
            for &v_target in &[v_b, v_c] {
                if !frozen[v_target] {
                    let d = (vertices[v_target] - p_a).length();
                    let cand = t_a + f_cost * d;
                    if cand < times[v_target] {
                        times[v_target] = cand;
                        heap.push(HeapEntry {
                            cost: cand,
                            vertex: v_target,
                        });
                    }
                }
            }

            // 2. 2D Triangle planar wavefront update to non-frozen vertices
            if frozen[v_b] && !frozen[v_c] {
                if let Some(cand) =
                    update_triangle_2d(p_a, vertices[v_b], vertices[v_c], t_a, times[v_b], f_cost)
                {
                    if cand < times[v_c] {
                        times[v_c] = cand;
                        heap.push(HeapEntry {
                            cost: cand,
                            vertex: v_c,
                        });
                    }
                }
            }

            if frozen[v_c] && !frozen[v_b] {
                if let Some(cand) =
                    update_triangle_2d(p_a, vertices[v_c], vertices[v_b], t_a, times[v_c], f_cost)
                {
                    if cand < times[v_b] {
                        times[v_b] = cand;
                        heap.push(HeapEntry {
                            cost: cand,
                            vertex: v_b,
                        });
                    }
                }
            }
        }
    }

    times
}

/// Solves the local 2D planar Eikonal equation across triangle (A, B, C)
/// with metric target gradient magnitude `cost`.
fn update_triangle_2d(
    p_a: DVec3,
    p_b: DVec3,
    p_c: DVec3,
    t_a: f64,
    t_b: f64,
    cost: f64,
) -> Option<f64> {
    let c = (p_b - p_a).length();
    let b = (p_c - p_a).length();
    let a = (p_c - p_b).length();

    if c <= 1e-12 || b <= 1e-12 || a <= 1e-12 {
        return None;
    }

    // Local 2D coordinates: A at (0, 0), B at (c, 0)
    let cos_alpha = ((b * b + c * c - a * a) / (2.0 * b * c)).clamp(-1.0, 1.0);
    let x_c = b * cos_alpha;
    let y_c = (b * b - x_c * x_c).max(0.0).sqrt();

    if y_c <= 1e-12 {
        return None;
    }

    // Gradient p = dT/dx = (T_B - T_A) / c
    let p = (t_b - t_a) / c;
    if p.abs() > cost {
        // Wave cannot cross edge AB continuously
        return None;
    }

    let q = (cost * cost - p * p).max(0.0).sqrt();

    // Upwind causality check: ray from virtual source S to C must pass through edge AB
    let x_0 = x_c - y_c * (p / q);
    if x_0 >= 0.0 && x_0 <= c {
        let t_c = t_a + p * x_c + q * y_c;
        Some(t_c)
    } else {
        None
    }
}

/// Evaluates barycentric interpolation of a per-vertex surface field at a query point `p`
/// on triangle `(v0, v1, v2)`.
pub fn interpolate_barycentric(
    v0: DVec3,
    v1: DVec3,
    v2: DVec3,
    t0: f64,
    t1: f64,
    t2: f64,
    p: DVec3,
) -> f64 {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let ep = p - v0;

    let d00 = e0.dot(e0);
    let d01 = e0.dot(e1);
    let d11 = e1.dot(e1);
    let dp0 = ep.dot(e0);
    let dp1 = ep.dot(e1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1e-12 {
        return t0.min(t1).min(t2);
    }

    let v = ((d11 * dp0 - d01 * dp1) / denom).clamp(0.0, 1.0);
    let w = ((d00 * dp1 - d01 * dp0) / denom).clamp(0.0, 1.0);
    let u = (1.0 - v - w).max(0.0);

    let sum = u + v + w;
    if sum <= 1e-12 {
        t0
    } else {
        (u * t0 + v * t1 + w * t2) / sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_triangle_fast_marching_matches_vertical_distance() {
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(10.0, 0.0, 0.0);
        let v2 = DVec3::new(0.0, 0.0, 10.0);
        let vertices = vec![v0, v1, v2];
        let faces = vec![[0, 1, 2]];

        let is_seed = |p: DVec3| p.z <= 0.01;
        let times = solve_surface_eikonal(&vertices, &faces, is_seed);

        assert_eq!(times[0], 0.0);
        assert!((times[2] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn cylinder_surface_propagation_is_strictly_monotonic_along_height() {
        let radius = 10.0;
        let height = 20.0;
        let num_radial = 16;
        let num_vertical = 10;

        let mut vertices = Vec::new();
        for z_step in 0..=num_vertical {
            let z = (z_step as f64 / num_vertical as f64) * height;
            for r_step in 0..num_radial {
                let theta = (r_step as f64 / num_radial as f64) * std::f64::consts::TAU;
                vertices.push(DVec3::new(radius * theta.cos(), radius * theta.sin(), z));
            }
        }

        let mut faces = Vec::new();
        for z_step in 0..num_vertical {
            for r_step in 0..num_radial {
                let next_r = (r_step + 1) % num_radial;
                let i00 = z_step * num_radial + r_step;
                let i01 = z_step * num_radial + next_r;
                let i10 = (z_step + 1) * num_radial + r_step;
                let i11 = (z_step + 1) * num_radial + next_r;

                faces.push([i00, i01, i11]);
                faces.push([i00, i11, i10]);
            }
        }

        let is_seed = |p: DVec3| p.z <= 0.01;
        let times = solve_surface_eikonal(&vertices, &faces, is_seed);

        for z_step in 1..=num_vertical {
            let z = (z_step as f64 / num_vertical as f64) * height;
            for r_step in 0..num_radial {
                let idx = z_step * num_radial + r_step;
                assert!(
                    times[idx] >= z - 1e-4,
                    "times[{}] = {}, expected >= {}",
                    idx,
                    times[idx],
                    z
                );
            }
        }
    }
}
