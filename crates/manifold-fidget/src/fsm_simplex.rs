//! Local 3D anisotropic simplex quadratic solver for Cartesian grids.
//!
//! Solves the local anisotropic Eikonal equation:
//!
//! $$\nabla \phi^T \mathbf{D} \nabla \phi = 1$$
//!
//! across the 3D, 2D, and 1D faces of a Cartesian grid cell simplex, enforcing
//! upwind causality along the characteristic group velocity vector $\mathbf{v}_g = \mathbf{D} \nabla \phi$.

use crate::fsm_tensor::MetricTensor3;
use glam::DVec3;

/// Solves the local anisotropic Eikonal update for a grid node given its upwind
/// neighbors along the $X, Y, Z$ axes in a coordinate octant defined by step signs
/// $(s_x, s_y, s_z) \in \{-1.0, +1.0\}^3$.
///
/// - `phi_x, phi_y, phi_z`: neighbor arrival values (may be `f64::INFINITY` if unreached or void).
/// - `sx, sy, sz`: direction signs from neighbor to current node ($+1.0$ if neighbor is at $x-1$, $-1.0$ if at $x+1$).
/// - `d`: local symmetric positive-definite velocity tensor $\mathbf{D} = \mathbf{M}^{-1}$.
/// - `h`: grid spacing.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn solve_anisotropic_octant_simplex(
    phi_x: f64,
    phi_y: f64,
    phi_z: f64,
    sx: f64,
    sy: f64,
    sz: f64,
    d: &MetricTensor3,
    h: f64,
) -> f64 {
    let mut best_u = f64::INFINITY;

    // 1. Try 3D simplex interior if all three neighbors have finite values
    if phi_x.is_finite() && phi_y.is_finite() && phi_z.is_finite() {
        if let Some(u_3d) = solve_3d_simplex(phi_x, phi_y, phi_z, sx, sy, sz, d, h) {
            if u_3d < best_u {
                best_u = u_3d;
            }
        }
    }

    // 2. Try 2D faces
    if phi_x.is_finite() && phi_y.is_finite() {
        if let Some(u_xy) = solve_2d_face(phi_x, phi_y, sx, sy, d.mxx, d.myy, d.mxy, h) {
            if u_xy < best_u {
                best_u = u_xy;
            }
        }
    }
    if phi_y.is_finite() && phi_z.is_finite() {
        if let Some(u_yz) = solve_2d_face(phi_y, phi_z, sy, sz, d.myy, d.mzz, d.myz, h) {
            if u_yz < best_u {
                best_u = u_yz;
            }
        }
    }
    if phi_x.is_finite() && phi_z.is_finite() {
        if let Some(u_xz) = solve_2d_face(phi_x, phi_z, sx, sz, d.mxx, d.mzz, d.mxz, h) {
            if u_xz < best_u {
                best_u = u_xz;
            }
        }
    }

    // 3. Fallback to 1D edges
    if phi_x.is_finite() && d.mxx > 1e-12 {
        let u_1d = phi_x + h / d.mxx.sqrt();
        if u_1d < best_u {
            best_u = u_1d;
        }
    }
    if phi_y.is_finite() && d.myy > 1e-12 {
        let u_1d = phi_y + h / d.myy.sqrt();
        if u_1d < best_u {
            best_u = u_1d;
        }
    }
    if phi_z.is_finite() && d.mzz > 1e-12 {
        let u_1d = phi_z + h / d.mzz.sqrt();
        if u_1d < best_u {
            best_u = u_1d;
        }
    }

    best_u
}

/// Solves the 3D anisotropic simplex update.
#[allow(clippy::too_many_arguments)]
fn solve_3d_simplex(
    phi_x: f64,
    phi_y: f64,
    phi_z: f64,
    sx: f64,
    sy: f64,
    sz: f64,
    d: &MetricTensor3,
    h: f64,
) -> Option<f64> {
    let inv_h = 1.0 / h;
    let a = DVec3::new(sx * inv_h, sy * inv_h, sz * inv_h);
    let b = DVec3::new(sx * phi_x * inv_h, sy * phi_y * inv_h, sz * phi_z * inv_h);

    let quad_a = d.quadform(a);
    if quad_a <= 1e-15 {
        return None;
    }

    let d_b = d.transform(b);
    let a_dot_db = a.dot(d_b);
    let quad_b = b.dot(d_b);

    let discr = a_dot_db * a_dot_db - quad_a * (quad_b - 1.0);
    if discr < 0.0 {
        return None;
    }

    let u = (a_dot_db + discr.sqrt()) / quad_a;

    // Upwind causality check: u must strictly exceed neighbor arrival times
    if u <= phi_x || u <= phi_y || u <= phi_z {
        return None;
    }

    // Characteristic ray direction v_g = D * p
    let p = u * a - b;
    let v_g = d.transform(p);

    // Group velocity must point into the current node from all 3 directions
    if sx * v_g.x > 0.0 && sy * v_g.y > 0.0 && sz * v_g.z > 0.0 {
        Some(u)
    } else {
        None
    }
}

/// Solves the 2D anisotropic planar face update for axes $i$ and $j$.
#[allow(clippy::too_many_arguments)]
fn solve_2d_face(
    phi_i: f64,
    phi_j: f64,
    si: f64,
    sj: f64,
    d_ii: f64,
    d_jj: f64,
    d_ij: f64,
    h: f64,
) -> Option<f64> {
    let inv_h = 1.0 / h;
    let a_i = si * inv_h;
    let a_j = sj * inv_h;
    let b_i = si * phi_i * inv_h;
    let b_j = sj * phi_j * inv_h;

    // Quadform A = a^T D a
    let quad_a = a_i * (d_ii * a_i + d_ij * a_j) + a_j * (d_ij * a_i + d_jj * a_j);
    if quad_a <= 1e-15 {
        return None;
    }

    // D * b
    let db_i = d_ii * b_i + d_ij * b_j;
    let db_j = d_ij * b_i + d_jj * b_j;

    let a_dot_db = a_i * db_i + a_j * db_j;
    let quad_b = b_i * db_i + b_j * db_j;

    let discr = a_dot_db * a_dot_db - quad_a * (quad_b - 1.0);
    if discr < 0.0 {
        return None;
    }

    let u = (a_dot_db + discr.sqrt()) / quad_a;

    if u <= phi_i || u <= phi_j {
        return None;
    }

    // 2D group velocity v_g = D * p
    let p_i = u * a_i - b_i;
    let p_j = u * a_j - b_j;
    let vg_i = d_ii * p_i + d_ij * p_j;
    let vg_j = d_ij * p_i + d_jj * p_j;

    if si * vg_i > 0.0 && sj * vg_j > 0.0 {
        Some(u)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_diagonal_simplex_matches_analytical_distance() {
        let h = 1.0;
        let d = MetricTensor3::identity(); // D = I, speed = 1.0

        // Suppose node at (1, 1, 1) is being evaluated from neighbors:
        // (0, 1, 1) with phi = 1.0
        // (1, 0, 1) with phi = 1.0
        // (1, 1, 0) with phi = 1.0
        // Sourced from (0, 0, 0) with phi = 0.
        // True distance from (0, 0, 0) to (1, 1, 1) is sqrt(3) ~= 1.73205
        let u = solve_anisotropic_octant_simplex(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, &d, h);

        // Analytical solution to (u - 1)^2 * 3 = 1 => u = 1 + 1/sqrt(3) ~= 1.57735
        // (Cartesian discretization under Godunov)
        let expected = 1.0 + 1.0 / 3.0f64.sqrt();
        assert!(
            (u - expected).abs() < 1e-9,
            "Expected {}, got {}",
            expected,
            u
        );
    }

    #[test]
    fn anisotropic_1d_edge_reflects_directional_speed() {
        let h = 0.5;
        // D_xx = 4.0 => speed_x = 2.0
        let d = MetricTensor3::from_diagonal(4.0, 1.0, 1.0);

        let u = solve_anisotropic_octant_simplex(
            2.0,
            f64::INFINITY,
            f64::INFINITY,
            1.0,
            1.0,
            1.0,
            &d,
            h,
        );

        // Arrival time u = 2.0 + 0.5 / 2.0 = 2.25
        assert!((u - 2.25).abs() < 1e-9, "Expected 2.25, got {}", u);
    }

    #[test]
    fn causality_falls_back_to_2d_when_3d_characteristic_leaves_simplex() {
        let h = 1.0;
        let d = MetricTensor3::identity();

        // One neighbor is very far away, so characteristic lies on the 2D plane of the other two
        let u = solve_anisotropic_octant_simplex(1.0, 1.0, 100.0, 1.0, 1.0, 1.0, &d, h);

        // Solves in 2D face: (u - 1)^2 * 2 = 1 => u = 1 + 1/sqrt(2) ~= 1.7071
        let expected = 1.0 + 1.0 / 2.0f64.sqrt();
        assert!(
            (u - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            u
        );
    }
}
