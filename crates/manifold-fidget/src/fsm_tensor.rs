//! 3D symmetric positive-definite metric and velocity tensors for anisotropic
//! Eikonal order field evaluation via the Fast Sweeping Method (FSM).

use glam::DVec3;

/// Symmetric 3x3 tensor represented by its 6 unique components:
///
/// ```text
/// [ mxx  mxy  mxz ]
/// [ mxy  myy  myz ]
/// [ mxz  myz  mzz ]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricTensor3 {
    pub mxx: f64,
    pub myy: f64,
    pub mzz: f64,
    pub mxy: f64,
    pub mxz: f64,
    pub myz: f64,
}

impl Default for MetricTensor3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl MetricTensor3 {
    /// Standard isotropic identity metric tensor: $M = \mathbf{I}$.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            mxx: 1.0,
            myy: 1.0,
            mzz: 1.0,
            mxy: 0.0,
            mxz: 0.0,
            myz: 0.0,
        }
    }

    /// Constructs a diagonal metric tensor:
    ///
    /// $$M = \operatorname{diag}(d_x, d_y, d_z)$$
    #[must_use]
    pub const fn from_diagonal(dx: f64, dy: f64, dz: f64) -> Self {
        Self {
            mxx: dx,
            myy: dy,
            mzz: dz,
            mxy: 0.0,
            mxz: 0.0,
            myz: 0.0,
        }
    }

    /// Constructs an anisotropic metric tensor aligned with an outward unit normal $\mathbf{n}$:
    ///
    /// $$M = \lambda_n \mathbf{n}\mathbf{n}^T + \lambda_t (\mathbf{I} - \mathbf{n}\mathbf{n}^T)$$
    ///
    /// where $\lambda_n$ is the eigenvalue along the normal direction, and $\lambda_t$
    /// is the eigenvalue across the tangent plane orthogonal to $\mathbf{n}$.
    ///
    /// - For **near-tangency** (level sets conforming parallel to the surface), set $\lambda_n \gg \lambda_t$.
    /// - For **near-orthogonality** (level sets intersecting normal to the surface), set $\lambda_t \gg \lambda_n$.
    #[must_use]
    pub fn from_normal_and_aspect(normal: DVec3, lambda_normal: f64, lambda_tangent: f64) -> Self {
        let len_sq = normal.length_squared();
        if len_sq <= 1e-12 {
            return Self::from_diagonal(lambda_tangent, lambda_tangent, lambda_tangent);
        }
        let n = normal / len_sq.sqrt();
        let delta = lambda_normal - lambda_tangent;

        Self {
            mxx: lambda_tangent + delta * n.x * n.x,
            myy: lambda_tangent + delta * n.y * n.y,
            mzz: lambda_tangent + delta * n.z * n.z,
            mxy: delta * n.x * n.y,
            mxz: delta * n.x * n.z,
            myz: delta * n.y * n.z,
        }
    }

    /// Evaluates the quadratic form $\mathbf{v}^T M \mathbf{v}$.
    #[must_use]
    #[inline]
    pub fn quadform(&self, v: DVec3) -> f64 {
        v.x * (self.mxx * v.x + 2.0 * self.mxy * v.y + 2.0 * self.mxz * v.z)
            + v.y * (self.myy * v.y + 2.0 * self.myz * v.z)
            + v.z * (self.mzz * v.z)
    }

    /// Evaluates the Riemannian norm $\|\mathbf{v}\|_M = \sqrt{\mathbf{v}^T M \mathbf{v}}$.
    #[must_use]
    #[inline]
    pub fn norm(&self, v: DVec3) -> f64 {
        self.quadform(v).max(0.0).sqrt()
    }

    /// Computes the matrix-vector product $M \mathbf{v}$.
    #[must_use]
    #[inline]
    pub fn transform(&self, v: DVec3) -> DVec3 {
        DVec3::new(
            self.mxx * v.x + self.mxy * v.y + self.mxz * v.z,
            self.mxy * v.x + self.myy * v.y + self.myz * v.z,
            self.mxz * v.x + self.myz * v.y + self.mzz * v.z,
        )
    }

    /// Computes the exact analytical determinant of the symmetric matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.mxx * (self.myy * self.mzz - self.myz * self.myz)
            - self.mxy * (self.mxy * self.mzz - self.myz * self.mxz)
            + self.mxz * (self.mxy * self.myz - self.myy * self.mxz)
    }

    /// Computes the analytical inverse $M^{-1}$, or returns `None` if the tensor is singular or ill-conditioned.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let c_xx = self.myy * self.mzz - self.myz * self.myz;
        let c_yy = self.mxx * self.mzz - self.mxz * self.mxz;
        let c_zz = self.mxx * self.myy - self.mxy * self.mxy;

        let c_xy = self.mxz * self.myz - self.mxy * self.mzz;
        let c_xz = self.mxy * self.myz - self.mxz * self.myy;
        let c_yz = self.mxy * self.mxz - self.mxx * self.myz;

        let det = self.mxx * c_xx + self.mxy * c_xy + self.mxz * c_xz;
        if det.abs() <= 1e-15 || !det.is_finite() {
            return None;
        }

        let inv_det = 1.0 / det;
        Some(Self {
            mxx: c_xx * inv_det,
            myy: c_yy * inv_det,
            mzz: c_zz * inv_det,
            mxy: c_xy * inv_det,
            mxz: c_xz * inv_det,
            myz: c_yz * inv_det,
        })
    }

    /// Component-wise linear interpolation between `self` and `other`: $(1-t)\,\text{self} + t\,\text{other}$.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t_clamped = t.clamp(0.0, 1.0);
        let s = 1.0 - t_clamped;
        Self {
            mxx: s * self.mxx + t_clamped * other.mxx,
            myy: s * self.myy + t_clamped * other.myy,
            mzz: s * self.mzz + t_clamped * other.mzz,
            mxy: s * self.mxy + t_clamped * other.mxy,
            mxz: s * self.mxz + t_clamped * other.mxz,
            myz: s * self.myz + t_clamped * other.myz,
        }
    }

    /// Matrix addition $A + B$.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            mxx: self.mxx + other.mxx,
            myy: self.myy + other.myy,
            mzz: self.mzz + other.mzz,
            mxy: self.mxy + other.mxy,
            mxz: self.mxz + other.mxz,
            myz: self.myz + other.myz,
        }
    }

    /// Scalar multiplication $s \cdot M$.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            mxx: self.mxx * s,
            myy: self.myy * s,
            mzz: self.mzz * s,
            mxy: self.mxy * s,
            mxz: self.mxz * s,
            myz: self.myz * s,
        }
    }
}

/// A 3D discrete grid of symmetric positive-definite metric tensors.
pub struct TensorGrid {
    pub min_corner: DVec3,
    pub dims: [usize; 3],
    pub h: f64,
    pub tensors: Vec<MetricTensor3>,
}

impl TensorGrid {
    /// Creates a uniform isotropic tensor grid where $M(x, y, z) = \mathbf{I}$ everywhere.
    #[must_use]
    pub fn new_isotropic(min_corner: DVec3, dims: [usize; 3], h: f64) -> Self {
        let total = dims[0] * dims[1] * dims[2];
        Self {
            min_corner,
            dims,
            h,
            tensors: vec![MetricTensor3::identity(); total],
        }
    }

    #[inline]
    #[must_use]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    #[inline]
    #[must_use]
    pub fn get(&self, x: usize, y: usize, z: usize) -> MetricTensor3 {
        let i = self.idx(x, y, z);
        self.tensors[i]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, tensor: MetricTensor3) {
        let i = self.idx(x, y, z);
        self.tensors[i] = tensor;
    }

    /// World-space coordinate of node `(x, y, z)`.
    #[inline]
    #[must_use]
    pub fn node_pos(&self, x: usize, y: usize, z: usize) -> DVec3 {
        DVec3::new(
            self.min_corner.x + x as f64 * self.h,
            self.min_corner.y + y as f64 * self.h,
            self.min_corner.z + z as f64 * self.h,
        )
    }

    /// Blends surface anisotropic tensors into the grid within a `skin_depth_mm` layer
    /// using Gaussian spatial weighting.
    pub fn blend_surface_tensors(
        &mut self,
        surface_sdf: impl Fn(DVec3) -> (f64, DVec3),
        top_tangency_aspect: f64,
        wall_ortho_aspect: f64,
        skin_depth_mm: f64,
    ) {
        if skin_depth_mm <= 1e-6 {
            return;
        }

        let sigma = skin_depth_mm / 2.0;
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
        let [nx, ny, nz] = self.dims;

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let p = self.node_pos(x, y, z);
                    let (dist, normal) = surface_sdf(p);

                    // Only blend inside or near the surface skin layer
                    if dist.abs() <= skin_depth_mm {
                        let weight = (-dist * dist * inv_two_sigma_sq).exp().clamp(0.0, 1.0);
                        if weight > 1e-4 {
                            // Upward normal dot Z: +1 = flat top face, 0 = vertical wall, -1 = underside
                            let n_z = normal.z.clamp(-1.0, 1.0);
                            let target_tensor = if n_z > 0.3 {
                                // Upward facing surface: conform parallel (tangent) to surface
                                MetricTensor3::from_normal_and_aspect(
                                    normal,
                                    top_tangency_aspect,
                                    1.0 / top_tangency_aspect,
                                )
                            } else if n_z.abs() <= 0.3 {
                                // Vertical or steep wall: intersect orthogonal to surface
                                MetricTensor3::from_normal_and_aspect(
                                    normal,
                                    1.0 / wall_ortho_aspect,
                                    wall_ortho_aspect,
                                )
                            } else {
                                MetricTensor3::identity()
                            };

                            let current = self.get(x, y, z);
                            let blended = current.lerp(&target_tensor, weight);
                            self.set(x, y, z, blended);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_tensor_norm_matches_euclidean_length() {
        let m = MetricTensor3::identity();
        let v = DVec3::new(3.0, 4.0, 12.0);
        let norm = m.norm(v);
        assert!((norm - 13.0).abs() < 1e-9);
    }

    #[test]
    fn spectral_decomposition_yields_expected_eigenvalues() {
        let normal = DVec3::new(0.0, 0.0, 1.0);
        let lambda_n = 4.0;
        let lambda_t = 0.25;
        let m = MetricTensor3::from_normal_and_aspect(normal, lambda_n, lambda_t);

        // Vector along normal
        let vn = DVec3::new(0.0, 0.0, 2.0);
        let norm_n_sq = m.quadform(vn);
        assert!((norm_n_sq - 4.0 * (2.0 * 2.0)).abs() < 1e-9);

        // Vector in tangent plane
        let vt = DVec3::new(2.0, 0.0, 0.0);
        let norm_t_sq = m.quadform(vt);
        assert!((norm_t_sq - 0.25 * (2.0 * 2.0)).abs() < 1e-9);
    }

    #[test]
    fn tensor_inverse_multiplication_yields_identity() {
        let normal = DVec3::new(1.0, 2.0, 3.0).normalize();
        let m = MetricTensor3::from_normal_and_aspect(normal, 3.0, 0.5);
        let inv = m.inverse().expect("tensor should be invertible");

        // Verify M * inv * v == v for test vectors
        for v in [DVec3::X, DVec3::Y, DVec3::Z, DVec3::new(1.2, -3.4, 5.6)] {
            let transformed = m.transform(inv.transform(v));
            assert!(
                (transformed - v).length() < 1e-9,
                "Expected {:?}, got {:?}",
                v,
                transformed
            );
        }
    }

    #[test]
    fn tensor_lerp_blends_smoothly() {
        let a = MetricTensor3::identity();
        let b = MetricTensor3::from_diagonal(2.0, 4.0, 8.0);
        let mid = a.lerp(&b, 0.5);

        assert!((mid.mxx - 1.5).abs() < 1e-9);
        assert!((mid.myy - 2.5).abs() < 1e-9);
        assert!((mid.mzz - 4.5).abs() < 1e-9);
    }
}
