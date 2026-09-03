//! Triply Periodic Minimal Surfaces (TPMS) functional representations.
//!
//! Provides continuous, implicit minimal surface geometries (Gyroid, Schwarz Diamond,
//! Schwarz Primitive) used for self-supporting 3D volumetric lattice infill.

use crate::{FieldSample, ScalarField};
use glam::DVec3;
use std::f64::consts::PI;

/// Built-in Triply Periodic Minimal Surface (TPMS) geometric equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TpmsKind {
    /// Gyroid surface: $\sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx) = 0$.
    #[default]
    Gyroid,
    /// Schwarz Diamond (D) surface: $\cos(kx)\cos(ky)\cos(kz) - \sin(kx)\sin(ky)\sin(kz) = 0$.
    SchwarzD,
    /// Schwarz Primitive (P) surface: $\cos(kx) + \cos(ky) + \cos(kz) = 0$.
    SchwarzP,
}

/// A [`ScalarField`] implementing a 3D TPMS implicit surface.
#[derive(Debug, Clone, Copy)]
pub struct TpmsField {
    pub kind: TpmsKind,
    pub wavelength: f64,
    pub thickness_threshold: f64,
}

impl TpmsField {
    /// Builds a new `TpmsField` with the given spatial `wavelength` and `thickness_threshold`.
    pub fn new(kind: TpmsKind, wavelength: f64, thickness_threshold: f64) -> Self {
        Self {
            kind,
            wavelength: wavelength.abs().max(1e-4),
            thickness_threshold,
        }
    }

    /// Derives the spatial wavelength $\lambda$ from `infill_line_width` and fractional `density` $\in (0, 1]$.
    pub fn wavelength_for_density(line_width: f64, density: f64) -> f64 {
        let d = density.clamp(0.01, 1.0);
        let lw = line_width.abs().max(1e-4);
        lw / (d * 0.35)
    }
}

impl ScalarField for TpmsField {
    fn sample(&self, p: DVec3) -> FieldSample {
        let k = (2.0 * PI) / self.wavelength;
        let x = p.x * k;
        let y = p.y * k;
        let z = p.z * k;

        let (sx, cx) = (x.sin(), x.cos());
        let (sy, cy) = (y.sin(), y.cos());
        let (sz, cz) = (z.sin(), z.cos());

        match self.kind {
            TpmsKind::Gyroid => {
                let val = sx * cy + sy * cz + sz * cx;
                let grad_x = k * (cx * cy - sz * sx);
                let grad_y = k * (cy * cz - sx * sy);
                let grad_z = k * (cz * cx - sy * sz);
                FieldSample {
                    value: val,
                    gradient: DVec3::new(grad_x, grad_y, grad_z),
                }
            }
            TpmsKind::SchwarzD => {
                let val = cx * cy * cz - sx * sy * sz;
                let grad_x = -k * (sx * cy * cz + cx * sy * sz);
                let grad_y = -k * (cx * sy * cz + sx * cy * sz);
                let grad_z = -k * (cx * cy * sz + sx * sy * cz);
                FieldSample {
                    value: val,
                    gradient: DVec3::new(grad_x, grad_y, grad_z),
                }
            }
            TpmsKind::SchwarzP => {
                let val = cx + cy + cz;
                let grad_x = -k * sx;
                let grad_y = -k * sy;
                let grad_z = -k * sz;
                FieldSample {
                    value: val,
                    gradient: DVec3::new(grad_x, grad_y, grad_z),
                }
            }
        }
    }
}

/// A 3D Constructive Solid Geometry (CSG) intersection field that clips a TPMS
/// implicit surface strictly inside a solid volume signed-distance field:
/// $$F(\mathbf{x}) = \max(T_{\text{TPMS}}(\mathbf{x}), \; S(\mathbf{x}) - \text{infill\_iso})$$
#[derive(Debug, Clone, Copy)]
pub struct ClippedTpmsField<'a, S: ScalarField> {
    pub tpms: TpmsField,
    pub sdf: &'a S,
    pub infill_iso: f64,
}

impl<'a, S: ScalarField> ClippedTpmsField<'a, S> {
    #[must_use]
    pub fn new(tpms: TpmsField, sdf: &'a S, infill_iso: f64) -> Self {
        Self {
            tpms,
            sdf,
            infill_iso,
        }
    }
}

impl<'a, S: ScalarField> ScalarField for ClippedTpmsField<'a, S> {
    fn sample(&self, p: DVec3) -> FieldSample {
        let sdf_val = self.sdf.sample(p).value;
        let tpms_val = self.tpms.sample(p).value;
        let csg_val = tpms_val.max(sdf_val - self.infill_iso);
        FieldSample {
            value: csg_val,
            gradient: DVec3::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gyroid_evaluates_at_origin() {
        let field = TpmsField::new(TpmsKind::Gyroid, 10.0, 0.0);
        let s = field.sample(DVec3::ZERO);
        assert!((s.value - 0.0).abs() < 1e-6);
        assert!(s.gradient.length() > 0.0);
    }

    #[test]
    fn schwarz_p_evaluates_at_origin() {
        let field = TpmsField::new(TpmsKind::SchwarzP, 10.0, 0.0);
        let s = field.sample(DVec3::ZERO);
        // cos(0) + cos(0) + cos(0) = 3.0
        assert!((s.value - 3.0).abs() < 1e-6);
    }

    #[test]
    fn schwarz_d_evaluates_at_origin() {
        let field = TpmsField::new(TpmsKind::SchwarzD, 10.0, 0.0);
        let s = field.sample(DVec3::ZERO);
        // cos(0)cos(0)cos(0) - sin(0)sin(0)sin(0) = 1.0
        assert!((s.value - 1.0).abs() < 1e-6);
    }
}
