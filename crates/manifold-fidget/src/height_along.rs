//! [`HeightAlong`]: projects a world point onto "distance along the
//! build/nozzle direction from the seed reference," for [`SlopeProfile`]
//! lookup (see `slope_profile.rs`).
//!
//! Today's slicing convention (see `manifold-core`'s `order_field.rs`)
//! treats "height" as a fixed axis (`BUILD_DIRECTION`) and a fixed seed
//! reference (the mesh bounding-box minimum along that axis); this trait
//! exists so a future spatially-varying or tilting-nozzle direction can
//! implement the same interface without [`SlopeProfile`] or its callers
//! needing to change at all — they only ever call
//! [`HeightAlong::height`].
//!
//! [`SlopeProfile`]: crate::slope_profile::SlopeProfile

use glam::DVec3;

/// Projects a world point `p` onto "distance along the build/nozzle
/// direction from the seed reference."
///
/// Implementations are expected to be cheap and side-effect-free, since
/// they may be called once per grid node during an Eikonal march.
pub trait HeightAlong {
    /// Returns the height of `p` along this implementation's direction,
    /// relative to its seed reference.
    fn height(&self, p: DVec3) -> f64;
}

/// A [`HeightAlong`] implementation for today's convention: a single fixed
/// axis and a single fixed seed reference point, matching
/// `manifold-core`'s `order_field.rs` usage of `BUILD_DIRECTION` and the
/// mesh bounding-box minimum.
///
/// `height(p) = (p - seed_reference).dot(axis)`.
///
/// # Degenerate input
///
/// `axis` is not required to be normalized or non-degenerate (e.g. a
/// zero-length or NaN-containing vector is accepted). [`ConstantAxisHeight::new`]
/// does not validate or normalize `axis` — [`DVec3::dot`] does not panic
/// on a zero-length or NaN vector, so a degenerate `axis` simply
/// propagates a NaN (or zero) result through [`HeightAlong::height`] per
/// standard IEEE 754 floating-point semantics, rather than panicking.
/// Callers that need a strictly non-degenerate axis are responsible for
/// validating it before construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantAxisHeight {
    axis: DVec3,
    seed_reference: DVec3,
}

impl ConstantAxisHeight {
    /// Builds a [`ConstantAxisHeight`] from a fixed `axis` and
    /// `seed_reference` point.
    ///
    /// See the struct docs for degenerate/NaN `axis` handling: no
    /// validation is performed here, and no panic occurs for any input.
    pub fn new(axis: DVec3, seed_reference: DVec3) -> Self {
        Self {
            axis,
            seed_reference,
        }
    }
}

impl HeightAlong for ConstantAxisHeight {
    fn height(&self, p: DVec3) -> f64 {
        (p - self.seed_reference).dot(self.axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn projects_along_z_axis_from_nontrivial_seed_reference() {
        let height_along = ConstantAxisHeight::new(DVec3::Z, DVec3::new(1.0, 2.0, 5.0));

        // Only the Z component beyond the seed reference's Z should
        // contribute; X/Y offsets should not affect the result.
        assert!(approx_eq(
            height_along.height(DVec3::new(1.0, 2.0, 5.0)),
            0.0,
            1e-9
        ));
        assert!(approx_eq(
            height_along.height(DVec3::new(100.0, -50.0, 15.0)),
            10.0,
            1e-9
        ));
        assert!(approx_eq(
            height_along.height(DVec3::new(0.0, 0.0, 0.0)),
            -5.0,
            1e-9
        ));
    }

    #[test]
    fn projects_along_non_axis_aligned_unit_vector() {
        // A unit vector along the diagonal of the XY plane's positive
        // quadrant: (1, 1, 0) normalized.
        let axis = DVec3::new(1.0, 1.0, 0.0).normalize();
        let height_along = ConstantAxisHeight::new(axis, DVec3::ZERO);

        // Point (1, 1, 0) projected onto the (1,1,0)-normalized axis:
        // dot((1,1,0), (1,1,0)/sqrt(2)) = 2/sqrt(2) = sqrt(2).
        let expected = 2.0_f64 / 2.0_f64.sqrt();
        assert!(approx_eq(
            height_along.height(DVec3::new(1.0, 1.0, 0.0)),
            expected,
            1e-9
        ));

        // A point purely along Z (perpendicular to the axis) should
        // project to 0.
        assert!(approx_eq(
            height_along.height(DVec3::new(0.0, 0.0, 42.0)),
            0.0,
            1e-9
        ));
    }

    #[test]
    fn height_is_invariant_to_orthogonal_displacement() {
        // Axis along Z; displacing p purely within the XY plane (orthogonal
        // to the axis) must not change the projected height.
        let axis = DVec3::Z;
        let height_along = ConstantAxisHeight::new(axis, DVec3::new(1.0, 2.0, 3.0));

        let base = height_along.height(DVec3::new(1.0, 2.0, 10.0));
        let shifted = height_along.height(DVec3::new(-40.0, 99.0, 10.0));

        assert!(approx_eq(base, shifted, 1e-9));
        assert!(approx_eq(base, 7.0, 1e-9));
    }

    #[test]
    fn projects_along_non_unit_length_diagonal_axis() {
        // A non-unit-length, non-axis-aligned axis: (2, 0, 2). Note this
        // is intentionally *not* normalized, unlike the unit-diagonal test
        // above, to cover the raw dot-product (no implicit normalization).
        let axis = DVec3::new(2.0, 0.0, 2.0);
        let seed_reference = DVec3::new(1.0, 1.0, 1.0);
        let height_along = ConstantAxisHeight::new(axis, seed_reference);

        let p = DVec3::new(4.0, 5.0, 2.0);
        // (p - seed_reference) = (3, 4, 1); dot((3,4,1), (2,0,2)) = 6 + 0 + 2 = 8.
        let expected = 8.0;
        assert!(approx_eq(height_along.height(p), expected, 1e-9));
    }

    #[test]
    fn nan_axis_does_not_panic_and_propagates_nan() {
        let height_along = ConstantAxisHeight::new(DVec3::new(f64::NAN, 0.0, 0.0), DVec3::ZERO);

        let result = height_along.height(DVec3::new(1.0, 2.0, 3.0));
        assert!(result.is_nan());
    }

    #[test]
    fn zero_length_axis_yields_zero_height_without_panicking() {
        let height_along = ConstantAxisHeight::new(DVec3::ZERO, DVec3::new(3.0, 4.0, 5.0));

        let result = height_along.height(DVec3::new(10.0, 20.0, 30.0));
        assert!(approx_eq(result, 0.0, 1e-9));
    }
}
