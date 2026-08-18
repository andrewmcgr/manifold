//! [`SlopeProfile`]: a piecewise angle-vs-height limit used to cap how
//! steep (how much overhang) an Eikonal order-field layer is allowed to be
//! at a given height along the print axis.
//!
//! A profile is a set of `(height_along_axis_mm, max_angle_deg)`
//! breakpoints: below the first breakpoint's height, that breakpoint's
//! angle applies; between two breakpoints, the *lower* breakpoint's angle
//! applies up to the next one; at or above the last breakpoint's height,
//! the last breakpoint's angle applies to all remaining height. This lets
//! a caller express e.g. "tighter overhang limit near the bed, looser
//! higher up" without the order-field code needing to know about height
//! bands at all — it just calls [`SlopeProfile::max_slope_at`] per point.
//!
//! A single-entry profile such as `[(f64::INFINITY, 45.0)]` expresses a
//! constant 45-degree limit everywhere: there is no separate "constant"
//! code path, it falls out of the same piecewise lookup because there is
//! nothing above the one breakpoint to switch to.
//!
//! Degenerate inputs are handled as documented graceful degradation rather
//! than panics, matching this crate's convention (see `eikonal.rs`) of
//! never `unwrap()`/panicking on degenerate geometry/config input:
//! - Empty breakpoints: treated as unconstrained (no additional slope
//!   limit), so [`SlopeProfile::max_slope_at`] returns [`UNCONSTRAINED_ANGLE_DEG`].
//! - Non-ascending breakpoints: tolerated by sorting on construction, so
//!   lookups never see out-of-order data or infinite-loop scanning for it.
//! - Zero/negative angle: clamped up to [`MIN_ANGLE_DEG`] so downstream
//!   slope-multiplier math never divides by zero or goes negative.
//! - Angle >= 90 degrees: clamped down to [`UNCONSTRAINED_ANGLE_DEG`], i.e.
//!   treated as "no limit" rather than producing an infinite or NaN slope
//!   multiplier downstream (tan(90 deg) is undefined).

/// Angle (degrees) used to mean "no slope limit" wherever a breakpoint's
/// angle would otherwise be undefined or >= 90 degrees.
pub const UNCONSTRAINED_ANGLE_DEG: f64 = 90.0 - 1e-6;

/// Smallest angle (degrees) a breakpoint is allowed to clamp down to, so a
/// zero/negative configured angle never reaches downstream slope math as
/// zero (which would demand a perfectly flat, unachievable layer).
pub const MIN_ANGLE_DEG: f64 = 1e-3;

/// A piecewise angle-vs-height overhang limit: see the module docs for the
/// full semantics and degenerate-input handling.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeProfile {
    /// `(height_along_axis_mm, max_angle_deg)` pairs, sorted ascending by
    /// height and with each angle already clamped to
    /// `[MIN_ANGLE_DEG, UNCONSTRAINED_ANGLE_DEG]`.
    breakpoints: Vec<(f64, f64)>,
}

impl SlopeProfile {
    /// Builds a [`SlopeProfile`] from `breakpoints`.
    ///
    /// Breakpoints are sorted ascending by height (so non-ascending input
    /// is tolerated rather than producing undefined lookup behavior), and
    /// every angle is clamped into `[MIN_ANGLE_DEG, UNCONSTRAINED_ANGLE_DEG]`
    /// (so a zero/negative or >=90-degree angle never reaches
    /// [`SlopeProfile::max_slope_at`] callers as-is).
    pub fn new(mut breakpoints: Vec<(f64, f64)>) -> Self {
        breakpoints.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, angle) in breakpoints.iter_mut() {
            *angle = angle.clamp(MIN_ANGLE_DEG, UNCONSTRAINED_ANGLE_DEG);
        }
        Self { breakpoints }
    }

    /// Returns the max overhang angle in degrees allowed at `height`.
    ///
    /// Semantics: finds the last breakpoint whose height is `<= height`
    /// and returns its angle; if `height` is below every breakpoint,
    /// returns the first (lowest-height) breakpoint's angle; if there are
    /// no breakpoints at all, returns [`UNCONSTRAINED_ANGLE_DEG`] (no
    /// additional limit).
    pub fn max_slope_at(&self, height: f64) -> f64 {
        if self.breakpoints.is_empty() {
            return UNCONSTRAINED_ANGLE_DEG;
        }

        // Breakpoints are sorted ascending by height (enforced in `new`),
        // so the last one with height <= query height is the applicable
        // one; if none qualify (height below the first breakpoint), fall
        // back to the first breakpoint's angle.
        match self.breakpoints.iter().rev().find(|(h, _)| *h <= height) {
            Some((_, angle)) => *angle,
            None => self.breakpoints[0].1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn single_entry_profile_is_constant_everywhere() {
        let profile = SlopeProfile::new(vec![(f64::INFINITY, 45.0)]);
        assert!(approx_eq(profile.max_slope_at(-1000.0), 45.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(0.0), 45.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(1e9), 45.0, 1e-9));
    }

    #[test]
    fn piecewise_lookup_uses_lower_breakpoints_angle_up_to_next() {
        let profile = SlopeProfile::new(vec![(0.0, 30.0), (10.0, 60.0), (20.0, 90.0 - 1e-6)]);

        assert!(approx_eq(profile.max_slope_at(0.0), 30.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(5.0), 30.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(10.0), 60.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(15.0), 60.0, 1e-9));
        // At/above the last breakpoint, its angle applies to all
        // remaining height.
        assert!(approx_eq(
            profile.max_slope_at(20.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
        assert!(approx_eq(
            profile.max_slope_at(1000.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
    }

    #[test]
    fn below_first_breakpoint_uses_first_breakpoints_angle() {
        let profile = SlopeProfile::new(vec![(10.0, 45.0), (20.0, 60.0)]);
        assert!(approx_eq(profile.max_slope_at(-5.0), 45.0, 1e-9));
    }

    #[test]
    fn empty_breakpoints_is_unconstrained_and_does_not_panic() {
        let profile = SlopeProfile::new(vec![]);
        assert!(approx_eq(
            profile.max_slope_at(0.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
        assert!(approx_eq(
            profile.max_slope_at(f64::NEG_INFINITY),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
    }

    #[test]
    fn non_ascending_breakpoints_are_sorted_not_panicking() {
        // Deliberately out of order; construction should sort rather than
        // panic or produce a nonsensical lookup.
        let profile = SlopeProfile::new(vec![(20.0, 90.0 - 1e-6), (0.0, 30.0), (10.0, 60.0)]);

        assert!(approx_eq(profile.max_slope_at(5.0), 30.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(15.0), 60.0, 1e-9));
        assert!(approx_eq(
            profile.max_slope_at(25.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
    }

    #[test]
    fn zero_or_negative_angle_is_clamped_to_min() {
        let profile = SlopeProfile::new(vec![(0.0, 0.0), (10.0, -30.0)]);
        assert!(approx_eq(profile.max_slope_at(0.0), MIN_ANGLE_DEG, 1e-9));
        assert!(approx_eq(profile.max_slope_at(10.0), MIN_ANGLE_DEG, 1e-9));
    }

    #[test]
    fn angle_at_or_above_ninety_is_clamped_to_unconstrained() {
        let profile = SlopeProfile::new(vec![(0.0, 90.0), (10.0, 180.0)]);
        assert!(approx_eq(
            profile.max_slope_at(0.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
        assert!(approx_eq(
            profile.max_slope_at(10.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
    }

    #[test]
    fn height_after_last_breakpoint_uses_last_breakpoints_angle() {
        // Distinct, non-clamped angles so the assertion can't pass by
        // accident via UNCONSTRAINED_ANGLE_DEG coincidence.
        let profile = SlopeProfile::new(vec![(0.0, 30.0), (10.0, 45.0)]);
        assert!(approx_eq(profile.max_slope_at(10.0), 45.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(11.0), 45.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(1e6), 45.0, 1e-9));
    }

    #[test]
    fn at_breakpoint_height_resolves_to_that_breakpoints_angle() {
        // Documents the boundary choice made by `max_slope_at`: querying
        // exactly at a breakpoint's height returns *that* breakpoint's
        // angle, not the previous segment's.
        let profile = SlopeProfile::new(vec![(0.0, 30.0), (10.0, 45.0), (20.0, 60.0)]);
        assert!(approx_eq(profile.max_slope_at(10.0), 45.0, 1e-9));
        assert!(approx_eq(profile.max_slope_at(20.0), 60.0, 1e-9));
    }

    #[test]
    fn clamped_angle_is_never_nan_or_negative() {
        let profile = SlopeProfile::new(vec![(0.0, 0.0), (10.0, -30.0), (20.0, 180.0)]);
        for h in [-1.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
            let angle = profile.max_slope_at(h);
            assert!(angle.is_finite());
            assert!(angle > 0.0);
        }
    }

    #[test]
    fn nan_height_query_does_not_panic() {
        let profile = SlopeProfile::new(vec![(0.0, 30.0), (10.0, 60.0)]);
        // NaN comparisons are always false, so `find` finds nothing and we
        // fall back to the first breakpoint's angle rather than panicking.
        let result = profile.max_slope_at(f64::NAN);
        assert!(result.is_finite());
    }
}
