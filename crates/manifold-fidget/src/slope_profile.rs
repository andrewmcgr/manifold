//! [`SlopeProfile`]: a toolhead clearance profile expressed as a series of
//! `(x_mm, z_mm)` points, used to cap how steep (how much overhang) an
//! Eikonal order-field layer is allowed to be at a given height along the print axis.
//!
//! A profile is a series of $(X, Z)$ points defining the radial clearance envelope of the
//! toolhead measured from the nozzle tip at $(X=0, Z=0)$:
//! - $X$: radial horizontal distance from nozzle center in mm ($X \ge 0$).
//! - $Z$: vertical clearance height above the nozzle tip in mm ($Z \ge 0$).
//!
//! At any query height $h$ (height above nozzle tip / feature), the toolhead envelope's
//! radial boundary $X(h)$ is interpolated piecewise linearly between adjacent $(X_i, Z_i)$ points.
//! The allowable slope angle $\theta(h)$ is the most restrictive line-of-sight angle to any point
//! on the envelope at or below height $h$:
//!
//! $$\tan \theta(h) = \min \left( \frac{h}{X(h)}, \min_{i: Z_i \le h, Z_i > 0} \frac{Z_i}{X_i} \right)$$
//!
//! Degenerate inputs are handled gracefully without panics:
//! - Empty points: unconstrained ($90^\circ$).
//! - Non-ascending points: sorted by $Z$ on construction.
//! - Negative or zero $X$: clamped to a minimum clearance radius ($10^{-3}\text{ mm}$).
//! - Negative $Z$: clamped to $0.0\text{ mm}$.

/// Angle (degrees) used to mean "no slope limit" wherever a point's
/// angle would otherwise be undefined or >= 90 degrees.
pub const UNCONSTRAINED_ANGLE_DEG: f64 = 90.0 - 1e-6;

/// Smallest angle (degrees) a slope profile is allowed to clamp down to, so a
/// near-zero configured slope never reaches downstream math as zero.
pub const MIN_ANGLE_DEG: f64 = 1e-3;

/// Minimum radial distance (mm) used to prevent division by zero.
pub const MIN_RADIUS_MM: f64 = 1e-6;

/// A toolhead clearance profile defined as a series of (x_mm, z_mm) points.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeProfile {
    /// `(x_mm, z_mm)` points sorted ascending by `z`, with `x >= MIN_RADIUS_MM`
    /// and `z >= 0.0`.
    points: Vec<(f64, f64)>,
}

impl SlopeProfile {
    /// Builds a [`SlopeProfile`] from `(x_mm, z_mm)` points.
    ///
    /// Points are sorted ascending by `z`, with `x` clamped to `>= MIN_RADIUS_MM`
    /// and `z` clamped to `>= 0.0`.
    pub fn new(mut points: Vec<(f64, f64)>) -> Self {
        for (x, z) in points.iter_mut() {
            *x = (*x).max(MIN_RADIUS_MM);
            *z = (*z).max(0.0);
        }
        points.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        });
        Self { points }
    }

    /// Builds a [`SlopeProfile`] representing a constant maximum slope angle in degrees.
    pub fn from_angle(angle_deg: f64) -> Self {
        let clamped = angle_deg.clamp(MIN_ANGLE_DEG, UNCONSTRAINED_ANGLE_DEG);
        let tan = clamped.to_radians().tan();
        let x = 1000.0 / tan;
        Self::new(vec![(0.0, 0.0), (x, 1000.0)])
    }

    /// Returns the raw `(x_mm, z_mm)` points defining the clearance envelope.
    pub fn points(&self) -> &[(f64, f64)] {
        &self.points
    }

    /// Returns the maximum allowable slope angle in degrees at `height` above the nozzle tip.
    pub fn max_slope_at(&self, height: f64) -> f64 {
        if self.points.is_empty() {
            return UNCONSTRAINED_ANGLE_DEG;
        }
        let slope = self.max_slope_multiplier_at(height);
        if !slope.is_finite() || slope >= 1e6 {
            return UNCONSTRAINED_ANGLE_DEG;
        }
        slope
            .atan()
            .to_degrees()
            .clamp(MIN_ANGLE_DEG, UNCONSTRAINED_ANGLE_DEG)
    }

    /// Returns the maximum allowable slope multiplier ($\tan \theta = \Delta z / \Delta r$)
    /// at `height` above the nozzle tip.
    pub fn max_slope_multiplier_at(&self, height: f64) -> f64 {
        if self.points.is_empty() {
            return UNCONSTRAINED_ANGLE_DEG.to_radians().tan();
        }
        let h = height.max(0.0);
        let n = self.points.len();

        // 1. Evaluate radius X(h) piecewise linearly
        let x_h = if n == 1 || h <= self.points[0].1 {
            self.points[0].0
        } else if h >= self.points[n - 1].1 {
            self.points[n - 1].0
        } else {
            let mut interp = self.points[n - 1].0;
            for i in 0..n - 1 {
                let (x0, z0) = self.points[i];
                let (x1, z1) = self.points[i + 1];
                if h >= z0 && h <= z1 {
                    let dz = z1 - z0;
                    if dz > 1e-9 {
                        let t = (h - z0) / dz;
                        interp = x0 + t * (x1 - x0);
                    } else {
                        interp = x0.max(x1);
                    }
                    break;
                }
            }
            interp
        };

        // 2. Direct slope to envelope at height h
        let mut min_slope = if h <= 1e-6 {
            if self.points[0].1 <= 1e-6 && n > 1 {
                let (x1, z1) = self.points[1];
                let (x0, _) = self.points[0];
                if z1 > 1e-6 {
                    z1 / x1.max(x0)
                } else {
                    MIN_ANGLE_DEG.to_radians().tan()
                }
            } else if self.points[0].1 > 1e-6 {
                self.points[0].1 / self.points[0].0
            } else {
                UNCONSTRAINED_ANGLE_DEG.to_radians().tan()
            }
        } else {
            h / x_h.max(MIN_RADIUS_MM)
        };

        // 3. Slope to any envelope vertex at or below h
        for &(px, pz) in &self.points {
            if pz <= h && pz > 1e-6 {
                let vertex_slope = pz / px.max(MIN_RADIUS_MM);
                min_slope = min_slope.min(vertex_slope);
            }
        }

        min_slope.max(MIN_ANGLE_DEG.to_radians().tan())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn constant_angle_profile_matches_everywhere() {
        let profile = SlopeProfile::from_angle(45.0);
        assert!(approx_eq(profile.max_slope_at(1.0), 45.0, 1e-3));
        assert!(approx_eq(profile.max_slope_at(10.0), 45.0, 1e-3));
        assert!(approx_eq(profile.max_slope_at(50.0), 45.0, 1e-3));
    }

    #[test]
    fn empty_points_is_unconstrained() {
        let profile = SlopeProfile::new(vec![]);
        assert!(approx_eq(
            profile.max_slope_at(0.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
        assert!(approx_eq(
            profile.max_slope_at(100.0),
            UNCONSTRAINED_ANGLE_DEG,
            1e-9
        ));
    }

    #[test]
    fn xz_points_enforce_toolhead_clearance_cone() {
        // Tip at radius 1mm (z=0), heater block radius 10mm at z=10mm (45 deg)
        let profile = SlopeProfile::new(vec![(1.0, 0.0), (10.0, 10.0)]);

        // At z=10, slope is 10/10 = 1.0 (45 deg)
        assert!(approx_eq(profile.max_slope_at(10.0), 45.0, 1e-2));

        // At z=5, x(5) = 1 + 0.5 * 9 = 5.5mm. slope = 5/5.5 = 0.909 (42.27 deg)
        let angle_5 = (5.0f64 / 5.5).atan().to_degrees();
        assert!(approx_eq(profile.max_slope_at(5.0), angle_5, 1e-2));

        // At z=20 (above top point, x remains 10mm): slope is bounded by z=10 obstacle (45 deg)
        assert!(approx_eq(profile.max_slope_at(20.0), 45.0, 1e-2));
    }

    #[test]
    fn non_ascending_points_are_sorted_safely() {
        let profile = SlopeProfile::new(vec![(10.0, 10.0), (1.0, 0.0)]);
        assert!(approx_eq(profile.max_slope_at(10.0), 45.0, 1e-2));
    }

    #[test]
    fn negative_or_zero_values_are_clamped() {
        let profile = SlopeProfile::new(vec![(0.0, -5.0), (-10.0, 10.0)]);
        assert!(profile.points()[0].0 >= MIN_RADIUS_MM);
        assert!(profile.points()[0].1 >= 0.0);
    }
}
