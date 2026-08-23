//! Machine (printer) definition: substrate, build volume, tools, and
//! kinematics capability.

use crate::{bounds::BoundingVolume, tool::Tool, transform::Transform};

/// Empty (unconstrained) default for [`Machine::eikonal_slope_profile`], used
/// by `#[serde(default = ...)]` so machine profiles saved before this field
/// existed still deserialize.
fn default_slope_profile() -> Vec<(f64, f64)> {
    Vec::new()
}

/// Describes the printer's physical envelope and kinematics.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Machine {
    /// Full position + orientation of the print substrate (build plate).
    ///
    /// Modeled as a transform (not a fixed Z=0 plane) so substrate
    /// reorientation can be introduced later without a schema change —
    /// see ROADMAP.md.
    pub substrate_transform: Transform,
    /// Build volume, in the substrate's local frame.
    pub build_volume: BoundingVolume,
    /// Tools available on this machine.
    pub tools: Vec<Tool>,
    /// Number of independently controllable motion axes.
    ///
    /// Fixed at 3 (X/Y/Z, non-tilting) for now; the field exists so the
    /// deferred multi-axis (tool-tilting / substrate-reorientation) work
    /// does not force a schema change later — see ROADMAP.md.
    pub axis_count: u8,
    /// Piecewise `(height_along_axis_mm, max_angle_deg)` breakpoints
    /// describing the maximum overhang/travel angle this machine can
    /// physically clear at a given height, converted via
    /// [`Machine::slope_profile`] into a
    /// `manifold_fidget::slope_profile::SlopeProfile`. Lives on `Machine`
    /// (not `SlicerConfig`) because it describes a physical property of the
    /// printer/gantry, not a per-slice setting — and it needs to be
    /// available for travel-collision routing (`toolpath`) even when the
    /// configured `order_field` is not `Eikonal`. Serde-friendly (a plain
    /// `Vec` of tuples) so it round-trips through saved JSON profiles (see
    /// `manifold-gui/src/profile.rs`).
    ///
    /// `#[serde(default = "default_slope_profile")]` so a saved machine
    /// profile from before this field existed still deserializes: a missing
    /// field falls back to `default_slope_profile()`, which is empty and
    /// therefore unconstrained.
    #[serde(default = "default_slope_profile")]
    pub eikonal_slope_profile: Vec<(f64, f64)>,
    /// Whether to use the stepper dynamic motor model for velocity and acceleration planning.
    #[serde(default)]
    pub use_stepper_dynamics: bool,
    /// Maximum available acceleration at zero velocity (a0, mm/s²). Default 20,000 mm/s².
    #[serde(default)]
    pub zero_speed_acceleration: Option<f64>,
    /// Theoretical maximum velocity where available motor torque/acceleration drops to zero (v_max, mm/s). Default 1,000 mm/s.
    #[serde(default)]
    pub max_available_speed: Option<f64>,
    /// Hard upper bound on acceleration (mm/s²). Defaults to 50% of zero_speed_acceleration (10,000 mm/s²).
    #[serde(default)]
    pub acceleration_limit: Option<f64>,
    /// Hard upper bound on velocity (mm/s). Defaults to 75% of max_available_speed (750 mm/s).
    #[serde(default)]
    pub speed_limit: Option<f64>,
}

impl Machine {
    /// Construct a 3-axis machine with the given build volume and tools.
    pub fn new(build_volume: BoundingVolume, tools: Vec<Tool>) -> Self {
        Self {
            substrate_transform: Transform::identity(),
            build_volume,
            tools,
            axis_count: 3,
            eikonal_slope_profile: Vec::new(),
            use_stepper_dynamics: false,
            zero_speed_acceleration: None,
            max_available_speed: None,
            acceleration_limit: None,
            speed_limit: None,
        }
    }

    /// Acceleration at zero velocity (a0, mm/s²). Default: 20,000 mm/s².
    pub fn zero_speed_acceleration(&self) -> f64 {
        self.zero_speed_acceleration.unwrap_or(20000.0)
    }

    /// Theoretical maximum velocity where motor torque reaches zero (v_max, mm/s). Default: 1,000 mm/s.
    pub fn max_available_speed(&self) -> f64 {
        self.max_available_speed.unwrap_or(1000.0)
    }

    /// Configured upper bound on acceleration (mm/s²). Defaults to 50% of `zero_speed_acceleration`.
    pub fn acceleration_limit(&self) -> f64 {
        self.acceleration_limit
            .unwrap_or_else(|| self.zero_speed_acceleration() * 0.5)
    }

    /// Configured upper bound on velocity (mm/s). Defaults to 75% of `max_available_speed`.
    pub fn speed_limit(&self) -> f64 {
        self.speed_limit
            .unwrap_or_else(|| self.max_available_speed() * 0.75)
    }

    /// Converts the serde-friendly `eikonal_slope_profile` breakpoints into
    /// a real `manifold_fidget::slope_profile::SlopeProfile`.
    ///
    /// Delegates entirely to `SlopeProfile::new`, which already sorts
    /// non-ascending breakpoints and clamps degenerate angles rather than
    /// panicking, so this adds no new panic path.
    #[must_use]
    pub fn slope_profile(&self) -> manifold_fidget::slope_profile::SlopeProfile {
        manifold_fidget::slope_profile::SlopeProfile::new(self.eikonal_slope_profile.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn new_machine_defaults_to_three_axes() {
        let machine = Machine::new(
            BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(200.0, 200.0, 200.0),
            },
            Vec::new(),
        );
        assert_eq!(machine.axis_count, 3);
    }

    #[test]
    fn machine_stepper_dynamics_defaults_and_overrides() {
        let mut machine = Machine::new(
            BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(200.0, 200.0, 200.0),
            },
            Vec::new(),
        );

        // Check defaults
        assert_eq!(machine.zero_speed_acceleration(), 20000.0);
        assert_eq!(machine.max_available_speed(), 1000.0);
        assert_eq!(machine.acceleration_limit(), 10000.0);
        assert_eq!(machine.speed_limit(), 750.0);

        // Check overrides
        machine.zero_speed_acceleration = Some(30000.0);
        machine.max_available_speed = Some(1200.0);
        // acceleration_limit defaults to 50% of updated zero_speed_acceleration
        assert_eq!(machine.acceleration_limit(), 15000.0);
        // speed_limit defaults to 75% of updated max_available_speed
        assert_eq!(machine.speed_limit(), 900.0);

        // Explicit limit overrides
        machine.acceleration_limit = Some(12000.0);
        machine.speed_limit = Some(800.0);
        assert_eq!(machine.acceleration_limit(), 12000.0);
        assert_eq!(machine.speed_limit(), 800.0);
    }
}
