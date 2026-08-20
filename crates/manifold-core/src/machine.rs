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
        }
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
}
