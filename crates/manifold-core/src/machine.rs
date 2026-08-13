//! Machine (printer) definition: substrate, build volume, tools, and
//! kinematics capability.

use crate::{bounds::BoundingVolume, tool::Tool, transform::Transform};

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
}

impl Machine {
    /// Construct a 3-axis machine with the given build volume and tools.
    pub fn new(build_volume: BoundingVolume, tools: Vec<Tool>) -> Self {
        Self {
            substrate_transform: Transform::identity(),
            build_volume,
            tools,
            axis_count: 3,
        }
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
