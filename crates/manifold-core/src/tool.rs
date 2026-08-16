//! Tool (nozzle/toolhead) definition.

use crate::{bounds::BoundingVolume, ids::ToolId, transform::Transform};
use glam::DVec3;

/// A single tool (nozzle/toolhead) mountable on a [`crate::machine::Machine`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tool {
    pub id: ToolId,
    /// Nozzle diameter in millimeters.
    pub nozzle_diameter: f64,
    /// Full position + orientation of this tool's mount on the machine.
    ///
    /// Modeled as a transform (not a Z-offset scalar) so tilting-tool
    /// kinematics can be introduced later without a schema change — see
    /// ROADMAP.md.
    pub mount: Transform,
    /// Bounding volume around the nozzle/toolhead, in the tool's local
    /// frame (relative to `mount`). Unused until the deferred
    /// collision-avoidance work lands — see ROADMAP.md.
    pub collision_envelope: BoundingVolume,
    /// Per-tool (per-filament) flow-rate scale applied to every extrusion
    /// this tool prints, on top of the geometric bead-volume calculation
    /// (see `crate::extrusion`) and any per-segment `Segment::extrusion_rate`.
    /// Lets a material that behaves differently (e.g. shrinks/expands more
    /// than nominal when extruded) be tuned without touching the shared
    /// bead-area math. Defaults to `1.0` (no adjustment).
    pub extrusion_multiplier: f64,
}

impl Tool {
    /// Construct a tool with the given id and nozzle diameter, mounted at
    /// the origin with a zero-radius (point-like) collision envelope.
    pub fn new(id: ToolId, nozzle_diameter: f64) -> Self {
        Self {
            id,
            nozzle_diameter,
            mount: Transform::identity(),
            collision_envelope: BoundingVolume::Sphere {
                center: DVec3::ZERO,
                radius: 0.0,
            },
            extrusion_multiplier: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tool_is_mounted_at_origin() {
        let tool = Tool::new(ToolId(1), 0.4);
        assert_eq!(tool.mount, Transform::identity());
    }
}
