//! Tool (nozzle/toolhead) definition.

use crate::{bounds::BoundingVolume, ids::ToolId, transform::Transform};
use glam::DVec3;

/// A single tool (nozzle/toolhead) mountable on a [`crate::machine::Machine`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tool {
    pub id: ToolId,
    /// Nozzle diameter in millimeters.
    pub nozzle_diameter: f64,
    /// Diameter (mm) of the nozzle tip's flat land, used by
    /// `toolpath::compensate_flat_nozzle`. Defaults to twice
    /// `nozzle_diameter` when `None`.
    #[serde(default)]
    pub nozzle_flat_diameter: Option<f64>,
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
    /// Target nozzle/hotend temperature for this tool in °C.
    /// Default: `240.0°C` when `None`.
    #[serde(default)]
    pub nozzle_temperature: Option<f64>,
}

impl Tool {
    /// Construct a tool with the given id and nozzle diameter, mounted at
    /// the origin with a zero-radius (point-like) collision envelope.
    pub fn new(id: ToolId, nozzle_diameter: f64) -> Self {
        Self {
            id,
            nozzle_diameter,
            nozzle_flat_diameter: None,
            mount: Transform::identity(),
            collision_envelope: BoundingVolume::Sphere {
                center: DVec3::ZERO,
                radius: 0.0,
            },
            extrusion_multiplier: 1.0,
            nozzle_temperature: None,
        }
    }

    /// Returns the target nozzle temperature in °C, defaulting to 240.0°C if unset.
    #[must_use]
    pub fn nozzle_temperature(&self) -> f64 {
        self.nozzle_temperature.unwrap_or(240.0)
    }

    /// Diameter (mm) of the nozzle tip's flat land, defaulting to twice
    /// [`Self::nozzle_diameter`] if unset.
    #[must_use]
    pub fn nozzle_flat_diameter(&self) -> f64 {
        self.nozzle_flat_diameter
            .unwrap_or(2.0 * self.nozzle_diameter)
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
