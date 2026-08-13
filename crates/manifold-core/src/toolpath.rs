//! Toolpath planning: layers -> ordered extrusion moves.

use crate::{slicing::Layer, Result, SlicerConfig};
use glam::DVec3;

/// A single continuous toolpath (e.g. one perimeter or infill pass).
#[derive(Debug, Clone, Default)]
pub struct Path {
    pub points: Vec<DVec3>,
    pub extruding: bool,
}

/// Plan toolpaths for a set of layers.
///
/// Placeholder implementation: real path planning (perimeters, infill,
/// non-planar toolpath deformation) lives here.
///
/// TODO(roadmap): Phase 2 (see ROADMAP.md) — add tool-change-aware
/// planning (per-tool paths, tool switch points) once Phase 0's `Tool`
/// model lands. Toolhead-vs-neighbor collision avoidance for simultaneous
/// multi-object printing is explicitly deferred (see ROADMAP.md); it will
/// use `Tool.collision_envelope` once tackled.
pub fn plan(_layers: &[Layer], _config: &SlicerConfig) -> Result<Vec<Path>> {
    Ok(Vec::new())
}
