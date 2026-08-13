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
pub fn plan(_layers: &[Layer], _config: &SlicerConfig) -> Result<Vec<Path>> {
    Ok(Vec::new())
}
