//! Non-planar slicing: mesh -> ordered layers of cross-section curves.

use crate::{mesh::Mesh, Result, SlicerConfig};

/// A single (possibly non-planar) slice layer.
#[derive(Debug, Clone, Default)]
pub struct Layer {
    pub index: usize,
}

/// Slice a mesh into layers according to `config`.
///
/// Placeholder implementation: real non-planar slicing logic lives here.
pub fn slice_mesh(_mesh: &Mesh, _config: &SlicerConfig) -> Result<Vec<Layer>> {
    Ok(Vec::new())
}
