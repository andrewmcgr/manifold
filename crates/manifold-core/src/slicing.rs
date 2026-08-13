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
///
/// TODO(roadmap): Phase 2 (see ROADMAP.md) — once Phase 0 lands, this needs
/// to operate per-`Object` (apply its transform, slice in world space) and
/// tag output layers by object id so same-Z layers across objects can be
/// merged. Collision-aware multi-object ordering is explicitly deferred
/// (see ROADMAP.md "Deferred / future work").
pub fn slice_mesh(_mesh: &Mesh, _config: &SlicerConfig) -> Result<Vec<Layer>> {
    Ok(Vec::new())
}
