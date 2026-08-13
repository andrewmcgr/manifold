//! Manifold slicing engine.
//!
//! `manifold-core` turns a mesh into non-planar toolpaths and emits Gcode.
//! It has no UI or CLI dependencies so it can run headless (e.g. embedded
//! in a service) or be driven by the `manifold-cli` front-end.

pub mod bounds;
pub mod error;
pub mod gcode;
pub mod ids;
pub mod machine;
pub mod material;
pub mod mesh;
pub mod object;
pub mod slicing;
pub mod stl;
pub mod threemf;
pub mod tool;
pub mod toolpath;
pub mod transform;
pub mod workspace;

pub use error::{Error, Result};
pub use workspace::Workspace;

/// Slicer configuration shared across the pipeline.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlicerConfig {
    pub layer_height: f64,
    pub nozzle_diameter: f64,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            layer_height: 0.2,
            nozzle_diameter: 0.4,
        }
    }
}

/// Run the full pipeline: workspace -> slice -> plan toolpaths -> emit Gcode.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the slicing/toolpath stages produce.
///
/// TODO(roadmap): Phase 2 (see ROADMAP.md) — this only slices the first
/// object in the workspace today; true multi-object/multi-tool slicing
/// (per-object transforms applied in world space, tool-change-aware
/// toolpath planning, tool-change Gcode) lands there.
pub fn slice_to_gcode(workspace: &Workspace) -> Result<String> {
    let object = workspace
        .objects
        .first()
        .ok_or_else(|| Error::InvalidMesh("workspace has no objects".to_string()))?;

    let layers = slicing::slice_mesh(&object.mesh, &workspace.config)?;
    let paths = toolpath::plan(&layers, &workspace.config)?;
    Ok(gcode::emit(&paths, &workspace.config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let cfg = SlicerConfig::default();
        assert!(cfg.layer_height > 0.0);
        assert!(cfg.nozzle_diameter > 0.0);
    }

    #[test]
    fn slice_to_gcode_rejects_empty_workspace() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let workspace = Workspace::new(Vec::new(), machine, SlicerConfig::default());

        let err = slice_to_gcode(&workspace).unwrap_err();
        assert!(matches!(err, Error::InvalidMesh(_)));
    }

    #[test]
    fn slice_to_gcode_slices_first_object() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let object = crate::object::Object::new(
            crate::ids::ObjectId(0),
            mesh::Mesh::default(),
            crate::ids::ToolId(0),
        );
        let workspace = Workspace::new(vec![object], machine, SlicerConfig::default());

        assert!(slice_to_gcode(&workspace).is_ok());
    }
}
