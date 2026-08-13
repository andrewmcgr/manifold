//! Manifold slicing engine.
//!
//! `manifold-core` turns a mesh into non-planar toolpaths and emits Gcode.
//! It has no UI or CLI dependencies so it can run headless (e.g. embedded
//! in a service) or be driven by the `manifold-cli` front-end.

pub mod error;
pub mod mesh;
pub mod slicing;
pub mod toolpath;
pub mod gcode;

pub use error::{Error, Result};

/// Slicer configuration shared across the pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

/// Run the full pipeline: load mesh -> slice -> plan toolpaths -> emit Gcode.
pub fn slice_to_gcode(mesh: &mesh::Mesh, config: &SlicerConfig) -> Result<String> {
    let layers = slicing::slice_mesh(mesh, config)?;
    let paths = toolpath::plan(&layers, config)?;
    Ok(gcode::emit(&paths, config))
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
}
