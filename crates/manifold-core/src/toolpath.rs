//! Toolpath planning: layers -> ordered extrusion moves.

use crate::{ids::ToolId, object::Object, slicing::Layer, Result, SlicerConfig};
use glam::DVec3;

/// A single continuous toolpath (e.g. one perimeter or infill pass).
#[derive(Debug, Clone, Default)]
pub struct Path {
    pub points: Vec<DVec3>,
    pub extruding: bool,
    /// The tool this path is printed with — looked up from the layer's
    /// source object's `Tool` assignment. Lets [`crate::gcode::emit`]
    /// insert tool-change Gcode between paths assigned to different
    /// tools.
    pub tool: ToolId,
}

/// Plan toolpaths for a set of layers, tagging each planned path with the
/// tool assigned to its source object.
///
/// Placeholder implementation: real path planning (perimeters, infill,
/// non-planar toolpath deformation) lives here.
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if a layer references an object
/// id not present in `objects`.
pub fn plan(layers: &[Layer], objects: &[Object], _config: &SlicerConfig) -> Result<Vec<Path>> {
    let mut paths = Vec::with_capacity(layers.len());
    for layer in layers {
        let object = objects
            .iter()
            .find(|object| object.id == layer.object)
            .ok_or_else(|| {
                crate::Error::InvalidMesh(format!(
                    "layer references unknown object {}",
                    layer.object
                ))
            })?;
        paths.push(Path {
            points: Vec::new(),
            extruding: false,
            tool: object.tool,
        });
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ids::ObjectId, mesh::Mesh};

    #[test]
    fn plan_tags_paths_with_objects_assigned_tool() {
        let objects = vec![
            Object::new(ObjectId(0), Mesh::default(), ToolId(0)),
            Object::new(ObjectId(1), Mesh::default(), ToolId(2)),
        ];
        let layers = vec![
            Layer {
                index: 0,
                object: ObjectId(1),
            },
            Layer {
                index: 0,
                object: ObjectId(0),
            },
        ];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert_eq!(paths[0].tool, ToolId(2));
        assert_eq!(paths[1].tool, ToolId(0));
    }

    #[test]
    fn plan_rejects_layer_with_unknown_object() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(99),
        }];

        let err = plan(&layers, &objects, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }
}
