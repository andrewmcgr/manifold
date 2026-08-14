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
/// Minimal implementation: emits one [`Path`] per contour loop in each
/// [`Layer`] (a layer with no loops contributes no paths). Real path
/// planning beyond this (multiple perimeters/shells, infill, travel-move
/// ordering/optimization, non-planar toolpath deformation) is future work.
///
/// # Errors
///
/// Returns [`crate::Error::InvalidMesh`] if a layer references an object
/// id not present in `objects`.
pub fn plan(layers: &[Layer], objects: &[Object], _config: &SlicerConfig) -> Result<Vec<Path>> {
    let mut paths = Vec::new();
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
        for loop_points in &layer.loops {
            paths.push(Path {
                points: loop_points.clone(),
                extruding: true,
                tool: object.tool,
            });
        }
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
        let loop_a = vec![
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let loop_b = vec![
            DVec3::ZERO,
            DVec3::new(2.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
        ];
        let layers = vec![
            Layer {
                index: 0,
                object: ObjectId(1),
                loops: vec![loop_a.clone()],
            },
            Layer {
                index: 0,
                object: ObjectId(0),
                loops: vec![loop_b.clone()],
            },
        ];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].tool, ToolId(2));
        assert_eq!(paths[0].points, loop_a);
        assert!(paths[0].extruding);
        assert_eq!(paths[1].tool, ToolId(0));
        assert_eq!(paths[1].points, loop_b);
        assert!(paths[1].extruding);
    }

    #[test]
    fn plan_emits_no_paths_for_layer_with_no_loops() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            loops: Vec::new(),
        }];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert!(paths.is_empty());
    }

    #[test]
    fn plan_emits_one_path_per_loop_in_a_layer() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            loops: vec![
                vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
                vec![DVec3::new(2.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0)],
            ],
        }];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn plan_rejects_layer_with_unknown_object() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(99),
            loops: Vec::new(),
        }];

        let err = plan(&layers, &objects, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }
}
