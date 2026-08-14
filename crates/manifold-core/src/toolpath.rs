//! Toolpath planning: layers -> ordered extrusion moves.

use crate::{ids::ToolId, object::Object, slicing::Layer, Result, SlicerConfig};
use glam::DVec3;

/// Classification of a single toolpath segment (the move from one point to
/// the next along a [`Path`]). Real wall/inner-wall/infill/support/bridge/
/// overhang *detection* is future work — today's [`plan`] tags every
/// segment [`MoveKind::WallOuter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MoveKind {
    #[default]
    WallOuter,
    WallInner,
    Infill,
    Bridge,
    Overhang,
    Travel,
}

/// Per-segment motion metadata for one `points[i] -> points[i+1]` edge of a
/// [`Path]` (including the closing edge of a closed loop).
#[derive(Debug, Clone, Copy, Default)]
pub struct Segment {
    pub kind: MoveKind,
    pub speed: f64,
    pub extrusion_rate: f64,
    pub support_fraction: f64,
    /// The order-field value (see `manifold_fidget::order`) whose
    /// isosurface produced this segment's source [`Layer`]. Stored
    /// per-segment (rather than per-`Path`/per-`Layer`) so it can vary
    /// once non-planar order fields exist.
    pub order: f64,
}

/// A single continuous toolpath (e.g. one perimeter or infill pass).
///
/// Per-segment metadata is carried in a sibling `segments` vector: for a
/// closed loop of N `points`, there are N segments — `segments[i]`
/// describes the move `points[i] -> points[(i + 1) % points.len()]`, with
/// the last segment being the closing edge back to `points[0]`. This keeps
/// `points`/`segments` as parallel `Vec`s (`segments.len() == points.len()`)
/// rather than pairing them in a single `Vec<(DVec3, Segment)>`, so callers
/// that only need geometry (e.g. bounding-box/preview code) can read
/// `points` without also touching `segments`.
#[derive(Debug, Clone, Default)]
pub struct Path {
    pub points: Vec<DVec3>,
    pub segments: Vec<Segment>,
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
            // Placeholder metadata: real wall/inner-wall/infill/support/
            // bridge/overhang classification and speed/extrusion-rate
            // planning is future work (see toolpath-metadata-phase12
            // subtask 03). Fixed sane defaults are used here rather than
            // new `SlicerConfig` fields, since these values aren't yet
            // meaningfully configurable.
            let segments = loop_points
                .iter()
                .map(|_| Segment {
                    kind: MoveKind::WallOuter,
                    speed: 60.0,
                    extrusion_rate: 1.0,
                    support_fraction: 0.0,
                    order: layer.order,
                })
                .collect();
            paths.push(Path {
                points: loop_points.clone(),
                segments,
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
                order: 0.0,
                loops: vec![loop_a.clone()],
            },
            Layer {
                index: 0,
                object: ObjectId(0),
                order: 0.0,
                loops: vec![loop_b.clone()],
            },
        ];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].tool, ToolId(2));
        assert_eq!(paths[0].points, loop_a);
        assert_eq!(paths[0].segments.len(), paths[0].points.len());
        assert!(paths[0]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallOuter));
        assert!(paths[0].segments.iter().all(|segment| segment.order == 0.0));
        assert!(paths[0]
            .segments
            .iter()
            .all(|segment| segment.support_fraction == 0.0));
        assert_eq!(paths[1].tool, ToolId(0));
        assert_eq!(paths[1].points, loop_b);
        assert_eq!(paths[1].segments.len(), paths[1].points.len());
        assert!(paths[1]
            .segments
            .iter()
            .all(|segment| segment.kind == MoveKind::WallOuter));
        assert!(paths[1].segments.iter().all(|segment| segment.order == 0.0));
        assert!(paths[1]
            .segments
            .iter()
            .all(|segment| segment.support_fraction == 0.0));
    }

    #[test]
    fn plan_stamps_segment_order_from_the_source_layer() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.75,
            loops: vec![vec![
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ]],
        }];

        let paths = plan(&layers, &objects, &SlicerConfig::default()).unwrap();

        assert_eq!(paths.len(), 1);
        assert!(paths[0]
            .segments
            .iter()
            .all(|segment| segment.order == 0.75));
    }

    #[test]
    fn plan_emits_no_paths_for_layer_with_no_loops() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let layers = vec![Layer {
            index: 0,
            object: ObjectId(0),
            order: 0.0,
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
            order: 0.0,
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
            order: 0.0,
            loops: Vec::new(),
        }];

        let err = plan(&layers, &objects, &SlicerConfig::default()).unwrap_err();
        assert!(matches!(err, crate::Error::InvalidMesh(_)));
    }
}
