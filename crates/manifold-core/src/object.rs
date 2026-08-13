//! Object: a mesh instance placed in a workspace.

use crate::{
    bounds::BoundingVolume,
    ids::{ObjectId, ToolId},
    mesh::Mesh,
    transform::Transform,
};
use glam::DVec3;

/// A single mesh instance placed within a [`crate::workspace::Workspace`],
/// assigned to a tool.
#[derive(Debug, Clone, PartialEq)]
pub struct Object {
    pub id: ObjectId,
    pub mesh: Mesh,
    /// Full position/rotation/scale placement of this object in world
    /// space.
    pub transform: Transform,
    /// The tool assigned to print this object.
    pub tool: ToolId,
}

impl Object {
    /// Construct an object at the identity transform.
    pub fn new(id: ObjectId, mesh: Mesh, tool: ToolId) -> Self {
        Self {
            id,
            mesh,
            transform: Transform::identity(),
            tool,
        }
    }
}

/// Re-center a freshly-loaded group of objects on a machine's bed: the
/// combined world-space bounding box of every object's mesh (after its
/// existing `transform`) is translated so it sits XY-centered on
/// `build_volume` and resting on its floor (minimum Z) — without
/// disturbing the objects' relative arrangement (e.g. a 3MF assembly's
/// build-item transforms). Objects with an empty mesh are ignored; a no-op
/// if every object is empty or `objects` is empty.
pub fn center_on_bed(objects: &mut [Object], build_volume: &BoundingVolume) {
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    let mut any = false;

    for object in objects.iter() {
        let Some((local_min, local_max)) = object.mesh.bounding_box() else {
            continue;
        };
        any = true;
        for corner in bounding_box_corners(local_min, local_max) {
            let world = object.transform.transform_point(corner);
            min = min.min(world);
            max = max.max(world);
        }
    }

    if !any {
        return;
    }

    let (bed_min, bed_max) = build_volume.bounding_box();
    let bed_center_xy = (bed_min + bed_max) * 0.5;
    let combined_center_xy = (min + max) * 0.5;
    let offset = DVec3::new(
        bed_center_xy.x - combined_center_xy.x,
        bed_center_xy.y - combined_center_xy.y,
        bed_min.z - min.z,
    );

    for object in objects.iter_mut() {
        object.transform = object.transform.then_translate(offset);
    }
}

fn bounding_box_corners(min: DVec3, max: DVec3) -> [DVec3; 8] {
    [
        DVec3::new(min.x, min.y, min.z),
        DVec3::new(max.x, min.y, min.z),
        DVec3::new(min.x, max.y, min.z),
        DVec3::new(max.x, max.y, min.z),
        DVec3::new(min.x, min.y, max.z),
        DVec3::new(max.x, min.y, max.z),
        DVec3::new(min.x, max.y, max.z),
        DVec3::new(max.x, max.y, max.z),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_object_is_placed_at_identity_transform() {
        let object = Object::new(ObjectId(1), Mesh::default(), ToolId(1));
        assert_eq!(object.transform, Transform::identity());
    }

    #[test]
    fn center_on_bed_ignores_empty_object_list() {
        let mut objects: Vec<Object> = Vec::new();
        center_on_bed(
            &mut objects,
            &BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(200.0, 200.0, 200.0),
            },
        );
        assert!(objects.is_empty());
    }

    #[test]
    fn center_on_bed_xy_centers_and_rests_on_floor() {
        let mesh = Mesh::new(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(10.0, 0.0, 0.0),
                DVec3::new(0.0, 10.0, 0.0),
                DVec3::new(0.0, 0.0, 20.0),
            ],
            vec![0, 1, 2, 0, 1, 3],
        );
        let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
        let build_volume = BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        };

        center_on_bed(&mut objects, &build_volume);

        let (world_min, world_max) = objects[0]
            .mesh
            .bounding_box()
            .map(|(min, max)| {
                (
                    objects[0].transform.transform_point(min),
                    objects[0].transform.transform_point(max),
                )
            })
            .unwrap();
        // XY footprint centered on the 200x200 bed: (0,0)-(10,10) local ->
        // (95,95)-(105,105) world.
        assert_eq!(world_min, DVec3::new(95.0, 95.0, 0.0));
        assert_eq!(world_max, DVec3::new(105.0, 105.0, 20.0));
    }

    #[test]
    fn center_on_bed_preserves_relative_arrangement() {
        let mesh_a = Mesh::new(vec![DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0)], vec![0, 0, 1]);
        let mesh_b = mesh_a.clone();
        let object_a = Object::new(ObjectId(0), mesh_a, ToolId(0));
        let mut object_b = Object::new(ObjectId(1), mesh_b, ToolId(0));
        object_b.transform = Transform::from_translation(DVec3::new(5.0, 0.0, 0.0));
        let mut objects = vec![object_a.clone(), object_b.clone()];

        center_on_bed(
            &mut objects,
            &BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(200.0, 200.0, 200.0),
            },
        );

        let a_origin = objects[0].transform.transform_point(DVec3::ZERO);
        let b_origin = objects[1].transform.transform_point(DVec3::ZERO);
        // The 5mm separation between the two objects' origins must survive
        // centering — only the group as a whole moves.
        assert_eq!(b_origin - a_origin, DVec3::new(5.0, 0.0, 0.0));
        let _ = (object_a, object_b);
    }
}
