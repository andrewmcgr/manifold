//! Object: a mesh instance placed in a workspace.

use crate::{
    ids::{ObjectId, ToolId},
    mesh::Mesh,
    transform::Transform,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_object_is_placed_at_identity_transform() {
        let object = Object::new(ObjectId(1), Mesh::default(), ToolId(1));
        assert_eq!(object.transform, Transform::identity());
    }
}
