//! Object print-ordering strategies.
//!
//! Multi-object printing needs to decide what order (and, eventually,
//! interleaving) objects are printed in. Collision-aware ordering is
//! deferred (see ROADMAP.md "Deferred / future work"), so for now this
//! module exists to make that decision **pluggable** rather than
//! hardcoded: `SlicerConfig::object_ordering` selects an
//! [`ObjectOrderingKind`], resolved to a concrete [`ObjectOrderStrategy`]
//! via [`strategy_for`]. Adding a new algorithm later means adding an enum
//! variant, a struct implementing the trait, and one match arm — no
//! changes to the slicing/toolpath/gcode pipeline itself.

use crate::{ids::ObjectId, object::Object, Result};

/// Determines the order in which objects in a workspace are printed.
///
/// Implementations may inspect object geometry, bounds, or tool
/// assignment, but must not mutate the objects they're given — ordering
/// is a read-only decision made before slicing.
pub trait ObjectOrderStrategy {
    /// Return the order (by [`ObjectId`]) in which `objects` should be
    /// sliced/printed.
    fn order(&self, objects: &[Object]) -> Result<Vec<ObjectId>>;
}

/// Print objects one fully at a time, in workspace declaration order.
///
/// The only strategy implemented today (see ROADMAP.md open decision #2):
/// naive-simultaneous and collision-aware interleaving are future work.
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialOrder;

impl ObjectOrderStrategy for SequentialOrder {
    fn order(&self, objects: &[Object]) -> Result<Vec<ObjectId>> {
        Ok(objects.iter().map(|object| object.id).collect())
    }
}

/// Selects which [`ObjectOrderStrategy`] `slice_to_gcode` uses. Persisted
/// on [`crate::SlicerConfig`] like any other slicing parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ObjectOrderingKind {
    /// Whole-object-at-a-time, in declaration order. See [`SequentialOrder`].
    #[default]
    Sequential,
}

/// Resolve a config-level [`ObjectOrderingKind`] to a concrete strategy.
pub fn strategy_for(kind: ObjectOrderingKind) -> Box<dyn ObjectOrderStrategy> {
    match kind {
        ObjectOrderingKind::Sequential => Box::new(SequentialOrder),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ids::ToolId, mesh::Mesh};

    #[test]
    fn sequential_order_preserves_declaration_order() {
        let objects = vec![
            Object::new(ObjectId(2), Mesh::default(), ToolId(0)),
            Object::new(ObjectId(0), Mesh::default(), ToolId(0)),
            Object::new(ObjectId(1), Mesh::default(), ToolId(0)),
        ];

        let order = SequentialOrder.order(&objects).unwrap();

        assert_eq!(order, vec![ObjectId(2), ObjectId(0), ObjectId(1)]);
    }

    #[test]
    fn sequential_order_is_empty_for_empty_workspace() {
        let order = SequentialOrder.order(&[]).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn strategy_for_sequential_kind_produces_declaration_order() {
        let objects = vec![Object::new(ObjectId(0), Mesh::default(), ToolId(0))];
        let strategy = strategy_for(ObjectOrderingKind::Sequential);

        assert_eq!(strategy.order(&objects).unwrap(), vec![ObjectId(0)]);
    }
}
