//! Workspace: the top-level input to the slicing pipeline.

use crate::{machine::Machine, object::Object, SlicerConfig};

/// The complete input to [`crate::slice_to_gcode`]: every object to
/// print, the machine printing them, and shared slicing configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Workspace {
    pub objects: Vec<Object>,
    pub machine: Machine,
    pub config: SlicerConfig,
}

impl Workspace {
    /// Construct a workspace from its parts.
    pub fn new(objects: Vec<Object>, machine: Machine, config: SlicerConfig) -> Self {
        Self {
            objects,
            machine,
            config,
        }
    }
}
