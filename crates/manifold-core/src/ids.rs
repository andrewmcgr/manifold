//! Strongly-typed identifiers for domain entities.
//!
//! Wrapping plain `u32`s in newtypes prevents accidentally mixing a
//! `ToolId` with an `ObjectId` (or a raw index) at call sites.

use std::fmt;

/// Identifies a [`crate::tool::Tool`] within a [`crate::machine::Machine`].
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ToolId(pub u32);

/// Identifies a [`crate::material::Material`].
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct MaterialId(pub u32);

/// Identifies an [`crate::object::Object`] within a [`crate::workspace::Workspace`].
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct ObjectId(pub u32);

impl ToolId {
    /// The raw underlying value.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl MaterialId {
    /// The raw underlying value.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl ObjectId {
    /// The raw underlying value.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ToolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for MaterialId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for ToolId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<u32> for MaterialId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<u32> for ObjectId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_roundtrips_through_raw_value() {
        let id = ToolId::from(3);
        assert_eq!(id.get(), 3);
        assert_eq!(ToolId(3), id);
    }

    #[test]
    fn id_displays_as_raw_value() {
        assert_eq!(ObjectId(7).to_string(), "7");
    }
}
