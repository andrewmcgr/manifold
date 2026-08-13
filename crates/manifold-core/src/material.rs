//! Material definition.

use crate::ids::MaterialId;

/// A printable material.
///
/// Extend with more fields (extrusion temperature, etc.) here as later
/// phases need them — not as loose function arguments, per
/// `CODE_STYLE.md`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Material {
    pub id: MaterialId,
    pub name: String,
}

impl Material {
    /// Construct a material with the given id and name.
    pub fn new(id: MaterialId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_material_stores_given_name() {
        let material = Material::new(MaterialId(1), "PLA");
        assert_eq!(material.name, "PLA");
    }
}
