//! Settings profiles: save/load named `Machine` + `SlicerConfig` presets to
//! JSON files so a user doesn't have to re-enter bed size, tool layout,
//! layer height, etc. each session (Phase 10, see ROADMAP.md).

use manifold_core::{machine::Machine, SlicerConfig};
use std::path::Path;

/// A saved preset bundling the machine definition and slicing settings.
///
/// Deliberately excludes `objects`/`selected` — profiles capture printer +
/// slicer setup, not a specific in-progress project.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Profile {
    pub machine: Machine,
    pub config: SlicerConfig,
}

impl Profile {
    /// Serializes `self` as pretty-printed JSON and writes it to `path`.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Reads and deserializes a profile previously written by [`Self::save`].
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let profile = serde_json::from_str(&json)?;
        Ok(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifold_core::bounds::BoundingVolume;
    use manifold_core::ids::ToolId;
    use manifold_core::tool::Tool;
    use manifold_core::transform::Transform;
    use manifold_core::SlicerConfig;

    fn sample_profile() -> Profile {
        Profile {
            machine: Machine {
                substrate_transform: Transform::identity(),
                build_volume: BoundingVolume::Aabb {
                    min: glam::DVec3::ZERO,
                    max: glam::DVec3::new(200.0, 200.0, 200.0),
                },
                tools: vec![Tool::new(ToolId(0), 0.4)],
                axis_count: 3,
            },
            config: SlicerConfig {
                layer_height: 0.2,
                nozzle_diameter: 0.4,
                object_ordering: Default::default(),
                wall_line_width: 0.4,
                shell_thickness: 0.4,
                wall_offset: 0.2,
            },
        }
    }

    #[test]
    fn save_then_load_roundtrips_profile() {
        let dir =
            std::env::temp_dir().join(format!("manifold-gui-profile-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("profile.json");

        let profile = sample_profile();
        profile.save(&path).expect("save profile");
        let loaded = Profile::load(&path).expect("load profile");

        assert_eq!(loaded, profile);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
