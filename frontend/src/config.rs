use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Shape of the 3D printer build plate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BedShape {
    Rectangular,
    Circular,
}

/// Configuration for the printer build volume and bed dimensions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintBedConfig {
    pub shape: BedShape,
    /// Bed width along X axis (mm), or diameter if Circular.
    pub width_x: f32,
    /// Bed depth along Y axis (mm).
    pub depth_y: f32,
    /// Maximum printable Z height (mm).
    pub height_z: f32,
    /// Grid spacing for the viewport visualization (mm).
    pub grid_spacing: f32,
    /// Show the 3D build volume wireframe box.
    pub show_volume_box: bool,
}

impl Default for PrintBedConfig {
    fn default() -> Self {
        Self {
            shape: BedShape::Rectangular,
            width_x: 256.0,
            depth_y: 256.0,
            height_z: 256.0,
            grid_spacing: 20.0,
            show_volume_box: true,
        }
    }
}

/// Parameters controlling automatic part arrangement on the build plate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrangementConfig {
    /// Minimum XY clearance between adjacent objects (mm).
    pub part_clearance_mm: f32,
    /// Minimum margin from the outer edge of the build plate (mm).
    pub bed_edge_margin_mm: f32,
    /// Angular step size (radians) when searching along the Archimedean spiral.
    pub spiral_angle_step_rad: f32,
    /// Radial pitch per revolution (mm) of the Archimedean spiral.
    pub spiral_radial_pitch_mm: f32,
}

impl Default for ArrangementConfig {
    fn default() -> Self {
        Self {
            part_clearance_mm: 4.0,
            bed_edge_margin_mm: 5.0,
            spiral_angle_step_rad: 0.20,
            spiral_radial_pitch_mm: 8.0,
        }
    }
}

/// Provision for non-planar slicing process parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlicingProcessConfig {
    pub base_layer_height_mm: f32,
    pub min_layer_height_mm: f32,
    pub max_layer_height_mm: f32,
    pub max_nonplanar_slope_deg: f32,
    pub perimeter_count: u32,
    pub infill_density_percent: f32,
}

impl Default for SlicingProcessConfig {
    fn default() -> Self {
        Self {
            base_layer_height_mm: 0.20,
            min_layer_height_mm: 0.08,
            max_layer_height_mm: 0.32,
            max_nonplanar_slope_deg: 35.0,
            perimeter_count: 3,
            infill_density_percent: 20.0,
        }
    }
}

/// Provision for OpenVDB volumetric Eikonal field parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EikonalFieldConfig {
    pub voxel_size_mm: f32,
    pub narrow_band_width_voxels: u32,
    pub curvature_regularization: f32,
}

impl Default for EikonalFieldConfig {
    fn default() -> Self {
        Self {
            voxel_size_mm: 0.50,
            narrow_band_width_voxels: 6,
            curvature_regularization: 0.05,
        }
    }
}

/// Provision for toolhead collision cone and multi-axis machine kinematics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MachineKinematicsConfig {
    pub nozzle_diameter_mm: f32,
    pub nozzle_clearance_cone_angle_deg: f32,
    pub nozzle_clearance_height_mm: f32,
    pub five_axis_enabled: bool,
}

impl Default for MachineKinematicsConfig {
    fn default() -> Self {
        Self {
            nozzle_diameter_mm: 0.40,
            nozzle_clearance_cone_angle_deg: 45.0,
            nozzle_clearance_height_mm: 12.0,
            five_axis_enabled: false,
        }
    }
}

/// Top-level extensible slicer configuration stored as a Bevy resource and
/// persisted inside `.3mf` project files under `/Metadata/manifold2_config.toml`.
#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct SlicerConfig {
    pub bed: PrintBedConfig,
    pub arrangement: ArrangementConfig,
    pub slicing: SlicingProcessConfig,
    pub eikonal: EikonalFieldConfig,
    pub machine: MachineKinematicsConfig,
}

impl SlicerConfig {
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
}
