use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Orthonormal right-handed basis defining the nozzle's local coordinate frame in bed coordinates:
/// - `nozzle_axis` ($\hat{\mathbf{z}}_{\text{nozzle}}$): points opposite to the nozzle discharge direction
///   (i.e. from the nozzle tip up along the nozzle body).
/// - `travel_axis` ($\hat{\mathbf{x}}_{\text{nozzle}}$): points in the direction of toolhead travel,
///   orthogonalized against `nozzle_axis`.
/// - `lateral_axis` ($\hat{\mathbf{y}}_{\text{nozzle}}$): completes the right-handed basis via
///   `nozzle_axis.cross(travel_axis)` so that `travel_axis × lateral_axis = nozzle_axis`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NozzleFrame {
    pub nozzle_axis: Vec3,
    pub travel_axis: Vec3,
    pub lateral_axis: Vec3,
}

impl Default for NozzleFrame {
    fn default() -> Self {
        Self::from_nozzle_and_travel(Vec3::Z, Vec3::X)
    }
}

impl NozzleFrame {
    /// Constructs a right-handed orthonormal basis from a desired `nozzle_axis` (pointing
    /// opposite to the nozzle discharge direction) and a `travel_direction` in bed coordinates.
    pub fn from_nozzle_and_travel(nozzle_axis: Vec3, travel_direction: Vec3) -> Self {
        let z_nozzle = nozzle_axis.try_normalize().unwrap_or(Vec3::Z);

        // Project travel_direction onto the plane orthogonal to z_nozzle
        let projected_travel = travel_direction - z_nozzle * travel_direction.dot(z_nozzle);
        let x_travel = if let Some(unit_t) = projected_travel.try_normalize() {
            unit_t
        } else {
            // Deterministic fallback when travel_direction is zero or collinear with z_nozzle
            let ref_axis = if z_nozzle.x.abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            (ref_axis - z_nozzle * ref_axis.dot(z_nozzle)).normalize()
        };

        // Right-hand rule: X_travel × Y_lateral = Z_nozzle  =>  Y_lateral = Z_nozzle × X_travel
        let y_lateral = z_nozzle.cross(x_travel).normalize();

        Self {
            nozzle_axis: z_nozzle,
            travel_axis: x_travel,
            lateral_axis: y_lateral,
        }
    }

    /// Transforms a bed-space point into the nozzle's local coordinate frame
    /// `(x_along_travel, y_lateral, z_above_nozzle_tip)` centered at `nozzle_tip_bed`.
    pub fn bed_to_nozzle_local(&self, nozzle_tip_bed: Vec3, point_bed: Vec3) -> Vec3 {
        let rel = point_bed - nozzle_tip_bed;
        Vec3::new(
            rel.dot(self.travel_axis),
            rel.dot(self.lateral_axis),
            rel.dot(self.nozzle_axis),
        )
    }

    /// Transforms a point from the nozzle's local frame `(x_travel, y_lateral, z_nozzle)`
    /// back into bed coordinates given the nozzle tip position `nozzle_tip_bed`.
    pub fn nozzle_local_to_bed(&self, nozzle_tip_bed: Vec3, local_pt: Vec3) -> Vec3 {
        nozzle_tip_bed
            + self.travel_axis * local_pt.x
            + self.lateral_axis * local_pt.y
            + self.nozzle_axis * local_pt.z
    }
}

/// A single `(X, Z)` vertex in a tool's revolved clearance profile, expressed in the
/// nozzle's local frame:
/// - `x_radius_mm`: radial distance from the nozzle centerline ($\ge 0$ mm).
/// - `z_height_mm`: axial distance above the nozzle tip along `nozzle_axis` ($\ge 0$ mm).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClearancePointXZ {
    pub x_radius_mm: f32,
    pub z_height_mm: f32,
}

impl ClearancePointXZ {
    pub const fn new(x_radius_mm: f32, z_height_mm: f32) -> Self {
        Self {
            x_radius_mm,
            z_height_mm,
        }
    }
}

/// Properties, rotational axis capabilities, and local-frame clearance envelope for one machine extrusion tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolConfig {
    pub id: u32,
    pub name: String,
    /// Orifice diameter of the nozzle in millimeters (mm).
    pub nozzle_diameter_mm: f32,
    /// Target extrusion temperature in degrees Celsius (°C).
    pub temperature_celsius: f32,
    /// Whether this tool has rotational axes capable of orienting the nozzle away from the bed normal.
    pub has_rotational_axes: bool,
    /// Maximum allowable angle (in degrees) between the tool axis (`nozzle_axis`) and the bed normal (`+Z_bed`).
    pub max_angle_from_bed_normal_deg: f32,
    /// Piecewise-linear clearance profile in the nozzle frame `(radius_x_mm, height_z_mm)`,
    /// ordered by ascending `z_height_mm` from the nozzle tip (`z = 0`).
    pub clearance_profile_xz: Vec<ClearancePointXZ>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self::standard_tool(0, "T0 - 0.4mm Standard", 0.40, 215.0)
    }
}

impl ToolConfig {
    /// Creates a standard extrusion tool with a default multi-segment XZ clearance profile
    /// modeling the nozzle tip flat, brass cone, heater block, and toolhead shroud.
    pub fn standard_tool(
        id: u32,
        name: impl Into<String>,
        nozzle_diameter_mm: f32,
        temperature_celsius: f32,
    ) -> Self {
        let tip_outer_r = (nozzle_diameter_mm * 1.25).max(0.4);
        Self {
            id,
            name: name.into(),
            nozzle_diameter_mm,
            temperature_celsius,
            has_rotational_axes: true,
            max_angle_from_bed_normal_deg: 45.0,
            clearance_profile_xz: vec![
                ClearancePointXZ::new(tip_outer_r, 0.0),
                ClearancePointXZ::new(2.5, 2.0),
                ClearancePointXZ::new(9.0, 6.0),
                ClearancePointXZ::new(22.0, 18.0),
                ClearancePointXZ::new(28.0, 45.0),
            ],
        }
    }

    /// Evaluates the interpolated clearance radius (mm) at a given height `z_mm` above the nozzle tip
    /// in the nozzle's local frame.
    pub fn clearance_radius_at_z(&self, z_mm: f32) -> Option<f32> {
        if self.clearance_profile_xz.is_empty() || z_mm < 0.0 {
            return None;
        }
        let first = self.clearance_profile_xz[0];
        if z_mm <= first.z_height_mm {
            return Some(first.x_radius_mm);
        }
        for pair in self.clearance_profile_xz.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if z_mm >= a.z_height_mm && z_mm <= b.z_height_mm {
                let dz = b.z_height_mm - a.z_height_mm;
                if dz.abs() < 1.0e-6 {
                    return Some(a.x_radius_mm.max(b.x_radius_mm));
                }
                let t = (z_mm - a.z_height_mm) / dz;
                return Some(a.x_radius_mm + t * (b.x_radius_mm - a.x_radius_mm));
            }
        }
        self.clearance_profile_xz.last().map(|p| p.x_radius_mm)
    }
}

/// Non-constant layer thickness specification for conformal / non-planar slicing,
/// represented as `(target_mm, min_mm, max_mm, first_layer_mm, top_layer_mm)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LayerHeightConfig {
    /// Nominal target layer thickness (mm).
    pub target_mm: f32,
    /// Minimum allowable layer thickness in compressed/high-curvature zones (mm).
    pub min_mm: f32,
    /// Maximum allowable layer thickness in expanded conformal zones (mm).
    pub max_mm: f32,
    /// Dedicated first-layer thickness for the initial isosurface off the bed (mm).
    pub first_layer_mm: f32,
    /// Dedicated top-layer thickness setting the final isosurface in each branch (mm).
    pub top_layer_mm: f32,
}

impl Default for LayerHeightConfig {
    fn default() -> Self {
        Self {
            target_mm: 0.20,
            min_mm: 0.08,
            max_mm: 0.32,
            first_layer_mm: 0.20,
            top_layer_mm: 0.20,
        }
    }
}

impl LayerHeightConfig {
    /// Ensures `0.01 <= min_mm <= target_mm <= max_mm` and positive `first_layer_mm`/`top_layer_mm`.
    pub fn sanitize(&mut self) {
        self.min_mm = self.min_mm.max(0.01);
        if self.max_mm < self.min_mm {
            self.max_mm = self.min_mm;
        }
        self.target_mm = self.target_mm.clamp(self.min_mm, self.max_mm);
        self.first_layer_mm = self.first_layer_mm.clamp(0.01, self.max_mm.max(1.0));
        self.top_layer_mm = self.top_layer_mm.clamp(0.01, self.max_mm.max(1.0));
    }
}

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

/// Configurable variable extrusion line width `(target_mm, min_mm, max_mm)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LineWidthConfig {
    /// Nominal target extrusion line width (mm).
    pub target_mm: f32,
    /// Minimum allowable line width when compressing to fit narrow spaces (mm).
    pub min_mm: f32,
    /// Maximum allowable line width when expanding to fill residual gaps (mm).
    pub max_mm: f32,
}

impl Default for LineWidthConfig {
    fn default() -> Self {
        Self {
            target_mm: 0.40,
            min_mm: 0.28,
            max_mm: 0.56,
        }
    }
}

impl LineWidthConfig {
    pub const fn new(target_mm: f32, min_mm: f32, max_mm: f32) -> Self {
        Self {
            target_mm,
            min_mm,
            max_mm,
        }
    }

    /// Ensures `0.05 <= min_mm <= target_mm <= max_mm`.
    pub fn sanitize(&mut self) {
        self.min_mm = self.min_mm.max(0.05);
        if self.max_mm < self.min_mm {
            self.max_mm = self.min_mm;
        }
        self.target_mm = self.target_mm.clamp(self.min_mm, self.max_mm);
    }
}

/// Pattern used to generate solid infill inside the outer shell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolidInfillPattern {
    /// Monotonic directional flood fill across the remaining shell region.
    #[default]
    Monotonic,
    /// Continued inward SDF/geodesic insetting across the remaining shell region.
    Concentric,
}

impl SolidInfillPattern {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Monotonic => 0,
            Self::Concentric => 1,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Concentric,
            _ => Self::Monotonic,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Monotonic => "Monotonic (Flood Fill)",
            Self::Concentric => "Concentric (Continued Insetting)",
        }
    }
}

/// Pattern used to generate sparse infill inside the interior core ($\phi_{\text{sdf}}(\mathbf{x}) < -T_{\text{shell}}$).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SparseInfillPattern {
    /// Alternating-angle rectilinear scanlines with Union-Find island splitting & boustrophedon chaining.
    Monotonic,
    /// 3D isotropic space-diagonal truss (dual orthogonal ±45° scanlines with √2 phase drift).
    Cubic,
    /// Triply Periodic Minimal Surface (Gyroid: sin X cos Y + sin Y cos Z + sin Z cos X = 0).
    #[default]
    Gyroid,
    /// Triply Periodic Minimal Surface (Schwarz D / Diamond lattice).
    SchwarzD,
    /// Triply Periodic Minimal Surface (Schwarz P / Primitive lattice).
    SchwarzP,
    /// Iterative inward geodesic/SDF offset rings at spacing w / ρ.
    Concentric,
    /// Iterative inward geodesic/SDF offset rings at 1× line width (100% concentric perimeters).
    AllWalls,
}

impl SparseInfillPattern {
    pub const ALL: [Self; 7] = [
        Self::Monotonic,
        Self::Cubic,
        Self::Gyroid,
        Self::SchwarzD,
        Self::SchwarzP,
        Self::Concentric,
        Self::AllWalls,
    ];

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Monotonic => 0,
            Self::Cubic => 1,
            Self::Gyroid => 2,
            Self::SchwarzD => 3,
            Self::SchwarzP => 4,
            Self::Concentric => 5,
            Self::AllWalls => 6,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Monotonic,
            1 => Self::Cubic,
            2 => Self::Gyroid,
            3 => Self::SchwarzD,
            4 => Self::SchwarzP,
            5 => Self::Concentric,
            6 => Self::AllWalls,
            _ => Self::Gyroid,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Monotonic => "Monotonic (Rectilinear)",
            Self::Cubic => "Cubic (3D Space-Diagonal Truss)",
            Self::Gyroid => "Gyroid (TPMS)",
            Self::SchwarzD => "Schwarz D (Diamond TPMS)",
            Self::SchwarzP => "Schwarz P (Primitive TPMS)",
            Self::Concentric => "Concentric (Inward Offset)",
            Self::AllWalls => "All Walls (100% Concentric)",
        }
    }
}

/// Non-planar slicing process & shell/sparse toolpath generation configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SlicingProcessConfig {
    /// Variable non-planar layer height (`target_mm`, `min_mm`, `max_mm`, `first_layer_mm`, `top_layer_mm`).
    pub layer_height: LayerHeightConfig,
    pub max_nonplanar_slope_deg: f32,
    pub perimeter_count: u32,
    /// Sparse infill density percentage (`0.0..=100.0`).
    pub infill_density_percent: f32,
    /// Base in-plane rotation angle for scanline infill patterns (`0.0..=180.0` degrees).
    pub infill_angle_deg: f32,
    /// Thickness of the outer shell (mm), defining the shell region $-T_{\text{shell}} \le \phi_{\text{sdf}}(\mathbf{x}) \le 0$.
    pub shell_thickness_mm: f32,
    /// Number of inner walls immediately inside the outer wall (in addition to the 1 outer wall).
    pub inner_wall_count: u32,
    /// Variable line width `(target_mm, min_mm, max_mm)` for outer walls contacting $\phi_{\text{sdf}} = 0$.
    pub outer_wall_width: LineWidthConfig,
    /// Variable line width `(target_mm, min_mm, max_mm)` for inner walls.
    pub inner_wall_width: LineWidthConfig,
    /// Variable line width `(target_mm, min_mm, max_mm)` for solid infill and shell gap fill.
    pub solid_infill_width: LineWidthConfig,
    /// Solid infill pattern within the shell (`Monotonic` or `Concentric`).
    pub solid_infill_pattern: SolidInfillPattern,
    /// Variable line width `(target_mm, min_mm, max_mm)` for sparse infill inside the core ($\phi_{\text{sdf}} < -T_{\text{shell}}$).
    pub sparse_infill_width: LineWidthConfig,
    /// Sparse infill pattern inside the interior core (`Monotonic`, `Cubic`, `Gyroid`, `SchwarzD`, `SchwarzP`, `Concentric`, `AllWalls`).
    pub sparse_infill_pattern: SparseInfillPattern,
    /// Minimum continuous extrusion length (mm); defaults to $2 \times d_{\text{nozzle}}$.
    /// Individual segments of a continuous extrusion may be shorter than this.
    pub min_extrusion_length_mm: f32,
}

impl Default for SlicingProcessConfig {
    fn default() -> Self {
        Self {
            layer_height: LayerHeightConfig::default(),
            max_nonplanar_slope_deg: 35.0,
            perimeter_count: 3,
            infill_density_percent: 20.0,
            infill_angle_deg: 45.0,
            shell_thickness_mm: 1.60,
            inner_wall_count: 2,
            outer_wall_width: LineWidthConfig::new(0.40, 0.28, 0.56),
            inner_wall_width: LineWidthConfig::new(0.42, 0.28, 0.58),
            solid_infill_width: LineWidthConfig::new(0.42, 0.26, 0.60),
            solid_infill_pattern: SolidInfillPattern::Monotonic,
            sparse_infill_width: LineWidthConfig::new(0.45, 0.30, 0.65),
            sparse_infill_pattern: SparseInfillPattern::Gyroid,
            min_extrusion_length_mm: 0.80,
        }
    }
}

impl SlicingProcessConfig {
    pub fn sanitize(&mut self) {
        self.layer_height.sanitize();
        self.infill_density_percent = self.infill_density_percent.clamp(0.0, 100.0);
        self.infill_angle_deg = self.infill_angle_deg.clamp(0.0, 180.0);
        self.shell_thickness_mm = self.shell_thickness_mm.max(0.05);
        self.outer_wall_width.sanitize();
        self.inner_wall_width.sanitize();
        self.solid_infill_width.sanitize();
        self.sparse_infill_width.sanitize();
        self.min_extrusion_length_mm = self.min_extrusion_length_mm.max(0.0);
    }
}

/// Mode for a surface-face orientation hint selected on a STEP B-Rep face or near-planar mesh patch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceHintMode {
    /// Suggests isosurface normal $\nabla \tau$ orthogonal to the selected face
    /// ($\nabla \tau \parallel \pm \hat{\mathbf{n}}_{\text{face}}$), making layers tangent/conformal to the surface.
    #[default]
    NormalOrthogonalToSurface,
    /// Suggests isosurface normal $\nabla \tau$ tangent to the selected face
    /// ($\nabla \tau \perp \hat{\mathbf{n}}_{\text{face}}$), making layers meet the surface orthogonally ($90^\circ$).
    NormalTangentToSurface,
}

impl SurfaceHintMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::NormalOrthogonalToSurface => "Normal ⊥ Surface (Conformal Layers)",
            Self::NormalTangentToSurface => "Normal ∥ Surface (⊥ Layers)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::NormalOrthogonalToSurface => "⊥ Normal",
            Self::NormalTangentToSurface => "∥ Normal",
        }
    }
}

/// A soft orientation hint assigned to a selectable `SurfacePatch` on an object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFaceHint {
    pub object_id: u64,
    pub patch_id: u32,
    pub mode: SurfaceHintMode,
    pub influence_depth_mm: f32,
    pub weight: f32,
}

/// A soft orientation hint placed at a 3D point in space (stored in object-local coordinates
/// so it follows the object if moved/rotated on the build plate).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointNormalHint {
    pub id: u32,
    pub object_id: u64,
    /// Position in the object's local coordinate frame (mm).
    pub local_position: [f32; 3],
    /// Desired unit normal direction $\hat{\mathbf{d}}$ for $\nabla \tau$ in the object's local coordinate frame.
    pub local_direction: [f32; 3],
    pub influence_radius_mm: f32,
    pub weight: f32,
}

/// OpenVDB volumetric Eikonal and SDF field parameters and soft orientation hints.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EikonalFieldConfig {
    pub voxel_size_mm: f32,
    pub narrow_band_width_voxels: u32,
    pub curvature_regularization: f32,
    /// Number of outer constraint & tool-orientation relaxation iterations (`>= 1`).
    pub solver_iterations: u32,
    /// Default interior influence depth (mm) for newly placed surface-face hints.
    pub default_surface_hint_depth_mm: f32,
    /// Default spherical influence radius (mm) for newly placed 3D point hints.
    pub default_point_hint_radius_mm: f32,
    /// Default soft blend weight (`0.0..=1.0`) for newly placed hints.
    pub default_hint_weight: f32,
    /// Active surface-face hints placed on STEP B-Rep faces or contiguous planar mesh patches.
    pub surface_hints: Vec<SurfaceFaceHint>,
    /// Active 3D point normal hints placed in space.
    pub point_hints: Vec<PointNormalHint>,
}

impl Default for EikonalFieldConfig {
    fn default() -> Self {
        Self {
            voxel_size_mm: 0.50,
            narrow_band_width_voxels: 6,
            curvature_regularization: 0.05,
            solver_iterations: 4,
            default_surface_hint_depth_mm: 12.0,
            default_point_hint_radius_mm: 18.0,
            default_hint_weight: 0.85,
            surface_hints: Vec::new(),
            point_hints: Vec::new(),
        }
    }
}

/// Multi-axis machine kinematics state in bed coordinates (6-axis robot arm nozzle
/// orientation & 5-axis tilting build plate gravity orientation).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MachineKinematicsConfig {
    /// Unit vector pointing opposite to the nozzle discharge direction in bed coordinates.
    /// Currently defaults to `[0.0, 0.0, 1.0]` (parallel to +Z), but will vary dynamically.
    pub nozzle_axis_bed: [f32; 3],
    /// Unit vector pointing in the direction of gravity in bed coordinates.
    /// Defaults to `[0.0, 0.0, -1.0]`, and rotates as the machine tilts the bed.
    pub gravity_direction_bed: [f32; 3],
    /// Bed tilt around the bed X axis (degrees) for 5-axis bed-tilt kinematics.
    pub bed_tilt_rx_deg: f32,
    /// Bed tilt around the bed Y axis (degrees) for 5-axis bed-tilt kinematics.
    pub bed_tilt_ry_deg: f32,
    /// Whether to draw the active tool's nozzle frame, XZ clearance profile, and gravity
    /// vector gizmo in the 3D viewport.
    pub show_nozzle_clearance_gizmo: bool,
}

impl Default for MachineKinematicsConfig {
    fn default() -> Self {
        Self {
            nozzle_axis_bed: [0.0, 0.0, 1.0],
            gravity_direction_bed: [0.0, 0.0, -1.0],
            bed_tilt_rx_deg: 0.0,
            bed_tilt_ry_deg: 0.0,
            show_nozzle_clearance_gizmo: false,
        }
    }
}

impl MachineKinematicsConfig {
    /// Updates `gravity_direction_bed` from the 5-axis bed tilt angles (`bed_tilt_rx_deg`, `bed_tilt_ry_deg`).
    /// When the bed tilts by $R_{\text{bed}}$, the world gravity vector $(0, 0, -1)$ expressed in
    /// bed coordinates becomes $R_{\text{bed}}^{-1} (0, 0, -1)$.
    pub fn update_gravity_from_bed_tilt(&mut self) {
        let rx = self.bed_tilt_rx_deg.to_radians();
        let ry = self.bed_tilt_ry_deg.to_radians();
        let bed_rot = Quat::from_euler(EulerRot::XYZ, rx, ry, 0.0);
        let g_bed = (bed_rot.inverse() * (-Vec3::Z)).normalize_or_zero();
        let g = if g_bed.length_squared() > 0.5 {
            g_bed
        } else {
            -Vec3::Z
        };
        self.gravity_direction_bed = g.to_array();
    }

    /// Queries the nozzle's local orthonormal frame at `position_bed` moving along `travel_dir_bed`.
    pub fn query_nozzle_frame(&self, _position_bed: Vec3, travel_dir_bed: Vec3) -> NozzleFrame {
        let axis = Vec3::from_array(self.nozzle_axis_bed);
        NozzleFrame::from_nozzle_and_travel(axis, travel_dir_bed)
    }

    /// Queries the unit gravity direction vector in bed coordinates at `position_bed`.
    pub fn query_gravity_bed(&self, _position_bed: Vec3) -> Vec3 {
        Vec3::from_array(self.gravity_direction_bed)
            .try_normalize()
            .unwrap_or(-Vec3::Z)
    }
}

/// Ordering strategy for multiple walls within an island on a layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WallPrintOrder {
    /// Prints inner walls `[n-1 .. 2]`, then outer wall `0`, then backing inner wall `1`
    /// so the outer wall sets before backing pressure is applied.
    #[default]
    InnerOuterInner,
    /// Prints outer wall `0` first, followed by inner walls `1 .. n`.
    OutsideIn,
}

impl WallPrintOrder {
    pub const ALL: [Self; 2] = [Self::InnerOuterInner, Self::OutsideIn];

    pub fn label(self) -> &'static str {
        match self {
            Self::InnerOuterInner => "Inner -> Outer -> Backing (In-Out-In)",
            Self::OutsideIn => "Outside -> In",
        }
    }
}

/// Configuration for toolpath-to-Gcode kinematic planning, flow/rheology compensation,
/// dynamic Pressure Advance, retractions, and start/end G-code templates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GcodeProcessConfig {
    /// Heated bed target temperature (°C).
    pub bed_temperature_celsius: f32,
    /// Raw filament diameter (mm), typically `1.75` or `2.85`.
    pub filament_diameter_mm: f32,
    /// Global extrusion flow multiplier (`0.5..=1.5`).
    pub extrusion_multiplier: f32,
    /// First-layer extrusion multiplier (`0.7..=1.5`).
    pub first_layer_flow_multiplier: f32,
    /// Wall printing order (`InnerOuterInner` or `OutsideIn`).
    pub wall_print_order: WallPrintOrder,
    /// When `false` (default for rectangular beds), G-code XY coordinates place `(0, 0)` at the
    /// front-left corner of the bed (`[0..width_x, 0..depth_y]`). When `true`, `(0, 0)` is at bed center.
    pub gcode_origin_at_bed_center: bool,
    /// Emit Klipper `[exclude_object]` commands (`EXCLUDE_OBJECT_DEFINE`, `EXCLUDE_OBJECT_START`, `EXCLUDE_OBJECT_END`).
    pub emit_exclude_object: bool,
    /// Embed base64-encoded PNG preview thumbnails (`32x32` and `300x300`) in the G-code header for Moonraker/Mainsail/Fluidd.
    pub emit_thumbnails: bool,

    // --- Linear Speeds (mm/s) ---
    pub first_layer_speed_mms: f32,
    pub outer_wall_speed_mms: f32,
    pub inner_wall_speed_mms: f32,
    pub solid_infill_speed_mms: f32,
    pub sparse_infill_speed_mms: f32,
    pub gap_fill_speed_mms: f32,
    pub travel_speed_mms: f32,
    pub z_hop_speed_mms: f32,

    // --- Accelerations (mm/s²) & Klipper Lookahead ---
    pub outer_wall_accel_mms2: f32,
    pub inner_wall_accel_mms2: f32,
    pub infill_accel_mms2: f32,
    pub travel_accel_mms2: f32,
    pub square_corner_velocity_mms: f32,
    pub minimum_cruise_ratio: f32,
    pub max_volumetric_flow_mm3s: f32,
    pub use_stepper_dynamics: bool,
    pub max_stepper_speed_mms: f32,
    pub z_travel_penalty: f32,

    // --- Seam, Scarf Joint, Wipe & Retraction ---
    pub enable_scarf_joint: bool,
    pub scarf_length_mm: f32,
    pub scarf_start_height_ratio: f32,
    pub seam_gap_mm: f32,
    pub pre_retract_taper_mm: f32,
    pub wipe_length_mm: f32,
    pub retract_length_mm: f32,
    pub retract_speed_mms: f32,
    pub min_travel_for_retract_mm: f32,
    pub z_hop_mm: f32,
    pub ooze_relaxation_extra_prime_mm: f32,
    pub ooze_relaxation_tau_s: f32,

    // --- Non-Newtonian Rheology, Pressure Advance & Corner/Transient Flow ---
    pub enable_flat_nozzle_comp: bool,
    pub enable_corner_flow_comp: bool,
    pub corner_flow_comp_ratio: f32,
    pub enable_transient_pressure_comp: bool,
    pub transient_pressure_exponent: f32,
    pub transient_min_flow_multiplier: f32,
    pub enable_dynamic_pa: bool,
    pub enable_slicer_pa_subdivision: bool,
    pub pa_low_flow_s: f32,
    pub pa_high_flow_s: f32,
    pub pa_ref_low_flow_mm3s: f32,
    pub pa_ref_high_flow_mm3s: f32,
    pub swell_low: f32,
    pub swell_high: f32,
    pub fan_speed_percent: f32,

    // --- Start & End G-code Templates ---
    pub start_gcode_template: String,
    pub end_gcode_template: String,
}

impl Default for GcodeProcessConfig {
    fn default() -> Self {
        Self {
            bed_temperature_celsius: 60.0,
            filament_diameter_mm: 1.75,
            extrusion_multiplier: 1.0,
            first_layer_flow_multiplier: 1.02,
            wall_print_order: WallPrintOrder::InnerOuterInner,
            gcode_origin_at_bed_center: false,
            emit_exclude_object: true,
            emit_thumbnails: true,
            first_layer_speed_mms: 45.0,
            outer_wall_speed_mms: 65.0,
            inner_wall_speed_mms: 100.0,
            solid_infill_speed_mms: 130.0,
            sparse_infill_speed_mms: 160.0,
            gap_fill_speed_mms: 60.0,
            travel_speed_mms: 250.0,
            z_hop_speed_mms: 25.0,
            outer_wall_accel_mms2: 2500.0,
            inner_wall_accel_mms2: 4500.0,
            infill_accel_mms2: 6500.0,
            travel_accel_mms2: 9000.0,
            square_corner_velocity_mms: 5.0,
            minimum_cruise_ratio: 0.50,
            max_volumetric_flow_mm3s: 18.0,
            use_stepper_dynamics: true,
            max_stepper_speed_mms: 400.0,
            z_travel_penalty: 4.0,
            enable_scarf_joint: true,
            scarf_length_mm: 8.0,
            scarf_start_height_ratio: 0.12,
            seam_gap_mm: 0.06,
            pre_retract_taper_mm: 1.0,
            wipe_length_mm: 1.5,
            retract_length_mm: 0.80,
            retract_speed_mms: 45.0,
            min_travel_for_retract_mm: 1.50,
            z_hop_mm: 0.40,
            ooze_relaxation_extra_prime_mm: -0.02,
            ooze_relaxation_tau_s: 1.50,
            enable_flat_nozzle_comp: true,
            enable_corner_flow_comp: true,
            corner_flow_comp_ratio: 0.85,
            enable_transient_pressure_comp: true,
            transient_pressure_exponent: 0.35,
            transient_min_flow_multiplier: 0.72,
            enable_dynamic_pa: true,
            enable_slicer_pa_subdivision: false,
            pa_low_flow_s: 0.042,
            pa_high_flow_s: 0.024,
            pa_ref_low_flow_mm3s: 2.0,
            pa_ref_high_flow_mm3s: 14.0,
            swell_low: 1.03,
            swell_high: 1.10,
            fan_speed_percent: 100.0,
            start_gcode_template: concat!(
                "; --- Manifold2 Non-Planar Print Start ---\n",
                "; Print Bounds: X[{bbox_min_x}..{bbox_max_x}] Y[{bbox_min_y}..{bbox_max_y}] Z[{bbox_min_z}..{bbox_max_z}]\n",
                "G90 ; Absolute positioning\n",
                "M83 ; Relative extrusion\n",
                "M140 S{bed_temp} ; Set bed temperature\n",
                "M104 S{nozzle_temp} T{tool_id} ; Set nozzle temperature\n",
                "G28 ; Home all axes\n",
                "T{tool_id} ; Select active tool\n",
                "M190 S{bed_temp} ; Wait for bed temperature\n",
                "M109 S{nozzle_temp} T{tool_id} ; Wait for nozzle temperature\n",
                "G92 E0 ; Reset extruder"
            )
            .to_string(),
            end_gcode_template: concat!(
                "; --- Manifold2 Non-Planar Print End ---\n",
                "M400 ; Wait for moves to finish\n",
                "M104 S0 ; Turn off hotend\n",
                "M140 S0 ; Turn off bed\n",
                "M107 ; Turn off part cooling fan\n",
                "G28 X Y ; Park toolhead\n",
                "M84 ; Disable steppers"
            )
            .to_string(),
        }
    }
}

impl GcodeProcessConfig {
    pub fn sanitize(&mut self) {
        self.bed_temperature_celsius = self.bed_temperature_celsius.clamp(0.0, 150.0);
        self.filament_diameter_mm = self.filament_diameter_mm.clamp(1.0, 3.5);
        self.extrusion_multiplier = self.extrusion_multiplier.clamp(0.2, 2.0);
        self.first_layer_flow_multiplier = self.first_layer_flow_multiplier.clamp(0.5, 2.0);
        self.first_layer_speed_mms = self.first_layer_speed_mms.clamp(5.0, 500.0);
        self.outer_wall_speed_mms = self.outer_wall_speed_mms.clamp(5.0, 600.0);
        self.inner_wall_speed_mms = self.inner_wall_speed_mms.clamp(5.0, 600.0);
        self.solid_infill_speed_mms = self.solid_infill_speed_mms.clamp(5.0, 800.0);
        self.sparse_infill_speed_mms = self.sparse_infill_speed_mms.clamp(5.0, 800.0);
        self.gap_fill_speed_mms = self.gap_fill_speed_mms.clamp(5.0, 500.0);
        self.travel_speed_mms = self.travel_speed_mms.clamp(20.0, 1200.0);
        self.z_hop_speed_mms = self.z_hop_speed_mms.clamp(2.0, 200.0);
        self.outer_wall_accel_mms2 = self.outer_wall_accel_mms2.clamp(100.0, 50000.0);
        self.inner_wall_accel_mms2 = self.inner_wall_accel_mms2.clamp(100.0, 50000.0);
        self.infill_accel_mms2 = self.infill_accel_mms2.clamp(100.0, 50000.0);
        self.travel_accel_mms2 = self.travel_accel_mms2.clamp(100.0, 80000.0);
        self.square_corner_velocity_mms = self.square_corner_velocity_mms.clamp(0.5, 50.0);
        self.minimum_cruise_ratio = self.minimum_cruise_ratio.clamp(0.0, 0.95);
        self.max_volumetric_flow_mm3s = self.max_volumetric_flow_mm3s.clamp(1.0, 100.0);
        self.max_stepper_speed_mms = self.max_stepper_speed_mms.clamp(50.0, 2000.0);
        self.scarf_length_mm = self.scarf_length_mm.clamp(1.0, 40.0);
        self.scarf_start_height_ratio = self.scarf_start_height_ratio.clamp(0.02, 0.50);
        self.seam_gap_mm = self.seam_gap_mm.clamp(0.0, 1.0);
        self.pre_retract_taper_mm = self.pre_retract_taper_mm.clamp(0.0, 10.0);
        self.wipe_length_mm = self.wipe_length_mm.clamp(0.0, 10.0);
        self.retract_length_mm = self.retract_length_mm.clamp(0.0, 10.0);
        self.retract_speed_mms = self.retract_speed_mms.clamp(5.0, 150.0);
        self.min_travel_for_retract_mm = self.min_travel_for_retract_mm.clamp(0.1, 20.0);
        self.z_hop_mm = self.z_hop_mm.clamp(0.0, 5.0);
        self.pa_low_flow_s = self.pa_low_flow_s.clamp(0.0, 0.5);
        self.pa_high_flow_s = self.pa_high_flow_s.clamp(0.0, 0.5);
        self.swell_low = self.swell_low.clamp(1.0, 1.8);
        self.swell_high = self.swell_high.clamp(1.0, 1.8);
        self.fan_speed_percent = self.fan_speed_percent.clamp(0.0, 100.0);
    }

    /// Effective minimum travel distance before triggering a retraction, accounting for scarf joint length.
    pub fn effective_min_travel_for_retract(&self) -> f32 {
        if self.enable_scarf_joint {
            self.min_travel_for_retract_mm + 0.35 * self.scarf_length_mm
        } else {
            self.min_travel_for_retract_mm
        }
    }
}

/// Top-level extensible slicer configuration stored as a Bevy resource and
/// persisted inside `.3mf` project files under `/Metadata/manifold2_config.toml`.
#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SlicerConfig {
    pub bed: PrintBedConfig,
    pub arrangement: ArrangementConfig,
    pub slicing: SlicingProcessConfig,
    pub gcode: GcodeProcessConfig,
    pub eikonal: EikonalFieldConfig,
    pub machine: MachineKinematicsConfig,
    pub tools: Vec<ToolConfig>,
    pub active_tool_index: usize,
}

impl Default for SlicerConfig {
    fn default() -> Self {
        Self {
            bed: PrintBedConfig::default(),
            arrangement: ArrangementConfig::default(),
            slicing: SlicingProcessConfig::default(),
            gcode: GcodeProcessConfig::default(),
            eikonal: EikonalFieldConfig::default(),
            machine: MachineKinematicsConfig::default(),
            tools: vec![
                ToolConfig::standard_tool(0, "T0 - 0.4mm Standard", 0.40, 215.0),
                ToolConfig::standard_tool(1, "T1 - 0.6mm High-Flow", 0.60, 230.0),
            ],
            active_tool_index: 0,
        }
    }
}

impl SlicerConfig {
    /// Returns the currently selected tool configuration (creating a default if `tools` was emptied).
    pub fn active_tool(&self) -> Option<&ToolConfig> {
        if self.tools.is_empty() {
            None
        } else {
            let idx = self.active_tool_index.min(self.tools.len() - 1);
            self.tools.get(idx)
        }
    }

    /// Queries the nozzle's local orthonormal frame at `position_bed` moving along `travel_dir_bed`.
    pub fn query_nozzle_frame(&self, position_bed: Vec3, travel_dir_bed: Vec3) -> NozzleFrame {
        self.machine.query_nozzle_frame(position_bed, travel_dir_bed)
    }

    /// Queries the unit gravity direction vector in bed coordinates at `position_bed`.
    pub fn query_gravity_bed(&self, position_bed: Vec3) -> Vec3 {
        self.machine.query_gravity_bed(position_bed)
    }

    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nozzle_frame_right_handed_orthonormal_basis() {
        // Default vertical nozzle (+Z opposite to nozzle discharge) moving along +X
        let frame = NozzleFrame::from_nozzle_and_travel(Vec3::Z, Vec3::X);
        assert!((frame.nozzle_axis - Vec3::Z).length() < 1e-6);
        assert!((frame.travel_axis - Vec3::X).length() < 1e-6);
        assert!((frame.lateral_axis - Vec3::Y).length() < 1e-6);

        // Verify right-hand rule: travel_axis × lateral_axis == nozzle_axis
        let cross = frame.travel_axis.cross(frame.lateral_axis);
        assert!((cross - frame.nozzle_axis).length() < 1e-6);

        // Tilted 6-axis robot nozzle with non-orthogonal travel direction
        let tilted_nozzle = Vec3::new(0.3, -0.4, 0.8660254).normalize();
        let raw_travel = Vec3::new(1.0, 0.5, 0.2);
        let tilted_frame = NozzleFrame::from_nozzle_and_travel(tilted_nozzle, raw_travel);

        assert!((tilted_frame.nozzle_axis.length() - 1.0).abs() < 1e-6);
        assert!((tilted_frame.travel_axis.length() - 1.0).abs() < 1e-6);
        assert!((tilted_frame.lateral_axis.length() - 1.0).abs() < 1e-6);
        assert!(tilted_frame.nozzle_axis.dot(tilted_frame.travel_axis).abs() < 1e-6);
        assert!(tilted_frame.nozzle_axis.dot(tilted_frame.lateral_axis).abs() < 1e-6);
        assert!(tilted_frame.travel_axis.dot(tilted_frame.lateral_axis).abs() < 1e-6);
        assert!(
            (tilted_frame.travel_axis.cross(tilted_frame.lateral_axis) - tilted_frame.nozzle_axis)
                .length()
                < 1e-5
        );
    }

    #[test]
    fn test_gravity_bed_tilt_and_tool_clearance_profile() {
        let mut config = SlicerConfig::default();
        assert_eq!(config.query_gravity_bed(Vec3::ZERO), -Vec3::Z);

        // Tilt bed by 30 degrees around X axis
        config.machine.bed_tilt_rx_deg = 30.0;
        config.machine.update_gravity_from_bed_tilt();
        let g = config.query_gravity_bed(Vec3::ZERO);
        assert!((g.length() - 1.0).abs() < 1e-6);
        assert!(g.y.abs() > 0.45 && g.z < -0.80);

        // Verify tool clearance profile interpolation in nozzle frame
        let tool = config.active_tool().unwrap();
        assert!((tool.nozzle_diameter_mm - 0.40).abs() < 1e-6);
        assert!((tool.temperature_celsius - 215.0).abs() < 1e-6);
        let r0 = tool.clearance_radius_at_z(0.0).unwrap();
        let r1 = tool.clearance_radius_at_z(1.0).unwrap();
        let r2 = tool.clearance_radius_at_z(2.0).unwrap();
        assert!(r0 < r1 && r1 < r2);

        // Verify variable layer height sanitization and TOML roundtrip
        config.slicing.layer_height = LayerHeightConfig {
            target_mm: 0.20,
            min_mm: 0.08,
            max_mm: 0.32,
            first_layer_mm: 0.24,
            top_layer_mm: 0.16,
        };
        let toml_str = config.to_toml_string().unwrap();
        let decoded = SlicerConfig::from_toml_str(&toml_str).unwrap();
        assert_eq!(decoded.tools.len(), 2);
        assert_eq!(decoded.slicing.layer_height, config.slicing.layer_height);
    }
}
