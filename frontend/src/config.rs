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
    /// Overhang angle threshold from vertical (degrees, `5.0..=85.0`) derived from the `"overhang"`
    /// field above which unsupported extrusions switch to Bridging or Wave Overhangs.
    pub overhang_angle_threshold_deg: f32,
    /// Maximum unsupported 3D straight-line span length (mm) for circular Bridging extrusions.
    pub max_bridge_length_mm: f32,
    /// Lateral overlap fraction (`0.01..=0.35`, default `0.08` = 8%) of a standard line width
    /// maintained between each inside-out Wave Overhang extrusion and the preceding extrusion.
    pub wave_overhang_overlap_fraction: f32,
    /// Enable automatic support detection and support toolpath generation for regions unreachable
    /// by the constrained Eikonal order field.
    pub enable_support_generation: bool,
    /// Seed the constrained Eikonal order field $\tau(\mathbf{x})$ from support contact regions
    /// so unreachable cantilevers and hanging islands are reached causally.
    pub seed_order_field_from_supports: bool,
    /// When true, only generate supports for regions that are actually causally unreachable by
    /// the constrained order field (floating islands / downward-hanging local minima) and two-pier
    /// bridges whose span exceeds `max_bridge_length_mm` (leaving reachable overhangs to conformal
    /// layers and inside-out wave overhangs).
    pub support_only_unreachable_and_bridges: bool,
    /// Vertical clearance gap (mm) along anti-gravity between support interface/floor and the part surface.
    pub support_z_gap_mm: f32,
    /// Lateral clearance gap (mm) between support structures and the object boundary $\phi_{\text{sdf}} = 0$.
    pub support_xy_gap_mm: f32,
    /// Number of dense support interface layers immediately beneath unreachable overhang roofs.
    pub support_interface_layers: u32,
    /// Line spacing (mm) for coarse support base scanlines.
    pub support_base_spacing_mm: f32,
    /// Line spacing (mm) for dense support interface scanlines.
    pub support_interface_spacing_mm: f32,
    /// Extruded line width (mm) for support structures.
    pub support_line_width_mm: f32,
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
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 40.0,
            wave_overhang_overlap_fraction: 0.08,
            enable_support_generation: true,
            seed_order_field_from_supports: true,
            support_only_unreachable_and_bridges: false,
            support_z_gap_mm: 0.20,
            support_xy_gap_mm: 0.50,
            support_interface_layers: 2,
            support_base_spacing_mm: 2.50,
            support_interface_spacing_mm: 0.60,
            support_line_width_mm: 0.40,
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
        self.overhang_angle_threshold_deg = self.overhang_angle_threshold_deg.clamp(5.0, 85.0);
        self.max_bridge_length_mm = self.max_bridge_length_mm.clamp(0.0, 500.0);
        self.wave_overhang_overlap_fraction = self.wave_overhang_overlap_fraction.clamp(0.01, 0.35);
        self.support_z_gap_mm = self.support_z_gap_mm.clamp(0.05, 5.0);
        self.support_xy_gap_mm = self.support_xy_gap_mm.clamp(0.10, 10.0);
        self.support_interface_layers = self.support_interface_layers.min(16);
        self.support_base_spacing_mm = self.support_base_spacing_mm.clamp(0.40, 20.0);
        self.support_interface_spacing_mm = self.support_interface_spacing_mm.clamp(0.20, 5.0);
        self.support_line_width_mm = self.support_line_width_mm.clamp(0.15, 3.0);
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

/// Computes the default fine OpenVDB voxel size (mm) from the active tool's nozzle diameter
/// $d_{\text{nozzle}}$ and target layer height $h_{\text{target}}$:
/// $$\Delta x_{\text{fine}} = \text{clamp}\!\left(0.375\,d_{\text{nozzle}} + 0.50\,h_{\text{target}},\; 0.08,\; 2.00\right)$$
/// For a standard $0.40\text{ mm}$ nozzle and $0.20\text{ mm}$ layer height, this yields
/// $\Delta x_{\text{fine}} = 0.25\text{ mm}$ (placing $\sim 3\text{--}5$ fine voxels across
/// thin walls of $0.8\text{--}1.2\text{ mm}$), and scales proportionally for larger nozzles
/// and thicker layers where the physical deposition process sets the achievable feature detail.
pub fn compute_auto_fine_voxel_size_mm(
    nozzle_diameter_mm: f32,
    target_layer_height_mm: f32,
) -> f32 {
    let d = nozzle_diameter_mm.max(0.10);
    let h = target_layer_height_mm.max(0.02);
    let raw = 0.375 * d + 0.50 * h;
    ((raw * 100.0).round() / 100.0).clamp(0.08, 2.00)
}

/// OpenVDB volumetric Eikonal and SDF field parameters and soft orientation hints.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EikonalFieldConfig {
    pub voxel_size_mm: f32,
    /// Automatically scale the fine grid `voxel_size_mm` from the active nozzle diameter
    /// and target layer height (`0.375 * nozzle_diameter_mm + 0.50 * target_layer_height_mm`).
    pub auto_voxel_size: bool,
    /// Enable Dual-Grid Multi-Resolution (coarse full-interior core + fine boundary-shell narrow band)
    /// to support large build volumes without excessive memory or compute.
    pub enable_multi_resolution_grid: bool,
    /// Coarse-to-fine voxel size multiplier (`2..=8`, default `4`) used for the interior core grid.
    pub coarse_grid_ratio: u32,
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
    /// Enable automatic overhang-driven conformal vs. orthogonal surface orientation bias
    /// using the gravity-aware `"overhang"` scalar field ($o = -\nabla \phi_{\text{sdf}} \cdot \hat{\mathbf{g}}_{\text{bed}}$).
    pub enable_overhang_orientation_bias: bool,
    /// Soft blend weight (`0.0..=1.0`, default `0.75`) for the overhang-driven surface orientation bias.
    pub overhang_bias_weight: f32,
    /// Supported surface angle from vertical (degrees, default `30.0`, i.e., `-30°` signed overhang angle)
    /// marking the lower boundary of the orthogonal-to-surface regime ($\nabla \tau \perp \hat{\mathbf{n}}_{\text{surf}}$).
    /// Shallower supported surfaces ($< -30^\circ$) bias toward conformal ($\nabla \tau \parallel +\hat{\mathbf{n}}_{\text{surf}}$).
    pub orthogonal_supported_angle_deg: f32,
    /// Overhang surface angle past vertical (degrees, default `60.0`, i.e., `+60°` signed overhang angle)
    /// marking the upper boundary of the orthogonal-to-surface regime ($\nabla \tau \perp \hat{\mathbf{n}}_{\text{surf}}$).
    /// Steeper overhangs ($> +60^\circ$) bias toward conformal ($\nabla \tau \parallel -\hat{\mathbf{n}}_{\text{surf}}$).
    pub orthogonal_overhang_angle_deg: f32,
    /// Smooth transition width (degrees, default `15.0`) over which the weight transitions ($C^1$ cubic
    /// Hermite `smoothstep`) between orthogonal and conformal around `-orthogonal_supported_angle_deg`
    /// and `+orthogonal_overhang_angle_deg`.
    pub overhang_bias_transition_deg: f32,
    /// Active surface-face hints placed on STEP B-Rep faces or contiguous planar mesh patches.
    pub surface_hints: Vec<SurfaceFaceHint>,
    /// Active 3D point normal hints placed in space.
    pub point_hints: Vec<PointNormalHint>,
}

impl Default for EikonalFieldConfig {
    fn default() -> Self {
        Self {
            voxel_size_mm: 0.25,
            auto_voxel_size: true,
            enable_multi_resolution_grid: true,
            coarse_grid_ratio: 4,
            narrow_band_width_voxels: 6,
            curvature_regularization: 0.05,
            solver_iterations: 4,
            default_surface_hint_depth_mm: 12.0,
            default_point_hint_radius_mm: 18.0,
            default_hint_weight: 0.85,
            enable_overhang_orientation_bias: true,
            overhang_bias_weight: 0.75,
            orthogonal_supported_angle_deg: 30.0,
            orthogonal_overhang_angle_deg: 60.0,
            overhang_bias_transition_deg: 15.0,
            surface_hints: Vec::new(),
            point_hints: Vec::new(),
        }
    }
}

impl EikonalFieldConfig {
    pub fn sanitize(&mut self) {
        self.voxel_size_mm = self.voxel_size_mm.clamp(0.05, 5.0);
        self.coarse_grid_ratio = self.coarse_grid_ratio.clamp(2, 8);
        self.narrow_band_width_voxels = self.narrow_band_width_voxels.clamp(2, 32);
        self.curvature_regularization = self.curvature_regularization.clamp(0.0, 1.0);
        self.solver_iterations = self.solver_iterations.clamp(1, 16);
        self.overhang_bias_weight = self.overhang_bias_weight.clamp(0.0, 1.0);
        self.orthogonal_supported_angle_deg = self.orthogonal_supported_angle_deg.clamp(0.0, 89.0);
        self.orthogonal_overhang_angle_deg = self.orthogonal_overhang_angle_deg.clamp(0.0, 89.0);
        self.overhang_bias_transition_deg = self.overhang_bias_transition_deg.clamp(1.0, 45.0);
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
    /// Filament material type identifier (e.g., `"PLA"`, `"PETG"`, `"ABS"`, `"ASA"`, `"TPU"`, `"PA-CF"`).
    pub filament_type: String,
    /// Raw filament diameter (mm), typically `1.75` or `2.85`.
    pub filament_diameter_mm: f32,
    /// Filament bulk density (g/cm³), e.g., `1.24` for PLA, `1.27` for PETG, `1.04` for ABS.
    pub filament_density_g_cm3: f32,
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
    pub bridge_speed_mms: f32,
    pub wave_overhang_speed_mms: f32,
    pub support_speed_mms: f32,
    pub support_interface_speed_mms: f32,
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

    // --- Non-Newtonian Rheology, Pressure Advance, Die Swell & Cooling ---
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
    /// Additional extrusion multiplier for circular Bridge extrusions (applied on top of PA, die swell, etc.).
    pub bridge_flow_multiplier: f32,
    /// Additional extrusion multiplier for Wave Overhang extrusions (applied on top of overlap volume, PA, die swell, etc.).
    pub wave_overhang_flow_multiplier: f32,
    pub fan_speed_percent: f32,
    pub first_layer_fan_speed_percent: f32,
    pub disable_fan_first_layers: u32,
    /// Part cooling fan speed (%) specifically for circular Bridge extrusions.
    pub bridge_fan_speed_percent: f32,
    /// Part cooling fan speed (%) specifically for Wave Overhang extrusions.
    pub wave_overhang_fan_speed_percent: f32,

    // --- Start & End G-code Templates ---
    pub start_gcode_template: String,
    pub end_gcode_template: String,
}

impl Default for GcodeProcessConfig {
    fn default() -> Self {
        Self {
            bed_temperature_celsius: 60.0,
            filament_type: "PLA".to_string(),
            filament_diameter_mm: 1.75,
            filament_density_g_cm3: 1.24,
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
            bridge_speed_mms: 35.0,
            wave_overhang_speed_mms: 30.0,
            support_speed_mms: 80.0,
            support_interface_speed_mms: 45.0,
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
            bridge_flow_multiplier: 0.95,
            wave_overhang_flow_multiplier: 1.00,
            fan_speed_percent: 100.0,
            first_layer_fan_speed_percent: 0.0,
            disable_fan_first_layers: 1,
            bridge_fan_speed_percent: 100.0,
            wave_overhang_fan_speed_percent: 100.0,
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
        if self.filament_type.trim().is_empty() {
            self.filament_type = "PLA".to_string();
        }
        self.filament_diameter_mm = self.filament_diameter_mm.clamp(1.0, 3.5);
        self.filament_density_g_cm3 = self.filament_density_g_cm3.clamp(0.5, 4.0);
        self.extrusion_multiplier = self.extrusion_multiplier.clamp(0.2, 2.0);
        self.first_layer_flow_multiplier = self.first_layer_flow_multiplier.clamp(0.5, 2.0);
        self.first_layer_speed_mms = self.first_layer_speed_mms.clamp(5.0, 500.0);
        self.outer_wall_speed_mms = self.outer_wall_speed_mms.clamp(5.0, 600.0);
        self.inner_wall_speed_mms = self.inner_wall_speed_mms.clamp(5.0, 600.0);
        self.solid_infill_speed_mms = self.solid_infill_speed_mms.clamp(5.0, 800.0);
        self.sparse_infill_speed_mms = self.sparse_infill_speed_mms.clamp(5.0, 800.0);
        self.gap_fill_speed_mms = self.gap_fill_speed_mms.clamp(5.0, 500.0);
        self.bridge_speed_mms = self.bridge_speed_mms.clamp(5.0, 500.0);
        self.wave_overhang_speed_mms = self.wave_overhang_speed_mms.clamp(5.0, 500.0);
        self.support_speed_mms = self.support_speed_mms.clamp(5.0, 600.0);
        self.support_interface_speed_mms = self.support_interface_speed_mms.clamp(5.0, 500.0);
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
        self.z_travel_penalty = self.z_travel_penalty.clamp(1.0, 20.0);
        self.scarf_length_mm = self.scarf_length_mm.clamp(1.0, 40.0);
        self.scarf_start_height_ratio = self.scarf_start_height_ratio.clamp(0.02, 0.50);
        self.seam_gap_mm = self.seam_gap_mm.clamp(0.0, 1.0);
        self.pre_retract_taper_mm = self.pre_retract_taper_mm.clamp(0.0, 10.0);
        self.wipe_length_mm = self.wipe_length_mm.clamp(0.0, 10.0);
        self.retract_length_mm = self.retract_length_mm.clamp(0.0, 10.0);
        self.retract_speed_mms = self.retract_speed_mms.clamp(5.0, 150.0);
        self.min_travel_for_retract_mm = self.min_travel_for_retract_mm.clamp(0.1, 20.0);
        self.z_hop_mm = self.z_hop_mm.clamp(0.0, 5.0);
        self.ooze_relaxation_extra_prime_mm =
            self.ooze_relaxation_extra_prime_mm.clamp(-0.5, 0.5);
        self.ooze_relaxation_tau_s = self.ooze_relaxation_tau_s.clamp(0.1, 30.0);
        self.corner_flow_comp_ratio = self.corner_flow_comp_ratio.clamp(0.0, 1.0);
        self.transient_pressure_exponent = self.transient_pressure_exponent.clamp(0.05, 1.0);
        self.transient_min_flow_multiplier = self.transient_min_flow_multiplier.clamp(0.40, 1.0);
        self.pa_low_flow_s = self.pa_low_flow_s.clamp(0.0, 0.5);
        self.pa_high_flow_s = self.pa_high_flow_s.clamp(0.0, 0.5);
        self.pa_ref_low_flow_mm3s = self.pa_ref_low_flow_mm3s.clamp(0.1, 30.0);
        self.pa_ref_high_flow_mm3s = self
            .pa_ref_high_flow_mm3s
            .clamp(self.pa_ref_low_flow_mm3s + 0.5, 100.0);
        self.swell_low = self.swell_low.clamp(1.0, 2.0);
        self.swell_high = self.swell_high.clamp(self.swell_low, 2.0);
        self.bridge_flow_multiplier = self.bridge_flow_multiplier.clamp(0.2, 2.0);
        self.wave_overhang_flow_multiplier = self.wave_overhang_flow_multiplier.clamp(0.2, 2.0);
        self.fan_speed_percent = self.fan_speed_percent.clamp(0.0, 100.0);
        self.first_layer_fan_speed_percent = self.first_layer_fan_speed_percent.clamp(0.0, 100.0);
        self.disable_fan_first_layers = self.disable_fan_first_layers.min(20);
        self.bridge_fan_speed_percent = self.bridge_fan_speed_percent.clamp(0.0, 100.0);
        self.wave_overhang_fan_speed_percent = self.wave_overhang_fan_speed_percent.clamp(0.0, 100.0);
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
    pub printer_preset_name: String,
    pub tool_preset_name: String,
    pub filament_preset_name: String,
    pub process_preset_name: String,
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
            printer_preset_name: "256³ CoreXY (Standard Klipper)".to_string(),
            tool_preset_name: "0.4mm Standard 5/6-Axis Conical".to_string(),
            filament_preset_name: "Generic PLA".to_string(),
            process_preset_name: "0.20mm Standard Non-Planar".to_string(),
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

    /// Updates `eikonal.voxel_size_mm` from the active tool's `nozzle_diameter_mm` and
    /// `slicing.layer_height.target_mm` when `eikonal.auto_voxel_size` is enabled.
    pub fn sync_auto_voxel_size(&mut self) {
        if self.eikonal.auto_voxel_size {
            let nozzle_mm = self
                .active_tool()
                .map(|t| t.nozzle_diameter_mm)
                .unwrap_or(0.40);
            self.eikonal.voxel_size_mm = compute_auto_fine_voxel_size_mm(
                nozzle_mm,
                self.slicing.layer_height.target_mm,
            );
        }
        self.eikonal.coarse_grid_ratio = self.eikonal.coarse_grid_ratio.clamp(2, 8);
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

/// Structured documentation entry for a single configuration parameter.
/// Used both to render interactive `?` help popups in the UI and to generate
/// complete Markdown reference documentation via [`generate_parameter_docs_markdown`].
#[derive(Clone, Copy, Debug)]
pub struct ParameterDoc {
    /// Hierarchical dotted parameter key (e.g. `"gcode.extrusion_multiplier"`).
    pub key: &'static str,
    /// Human-readable category grouping.
    pub category: &'static str,
    /// Display title for popups and reference tables.
    pub title: &'static str,
    /// Physical units or value range description (empty if unitless/boolean).
    pub units: &'static str,
    /// Default value formatted as a string.
    pub default_value: &'static str,
    /// Concise 1-sentence summary shown at the top of the help popup.
    pub summary: &'static str,
    /// Detailed explanation of the physical/geometric behavior and tuning advice.
    pub details: &'static str,
}

/// Complete structured parameter documentation catalog for Manifold2.
pub const PARAMETER_DOCS: &[ParameterDoc] = &[
    // --- Eikonal Field & Orientation Hints ---
    ParameterDoc {
        key: "eikonal.voxel_size_mm",
        category: "Volumetric Fields & Eikonal Solver",
        title: "OpenVDB Fine Voxel Size",
        units: "mm",
        default_value: "0.25 mm",
        summary: "Spatial resolution of the fine OpenVDB boundary-shell signed-distance and Eikonal grids.",
        details: "Smaller voxel sizes capture finer geometric detail, thin walls, and sharper non-planar curvature transitions. When 'Auto Voxel Size' is enabled, this scales automatically with nozzle diameter and target layer height (0.375 * d_nozzle + 0.50 * h_target).",
    },
    ParameterDoc {
        key: "eikonal.auto_voxel_size",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Auto Fine Voxel Size (Nozzle + Layer Height)",
        units: "",
        default_value: "true",
        summary: "Automatically scales the fine OpenVDB voxel size from nozzle diameter and target layer height.",
        details: "Sets the fine voxel size to 0.375 * d_nozzle + 0.50 * h_target (0.25 mm for a 0.4 mm nozzle at 0.2 mm layers; 0.45 mm for a 0.8 mm nozzle at 0.3 mm layers) so that thin-walled curved parts (~2x nozzle diameter) always span at least 3-5 fine voxels while scaling appropriately for larger nozzles.",
    },
    ParameterDoc {
        key: "eikonal.enable_multi_resolution_grid",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Dual-Grid Multi-Resolution (Coarse Core + Fine Shell)",
        units: "",
        default_value: "true",
        summary: "Uses a coarse full-interior grid for global Eikonal propagation and a fine narrow-band grid in the outer shell.",
        details: "Eliminates the O(L^3 / dx_fine^3) memory and fast-sweeping bottleneck on large build volumes. The deep interior core (sparse infill region) is solved on a coarse grid (e.g. 4x voxel size = 64x fewer interior voxels), while the outer boundary shell (-T_shell <= phi_sdf <= 0) is voxelized and refined at fine resolution with conformal shell mesh subdivision.",
    },
    ParameterDoc {
        key: "eikonal.coarse_grid_ratio",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Coarse Interior Grid Ratio",
        units: "× fine voxel",
        default_value: "4×",
        summary: "Multiplier between the coarse interior core voxel size and the fine boundary-shell voxel size.",
        details: "A 4× ratio reduces interior core voxel count and Eikonal sweeping cost by 64× (4^3) while keeping global arrival-time wavefronts, branch saddle detection, and sparse infill layer spacing smooth and accurate.",
    },
    ParameterDoc {
        key: "eikonal.narrow_band_width_voxels",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Narrow-Band Half-Width",
        units: "voxels",
        default_value: "6.0 voxels",
        summary: "Half-width of the active OpenVDB level-set narrow band around the object boundary.",
        details: "Controls how many voxel layers outside the surface are allocated in the sparse OpenVDB tree before flood-filling the interior.",
    },
    ParameterDoc {
        key: "eikonal.solver_iterations",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Eikonal Solver Iterations",
        units: "sweeps",
        default_value: "4",
        summary: "Number of Gauss-Seidel / Fast-Sweeping passes for the anisotropic Eikonal field.",
        details: "More iterations improve convergence when strong surface/point orientation hints or complex multi-branch topologies bend the arrival-time field.",
    },
    ParameterDoc {
        key: "eikonal.curvature_regularization",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Curvature Regularization",
        units: "0.0 .. 1.0",
        default_value: "0.15",
        summary: "Laplacian smoothing weight applied to the target director field before Eikonal integration.",
        details: "Prevents sharp creases or self-intersecting shocks in the conformal slicing isosurfaces when adjacent surface hints disagree.",
    },
    ParameterDoc {
        key: "eikonal.default_surface_hint_depth_mm",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Default Surface Hint Depth",
        units: "mm",
        default_value: "8.0 mm",
        summary: "Default inward influence distance for newly placed surface tangency/orthogonality hints.",
        details: "Determines how deep into the part interior a B-Rep face or mesh patch hint steers the Eikonal gradient (∇τ).",
    },
    ParameterDoc {
        key: "eikonal.default_point_hint_radius_mm",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Default 3D Point Hint Radius",
        units: "mm",
        default_value: "12.0 mm",
        summary: "Default spherical influence radius for newly placed 3D point orientation hints.",
        details: "Within this radius, the Eikonal layer normal is softly steered toward the 3D arrow direction using a smooth Wendland radial falloff.",
    },
    ParameterDoc {
        key: "eikonal.default_hint_weight",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Default Hint Soft Weight",
        units: "0.05 .. 1.0",
        default_value: "0.85",
        summary: "Relative blending strength of newly placed orientation hints against gravity/nozzle defaults.",
        details: "Hints act as soft constraints: they steer the layer normal where feasible but never override hard toolhead clearance or maximum slope limits.",
    },
    ParameterDoc {
        key: "eikonal.enable_overhang_orientation_bias",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Overhang Conformal / Orthogonal Surface Bias",
        units: "",
        default_value: "true",
        summary: "Automatically steers Eikonal layer orientation between orthogonal and conformal based on the gravity-aware overhang field.",
        details: "Uses the local signed overhang angle θ_ovh = arcsin(∇φ_sdf · g_bed) ∈ [-90°, +90°] to bias slicing isosurfaces toward orthogonal (∇τ ⊥ n_surf) on walls and moderate slopes/overhangs (default 30° supported to 60° overhang), and toward conformal (∇τ ∥ ±n_surf) on shallow supported roofs/domes (< -30°) and steep overhangs/ceilings (> +60°).",
    },
    ParameterDoc {
        key: "eikonal.overhang_bias_weight",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Overhang Surface Bias Weight",
        units: "0.0 .. 1.0",
        default_value: "0.75",
        summary: "Soft blending weight of the automatic overhang-driven conformal/orthogonal surface orientation bias.",
        details: "Controls how strongly the boundary overhang regime steers the Eikonal layer normal and 5/6-axis tool orientation before hard tool clearance cone clamping. Explicit user surface or point hints always take precedence where placed.",
    },
    ParameterDoc {
        key: "eikonal.orthogonal_supported_angle_deg",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Orthogonal Band Supported Limit",
        units: "° supported",
        default_value: "30.0°",
        summary: "Supported slope angle from vertical below which the surface bias transitions from conformal to orthogonal.",
        details: "Corresponds to signed overhang angle θ_min = -orthogonal_supported_angle_deg (default -30°, i.e., 30° supported). Upward-facing surfaces flatter than this angle (θ_ovh < -30°, such as shallow domes and top skins) bias toward conformal layers (∇τ ∥ +n_surf) to eliminate stair-stepping, while steeper walls (θ_ovh ≥ -30°) bias toward orthogonal layers (∇τ ⊥ n_surf).",
    },
    ParameterDoc {
        key: "eikonal.orthogonal_overhang_angle_deg",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Orthogonal Band Overhang Limit",
        units: "° overhang",
        default_value: "60.0°",
        summary: "Overhang angle past vertical up to which the surface bias stays orthogonal before transitioning to conformal.",
        details: "Corresponds to signed overhang angle θ_max = +orthogonal_overhang_angle_deg (default +60°, i.e., 60° overhang). Overhangs up to this angle bias toward orthogonal layers (∇τ ⊥ n_surf) so each layer tilts to support the overhang face at 90°, whereas steeper overhangs and bridge ceilings (θ_ovh > +60°) transition smoothly to conformal layers (∇τ ∥ -n_surf).",
    },
    ParameterDoc {
        key: "eikonal.overhang_bias_transition_deg",
        category: "Volumetric Fields & Eikonal Solver",
        title: "Overhang Bias Smooth Transition Width",
        units: "°",
        default_value: "15.0°",
        summary: "Angular width of the C¹ cubic Hermite smoothstep transition between orthogonal and conformal regimes.",
        details: "Blends the orthogonal and conformal target directions smoothly over [θ_min - 0.5·Δθ, θ_min + 0.5·Δθ] and [θ_max - 0.5·Δθ, θ_max + 0.5·Δθ] so the target director field has no angular discontinuities across curved surfaces.",
    },

    // --- Non-Planar Slicing & Variable Layer Height ---
    ParameterDoc {
        key: "slicing.layer_height.target_mm",
        category: "Non-Planar Layer Slicing",
        title: "Target Layer Thickness",
        units: "mm",
        default_value: "0.20 mm",
        summary: "Nominal layer thickness along the local surface normal between consecutive conformal isosurfaces.",
        details: "Because non-planar isosurfaces expand and contract across curved geometry, the bidirectional branch selector chooses isovalues so the mean normal spacing matches this target while respecting [Min, Max] bounds.",
    },
    ParameterDoc {
        key: "slicing.layer_height.min_mm",
        category: "Non-Planar Layer Slicing",
        title: "Minimum Layer Thickness",
        units: "mm",
        default_value: "0.08 mm",
        summary: "Lower bound on local layer thickness where conformal layers pinch or converge.",
        details: "Prevents nozzle scraping or starved extrusion when non-planar layers compress over high-curvature regions.",
    },
    ParameterDoc {
        key: "slicing.layer_height.max_mm",
        category: "Non-Planar Layer Slicing",
        title: "Maximum Layer Thickness",
        units: "mm",
        default_value: "0.32 mm",
        summary: "Upper bound on local layer thickness where conformal layers fan out.",
        details: "Should not exceed ~75-80% of the nozzle diameter so extruded beads maintain adequate inter-layer squish and bond strength.",
    },
    ParameterDoc {
        key: "slicing.layer_height.first_layer_mm",
        category: "Non-Planar Layer Slicing",
        title: "First Layer Thickness",
        units: "mm",
        default_value: "0.20 mm",
        summary: "Target thickness of the initial build-plate adhesion layer.",
        details: "A slightly thicker first layer (e.g. 0.20-0.25 mm for a 0.4 mm nozzle) tolerates minor bed unevenness and improves first-layer adhesion.",
    },
    ParameterDoc {
        key: "slicing.layer_height.top_layer_mm",
        category: "Non-Planar Layer Slicing",
        title: "Top Layer Thickness",
        units: "mm",
        default_value: "0.20 mm",
        summary: "Target thickness for terminal top surface layers of each branch.",
        details: "Reducing top layer thickness produces smoother conformal top skins with finer step resolution.",
    },
    ParameterDoc {
        key: "slicing.max_nonplanar_slope_deg",
        category: "Non-Planar Layer Slicing",
        title: "Maximum Non-Planar Slope Angle",
        units: "deg (°)",
        default_value: "35.0°",
        summary: "Maximum allowed angle between the isosurface normal (∇τ) and the nozzle axis.",
        details: "Clamps the Eikonal gradient cone so conformal layers never tilt steeper than your nozzle/fan shroud clearance allows.",
    },

    // --- Shell & Toolpath Generation ---
    ParameterDoc {
        key: "slicing.shell_thickness_mm",
        category: "Shell & Toolpath Generation",
        title: "Conformal 3D Shell Thickness",
        units: "mm",
        default_value: "0.80 mm",
        summary: "True 3D Euclidean thickness of the dense outer skin measured via the object's signed distance field (-φ_sdf <= T_shell).",
        details: "Unlike planar slicers that count top/bottom layers vertically, Manifold2 enforces a uniform 3D shell thickness everywhere (roofs, floors, and sloped walls) using the OpenVDB SDF.",
    },
    ParameterDoc {
        key: "slicing.inner_wall_count",
        category: "Shell & Toolpath Generation",
        title: "Inner Wall Loop Count",
        units: "loops",
        default_value: "2",
        summary: "Number of concentric inner perimeter loops printed inside the outer wall.",
        details: "Inner walls back up the outer perimeter and provide structural stiffness before transitioning to solid or sparse infill.",
    },
    ParameterDoc {
        key: "slicing.min_extrusion_length_mm",
        category: "Shell & Toolpath Generation",
        title: "Minimum Extrusion Path Length",
        units: "mm",
        default_value: "0.80 mm",
        summary: "Shortest isolated gap-fill or infill segment that will be emitted instead of culled.",
        details: "Filters out microscopic islands shorter than ~2x nozzle diameter that would otherwise cause blobs and unnecessary retractions.",
    },
    ParameterDoc {
        key: "slicing.solid_infill_pattern",
        category: "Shell & Toolpath Generation",
        title: "Solid Shell Infill Pattern",
        units: "",
        default_value: "Monotonic",
        summary: "Toolpath pattern used inside the dense 3D shell band (-φ_sdf <= T_shell).",
        details: "Monotonic lines alternate direction by 90° each layer for flat, ridge-free skins; Concentric follows the 3D shell boundary inward.",
    },
    ParameterDoc {
        key: "slicing.sparse_infill_pattern",
        category: "Shell & Toolpath Generation",
        title: "Sparse Core Infill Pattern",
        units: "",
        default_value: "Gyroid (3D TPMS)",
        summary: "Volumetric infill pattern inside the core interior (-φ_sdf > T_shell).",
        details: "Supports true 3D Triply Periodic Minimal Surfaces (Gyroid, Schwarz-P, Schwarz-D) intersected directly with curved non-planar isosurfaces, as well as Rectilinear, Grid, Triangles, Honeycomb, and Cubic.",
    },
    ParameterDoc {
        key: "slicing.infill_density_percent",
        category: "Shell & Toolpath Generation",
        title: "Sparse Infill Density",
        units: "%",
        default_value: "20.0%",
        summary: "Target volume fraction of sparse infill in the core interior.",
        details: "Determines the spatial period/pitch of TPMS lattices or line spacing of 2D/3D infill patterns. Set to 0% for hollow parts or 100% for solid parts.",
    },
    ParameterDoc {
        key: "slicing.infill_angle_deg",
        category: "Shell & Toolpath Generation",
        title: "Base Infill Orientation Angle",
        units: "deg (°)",
        default_value: "45.0°",
        summary: "In-plane rotation angle of infill lattices and monotonic/rectilinear rasters in bed XY coordinates.",
        details: "Rotates the infill coordinate frame around the bed Z axis to align structural ribs with primary load directions.",
    },
    ParameterDoc {
        key: "slicing.line_widths",
        category: "Shell & Toolpath Generation",
        title: "Arachne-Style Variable Line Widths (Target / Min / Max)",
        units: "mm",
        default_value: "0.42 / 0.28 / 0.56 mm",
        summary: "Target, minimum, and maximum extruded bead widths for Outer Walls, Inner Walls, Solid Infill, and Sparse Infill.",
        details: "Medial-axis adaptive bead sizing dynamically widens or narrows walls within [Min, Max] to fill thin wedges without leaving voids or requiring tiny gap-fill fragments.",
    },
    ParameterDoc {
        key: "slicing.overhang_angle_threshold_deg",
        category: "Shell & Toolpath Generation",
        title: "Overhang Angle Threshold",
        units: "deg (°)",
        default_value: "45.0°",
        summary: "Overhang angle from vertical (derived from the gravity-aware overhang field) above which unsupported extrusions use Bridging or Wave Overhangs.",
        details: "Evaluated against the true bed-coordinate gravity direction via the OpenVDB overhang field.",
    },
    ParameterDoc {
        key: "slicing.max_bridge_length_mm",
        category: "Shell & Toolpath Generation",
        title: "Maximum Bridge Length",
        units: "mm",
        default_value: "40.0 mm",
        summary: "Maximum 3D straight-line span length for circular Bridging extrusions across unsupported gaps.",
        details: "Unsupported spans with accessible supported anchors on both sides within this distance are bridged with straight-line circular extrusions; longer or one-sided spans use inside-out Wave Overhangs.",
    },
    ParameterDoc {
        key: "slicing.wave_overhang_overlap_fraction",
        category: "Shell & Toolpath Generation",
        title: "Wave Overhang Overlap Fraction",
        units: "ratio",
        default_value: "0.08 (8%)",
        summary: "Lateral overlap with the adjacent existing extrusion as a fraction of standard line width.",
        details: "Wave overhang passes are ordered inside-out and maintain continuous contact at this overlap even when line width changes or the 3D centerline rotates around the existing extrusion, with extrusion volume scaled accordingly.",
    },
    ParameterDoc {
        key: "slicing.enable_support_generation",
        category: "Support Generation",
        title: "Generate Support Structures",
        units: "bool",
        default_value: "true",
        summary: "Generates gravity-aligned support base and support interface toolpaths for regions unreachable by the constrained order field.",
        details: "Detects floating islands, downward-hanging local minima, and non-bridgeable overhangs that exceed the constrained Eikonal layer cone, and projects support columns along gravity (+g_bed) to the build plate or lower part surface.",
    },
    ParameterDoc {
        key: "slicing.seed_order_field_from_supports",
        category: "Support Generation",
        title: "Seed Order Field from Support Contacts",
        units: "bool",
        default_value: "true",
        summary: "Seeds the constrained Eikonal order field τ(x) at unreachable support contact patches.",
        details: "Anchors unreachable boundary overhangs and hanging islands to their anti-gravity height so the constrained order field propagates causally upward from the support interface.",
    },
    ParameterDoc {
        key: "slicing.support_only_unreachable_and_bridges",
        category: "Support Generation",
        title: "Only Support Unreachable Regions & Long Bridges",
        units: "bool",
        default_value: "false",
        summary: "Restricts support generation to regions that are causally unreachable by the constrained order field (floating islands / downward-hanging local minima) and two-pier bridges exceeding `Max Bridge Length`.",
        details: "When enabled, causally reachable one-sided overhangs and cantilevers are left unsupported so they can be printed via non-planar conformal layers or inside-out wave overhangs.",
    },
    ParameterDoc {
        key: "slicing.support_z_gap_mm",
        category: "Support Generation",
        title: "Support Z Contact Gap",
        units: "mm",
        default_value: "0.20 mm",
        summary: "Vertical separation along anti-gravity between the support interface/floor and the object surface.",
        details: "Allows supports to detach cleanly from the printed part without fusing.",
    },
    ParameterDoc {
        key: "slicing.support_xy_gap_mm",
        category: "Support Generation",
        title: "Support XY Clearance Gap",
        units: "mm",
        default_value: "0.50 mm",
        summary: "Minimum 3D signed-distance clearance between support columns and adjacent part walls.",
        details: "Prevents support scanlines from colliding with or scarring vertical/sloped object walls.",
    },
    ParameterDoc {
        key: "slicing.support_interface_layers",
        category: "Support Generation",
        title: "Support Interface Layers",
        units: "layers",
        default_value: "2",
        summary: "Number of dense support interface layers printed immediately below overhanging contact roofs.",
        details: "Provides a smooth, closely-spaced support roof beneath unreachable overhangs while keeping the lower support base coarse and fast to print.",
    },
    ParameterDoc {
        key: "slicing.support_base_spacing_mm",
        category: "Support Generation",
        title: "Support Base Line Spacing",
        units: "mm",
        default_value: "2.50 mm",
        summary: "Centerline spacing for coarse support base scanlines.",
        details: "Larger spacing reduces support material consumption and print time.",
    },
    ParameterDoc {
        key: "slicing.support_interface_spacing_mm",
        category: "Support Generation",
        title: "Support Interface Line Spacing",
        units: "mm",
        default_value: "0.60 mm",
        summary: "Centerline spacing for dense support interface layers directly beneath the part.",
        details: "Tight spacing supports overhanging perimeters and skins cleanly.",
    },
    ParameterDoc {
        key: "gcode.bridge_speed_mms",
        category: "Speeds & Accelerations",
        title: "Bridge Print Speed",
        units: "mm/s",
        default_value: "35.0 mm/s",
        summary: "Feedrate for straight-line circular Bridge extrusions.",
        details: "Controlled bridging speed keeps the free-spanning circular filament under steady tension while solidifying in the cooling airflow.",
    },
    ParameterDoc {
        key: "gcode.wave_overhang_speed_mms",
        category: "Speeds & Accelerations",
        title: "Wave Overhang Print Speed",
        units: "mm/s",
        default_value: "30.0 mm/s",
        summary: "Feedrate for inside-out Wave Overhang extrusions.",
        details: "Moderate speed gives the laterally-overlapping bead time to bond and solidify against the adjacent supported extrusion.",
    },
    ParameterDoc {
        key: "gcode.bridge_flow_multiplier",
        category: "Filament, Multipliers & Cooling",
        title: "Bridge Extrusion Multiplier",
        units: "ratio",
        default_value: "0.95",
        summary: "Dedicated flow multiplier for circular Bridge extrusions, applied in addition to global EM, die swell, and PA multipliers.",
        details: "Slight under-extrusion (< 1.0) keeps free-air circular bridge strands taut and reduces sag across unsupported gaps.",
    },
    ParameterDoc {
        key: "gcode.wave_overhang_flow_multiplier",
        category: "Filament, Multipliers & Cooling",
        title: "Wave Overhang Extrusion Multiplier",
        units: "ratio",
        default_value: "1.00",
        summary: "Dedicated flow multiplier for Wave Overhang extrusions, applied in addition to the overlap volume adjustment, global EM, die swell, corner, and PA multipliers.",
        details: "Scales the overlap-adjusted extrusion volume of inside-out Wave Overhang passes.",
    },
    ParameterDoc {
        key: "gcode.bridge_fan_speed_percent",
        category: "Filament, Multipliers & Cooling",
        title: "Bridge Cooling Fan Speed",
        units: "%",
        default_value: "100%",
        summary: "Part cooling fan duty cycle during circular Bridge extrusions.",
        details: "Emits M106 before bridge spans to rapidly solidify free-spanning circular extrusions.",
    },
    ParameterDoc {
        key: "gcode.wave_overhang_fan_speed_percent",
        category: "Filament, Multipliers & Cooling",
        title: "Wave Overhang Cooling Fan Speed",
        units: "%",
        default_value: "100%",
        summary: "Part cooling fan duty cycle during inside-out Wave Overhang extrusions.",
        details: "Emits M106 before wave overhang passes to prevent curling or sagging on unsupported perimeters.",
    },

    // --- Filament, Multipliers & Cooling ---
    ParameterDoc {
        key: "gcode.wall_print_order",
        category: "G-code General & Filament",
        title: "Wall Printing Order",
        units: "",
        default_value: "Inner -> Outer -> Backing (In-Out-In)",
        summary: "Sequence in which concentric perimeter loops are extruded on each layer.",
        details: "In-Out-In prints deep inner walls [n..2], then the visible outer wall [0] unconfined for maximum dimensional accuracy, and finally backing wall [1] to lock the outer wall in place. Outside-In prints [0..n] sequentially.",
    },
    ParameterDoc {
        key: "gcode.emit_exclude_object",
        category: "G-code General & Filament",
        title: "Emit Klipper EXCLUDE_OBJECT Commands",
        units: "bool",
        default_value: "true",
        summary: "Emits Klipper EXCLUDE_OBJECT_DEFINE, EXCLUDE_OBJECT_START, and EXCLUDE_OBJECT_END macros.",
        details: "Allows cancelling individual failed parts mid-print in Mainsail/Fluidd/KlipperScreen using the convex-hull footprint polygon.",
    },
    ParameterDoc {
        key: "gcode.emit_thumbnails",
        category: "G-code General & Filament",
        title: "Embed Moonraker 3D Preview Thumbnails",
        units: "bool",
        default_value: "true",
        summary: "Renders 32x32 and 300x300 isometric 3D toolpath PNG thumbnails into the G-code header.",
        details: "Formatted as PrusaSlicer/Moonraker `; thumbnail begin WxH LEN` base64 blocks so Mainsail, Fluidd, and KlipperScreen display a 3D preview of the non-planar toolpaths.",
    },
    ParameterDoc {
        key: "gcode.filament_type",
        category: "Filament, Multipliers & Cooling",
        title: "Filament Material Type",
        units: "",
        default_value: "PLA",
        summary: "Material identifier written to Moonraker G-code metadata (e.g. PLA, PETG, ABS, ASA, TPU, PA-CF).",
        details: "Used by Moonraker, Mainsail, Fluidd, and Spoolman to match filament spools and display material badges.",
    },
    ParameterDoc {
        key: "gcode.filament_diameter_mm",
        category: "Filament, Multipliers & Cooling",
        title: "Filament Diameter",
        units: "mm",
        default_value: "1.75 mm",
        summary: "Nominal diameter of the raw feedstock filament entering the extruder.",
        details: "Determines the cross-sectional area A_fil = π*(D_fil/2)² used to convert volumetric extrusion into E-axis stepper distance.",
    },
    ParameterDoc {
        key: "gcode.filament_density_g_cm3",
        category: "Filament, Multipliers & Cooling",
        title: "Filament Density",
        units: "g/cm³",
        default_value: "1.24 g/cm³",
        summary: "Bulk mass density of the filament material in grams per cubic centimeter.",
        details: "Used to compute total part weight in grams for the UI and Moonraker metadata (typical values: PLA 1.24, PETG 1.27, ABS 1.04, ASA 1.07, TPU 1.21).",
    },
    ParameterDoc {
        key: "gcode.extrusion_multiplier",
        category: "Filament, Multipliers & Cooling",
        title: "Global Extrusion Multiplier",
        units: "ratio (×)",
        default_value: "1.00×",
        summary: "Global volumetric flow scaling factor applied to all extruding moves.",
        details: "Use to calibrate slight over- or under-extrusion for specific filament batches (e.g. 0.95-0.98 for moisture-sensitive or foaming filaments).",
    },
    ParameterDoc {
        key: "gcode.first_layer_flow_multiplier",
        category: "Filament, Multipliers & Cooling",
        title: "First-Layer Flow Multiplier",
        units: "ratio (×)",
        default_value: "1.02×",
        summary: "Additional flow multiplier applied on the first layer (layer_index == 0).",
        details: "Slightly increasing first-layer flow (e.g. 1.02-1.05×) helps fill micro-texture on textured PEI plates for reliable bed adhesion.",
    },
    ParameterDoc {
        key: "gcode.fan_speed_percent",
        category: "Filament, Multipliers & Cooling",
        title: "Part Cooling Fan Speed",
        units: "%",
        default_value: "100%",
        summary: "Steady-state part cooling fan duty cycle (M106 S0..255) after the initial fan-disabled layers.",
        details: "Crucial for non-planar overhangs and bridges. Typically 100% for PLA, 30-50% for PETG, and 0-20% for ABS/ASA inside an enclosure.",
    },
    ParameterDoc {
        key: "gcode.first_layer_fan_speed_percent",
        category: "Filament, Multipliers & Cooling",
        title: "First-Layer Fan Speed",
        units: "%",
        default_value: "0%",
        summary: "Part cooling fan speed applied during the initial `Disable Fan First Layers` count.",
        details: "Usually kept at 0% (emits M107) on the first layer so the polymer bonds to the heated bed without thermal shock or warping.",
    },
    ParameterDoc {
        key: "gcode.disable_fan_first_layers",
        category: "Filament, Multipliers & Cooling",
        title: "Use First-Layer Fan Speed for First N Layers",
        units: "layers",
        default_value: "1 layer",
        summary: "Number of initial layers that use `First-Layer Fan Speed` before ramping to `Part Cooling Fan Speed`.",
        details: "Setting to 1-3 layers prevents base-corner lifting before full cooling kicks in.",
    },

    // --- Speeds, Accelerations & Stepper Kinematics ---
    ParameterDoc {
        key: "gcode.wall_speeds_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Outer & Inner Wall Speeds",
        units: "mm/s",
        default_value: "65 / 100 mm/s",
        summary: "Target linear feedrates for visible outer perimeters and structural inner perimeters.",
        details: "Moderate outer wall speeds minimize ringing and gloss variations; inner walls can run faster while remaining capped by Max Volumetric Flow.",
    },
    ParameterDoc {
        key: "gcode.infill_speeds_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Solid & Sparse Infill Speeds",
        units: "mm/s",
        default_value: "130 / 160 mm/s",
        summary: "Target linear feedrates for dense 3D shell infill and interior sparse infill.",
        details: "Automatically throttled if `width × height × speed` would exceed `Max Volumetric Flow (mm³/s)`.",
    },
    ParameterDoc {
        key: "gcode.gap_fill_speed_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Gap Fill Speed",
        units: "mm/s",
        default_value: "60 mm/s",
        summary: "Target linear feedrate for narrow medial-axis gap-fill beads.",
        details: "Kept conservative because gap-fill paths often have rapidly varying width and high curvature.",
    },
    ParameterDoc {
        key: "gcode.first_layer_and_travel_speed_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "First-Layer & Rapid Travel Speeds",
        units: "mm/s",
        default_value: "45 / 250 mm/s",
        summary: "Feedrate for all extruding moves on the first layer and non-extruding rapid G0 moves.",
        details: "Slower first-layer speeds ensure consistent melt wetting on the build plate; high travel speeds minimize ooze time during non-extruding hops.",
    },
    ParameterDoc {
        key: "gcode.z_hop_speed_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Nozzle Clearance Lift (Z-Hop) Speed",
        units: "mm/s",
        default_value: "25 mm/s",
        summary: "Feedrate for clearance lifts along the local nozzle axis (`z_nozzle`) before/after travels.",
        details: "In Manifold2, clearance hops lift along the local nozzle axis (Rule 1), not hardcoded +Z. Keep within leadscrew/z-belt torque limits.",
    },
    ParameterDoc {
        key: "gcode.accelerations_mms2",
        category: "Speeds, Accelerations & Kinematics",
        title: "Feature Accelerations (Outer Wall / Inner Wall / Infill / Travel)",
        units: "mm/s²",
        default_value: "2500 / 4500 / 6500 / 9000 mm/s²",
        summary: "Per-feature acceleration limits emitted via deadband-compacted `SET_VELOCITY_LIMIT ACCEL=...`.",
        details: "Lower outer-wall acceleration preserves surface finish below input-shaper vibration thresholds, while higher inner-wall, infill, and travel accelerations reduce print time.",
    },
    ParameterDoc {
        key: "gcode.square_corner_velocity_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Square Corner Velocity (SCV)",
        units: "mm/s",
        default_value: "5.0 mm/s",
        summary: "Klipper instantaneous junction velocity at a 90° corner.",
        details: "Controls both the chained lookahead planner's junction speeds and the centripetal junction smoothing arc radius R_eff used for corner over-extrusion compensation.",
    },
    ParameterDoc {
        key: "gcode.minimum_cruise_ratio",
        category: "Speeds, Accelerations & Kinematics",
        title: "Minimum Cruise Ratio",
        units: "0.0 .. 0.90",
        default_value: "0.50",
        summary: "Klipper `minimum_cruise_ratio` enforcing a minimum constant-speed fraction on short back-to-back moves.",
        details: "Prevents pure triangular velocity profiles on dense non-planar polylines that would otherwise excite high-frequency resonance.",
    },
    ParameterDoc {
        key: "gcode.max_volumetric_flow_mm3s",
        category: "Speeds, Accelerations & Kinematics",
        title: "Maximum Hotend Volumetric Flow",
        units: "mm³/s",
        default_value: "18.0 mm³/s",
        summary: "Hard ceiling on volumetric extrusion rate Q = width × height × speed.",
        details: "Because non-planar layers vary continuously in both thickness (h) and Arachne width (w), the planner dynamically caps segment speed to `Q_max / (w × h)` so the hotend melt zone is never outrun.",
    },
    ParameterDoc {
        key: "gcode.use_stepper_dynamics",
        category: "Speeds, Accelerations & Kinematics",
        title: "Stepper Back-EMF Torque Roll-off",
        units: "bool",
        default_value: "true",
        summary: "Models high-speed stepper motor back-EMF torque drop (`a_eff(v) = a_0 * (1 - v / v_max_stepper)`).",
        details: "Prevents skipped steps during fast travels and high-flow infills by replacing constant acceleration with exponential velocity-dependent acceleration envelopes.",
    },
    ParameterDoc {
        key: "gcode.max_stepper_speed_mms",
        category: "Speeds, Accelerations & Kinematics",
        title: "Stepper Pull-Out Velocity (v_max_stepper)",
        units: "mm/s",
        default_value: "400 mm/s",
        summary: "Asymptotic stepper velocity where back-EMF reduces available acceleration torque to zero.",
        details: "Used when `Stepper Back-EMF Torque Roll-off` is enabled to bound reachable peak speeds on long segments.",
    },
    ParameterDoc {
        key: "gcode.z_travel_penalty",
        category: "Speeds, Accelerations & Kinematics",
        title: "Z-Axis Travel Distance Penalty",
        units: "weight",
        default_value: "4.0×",
        summary: "Penalty multiplier applied to vertical ΔZ displacements during nearest-neighbor path ordering and seam placement.",
        details: "Discourages unnecessary up-and-down non-planar jumps across curved layers when ordering islands and infill lines.",
    },

    // --- Seams, 3D Scarf Joint, Wipe & Retraction ---
    ParameterDoc {
        key: "gcode.enable_scarf_joint",
        category: "Seams, Scarf Joint & Retraction",
        title: "3D Conformal Scarf Joint",
        units: "bool",
        default_value: "true",
        summary: "Replaces abrupt loop start/stop seams with a 3D helical ramp along the local nozzle axis (`z_nozzle`).",
        details: "Ramps bead height and flow linearly over `Scarf Length (mm)` at the start and end of closed loops so the overlap blends seamlessly.",
    },
    ParameterDoc {
        key: "gcode.scarf_length_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Scarf Joint Ramp Length",
        units: "mm",
        default_value: "8.0 mm",
        summary: "Arc length along the closed perimeter over which the 3D scarf joint ramps from `Scarf Start Height Ratio` to 100%.",
        details: "Longer ramps (6-12 mm) make outer-wall seams virtually invisible on smooth non-planar surfaces.",
    },
    ParameterDoc {
        key: "gcode.scarf_start_height_ratio",
        category: "Seams, Scarf Joint & Retraction",
        title: "Scarf Start Height Ratio",
        units: "0.02 .. 0.50",
        default_value: "0.12",
        summary: "Initial fraction of full layer height at the leading tip of the scarf ramp.",
        details: "Starts the scarf wedge at `h_start = ratio × h_layer` and finishes the loop at `(1 - ratio) × h_layer` so the two wedges sum to 100% height.",
    },
    ParameterDoc {
        key: "gcode.seam_gap_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Loop Seam Coast Gap",
        units: "mm",
        default_value: "0.06 mm",
        summary: "Arc length trimmed from the end of a closed loop before the seam to prevent overlap bulges.",
        details: "Compensates for residual nozzle pressure at loop closure.",
    },
    ParameterDoc {
        key: "gcode.pre_retract_taper_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Pre-Retract Flow Taper Length",
        units: "mm",
        default_value: "1.0 mm",
        summary: "Distance before the end of an extrusion chain over which flow tapers slightly to bleed melt pressure.",
        details: "Reduces stringing and end-of-path blobs before a retraction + wipe.",
    },
    ParameterDoc {
        key: "gcode.wipe_length_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Post-Extrusion Wipe Distance",
        units: "mm",
        default_value: "1.5 mm",
        summary: "Non-extruding wipe travel distance along the path before lifting the nozzle.",
        details: "On closed loops, wipes forward across the seam; on open paths, wipes backward along the just-printed trace so the nozzle never protrudes outside the part.",
    },
    ParameterDoc {
        key: "gcode.retract_length_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Nominal Retraction Length",
        units: "mm",
        default_value: "0.80 mm",
        summary: "Baseline filament pullback distance before travel moves exceeding `Min Travel for Retract`.",
        details: "Adapted dynamically based on residual melt-zone pressure (`K_pa × Q`). Typical values: 0.4-1.0 mm for direct-drive, 3.0-6.0 mm for Bowden.",
    },
    ParameterDoc {
        key: "gcode.retract_speed_mms",
        category: "Seams, Scarf Joint & Retraction",
        title: "Retract & Unretract Speed",
        units: "mm/s",
        default_value: "45 mm/s",
        summary: "Extruder feedrate used for G1 E-retract and E+unretract moves.",
        details: "Fast enough to break filament strands cleanly without grinding the filament drive gear.",
    },
    ParameterDoc {
        key: "gcode.min_travel_for_retract_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Minimum Travel Distance for Retraction",
        units: "mm",
        default_value: "1.50 mm",
        summary: "Shortest non-extruding travel move that triggers a retraction and nozzle clearance hop.",
        details: "Suppresses redundant retractions between adjacent walls or tightly spaced infill lines.",
    },
    ParameterDoc {
        key: "gcode.z_hop_mm",
        category: "Seams, Scarf Joint & Retraction",
        title: "Nozzle Clearance Hop Height",
        units: "mm",
        default_value: "0.40 mm",
        summary: "Distance lifted along the local nozzle axis (`z_nozzle`) during retracted travel moves.",
        details: "Essential for non-planar printing so the nozzle clears curved crests between islands.",
    },
    ParameterDoc {
        key: "gcode.ooze_relaxation",
        category: "Seams, Scarf Joint & Retraction",
        title: "Thermal Ooze & Die-Swell Unretract Relaxation",
        units: "mm / s",
        default_value: "-0.02 mm / τ = 1.50 s",
        summary: "Adjusts unretract prime length as an exponential function of travel duration (`1 - exp(-t / τ)`).",
        details: "Short hops slightly under-prime to avoid viscoelastic swell blobs; long travels decay toward full prime as melt pressure relaxes.",
    },

    // --- Non-Newtonian Rheology, Die Swell & Transient Flow ---
    ParameterDoc {
        key: "gcode.enable_flat_nozzle_comp",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Flat-Nozzle Annular Slope Clearance",
        units: "bool",
        default_value: "true",
        summary: "Elevates the nozzle tip along `z_nozzle` on sloped outer walls (`Δz = R_flat * sin(α) * (1 - cos(α))`).",
        details: "Prevents the flat brass land surrounding the nozzle orifice from gouging the uphill side of a sloped conformal layer.",
    },
    ParameterDoc {
        key: "gcode.pa_low_high_s",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Shear-Thinning Pressure Advance (k_low / k_high)",
        units: "s",
        default_value: "0.042 s / 0.024 s",
        summary: "Pressure Advance time constants at low and high reference volumetric flow rates.",
        details: "Polymer melts are shear-thinning (power-law index n < 1): effective PA decreases at higher volumetric flows. Manifold2 interpolates k(Q) via a power law between Q_low and Q_high.",
    },
    ParameterDoc {
        key: "gcode.pa_ref_flows_mm3s",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "PA Reference Flow Rates (Q_low / Q_high)",
        units: "mm³/s",
        default_value: "2.0 / 14.0 mm³/s",
        summary: "Low and high volumetric flow calibration anchor points for power-law Pressure Advance and Die Swell.",
        details: "Defines the flow range `[Q_low, Q_high]` over which `k_low..k_high` and the low/high-flow die-swell multipliers are calibrated.",
    },
    ParameterDoc {
        key: "gcode.swell_multipliers",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Low-Flow & High-Flow Die-Swell Extrusion Multipliers",
        units: "ratio (×)",
        default_value: "0.971× (low) / 0.909× (high)",
        summary: "Reciprocal viscoelastic die-swell extrusion multipliers (`1 / swell_low` and `1 / swell_high`) at `Q_low` and `Q_high`.",
        details: "Viscoelastic polymer melts expand radially exiting the nozzle orifice (die swell increases with shear rate). Expressing die swell as reciprocal extrusion multipliers (< 1.0×) scales down flow and insets outer walls at high shear rates to prevent over-extrusion.",
    },
    ParameterDoc {
        key: "gcode.enable_dynamic_pa",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Emit Dynamic SET_PRESSURE_ADVANCE Commands",
        units: "bool",
        default_value: "true",
        summary: "Emits deadband-compacted `SET_PRESSURE_ADVANCE ADVANCE=...` updates as segment volumetric flow changes.",
        details: "Automatically adapts Klipper's Pressure Advance across slow outer walls, variable-thickness non-planar layers, and high-flow infill.",
    },
    ParameterDoc {
        key: "gcode.enable_slicer_pa_subdivision",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Slicer-Side Trapezoidal PA Subdivision",
        units: "bool",
        default_value: "false",
        summary: "Subdivides each segment into entry-accel, cruise, and exit-decel sub-segments with pre-compensated E values (`ADVANCE=0`).",
        details: "Useful for multi-axis controllers or firmwares without native nonlinear PA; disables firmware PA (`SET_PRESSURE_ADVANCE ADVANCE=0.0000`) when active.",
    },
    ParameterDoc {
        key: "gcode.corner_flow_comp",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Klipper SCV Corner Flow Compensation",
        units: "bool / ratio",
        default_value: "true (ratio 0.85)",
        summary: "Deducts both geometric inner-corner diamond overlap and Klipper SCV centripetal shortcut volume at corners.",
        details: "Eliminates corner bulging on sharp perimeter turns by reducing extrusion on the segments immediately entering and exiting each corner.",
    },
    ParameterDoc {
        key: "gcode.transient_pressure_comp",
        category: "Rheology, Die Swell & Dynamic PA",
        title: "Melt-Zone Transient Pressure Compensation",
        units: "bool / exp / min×",
        default_value: "true (exp 0.35, min 0.72×)",
        summary: "First-order lag model of nozzle melt-chamber pressure across variable-width/height segments.",
        details: "When stepping down from high-flow to low-flow segments, temporarily throttles extrusion (`>= Min Multiplier`) while stored melt-zone pressure discharges.",
    },

    // --- Start & End G-code Templates ---
    ParameterDoc {
        key: "gcode.bed_temperature_celsius",
        category: "Start & End G-code Templates",
        title: "Heated Bed Temperature",
        units: "°C",
        default_value: "60 °C",
        summary: "Target build plate temperature substituted into `{bed_temp}` (`M140`/`M190`) and Moonraker metadata.",
        details: "Typical values: 55-65 °C for PLA, 70-85 °C for PETG, 95-110 °C for ABS/ASA.",
    },
    ParameterDoc {
        key: "gcode.templates",
        category: "Start & End G-code Templates",
        title: "Start & End G-code Templates",
        units: "G-code",
        default_value: "Standard Klipper M140/M104/G28/M190/M109",
        summary: "Customizable start and end G-code blocks with `{bed_temp}`, `{nozzle_temp}`, `{tool_id}`, and bounding-box placeholders.",
        details: "Supports `{tool_id}`, `{nozzle_temp}`, `{bed_temp}`, `{nozzle_diameter}`, `{bbox_min_x..z}`, `{bbox_max_x..z}`, `{bbox_center_x..y}`, and `{layer_count}`.",
    },

    // --- Machine Tools & Multi-Axis Kinematics ---
    ParameterDoc {
        key: "tools.nozzle_diameter_mm",
        category: "Machine Tools & Multi-Axis Kinematics",
        title: "Tool Nozzle Orifice Diameter",
        units: "mm",
        default_value: "0.40 mm",
        summary: "Orifice diameter of the active extrusion tool.",
        details: "Sets baseline extrusion widths, minimum path length presets, and flat-nozzle land radius (`R_flat = D_nozzle`).",
    },
    ParameterDoc {
        key: "tools.temperature_celsius",
        category: "Machine Tools & Multi-Axis Kinematics",
        title: "Tool Hotend Temperature",
        units: "°C",
        default_value: "215 °C",
        summary: "Target hotend temperature for the selected tool (`{nozzle_temp}`).",
        details: "Emitted via `M104`/`M109` in the start G-code and recorded in Moonraker metadata.",
    },
    ParameterDoc {
        key: "tools.rotational_axes",
        category: "Machine Tools & Multi-Axis Kinematics",
        title: "5/6-Axis Rotational Toolhead & Max Tilt Angle",
        units: "bool / deg (°)",
        default_value: "true / 45.0°",
        summary: "Indicates whether the toolhead can tilt its nozzle axis (`z_nozzle`) away from the bed normal up to `Max Angle`.",
        details: "Follows AGENTS.md Rule 1: all tool clearance profiles and non-planar offsets are evaluated in the local orthonormal nozzle frame `(x_nozzle, y_nozzle, z_nozzle)`.",
    },
    ParameterDoc {
        key: "tools.clearance_profile_xz",
        category: "Machine Tools & Multi-Axis Kinematics",
        title: "Nozzle Clearance Profile (XZ in Nozzle Frame)",
        units: "mm (X radius, Z height)",
        default_value: "4-point cone profile",
        summary: "Piecewise-linear radial envelope `(X_radius, Z_height)` of the nozzle tip, heater block, and fan shroud.",
        details: "Defined in the local nozzle frame (`Z >= 0` up along `nozzle_axis`, `X >= 0` radial distance) to bound collision-free non-planar slopes.",
    },
    ParameterDoc {
        key: "machine.bed_tilt_deg",
        category: "Machine Tools & Multi-Axis Kinematics",
        title: "5-Axis Build Plate Tilt (Rx / Ry)",
        units: "deg (°)",
        default_value: "0.0° / 0.0°",
        summary: "Tilts the build plate relative to world gravity, updating the bed-coordinate gravity vector `g_bed`.",
        details: "Follows AGENTS.md Rule 2: overhang fields and gravity-aware calculations always query `g_bed` rather than assuming gravity is `(0, 0, -1)`.",
    },

    // --- Print Bed & Auto-Arrangement ---
    ParameterDoc {
        key: "bed.geometry",
        category: "Print Bed & Arrangement",
        title: "Print Bed Shape & Dimensions (Width X / Depth Y / Max Z)",
        units: "mm",
        default_value: "Rectangular 256 × 256 × 256 mm",
        summary: "Physical build volume dimensions and shape (Rectangular or Circular Delta).",
        details: "Used for viewport grid rendering, spiral auto-arrangement bounds, and front-left vs. center G-code coordinate origin translation.",
    },
    ParameterDoc {
        key: "gcode.gcode_origin_at_bed_center",
        category: "Print Bed & Arrangement",
        title: "G-code Origin at Bed Center (0, 0)",
        units: "bool",
        default_value: "false (Front-Left for Rectangular)",
        summary: "Controls whether exported G-code places `(X0, Y0)` at the bed center or at the front-left corner.",
        details: "Standard Cartesian/CoreXY printers (Voron, Bambu, Prusa) home `(0, 0)` at the front-left corner (`[0..W, 0..D]`), whereas Delta printers place `(0, 0)` at the bed center.",
    },
    ParameterDoc {
        key: "arrangement.margins_mm",
        category: "Print Bed & Arrangement",
        title: "Auto-Arrangement Part Clearance & Bed Margin",
        units: "mm",
        default_value: "4.0 mm clearance / 5.0 mm margin",
        summary: "Minimum XY gap between parts and safety margin from the build plate edge during spiral auto-arrangement.",
        details: "Increase `Part Clearance` when printing tall non-planar parts sequentially or when using wide toolhead shrouds.",
    },
];

/// Looks up the structured documentation entry for `key` in [`PARAMETER_DOCS`].
pub fn lookup_parameter_doc(key: &str) -> Option<&'static ParameterDoc> {
    PARAMETER_DOCS.iter().find(|doc| doc.key == key)
}

/// Generates a complete Markdown reference document from [`PARAMETER_DOCS`], grouped by category.
pub fn generate_parameter_docs_markdown() -> String {
    let mut out = String::with_capacity(16 * 1024);
    out.push_str("# Manifold2 Parameter Reference\n\n");
    let mut current_category = "";
    for doc in PARAMETER_DOCS {
        if doc.category != current_category {
            current_category = doc.category;
            out.push_str(&format!("## {current_category}\n\n"));
        }
        out.push_str(&format!("### {} (`{}`)\n", doc.title, doc.key));
        if !doc.units.is_empty() {
            out.push_str(&format!(
                "- **Default**: `{}` ({})\n",
                doc.default_value, doc.units
            ));
        } else {
            out.push_str(&format!("- **Default**: `{}`\n", doc.default_value));
        }
        out.push_str(&format!("- **Summary**: {}\n", doc.summary));
        out.push_str(&format!("- **Details**: {}\n\n", doc.details));
    }
    out
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

    #[test]
    fn test_parameter_docs_catalog_and_markdown_export() {
        assert!(!PARAMETER_DOCS.is_empty());
        for (i, doc) in PARAMETER_DOCS.iter().enumerate() {
            assert!(!doc.key.is_empty());
            assert!(!doc.title.is_empty());
            assert!(!doc.summary.is_empty());
            assert!(!doc.details.is_empty());
            for other in &PARAMETER_DOCS[..i] {
                assert_ne!(
                    doc.key, other.key,
                    "Duplicate ParameterDoc key: {}",
                    doc.key
                );
            }
        }
        assert!(lookup_parameter_doc("gcode.swell_multipliers").is_some());
        assert!(lookup_parameter_doc("gcode.fan_speed_percent").is_some());
        assert!(lookup_parameter_doc("eikonal.auto_voxel_size").is_some());
        assert!(lookup_parameter_doc("eikonal.enable_multi_resolution_grid").is_some());
        assert!(lookup_parameter_doc("eikonal.coarse_grid_ratio").is_some());
        let md = generate_parameter_docs_markdown();
        assert!(md.starts_with("# Manifold2 Parameter Reference"));
        assert!(md.contains("gcode.swell_multipliers"));
    }

    #[test]
    fn test_auto_fine_voxel_size_scaling_with_nozzle_and_layer_height() {
        // Standard 0.40 mm nozzle + 0.20 mm target layer height => 0.25 mm fine voxel size
        let v_std = compute_auto_fine_voxel_size_mm(0.40, 0.20);
        assert!((v_std - 0.25).abs() < 1.0e-5, "Expected 0.25 mm, got {v_std}");

        // 0.60 mm nozzle + 0.30 mm target layer height => 0.375 * 0.60 + 0.15 = 0.375 -> 0.38 mm
        let v_06 = compute_auto_fine_voxel_size_mm(0.60, 0.30);
        assert!((v_06 - 0.38).abs() < 1.0e-5, "Expected 0.38 mm, got {v_06}");

        // 0.80 mm nozzle + 0.40 mm target layer height => 0.375 * 0.80 + 0.20 = 0.50 mm
        let v_08 = compute_auto_fine_voxel_size_mm(0.80, 0.40);
        assert!((v_08 - 0.50).abs() < 1.0e-5, "Expected 0.50 mm, got {v_08}");

        let mut cfg = SlicerConfig::default();
        assert!(cfg.eikonal.auto_voxel_size);
        assert!(cfg.eikonal.enable_multi_resolution_grid);
        assert_eq!(cfg.eikonal.coarse_grid_ratio, 4);
        assert!((cfg.eikonal.voxel_size_mm - 0.25).abs() < 1.0e-5);
        cfg.active_tool_index = 1; // T1 - 0.6mm High-Flow
        cfg.slicing.layer_height.target_mm = 0.30;
        cfg.sync_auto_voxel_size();
        assert!((cfg.eikonal.voxel_size_mm - 0.38).abs() < 1.0e-5);
    }
}
