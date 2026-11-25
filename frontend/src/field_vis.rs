use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::tasks::IoTaskPool;

use crate::bridge::ffi::{
    self, ExtrusionPath, FieldSlice2D, FieldStats, LayerHeightSpec, LineWidthSpec,
    ObjectToolpathsResult, SelectedIsosurfacesResult, ShellPathSpec, SupportSpec,
};
use crate::config::{NozzleFrame, ObjectFilamentMode, SlicerConfig};
use crate::scene::{BuildPlateState, CadObjectEntity};

/// Cross-section cutting plane orientation in the object's coordinate frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SliceAxis {
    X,
    Y,
    #[default]
    Z,
}

impl SliceAxis {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::X => "YZ Plane (X-Slice)",
            Self::Y => "XZ Plane (Y-Slice)",
            Self::Z => "XY Plane (Z-Slice)",
        }
    }
}

/// Controls how a non-SDF isosurface visualizes the region where it intersects the solid object ($\phi_{\text{sdf}} \le 0$).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IsoIntersectionMode {
    /// Highlights the object-intersecting region ($\phi_{\text{sdf}} \le 0$) in vivid colors
    /// and dims the exterior region ($\phi_{\text{sdf}} > 0$) in muted dark slate.
    HighlightInsideDimOutside,
    /// Clips away all exterior triangles ($\phi_{\text{sdf}} > 0$), rendering only the exact
    /// portion of the isosurface that intersects the solid object ($\phi_{\text{sdf}} \le 0$).
    #[default]
    ClipToObjectInterior,
    /// Renders the entire unclipped isosurface uniformly.
    FullUnclipped,
}

impl IsoIntersectionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::HighlightInsideDimOutside => "Highlight Inside (φ <= 0) & Dim Exterior",
            Self::ClipToObjectInterior => "Clip to Object Interior Only (φ <= 0)",
            Self::FullUnclipped => "Full Unclipped Isosurface",
        }
    }
}

/// Colormap palettes for 2D scalar field slice visualization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FieldColormap {
    /// Blue ($\phi < 0$) -> Crisp White ($\phi = 0$) -> Red ($\phi > 0$), ideal for SDFs & overhangs.
    #[default]
    CoolWarmSigned,
    /// Perceptual multi-hue gradient across `[min, max]`, ideal for Eikonal / geodesic / thermal fields.
    ViridisSequential,
    /// High-contrast alternating level-set zebra bands for inspecting iso-contour spacing.
    ZebraIsoBands,
}

impl FieldColormap {
    pub fn label(self) -> &'static str {
        match self {
            Self::CoolWarmSigned => "Signed Cool-Warm (Blue / 0=White / Red)",
            Self::ViridisSequential => "Viridis Sequential ([Min .. Max])",
            Self::ZebraIsoBands => "Zebra Iso-Contour Bands",
        }
    }
}

/// Color mode for 3D visualization of the CAD object's outer surface mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ObjectSurfaceColorMode {
    /// Standard solid/translucent CAD material color.
    #[default]
    DefaultCad,
    /// Colors the object surface by overhang angle relative to the queried gravity vector
    /// $\hat{\mathbf{g}}_{\text{bed}}$ in bed coordinates:
    /// $\theta_{\text{overhang}} = \arccos(-\hat{\mathbf{n}}_{\text{bed}} \cdot \hat{\mathbf{g}}_{\text{bed}}) \in [0^\circ, 180^\circ]$
    /// ($0^\circ$ = upward top surface, $90^\circ$ = vertical wall, $135^\circ$ = $45^\circ$ overhang, $180^\circ$ = downward bridge).
    Overhang,
    /// Colors the object surface by the angle between the local tool/nozzle axis $\hat{\mathbf{z}}_{\text{nozzle}}$
    /// (from the solved 5/6-axis `toolAxisGrid` field when available, or the queried machine `NozzleFrame`)
    /// and the outward surface normal $\hat{\mathbf{n}}_{\text{bed}}$:
    /// $\theta_{\text{tool}} = \arccos(\hat{\mathbf{z}}_{\text{nozzle}} \cdot \hat{\mathbf{n}}_{\text{bed}}) \in [0^\circ, 180^\circ]$
    /// ($0^\circ$ = nozzle perpendicular/conformal to surface, $90^\circ$ = nozzle tangent to surface, $180^\circ$ = opposite).
    ToolAngleFromNormal,
    /// Colors the object surface by the `"support_needed"` field ($0\%$ = reachable by constrained order
    /// field $\tau(\mathbf{x})$ or valid bridge; $100\%$ = unreachable floating island, downward local minimum,
    /// or un-bridged overhang exceeding the constrained tool/clearance cone).
    SupportNeeded,
}

impl ObjectSurfaceColorMode {
    pub const ALL: [Self; 4] = [
        Self::DefaultCad,
        Self::Overhang,
        Self::ToolAngleFromNormal,
        Self::SupportNeeded,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::DefaultCad => "Default CAD Material",
            Self::Overhang => "Overhang vs. Gravity (0°=Up / 90°=Vertical / 180°=Down)",
            Self::ToolAngleFromNormal => "Tool Angle from Surface Normal (0°=⊥ / 90°=∥ / 180°=Opp)",
            Self::SupportNeeded => "Unreachable / Support Needed (0%=Supported / 100%=Needs Support)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::DefaultCad => "Default CAD Material",
            Self::Overhang => "Overhang vs. Gravity (°)",
            Self::ToolAngleFromNormal => "Tool Angle from Normal (°)",
            Self::SupportNeeded => "Unreachable / Support Needed (%)",
        }
    }
}

/// Color mode for 3D visualization of the selected slicing isosurfaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelectedLayersColorMode {
    /// Colors each layer by its mean physical layer thickness between `[min_mm, target_mm, max_mm]`
    /// (Blue = thin/min, Emerald Green = target, Amber/Red = thick/max).
    #[default]
    ByLayerThickness,
    /// Colors each layer by its topological Reeb-graph `branch_id`.
    ByBranch,
    /// Alternating warm/cool bands by sequential layer index.
    AlternatingBands,
}

impl SelectedLayersColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::ByLayerThickness => "Layer Thickness (Min=Blue / Target=Green / Max=Amber)",
            Self::ByBranch => "Topological Branch ID",
            Self::AlternatingBands => "Alternating Layer Bands",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ByLayerThickness => "Layer Thickness Δh (mm)",
            Self::ByBranch => "Topological Branch ID",
            Self::AlternatingBands => "Alternating Layer Bands",
        }
    }
}

/// Color mode for 3D visualization of generated non-planar toolpaths and G-code variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolpathColorMode {
    /// Colors paths by feature type:
    /// - Outer Wall: Coral/Orange
    /// - Inner Wall: Emerald/Lime Green
    /// - Solid Infill: Sky Blue/Cyan
    /// - Sparse Infill: Warm Gold/Amber
    /// - Gap Fill: Vivid Magenta
    /// - Scarf / Wipe: Violet / Slate
    /// - Travel / Z-Hop: Cyan / Yellow
    #[default]
    ByFeatureType,
    /// Colors each segment by its local line width $W$ (`mm`).
    ByLineWidth,
    /// Colors each segment by its local layer height $H$ (`mm`).
    ByLayerHeight,
    /// Colors each segment by its planned kinematic feedrate speed $v$ (`mm/s`).
    BySpeed,
    /// Colors each segment by its signed tangential acceleration $a$ (`mm/s²`).
    ByAcceleration,
    /// Colors each segment by its actual volumetric flow rate $Q_{\text{actual}}$ (`mm³/s`).
    ByVolumetricFlow,
    /// Colors each segment by its corner + transient pressure flow multiplier $M_{\text{total}}$.
    ByExtrusionMultiplier,
    /// Colors each segment by its dynamic non-Newtonian pressure advance $k_{\text{PA}}$ (`s`).
    ByPressureAdvance,
    /// Colors each segment by its modeled melt-zone pressure $P_{\text{melt}}$ (`MPa` proxy).
    ByMeltZonePressure,
    /// Colors each segment by the angular deviation $\theta_{\text{tool,iso}} = \arccos(\hat{\mathbf{z}}_{\text{nozzle}} \cdot \hat{\mathbf{n}}_{\text{iso}})$ (degrees) between the tool axis and the local slicing isosurface normal.
    ByToolIsoNormalAngle,
    /// Highlights segments where the tool axis is actively oriented using the outward object surface normal $\nabla \phi_{\text{sdf}}$ vs. the slicing isosurface normal $\hat{\mathbf{n}}_{\text{iso}}$.
    BySurfaceNormalUsage,
    /// Colors extrusions by the continuous signed distance $\phi_{\text{sdf}}(\mathbf{x})$ (mm) sampled at the nozzle tip (negative inside CAD solid, 0 on CAD surface, positive outside).
    ByNozzleSdf,
    /// Colors each segment by its 3MF filament / colour slot (`filament_slot`) and mapped machine tool.
    ByToolOrFilament,
    /// Colors each segment by its simulated Kubelka–Munk multi-layer optical print appearance (front-lit or back-lit lithophane).
    ByOpticalComposite,
}

impl ToolpathColorMode {
    pub const ALL: [Self; 14] = [
        Self::ByFeatureType,
        Self::ByToolOrFilament,
        Self::ByOpticalComposite,
        Self::ByLineWidth,
        Self::ByLayerHeight,
        Self::BySpeed,
        Self::ByAcceleration,
        Self::ByVolumetricFlow,
        Self::ByExtrusionMultiplier,
        Self::ByPressureAdvance,
        Self::ByMeltZonePressure,
        Self::ByToolIsoNormalAngle,
        Self::BySurfaceNormalUsage,
        Self::ByNozzleSdf,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ByFeatureType => "Feature Type (Outer / Inner / Solid / Sparse / Scarf / Wipe)",
            Self::ByToolOrFilament => "Tool / Filament Colour (3MF Multi-Colour Slot)",
            Self::ByOpticalComposite => {
                "Simulated Print Appearance (Kubelka–Munk Optical Stack)"
            }
            Self::ByLineWidth => "Line Width W (mm)",
            Self::ByLayerHeight => "Layer Height H (mm)",
            Self::BySpeed => "Planned Speed v (mm/s)",
            Self::ByAcceleration => "Acceleration a (mm/s²)",
            Self::ByVolumetricFlow => "Actual Volumetric Flow Q (mm³/s)",
            Self::ByExtrusionMultiplier => "Flow Multiplier M (Corner + Transient)",
            Self::ByPressureAdvance => "Dynamic Pressure Advance k_PA (s)",
            Self::ByMeltZonePressure => "Melt-Zone Pressure P_melt (MPa)",
            Self::ByToolIsoNormalAngle => "Angle: Tool vs. Isosurface Normal (°)",
            Self::BySurfaceNormalUsage => "Tool Orientation Source (Object Surface vs. Isosurface Normal)",
            Self::ByNozzleSdf => "Nozzle SDF Value (φ_sdf mm)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ByFeatureType => "Feature / Line Type",
            Self::ByToolOrFilament => "Tool / Filament Colour",
            Self::ByOpticalComposite => "Simulated Optical Print",
            Self::ByLineWidth => "Line Width W (mm)",
            Self::ByLayerHeight => "Layer Height H (mm)",
            Self::BySpeed => "Planned Speed v (mm/s)",
            Self::ByAcceleration => "Acceleration a (mm/s²)",
            Self::ByVolumetricFlow => "Volumetric Flow Q (mm³/s)",
            Self::ByExtrusionMultiplier => "Flow Multiplier M (×)",
            Self::ByPressureAdvance => "Dynamic PA k_PA (s)",
            Self::ByMeltZonePressure => "Melt-Zone Pressure P_melt",
            Self::ByToolIsoNormalAngle => "Tool vs. Isosurface Normal (°)",
            Self::BySurfaceNormalUsage => "Surface vs. Isosurface Normal",
            Self::ByNozzleSdf => "Nozzle SDF (mm)",
        }
    }
}

/// Returns the nominal positive layer height reference `h_ref` (mm) used to anchor the near-zero
/// resolution of the nozzle-SDF colormap (`±h_ref / 4` stands out clearly around `0.0 mm`).
pub fn nozzle_sdf_reference_layer_height_mm(config: &SlicerConfig) -> f32 {
    let lh = config.slicing.layer_height;
    let lo = if lh.min_mm.is_finite() {
        lh.min_mm.max(0.02)
    } else {
        0.02
    };
    let hi = if lh.max_mm.is_finite() {
        lh.max_mm.max(lo)
    } else {
        lo.max(1.0)
    };
    let target = if lh.target_mm.is_finite() && lh.target_mm > 0.0 {
        lh.target_mm
    } else {
        0.5 * (lo + hi)
    };
    target.clamp(lo, hi)
}

/// Maps signed SDF distance (mm) non-linearly into `[-1.0, 1.0]` anchored to `h_ref = layer_height_mm`:
/// `0 -> 0`, `±h_ref/4 -> ±1/3`, `±h_ref/2 -> ±1/2`, `±h_ref -> ±2/3`, `±2*h_ref -> ±4/5`, `±∞ -> ±1`.
pub fn nozzle_sdf_signed_t(sdf_mm: f32, layer_height_mm: f32) -> f32 {
    if !sdf_mm.is_finite() {
        return 0.0;
    }
    let h_ref = if layer_height_mm.is_finite() && layer_height_mm > 0.0 {
        layer_height_mm.max(0.02)
    } else {
        0.20
    };
    let k = 0.5 * h_ref;
    (sdf_mm / (sdf_mm.abs() + k)).clamp(-1.0, 1.0)
}

fn lerp_rgb_stops(t: f32, stops: &[(f32, [f32; 3])]) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let u = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
            ];
        }
    }
    stops.last().map(|s| s.1).unwrap_or([0.5, 0.5, 0.5])
}

/// Evaluates the signed diverging nozzle-SDF colormap directly from normalized coordinate `signed_t ∈ [-1.0, 1.0]`:
/// - `0.0`: crisp white (`CAD surface`)
/// - `-1/3` (`-h/4`): vivid electric turquoise; `-1/2` (`-h/2`): sky-azure; `-2/3` (`-h`): royal blue; `-1`: deep indigo
/// - `+1/3` (`+h/4`): vivid gold-amber; `+1/2` (`+h/2`): tangerine-orange; `+2/3` (`+h`): scarlet-red; `+1`: dark maroon
pub fn sample_nozzle_sdf_rgba_for_signed_t(signed_t: f32) -> [f32; 4] {
    if !signed_t.is_finite() {
        return [0.5, 0.5, 0.5, 1.0];
    }
    let t = signed_t.abs().clamp(0.0, 1.0);
    let rgb = if signed_t < 0.0 {
        lerp_rgb_stops(
            t,
            &[
                (0.0, [0.98, 0.98, 0.99]),
                (0.18, [0.46, 0.95, 0.88]),
                (1.0 / 3.0, [0.04, 0.86, 0.78]),
                (0.50, [0.04, 0.64, 0.98]),
                (2.0 / 3.0, [0.12, 0.34, 0.98]),
                (0.84, [0.22, 0.14, 0.76]),
                (1.0, [0.10, 0.05, 0.36]),
            ],
        )
    } else {
        lerp_rgb_stops(
            t,
            &[
                (0.0, [0.98, 0.98, 0.99]),
                (0.18, [1.00, 0.93, 0.40]),
                (1.0 / 3.0, [1.00, 0.78, 0.04]),
                (0.50, [1.00, 0.44, 0.04]),
                (2.0 / 3.0, [0.94, 0.14, 0.10]),
                (0.84, [0.70, 0.04, 0.30]),
                (1.0, [0.36, 0.02, 0.22]),
            ],
        )
    };
    [rgb[0], rgb[1], rgb[2], 1.0]
}

/// Signed diverging colormap for nozzle-tip SDF values (mm) with heightened near-zero contrast
/// anchored to `layer_height_mm` (`±layer_height_mm / 4` maps to saturated turquoise / gold-amber).
pub fn sample_nozzle_sdf_rgba(sdf_mm: f32, layer_height_mm: f32) -> [f32; 4] {
    if !sdf_mm.is_finite() {
        return [0.5, 0.5, 0.5, 1.0];
    }
    sample_nozzle_sdf_rgba_for_signed_t(nozzle_sdf_signed_t(sdf_mm, layer_height_mm))
}

/// Computes observed signed nozzle-SDF range `(min_sdf_mm, max_sdf_mm)` across raw object toolpaths,
/// excluding auxiliary paths (`Support`, `SupportInterface`, `PurgeTower`) unless no part points exist.
pub(crate) fn raw_toolpaths_sdf_observed_range(tp_res: &ObjectToolpathsResult) -> Option<(f32, f32)> {
    let mut part_min = f32::INFINITY;
    let mut part_max = f32::NEG_INFINITY;
    let mut all_min = f32::INFINITY;
    let mut all_max = f32::NEG_INFINITY;
    for layer in &tp_res.layers {
        for path in &layer.paths {
            let is_aux = matches!(path.feature_kind, 7 | 8 | 13);
            for pt in &path.points {
                if pt.sdf_mm.is_finite() {
                    all_min = all_min.min(pt.sdf_mm);
                    all_max = all_max.max(pt.sdf_mm);
                    if !is_aux {
                        part_min = part_min.min(pt.sdf_mm);
                        part_max = part_max.max(pt.sdf_mm);
                    }
                }
            }
        }
    }
    if part_min <= part_max {
        Some((part_min, part_max))
    } else if all_min <= all_max {
        Some((all_min, all_max))
    } else {
        None
    }
}

#[inline]
fn scale_srgba_to_linear(mut rgba: [f32; 4], brightness_scale: f32) -> [f32; 4] {
    rgba[0] = (rgba[0] * brightness_scale).clamp(0.0, 1.0);
    rgba[1] = (rgba[1] * brightness_scale).clamp(0.0, 1.0);
    rgba[2] = (rgba[2] * brightness_scale).clamp(0.0, 1.0);
    srgba_to_linear_rgba(rgba)
}

#[inline]
fn nozzle_sdf_segment_linear_colors(
    s0: f32,
    sm: f32,
    s1: f32,
    layer_height_mm: f32,
    brightness_scale: f32,
) -> ([f32; 4], Option<[f32; 4]>, [f32; 4]) {
    let c0 = scale_srgba_to_linear(
        sample_nozzle_sdf_rgba(s0, layer_height_mm),
        brightness_scale,
    );
    let c1 = scale_srgba_to_linear(
        sample_nozzle_sdf_rgba(s1, layer_height_mm),
        brightness_scale,
    );
    let cm_exact = scale_srgba_to_linear(
        sample_nozzle_sdf_rgba(sm, layer_height_mm),
        brightness_scale,
    );
    let max_rgb_dev = (0..3)
        .map(|k| (cm_exact[k] - 0.5 * (c0[k] + c1[k])).abs())
        .fold(0.0f32, f32::max);
    let cm = if (s0 * s1 < 0.0) || max_rgb_dev > 0.03 {
        Some(cm_exact)
    } else {
        None
    };
    (c0, cm, c1)
}

/// Geometry & PBR rendering quality mode for 3D extruded bead previews.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ToolpathQualityMode {
    /// Lightweight 4-sided diamond prism cross-section merged into a single mesh for maximum interactive FPS.
    #[default]
    Fast,
    /// High-quality 12-vertex pressed-stadium (capsule) beads with continuous mitered joints,
    /// convex bullnose endcaps, per-filament Bevy PBR materials (`clearcoat`, polymer roughness, IOR),
    /// TD-driven multi-step optical transmission, and soft cascaded directional shadows.
    HighQualityPbr,
}

impl ToolpathQualityMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast (4-Sided Tubes)",
            Self::HighQualityPbr => "High-Quality PBR (Capsule + Translucency)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::HighQualityPbr => "High-Quality PBR",
        }
    }
}

/// Persistent Bevy entity + asset handles for one `(ShellDepthTier, MaterialSlot)` sub-mesh in `HighQualityPbr` mode.
#[derive(Clone, Debug)]
pub struct HqToolpathSubMesh {
    pub entity: Entity,
    pub mesh_handle: Handle<Mesh>,
    pub mat_handle: Handle<StandardMaterial>,
    pub tier: u8,
    pub mat_slot: u16,
    pub mean_thickness_mm: f32,
}

/// Marker component for the 3D isosurface mesh entity extracted via `openvdb::tools::volumeToMesh`.
#[derive(Component)]
pub struct FieldIsosurfaceEntity;

/// Marker component for the 3D cross-section slice plane entity sampled via `openvdb::tools::GridSampler`.
#[derive(Component)]
pub struct FieldSliceEntity;

/// Marker component for the 3D combined mesh entity of selected slicing isosurfaces.
#[derive(Component)]
pub struct SelectedLayersMeshEntity;

/// Marker component for the 3D variable-width extruded bead mesh entity of generated toolpaths.
#[derive(Component)]
pub struct ToolpathsMeshEntity;

/// Snapshot of the toolhead position, local orthonormal `NozzleFrame` (`AGENTS.md` Rule 1),
/// bed-coordinate gravity direction (`AGENTS.md` Rule 2), and 5-axis rotary angles `(B, C)`
/// at the active layer and within-layer scrub position.
#[derive(Clone, Copy, Debug)]
pub struct ScrubbedToolheadPose {
    /// Nozzle tip position in 3D bed/scene coordinates (`mm`).
    pub tip_bed: Vec3,
    /// Start position of the active scrubbed segment in 3D bed/scene coordinates (`mm`).
    pub seg_start_bed: Vec3,
    /// Orthonormal local nozzle frame $(\hat{\mathbf{x}}_{\text{travel}}, \hat{\mathbf{y}}_{\text{lateral}}, \hat{\mathbf{z}}_{\text{nozzle}})$ in bed coordinates.
    pub frame_bed: NozzleFrame,
    /// Unit gravity direction vector $\hat{\mathbf{g}}_{\text{bed}}$ in bed coordinates at this posture.
    pub gravity_bed: Vec3,
    /// Active extruder tool index (`0` for `T0`, `1` for `T1`, ...).
    pub tool_id: usize,
    /// Instantaneous 5-axis polar tilt `B` and azimuthal rotation `C` angles (degrees).
    pub b_deg: f32,
    pub c_deg: f32,
    /// Active 0-based layer index.
    pub layer_index: usize,
    /// Active 1-based move index within the visible moves of `layer_index`.
    pub move_index: usize,
    /// Total visible moves on `layer_index`.
    pub total_layer_moves: usize,
    /// Human-readable feature label of the active move (e.g., `"Outer Wall"`, `"Solid Infill"`).
    pub move_label: &'static str,
    /// Planned cruise speed (`mm/s`), line width (`mm`), and layer height (`mm`) of the active move.
    pub speed_mms: f32,
    pub width_mm: f32,
    pub height_mm: f32,
    /// Angular deviation (degrees) between `nozzle_axis` and the local slicing isosurface normal at the active move.
    pub tool_iso_angle_deg: f32,
    /// `true` when the active move's nozzle axis is actively oriented using the outward object surface normal.
    pub uses_surface_normal: bool,
    /// Signed distance $\phi_{\text{sdf}}(\mathbf{x})$ (mm) sampled at the nozzle tip (negative inside solid, 0 on CAD surface, positive outside).
    pub sdf_mm: f32,
}

/// Generic scalar field visualization state resource supporting any named `openvdb::FloatGrid`
/// on the currently selected object (`"sdf"`, `"eikonal"`, `"overhang"`, and future fields),
/// plus 3D/2D visualization of the selected non-planar slicing isosurfaces and shell toolpaths.
#[derive(Resource)]
pub struct FieldVisualizationState {
    pub active_field_name: String,
    pub available_fields: Vec<String>,
    pub cached_stats: Option<FieldStats>,
    pub last_object_id: Option<u64>,

    /// Opacity of the selected CAD solid mesh (`0.0..=1.0`) so interior slices/isosurfaces are visible.
    pub solid_opacity: f32,

    // --- Object Surface Coloring Controls (Overhang & Tool Angle from Normal) ---
    pub surface_color_mode: ObjectSurfaceColorMode,
    pub dirty_surface_colors: bool,
    pub last_surface_color_transform: Option<Transform>,
    pub last_surface_color_gravity_bed: [f32; 3],
    pub last_surface_color_nozzle_bed: [f32; 3],
    pub last_solved_field_rotation: Option<Quat>,
    pub surface_color_min_deg: f32,
    pub surface_color_max_deg: f32,
    pub surface_color_mean_deg: f32,
    pub surface_color_uses_solved_tool: bool,

    // --- 3D Isosurface & Object Intersection Controls ---
    pub show_isosurface: bool,
    pub isovalue: f32,
    pub adaptivity: f32,
    pub intersection_mode: IsoIntersectionMode,
    pub show_intersection_contour: bool,
    pub isosurface_entity: Option<Entity>,
    pub isosurface_mesh_handle: Option<Handle<Mesh>>,
    pub isosurface_mat_handle: Option<Handle<StandardMaterial>>,
    pub isosurface_triangles: usize,
    pub interior_triangles: usize,
    pub exterior_triangles: usize,
    pub contour_segments_local: Vec<[Vec3; 2]>,
    pub dirty_isosurface: bool,

    // --- Selected Slicing Isosurfaces Visualization Controls ---
    pub show_selected_layers: bool,
    pub show_selected_layer_surfaces: bool,
    pub show_selected_layer_contours: bool,
    pub selected_layers_color_mode: SelectedLayersColorMode,
    pub selected_layer_stride: usize,
    pub selected_layer_min_idx: usize,
    pub selected_layer_max_idx: usize,
    /// When true, scrubbing the active layer (`selected_layer_max_idx`) locks `selected_layer_min_idx = selected_layer_max_idx`
    /// so only a single layer/isosurface is shown at a time.
    pub single_layer_scrub_mode: bool,
    /// Optional 1-based move limit within the active top layer (`selected_layer_max_idx`).
    /// When `None`, all moves (`100%`) of the active layer are displayed.
    pub layer_move_scrub_limit: Option<usize>,
    pub highlighted_selected_layer: Option<usize>,
    pub last_highlighted_selected_layer: Option<usize>,
    pub cached_selected_layers: Option<SelectedIsosurfacesResult>,
    pub selected_layers_entity: Option<Entity>,
    pub selected_layers_mesh_handle: Option<Handle<Mesh>>,
    pub selected_layers_mat_handle: Option<Handle<StandardMaterial>>,
    pub dirty_selected_layers: bool,
    pub dirty_selected_layers_mesh: bool,

    // --- Non-Planar Shell & Sparse Infill Toolpaths & Planned G-code Visualization Controls ---
    pub show_toolpaths: bool,
    pub show_outer_walls: bool,
    pub show_inner_walls: bool,
    pub show_bridges: bool,
    pub show_wave_overhangs: bool,
    pub show_solid_infill: bool,
    pub show_sparse_infill: bool,
    pub show_gap_fill: bool,
    pub show_supports: bool,
    pub show_support_interface: bool,
    pub show_purge_tower: bool,
    pub show_scarf_seams: bool,
    pub show_wipes: bool,
    pub show_travel_moves: bool,
    pub show_z_hops: bool,
    pub show_toolpath_centerlines: bool,
    pub toolpath_color_mode: ToolpathColorMode,
    pub toolpath_quality_mode: ToolpathQualityMode,
    /// Computes each filament's physical single-bead opacity directly from its Kubelka–Munk
    /// Transmission Distance (`TD`) and local bead thickness (`1 - T_eff(d, TD)`).
    pub enable_pbr_translucency: bool,
    /// Enables millimeter-calibrated cascaded directional shadows in `HighQualityPbr` mode.
    pub enable_hq_shadows: bool,
    /// Opacity multiplier (`0.10..=1.50`, `1.00` = exact physical TD opacity when `enable_pbr_translucency` is active).
    pub toolpath_opacity: f32,
    /// Boosts rear backlight illumination in 3D to preview backlit lithophane glow through translucent extrusions.
    pub lithophane_backlight_preview: bool,
    pub cached_unmixed_toolpaths: Option<ObjectToolpathsResult>,
    pub cached_toolpaths: Option<ObjectToolpathsResult>,
    pub cached_planned_gcode: Option<crate::gcode::PlannedGcodeResult>,
    /// Per-object cached unmixed toolpaths keyed by `object_id` across the build plate.
    pub plate_unmixed_toolpaths: std::collections::HashMap<u64, ObjectToolpathsResult>,
    /// Per-object cached optical-mixed toolpaths keyed by `object_id` across the build plate.
    pub plate_mixed_toolpaths: std::collections::HashMap<u64, ObjectToolpathsResult>,
    /// True when `cached_planned_gcode` segments are stored in scene world coordinates
    /// (so the toolpath mesh entity is placed at `Transform::IDENTITY`).
    pub planned_gcode_is_world_space: bool,
    /// Last active build plate index observed by `sync_field_visualization_system`.
    pub last_active_plate_index: Option<usize>,
    /// Cached signature of active plate printable objects & transforms used to build `cached_planned_gcode`.
    pub last_plate_gcode_signature: u64,
    pub toolpaths_entity: Option<Entity>,
    pub toolpaths_mesh_handle: Option<Handle<Mesh>>,
    pub toolpaths_mat_handle: Option<Handle<StandardMaterial>>,
    pub hq_toolpath_parts: Vec<HqToolpathSubMesh>,
    pub last_hq_pbr_sync_sig: Option<u64>,
    pub dirty_toolpaths: bool,
    pub dirty_color_mixing: bool,
    pub dirty_planned_gcode: bool,
    pub dirty_toolpaths_mesh: bool,
    pub dirty_triangle_slot_remap: bool,

    // --- 2D Cross-Section Slice Controls & Persistent Handles ---
    pub show_slice_plane: bool,
    pub slice_axis: SliceAxis,
    pub slice_normalized_pos: f32,
    pub slice_resolution: u32,
    pub colormap: FieldColormap,
    pub show_iso_contours: bool,
    pub fade_exterior_background: bool,
    pub slice_entity: Option<Entity>,
    pub slice_mesh_handle: Option<Handle<Mesh>>,
    pub slice_mat_handle: Option<Handle<StandardMaterial>>,
    pub slice_image_handle: Option<Handle<Image>>,
    pub slice_plane_coord: f32,
    pub dirty_slice: bool,

    /// Background task computing OpenVDB scalar fields for `(object_id, status_target, computed_object_ids)`.
    pub pending_field_task: Option<(u64, String, Vec<u64>, bevy::tasks::Task<bool>)>,
    /// Background task computing selected slicing isosurfaces and/or non-planar shell toolpaths.
    pub pending_slicing_task: Option<(
        u64,
        bevy::tasks::Task<(
            Option<SelectedIsosurfacesResult>,
            Vec<(u64, ObjectToolpathsResult)>,
        )>,
    )>,
    /// Shared live progress tracker for background post-slicing (colour mixing, G-code planning, mesh building).
    pub post_slice_progress: std::sync::Arc<PostSliceProgress>,
    /// Background task executing colour mixing, G-code planning, and/or 3D toolpath mesh buffer construction.
    pub pending_post_slice_task: Option<bevy::tasks::Task<PostSliceTaskOutput>>,
    /// True when the active `pending_post_slice_task` should display its progress bar immediately.
    pub post_slice_show_progress: bool,
    /// True while the user is actively holding down the pointer on a UI slider/drag widget.
    pub ui_pointer_down: bool,
}

/// Thread-safe progress tracker for background Rust post-slicing tasks (colour mixing, G-code planning, mesh build).
#[derive(Debug)]
pub struct PostSliceProgress {
    fraction_bits: std::sync::atomic::AtomicU32,
    stage: std::sync::Mutex<String>,
}

impl Default for PostSliceProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl PostSliceProgress {
    pub fn new() -> Self {
        Self {
            fraction_bits: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            stage: std::sync::Mutex::new("Preparing toolpaths...".to_string()),
        }
    }

    pub fn set(&self, fraction: f32, stage_msg: impl Into<String>) {
        self.fraction_bits.store(
            fraction.clamp(0.0, 1.0).to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Ok(mut guard) = self.stage.lock() {
            *guard = stage_msg.into();
        }
    }

    pub fn get(&self) -> (f32, String) {
        let frac = f32::from_bits(
            self.fraction_bits
                .load(std::sync::atomic::Ordering::Relaxed),
        )
        .clamp(0.0, 1.0);
        let stg = self
            .stage
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| "Processing toolpaths...".to_string());
        (frac, stg)
    }
}

/// Pre-built 3D toolpath vertex/index buffers produced on a background `IoTaskPool` thread.
pub enum BuiltToolpathGeometry {
    Fast {
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        colors: Vec<[f32; 4]>,
        indices: Vec<u32>,
    },
    HighQualityPbr(Vec<HqBucketMesh>),
}

/// Result returned by a background `pending_post_slice_task`.
pub struct PostSliceTaskOutput {
    pub plate_index: usize,
    pub selected_obj_id: Option<u64>,
    pub invalidated_other_plates_mixed: bool,
    pub reset_layer_range: bool,
    pub remapped_triangle_slots: Vec<(u64, Vec<u8>)>,
    pub updated_mixed_toolpaths: Option<(
        std::collections::HashMap<u64, ObjectToolpathsResult>,
        Option<ObjectToolpathsResult>,
    )>,
    pub updated_planned_gcode: Option<Option<crate::gcode::PlannedGcodeResult>>,
    pub built_mesh: Option<(BuiltToolpathGeometry, Transform)>,
}

/// Decision flags computed on each frame for `pending_post_slice_task` scheduling and
/// synchronous stale-cache invalidation, scoped strictly to the active build plate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PostSliceWorkDecision {
    pub active_plate_has_plannable_tp: bool,
    pub clear_stale_planned_gcode: bool,
    pub clear_stale_mixed_toolpaths: bool,
    pub want_remap: bool,
    pub want_mix: bool,
    pub want_plan: bool,
    pub want_heavy_mesh: bool,
    pub invalidate_other_plates_mixed: bool,
}

#[inline]
pub(crate) fn is_valid_nonempty_toolpaths(tp: &ObjectToolpathsResult) -> bool {
    tp.valid && !tp.layers.is_empty()
}

/// Evaluates whether the active build plate requires triangle colour remapping,
/// filament/optical mixing, G-code planning, or background 3D mesh construction.
pub(crate) fn compute_post_slice_work_decision(
    vis_state: &FieldVisualizationState,
    active_printable_ids: &[u64],
    selected_obj_id: Option<u64>,
    selected_obj_printable: bool,
) -> PostSliceWorkDecision {
    let sel_has_unmixed = vis_state
        .cached_unmixed_toolpaths
        .as_ref()
        .is_some_and(is_valid_nonempty_toolpaths);
    let sel_has_mixed = vis_state
        .cached_toolpaths
        .as_ref()
        .is_some_and(is_valid_nonempty_toolpaths);

    let active_has_unmixed = active_printable_ids.iter().any(|id| {
        vis_state
            .plate_unmixed_toolpaths
            .get(id)
            .is_some_and(is_valid_nonempty_toolpaths)
    }) || (selected_obj_printable && sel_has_unmixed);

    let has_any_active_or_sel_unmixed = active_has_unmixed || sel_has_unmixed;
    let sel_needs_mix = sel_has_unmixed && !sel_has_mixed;
    let active_plate_missing_mix = active_printable_ids.iter().any(|id| {
        vis_state
            .plate_unmixed_toolpaths
            .get(id)
            .is_some_and(is_valid_nonempty_toolpaths)
            && !vis_state
                .plate_mixed_toolpaths
                .get(id)
                .is_some_and(is_valid_nonempty_toolpaths)
    });

    let active_plate_has_plannable_tp = active_printable_ids.iter().any(|id| {
        vis_state
            .plate_unmixed_toolpaths
            .get(id)
            .is_some_and(is_valid_nonempty_toolpaths)
            || vis_state
                .plate_mixed_toolpaths
                .get(id)
                .is_some_and(is_valid_nonempty_toolpaths)
            || (selected_obj_id == Some(*id) && (sel_has_unmixed || sel_has_mixed))
    });

    let want_remap = vis_state.dirty_triangle_slot_remap;
    let want_mix = (has_any_active_or_sel_unmixed && vis_state.dirty_color_mixing)
        || sel_needs_mix
        || active_plate_missing_mix;
    let invalidate_other_plates_mixed = want_mix && vis_state.dirty_color_mixing;
    let clear_stale_mixed_toolpaths =
        !has_any_active_or_sel_unmixed && vis_state.dirty_color_mixing;
    let want_plan = active_plate_has_plannable_tp
        && (want_mix || vis_state.dirty_planned_gcode || vis_state.cached_planned_gcode.is_none());
    let clear_stale_planned_gcode = !active_plate_has_plannable_tp && vis_state.dirty_planned_gcode;
    let has_renderable_toolpaths = vis_state.cached_planned_gcode.is_some() || sel_has_mixed;
    let want_heavy_mesh = !want_mix
        && !want_plan
        && !clear_stale_planned_gcode
        && vis_state.show_toolpaths
        && vis_state.dirty_toolpaths_mesh
        && has_renderable_toolpaths
        && (vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr
            || vis_state.toolpath_color_mode == ToolpathColorMode::ByOpticalComposite);

    PostSliceWorkDecision {
        active_plate_has_plannable_tp,
        clear_stale_planned_gcode,
        clear_stale_mixed_toolpaths,
        want_remap,
        want_mix,
        want_plan,
        want_heavy_mesh,
        invalidate_other_plates_mixed,
    }
}

/// Lightweight `Copy` snapshot of toolpath visibility and layer-scrub filters for background mesh building.
#[derive(Debug, Clone, Copy)]
pub struct ToolpathMeshFilter {
    pub show_outer_walls: bool,
    pub show_inner_walls: bool,
    pub show_bridges: bool,
    pub show_wave_overhangs: bool,
    pub show_solid_infill: bool,
    pub show_sparse_infill: bool,
    pub show_gap_fill: bool,
    pub show_supports: bool,
    pub show_support_interface: bool,
    pub show_purge_tower: bool,
    pub show_scarf_seams: bool,
    pub show_wipes: bool,
    pub show_travel_moves: bool,
    pub show_z_hops: bool,
    pub selected_layer_stride: usize,
    pub selected_layer_min_idx: usize,
    pub selected_layer_max_idx: usize,
    pub highlighted_selected_layer: Option<usize>,
    pub layer_move_scrub_limit: Option<usize>,
    pub toolpath_color_mode: ToolpathColorMode,
    pub toolpath_quality_mode: ToolpathQualityMode,
}

impl Default for FieldVisualizationState {
    fn default() -> Self {
        Self {
            active_field_name: "sdf".to_string(),
            available_fields: Vec::new(),
            cached_stats: None,
            last_object_id: None,
            solid_opacity: 0.25,
            surface_color_mode: ObjectSurfaceColorMode::DefaultCad,
            dirty_surface_colors: true,
            last_surface_color_transform: None,
            last_surface_color_gravity_bed: [0.0, 0.0, -1.0],
            last_surface_color_nozzle_bed: [0.0, 0.0, 1.0],
            last_solved_field_rotation: None,
            surface_color_min_deg: 0.0,
            surface_color_max_deg: 180.0,
            surface_color_mean_deg: 90.0,
            surface_color_uses_solved_tool: false,
            show_isosurface: false,
            isovalue: 0.0,
            adaptivity: 0.0,
            intersection_mode: IsoIntersectionMode::ClipToObjectInterior,
            show_intersection_contour: true,
            isosurface_entity: None,
            isosurface_mesh_handle: None,
            isosurface_mat_handle: None,
            isosurface_triangles: 0,
            interior_triangles: 0,
            exterior_triangles: 0,
            contour_segments_local: Vec::new(),
            dirty_isosurface: true,
            show_selected_layers: false,
            show_selected_layer_surfaces: true,
            show_selected_layer_contours: true,
            selected_layers_color_mode: SelectedLayersColorMode::ByLayerThickness,
            selected_layer_stride: 1,
            selected_layer_min_idx: 0,
            selected_layer_max_idx: 0,
            single_layer_scrub_mode: false,
            layer_move_scrub_limit: None,
            highlighted_selected_layer: None,
            last_highlighted_selected_layer: None,
            cached_selected_layers: None,
            selected_layers_entity: None,
            selected_layers_mesh_handle: None,
            selected_layers_mat_handle: None,
            dirty_selected_layers: true,
            dirty_selected_layers_mesh: true,
            show_toolpaths: false,
            show_outer_walls: true,
            show_inner_walls: true,
            show_bridges: true,
            show_wave_overhangs: true,
            show_solid_infill: true,
            show_sparse_infill: true,
            show_gap_fill: true,
            show_supports: true,
            show_support_interface: true,
            show_purge_tower: true,
            show_scarf_seams: true,
            show_wipes: true,
            show_travel_moves: false,
            show_z_hops: false,
            show_toolpath_centerlines: false,
            toolpath_color_mode: ToolpathColorMode::ByFeatureType,
            toolpath_quality_mode: ToolpathQualityMode::Fast,
            enable_pbr_translucency: true,
            enable_hq_shadows: false,
            toolpath_opacity: 1.0,
            lithophane_backlight_preview: false,
            cached_unmixed_toolpaths: None,
            cached_toolpaths: None,
            cached_planned_gcode: None,
            plate_unmixed_toolpaths: std::collections::HashMap::new(),
            plate_mixed_toolpaths: std::collections::HashMap::new(),
            planned_gcode_is_world_space: false,
            last_active_plate_index: None,
            last_plate_gcode_signature: 0,
            toolpaths_entity: None,
            toolpaths_mesh_handle: None,
            toolpaths_mat_handle: None,
            hq_toolpath_parts: Vec::new(),
            last_hq_pbr_sync_sig: None,
            dirty_toolpaths: true,
            dirty_color_mixing: false,
            dirty_planned_gcode: true,
            dirty_toolpaths_mesh: true,
            dirty_triangle_slot_remap: false,
            show_slice_plane: false,
            slice_axis: SliceAxis::Z,
            slice_normalized_pos: 0.5,
            slice_resolution: 160,
            colormap: FieldColormap::CoolWarmSigned,
            show_iso_contours: true,
            fade_exterior_background: true,
            slice_entity: None,
            slice_mesh_handle: None,
            slice_mat_handle: None,
            slice_image_handle: None,
            slice_plane_coord: 0.0,
            dirty_slice: true,
            pending_field_task: None,
            pending_slicing_task: None,
            post_slice_progress: std::sync::Arc::new(PostSliceProgress::new()),
            pending_post_slice_task: None,
            post_slice_show_progress: false,
            ui_pointer_down: false,
        }
    }
}

impl FieldVisualizationState {
    /// Returns `true` when a C++ OpenVDB field computation or slicing task is actively running.
    pub fn is_backend_busy(&self) -> bool {
        self.pending_field_task.is_some() || self.pending_slicing_task.is_some()
    }

    /// Returns `true` when any background field, slicing, or post-slice G-code/mesh task is running.
    pub fn is_busy(&self) -> bool {
        self.pending_field_task.is_some()
            || self.pending_slicing_task.is_some()
            || self.pending_post_slice_task.is_some()
    }

    /// Returns the active background task progress `(fraction_0_to_1, stage_description)` if a
    /// field computation, non-planar slicing, or post-slice colour/G-code/mesh task is running.
    pub fn active_progress(&self) -> Option<(f32, String)> {
        if self.pending_field_task.is_some() || self.pending_slicing_task.is_some() {
            let prog = ffi::get_field_compute_progress();
            let stage = if prog.stage.trim().is_empty() {
                if self.pending_field_task.is_some() {
                    "Computing OpenVDB scalar fields...".to_string()
                } else {
                    "Generating non-planar shell toolpaths...".to_string()
                }
            } else {
                prog.stage
            };
            return Some((prog.fraction.clamp(0.0, 1.0), stage));
        }
        if self.pending_post_slice_task.is_some() && self.post_slice_show_progress {
            return Some(self.post_slice_progress.get());
        }
        None
    }

    pub fn toolpath_mesh_filter(&self) -> ToolpathMeshFilter {
        ToolpathMeshFilter {
            show_outer_walls: self.show_outer_walls,
            show_inner_walls: self.show_inner_walls,
            show_bridges: self.show_bridges,
            show_wave_overhangs: self.show_wave_overhangs,
            show_solid_infill: self.show_solid_infill,
            show_sparse_infill: self.show_sparse_infill,
            show_gap_fill: self.show_gap_fill,
            show_supports: self.show_supports,
            show_support_interface: self.show_support_interface,
            show_purge_tower: self.show_purge_tower,
            show_scarf_seams: self.show_scarf_seams,
            show_wipes: self.show_wipes,
            show_travel_moves: self.show_travel_moves,
            show_z_hops: self.show_z_hops,
            selected_layer_stride: self.selected_layer_stride.max(1),
            selected_layer_min_idx: self.selected_layer_min_idx,
            selected_layer_max_idx: self.selected_layer_max_idx,
            highlighted_selected_layer: self.highlighted_selected_layer,
            layer_move_scrub_limit: self.layer_move_scrub_limit,
            toolpath_color_mode: self.toolpath_color_mode,
            toolpath_quality_mode: self.toolpath_quality_mode,
        }
    }

    fn clear_view_caches_and_mark_dirty(&mut self) {
        self.cached_stats = None;
        self.cached_selected_layers = None;
        self.cached_unmixed_toolpaths = None;
        self.cached_toolpaths = None;
        self.cached_planned_gcode = None;
        self.planned_gcode_is_world_space = false;
        self.last_plate_gcode_signature = 0;
        self.layer_move_scrub_limit = None;
        self.dirty_isosurface = true;
        self.dirty_selected_layers = true;
        self.dirty_selected_layers_mesh = true;
        self.dirty_toolpaths = true;
        self.dirty_color_mixing = false;
        self.dirty_planned_gcode = true;
        self.dirty_toolpaths_mesh = true;
        self.dirty_triangle_slot_remap = false;
        self.dirty_slice = true;
        self.dirty_surface_colors = true;
    }

    pub fn mark_all_dirty(&mut self) {
        self.plate_unmixed_toolpaths.clear();
        self.plate_mixed_toolpaths.clear();
        self.clear_view_caches_and_mark_dirty();
    }

    /// Invalidates cached field/slicing results for `object_ids` while preserving cached
    /// toolpaths for untouched objects on other build plates.
    pub fn mark_objects_dirty(&mut self, object_ids: &[u64]) {
        for id in object_ids {
            self.plate_unmixed_toolpaths.remove(id);
            self.plate_mixed_toolpaths.remove(id);
        }
        self.clear_view_caches_and_mark_dirty();
    }

    /// Spawns an asynchronous background task on `IoTaskPool` to compute selected slicing
    /// isosurfaces and/or non-planar shell toolpaths without blocking the UI thread.
    pub fn spawn_slicing_task(
        &mut self,
        object_id: u64,
        config: &SlicerConfig,
        need_layers: bool,
        need_toolpaths: bool,
    ) {
        self.spawn_slicing_task_for_objects(
            object_id,
            vec![object_id],
            config,
            need_layers,
            need_toolpaths,
        );
    }

    /// Spawns an asynchronous background task on `IoTaskPool` to compute selected slicing
    /// isosurfaces for `object_id` and/or non-planar shell toolpaths for all `printable_object_ids`
    /// on the active build plate.
    pub fn spawn_slicing_task_for_objects(
        &mut self,
        object_id: u64,
        printable_object_ids: Vec<u64>,
        config: &SlicerConfig,
        need_layers: bool,
        need_toolpaths: bool,
    ) {
        if self.is_busy() || (!need_layers && !need_toolpaths) {
            return;
        }

        if need_layers {
            self.dirty_selected_layers = false;
        }
        if need_toolpaths {
            self.dirty_toolpaths = false;
        }

        let lh = config.slicing.layer_height;
        let lh_spec = LayerHeightSpec {
            target_mm: lh.target_mm,
            min_mm: lh.min_mm,
            max_mm: lh.max_mm,
            first_layer_mm: lh.first_layer_mm,
            top_layer_mm: lh.top_layer_mm,
        };
        let s = &config.slicing;
        let nozzle_diameter_mm = config
            .tools
            .get(config.active_tool_index)
            .or_else(|| config.tools.first())
            .map(|t| t.nozzle_diameter_mm)
            .unwrap_or(0.4);
        let shell_spec = ShellPathSpec {
            shell_thickness_mm: s.shell_thickness_mm,
            inner_wall_count: s.inner_wall_count,
            outer_wall_width: LineWidthSpec {
                target_mm: s.outer_wall_width.target_mm,
                min_mm: s.outer_wall_width.min_mm,
                max_mm: s.outer_wall_width.max_mm,
            },
            inner_wall_width: LineWidthSpec {
                target_mm: s.inner_wall_width.target_mm,
                min_mm: s.inner_wall_width.min_mm,
                max_mm: s.inner_wall_width.max_mm,
            },
            solid_infill_width: LineWidthSpec {
                target_mm: s.solid_infill_width.target_mm,
                min_mm: s.solid_infill_width.min_mm,
                max_mm: s.solid_infill_width.max_mm,
            },
            solid_infill_pattern: s.solid_infill_pattern.as_u8(),
            min_extrusion_length_mm: s.min_extrusion_length_mm,
            sparse_infill_density: (s.infill_density_percent / 100.0).clamp(0.0, 1.0),
            sparse_infill_width: LineWidthSpec {
                target_mm: s.sparse_infill_width.target_mm,
                min_mm: s.sparse_infill_width.min_mm,
                max_mm: s.sparse_infill_width.max_mm,
            },
            sparse_infill_pattern: s.sparse_infill_pattern.as_u8(),
            infill_angle_deg: s.infill_angle_deg,
            nozzle_diameter_mm,
            overhang_angle_threshold_deg: s.overhang_angle_threshold_deg,
            max_bridge_length_mm: s.max_bridge_length_mm,
            wave_overhang_overlap_fraction: s.wave_overhang_overlap_fraction,
            one_perimeter_on_conformal_patches: s.one_perimeter_on_conformal_patches,
            orient_outer_surfaces_to_normal: s.orient_outer_surfaces_to_normal,
            outer_surface_parallel_angle_deg: s.outer_surface_parallel_angle_deg,
            brick_layer_mode: s.brick_layer_mode.as_u8(),
            enable_top_surface_lofting: s.enable_top_surface_lofting,
        };
        let support_spec = SupportSpec {
            enabled: s.enable_support_generation,
            support_tool_index: s.support_tool_index.map(|idx| idx as i32).unwrap_or(-1),
            support_style: s.support_style.as_u8(),
            on_bed_only: s.support_on_bed_only,
            seed_order_field: s.seed_order_field_from_supports,
            only_unreachable_and_long_bridges: s.support_only_unreachable_and_bridges,
            max_bridge_length_mm: s.max_bridge_length_mm,
            z_gap_mm: s.support_z_gap_mm,
            xy_gap_mm: s.support_xy_gap_mm,
            interface_layers: s.support_interface_layers,
            bed_interface_layers: s.support_bed_interface_layers,
            base_spacing_mm: s.support_base_spacing_mm,
            interface_spacing_mm: s.support_interface_spacing_mm,
            line_width_mm: s.support_line_width_mm,
            tree_branch_angle_deg: s.tree_branch_angle_deg,
            tree_tip_diameter_mm: s.tree_tip_diameter_mm,
            tree_trunk_max_diameter_mm: s.tree_trunk_max_diameter_mm,
            tree_branch_wall_count: s.tree_branch_wall_count,
        };
        let skin_spec = ffi::ConformalSkinSpec {
            auto_enabled: s.enable_auto_conformal_skin,
            default_skin_layers: s.conformal_skin_layers,
            min_oblique_angle_deg: s.conformal_skin_min_oblique_deg,
            max_slope_angle_deg: s.conformal_skin_max_slope_deg,
        };
        let adaptivity = self.adaptivity;

        let task = IoTaskPool::get().spawn(async move {
            ffi::set_slice_layer_height(lh_spec);
            ffi::set_shell_path_spec(shell_spec);
            ffi::set_support_spec(support_spec);
            ffi::set_conformal_skin_spec(skin_spec);

            let layers_res = if need_layers {
                Some(ffi::select_slicing_isosurfaces(object_id, adaptivity))
            } else {
                None
            };
            let mut tp_list = Vec::new();
            if need_toolpaths {
                let ids = if printable_object_ids.is_empty() {
                    vec![object_id]
                } else {
                    printable_object_ids
                };
                for oid in ids {
                    let tp = ffi::generate_object_shell_toolpaths(oid, adaptivity);
                    tp_list.push((oid, tp));
                }
            }
            (layers_res, tp_list)
        });

        self.pending_slicing_task = Some((object_id, task));
    }

    pub fn apply_slicing_task_result(
        &mut self,
        layers_opt: Option<SelectedIsosurfacesResult>,
        plate_toolpaths: Vec<(u64, ObjectToolpathsResult)>,
    ) {
        if let Some(res) = layers_opt {
            if res.valid && !res.layers.is_empty() {
                let max_idx = res.layers.len() - 1;
                self.selected_layer_min_idx = 0;
                self.selected_layer_max_idx = max_idx;
                self.single_layer_scrub_mode = false;
                self.layer_move_scrub_limit = None;
                self.cached_selected_layers = Some(res);
            } else {
                self.selected_layer_min_idx = 0;
                self.selected_layer_max_idx = 0;
                self.single_layer_scrub_mode = false;
                self.layer_move_scrub_limit = None;
                self.cached_selected_layers = None;
            }
            self.dirty_selected_layers_mesh = true;
            self.dirty_slice = true;
        }

        if !plate_toolpaths.is_empty() {
            let mut any_valid = false;
            for (oid, res) in plate_toolpaths {
                if is_valid_nonempty_toolpaths(&res) {
                    any_valid = true;
                    if self.selected_layer_max_idx == 0 && res.layers.len() > 1 {
                        self.selected_layer_max_idx = res.layers.len() - 1;
                    }
                    if Some(oid) == self.last_object_id {
                        self.cached_unmixed_toolpaths = Some(res.clone());
                        self.cached_toolpaths = None;
                    }
                    self.plate_unmixed_toolpaths.insert(oid, res);
                    // Force fresh colour mixing only for newly sliced objects without discarding
                    // valid mixes cached for objects on other build plates.
                    self.plate_mixed_toolpaths.remove(&oid);
                } else {
                    self.plate_unmixed_toolpaths.remove(&oid);
                    self.plate_mixed_toolpaths.remove(&oid);
                    if Some(oid) == self.last_object_id {
                        self.cached_unmixed_toolpaths = None;
                        self.cached_toolpaths = None;
                    }
                }
            }
            self.layer_move_scrub_limit = None;
            if !any_valid {
                self.cached_unmixed_toolpaths = None;
                self.cached_toolpaths = None;
                self.cached_planned_gcode = None;
                self.dirty_color_mixing = false;
            }
            self.dirty_planned_gcode = true;
            self.dirty_toolpaths_mesh = true;
        }
    }

    pub fn refresh_stats_for(&mut self, object_id: u64) {
        if self.active_field_name.is_empty() {
            return;
        }
        let stats = ffi::get_object_field_stats(object_id, &self.active_field_name);
        if stats.valid {
            if self.isovalue < stats.min_value || self.isovalue > stats.max_value {
                self.isovalue = if stats.min_value <= 0.0 && 0.0 <= stats.max_value {
                    0.0
                } else {
                    0.5 * (stats.min_value + stats.max_value)
                };
            }
            self.cached_stats = Some(stats);
        }
    }
}

/// Applies Virtual Filament integer-ratio dithering (`config.has_virtual_filament_slots()`) and/or
/// Kubelka-Munk optical filament mixing (`config.slicing.color_mixing`) to `raw_toolpaths`
/// for `object_id` when at least 2 extrusion tools are configured, or returns a clone of `raw_toolpaths`.
pub fn build_mixed_toolpaths_for_object(
    object_id: u64,
    raw_toolpaths: &ObjectToolpathsResult,
    config: &SlicerConfig,
) -> ObjectToolpathsResult {
    build_mixed_toolpaths_for_object_with_mode(
        object_id,
        raw_toolpaths,
        config,
        ObjectFilamentMode::ProjectDefault,
        None,
    )
}

/// Applies per-object filament assignment (`ObjectFilamentMode`), Virtual Filament integer-ratio
/// dithering, and/or Kubelka–Munk optical colour mixing (CMYKW Front-Lit Reflectance or Back-Lit
/// Lithophane) to `raw_toolpaths` for `object_id`.
pub fn build_mixed_toolpaths_for_object_with_mode(
    object_id: u64,
    raw_toolpaths: &ObjectToolpathsResult,
    config: &SlicerConfig,
    filament_mode: ObjectFilamentMode,
    custom_color_srgba: Option<[f32; 4]>,
) -> ObjectToolpathsResult {
    if !raw_toolpaths.valid {
        return raw_toolpaths.clone();
    }
    let support_tool = if config.slicing.enable_support_generation {
        Some(config.support_tool_index_resolved())
    } else {
        None
    };

    let apply_infill_override = |tp: &mut ObjectToolpathsResult| {
        match config.slicing.infill_filament_mode {
            crate::config::InfillFilamentMode::SingleTool(t_idx) => {
                let clamped = t_idx.min(config.tools.len().saturating_sub(1));
                let direct_slot = crate::color_optics::encode_direct_tool_slot(clamped);
                for layer in &mut tp.layers {
                    for path in &mut layer.paths {
                        if path.feature_kind == 4 {
                            path.filament_slot = direct_slot;
                        }
                    }
                }
            }
            crate::config::InfillFilamentMode::FilamentSlot(s_idx) => {
                for layer in &mut tp.layers {
                    for path in &mut layer.paths {
                        if path.feature_kind == 4 {
                            path.filament_slot = s_idx;
                        }
                    }
                }
            }
            crate::config::InfillFilamentMode::FollowObject
            | crate::config::InfillFilamentMode::AutoPurgeIntoInfill => {}
        }
    };

    match filament_mode {
        ObjectFilamentMode::SingleTool(tool_idx) => {
            let clamped_tool = tool_idx.min(config.tools.len().saturating_sub(1));
            let direct_slot = crate::color_optics::encode_direct_tool_slot(clamped_tool);
            let mut out = raw_toolpaths.clone();
            for layer in &mut out.layers {
                for path in &mut layer.paths {
                    if path.feature_kind != 7 && path.feature_kind != 8 {
                        path.filament_slot = direct_slot;
                    }
                }
            }
            apply_infill_override(&mut out);
            if matches!(
                config.slicing.infill_filament_mode,
                crate::config::InfillFilamentMode::FilamentSlot(_)
            ) && config.has_virtual_filament_slots()
                && config.tools.len() >= 2
            {
                return crate::color_optics::apply_virtual_filament_dithering_to_toolpaths(
                    &out,
                    &config.filament_slots,
                    config.tools.len(),
                    config.slicing.color_mixing.min_short_extrusion_mm,
                    support_tool,
                );
            }
            return out;
        }
        ObjectFilamentMode::FilamentSlot(slot_idx) => {
            let mut assigned = raw_toolpaths.clone();
            for layer in &mut assigned.layers {
                for path in &mut layer.paths {
                    if path.feature_kind != 7 && path.feature_kind != 8 {
                        path.filament_slot = slot_idx;
                    }
                }
            }
            apply_infill_override(&mut assigned);
            let is_virtual = config
                .filament_slots
                .get(slot_idx as usize)
                .map_or(false, |s| s.is_virtual_mix())
                || config.has_virtual_filament_slots();
            if is_virtual && config.tools.len() >= 2 {
                return crate::color_optics::apply_virtual_filament_dithering_to_toolpaths(
                    &assigned,
                    &config.filament_slots,
                    config.tools.len(),
                    config.slicing.color_mixing.min_short_extrusion_mm,
                    support_tool,
                );
            }
            return assigned;
        }
        ObjectFilamentMode::DirectSlotsNoOptical => {
            let mut assigned = raw_toolpaths.clone();
            apply_infill_override(&mut assigned);
            if config.has_virtual_filament_slots() && config.tools.len() >= 2 {
                return crate::color_optics::apply_virtual_filament_dithering_to_toolpaths(
                    &assigned,
                    &config.filament_slots,
                    config.tools.len(),
                    config.slicing.color_mixing.min_short_extrusion_mm,
                    support_tool,
                );
            }
            return assigned;
        }
        ObjectFilamentMode::ProjectDefault
        | ObjectFilamentMode::OpticalFrontLit
        | ObjectFilamentMode::OpticalBackLitLithophane => {}
    }

    let mut base_tp = raw_toolpaths.clone();
    apply_infill_override(&mut base_tp);

    if config.tools.len() < 2 {
        return base_tp;
    }

    let want_optical = match filament_mode {
        ObjectFilamentMode::OpticalFrontLit | ObjectFilamentMode::OpticalBackLitLithophane => true,
        _ => config.slicing.color_mixing.enabled,
    };
    let has_virtual = config.has_virtual_filament_slots();
    if !has_virtual && !want_optical {
        return base_tp;
    }

    let after_virtual = if has_virtual {
        crate::color_optics::apply_virtual_filament_dithering_to_toolpaths(
            &base_tp,
            &config.filament_slots,
            config.tools.len(),
            config.slicing.color_mixing.min_short_extrusion_mm,
            support_tool,
        )
    } else {
        base_tp
    };

    if !want_optical {
        return after_virtual;
    }

    let mut mix_cfg = config.slicing.color_mixing.clone();
    mix_cfg.enabled = true;
    match filament_mode {
        ObjectFilamentMode::OpticalFrontLit => {
            mix_cfg.lighting_mode = crate::color_optics::OpticalLightingMode::FrontLitReflectance;
        }
        ObjectFilamentMode::OpticalBackLitLithophane => {
            mix_cfg.lighting_mode = crate::color_optics::OpticalLightingMode::BackLitTransmittance;
        }
        _ => {}
    }

    let tool_optics = config.tool_optical_inputs();
    let mut slot_colors = config.imported_slot_colors_srgba();
    if let Some(custom_rgba) = custom_color_srgba {
        if slot_colors.is_empty() {
            slot_colors.push(custom_rgba);
        } else {
            for sc in &mut slot_colors {
                *sc = custom_rgba;
            }
        }
    } else if slot_colors.is_empty() {
        let fallback_rgba = config
            .active_tool()
            .map(|t| t.filament_color_srgba)
            .unwrap_or([0.24, 0.62, 0.92, 1.0]);
        slot_colors.push(fallback_rgba);
    }

    crate::color_optics::apply_optical_color_mixing_to_toolpaths(
        &after_virtual,
        &mix_cfg,
        &tool_optics,
        &slot_colors,
        config.slicing.layer_height.target_mm,
        support_tool,
        |flat_xyz| ffi::sample_object_surface_optics(object_id, flat_xyz),
    )
}

/// Synchronizes available OpenVDB scalar fields, updates 3D isosurfaces (`volumeToMesh`)
/// with exact object-intersection clipping/highlighting ($\phi_{\text{sdf}} \le 0$) and 3D
/// intersection contour curves ($\tau = c \cap \phi_{\text{sdf}} = 0$), renders the selected
/// slicing isosurfaces stack, and updates 2D cross-section slices (`GridSampler`) in-place.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn sync_field_visualization_system(
    mut commands: Commands,
    mut plate: ResMut<BuildPlateState>,
    config: Res<SlicerConfig>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gizmos: Gizmos,
    object_transforms: Query<
        &Transform,
        (
            With<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
            Without<ToolpathsMeshEntity>,
        ),
    >,
    mut iso_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldIsosurfaceEntity>,
            Without<CadObjectEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
            Without<ToolpathsMeshEntity>,
        ),
    >,
    mut slice_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldSliceEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<SelectedLayersMeshEntity>,
            Without<ToolpathsMeshEntity>,
        ),
    >,
    mut selected_layers_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<SelectedLayersMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<ToolpathsMeshEntity>,
        ),
    >,
    mut toolpaths_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<ToolpathsMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
) {
    let selected_id = plate.selected_object_id;

    // 0. Poll pending background post-slicing (colour mixing / G-code planning / 3D mesh build) task
    if let Some(ref mut task) = vis_state.pending_post_slice_task {
        if let Some(output) =
            bevy::tasks::block_on(bevy::tasks::futures_lite::future::poll_once(task))
        {
            vis_state.pending_post_slice_task = None;
            vis_state.post_slice_show_progress = false;

            if !output.remapped_triangle_slots.is_empty() {
                for (oid, slots) in output.remapped_triangle_slots {
                    if let Some(obj) = plate.objects.iter_mut().find(|o| o.object_id == oid) {
                        obj.triangle_filament_slots = slots;
                    }
                }
                vis_state.dirty_surface_colors = true;
            }
            if let Some((mixed_map, sel_mixed)) = output.updated_mixed_toolpaths {
                if output.invalidated_other_plates_mixed {
                    vis_state.plate_mixed_toolpaths = mixed_map;
                } else {
                    for (oid, mixed) in mixed_map {
                        vis_state.plate_mixed_toolpaths.insert(oid, mixed);
                    }
                }
                if output.selected_obj_id == selected_id {
                    vis_state.cached_toolpaths = sel_mixed;
                } else {
                    vis_state.cached_toolpaths = selected_id
                        .and_then(|id| vis_state.plate_mixed_toolpaths.get(&id).cloned());
                }
            }
            if output.plate_index == plate.active_plate_index {
                if let Some(planned_opt) = output.updated_planned_gcode {
                    if let Some(ref combined) = planned_opt {
                        if !combined.layers.is_empty() {
                            if output.reset_layer_range {
                                vis_state.selected_layer_min_idx = 0;
                                vis_state.selected_layer_max_idx = combined.layers.len() - 1;
                            } else {
                                vis_state.selected_layer_max_idx =
                                    vis_state.selected_layer_max_idx.min(combined.layers.len() - 1);
                                vis_state.selected_layer_min_idx = vis_state
                                    .selected_layer_min_idx
                                    .min(vis_state.selected_layer_max_idx);
                            }
                        }
                        vis_state.planned_gcode_is_world_space = true;
                    } else {
                        vis_state.planned_gcode_is_world_space = false;
                    }
                    vis_state.cached_planned_gcode = planned_opt;
                }
                if let Some((built_geom, render_tf)) = output.built_mesh {
                    if vis_state.show_toolpaths {
                        apply_built_toolpath_geometry_to_scene(
                            &mut commands,
                            &mut vis_state,
                            &config,
                            &mut meshes,
                            &mut materials,
                            &mut toolpaths_query,
                            built_geom,
                            render_tf,
                        );
                    }
                }
            } else if output.updated_planned_gcode.is_some() || output.built_mesh.is_some() {
                vis_state.dirty_planned_gcode = true;
                vis_state.dirty_toolpaths_mesh = true;
            }
        }
    }

    // 0b. Detect active build plate change (covers switching to empty plates as well)
    if vis_state.last_active_plate_index != Some(plate.active_plate_index) {
        vis_state.last_active_plate_index = Some(plate.active_plate_index);
        vis_state.cached_planned_gcode = None;
        vis_state.planned_gcode_is_world_space = false;
        vis_state.layer_move_scrub_limit = None;
        vis_state.single_layer_scrub_mode = false;
        vis_state.dirty_planned_gcode = true;
        vis_state.dirty_toolpaths_mesh = true;
    }

    // 1. Detect object selection change
    if vis_state.last_object_id != selected_id {
        if let Some(prev_id) = vis_state.last_object_id {
            if let Some(prev_obj) = plate.objects.iter().find(|o| o.object_id == prev_id) {
                if let Some(mesh) = meshes.get_mut(&prev_obj.mesh_handle) {
                    mesh.insert_attribute(
                        Mesh::ATTRIBUTE_COLOR,
                        default_cad_vertex_colors_for_object(prev_obj, &config),
                    );
                }
            }
        }
        vis_state.last_object_id = selected_id;
        vis_state.available_fields.clear();
        vis_state.cached_stats = None;
        vis_state.cached_selected_layers = None;
        vis_state.cached_unmixed_toolpaths = selected_id
            .and_then(|id| vis_state.plate_unmixed_toolpaths.get(&id).cloned());
        vis_state.cached_toolpaths = selected_id
            .and_then(|id| vis_state.plate_mixed_toolpaths.get(&id).cloned());
        vis_state.layer_move_scrub_limit = None;
        vis_state.contour_segments_local.clear();
        vis_state.dirty_isosurface = true;
        vis_state.dirty_selected_layers = true;
        vis_state.dirty_selected_layers_mesh = true;
        vis_state.dirty_toolpaths =
            vis_state.cached_unmixed_toolpaths.is_none() && vis_state.cached_toolpaths.is_none();
        vis_state.dirty_slice = true;
        vis_state.dirty_surface_colors = true;
        vis_state.last_surface_color_transform = None;
        vis_state.last_solved_field_rotation = None;

        if let Some(id) = selected_id {
            if !vis_state.is_backend_busy() {
                vis_state.available_fields = ffi::list_object_fields(id);
                if !vis_state.available_fields.is_empty()
                    && !vis_state
                        .available_fields
                        .contains(&vis_state.active_field_name)
                {
                    vis_state.active_field_name = vis_state.available_fields[0].clone();
                }
            }
        }
    }

    // Track changes to active plate printable objects & their transforms
    {
        use std::hash::{Hash, Hasher};
        let mut sig_hasher = std::collections::hash_map::DefaultHasher::new();
        plate.active_plate_index.hash(&mut sig_hasher);
        config.gcode.purge_mode.hash(&mut sig_hasher);
        config.gcode.min_purge_volume_mm3.to_bits().hash(&mut sig_hasher);
        config.gcode.purge_tower_auto_dimensions.hash(&mut sig_hasher);
        config.gcode.purge_tower_width_mm.to_bits().hash(&mut sig_hasher);
        config.gcode.purge_tower_depth_mm.to_bits().hash(&mut sig_hasher);
        config.gcode.purge_tower_brim.hash(&mut sig_hasher);
        config.gcode.purge_tower_auto_position.hash(&mut sig_hasher);
        config.gcode.purge_tower_position_xy[0].to_bits().hash(&mut sig_hasher);
        config.gcode.purge_tower_position_xy[1].to_bits().hash(&mut sig_hasher);
        config.slicing.infill_filament_mode.hash(&mut sig_hasher);
        config.slicing.support_tool_index.hash(&mut sig_hasher);
        config.slicing.support_interface_tool_index.hash(&mut sig_hasher);
        for o in &plate.objects {
            if o.plate_index == plate.active_plate_index {
                o.object_id.hash(&mut sig_hasher);
                o.printable.hash(&mut sig_hasher);
                o.filament_mode.hash(&mut sig_hasher);
                if let Some(cc) = o.custom_color_srgba {
                    for c in cc {
                        c.to_bits().hash(&mut sig_hasher);
                    }
                }
                let tf = object_transforms
                    .get(o.entity)
                    .copied()
                    .unwrap_or(Transform::IDENTITY);
                tf.translation.x.to_bits().hash(&mut sig_hasher);
                tf.translation.y.to_bits().hash(&mut sig_hasher);
                tf.translation.z.to_bits().hash(&mut sig_hasher);
                tf.rotation.x.to_bits().hash(&mut sig_hasher);
                tf.rotation.y.to_bits().hash(&mut sig_hasher);
                tf.rotation.z.to_bits().hash(&mut sig_hasher);
                tf.rotation.w.to_bits().hash(&mut sig_hasher);
            }
        }
        let plate_sig = sig_hasher.finish();
        if vis_state.last_plate_gcode_signature != plate_sig {
            vis_state.last_plate_gcode_signature = plate_sig;
            vis_state.dirty_planned_gcode = true;
        }
    }

    // If no object is selected, hide active field visualization entities (except standalone calibration toolpaths)
    let Some(obj_id) = selected_id else {
        if let Some(ent) = vis_state.isosurface_entity {
            if let Ok((_, mut vis)) = iso_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
        if let Some(ent) = vis_state.slice_entity {
            if let Ok((_, mut vis)) = slice_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
        if let Some(ent) = vis_state.selected_layers_entity {
            if let Ok((_, mut vis)) = selected_layers_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
        if vis_state.dirty_color_mixing {
            vis_state.dirty_color_mixing = false;
            vis_state.plate_mixed_toolpaths.clear();
        }
        if vis_state.dirty_planned_gcode {
            vis_state.dirty_planned_gcode = false;
            vis_state.cached_planned_gcode = None;
            vis_state.planned_gcode_is_world_space = false;
            vis_state.dirty_toolpaths_mesh = false;
        }
        if !vis_state.show_toolpaths || vis_state.cached_planned_gcode.is_none() {
            hide_all_toolpath_entities(&vis_state, &mut toolpaths_query);
            return;
        }

        // Render standalone planned G-code toolpaths (e.g., Calibration Patterns) at Transform::IDENTITY
        if vis_state.dirty_toolpaths_mesh {
            vis_state.dirty_toolpaths_mesh = false;
            rebuild_and_sync_toolpath_meshes(
                &mut commands,
                &mut vis_state,
                &config,
                &mut meshes,
                &mut materials,
                &mut toolpaths_query,
                Transform::IDENTITY,
            );
        }
        return;
    };

    let Some(scene_obj) = plate.objects.iter().find(|o| o.object_id == obj_id) else {
        return;
    };

    let obj_world_tf = object_transforms
        .get(scene_obj.entity)
        .copied()
        .unwrap_or(Transform::IDENTITY);

    if vis_state.cached_stats.is_none() && !vis_state.is_backend_busy() {
        vis_state.refresh_stats_for(obj_id);
    }

    // 1b. Update CAD object surface vertex colors (Default CAD / Overhang / Tool Angle from Normal)
    //     Deferred while `ui_pointer_down` is true so dragging colour pickers or ratio sliders never stalls the UI.
    let tf_changed = vis_state.last_surface_color_transform != Some(obj_world_tf);
    let grav_changed =
        vis_state.last_surface_color_gravity_bed != config.machine.gravity_direction_bed;
    let nozzle_changed = vis_state.last_surface_color_nozzle_bed != config.machine.nozzle_axis_bed;
    if !vis_state.ui_pointer_down
        && (vis_state.dirty_surface_colors
            || (vis_state.surface_color_mode != ObjectSurfaceColorMode::DefaultCad
                && (tf_changed || grav_changed || nozzle_changed)))
    {
        vis_state.dirty_surface_colors = false;
        vis_state.last_surface_color_transform = Some(obj_world_tf);
        vis_state.last_surface_color_gravity_bed = config.machine.gravity_direction_bed;
        vis_state.last_surface_color_nozzle_bed = config.machine.nozzle_axis_bed;

        if vis_state.surface_color_mode == ObjectSurfaceColorMode::DefaultCad {
            vis_state.surface_color_min_deg = 0.0;
            vis_state.surface_color_max_deg = 180.0;
            vis_state.surface_color_mean_deg = 0.0;
            vis_state.surface_color_uses_solved_tool = false;
            for obj in &plate.objects {
                if obj.plate_index == plate.active_plate_index
                    && (!obj.vertex_filament_slots.is_empty()
                        || !obj.vertex_colors_rgba.is_empty()
                        || obj.object_id == obj_id)
                {
                    if let Some(mesh) = meshes.get_mut(&obj.mesh_handle) {
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_COLOR,
                            default_cad_vertex_colors_for_object(obj, &config),
                        );
                    }
                }
            }
        } else {
            let solved_axes =
                if vis_state.surface_color_mode == ObjectSurfaceColorMode::ToolAngleFromNormal
                    && !vis_state.is_backend_busy()
                    && vis_state
                        .available_fields
                        .iter()
                        .any(|f| f == "tool_tilt_deg" || f == "eikonal")
                    && vis_state
                        .last_solved_field_rotation
                        .map_or(true, |r| r.angle_between(obj_world_tf.rotation) < 1.0e-3)
                {
                    ffi::get_object_vertex_tool_axes(obj_id)
                } else if vis_state.surface_color_mode == ObjectSurfaceColorMode::SupportNeeded
                    && !vis_state.is_backend_busy()
                    && vis_state
                        .available_fields
                        .iter()
                        .any(|f| f == "support_needed" || f == "eikonal")
                {
                    ffi::get_object_vertex_support_needed(obj_id)
                } else {
                    Vec::new()
                };

            let (colors, min_deg, max_deg, mean_deg, uses_solved) =
                compute_object_surface_vertex_colors(
                    &scene_obj.vertices,
                    &obj_world_tf,
                    &config,
                    vis_state.surface_color_mode,
                    &solved_axes,
                );
            vis_state.surface_color_min_deg = min_deg;
            vis_state.surface_color_max_deg = max_deg;
            vis_state.surface_color_mean_deg = mean_deg;
            vis_state.surface_color_uses_solved_tool = uses_solved;

            if let Some(mesh) = meshes.get_mut(&scene_obj.mesh_handle) {
                mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
            }
        }
    }

    // Check if hover highlight on the selected layers list changed
    if vis_state.last_highlighted_selected_layer != vis_state.highlighted_selected_layer {
        vis_state.last_highlighted_selected_layer = vis_state.highlighted_selected_layer;
        vis_state.dirty_selected_layers_mesh = true;
        vis_state.dirty_toolpaths_mesh = true;
        if vis_state.show_slice_plane && vis_state.show_selected_layers {
            vis_state.dirty_slice = true;
        }
    }

    // Spawn background task for selected slicing isosurfaces and/or shell toolpaths when needed
    let want_layers = vis_state.show_selected_layers && vis_state.dirty_selected_layers;
    let want_toolpaths = vis_state.show_toolpaths && vis_state.dirty_toolpaths;
    if (want_layers || want_toolpaths)
        && !vis_state.is_busy()
        && !vis_state.ui_pointer_down
        && vis_state.available_fields.iter().any(|f| f == "eikonal")
    {
        let need_layers =
            want_layers || (want_toolpaths && vis_state.cached_selected_layers.is_none());
        let printable_ids: Vec<u64> = plate
            .objects
            .iter()
            .filter(|o| o.plate_index == plate.active_plate_index && o.printable)
            .map(|o| o.object_id)
            .collect();
        vis_state.spawn_slicing_task_for_objects(
            obj_id,
            printable_ids,
            &config,
            need_layers,
            want_toolpaths,
        );
    }

    // 2. Update 3D Isosurface Mesh in-place (opaque depth-tested mesh with per-vertex colors)
    if !vis_state.show_isosurface {
        if let Some(ent) = vis_state.isosurface_entity {
            if let Ok((_, mut vis)) = iso_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
        vis_state.isosurface_triangles = 0;
        vis_state.interior_triangles = 0;
        vis_state.exterior_triangles = 0;
        vis_state.contour_segments_local.clear();
    } else if vis_state.dirty_isosurface && !vis_state.is_busy() {
        vis_state.dirty_isosurface = false;

        let iso_mesh = ffi::extract_field_isosurface(
            obj_id,
            &vis_state.active_field_name,
            vis_state.isovalue,
            vis_state.adaptivity,
        );

        vis_state.interior_triangles = iso_mesh.interior_indices.len() / 3;
        vis_state.exterior_triangles = iso_mesh.exterior_indices.len() / 3;

        // Cache 3D intersection boundary contour segments (\tau(x) = c \cap \phi_sdf(x) = 0)
        vis_state.contour_segments_local.clear();
        for seg in iso_mesh.intersection_contour.chunks_exact(6) {
            let p0 = Vec3::new(seg[0], seg[1], seg[2]);
            let p1 = Vec3::new(seg[3], seg[4], seg[5]);
            if p0.distance_squared(p1) > 1.0e-8 {
                vis_state.contour_segments_local.push([p0, p1]);
            }
        }

        let is_non_sdf = vis_state.active_field_name != "sdf";
        let active_indices: Vec<u32> = if is_non_sdf
            && vis_state.intersection_mode == IsoIntersectionMode::ClipToObjectInterior
        {
            iso_mesh.interior_indices.clone()
        } else {
            iso_mesh.indices.clone()
        };

        vis_state.isosurface_triangles = active_indices.len() / 3;

        if !iso_mesh.vertices.is_empty() && !active_indices.is_empty() {
            let mut positions = Vec::with_capacity(iso_mesh.vertices.len());
            let mut normals = Vec::with_capacity(iso_mesh.vertices.len());
            let mut colors = Vec::with_capacity(iso_mesh.vertices.len());

            // Determine interior SDF depth scale for rich shading inside the object
            let min_sdf = iso_mesh
                .vertex_sdf
                .iter()
                .copied()
                .fold(0.0f32, f32::min)
                .min(-1.0);

            for (i, v) in iso_mesh.vertices.iter().enumerate() {
                // Apply a tiny 0.015 mm normal bias so the phi=0 isosurface does not Z-fight with the CAD mesh
                positions.push([
                    v.x + v.nx * 0.015,
                    v.y + v.ny * 0.015,
                    v.z + v.nz * 0.015,
                ]);
                normals.push([v.nx, v.ny, v.nz]);

                let sdf_val = iso_mesh.vertex_sdf.get(i).copied().unwrap_or(0.0);
                let rgba = compute_isosurface_vertex_color(
                    &vis_state.active_field_name,
                    vis_state.intersection_mode,
                    sdf_val,
                    min_sdf,
                );
                colors.push(srgba_to_linear_rgba(rgba));
            }

            let bevy_mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
            .with_inserted_indices(Indices::U32(active_indices));

            if let (Some(ent), Some(ref mesh_h), Some(ref mat_h)) = (
                vis_state.isosurface_entity,
                &vis_state.isosurface_mesh_handle,
                &vis_state.isosurface_mat_handle,
            ) {
                meshes.insert(mesh_h, bevy_mesh);
                if let Some(mat) = materials.get_mut(mat_h) {
                    mat.base_color = Color::WHITE;
                }
                if let Ok((mut tf, mut vis)) = iso_query.get_mut(ent) {
                    *tf = obj_world_tf;
                    *vis = Visibility::Inherited;
                }
            } else {
                let mesh_h = meshes.add(bevy_mesh);
                let mat_h = materials.add(StandardMaterial {
                    // Base color is WHITE so per-vertex ATTRIBUTE_COLOR directly drives surface appearance
                    base_color: Color::WHITE,
                    metallic: 0.0,
                    perceptual_roughness: 0.58,
                    reflectance: 0.16,
                    alpha_mode: AlphaMode::Opaque,
                    depth_bias: 8.0,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                });
                let ent = commands
                    .spawn((
                        PbrBundle {
                            mesh: mesh_h.clone(),
                            material: mat_h.clone(),
                            transform: obj_world_tf,
                            ..default()
                        },
                        FieldIsosurfaceEntity,
                    ))
                    .id();
                vis_state.isosurface_entity = Some(ent);
                vis_state.isosurface_mesh_handle = Some(mesh_h);
                vis_state.isosurface_mat_handle = Some(mat_h);
            }
        } else if let Some(ent) = vis_state.isosurface_entity {
            if let Ok((_, mut vis)) = iso_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    }

    // 2b. Render 3D Object Intersection Boundary Contour (\tau(x) = c \cap \phi_sdf(x) = 0)
    if vis_state.show_isosurface
        && vis_state.show_intersection_contour
        && vis_state.active_field_name != "sdf"
        && !vis_state.contour_segments_local.is_empty()
    {
        let contour_color = Color::srgb(0.15, 1.0, 0.92);
        for &[p0, p1] in &vis_state.contour_segments_local {
            let w0 = obj_world_tf.transform_point(p0);
            let w1 = obj_world_tf.transform_point(p1);
            gizmos.line(w0, w1, contour_color);
        }
    }

    // 2c. Render Selected Slicing Isosurfaces Stack (3D Clipped Surfaces + Contours)
    if !vis_state.show_selected_layers || !vis_state.show_selected_layer_surfaces {
        if let Some(ent) = vis_state.selected_layers_entity {
            if let Ok((_, mut vis)) = selected_layers_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    } else if vis_state.dirty_selected_layers_mesh {
        vis_state.dirty_selected_layers_mesh = false;

        let stride = vis_state.selected_layer_stride.max(1);
        let min_idx = vis_state.selected_layer_min_idx;
        let max_idx = vis_state.selected_layer_max_idx;
        let highlighted = vis_state.highlighted_selected_layer;
        let color_mode = vis_state.selected_layers_color_mode;
        let lh_cfg = config.slicing.layer_height;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        if let Some(ref res) = vis_state.cached_selected_layers {
            let total = res.layers.len();
            let lo = min_idx.min(total.saturating_sub(1));
            let hi = max_idx.min(total.saturating_sub(1)).max(lo);

            let mut total_verts = 0usize;
            let mut total_inds = 0usize;
            for (idx, layer) in res.layers.iter().enumerate() {
                let is_hovered = highlighted == Some(idx);
                let in_range = idx >= lo && idx <= hi;
                let matches_stride = (idx.saturating_sub(lo) % stride == 0)
                    || layer.is_first_layer
                    || layer.is_top_layer;
                if !is_hovered && (!in_range || !matches_stride) {
                    continue;
                }
                total_verts += layer.vertices.len();
                total_inds += layer.indices.len();
            }
            positions.reserve(total_verts);
            normals.reserve(total_verts);
            colors.reserve(total_verts);
            indices.reserve(total_inds);

            for (idx, layer) in res.layers.iter().enumerate() {
                let is_hovered = highlighted == Some(idx);
                let in_range = idx >= lo && idx <= hi;
                let matches_stride = (idx.saturating_sub(lo) % stride == 0)
                    || layer.is_first_layer
                    || layer.is_top_layer;
                if !is_hovered && (!in_range || !matches_stride) {
                    continue;
                }
                if layer.vertices.is_empty() || layer.indices.is_empty() {
                    continue;
                }

                let base_v = positions.len() as u32;
                let mut rgba = compute_selected_layer_color(
                    layer,
                    color_mode,
                    lh_cfg.min_mm,
                    lh_cfg.target_mm,
                    lh_cfg.max_mm,
                );
                if is_hovered {
                    rgba = [1.0, 0.96, 0.25, 1.0];
                } else if highlighted.is_some() {
                    // Subtle dimming of non-hovered layers when inspecting a specific layer in the list
                    rgba[0] *= 0.65;
                    rgba[1] *= 0.65;
                    rgba[2] *= 0.65;
                }

                let linear_rgba = srgba_to_linear_rgba(rgba);
                for v in &layer.vertices {
                    positions.push([v.x, v.y, v.z]);
                    normals.push([v.nx, v.ny, v.nz]);
                    colors.push(linear_rgba);
                }
                for &i in &layer.indices {
                    indices.push(base_v + i);
                }
            }
        }

        if !positions.is_empty() && !indices.is_empty() {
            let bevy_mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
            .with_inserted_indices(Indices::U32(indices));

            if let (Some(ent), Some(ref mesh_h), Some(ref mat_h)) = (
                vis_state.selected_layers_entity,
                &vis_state.selected_layers_mesh_handle,
                &vis_state.selected_layers_mat_handle,
            ) {
                meshes.insert(mesh_h, bevy_mesh);
                if let Some(mat) = materials.get_mut(mat_h) {
                    mat.base_color = Color::WHITE;
                }
                if let Ok((mut tf, mut vis)) = selected_layers_query.get_mut(ent) {
                    *tf = obj_world_tf;
                    *vis = Visibility::Inherited;
                }
            } else {
                let mesh_h = meshes.add(bevy_mesh);
                let mat_h = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    metallic: 0.0,
                    perceptual_roughness: 0.58,
                    reflectance: 0.16,
                    alpha_mode: AlphaMode::Opaque,
                    depth_bias: 10.0,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                });
                let ent = commands
                    .spawn((
                        PbrBundle {
                            mesh: mesh_h.clone(),
                            material: mat_h.clone(),
                            transform: obj_world_tf,
                            ..default()
                        },
                        SelectedLayersMeshEntity,
                    ))
                    .id();
                vis_state.selected_layers_entity = Some(ent);
                vis_state.selected_layers_mesh_handle = Some(mesh_h);
                vis_state.selected_layers_mat_handle = Some(mat_h);
            }
        } else if let Some(ent) = vis_state.selected_layers_entity {
            if let Ok((_, mut vis)) = selected_layers_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    }

    // Draw 3D perimeter intersection contours for selected layers
    if vis_state.show_selected_layers && vis_state.show_selected_layer_contours {
        if let Some(ref res) = vis_state.cached_selected_layers {
            let total = res.layers.len();
            if total > 0 {
                let stride = vis_state.selected_layer_stride.max(1);
                let lo = vis_state.selected_layer_min_idx.min(total - 1);
                let hi = vis_state.selected_layer_max_idx.min(total - 1).max(lo);
                let highlighted = vis_state.highlighted_selected_layer;
                let lh_cfg = config.slicing.layer_height;

                for (idx, layer) in res.layers.iter().enumerate() {
                    let is_hovered = highlighted == Some(idx);
                    let in_range = idx >= lo && idx <= hi;
                    let matches_stride = (idx.saturating_sub(lo) % stride == 0)
                        || layer.is_first_layer
                        || layer.is_top_layer;
                    if !is_hovered && (!in_range || !matches_stride) {
                        continue;
                    }

                    let rgba = if is_hovered {
                        [1.0, 0.98, 0.20, 1.0]
                    } else {
                        let mut c = compute_selected_layer_color(
                            layer,
                            vis_state.selected_layers_color_mode,
                            lh_cfg.min_mm,
                            lh_cfg.target_mm,
                            lh_cfg.max_mm,
                        );
                        c[0] = (c[0] * 1.15).min(1.0);
                        c[1] = (c[1] * 1.15).min(1.0);
                        c[2] = (c[2] * 1.15).min(1.0);
                        c
                    };
                    let line_color = Color::srgb(rgba[0], rgba[1], rgba[2]);
                    let active_scrub_layer = highlighted.unwrap_or(hi);
                    let max_segs = if !vis_state.show_toolpaths && idx == active_scrub_layer {
                        vis_state.layer_move_scrub_limit.unwrap_or(usize::MAX)
                    } else {
                        usize::MAX
                    };

                    for seg in layer.contour_segments.chunks_exact(6).take(max_segs) {
                        let p0 = obj_world_tf.transform_point(Vec3::new(seg[0], seg[1], seg[2]));
                        let p1 = obj_world_tf.transform_point(Vec3::new(seg[3], seg[4], seg[5]));
                        gizmos.line(p0, p1, line_color);
                    }
                }
            }
        }
    }

    // 2d. Spawn asynchronous background task on `IoTaskPool` for triangle colour remapping,
    //     Virtual Filament dithering / Kubelka-Munk optical mixing, G-code & purge planning,
    //     and 3D toolpath mesh construction so the UI thread never blocks or beachballs!
    let active_printable_ids: Vec<u64> = plate
        .objects
        .iter()
        .filter(|o| o.plate_index == plate.active_plate_index && o.printable)
        .map(|o| o.object_id)
        .collect();
    let can_start_post_slice = vis_state.pending_post_slice_task.is_none()
        && !vis_state.is_backend_busy()
        && !vis_state.ui_pointer_down
        && !plate.is_dragging_purge_tower
        && !plate.is_dragging_object;

    if can_start_post_slice {
        let decision = compute_post_slice_work_decision(
            &vis_state,
            &active_printable_ids,
            Some(obj_id),
            scene_obj.printable,
        );
        if decision.clear_stale_mixed_toolpaths {
            vis_state.dirty_color_mixing = false;
            vis_state.plate_mixed_toolpaths.clear();
        }
        if decision.clear_stale_planned_gcode {
            vis_state.dirty_planned_gcode = false;
            vis_state.cached_planned_gcode = None;
            vis_state.planned_gcode_is_world_space = false;
            vis_state.dirty_toolpaths_mesh = true;
        }

        let want_remap = decision.want_remap;
        let want_mix = decision.want_mix;
        let want_plan = decision.want_plan;
        let want_heavy_mesh = decision.want_heavy_mesh;

        if want_remap || want_mix || want_plan || want_heavy_mesh {
            if want_remap {
                vis_state.dirty_triangle_slot_remap = false;
            }
            if want_mix {
                vis_state.dirty_color_mixing = false;
            }
            if want_plan {
                vis_state.dirty_planned_gcode = false;
            }
            let build_mesh_in_task =
                vis_state.show_toolpaths && (want_mix || want_plan || want_heavy_mesh);
            if build_mesh_in_task {
                vis_state.dirty_toolpaths_mesh = false;
            }

            let task_plate_index = plate.active_plate_index;
            let invalidate_other_plates_mixed = decision.invalidate_other_plates_mixed;
            let reset_layer_range = want_plan && vis_state.cached_planned_gcode.is_none();

            let progress = vis_state.post_slice_progress.clone();
            let initial_stage = if want_remap {
                "Remapping 3MF triangle colours to filament slots..."
            } else if want_mix {
                "Applying filament mappings & optical colour mixing..."
            } else if want_plan {
                "Planning non-planar G-code & purges..."
            } else {
                "Building 3D toolpath preview mesh..."
            };
            progress.set(0.02, initial_stage);
            vis_state.post_slice_show_progress = true;

            let config_clone = (*config).clone();
            let sel_oid = obj_id;
            let sel_obj_mode = scene_obj.filament_mode;
            let sel_obj_custom_col = scene_obj.custom_color_srgba;
            let sel_world_tf = obj_world_tf;
            let mut mesh_filter = vis_state.toolpath_mesh_filter();

            let remap_inputs: Vec<(u64, Vec<u32>, usize)> = if want_remap {
                plate
                    .objects
                    .iter()
                    .filter(|o| !o.triangle_colors_rgba.is_empty() && o.vertex_filament_slots.is_empty())
                    .map(|o| (o.object_id, o.triangle_colors_rgba.clone(), o.indices.len() / 3))
                    .collect()
            } else {
                Vec::new()
            };

            let unmixed_snapshot = if want_mix {
                let scoped_unmixed: std::collections::HashMap<u64, ObjectToolpathsResult> =
                    vis_state
                        .plate_unmixed_toolpaths
                        .iter()
                        .filter(|(&oid, _)| {
                            oid == sel_oid || active_printable_ids.contains(&oid)
                        })
                        .map(|(&oid, tp)| (oid, tp.clone()))
                        .collect();
                Some((scoped_unmixed, vis_state.cached_unmixed_toolpaths.clone()))
            } else {
                None
            };
            let existing_mixed_snapshot = if !want_mix && (want_plan || want_heavy_mesh) {
                let scoped_mixed: std::collections::HashMap<u64, ObjectToolpathsResult> =
                    vis_state
                        .plate_mixed_toolpaths
                        .iter()
                        .filter(|(&oid, _)| {
                            oid == sel_oid || active_printable_ids.contains(&oid)
                        })
                        .map(|(&oid, tp)| (oid, tp.clone()))
                        .collect();
                Some((scoped_mixed, vis_state.cached_toolpaths.clone()))
            } else {
                None
            };
            let existing_planned_for_mesh = if want_heavy_mesh {
                vis_state
                    .cached_planned_gcode
                    .as_ref()
                    .map(clone_planned_without_gcode_text)
            } else {
                None
            };
            let existing_planned_is_world = vis_state.planned_gcode_is_world_space;

            let obj_modes: std::collections::HashMap<u64, (ObjectFilamentMode, Option<[f32; 4]>)> =
                plate
                    .objects
                    .iter()
                    .map(|o| (o.object_id, (o.filament_mode, o.custom_color_srgba)))
                    .collect();

            let printable_specs: Vec<(u64, String, Transform)> = if want_plan {
                plate
                    .objects
                    .iter()
                    .filter(|o| o.plate_index == plate.active_plate_index && o.printable)
                    .map(|o| {
                        let o_tf = object_transforms
                            .get(o.entity)
                            .copied()
                            .unwrap_or(Transform::IDENTITY);
                        (o.object_id, o.name.clone(), o_tf)
                    })
                    .collect()
            } else {
                Vec::new()
            };

            let task = IoTaskPool::get().spawn(async move {
                // 1. Optional triangle colour -> slot remapping
                let mut remapped_triangle_slots: Vec<(u64, Vec<u8>)> = Vec::new();
                if !remap_inputs.is_empty() {
                    progress.set(0.03, "Remapping 3MF triangle colours to filament slots...");
                    let slot_colors = config_clone.imported_slot_colors_srgba();
                    if !slot_colors.is_empty() {
                        for (oid, tri_cols, num_tris) in remap_inputs {
                            let remapped = crate::color_optics::map_triangle_colors_to_slot_colors(
                                &tri_cols,
                                &slot_colors,
                            );
                            if remapped.len() == num_tris {
                                ffi::set_object_triangle_filament_slots(oid, &remapped);
                                remapped_triangle_slots.push((oid, remapped));
                            }
                        }
                    }
                }

                // 2. Optional Virtual Filament ratio dithering & Kubelka-Munk optical colour mixing
                let mut updated_mixed_toolpaths: Option<(
                    std::collections::HashMap<u64, ObjectToolpathsResult>,
                    Option<ObjectToolpathsResult>,
                )> = None;
                let (active_mixed_map, active_sel_mixed) =
                    if let Some((unmixed_map, cached_unmixed)) = unmixed_snapshot {
                        progress.set(0.06, "Applying filament mappings & optical colour mixing...");
                        let mut mixed_map: std::collections::HashMap<u64, ObjectToolpathsResult> =
                            std::collections::HashMap::with_capacity(unmixed_map.len().max(1));
                        let mut sel_mixed: Option<ObjectToolpathsResult> = None;
                        let total_objs = unmixed_map.len().max(1);
                        for (idx, (&oid, raw_tp)) in unmixed_map.iter().enumerate() {
                            progress.set(
                                0.06 + 0.12 * ((idx as f32) / (total_objs as f32)),
                                format!(
                                    "Applying filament & colour mixing ({}/{total_objs})...",
                                    idx + 1
                                ),
                            );
                            let (o_mode, o_custom_col) = obj_modes
                                .get(&oid)
                                .copied()
                                .unwrap_or((ObjectFilamentMode::ProjectDefault, None));
                            let mixed = build_mixed_toolpaths_for_object_with_mode(
                                oid,
                                raw_tp,
                                &config_clone,
                                o_mode,
                                o_custom_col,
                            );
                            if oid == sel_oid {
                                sel_mixed = Some(mixed.clone());
                            }
                            mixed_map.insert(oid, mixed);
                        }
                        if !mixed_map.contains_key(&sel_oid) {
                            if let Some(ref raw_tp) = cached_unmixed {
                                let mixed = build_mixed_toolpaths_for_object_with_mode(
                                    sel_oid,
                                    raw_tp,
                                    &config_clone,
                                    sel_obj_mode,
                                    sel_obj_custom_col,
                                );
                                sel_mixed = Some(mixed.clone());
                                mixed_map.insert(sel_oid, mixed);
                            }
                        }
                        updated_mixed_toolpaths = Some((mixed_map.clone(), sel_mixed.clone()));
                        (mixed_map, sel_mixed)
                    } else if let Some((mixed_map, sel_mixed)) = existing_mixed_snapshot {
                        (mixed_map, sel_mixed)
                    } else {
                        (std::collections::HashMap::new(), None)
                    };

                // 3. Optional G-code & purge planning across all printable objects on the active plate
                let updated_planned_gcode: Option<Option<crate::gcode::PlannedGcodeResult>> =
                    if want_plan {
                        let mut planned_objects: Vec<(
                            String,
                            crate::gcode::PlannedGcodeResult,
                            Transform,
                        )> = Vec::new();
                        let num_printable = printable_specs.len().max(1);
                        for (p_idx, (oid, o_name, o_tf)) in printable_specs.into_iter().enumerate()
                        {
                            let tp_opt = active_mixed_map.get(&oid).or_else(|| {
                                if oid == sel_oid {
                                    active_sel_mixed.as_ref()
                                } else {
                                    None
                                }
                            });
                            if let Some(tp_res) = tp_opt {
                                if tp_res.valid && !tp_res.layers.is_empty() {
                                    let obj_base_frac =
                                        0.18 + 0.64 * ((p_idx as f32) / (num_printable as f32));
                                    let obj_span_frac = 0.64 / (num_printable as f32);
                                    let prog_ref = &progress;
                                    let mut p =
                                        crate::gcode::plan_and_emit_gcode_for_object_with_progress(
                                            &o_name,
                                            oid,
                                            tp_res,
                                            o_tf,
                                            &config_clone,
                                            Some(&|sub_f, sub_msg| {
                                                prog_ref.set(
                                                    obj_base_frac + sub_f * obj_span_frac,
                                                    sub_msg,
                                                );
                                            }),
                                        );
                                    crate::gcode::refresh_planned_gcode_nozzle_sdf(
                                        &mut p,
                                        |xyz| ffi::sample_object_sdf(oid, xyz),
                                    );
                                    planned_objects.push((o_name, p, o_tf));
                                }
                            }
                        }

                        if planned_objects.is_empty() {
                            Some(None)
                        } else {
                            progress.set(0.83, "Combining plate G-code & metadata...");
                            let combined =
                                crate::gcode::combine_planned_objects_into_plate_result(
                                    &planned_objects,
                                    &config_clone,
                                );
                            Some(Some(combined))
                        }
                    } else {
                        None
                    };

                // 4. Optional 3D toolpath preview mesh construction
                let built_mesh = if build_mesh_in_task {
                    progress.set(0.87, "Building 3D toolpath preview mesh...");
                    let planned_ref = match &updated_planned_gcode {
                        Some(Some(ref p)) => Some(p),
                        Some(None) => None,
                        None => existing_planned_for_mesh.as_ref(),
                    };
                    if let Some(p) = planned_ref {
                        if !p.layers.is_empty() {
                            if reset_layer_range {
                                mesh_filter.selected_layer_min_idx = 0;
                                mesh_filter.selected_layer_max_idx = p.layers.len() - 1;
                            } else {
                                mesh_filter.selected_layer_max_idx =
                                    mesh_filter.selected_layer_max_idx.min(p.layers.len() - 1);
                                mesh_filter.selected_layer_min_idx = mesh_filter
                                    .selected_layer_min_idx
                                    .min(mesh_filter.selected_layer_max_idx);
                            }
                        }
                    }
                    let geom = mesh_filter.build_geometry(
                        planned_ref,
                        active_sel_mixed.as_ref(),
                        &config_clone,
                    );
                    let render_tf = if planned_ref.is_some()
                        && (updated_planned_gcode.is_some() || existing_planned_is_world)
                    {
                        Transform::IDENTITY
                    } else {
                        sel_world_tf
                    };
                    progress.set(0.99, "Finalizing 3D toolpath preview...");
                    Some((geom, render_tf))
                } else {
                    None
                };

                PostSliceTaskOutput {
                    plate_index: task_plate_index,
                    selected_obj_id: Some(sel_oid),
                    invalidated_other_plates_mixed: invalidate_other_plates_mixed,
                    reset_layer_range,
                    remapped_triangle_slots,
                    updated_mixed_toolpaths,
                    updated_planned_gcode,
                    built_mesh,
                }
            });

            vis_state.pending_post_slice_task = Some(task);
        }
    }

    let toolpath_render_tf =
        if vis_state.planned_gcode_is_world_space && vis_state.cached_planned_gcode.is_some() {
            Transform::IDENTITY
        } else {
            obj_world_tf
        };

    if !vis_state.show_toolpaths {
        hide_all_toolpath_entities(&vis_state, &mut toolpaths_query);
    } else if vis_state.dirty_toolpaths_mesh && vis_state.pending_post_slice_task.is_none() {
        vis_state.dirty_toolpaths_mesh = false;
        rebuild_and_sync_toolpath_meshes(
            &mut commands,
            &mut vis_state,
            &config,
            &mut meshes,
            &mut materials,
            &mut toolpaths_query,
            toolpath_render_tf,
        );
    }

    // Optional crisp gizmo centerlines and travel/Z-hop trajectories on top of toolpath beads
    if vis_state.show_toolpaths
        && (vis_state.show_toolpath_centerlines
            || vis_state.show_travel_moves
            || vis_state.show_z_hops)
    {
        if let Some(ref planned) = vis_state.cached_planned_gcode {
            let total = planned.layers.len();
            if total > 0 {
                let stride = vis_state.selected_layer_stride.max(1);
                let lo = vis_state.selected_layer_min_idx.min(total - 1);
                let hi = vis_state.selected_layer_max_idx.min(total - 1).max(lo);
                let highlighted = vis_state.highlighted_selected_layer;
                let active_scrub_layer = highlighted.unwrap_or(hi);

                for (idx, layer_p) in planned.layers.iter().enumerate() {
                    let is_hovered = highlighted == Some(idx);
                    let in_range = idx >= lo && idx <= hi;
                    let matches_stride =
                        (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
                    if !is_hovered && (!in_range || !matches_stride) {
                        continue;
                    }
                    let max_vis_moves = if idx == active_scrub_layer {
                        vis_state.layer_move_scrub_limit.unwrap_or(usize::MAX)
                    } else {
                        usize::MAX
                    };
                    let mut vis_move_count = 0usize;
                    for seg in &layer_p.segments {
                        if !vis_state.is_planned_segment_visible(seg) {
                            continue;
                        }
                        vis_move_count += 1;
                        if vis_move_count > max_vis_moves {
                            break;
                        }
                        let is_travel = matches!(
                            seg.kind,
                            crate::gcode::PlannedMoveKind::Travel
                                | crate::gcode::PlannedMoveKind::ZHop
                        );
                        if !is_travel && !vis_state.show_toolpath_centerlines {
                            continue;
                        }
                        let rgba = compute_planned_segment_color(
                            seg,
                            planned,
                            vis_state.toolpath_color_mode,
                            &config,
                        );
                        let line_col = Color::srgb(
                            (rgba[0] * 1.15).min(1.0),
                            (rgba[1] * 1.15).min(1.0),
                            (rgba[2] * 1.15).min(1.0),
                        );
                        let p0 = toolpath_render_tf.transform_point(seg.p0_local);
                        let p1 = toolpath_render_tf.transform_point(seg.p1_local);
                        gizmos.line(p0, p1, line_col);
                    }
                }
            }
        }
    }

    // 3. Update 2D Cross-Section Slice Plane in-place using AlphaMode::Mask(0.5)
    //    so hardware depth testing is active and there is zero transparency sort flicker!
    if !vis_state.show_slice_plane {
        if let Some(ent) = vis_state.slice_entity {
            if let Ok((_, mut vis)) = slice_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    } else if vis_state.dirty_slice && !vis_state.is_busy() {
        vis_state.dirty_slice = false;

        let slice = ffi::sample_field_slice(
            obj_id,
            &vis_state.active_field_name,
            vis_state.slice_axis.as_u8(),
            vis_state.slice_normalized_pos,
            vis_state.slice_resolution,
        );

        if slice.valid && slice.width > 1 && slice.height > 1 {
            vis_state.slice_plane_coord = slice.plane_coord;

            let mut selected_iso_list: Vec<f32> = Vec::new();
            let mut highlighted_iso: Option<f32> = None;
            if vis_state.show_selected_layers && vis_state.active_field_name == "eikonal" {
                if let Some(ref res) = vis_state.cached_selected_layers {
                    let total = res.layers.len();
                    if total > 0 {
                        let stride = vis_state.selected_layer_stride.max(1);
                        let lo = vis_state.selected_layer_min_idx.min(total - 1);
                        let hi = vis_state.selected_layer_max_idx.min(total - 1).max(lo);
                        for (idx, l) in res.layers.iter().enumerate() {
                            if vis_state.highlighted_selected_layer == Some(idx) {
                                highlighted_iso = Some(l.isovalue);
                            }
                            if idx >= lo
                                && idx <= hi
                                && ((idx.saturating_sub(lo) % stride == 0)
                                    || l.is_first_layer
                                    || l.is_top_layer)
                            {
                                selected_iso_list.push(l.isovalue);
                            }
                        }
                    }
                }
            }

            let rgba_bytes = rasterize_slice_to_rgba(
                &slice,
                &vis_state.active_field_name,
                vis_state.isovalue,
                vis_state.colormap,
                vis_state.show_iso_contours,
                vis_state.fade_exterior_background,
                &selected_iso_list,
                highlighted_iso,
            );

            let texture = Image::new(
                Extent3d {
                    width: slice.width,
                    height: slice.height,
                    depth_or_array_layers: 1,
                },
                TextureDimension::D2,
                rgba_bytes,
                TextureFormat::Rgba8UnormSrgb,
                RenderAssetUsages::default(),
            );

            let o = Vec3::from_array(slice.origin);
            let u = Vec3::from_array(slice.u_vec);
            let v = Vec3::from_array(slice.v_vec);
            let n = Vec3::from_array(slice.normal);

            let quad_positions = vec![
                o.to_array(),
                (o + u).to_array(),
                (o + u + v).to_array(),
                (o + v).to_array(),
            ];
            let quad_normals = vec![n.to_array(); 4];
            let quad_uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
            let quad_indices = vec![0, 1, 2, 0, 2, 3];

            let quad_mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, quad_positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, quad_normals)
            .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, quad_uvs)
            .with_inserted_indices(Indices::U32(quad_indices));

            let new_img_h = images.add(texture);

            if let (Some(ent), Some(mesh_h), Some(mat_h)) = (
                vis_state.slice_entity,
                vis_state.slice_mesh_handle.clone(),
                vis_state.slice_mat_handle.clone(),
            ) {
                meshes.insert(&mesh_h, quad_mesh);
                if let Some(old_img_h) = vis_state.slice_image_handle.replace(new_img_h.clone()) {
                    images.remove(&old_img_h);
                }
                // Mutate StandardMaterial so Bevy's MaterialPlugin rebuilds the GPU bind group
                // with the newly uploaded texture on this frame.
                if let Some(mat) = materials.get_mut(&mat_h) {
                    mat.base_color_texture = Some(new_img_h);
                }
                if let Ok((mut tf, mut vis)) = slice_query.get_mut(ent) {
                    *tf = obj_world_tf;
                    *vis = Visibility::Inherited;
                }
            } else {
                let mesh_h = meshes.add(quad_mesh);
                let mat_h = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    base_color_texture: Some(new_img_h.clone()),
                    unlit: true,
                    alpha_mode: AlphaMode::Mask(0.5),
                    depth_bias: 16.0,
                    double_sided: true,
                    cull_mode: None,
                    ..default()
                });
                let ent = commands
                    .spawn((
                        PbrBundle {
                            mesh: mesh_h.clone(),
                            material: mat_h.clone(),
                            transform: obj_world_tf,
                            ..default()
                        },
                        FieldSliceEntity,
                    ))
                    .id();
                vis_state.slice_entity = Some(ent);
                vis_state.slice_mesh_handle = Some(mesh_h);
                vis_state.slice_mat_handle = Some(mat_h);
                vis_state.slice_image_handle = Some(new_img_h);
            }
        }
    }

    // 4. Keep active isosurface, selected layers, toolpaths, & slice entities locked to the selected object's Transform
    if vis_state.show_isosurface {
        if let Some(ent) = vis_state.isosurface_entity {
            if let Ok((mut tf, _)) = iso_query.get_mut(ent) {
                if *tf != obj_world_tf {
                    *tf = obj_world_tf;
                }
            }
        }
    }
    if vis_state.show_selected_layers && vis_state.show_selected_layer_surfaces {
        if let Some(ent) = vis_state.selected_layers_entity {
            if let Ok((mut tf, _)) = selected_layers_query.get_mut(ent) {
                if *tf != obj_world_tf {
                    *tf = obj_world_tf;
                }
            }
        }
    }
    if vis_state.show_toolpaths {
        let mut tf_moved = false;
        if let Some(ent) = vis_state.toolpaths_entity {
            if let Ok((mut tf, vis)) = toolpaths_query.get_mut(ent) {
                if *vis != Visibility::Hidden && *tf != toolpath_render_tf {
                    *tf = toolpath_render_tf;
                    tf_moved = true;
                }
            }
        }
        for part in &vis_state.hq_toolpath_parts {
            if let Ok((mut tf, vis)) = toolpaths_query.get_mut(part.entity) {
                if *vis != Visibility::Hidden && *tf != toolpath_render_tf {
                    *tf = toolpath_render_tf;
                    tf_moved = true;
                }
            }
        }
        if tf_moved {
            vis_state.dirty_planned_gcode = true;
        }
    }
    if vis_state.show_slice_plane {
        if let Some(ent) = vis_state.slice_entity {
            if let Ok((mut tf, _)) = slice_query.get_mut(ent) {
                if *tf != obj_world_tf {
                    *tf = obj_world_tf;
                }
            }
        }
    }
}

impl FieldVisualizationState {
    /// Returns whether a `PlannedSegment` should be rendered given the current UI visibility toggles.
    pub fn is_planned_segment_visible(&self, seg: &crate::gcode::PlannedSegment) -> bool {
        match seg.kind {
            crate::gcode::PlannedMoveKind::OuterWall => self.show_outer_walls,
            crate::gcode::PlannedMoveKind::InnerWall => self.show_inner_walls,
            crate::gcode::PlannedMoveKind::Bridge => self.show_bridges,
            crate::gcode::PlannedMoveKind::WaveOverhang => self.show_wave_overhangs,
            crate::gcode::PlannedMoveKind::SolidInfill => self.show_solid_infill,
            crate::gcode::PlannedMoveKind::SparseInfill => self.show_sparse_infill,
            crate::gcode::PlannedMoveKind::GapFill => self.show_gap_fill,
            crate::gcode::PlannedMoveKind::Support => self.show_supports,
            crate::gcode::PlannedMoveKind::SupportInterface => self.show_support_interface,
            crate::gcode::PlannedMoveKind::PurgeTower => self.show_purge_tower,
            crate::gcode::PlannedMoveKind::ScarfSeam => self.show_scarf_seams,
            crate::gcode::PlannedMoveKind::Wipe => self.show_wipes,
            crate::gcode::PlannedMoveKind::Travel => self.show_travel_moves,
            crate::gcode::PlannedMoveKind::ZHop => self.show_z_hops,
        }
    }

    /// Returns whether a raw `ExtrusionPath` should be rendered given its `feature_kind`.
    pub fn is_raw_path_visible(&self, feature_kind: u8) -> bool {
        match feature_kind {
            0 => self.show_outer_walls,
            1 => self.show_inner_walls,
            2 => self.show_solid_infill,
            3 => self.show_gap_fill,
            4 => self.show_sparse_infill,
            5 => self.show_bridges,
            6 => self.show_wave_overhangs,
            7 => self.show_supports,
            8 => self.show_support_interface,
            _ => self.show_solid_infill,
        }
    }

    /// Returns the total number of available layers across planned G-code, raw toolpaths, or selected isosurfaces.
    pub fn total_layer_count(&self) -> usize {
        if self.show_toolpaths {
            if let Some(ref planned) = self.cached_planned_gcode {
                return planned.layers.len();
            }
            if let Some(ref tp_res) = self.cached_toolpaths {
                return tp_res.layers.len();
            }
        }
        if let Some(ref layers_res) = self.cached_selected_layers {
            if !layers_res.layers.is_empty() {
                return layers_res.layers.len();
            }
        }
        if let Some(ref planned) = self.cached_planned_gcode {
            return planned.layers.len();
        }
        if let Some(ref tp_res) = self.cached_toolpaths {
            return tp_res.layers.len();
        }
        0
    }

    /// Returns the currently active/inspected 0-based layer index (`highlighted_selected_layer` or `selected_layer_max_idx`).
    pub fn active_layer_index(&self) -> Option<usize> {
        let total = self.total_layer_count();
        if total == 0 {
            return None;
        }
        Some(
            self.highlighted_selected_layer
                .unwrap_or(self.selected_layer_max_idx)
                .min(total - 1),
        )
    }

    /// Returns the total number of visible moves/segments on the currently active layer.
    pub fn active_layer_total_moves(&self) -> usize {
        if self.show_toolpaths {
            if let Some(ref planned) = self.cached_planned_gcode {
                if planned.layers.is_empty() {
                    return 0;
                }
                let active_idx = self
                    .highlighted_selected_layer
                    .unwrap_or(self.selected_layer_max_idx)
                    .min(planned.layers.len() - 1);
                return planned.layers[active_idx]
                    .segments
                    .iter()
                    .filter(|s| self.is_planned_segment_visible(s))
                    .count();
            }
            if let Some(ref tp_res) = self.cached_toolpaths {
                if tp_res.layers.is_empty() {
                    return 0;
                }
                let active_idx = self
                    .highlighted_selected_layer
                    .unwrap_or(self.selected_layer_max_idx)
                    .min(tp_res.layers.len() - 1);
                return tp_res.layers[active_idx]
                    .paths
                    .iter()
                    .filter(|p| self.is_raw_path_visible(p.feature_kind) && p.points.len() >= 2)
                    .map(|p| {
                        if p.is_closed && p.points.len() >= 3 {
                            p.points.len()
                        } else {
                            p.points.len() - 1
                        }
                    })
                    .sum();
            }
        }
        if self.show_selected_layers {
            if let Some(ref layers_res) = self.cached_selected_layers {
                if !layers_res.layers.is_empty() {
                    let active_idx = self
                        .highlighted_selected_layer
                        .unwrap_or(self.selected_layer_max_idx)
                        .min(layers_res.layers.len() - 1);
                    return layers_res.layers[active_idx].contour_segments.len() / 6;
                }
            }
        }
        0
    }

    /// Computes the toolhead position, local `NozzleFrame` (`AGENTS.md` Rule 1), bed-coordinate
    /// gravity direction (`AGENTS.md` Rule 2), and 5-axis rotary angles `(B, C)` at the active
    /// layer and within-layer scrub position.
    pub fn compute_active_scrub_pose(
        &self,
        config: &SlicerConfig,
        obj_world_tf: Transform,
    ) -> Option<ScrubbedToolheadPose> {
        // 1. Planned G-code segments when toolpath visualization is enabled
        if self.show_toolpaths {
            if let Some(ref planned) = self.cached_planned_gcode {
                if !planned.layers.is_empty() {
                    let active_idx = self
                        .highlighted_selected_layer
                        .unwrap_or(self.selected_layer_max_idx)
                        .min(planned.layers.len() - 1);
                    let layer_p = &planned.layers[active_idx];
                    let total_vis = layer_p
                        .segments
                        .iter()
                        .filter(|s| self.is_planned_segment_visible(s))
                        .count();

                    let (seg_opt, move_idx, total_moves) = if total_vis > 0 {
                        let m_idx = self
                            .layer_move_scrub_limit
                            .unwrap_or(total_vis)
                            .clamp(1, total_vis);
                        let seg = layer_p
                            .segments
                            .iter()
                            .filter(|s| self.is_planned_segment_visible(s))
                            .nth(m_idx - 1);
                        (seg, m_idx, total_vis)
                    } else {
                        // Fallback if all visible feature types on this layer were unchecked
                        let len = layer_p.segments.len();
                        (layer_p.segments.last(), len.max(1), len.max(1))
                    };

                    if let Some(seg) = seg_opt {
                        let pose_tf = if self.planned_gcode_is_world_space {
                            Transform::IDENTITY
                        } else {
                            obj_world_tf
                        };
                        let seg_start_bed = pose_tf.transform_point(seg.p0_local);
                        let tip_bed = pose_tf.transform_point(seg.p1_local);
                        let travel_dir_bed = (tip_bed - seg_start_bed)
                            .try_normalize()
                            .unwrap_or(Vec3::X);
                        let nozzle_axis_bed = (pose_tf.rotation * seg.nozzle_local_1)
                            .try_normalize()
                            .or_else(|| seg.nozzle_bed_1.try_normalize())
                            .unwrap_or_else(|| {
                                config.query_nozzle_frame(tip_bed, travel_dir_bed).nozzle_axis
                            });
                        let frame_bed =
                            NozzleFrame::from_nozzle_and_travel(nozzle_axis_bed, travel_dir_bed);
                        let gravity_bed = if config.machine.gravity_couples_to_bed_tilt() {
                            config.machine.gravity_bed_at_bc(seg.b1_deg, seg.c1_deg)
                        } else {
                            config.query_gravity_bed(tip_bed)
                        };
                        return Some(ScrubbedToolheadPose {
                            tip_bed,
                            seg_start_bed,
                            frame_bed,
                            gravity_bed,
                            tool_id: seg.tool_id as usize,
                            b_deg: seg.b1_deg,
                            c_deg: seg.c1_deg,
                            layer_index: active_idx,
                            move_index: move_idx,
                            total_layer_moves: total_moves,
                            move_label: seg.kind.label(),
                            speed_mms: seg.cruise_speed_mms,
                            width_mm: seg.width_mm,
                            height_mm: seg.height_mm,
                            tool_iso_angle_deg: seg.tool_iso_angle_deg,
                            uses_surface_normal: seg.uses_surface_normal,
                            sdf_mm: seg.sdf_mm_1,
                        });
                    }
                }
            }

            // 2. Raw toolpath polylines when planned G-code is not yet available
            if let Some(ref tp_res) = self.cached_toolpaths {
                if !tp_res.layers.is_empty() {
                    let active_idx = self
                        .highlighted_selected_layer
                        .unwrap_or(self.selected_layer_max_idx)
                        .min(tp_res.layers.len() - 1);
                    let layer_tp = &tp_res.layers[active_idx];
                    let total_vis: usize = layer_tp
                        .paths
                        .iter()
                        .filter(|p| self.is_raw_path_visible(p.feature_kind) && p.points.len() >= 2)
                        .map(|p| {
                            if p.is_closed && p.points.len() >= 3 {
                                p.points.len()
                            } else {
                                p.points.len() - 1
                            }
                        })
                        .sum();

                    if total_vis > 0 {
                        let m_idx = self
                            .layer_move_scrub_limit
                            .unwrap_or(total_vis)
                            .clamp(1, total_vis);
                        let mut rem = m_idx - 1;
                        for path in &layer_tp.paths {
                            if !self.is_raw_path_visible(path.feature_kind) || path.points.len() < 2
                            {
                                continue;
                            }
                            let n = path.points.len();
                            let seg_count = if path.is_closed && n >= 3 { n } else { n - 1 };
                            if rem < seg_count {
                                let pt0 = &path.points[rem];
                                let pt1 = &path.points[(rem + 1) % n];
                                let seg_start_bed =
                                    obj_world_tf.transform_point(Vec3::new(pt0.x, pt0.y, pt0.z));
                                let tip_bed =
                                    obj_world_tf.transform_point(Vec3::new(pt1.x, pt1.y, pt1.z));
                                let travel_dir_bed = (tip_bed - seg_start_bed)
                                    .try_normalize()
                                    .unwrap_or(Vec3::X);
                                let nozzle_axis_bed = (obj_world_tf.rotation
                                    * Vec3::new(pt1.nx, pt1.ny, pt1.nz))
                                .try_normalize()
                                .unwrap_or_else(|| {
                                    config.query_nozzle_frame(tip_bed, travel_dir_bed).nozzle_axis
                                });
                                let frame_bed = NozzleFrame::from_nozzle_and_travel(
                                    nozzle_axis_bed,
                                    travel_dir_bed,
                                );
                                let (b_deg, c_deg, _) = config.machine.nozzle_axis_to_bc_deg(
                                    nozzle_axis_bed,
                                    0.0,
                                    0.0,
                                    config.slicing.max_nonplanar_slope_deg,
                                );
                                let gravity_bed = if config.machine.gravity_couples_to_bed_tilt() {
                                    config.machine.gravity_bed_at_bc(b_deg, c_deg)
                                } else {
                                    config.query_gravity_bed(tip_bed)
                                };
                                let kind_label =
                                    crate::gcode::PlannedMoveKind::from_feature_kind(
                                        path.feature_kind,
                                    )
                                    .label();
                                return Some(ScrubbedToolheadPose {
                                    tip_bed,
                                    seg_start_bed,
                                    frame_bed,
                                    gravity_bed,
                                    tool_id: config.active_tool_index,
                                    b_deg,
                                    c_deg,
                                    layer_index: active_idx,
                                    move_index: m_idx,
                                    total_layer_moves: total_vis,
                                    move_label: kind_label,
                                    speed_mms: 0.0,
                                    width_mm: pt1.width_mm,
                                    height_mm: pt1.height_mm,
                                    tool_iso_angle_deg: pt1.tool_iso_angle_deg,
                                    uses_surface_normal: pt1.uses_surface_normal,
                                    sdf_mm: pt1.sdf_mm,
                                });
                            }
                            rem -= seg_count;
                        }
                    }
                }
            }
        }

        // 3. Selected slicing isosurfaces when scrubbing layers/contours without toolpaths
        if self.show_selected_layers {
            if let Some(ref layers_res) = self.cached_selected_layers {
                if !layers_res.layers.is_empty() {
                    let active_idx = self
                        .highlighted_selected_layer
                        .unwrap_or(self.selected_layer_max_idx)
                        .min(layers_res.layers.len() - 1);
                    let layer = &layers_res.layers[active_idx];
                    let num_segs = layer.contour_segments.len() / 6;
                    if num_segs > 0 {
                        let m_idx = self
                            .layer_move_scrub_limit
                            .unwrap_or(num_segs)
                            .clamp(1, num_segs);
                        let base = (m_idx - 1) * 6;
                        let p0_local = Vec3::new(
                            layer.contour_segments[base],
                            layer.contour_segments[base + 1],
                            layer.contour_segments[base + 2],
                        );
                        let p1_local = Vec3::new(
                            layer.contour_segments[base + 3],
                            layer.contour_segments[base + 4],
                            layer.contour_segments[base + 5],
                        );
                        let mut best_normal_local = Vec3::Z;
                        let mut best_d2 = f32::MAX;
                        for v in &layer.vertices {
                            let vp = Vec3::new(v.x, v.y, v.z);
                            let d2 = vp.distance_squared(p1_local);
                            if d2 < best_d2 {
                                best_d2 = d2;
                                best_normal_local = Vec3::new(v.nx, v.ny, v.nz);
                            }
                        }
                        let seg_start_bed = obj_world_tf.transform_point(p0_local);
                        let tip_bed = obj_world_tf.transform_point(p1_local);
                        let travel_dir_bed = (tip_bed - seg_start_bed)
                            .try_normalize()
                            .unwrap_or(Vec3::X);
                        let nozzle_axis_bed = (obj_world_tf.rotation * best_normal_local)
                            .try_normalize()
                            .unwrap_or_else(|| {
                                config.query_nozzle_frame(tip_bed, travel_dir_bed).nozzle_axis
                            });
                        let frame_bed =
                            NozzleFrame::from_nozzle_and_travel(nozzle_axis_bed, travel_dir_bed);
                        let (b_deg, c_deg, _) = config.machine.nozzle_axis_to_bc_deg(
                            nozzle_axis_bed,
                            0.0,
                            0.0,
                            config.slicing.max_nonplanar_slope_deg,
                        );
                        let gravity_bed = if config.machine.gravity_couples_to_bed_tilt() {
                            config.machine.gravity_bed_at_bc(b_deg, c_deg)
                        } else {
                            config.query_gravity_bed(tip_bed)
                        };
                        return Some(ScrubbedToolheadPose {
                            tip_bed,
                            seg_start_bed,
                            frame_bed,
                            gravity_bed,
                            tool_id: config.active_tool_index,
                            b_deg,
                            c_deg,
                            layer_index: active_idx,
                            move_index: m_idx,
                            total_layer_moves: num_segs,
                            move_label: "Isosurface Contour",
                            speed_mms: 0.0,
                            width_mm: config.slicing.outer_wall_width.target_mm,
                            height_mm: layer.mean_thickness_mm,
                            tool_iso_angle_deg: 0.0,
                            uses_surface_normal: false,
                            sdf_mm: 0.0,
                        });
                    }
                }
            }
        }

        None
    }
}

/// Synchronizes 3D viewport key-light shadows, rear lithophane backlight intensity, and live
/// TD-computed extrusion opacity/PBR materials — gated by a deterministic content hash so GPU
/// material bind groups are never rebuilt redundantly during idle or camera-orbit frames.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn sync_viewport_pbr_quality_system(
    config: Res<SlicerConfig>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut camera_query: Query<&mut bevy::core_pipeline::core_3d::Camera3d>,
    mut key_lights: Query<
        &mut DirectionalLight,
        (
            With<crate::ViewportKeyLight>,
            Without<crate::ViewportRimBackLight>,
        ),
    >,
    mut rim_lights: Query<
        &mut DirectionalLight,
        (
            With<crate::ViewportRimBackLight>,
            Without<crate::ViewportKeyLight>,
        ),
    >,
) {
    use std::hash::{Hash, Hasher};

    let hq_pbr = vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr;
    let shadows_active = hq_pbr && vis_state.enable_hq_shadows;
    let backlight_active = vis_state.lithophane_backlight_preview
        || (config.slicing.color_mixing.enabled
            && config.slicing.color_mixing.lighting_mode
                != crate::color_optics::OpticalLightingMode::FrontLitReflectance);
    let opacity_bits = vis_state.toolpath_opacity.clamp(0.10, 1.50).to_bits();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    vis_state.toolpath_quality_mode.hash(&mut hasher);
    vis_state.enable_pbr_translucency.hash(&mut hasher);
    shadows_active.hash(&mut hasher);
    backlight_active.hash(&mut hasher);
    (vis_state.toolpath_color_mode as u8).hash(&mut hasher);
    opacity_bits.hash(&mut hasher);
    vis_state.hq_toolpath_parts.len().hash(&mut hasher);
    for tool in &config.tools {
        tool.filament_type.hash(&mut hasher);
        tool.name.hash(&mut hasher);
        tool.effective_transmission_distance_mm().to_bits().hash(&mut hasher);
        for c in tool.filament_color_srgba {
            c.to_bits().hash(&mut hasher);
        }
    }
    let sig = hasher.finish();

    if vis_state.last_hq_pbr_sync_sig == Some(sig) {
        return;
    }
    vis_state.last_hq_pbr_sync_sig = Some(sig);

    // Keep screen-space specular transmission disabled (0 steps): its spiral/checkerboard noise taps
    // shimmer on sub-millimeter beads without TAA, whereas TD-derived alpha opacity + diffuse_transmission
    // is completely noise-free and matches physical Kubelka–Munk hiding power directly.
    for mut cam3d in &mut camera_query {
        if cam3d.screen_space_specular_transmission_steps != 0 {
            cam3d.screen_space_specular_transmission_steps = 0;
        }
    }

    for mut key in &mut key_lights {
        key.shadows_enabled = shadows_active;
        key.illuminance = if backlight_active && hq_pbr {
            6_500.0
        } else {
            11_500.0
        };
    }

    for mut rim in &mut rim_lights {
        rim.illuminance = if backlight_active {
            24_000.0
        } else if hq_pbr {
            4_500.0
        } else {
            2_800.0
        };
    }

    if let Some(ref mat_h) = vis_state.toolpaths_mat_handle {
        if let Some(mat) = materials.get_mut(mat_h) {
            *mat = build_fast_toolpath_material(vis_state.toolpath_opacity);
        }
    }
    for part in &vis_state.hq_toolpath_parts {
        if let Some(mat) = materials.get_mut(&part.mat_handle) {
            *mat = build_hq_bucket_pbr_material(
                part.tier,
                part.mat_slot,
                part.mean_thickness_mm,
                &vis_state,
                &config,
            );
        }
    }
}

fn hide_all_toolpath_entities(
    vis_state: &FieldVisualizationState,
    toolpaths_query: &mut Query<
        (&mut Transform, &mut Visibility),
        (
            With<ToolpathsMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
) {
    if let Some(ent) = vis_state.toolpaths_entity {
        if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
            *vis = Visibility::Hidden;
        }
    }
    for part in &vis_state.hq_toolpath_parts {
        if let Ok((_, mut vis)) = toolpaths_query.get_mut(part.entity) {
            *vis = Visibility::Hidden;
        }
    }
}

fn build_fast_toolpath_material(opacity: f32) -> StandardMaterial {
    let alpha = opacity.clamp(0.10, 1.0);
    if alpha < 0.995 {
        StandardMaterial {
            base_color: Color::srgba(1.0, 1.0, 1.0, alpha),
            metallic: 0.0,
            perceptual_roughness: 0.52,
            reflectance: 0.22,
            alpha_mode: AlphaMode::Blend,
            depth_bias: 14.0,
            double_sided: false,
            cull_mode: Some(wgpu::Face::Back),
            ..default()
        }
    } else {
        StandardMaterial {
            base_color: Color::WHITE,
            metallic: 0.0,
            perceptual_roughness: 0.56,
            reflectance: 0.16,
            alpha_mode: AlphaMode::Opaque,
            depth_bias: 14.0,
            double_sided: true,
            cull_mode: None,
            ..default()
        }
    }
}

/// Synthesizes a flicker-free physically based `StandardMaterial` for a High-Quality extrusion
/// sub-mesh bucket `(tier, mat_slot)`:
/// - Computes physical single-bead opacity $\alpha(d, \text{TD}) = 1 - T_{\text{eff}}(d, \text{TD})$
///   directly from each filament's Kubelka–Munk `FilamentOpticalModel` and `mean_thickness_mm`.
/// - Assigns unique positive `depth_bias` (`12.xxx` Core -> `13.xxx` Inner Shell -> `14.xxx` Outer Shell)
///   eliminating both coplanar Z-fighting against the CAD/isosurface meshes and `Transparent3d` sort ties.
fn build_hq_bucket_pbr_material(
    tier: u8,
    mat_slot: u16,
    mean_thickness_mm: f32,
    vis_state: &FieldVisualizationState,
    config: &SlicerConfig,
) -> StandardMaterial {
    let opacity_mult = vis_state.toolpath_opacity.clamp(0.10, 1.50);

    let (ior, roughness, reflectance, metallic, clearcoat, cc_roughness, td_mm, color_srgb) =
        if (mat_slot as usize) < config.tools.len() {
            let tool = &config.tools[mat_slot as usize];
            let upper_type = tool.filament_type.to_ascii_uppercase();
            let upper_name = tool.name.to_ascii_uppercase();
            let is_silk = upper_type.contains("SILK") || upper_name.contains("SILK");
            let is_filled = upper_type.contains("CF")
                || upper_type.contains("GF")
                || upper_type.contains("MATTE")
                || upper_name.contains("MATTE");
            let is_metal = upper_type.contains("METAL")
                || upper_type.contains("BRONZE")
                || upper_type.contains("COPPER")
                || upper_type.contains("STEEL");

            let (ior, rough, refl, met, cc, cc_r) = if is_metal {
                (1.50, 0.36, 0.65, 0.72, 0.12, 0.24)
            } else if is_silk {
                (1.48, 0.22, 0.64, 0.26, 0.55, 0.16)
            } else if is_filled {
                (1.52, 0.76, 0.22, 0.04, 0.0, 0.50)
            } else if upper_type.contains("PETG")
                || upper_type.contains("PCTG")
                || upper_type.contains("PC")
                || upper_type.contains("CLEAR")
            {
                (1.57, 0.24, 0.54, 0.0, 0.44, 0.14)
            } else if upper_type.contains("ABS") || upper_type.contains("ASA") {
                (1.54, 0.50, 0.36, 0.0, 0.08, 0.34)
            } else if upper_type.contains("TPU")
                || upper_type.contains("TPE")
                || upper_type.contains("FLEX")
            {
                (1.50, 0.48, 0.34, 0.0, 0.12, 0.30)
            } else {
                (1.46, 0.33, 0.46, 0.02, 0.32, 0.20)
            };

            let td = tool.effective_transmission_distance_mm().clamp(0.1, 30.0);
            let rgb = [
                tool.filament_color_srgba[0].clamp(0.0, 1.0),
                tool.filament_color_srgba[1].clamp(0.0, 1.0),
                tool.filament_color_srgba[2].clamp(0.0, 1.0),
            ];
            (ior, rough, refl, met, cc, cc_r, td, rgb)
        } else if mat_slot == 256 {
            (1.46, 0.54, 0.30, 0.0, 0.06, 0.35, 1.8, [0.22, 0.88, 0.62])
        } else if mat_slot == 257 {
            (1.46, 0.50, 0.28, 0.0, 0.05, 0.35, 1.2, [0.68, 0.76, 0.86])
        } else {
            let td = config
                .active_tool()
                .map(|t| t.effective_transmission_distance_mm())
                .unwrap_or(2.5)
                .clamp(0.1, 30.0);
            (1.47, 0.34, 0.44, 0.02, 0.28, 0.22, td, [0.92, 0.92, 0.92])
        };

    let td_opacity_on = vis_state.enable_pbr_translucency
        && vis_state.toolpath_color_mode == ToolpathColorMode::ByToolOrFilament;

    // Compute physical single-bead opacity directly from the filament's Transmission Distance (TD)!
    let base_td_opacity = if td_opacity_on {
        crate::color_optics::td_opacity_for_thickness_mm(td_mm, mean_thickness_mm)
    } else {
        1.0
    };

    // Core/backing tier (`tier == 0`) forms a solid Z-buffered opaque foundation unless the user
    // explicitly drags `opacity_mult < 1.0` to X-ray inside the core.
    let final_alpha = if tier == 0 && opacity_mult >= 0.995 {
        1.0
    } else if td_opacity_on {
        (base_td_opacity * opacity_mult).clamp(0.08, 1.0)
    } else {
        opacity_mult.clamp(0.08, 1.0)
    };

    let trans_01 = ((td_mm - 0.45) / (td_mm + 3.0)).clamp(0.0, 0.90);
    let diffuse_trans = if vis_state.enable_pbr_translucency {
        (0.08 + 0.48 * trans_01).clamp(0.0, 0.62)
    } else {
        0.0
    };

    // Positive hardware Z-bias (> 10.0) prevents coplanar Z-fighting against the CAD mesh (0.0)
    // and selected isosurfaces (10.0), while unique per-slot fractional offsets guarantee
    // deterministic tie-free back-to-front ordering in `Transparent3d`: Core (12.xxx) < Inner (13.xxx) < Outer (14.xxx).
    let slot_tie_break = (mat_slot.min(300) as f32) * 0.002;
    let depth_bias = match tier {
        0 => 12.0 + slot_tie_break,
        1 => 13.0 + slot_tie_break,
        _ => 14.0 + slot_tie_break,
    };

    let alpha_mode = if final_alpha >= 0.985 {
        AlphaMode::Opaque
    } else {
        AlphaMode::Blend
    };

    StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, final_alpha),
        metallic,
        perceptual_roughness: roughness,
        reflectance,
        ior,
        clearcoat,
        clearcoat_perceptual_roughness: cc_roughness,
        diffuse_transmission: diffuse_trans,
        specular_transmission: 0.0,
        thickness: mean_thickness_mm.clamp(0.10, 2.50),
        attenuation_distance: if vis_state.enable_pbr_translucency {
            (0.55 * td_mm).clamp(0.08, 30.0)
        } else {
            f32::INFINITY
        },
        attenuation_color: Color::srgb(
            color_srgb[0].max(0.03),
            color_srgb[1].max(0.03),
            color_srgb[2].max(0.03),
        ),
        alpha_mode,
        depth_bias,
        double_sided: false,
        cull_mode: Some(wgpu::Face::Back),
        ..default()
    }
}

fn clone_planned_without_gcode_text(
    p: &crate::gcode::PlannedGcodeResult,
) -> crate::gcode::PlannedGcodeResult {
    crate::gcode::PlannedGcodeResult {
        layers: p.layers.clone(),
        gcode_text: String::new(),
        total_time_s: p.total_time_s,
        total_filament_mm: p.total_filament_mm,
        total_filament_volume_mm3: p.total_filament_volume_mm3,
        total_filament_grams: p.total_filament_grams,
        bbox_min_bed: p.bbox_min_bed,
        bbox_max_bed: p.bbox_max_bed,
        exclude_object_name: String::new(),
        exclude_object_center_bed: p.exclude_object_center_bed,
        exclude_object_polygon_bed: Vec::new(),
        extruding_move_count: p.extruding_move_count,
        travel_move_count: p.travel_move_count,
        retract_count: p.retract_count,
        toolchange_count: p.toolchange_count,
        min_speed_mms: p.min_speed_mms,
        max_speed_mms: p.max_speed_mms,
        min_accel_mms2: p.min_accel_mms2,
        max_accel_mms2: p.max_accel_mms2,
        min_flow_mm3s: p.min_flow_mm3s,
        max_flow_mm3s: p.max_flow_mm3s,
        min_flow_mult: p.min_flow_mult,
        max_flow_mult: p.max_flow_mult,
        min_pa_s: p.min_pa_s,
        max_pa_s: p.max_pa_s,
        min_pressure_mm3s: p.min_pressure_mm3s,
        max_pressure_mm3s: p.max_pressure_mm3s,
        min_width_mm: p.min_width_mm,
        max_width_mm: p.max_width_mm,
        min_height_mm: p.min_height_mm,
        max_height_mm: p.max_height_mm,
        min_tool_iso_angle_deg: p.min_tool_iso_angle_deg,
        max_tool_iso_angle_deg: p.max_tool_iso_angle_deg,
        min_sdf_mm: p.min_sdf_mm,
        max_sdf_mm: p.max_sdf_mm,
    }
}

impl ToolpathMeshFilter {
    #[inline]
    pub fn is_planned_segment_visible(&self, seg: &crate::gcode::PlannedSegment) -> bool {
        match seg.kind {
            crate::gcode::PlannedMoveKind::OuterWall => self.show_outer_walls,
            crate::gcode::PlannedMoveKind::InnerWall => self.show_inner_walls,
            crate::gcode::PlannedMoveKind::Bridge => self.show_bridges,
            crate::gcode::PlannedMoveKind::WaveOverhang => self.show_wave_overhangs,
            crate::gcode::PlannedMoveKind::SolidInfill => self.show_solid_infill,
            crate::gcode::PlannedMoveKind::SparseInfill => self.show_sparse_infill,
            crate::gcode::PlannedMoveKind::GapFill => self.show_gap_fill,
            crate::gcode::PlannedMoveKind::Support => self.show_supports,
            crate::gcode::PlannedMoveKind::SupportInterface => self.show_support_interface,
            crate::gcode::PlannedMoveKind::PurgeTower => self.show_purge_tower,
            crate::gcode::PlannedMoveKind::ScarfSeam => self.show_scarf_seams,
            crate::gcode::PlannedMoveKind::Wipe => self.show_wipes,
            crate::gcode::PlannedMoveKind::Travel => self.show_travel_moves,
            crate::gcode::PlannedMoveKind::ZHop => self.show_z_hops,
        }
    }

    #[inline]
    pub fn is_raw_path_visible(&self, feature_kind: u8) -> bool {
        match feature_kind {
            0 => self.show_outer_walls,
            1 => self.show_inner_walls,
            2 => self.show_solid_infill,
            3 => self.show_gap_fill,
            4 => self.show_sparse_infill,
            5 => self.show_bridges,
            6 => self.show_wave_overhangs,
            7 => self.show_supports,
            8 => self.show_support_interface,
            _ => self.show_solid_infill,
        }
    }

    pub fn build_geometry(
        &self,
        planned_opt: Option<&crate::gcode::PlannedGcodeResult>,
        raw_opt: Option<&ObjectToolpathsResult>,
        config: &SlicerConfig,
    ) -> BuiltToolpathGeometry {
        let stride = self.selected_layer_stride.max(1);
        let min_idx = self.selected_layer_min_idx;
        let max_idx = self.selected_layer_max_idx;
        let highlighted = self.highlighted_selected_layer;
        let color_mode = self.toolpath_color_mode;

        if self.toolpath_quality_mode == ToolpathQualityMode::Fast {
            let mut positions: Vec<[f32; 3]> = Vec::new();
            let mut normals: Vec<[f32; 3]> = Vec::new();
            let mut colors: Vec<[f32; 4]> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            if let Some(planned) = planned_opt {
                build_planned_toolpaths_bead_mesh_parallel(
                    planned,
                    self,
                    config,
                    min_idx,
                    max_idx,
                    stride,
                    highlighted,
                    color_mode,
                    &mut positions,
                    &mut normals,
                    &mut colors,
                    &mut indices,
                );
            } else if let Some(tp_res) = raw_opt {
                build_raw_toolpaths_bead_mesh_parallel(
                    tp_res,
                    self,
                    config,
                    min_idx,
                    max_idx,
                    stride,
                    highlighted,
                    color_mode,
                    &mut positions,
                    &mut normals,
                    &mut colors,
                    &mut indices,
                );
            }
            BuiltToolpathGeometry::Fast {
                positions,
                normals,
                colors,
                indices,
            }
        } else {
            let buckets = if let Some(planned) = planned_opt {
                build_hq_planned_toolpaths_buckets(
                    planned,
                    self,
                    config,
                    min_idx,
                    max_idx,
                    stride,
                    highlighted,
                    color_mode,
                )
            } else if let Some(tp_res) = raw_opt {
                build_hq_raw_toolpaths_buckets(
                    tp_res,
                    self,
                    config,
                    min_idx,
                    max_idx,
                    stride,
                    highlighted,
                    color_mode,
                )
            } else {
                Vec::new()
            };
            BuiltToolpathGeometry::HighQualityPbr(buckets)
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn apply_built_toolpath_geometry_to_scene(
    commands: &mut Commands,
    vis_state: &mut FieldVisualizationState,
    config: &SlicerConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    toolpaths_query: &mut Query<
        (&mut Transform, &mut Visibility),
        (
            With<ToolpathsMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
    built_geom: BuiltToolpathGeometry,
    target_tf: Transform,
) {
    match built_geom {
        BuiltToolpathGeometry::Fast {
            positions,
            normals,
            colors,
            indices,
        } => {
            for part in &vis_state.hq_toolpath_parts {
                if let Ok((_, mut vis)) = toolpaths_query.get_mut(part.entity) {
                    *vis = Visibility::Hidden;
                }
            }

            if !positions.is_empty() && !indices.is_empty() {
                let bevy_mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
                .with_inserted_indices(Indices::U32(indices));

                let fast_mat = build_fast_toolpath_material(vis_state.toolpath_opacity);
                if let (Some(ent), Some(ref mesh_h), Some(ref mat_h)) = (
                    vis_state.toolpaths_entity,
                    &vis_state.toolpaths_mesh_handle,
                    &vis_state.toolpaths_mat_handle,
                ) {
                    meshes.insert(mesh_h, bevy_mesh);
                    if let Some(mat) = materials.get_mut(mat_h) {
                        *mat = fast_mat;
                    }
                    if let Ok((mut tf, mut vis)) = toolpaths_query.get_mut(ent) {
                        *tf = target_tf;
                        *vis = Visibility::Inherited;
                    }
                } else {
                    let mesh_h = meshes.add(bevy_mesh);
                    let mat_h = materials.add(fast_mat);
                    let ent = commands
                        .spawn((
                            PbrBundle {
                                mesh: mesh_h.clone(),
                                material: mat_h.clone(),
                                transform: target_tf,
                                ..default()
                            },
                            ToolpathsMeshEntity,
                        ))
                        .id();
                    vis_state.toolpaths_entity = Some(ent);
                    vis_state.toolpaths_mesh_handle = Some(mesh_h);
                    vis_state.toolpaths_mat_handle = Some(mat_h);
                }
            } else if let Some(ent) = vis_state.toolpaths_entity {
                if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
                    *vis = Visibility::Hidden;
                }
            }
        }
        BuiltToolpathGeometry::HighQualityPbr(buckets) => {
            if let Some(ent) = vis_state.toolpaths_entity {
                if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
                    *vis = Visibility::Hidden;
                }
            }

            let used_count = buckets.len();
            for (slot_i, bucket) in buckets.into_iter().enumerate() {
                let bevy_mesh = Mesh::new(
                    PrimitiveTopology::TriangleList,
                    RenderAssetUsages::default(),
                )
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, bucket.positions)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, bucket.normals)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, bucket.colors)
                .with_inserted_indices(Indices::U32(bucket.indices));

                let pbr_mat = build_hq_bucket_pbr_material(
                    bucket.tier,
                    bucket.mat_slot,
                    bucket.mean_thickness_mm,
                    vis_state,
                    config,
                );

                if slot_i < vis_state.hq_toolpath_parts.len() {
                    let part = &mut vis_state.hq_toolpath_parts[slot_i];
                    part.tier = bucket.tier;
                    part.mat_slot = bucket.mat_slot;
                    part.mean_thickness_mm = bucket.mean_thickness_mm;
                    meshes.insert(&part.mesh_handle, bevy_mesh);
                    if let Some(mat) = materials.get_mut(&part.mat_handle) {
                        *mat = pbr_mat;
                    }
                    if let Ok((mut tf, mut vis)) = toolpaths_query.get_mut(part.entity) {
                        *tf = target_tf;
                        *vis = Visibility::Inherited;
                    }
                } else {
                    let mesh_h = meshes.add(bevy_mesh);
                    let mat_h = materials.add(pbr_mat);
                    let ent = commands
                        .spawn((
                            PbrBundle {
                                mesh: mesh_h.clone(),
                                material: mat_h.clone(),
                                transform: target_tf,
                                ..default()
                            },
                            bevy::pbr::TransmittedShadowReceiver,
                            ToolpathsMeshEntity,
                        ))
                        .id();
                    vis_state.hq_toolpath_parts.push(HqToolpathSubMesh {
                        entity: ent,
                        mesh_handle: mesh_h,
                        mat_handle: mat_h,
                        tier: bucket.tier,
                        mat_slot: bucket.mat_slot,
                        mean_thickness_mm: bucket.mean_thickness_mm,
                    });
                }
            }

            for unused in vis_state.hq_toolpath_parts.iter().skip(used_count) {
                if let Ok((_, mut vis)) = toolpaths_query.get_mut(unused.entity) {
                    *vis = Visibility::Hidden;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn rebuild_and_sync_toolpath_meshes(
    commands: &mut Commands,
    vis_state: &mut FieldVisualizationState,
    config: &SlicerConfig,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    toolpaths_query: &mut Query<
        (&mut Transform, &mut Visibility),
        (
            With<ToolpathsMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
    target_tf: Transform,
) {
    let filter = vis_state.toolpath_mesh_filter();
    let built_geom = filter.build_geometry(
        vis_state.cached_planned_gcode.as_ref(),
        vis_state.cached_toolpaths.as_ref(),
        config,
    );
    apply_built_toolpath_geometry_to_scene(
        commands,
        vis_state,
        config,
        meshes,
        materials,
        toolpaths_query,
        built_geom,
        target_tf,
    );
}

/// Builds 3D oriented bead prism geometry across visible layers of a `PlannedGcodeResult`
/// in parallel across CPU worker threads using `std::thread::scope`.
#[allow(clippy::too_many_arguments)]
fn build_planned_toolpaths_bead_mesh_parallel(
    planned: &crate::gcode::PlannedGcodeResult,
    filter: &ToolpathMeshFilter,
    config: &SlicerConfig,
    min_idx: usize,
    max_idx: usize,
    stride: usize,
    highlighted: Option<usize>,
    color_mode: ToolpathColorMode,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let total = planned.layers.len();
    if total == 0 {
        return;
    }
    let lo = min_idx.min(total - 1);
    let hi = max_idx.min(total - 1).max(lo);
    let active_scrub_layer = highlighted.unwrap_or(hi);

    let mut visible_layers: Vec<(usize, f32, usize)> = Vec::new();
    let mut total_est_segs = 0usize;
    for (idx, layer_p) in planned.layers.iter().enumerate() {
        let is_hovered = highlighted == Some(idx);
        let in_range = idx >= lo && idx <= hi;
        let matches_stride =
            (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
        if !is_hovered && (!in_range || !matches_stride) {
            continue;
        }
        let dim_factor = if is_hovered {
            1.15
        } else if highlighted.is_some() {
            0.55
        } else {
            1.0
        };
        let mut layer_segs = 0usize;
        for seg in &layer_p.segments {
            if filter.is_planned_segment_visible(seg) {
                layer_segs += 1;
            }
        }
        if idx == active_scrub_layer {
            if let Some(limit) = filter.layer_move_scrub_limit {
                layer_segs = layer_segs.min(limit);
            }
        }
        if layer_segs > 0 {
            visible_layers.push((idx, dim_factor, layer_segs));
            total_est_segs += layer_segs;
        }
    }

    if visible_layers.is_empty() {
        return;
    }

    positions.reserve(total_est_segs * 8);
    normals.reserve(total_est_segs * 8);
    colors.reserve(total_est_segs * 8);
    indices.reserve(total_est_segs * 36);

    let optical_composite = if color_mode == ToolpathColorMode::ByOpticalComposite {
        crate::color_optics::compute_planned_optical_composite_colors(planned, config)
    } else {
        Vec::new()
    };
    let opt_ref = optical_composite.as_slice();

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(visible_layers.len());

    if num_threads <= 1 || total_est_segs < 256 {
        for &(idx, dim_factor, max_layer_segs) in &visible_layers {
            let mut emitted = 0usize;
            for (seg_i, seg) in planned.layers[idx].segments.iter().enumerate() {
                if !filter.is_planned_segment_visible(seg) {
                    continue;
                }
                if emitted >= max_layer_segs {
                    break;
                }
                emitted += 1;
                let override_rgba = opt_ref
                    .get(idx)
                    .and_then(|layer_cols| layer_cols.get(seg_i))
                    .copied();
                append_planned_segment_bead_mesh(
                    seg,
                    planned,
                    color_mode,
                    override_rgba,
                    config,
                    dim_factor,
                    positions,
                    normals,
                    colors,
                    indices,
                );
            }
        }
        return;
    }

    let chunk_size = visible_layers.len().div_ceil(num_threads);
    let chunk_results: Vec<(
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 4]>,
        Vec<u32>,
    )> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for chunk in visible_layers.chunks(chunk_size) {
            handles.push(s.spawn(move || {
                let est: usize = chunk.iter().map(|c| c.2).sum();
                let mut c_pos = Vec::with_capacity(est * 8);
                let mut c_nrm = Vec::with_capacity(est * 8);
                let mut c_col = Vec::with_capacity(est * 8);
                let mut c_ind = Vec::with_capacity(est * 36);
                for &(idx, dim_factor, max_layer_segs) in chunk {
                    let mut emitted = 0usize;
                    for (seg_i, seg) in planned.layers[idx].segments.iter().enumerate() {
                        if !filter.is_planned_segment_visible(seg) {
                            continue;
                        }
                        if emitted >= max_layer_segs {
                            break;
                        }
                        emitted += 1;
                        let override_rgba = opt_ref
                            .get(idx)
                            .and_then(|layer_cols| layer_cols.get(seg_i))
                            .copied();
                        append_planned_segment_bead_mesh(
                            seg,
                            planned,
                            color_mode,
                            override_rgba,
                            config,
                            dim_factor,
                            &mut c_pos,
                            &mut c_nrm,
                            &mut c_col,
                            &mut c_ind,
                        );
                    }
                }
                (c_pos, c_nrm, c_col, c_ind)
            }));
        }
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .collect()
    });

    for (c_pos, c_nrm, c_col, c_ind) in chunk_results {
        if c_pos.is_empty() {
            continue;
        }
        let base_v = positions.len() as u32;
        positions.extend(c_pos);
        normals.extend(c_nrm);
        colors.extend(c_col);
        indices.extend(c_ind.into_iter().map(|i| i + base_v));
    }
}

/// Builds 3D oriented bead tube geometry across visible layers of an `ObjectToolpathsResult`
/// in parallel across CPU worker threads using `std::thread::scope`.
#[allow(clippy::too_many_arguments)]
fn build_raw_toolpaths_bead_mesh_parallel(
    tp_res: &ObjectToolpathsResult,
    filter: &ToolpathMeshFilter,
    config: &SlicerConfig,
    min_idx: usize,
    max_idx: usize,
    stride: usize,
    highlighted: Option<usize>,
    color_mode: ToolpathColorMode,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let total = tp_res.layers.len();
    if total == 0 {
        return;
    }
    let lo = min_idx.min(total - 1);
    let hi = max_idx.min(total - 1).max(lo);
    let active_scrub_layer = highlighted.unwrap_or(hi);

    let mut visible_layers: Vec<(usize, f32, usize, Option<usize>)> = Vec::new();
    let mut total_est_pts = 0usize;
    for (idx, layer_tp) in tp_res.layers.iter().enumerate() {
        let is_hovered = highlighted == Some(idx);
        let in_range = idx >= lo && idx <= hi;
        let matches_stride =
            (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
        if !is_hovered && (!in_range || !matches_stride) {
            continue;
        }
        let dim_factor = if is_hovered {
            1.15
        } else if highlighted.is_some() {
            0.55
        } else {
            1.0
        };
        let mut layer_pts = 0usize;
        for path in &layer_tp.paths {
            if filter.is_raw_path_visible(path.feature_kind) && path.points.len() >= 2 {
                layer_pts += path.points.len();
            }
        }
        let seg_limit = if idx == active_scrub_layer {
            filter.layer_move_scrub_limit
        } else {
            None
        };
        if layer_pts > 0 && seg_limit != Some(0) {
            visible_layers.push((idx, dim_factor, layer_pts, seg_limit));
            total_est_pts += layer_pts;
        }
    }

    if visible_layers.is_empty() {
        return;
    }

    positions.reserve(total_est_pts * 4);
    normals.reserve(total_est_pts * 4);
    colors.reserve(total_est_pts * 4);
    indices.reserve(total_est_pts * 24);

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(visible_layers.len());

    if num_threads <= 1 || total_est_pts < 256 {
        for &(idx, dim_factor, _, seg_limit_opt) in &visible_layers {
            let mut rem_segs = seg_limit_opt;
            for path in &tp_res.layers[idx].paths {
                if !filter.is_raw_path_visible(path.feature_kind) || path.points.len() < 2 {
                    continue;
                }
                if rem_segs == Some(0) {
                    break;
                }
                let emitted = append_extrusion_path_bead_mesh(
                    path,
                    color_mode,
                    config,
                    dim_factor,
                    rem_segs,
                    positions,
                    normals,
                    colors,
                    indices,
                );
                if let Some(ref mut rem) = rem_segs {
                    *rem = rem.saturating_sub(emitted);
                }
            }
        }
        return;
    }

    let chunk_size = visible_layers.len().div_ceil(num_threads);
    let chunk_results: Vec<(
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 4]>,
        Vec<u32>,
    )> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for chunk in visible_layers.chunks(chunk_size) {
            handles.push(s.spawn(move || {
                let est: usize = chunk.iter().map(|c| c.2).sum();
                let mut c_pos = Vec::with_capacity(est * 4);
                let mut c_nrm = Vec::with_capacity(est * 4);
                let mut c_col = Vec::with_capacity(est * 4);
                let mut c_ind = Vec::with_capacity(est * 24);
                for &(idx, dim_factor, _, seg_limit_opt) in chunk {
                    let mut rem_segs = seg_limit_opt;
                    for path in &tp_res.layers[idx].paths {
                        if !filter.is_raw_path_visible(path.feature_kind)
                            || path.points.len() < 2
                        {
                            continue;
                        }
                        if rem_segs == Some(0) {
                            break;
                        }
                        let emitted = append_extrusion_path_bead_mesh(
                            path,
                            color_mode,
                            config,
                            dim_factor,
                            rem_segs,
                            &mut c_pos,
                            &mut c_nrm,
                            &mut c_col,
                            &mut c_ind,
                        );
                        if let Some(ref mut rem) = rem_segs {
                            *rem = rem.saturating_sub(emitted);
                        }
                    }
                }
                (c_pos, c_nrm, c_col, c_ind)
            }));
        }
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .collect()
    });

    for (c_pos, c_nrm, c_col, c_ind) in chunk_results {
        if c_pos.is_empty() {
            continue;
        }
        let base_v = positions.len() as u32;
        positions.extend(c_pos);
        normals.extend(c_nrm);
        colors.extend(c_col);
        indices.extend(c_ind.into_iter().map(|i| i + base_v));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn push_fast_bead_ring(
    p: Vec3,
    frame: NozzleFrame,
    half_w: f32,
    half_h: f32,
    linear_rgba: [f32; 4],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
) {
    let v_top = p + frame.nozzle_axis * half_h;
    let v_right = p + frame.lateral_axis * half_w;
    let v_bottom = p - frame.nozzle_axis * half_h;
    let v_left = p - frame.lateral_axis * half_w;

    positions.push(v_top.to_array());
    normals.push(frame.nozzle_axis.to_array());
    colors.push(linear_rgba);

    positions.push(v_right.to_array());
    normals.push(frame.lateral_axis.to_array());
    colors.push(linear_rgba);

    positions.push(v_bottom.to_array());
    normals.push((-frame.nozzle_axis).to_array());
    colors.push(linear_rgba);

    positions.push(v_left.to_array());
    normals.push((-frame.lateral_axis).to_array());
    colors.push(linear_rgba);
}

/// Appends a single planned segment (`PlannedSegment`) as a 3D oriented bead prism
/// in its local `NozzleFrame` $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$.
#[allow(clippy::too_many_arguments)]
fn append_planned_segment_bead_mesh(
    seg: &crate::gcode::PlannedSegment,
    planned: &crate::gcode::PlannedGcodeResult,
    color_mode: ToolpathColorMode,
    override_rgba: Option<[f32; 4]>,
    config: &SlicerConfig,
    brightness_scale: f32,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let is_travel = matches!(
        seg.kind,
        crate::gcode::PlannedMoveKind::Travel | crate::gcode::PlannedMoveKind::ZHop
    );
    // Travel/Z-hop moves are drawn with gizmo lines
    if is_travel {
        return;
    }

    let delta = seg.p1_local - seg.p0_local;
    let len_sq = delta.length_squared();
    if len_sq < 1.0e-9 {
        return;
    }
    let raw_travel = delta / len_sq.sqrt();

    let n0 = seg
        .nozzle_local_0
        .try_normalize()
        .unwrap_or_else(|| Vec3::from_array(config.machine.nozzle_axis_bed));
    let n1 = seg.nozzle_local_1.try_normalize().unwrap_or(n0);

    let frame0 = NozzleFrame::from_nozzle_and_travel(n0, raw_travel);
    let frame1 = NozzleFrame::from_nozzle_and_travel(n1, raw_travel);

    let w_eff = if matches!(seg.kind, crate::gcode::PlannedMoveKind::Wipe) {
        (0.55 * seg.width_mm).max(0.10)
    } else {
        seg.width_mm
    };
    let h_eff = if matches!(seg.kind, crate::gcode::PlannedMoveKind::Wipe) {
        (0.55 * seg.height_mm).max(0.06)
    } else {
        seg.height_mm
    };

    let (half_w, half_h) = if seg.kind == crate::gcode::PlannedMoveKind::Bridge {
        let r = (0.48 * w_eff).max(0.03);
        (r, r)
    } else {
        ((0.48 * w_eff).max(0.03), (0.42 * h_eff).max(0.02))
    };

    let (c0, cm_opt, c1) =
        if color_mode == ToolpathColorMode::ByNozzleSdf && override_rgba.is_none() {
            nozzle_sdf_segment_linear_colors(
                seg.sdf_mm_0,
                seg.sdf_mm,
                seg.sdf_mm_1,
                nozzle_sdf_reference_layer_height_mm(config),
                brightness_scale,
            )
        } else {
            let rgba = override_rgba
                .unwrap_or_else(|| compute_planned_segment_color(seg, planned, color_mode, config));
            let lin = scale_srgba_to_linear(rgba, brightness_scale);
            (lin, None, lin)
        };

    let base_ring_start = positions.len() as u32;
    let mut ring_count = 0u32;

    push_fast_bead_ring(
        seg.p0_local,
        frame0,
        half_w,
        half_h,
        c0,
        positions,
        normals,
        colors,
    );
    ring_count += 1;

    if let Some(cm) = cm_opt {
        let pm = 0.5 * (seg.p0_local + seg.p1_local);
        let nm = (n0 + n1).try_normalize().unwrap_or(n0);
        let frame_m = NozzleFrame::from_nozzle_and_travel(nm, raw_travel);
        push_fast_bead_ring(pm, frame_m, half_w, half_h, cm, positions, normals, colors);
        ring_count += 1;
    }

    push_fast_bead_ring(
        seg.p1_local,
        frame1,
        half_w,
        half_h,
        c1,
        positions,
        normals,
        colors,
    );
    ring_count += 1;

    let r_first = base_ring_start;
    indices.extend_from_slice(&[r_first, r_first + 2, r_first + 1, r_first, r_first + 3, r_first + 2]);
    for ri in 0..(ring_count - 1) {
        let r0 = base_ring_start + ri * 4;
        let r1 = r0 + 4;
        for k in 0..4u32 {
            let k_next = (k + 1) % 4;
            let a0 = r0 + k;
            let a1 = r0 + k_next;
            let b0 = r1 + k;
            let b1 = r1 + k_next;
            indices.extend_from_slice(&[a0, b0, b1, a0, b1, a1]);
        }
    }
    let r_last = base_ring_start + (ring_count - 1) * 4;
    indices.extend_from_slice(&[r_last, r_last + 1, r_last + 2, r_last, r_last + 2, r_last + 3]);
}

/// Converts a single sRGB channel ($[0, 1]$) into Linear RGB ($[0, 1]$) for `Mesh::ATTRIBUTE_COLOR`.
#[inline]
fn srgb_channel_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts an sRGBA color (`[f32; 4]`, matching `egui` legend swatches) into Linear RGBA
/// required by Bevy's 3D `Mesh::ATTRIBUTE_COLOR` PBR pipeline so 3D vertex colors are not
/// double-gamma-encoded (which otherwise lifts secondary channels and washes out saturation).
#[inline]
pub fn srgba_to_linear_rgba(srgba: [f32; 4]) -> [f32; 4] {
    [
        srgb_channel_to_linear(srgba[0]),
        srgb_channel_to_linear(srgba[1]),
        srgb_channel_to_linear(srgba[2]),
        srgba[3],
    ]
}

/// High-contrast 6-stop perceptual color ramp for toolpath and layer scalar parameters:
/// `0.00` Deep Indigo -> `0.20` Electric Cyan -> `0.40` Emerald Green ->
/// `0.60` Solar Yellow -> `0.80` Blazing Orange-Red -> `1.00` Hot Crimson-Magenta.
pub fn high_contrast_parameter_ramp(t: f32) -> [f32; 4] {
    const STOPS: [(f32, [f32; 3]); 6] = [
        (0.00, [0.16, 0.14, 0.88]), // Deep Indigo
        (0.20, [0.02, 0.80, 0.98]), // Electric Cyan
        (0.40, [0.10, 0.92, 0.34]), // Vivid Emerald Green
        (0.60, [0.98, 0.90, 0.08]), // Bright Solar Yellow
        (0.80, [0.98, 0.34, 0.06]), // Blazing Orange-Red
        (1.00, [0.92, 0.06, 0.54]), // Hot Crimson-Magenta
    ];
    let t = t.clamp(0.0, 1.0);
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let u = ((t - t0) / (t1 - t0).max(1.0e-5)).clamp(0.0, 1.0);
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
                1.0,
            ];
        }
    }
    let last = STOPS[STOPS.len() - 1].1;
    [last[0], last[1], last[2], 1.0]
}

/// Distinct 8-color categorical palette for branch-aware slicing visualization.
pub const BRANCH_PALETTE: [[f32; 3]; 8] = [
    [0.20, 0.86, 0.62],
    [0.96, 0.62, 0.20],
    [0.36, 0.68, 0.98],
    [0.94, 0.38, 0.76],
    [0.72, 0.90, 0.24],
    [0.98, 0.42, 0.38],
    [0.58, 0.48, 0.96],
    [0.24, 0.90, 0.92],
];

/// Returns per-vertex Linear RGBA colors for a `SceneObject` in `DefaultCad` surface color mode,
/// honoring per-object `SingleTool`, `FilamentSlot`, or `custom_color_srgba` overrides when present.
pub fn default_cad_vertex_colors_for_object(
    obj: &crate::scene::SceneObject,
    config: &SlicerConfig,
) -> Vec<[f32; 4]> {
    match obj.filament_mode {
        ObjectFilamentMode::SingleTool(t) => {
            if let Some(tool) = config.tools.get(t).or_else(|| config.tools.first()) {
                let lin = if obj.vertex_filament_slots.is_empty() && obj.vertex_colors_rgba.is_empty()
                {
                    // Base material already applies `resolved_cad_tint_srgba`, so keep vertex attribute white
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    srgba_to_linear_rgba(tool.filament_color_srgba)
                };
                return vec![lin; obj.vertices.len()];
            }
        }
        ObjectFilamentMode::FilamentSlot(s) => {
            let lin = if obj.vertex_filament_slots.is_empty() && obj.vertex_colors_rgba.is_empty() {
                [1.0, 1.0, 1.0, 1.0]
            } else {
                srgba_to_linear_rgba(config.color_srgba_for_filament_slot(s))
            };
            return vec![lin; obj.vertices.len()];
        }
        _ => {
            if let Some(c) = obj.custom_color_srgba {
                if obj.vertex_colors_rgba.is_empty() {
                    let lin = if obj.vertex_filament_slots.is_empty() {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        srgba_to_linear_rgba(c)
                    };
                    return vec![lin; obj.vertices.len()];
                }
            }
        }
    }
    default_cad_vertex_colors(
        obj.vertices.len(),
        &obj.vertex_filament_slots,
        &obj.vertex_colors_rgba,
        config,
    )
}

/// Returns per-vertex Linear RGBA colors for a CAD object in `DefaultCad` surface color mode:
/// uses discrete 3MF/OBJ filament slot colours when multiple slots (or explicit slot mappings) are
/// present without continuous texture/vertex colours, else uses true-colour/textured `vertex_colors_rgba`,
/// or uniform white (`[1.0, 1.0, 1.0, 1.0]`).
pub fn default_cad_vertex_colors(
    vertex_count: usize,
    vertex_filament_slots: &[u8],
    vertex_colors_rgba: &[[u8; 4]],
    config: &SlicerConfig,
) -> Vec<[f32; 4]> {
    let has_slots = vertex_filament_slots.len() == vertex_count && !vertex_filament_slots.is_empty();
    let has_true_colors = vertex_colors_rgba.len() == vertex_count && !vertex_colors_rgba.is_empty();

    if has_true_colors && !has_slots {
        vertex_colors_rgba
            .iter()
            .map(|&[r, g, b, a]| {
                srgba_to_linear_rgba([
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                    a as f32 / 255.0,
                ])
            })
            .collect()
    } else if has_slots {
        // Precompute all 256 possible filament slot Linear RGBA colours once so per-vertex
        // lookup across hundreds of thousands of vertices is a single array index operation.
        let mut slot_lut = [[1.0f32; 4]; 256];
        for s in 0..=255u8 {
            slot_lut[s as usize] = srgba_to_linear_rgba(config.color_srgba_for_filament_slot(s));
        }
        vertex_filament_slots
            .iter()
            .map(|&slot| slot_lut[slot as usize])
            .collect()
    } else if has_true_colors {
        vertex_colors_rgba
            .iter()
            .map(|&[r, g, b, a]| {
                srgba_to_linear_rgba([
                    r as f32 / 255.0,
                    g as f32 / 255.0,
                    b as f32 / 255.0,
                    a as f32 / 255.0,
                ])
            })
            .collect()
    } else {
        vec![[1.0f32, 1.0, 1.0, 1.0]; vertex_count]
    }
}

/// Computes per-vertex linear RGBA colors (`Mesh::ATTRIBUTE_COLOR`) and summary angle statistics
/// `(min_deg, max_deg, mean_deg, uses_solved_tool)` for an object's surface mesh under
/// `ObjectSurfaceColorMode`.
///
/// Obeys `AGENTS.md` kinematic invariants:
/// - Overhang is always measured relative to the queried gravity vector in bed coordinates
///   (`config.query_gravity_direction(p_bed)`), never hardcoding `(0, 0, -1)`.
/// - Tool angle from normal is always measured between the outward surface normal in bed
///   coordinates and the local nozzle/tool axis (`solved_local_tool_axes` rotated into bed
///   coordinates when available, or `config.query_nozzle_frame(p_bed, Vec3::X).nozzle_axis`),
///   never hardcoding `(0, 0, 1)`.
pub fn compute_object_surface_vertex_colors(
    vertices: &[ffi::RenderVertex],
    obj_transform: &Transform,
    config: &SlicerConfig,
    mode: ObjectSurfaceColorMode,
    solved_local_tool_axes: &[f32],
) -> (Vec<[f32; 4]>, f32, f32, f32, bool) {
    if vertices.is_empty() || mode == ObjectSurfaceColorMode::DefaultCad {
        return (
            vec![[1.0f32, 1.0, 1.0, 1.0]; vertices.len()],
            0.0,
            180.0,
            0.0,
            false,
        );
    }

    let has_solved_axes = solved_local_tool_axes.len() == vertices.len() * 3;
    let has_support_vals =
        mode == ObjectSurfaceColorMode::SupportNeeded && solved_local_tool_axes.len() == vertices.len();
    let mut colors = Vec::with_capacity(vertices.len());
    let mut min_deg = f32::INFINITY;
    let mut max_deg = f32::NEG_INFINITY;
    let mut sum_deg = 0.0_f64;

    for (i, v) in vertices.iter().enumerate() {
        let p_local = Vec3::new(v.x, v.y, v.z);
        let p_bed = obj_transform.transform_point(p_local);
        let n_local = Vec3::new(v.nx, v.ny, v.nz);
        let n_bed = (obj_transform.rotation * n_local)
            .try_normalize()
            .unwrap_or(Vec3::Z);

        let (angle_deg, t) = match mode {
            ObjectSurfaceColorMode::DefaultCad => (0.0, 0.0),
            ObjectSurfaceColorMode::Overhang => {
                let g_bed = config.query_gravity_bed(p_bed);
                // -n_bed . g_bed == +1 when outward normal points opposite to gravity (upward top),
                // 0 on vertical walls, and -1 on downward-facing overhangs/ceilings.
                let cos_up = (-n_bed.dot(g_bed)).clamp(-1.0, 1.0);
                let deg = cos_up.acos().to_degrees();
                (deg, (deg / 180.0).clamp(0.0, 1.0))
            }
            ObjectSurfaceColorMode::ToolAngleFromNormal => {
                let z_nozzle_bed = if has_solved_axes {
                    let ax = solved_local_tool_axes[i * 3];
                    let ay = solved_local_tool_axes[i * 3 + 1];
                    let az = solved_local_tool_axes[i * 3 + 2];
                    (obj_transform.rotation * Vec3::new(ax, ay, az))
                        .try_normalize()
                        .unwrap_or_else(|| config.query_nozzle_frame(p_bed, Vec3::X).nozzle_axis)
                } else {
                    config.query_nozzle_frame(p_bed, Vec3::X).nozzle_axis
                };
                let cos_tool = n_bed.dot(z_nozzle_bed).clamp(-1.0, 1.0);
                let deg = cos_tool.acos().to_degrees();
                (deg, (deg / 180.0).clamp(0.0, 1.0))
            }
            ObjectSurfaceColorMode::SupportNeeded => {
                let s = if has_support_vals {
                    solved_local_tool_axes[i].clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (s * 100.0, s)
            }
        };

        min_deg = min_deg.min(angle_deg);
        max_deg = max_deg.max(angle_deg);
        sum_deg += angle_deg as f64;

        let srgba = high_contrast_parameter_ramp(t);
        colors.push(srgba_to_linear_rgba(srgba));
    }

    let mean_deg = (sum_deg / (vertices.len() as f64)) as f32;
    let uses_solved = (has_solved_axes && mode == ObjectSurfaceColorMode::ToolAngleFromNormal)
        || has_support_vals;
    (
        colors,
        if min_deg.is_finite() { min_deg } else { 0.0 },
        if max_deg.is_finite() { max_deg } else { 180.0 },
        mean_deg,
        uses_solved,
    )
}

/// Samples the 2D slice plane `FieldColormap` at normalized position `t` in `[0.0, 1.0]`
/// given `[min_v, max_v]` (used by both the 2D slice rasterizer and the UI legend bar).
pub fn sample_slice_colormap_rgba(
    colormap: FieldColormap,
    t: f32,
    min_v: f32,
    max_v: f32,
) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let val = min_v + (max_v - min_v) * t;
    let (r, g, b) = match colormap {
        FieldColormap::CoolWarmSigned => {
            let abs_max = min_v.abs().max(max_v.abs()).max(1e-4);
            let signed_t = (val / abs_max).clamp(-1.0, 1.0);
            if signed_t < 0.0 {
                let s = (-signed_t).clamp(0.0, 1.0);
                (1.0 - 0.88 * s, 1.0 - 0.45 * s, 1.0 - 0.05 * s * s)
            } else {
                let s = signed_t.clamp(0.0, 1.0);
                (1.0 - 0.08 * s * s, 1.0 - 0.72 * s, 1.0 - 0.88 * s)
            }
        }
        FieldColormap::ViridisSequential => viridis_approx(t),
        FieldColormap::ZebraIsoBands => {
            let wave = (t * 16.0 * std::f32::consts::PI).sin();
            let band = if wave >= 0.0 { 0.88 } else { 0.18 };
            let (vr, vg, vb) = viridis_approx(t);
            (vr * band, vg * band, vb * band)
        }
    };
    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0), 1.0]
}

/// Computes the RGBA color for a `PlannedSegment` according to `ToolpathColorMode`.
fn compute_planned_segment_color(
    seg: &crate::gcode::PlannedSegment,
    planned: &crate::gcode::PlannedGcodeResult,
    mode: ToolpathColorMode,
    config: &SlicerConfig,
) -> [f32; 4] {
    match seg.kind {
        crate::gcode::PlannedMoveKind::Travel => {
            if mode == ToolpathColorMode::BySpeed {
                let denom = (planned.max_speed_mms - planned.min_speed_mms).max(1.0e-3);
                let t = (seg.cruise_speed_mms - planned.min_speed_mms) / denom;
                return high_contrast_parameter_ramp(t);
            }
            return [0.28, 0.86, 0.98, 1.0]; // Cyan travel line
        }
        crate::gcode::PlannedMoveKind::ZHop => {
            return [0.98, 0.92, 0.22, 1.0]; // Bright yellow nozzle-axis hop
        }
        _ => {}
    }

    match mode {
        ToolpathColorMode::ByFeatureType => match seg.kind {
            crate::gcode::PlannedMoveKind::ScarfSeam => [0.76, 0.44, 0.98, 1.0], // Violet scarf seam
            crate::gcode::PlannedMoveKind::Wipe => [0.68, 0.76, 0.86, 1.0],      // Cool silver wipe
            crate::gcode::PlannedMoveKind::PurgeTower => [0.96, 0.44, 0.64, 1.0], // Rose/coral purge tower
            _ => compute_toolpath_point_color(
                seg.kind as u8,
                seg.wall_index,
                seg.filament_slot,
                seg.width_mm,
                seg.tool_iso_angle_deg,
                seg.uses_surface_normal,
                seg.sdf_mm,
                ToolpathColorMode::ByFeatureType,
                config,
            ),
        },
        ToolpathColorMode::ByToolOrFilament | ToolpathColorMode::ByOpticalComposite => {
            if seg.kind.is_support()
                && config.slicing.support_tool_index.is_none()
                && config.slicing.support_interface_tool_index.is_none()
            {
                [0.22, 0.88, 0.62, 1.0]
            } else {
                config.color_srgba_for_filament_slot(seg.filament_slot)
            }
        }
        ToolpathColorMode::ByLineWidth => {
            let denom = (planned.max_width_mm - planned.min_width_mm).max(1.0e-4);
            let t = (seg.width_mm - planned.min_width_mm) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByLayerHeight => {
            let denom = (planned.max_height_mm - planned.min_height_mm).max(1.0e-4);
            let t = (seg.height_mm - planned.min_height_mm) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::BySpeed => {
            let denom = (planned.max_speed_mms - planned.min_speed_mms).max(1.0e-3);
            let t = (seg.cruise_speed_mms - planned.min_speed_mms) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByAcceleration => {
            let denom = (planned.max_accel_mms2 - planned.min_accel_mms2).max(1.0);
            let t = (seg.accel_mms2 - planned.min_accel_mms2) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByVolumetricFlow => {
            let denom = (planned.max_flow_mm3s - planned.min_flow_mm3s).max(1.0e-3);
            let t = (seg.volumetric_flow_mm3s - planned.min_flow_mm3s) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByExtrusionMultiplier => {
            let denom = (planned.max_flow_mult - planned.min_flow_mult).max(1.0e-3);
            let t = (seg.extrusion_multiplier - planned.min_flow_mult) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByPressureAdvance => {
            let denom = (planned.max_pa_s - planned.min_pa_s).max(1.0e-5);
            let t = (seg.pressure_advance_s - planned.min_pa_s) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByMeltZonePressure => {
            let denom = (planned.max_pressure_mm3s - planned.min_pressure_mm3s).max(1.0e-4);
            let t = (seg.melt_pressure_mm3s - planned.min_pressure_mm3s) / denom;
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::ByToolIsoNormalAngle => {
            let lo = planned.min_tool_iso_angle_deg;
            let hi = planned.max_tool_iso_angle_deg.max(lo + 5.0);
            let t = ((seg.tool_iso_angle_deg - lo) / (hi - lo)).clamp(0.0, 1.0);
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::BySurfaceNormalUsage => {
            if seg.uses_surface_normal {
                [0.14, 0.92, 0.46, 1.0] // Vivid emerald green: Object Surface Normal
            } else if matches!(
                seg.kind,
                crate::gcode::PlannedMoveKind::OuterWall
                    | crate::gcode::PlannedMoveKind::ScarfSeam
                    | crate::gcode::PlannedMoveKind::WaveOverhang
            ) {
                [0.98, 0.62, 0.16, 1.0] // Warm amber-orange: Isosurface Normal (Outer Fallback)
            } else {
                [0.24, 0.52, 0.92, 1.0] // Cool sapphire blue: Isosurface Normal (Interior / Other)
            }
        }
        ToolpathColorMode::ByNozzleSdf => sample_nozzle_sdf_rgba(
            seg.sdf_mm,
            nozzle_sdf_reference_layer_height_mm(config),
        ),
    }
}

/// Expands a continuous `ExtrusionPath` into a 3D variable-width bead tube mesh oriented
/// in each point's local `NozzleFrame` $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$.
#[allow(clippy::too_many_arguments)]
fn append_extrusion_path_bead_mesh(
    path: &ExtrusionPath,
    color_mode: ToolpathColorMode,
    config: &SlicerConfig,
    brightness_scale: f32,
    max_segments: Option<usize>,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) -> usize {
    let n = path.points.len();
    if n < 2 || max_segments == Some(0) {
        return 0;
    }

    let full_seg_count = if path.is_closed && n >= 3 { n } else { n - 1 };
    let emit_seg_count = max_segments
        .map_or(full_seg_count, |lim| lim.min(full_seg_count));
    if emit_seg_count == 0 {
        return 0;
    }

    let base_ring_start = positions.len() as u32;

    for i in 0..n {
        let pt = &path.points[i];
        let p_curr = Vec3::new(pt.x, pt.y, pt.z);
        let p_prev = if i > 0 {
            let pv = &path.points[i - 1];
            Vec3::new(pv.x, pv.y, pv.z)
        } else if path.is_closed {
            let pv = &path.points[n - 1];
            Vec3::new(pv.x, pv.y, pv.z)
        } else {
            p_curr
        };
        let p_next = if i + 1 < n {
            let nv = &path.points[i + 1];
            Vec3::new(nv.x, nv.y, nv.z)
        } else if path.is_closed {
            let nv = &path.points[0];
            Vec3::new(nv.x, nv.y, nv.z)
        } else {
            p_curr
        };

        let raw_travel = (p_next - p_prev)
            .try_normalize()
            .unwrap_or(Vec3::X);
        let nozzle_axis = Vec3::new(pt.nx, pt.ny, pt.nz)
            .try_normalize()
            .unwrap_or_else(|| Vec3::from_array(config.machine.nozzle_axis_bed));
        let frame = NozzleFrame::from_nozzle_and_travel(nozzle_axis, raw_travel);

        let (half_w, half_h) = if path.feature_kind == 5 {
            let r = (0.48 * pt.width_mm).max(0.03);
            (r, r)
        } else {
            ((0.48 * pt.width_mm).max(0.03), (0.42 * pt.height_mm).max(0.02))
        };

        let rgba = compute_toolpath_point_color(
            path.feature_kind,
            path.wall_index,
            path.filament_slot,
            pt.width_mm,
            pt.tool_iso_angle_deg,
            pt.uses_surface_normal,
            pt.sdf_mm,
            color_mode,
            config,
        );
        let linear_rgba = scale_srgba_to_linear(rgba, brightness_scale);

        push_fast_bead_ring(
            p_curr,
            frame,
            half_w,
            half_h,
            linear_rgba,
            positions,
            normals,
            colors,
        );
    }

    for s in 0..emit_seg_count {
        let r0 = base_ring_start + (s as u32) * 4;
        let r1 = base_ring_start + (((s + 1) % n) as u32) * 4;
        for k in 0..4u32 {
            let k_next = (k + 1) % 4;
            let a0 = r0 + k;
            let a1 = r0 + k_next;
            let b0 = r1 + k;
            let b1 = r1 + k_next;
            indices.extend_from_slice(&[a0, b0, b1, a0, b1, a1]);
        }
    }

    // End caps for open or partially-scrubbed extrusion polylines
    if !path.is_closed || emit_seg_count < full_seg_count {
        let r_start = base_ring_start;
        indices.extend_from_slice(&[r_start, r_start + 2, r_start + 1, r_start, r_start + 3, r_start + 2]);
        let r_end = base_ring_start + ((emit_seg_count % n) as u32) * 4;
        indices.extend_from_slice(&[r_end, r_end + 1, r_end + 2, r_end, r_end + 2, r_end + 3]);
    }

    emit_seg_count
}

// ============================================================================
// High-Quality Pressed-Stadium (Capsule) Welded Bead Mesher & PBR Bucketing
// ============================================================================

const HQ_RING_VERTS: usize = 12;

/// Assembled triangle mesh for one `(tier, mat_slot)` PBR sub-mesh bucket.
pub struct HqBucketMesh {
    pub tier: u8,
    pub mat_slot: u16,
    pub mean_thickness_mm: f32,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Copy)]
struct HqChainNode {
    pos: Vec3,
    nozzle_axis: Vec3,
    width_mm: f32,
    height_mm: f32,
    linear_rgba: [f32; 4],
    is_circular: bool,
}

#[derive(Default)]
struct HqBucketAccum {
    thickness_sum: f64,
    thickness_count: usize,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

#[inline]
fn classify_hq_planned_bucket(
    seg: &crate::gcode::PlannedSegment,
    color_mode: ToolpathColorMode,
    config: &SlicerConfig,
) -> (u8, u16) {
    let tier = match seg.kind {
        crate::gcode::PlannedMoveKind::SparseInfill
        | crate::gcode::PlannedMoveKind::Support
        | crate::gcode::PlannedMoveKind::SupportInterface
        | crate::gcode::PlannedMoveKind::Wipe => 0u8,
        crate::gcode::PlannedMoveKind::InnerWall => {
            if seg.wall_index >= 2 {
                0u8
            } else {
                1u8
            }
        }
        _ => 2u8,
    };

    let mat_slot = if color_mode == ToolpathColorMode::ByToolOrFilament {
        if seg.kind.is_support() {
            256u16
        } else if seg.kind == crate::gcode::PlannedMoveKind::Wipe {
            257u16
        } else {
            config
                .tool_index_for_filament_slot(seg.filament_slot)
                .min(255) as u16
        }
    } else {
        258u16
    };
    (tier, mat_slot)
}

#[inline]
fn classify_hq_raw_bucket(
    path: &ExtrusionPath,
    color_mode: ToolpathColorMode,
    config: &SlicerConfig,
) -> (u8, u16) {
    let tier = match path.feature_kind {
        4 | 7 | 8 => 0u8,
        1 => {
            if path.wall_index >= 2 {
                0u8
            } else {
                1u8
            }
        }
        _ => 2u8,
    };

    let mat_slot = if color_mode == ToolpathColorMode::ByToolOrFilament {
        if path.feature_kind == 7 || path.feature_kind == 8 {
            256u16
        } else {
            config
                .tool_index_for_filament_slot(path.filament_slot)
                .min(255) as u16
        }
    } else {
        258u16
    };
    (tier, mat_slot)
}

/// Builds parallel per-`(tier, mat_slot)` 12-vertex pressed-stadium welded chain meshes
/// across visible layers of a `PlannedGcodeResult`.
#[allow(clippy::too_many_arguments)]
fn build_hq_planned_toolpaths_buckets(
    planned: &crate::gcode::PlannedGcodeResult,
    filter: &ToolpathMeshFilter,
    config: &SlicerConfig,
    min_idx: usize,
    max_idx: usize,
    stride: usize,
    highlighted: Option<usize>,
    color_mode: ToolpathColorMode,
) -> Vec<HqBucketMesh> {
    use std::collections::BTreeMap;

    let total = planned.layers.len();
    if total == 0 {
        return Vec::new();
    }
    let lo = min_idx.min(total - 1);
    let hi = max_idx.min(total - 1).max(lo);
    let active_scrub_layer = highlighted.unwrap_or(hi);

    let mut visible_layers: Vec<(usize, f32, usize)> = Vec::new();
    for (idx, layer_p) in planned.layers.iter().enumerate() {
        let is_hovered = highlighted == Some(idx);
        let in_range = idx >= lo && idx <= hi;
        let matches_stride =
            (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
        if !is_hovered && (!in_range || !matches_stride) {
            continue;
        }
        let dim_factor = if is_hovered {
            1.15
        } else if highlighted.is_some() {
            0.55
        } else {
            1.0
        };
        let mut layer_segs = 0usize;
        for seg in &layer_p.segments {
            if filter.is_planned_segment_visible(seg) {
                layer_segs += 1;
            }
        }
        if idx == active_scrub_layer {
            if let Some(limit) = filter.layer_move_scrub_limit {
                layer_segs = layer_segs.min(limit);
            }
        }
        if layer_segs > 0 {
            visible_layers.push((idx, dim_factor, layer_segs));
        }
    }

    if visible_layers.is_empty() {
        return Vec::new();
    }

    let optical_composite = if color_mode == ToolpathColorMode::ByOpticalComposite {
        crate::color_optics::compute_planned_optical_composite_colors(planned, config)
    } else {
        Vec::new()
    };
    let opt_ref = optical_composite.as_slice();
    let fallback_nozzle = Vec3::from_array(config.machine.nozzle_axis_bed);

    let build_chunk = |chunk: &[(usize, f32, usize)]| -> BTreeMap<(u8, u16), HqBucketAccum> {
        let mut local_map: BTreeMap<(u8, u16), HqBucketAccum> = BTreeMap::new();
        let mut chain: Vec<HqChainNode> = Vec::with_capacity(64);
        let mut current_key: Option<(u8, u16)> = None;

        for &(idx, dim_factor, max_layer_segs) in chunk {
            let mut emitted = 0usize;
            for (seg_i, seg) in planned.layers[idx].segments.iter().enumerate() {
                if !filter.is_planned_segment_visible(seg) {
                    continue;
                }
                if emitted >= max_layer_segs {
                    break;
                }
                emitted += 1;

                if matches!(
                    seg.kind,
                    crate::gcode::PlannedMoveKind::Travel | crate::gcode::PlannedMoveKind::ZHop
                ) {
                    if let Some(prev_k) = current_key.take() {
                        let acc = local_map.entry(prev_k).or_default();
                        append_hq_welded_chain_stadium_mesh(&chain, fallback_nozzle, acc);
                        chain.clear();
                    }
                    continue;
                }

                let len_sq = (seg.p1_local - seg.p0_local).length_squared();
                if len_sq < 1.0e-9 {
                    continue;
                }

                let key = classify_hq_planned_bucket(seg, color_mode, config);
                let w_eff = if seg.kind == crate::gcode::PlannedMoveKind::Wipe {
                    (0.55 * seg.width_mm).max(0.10)
                } else {
                    seg.width_mm.max(0.06)
                };
                let h_eff = if seg.kind == crate::gcode::PlannedMoveKind::Wipe {
                    (0.55 * seg.height_mm).max(0.06)
                } else {
                    seg.height_mm.max(0.04)
                };
                let is_circ = seg.kind == crate::gcode::PlannedMoveKind::Bridge;

                let (c0, cm_opt, c1) =
                    if color_mode == ToolpathColorMode::ByNozzleSdf && opt_ref.is_empty() {
                        nozzle_sdf_segment_linear_colors(
                            seg.sdf_mm_0,
                            seg.sdf_mm,
                            seg.sdf_mm_1,
                            nozzle_sdf_reference_layer_height_mm(config),
                            dim_factor,
                        )
                    } else {
                        let rgba = opt_ref
                            .get(idx)
                            .and_then(|lc| lc.get(seg_i))
                            .copied()
                            .unwrap_or_else(|| {
                                compute_planned_segment_color(seg, planned, color_mode, config)
                            });
                        let lin = scale_srgba_to_linear(rgba, dim_factor);
                        (lin, None, lin)
                    };

                let n0 = seg.nozzle_local_0.try_normalize().unwrap_or(fallback_nozzle);
                let n1 = seg.nozzle_local_1.try_normalize().unwrap_or(n0);

                let contiguous = current_key == Some(key)
                    && chain
                        .last()
                        .is_some_and(|last| last.pos.distance_squared(seg.p0_local) <= 1.0e-4);

                if !contiguous {
                    if let Some(prev_k) = current_key.take() {
                        let acc = local_map.entry(prev_k).or_default();
                        append_hq_welded_chain_stadium_mesh(&chain, fallback_nozzle, acc);
                        chain.clear();
                    }
                    current_key = Some(key);
                    chain.push(HqChainNode {
                        pos: seg.p0_local,
                        nozzle_axis: n0,
                        width_mm: w_eff,
                        height_mm: h_eff,
                        linear_rgba: c0,
                        is_circular: is_circ,
                    });
                }
                if let Some(cm) = cm_opt {
                    chain.push(HqChainNode {
                        pos: 0.5 * (seg.p0_local + seg.p1_local),
                        nozzle_axis: (n0 + n1).try_normalize().unwrap_or(n0),
                        width_mm: w_eff,
                        height_mm: h_eff,
                        linear_rgba: cm,
                        is_circular: is_circ,
                    });
                }
                chain.push(HqChainNode {
                    pos: seg.p1_local,
                    nozzle_axis: n1,
                    width_mm: w_eff,
                    height_mm: h_eff,
                    linear_rgba: c1,
                    is_circular: is_circ,
                });
            }

            if let Some(prev_k) = current_key.take() {
                let acc = local_map.entry(prev_k).or_default();
                append_hq_welded_chain_stadium_mesh(&chain, fallback_nozzle, acc);
                chain.clear();
            }
        }
        local_map
    };

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(visible_layers.len());

    let merged_map = if num_threads <= 1 || visible_layers.len() < 4 {
        build_chunk(&visible_layers)
    } else {
        let chunk_size = visible_layers.len().div_ceil(num_threads);
        let partial_maps: Vec<BTreeMap<(u8, u16), HqBucketAccum>> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(num_threads);
            for chunk in visible_layers.chunks(chunk_size) {
                handles.push(s.spawn(move || build_chunk(chunk)));
            }
            handles.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        let mut combined: BTreeMap<(u8, u16), HqBucketAccum> = BTreeMap::new();
        for part_map in partial_maps {
            for (k, src) in part_map {
                if src.positions.is_empty() {
                    continue;
                }
                let dst = combined.entry(k).or_default();
                let base_v = dst.positions.len() as u32;
                dst.thickness_sum += src.thickness_sum;
                dst.thickness_count += src.thickness_count;
                dst.positions.extend(src.positions);
                dst.normals.extend(src.normals);
                dst.colors.extend(src.colors);
                dst.indices.extend(src.indices.into_iter().map(|i| i + base_v));
            }
        }
        combined
    };

    finalize_hq_buckets(merged_map)
}

/// Builds per-`(tier, mat_slot)` 12-vertex pressed-stadium meshes for raw `ObjectToolpathsResult`.
#[allow(clippy::too_many_arguments)]
fn build_hq_raw_toolpaths_buckets(
    tp_res: &ObjectToolpathsResult,
    filter: &ToolpathMeshFilter,
    config: &SlicerConfig,
    min_idx: usize,
    max_idx: usize,
    stride: usize,
    highlighted: Option<usize>,
    color_mode: ToolpathColorMode,
) -> Vec<HqBucketMesh> {
    use std::collections::BTreeMap;

    let total = tp_res.layers.len();
    if total == 0 {
        return Vec::new();
    }
    let lo = min_idx.min(total - 1);
    let hi = max_idx.min(total - 1).max(lo);
    let active_scrub_layer = highlighted.unwrap_or(hi);
    let fallback_nozzle = Vec3::from_array(config.machine.nozzle_axis_bed);

    let mut buckets: BTreeMap<(u8, u16), HqBucketAccum> = BTreeMap::new();
    let mut chain: Vec<HqChainNode> = Vec::new();

    for (idx, layer_tp) in tp_res.layers.iter().enumerate() {
        let is_hovered = highlighted == Some(idx);
        let in_range = idx >= lo && idx <= hi;
        let matches_stride =
            (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
        if !is_hovered && (!in_range || !matches_stride) {
            continue;
        }
        let dim_factor = if is_hovered {
            1.15
        } else if highlighted.is_some() {
            0.55
        } else {
            1.0
        };
        let mut rem_segs = if idx == active_scrub_layer {
            filter.layer_move_scrub_limit
        } else {
            None
        };

        for path in &layer_tp.paths {
            if !filter.is_raw_path_visible(path.feature_kind) || path.points.len() < 2 {
                continue;
            }
            if rem_segs == Some(0) {
                break;
            }
            let n = path.points.len();
            let full_segs = if path.is_closed && n >= 3 { n } else { n - 1 };
            let emit_segs = rem_segs.map_or(full_segs, |lim| lim.min(full_segs));
            if emit_segs == 0 {
                continue;
            }

            chain.clear();
            let total_nodes = emit_segs + 1;
            chain.reserve(total_nodes);
            for step in 0..total_nodes {
                let pt = &path.points[step % n];
                let rgba = compute_toolpath_point_color(
                    path.feature_kind,
                    path.wall_index,
                    path.filament_slot,
                    pt.width_mm,
                    pt.tool_iso_angle_deg,
                    pt.uses_surface_normal,
                    pt.sdf_mm,
                    color_mode,
                    config,
                );
                chain.push(HqChainNode {
                    pos: Vec3::new(pt.x, pt.y, pt.z),
                    nozzle_axis: Vec3::new(pt.nx, pt.ny, pt.nz)
                        .try_normalize()
                        .unwrap_or(fallback_nozzle),
                    width_mm: pt.width_mm.max(0.06),
                    height_mm: pt.height_mm.max(0.04),
                    linear_rgba: scale_srgba_to_linear(rgba, dim_factor),
                    is_circular: path.feature_kind == 5,
                });
            }

            let key = classify_hq_raw_bucket(path, color_mode, config);
            let acc = buckets.entry(key).or_default();
            append_hq_welded_chain_stadium_mesh(&chain, fallback_nozzle, acc);

            if let Some(ref mut rem) = rem_segs {
                *rem = rem.saturating_sub(emit_segs);
            }
        }
    }

    finalize_hq_buckets(buckets)
}

fn finalize_hq_buckets(
    map: std::collections::BTreeMap<(u8, u16), HqBucketAccum>,
) -> Vec<HqBucketMesh> {
    let mut out = Vec::with_capacity(map.len());
    for ((tier, mat_slot), acc) in map {
        if acc.positions.is_empty() || acc.indices.is_empty() {
            continue;
        }
        let mean_thickness_mm = if acc.thickness_count > 0 {
            (acc.thickness_sum / (acc.thickness_count as f64)) as f32
        } else {
            0.40
        };
        out.push(HqBucketMesh {
            tier,
            mat_slot,
            mean_thickness_mm,
            positions: acc.positions,
            normals: acc.normals,
            colors: acc.colors,
            indices: acc.indices,
        });
    }
    out
}

/// Emits a continuous welded 12-vertex pressed-stadium (capsule) extrusion chain in each node's
/// local `NozzleFrame` $(\hat{\mathbf{x}}_{\text{travel}}, \hat{\mathbf{y}}_{\text{lateral}}, \hat{\mathbf{z}}_{\text{nozzle}})$
/// (`AGENTS.md` Rule 1 — never assuming `+Z`), splitting cleanly at self-intersecting / acute
/// miter corners into rounded bullnose sub-chains so sharp U-turns never fold inside-out.
fn append_hq_welded_chain_stadium_mesh(
    raw_nodes: &[HqChainNode],
    fallback_nozzle: Vec3,
    acc: &mut HqBucketAccum,
) {
    if raw_nodes.len() < 2 {
        return;
    }

    // Deduplicate near-coincident consecutive points while preserving closed-loop closure at the end
    let mut nodes: Vec<HqChainNode> = Vec::with_capacity(raw_nodes.len());
    for (idx, &nd) in raw_nodes.iter().enumerate() {
        if let Some(prev) = nodes.last() {
            let is_final = idx + 1 == raw_nodes.len();
            if prev.pos.distance_squared(nd.pos) < 1.0e-8 && !is_final {
                continue;
            }
        }
        nodes.push(nd);
    }
    if nodes.len() < 2 {
        return;
    }

    let is_closed =
        nodes.len() >= 4 && nodes[0].pos.distance_squared(nodes.last().unwrap().pos) <= 1.0e-5;
    if is_closed {
        nodes.pop(); // Ring `n - 1` wraps seamlessly back to Ring `0` with zero internal endcap!
    }
    let n = nodes.len();
    if n < 2 || (is_closed && n < 3) {
        return;
    }

    let hq_half_width = |nd: &HqChainNode| -> f32 {
        if nd.is_circular {
            0.5 * (std::f32::consts::FRAC_2_PI * 2.0 * nd.width_mm * nd.height_mm)
                .sqrt()
                .max(0.05)
        } else {
            (0.495 * nd.width_mm).max(0.028)
        }
    };

    // Detect folding / self-intersecting acute miters so sharp turns split into capped sub-chains
    let mut split_indices: Vec<usize> = Vec::new();
    if n >= 3 {
        for i in 0..n {
            if !is_closed && (i == 0 || i + 1 == n) {
                continue;
            }
            let prev_idx = (i + n - 1) % n;
            let next_idx = (i + 1) % n;
            let v_in = nodes[i].pos - nodes[prev_idx].pos;
            let v_out = nodes[next_idx].pos - nodes[i].pos;
            let len_in = v_in.length();
            let len_out = v_out.length();
            if len_in <= 1.0e-4 || len_out <= 1.0e-4 {
                continue;
            }
            let z_noz = nodes[i].nozzle_axis.try_normalize().unwrap_or(fallback_nozzle);
            let f_in = NozzleFrame::from_nozzle_and_travel(z_noz, v_in / len_in);
            let f_out = NozzleFrame::from_nozzle_and_travel(z_noz, v_out / len_out);
            let bisect_travel = (f_in.travel_axis + f_out.travel_axis)
                .try_normalize()
                .unwrap_or(f_out.travel_axis);
            let f_mid = NozzleFrame::from_nozzle_and_travel(z_noz, bisect_travel);
            let cos_half = f_mid.lateral_axis.dot(f_in.lateral_axis).abs().max(0.60);
            let miter_scale = (1.0 / cos_half).clamp(1.0, 1.62);
            let half_w = hq_half_width(&nodes[i]);
            let proj_in = half_w * miter_scale * f_mid.lateral_axis.dot(f_in.travel_axis).abs();
            let proj_out = half_w * miter_scale * f_mid.lateral_axis.dot(f_out.travel_axis).abs();
            let turn_cos = f_in.travel_axis.dot(f_out.travel_axis);

            if turn_cos < -0.15 || proj_in > 0.48 * len_in || proj_out > 0.48 * len_out {
                split_indices.push(i);
            }
        }
    }

    if split_indices.is_empty() {
        append_hq_welded_subchain_stadium_mesh(&nodes, is_closed, fallback_nozzle, acc);
        return;
    }

    if is_closed {
        let base = split_indices[0];
        let mut rotated: Vec<HqChainNode> = Vec::with_capacity(n + 1);
        rotated.extend_from_slice(&nodes[base..]);
        rotated.extend_from_slice(&nodes[..base]);
        rotated.push(nodes[base]);

        let mut boundaries: Vec<usize> = Vec::with_capacity(split_indices.len() + 2);
        boundaries.push(0);
        for &s in split_indices.iter().skip(1) {
            boundaries.push((s + n - base) % n);
        }
        boundaries.push(n);

        for w in boundaries.windows(2) {
            if w[1] > w[0] {
                append_hq_welded_subchain_stadium_mesh(
                    &rotated[w[0]..=w[1]],
                    false,
                    fallback_nozzle,
                    acc,
                );
            }
        }
    } else {
        let mut boundaries: Vec<usize> = Vec::with_capacity(split_indices.len() + 2);
        boundaries.push(0);
        boundaries.extend_from_slice(&split_indices);
        boundaries.push(n - 1);

        for w in boundaries.windows(2) {
            if w[1] > w[0] {
                append_hq_welded_subchain_stadium_mesh(
                    &nodes[w[0]..=w[1]],
                    false,
                    fallback_nozzle,
                    acc,
                );
            }
        }
    }
}

fn append_hq_welded_subchain_stadium_mesh(
    nodes: &[HqChainNode],
    is_closed: bool,
    fallback_nozzle: Vec3,
    acc: &mut HqBucketAccum,
) {
    let n = nodes.len();
    if n < 2 || (is_closed && n < 3) {
        return;
    }

    // Precomputed stadium meniscus angles & micro-groove AO factors:
    // 0..6: right meniscus [-pi/2 .. +pi/2], 6..12: left meniscus [+pi/2 .. +3pi/2]
    const ANGLES_COS_SIN_SIDE_AO: [(f32, f32, f32, f32); HQ_RING_VERTS] = [
        (0.0, -1.0, 1.0, 0.80),          // 0: bottom-right crease
        (0.58778524, -0.809017, 1.0, 0.91), // 1: -54°
        (0.95105654, -0.309017, 1.0, 1.00), // 2: -18°
        (0.95105654, 0.309017, 1.0, 1.00),  // 3: +18°
        (0.58778524, 0.809017, 1.0, 0.94),  // 4: +54°
        (0.0, 1.0, 1.0, 0.98),           // 5: top-right ironed land
        (0.0, 1.0, -1.0, 0.98),          // 6: top-left ironed land
        (-0.58778524, 0.809017, -1.0, 0.94), // 7: +126°
        (-0.95105654, 0.309017, -1.0, 1.00), // 8: +162°
        (-0.95105654, -0.309017, -1.0, 1.00), // 9: +198°
        (-0.58778524, -0.809017, -1.0, 0.91), // 10: +234°
        (0.0, -1.0, -1.0, 0.80),         // 11: bottom-left crease
    ];

    let ring_base = acc.positions.len() as u32;
    acc.positions.reserve(n * HQ_RING_VERTS + 2);
    acc.normals.reserve(n * HQ_RING_VERTS + 2);
    acc.colors.reserve(n * HQ_RING_VERTS + 2);
    let num_segs = if is_closed { n } else { n - 1 };
    acc.indices.reserve(num_segs * HQ_RING_VERTS * 6 + HQ_RING_VERTS * 6);

    let mut first_frame = NozzleFrame::from_nozzle_and_travel(fallback_nozzle, Vec3::X);
    let mut last_frame = first_frame;
    let mut first_cap_ext = 0.04_f32;
    let mut last_cap_ext = 0.04_f32;

    for i in 0..n {
        let nd = &nodes[i];
        let z_noz = nd.nozzle_axis.try_normalize().unwrap_or(fallback_nozzle);

        let dir_in_opt = if i > 0 {
            (nd.pos - nodes[i - 1].pos).try_normalize()
        } else if is_closed {
            (nd.pos - nodes[n - 1].pos).try_normalize()
        } else {
            None
        };
        let dir_out_opt = if i + 1 < n {
            (nodes[i + 1].pos - nd.pos).try_normalize()
        } else if is_closed {
            (nodes[0].pos - nd.pos).try_normalize()
        } else {
            None
        };

        let (frame, miter_scale) = match (dir_in_opt, dir_out_opt) {
            (Some(d_in), Some(d_out)) => {
                let f_in = NozzleFrame::from_nozzle_and_travel(z_noz, d_in);
                let f_out = NozzleFrame::from_nozzle_and_travel(z_noz, d_out);
                let bisect_travel = (f_in.travel_axis + f_out.travel_axis)
                    .try_normalize()
                    .unwrap_or(f_out.travel_axis);
                let f_mid = NozzleFrame::from_nozzle_and_travel(z_noz, bisect_travel);
                let cos_half = f_mid.lateral_axis.dot(f_in.lateral_axis).abs().max(0.60);
                (f_mid, (1.0 / cos_half).clamp(1.0, 1.62))
            }
            (None, Some(d_out)) => (NozzleFrame::from_nozzle_and_travel(z_noz, d_out), 1.0),
            (Some(d_in), None) => (NozzleFrame::from_nozzle_and_travel(z_noz, d_in), 1.0),
            (None, None) => (NozzleFrame::from_nozzle_and_travel(z_noz, Vec3::X), 1.0),
        };

        let (flat_a, r_lat, b_vert) = if nd.is_circular {
            // Free-air stretched circular bridge cross-section preserving exact bead area W * H
            let r_eq = (0.5 * (std::f32::consts::FRAC_2_PI * 2.0 * nd.width_mm * nd.height_mm).sqrt())
                .max(0.025);
            (0.002, r_eq * miter_scale, r_eq)
        } else {
            let b = (0.495 * nd.height_mm).max(0.018);
            let half_w = (0.495 * nd.width_mm).max(0.028);
            let r_m = b.min(half_w);
            let a = (half_w - r_m).max(0.002) * miter_scale;
            (a, r_m * miter_scale, b)
        };

        if i == 0 {
            first_frame = frame;
            first_cap_ext = 0.38 * b_vert.max(flat_a + r_lat);
        }
        if i + 1 == n {
            last_frame = frame;
            last_cap_ext = 0.38 * b_vert.max(flat_a + r_lat);
        }

        acc.thickness_sum += 0.5 * ((nd.width_mm + nd.height_mm) as f64);
        acc.thickness_count += 1;

        for &(cos_t, sin_t, side_sign, ao) in &ANGLES_COS_SIN_SIDE_AO {
            let local_y = side_sign * flat_a + r_lat * cos_t;
            let local_z = b_vert * sin_t;
            let v_pos = nd.pos + frame.lateral_axis * local_y + frame.nozzle_axis * local_z;
            let v_nrm = (frame.lateral_axis * cos_t + frame.nozzle_axis * sin_t)
                .try_normalize()
                .unwrap_or(frame.nozzle_axis);

            acc.positions.push(v_pos.to_array());
            acc.normals.push(v_nrm.to_array());
            acc.colors.push([
                nd.linear_rgba[0] * ao,
                nd.linear_rgba[1] * ao,
                nd.linear_rgba[2] * ao,
                nd.linear_rgba[3],
            ]);
        }
    }

    // Connect welded rings with outward FrontFace::Ccw triangles
    let ring_stride = HQ_RING_VERTS as u32;
    for s in 0..num_segs {
        let r0 = ring_base + (s as u32) * ring_stride;
        let r1 = ring_base + (((s + 1) % n) as u32) * ring_stride;
        for k in 0..ring_stride {
            let k_next = (k + 1) % ring_stride;
            let a0 = r0 + k;
            let a1 = r0 + k_next;
            let b0 = r1 + k;
            let b1 = r1 + k_next;
            acc.indices.extend_from_slice(&[a0, a1, b1, a0, b1, b0]);
        }
    }

    // Convex 3D bullnose endcaps at open chain start & end
    if !is_closed {
        let n0 = &nodes[0];
        let start_apex_idx = acc.positions.len() as u32;
        let start_apex_pos = n0.pos - first_frame.travel_axis * first_cap_ext;
        acc.positions.push(start_apex_pos.to_array());
        acc.normals.push((-first_frame.travel_axis).to_array());
        acc.colors.push(n0.linear_rgba);

        let r0 = ring_base;
        for k in 0..ring_stride {
            let k_next = (k + 1) % ring_stride;
            acc.indices
                .extend_from_slice(&[start_apex_idx, r0 + k_next, r0 + k]);
        }

        let n_end = &nodes[n - 1];
        let end_apex_idx = acc.positions.len() as u32;
        let end_apex_pos = n_end.pos + last_frame.travel_axis * last_cap_ext;
        acc.positions.push(end_apex_pos.to_array());
        acc.normals.push(last_frame.travel_axis.to_array());
        acc.colors.push(n_end.linear_rgba);

        let r_last = ring_base + ((n - 1) as u32) * ring_stride;
        for k in 0..ring_stride {
            let k_next = (k + 1) % ring_stride;
            acc.indices
                .extend_from_slice(&[end_apex_idx, r_last + k, r_last + k_next]);
        }
    }
}

/// Computes the RGBA color for a toolpath point according to `ToolpathColorMode`.
fn compute_toolpath_point_color(
    feature_kind: u8,
    wall_index: u32,
    filament_slot: u8,
    width_mm: f32,
    tool_iso_angle_deg: f32,
    uses_surface_normal: bool,
    sdf_mm: f32,
    mode: ToolpathColorMode,
    config: &SlicerConfig,
) -> [f32; 4] {
    match mode {
        ToolpathColorMode::ByFeatureType => match feature_kind {
            0 => [0.98, 0.44, 0.16, 1.0], // Outer Wall: vivid coral-orange
            1 => {
                // Inner Wall: emerald/lime green (with slight shade variation per wall index)
                let shade = (1.0 - 0.10 * ((wall_index.saturating_sub(1) % 4) as f32)).clamp(0.65, 1.0);
                [0.24 * shade, 0.90 * shade, 0.42 * shade, 1.0]
            }
            2 => [0.22, 0.76, 0.98, 1.0], // Solid Infill: bright sky-blue/cyan
            4 => [0.98, 0.78, 0.18, 1.0], // Sparse Infill: warm gold/amber
            5 => [0.18, 0.52, 0.98, 1.0], // Bridge: electric royal blue
            6 => {
                // Wave Overhang: vivid orchid-rose (with slight inside-out pass shade)
                let shade = (1.0 - 0.08 * ((wall_index.saturating_sub(1) % 4) as f32)).clamp(0.70, 1.0);
                [0.98 * shade, 0.26 * shade, 0.68 * shade, 1.0]
            }
            7 => [0.22, 0.88, 0.62, 1.0], // Support Base: seafoam/mint green
            8 => [0.12, 0.96, 0.88, 1.0], // Support Interface: bright aquamarine
            _ => [0.98, 0.32, 0.88, 1.0], // Gap Fill: vivid magenta
        },
        ToolpathColorMode::ByToolOrFilament | ToolpathColorMode::ByOpticalComposite => {
            if feature_kind == 7 || feature_kind == 8 {
                [0.22, 0.88, 0.62, 1.0]
            } else {
                config.color_srgba_for_filament_slot(filament_slot)
            }
        }
        ToolpathColorMode::ByToolIsoNormalAngle => {
            let max_ang = config
                .active_tool()
                .map(|t| t.max_angle_from_bed_normal_deg)
                .unwrap_or(45.0)
                .max(15.0);
            let t = (tool_iso_angle_deg / max_ang).clamp(0.0, 1.0);
            high_contrast_parameter_ramp(t)
        }
        ToolpathColorMode::BySurfaceNormalUsage => {
            if uses_surface_normal {
                [0.14, 0.92, 0.46, 1.0] // Vivid emerald green: Object Surface Normal
            } else if feature_kind == 0 || feature_kind == 6 {
                [0.98, 0.62, 0.16, 1.0] // Warm amber-orange: Isosurface Normal (Outer Fallback)
            } else {
                [0.24, 0.52, 0.92, 1.0] // Cool sapphire blue: Isosurface Normal (Interior / Other)
            }
        }
        ToolpathColorMode::ByNozzleSdf => sample_nozzle_sdf_rgba(
            sdf_mm,
            nozzle_sdf_reference_layer_height_mm(config),
        ),
        ToolpathColorMode::ByLineWidth | _ => {
            let spec = match feature_kind {
                0 | 6 => config.slicing.outer_wall_width,
                1 => config.slicing.inner_wall_width,
                4 => config.slicing.sparse_infill_width,
                _ => config.slicing.solid_infill_width,
            };
            let denom = (spec.max_mm - spec.min_mm).max(1.0e-4);
            let t = ((width_mm - spec.min_mm) / denom).clamp(0.0, 1.0);
            high_contrast_parameter_ramp(t)
        }
    }
}

/// Computes the RGBA color for a selected slicing isosurface layer according to `SelectedLayersColorMode`.
fn compute_selected_layer_color(
    layer: &ffi::SelectedLayerInfo,
    mode: SelectedLayersColorMode,
    min_mm: f32,
    _target_mm: f32,
    max_mm: f32,
) -> [f32; 4] {
    match mode {
        SelectedLayersColorMode::ByLayerThickness => {
            if layer.is_first_layer {
                return [0.18, 0.92, 0.96, 1.0]; // Crisp cyan for bed-anchored first layer
            }
            if layer.is_top_layer {
                return [0.96, 0.42, 0.92, 1.0]; // Vivid magenta-violet for branch-top layer
            }
            let h = layer.mean_thickness_mm;
            let denom = (max_mm - min_mm).max(1.0e-4);
            let t = ((h - min_mm) / denom).clamp(0.0, 1.0);
            high_contrast_parameter_ramp(t)
        }
        SelectedLayersColorMode::ByBranch => {
            let rgb = BRANCH_PALETTE[(layer.branch_id as usize) % BRANCH_PALETTE.len()];
            [rgb[0], rgb[1], rgb[2], 1.0]
        }
        SelectedLayersColorMode::AlternatingBands => {
            if layer.layer_index % 2 == 0 {
                [0.20, 0.84, 0.76, 1.0]
            } else {
                [0.96, 0.66, 0.22, 1.0]
            }
        }
    }
}

/// Computes per-vertex RGBA color for an extracted isosurface mesh so that for non-SDF fields,
/// the region intersecting the solid object ($\phi_{\text{sdf}} \le 0$) is sharply distinguished
/// from the exterior extension ($\phi_{\text{sdf}} > 0$).
fn compute_isosurface_vertex_color(
    field_name: &str,
    mode: IsoIntersectionMode,
    sdf_val: f32,
    min_sdf: f32,
) -> [f32; 4] {
    if field_name == "sdf" {
        return [0.16, 0.88, 0.58, 1.0];
    }

    let base_inside = match field_name {
        "eikonal" => [1.0, 0.68, 0.14],
        "overhang" => [0.22, 0.92, 0.72],
        "support_needed" => [0.98, 0.28, 0.36],
        _ => [0.85, 0.44, 0.98],
    };

    if mode == IsoIntersectionMode::FullUnclipped {
        return [base_inside[0], base_inside[1], base_inside[2], 1.0];
    }

    if sdf_val <= 0.0 {
        // Inside the object (\phi_sdf <= 0): shade smoothly from bright warm gold at the
        // perimeter (\phi_sdf = 0) to vivid field color toward the interior medial core.
        let depth_t = (sdf_val / min_sdf.min(-0.5)).clamp(0.0, 1.0);
        let r = (1.0 * (1.0 - 0.35 * depth_t) * base_inside[0] + 0.18 * (1.0 - depth_t)).clamp(0.0, 1.0);
        let g = (base_inside[1] * (0.85 + 0.15 * (1.0 - depth_t)) + 0.15 * depth_t).clamp(0.0, 1.0);
        let b = (base_inside[2] + 0.45 * depth_t).clamp(0.0, 1.0);
        [r, g, b, 1.0]
    } else {
        // Outside the object (\phi_sdf > 0): dim, cool dark slate-blue so the intersecting
        // interior patch stands out unmistakably.
        [0.18, 0.21, 0.29, 1.0]
    }
}

/// Converts a 2D grid of scalar values (`FieldSlice2D`) into an RGBA8 pixel buffer
/// using the selected `FieldColormap`, highlighting the object boundary ($\phi_{\text{sdf}} = 0$),
/// the active isosurface intersection contour ($v = \text{isovalue}$ inside $\phi_{\text{sdf}} \le 0$),
/// and any selected slicing isosurface contours when `selected_isovalues` is non-empty.
#[allow(clippy::too_many_arguments)]
fn rasterize_slice_to_rgba(
    slice: &FieldSlice2D,
    field_name: &str,
    active_isovalue: f32,
    colormap: FieldColormap,
    show_contours: bool,
    fade_background: bool,
    selected_isovalues: &[f32],
    highlighted_isovalue: Option<f32>,
) -> Vec<u8> {
    let total_pixels = (slice.width as usize) * (slice.height as usize);
    let mut out = Vec::with_capacity(total_pixels * 4);

    let min_v = slice.min_value;
    let max_v = slice.max_value;
    let span = (max_v - min_v).max(1e-5);
    let bg = slice.background_value;
    let bg_tol = span * 0.025;
    let is_non_sdf = field_name != "sdf";

    let max_sdf = slice
        .sdf_values
        .iter()
        .copied()
        .fold(1.0f32, f32::max)
        .max(1.0);
    let exterior_halo_limit = 0.45 * max_sdf;

    for (idx, &val) in slice.values.iter().enumerate() {
        if !val.is_finite() {
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }

        let sdf_v = slice.sdf_values.get(idx).copied().unwrap_or(val);
        let is_bg = ((val - bg).abs() <= bg_tol && sdf_v > 0.0) || (sdf_v > exterior_halo_limit);
        if fade_background && is_bg {
            // Alpha = 0 so AlphaMode::Mask(0.5) discards exterior background cleanly
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }

        let (mut r, mut g, mut b) = match colormap {
            FieldColormap::CoolWarmSigned => {
                let abs_max = min_v.abs().max(max_v.abs()).max(1e-4);
                let signed_t = (val / abs_max).clamp(-1.0, 1.0);
                if signed_t < 0.0 {
                    let s = (-signed_t).clamp(0.0, 1.0);
                    (
                        1.0 - 0.88 * s,
                        1.0 - 0.45 * s,
                        1.0 - 0.05 * s * s,
                    )
                } else {
                    let s = signed_t.clamp(0.0, 1.0);
                    (
                        1.0 - 0.08 * s * s,
                        1.0 - 0.72 * s,
                        1.0 - 0.88 * s,
                    )
                }
            }
            FieldColormap::ViridisSequential => {
                let t = ((val - min_v) / span).clamp(0.0, 1.0);
                viridis_approx(t)
            }
            FieldColormap::ZebraIsoBands => {
                let t = ((val - min_v) / span).clamp(0.0, 1.0);
                let wave = (t * 16.0 * std::f32::consts::PI).sin();
                let band = if wave >= 0.0 { 0.88 } else { 0.18 };
                let (vr, vg, vb) = viridis_approx(t);
                (vr * band, vg * band, vb * band)
            }
        };

        // Dim exterior region (\phi_sdf > 0) slightly on non-SDF slices so the object interior pops
        if is_non_sdf && sdf_v > 0.0 {
            r *= 0.45;
            g *= 0.45;
            b *= 0.50;
        }

        if show_contours {
            let iso_band = span * 0.014;
            let selected_band = span * 0.008;
            if is_non_sdf && sdf_v.abs() < 0.22 {
                // Crisp white outline along the object boundary (\phi_sdf = 0)
                r = 0.96;
                g = 0.98;
                b = 1.0;
            } else if is_non_sdf
                && sdf_v <= 0.0
                && highlighted_isovalue.is_some_and(|hi| (val - hi).abs() < iso_band * 1.25)
            {
                // Hovered selected layer contour in bright gold
                r = 1.0;
                g = 0.96;
                b = 0.22;
            } else if is_non_sdf && (val - active_isovalue).abs() < iso_band && sdf_v <= 0.0 {
                // Bright cyan highlight where the active non-SDF isosurface intersects the object!
                r = 0.15;
                g = 1.0;
                b = 0.92;
            } else if is_non_sdf
                && sdf_v <= 0.0
                && !selected_isovalues.is_empty()
                && selected_isovalues.iter().any(|&iso| (val - iso).abs() < selected_band)
            {
                // Selected slicing layer contours overlaid on the 2D cross-section plane
                r = 0.25;
                g = 0.96;
                b = 0.58;
            } else if !is_non_sdf && min_v < 0.0 && max_v > 0.0 && val.abs() < iso_band {
                r = 1.0;
                g = 1.0;
                b = 1.0;
            } else if selected_isovalues.is_empty() {
                let norm = ((val - min_v) / span).clamp(0.0, 1.0);
                let contour_phase = (norm * 12.0).fract();
                if contour_phase < 0.06 || contour_phase > 0.94 {
                    r *= 0.35;
                    g *= 0.35;
                    b *= 0.35;
                }
            }
        }

        out.push((r.clamp(0.0, 1.0) * 255.0) as u8);
        out.push((g.clamp(0.0, 1.0) * 255.0) as u8);
        out.push((b.clamp(0.0, 1.0) * 255.0) as u8);
        out.push(255);
    }

    out
}

fn viridis_approx(t: f32) -> (f32, f32, f32) {
    let x = t.clamp(0.0, 1.0);
    let r = (0.267 + x * (-0.004 + x * (2.780 - x * 2.050))).clamp(0.0, 1.0);
    let g = (0.005 + x * (1.380 - x * 0.480)).clamp(0.0, 1.0);
    let b = (0.329 + x * (1.450 + x * (-2.650 + x * 1.100))).clamp(0.0, 1.0);
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{ffi, TEST_BACKEND_LOCK};

    #[test]
    fn test_openvdb_sdf_and_generic_field_visualization_pipeline() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let positions: [f32; 24] = [
            -10.0, -10.0, 0.0,
             10.0, -10.0, 0.0,
             10.0,  10.0, 0.0,
            -10.0,  10.0, 0.0,
            -10.0, -10.0, 20.0,
             10.0, -10.0, 20.0,
             10.0,  10.0, 20.0,
            -10.0,  10.0, 20.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, // bottom
            4, 5, 6, 4, 6, 7, // top
            0, 1, 5, 0, 5, 4, // front
            1, 2, 6, 1, 6, 5, // right
            2, 3, 7, 2, 7, 6, // back
            3, 0, 4, 3, 4, 7, // left
        ];

        let part = ffi::register_triangle_mesh("test_sdf_cube", &positions, &indices);
        assert!(part.object_id > 0);

        // Fields are not computed on initial object load
        assert!(ffi::list_object_fields(part.object_id).is_empty());

        // Compute fields on demand and verify that voxel resolution preserves full interior & exterior millimeter range
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 6.0));
        let fields = ffi::list_object_fields(part.object_id);
        assert!(fields.contains(&"sdf".to_string()));
        assert!(fields.contains(&"eikonal".to_string()));
        assert!(fields.contains(&"overhang".to_string()));
        let sdf_stats = ffi::get_object_field_stats(part.object_id, "sdf");
        assert!(sdf_stats.valid);
        assert!(sdf_stats.active_voxel_count > 0);
        assert!(
            sdf_stats.min_value < -5.0 && sdf_stats.max_value > 3.0,
            "Expected full interior/exterior SDF range in mm regardless of voxel size, got [{}, {}]",
            sdf_stats.min_value,
            sdf_stats.max_value
        );

        // Verify non-SDF ("eikonal") isosurface extraction splits interior (\phi <= 0)
        // vs exterior (\phi > 0) triangles and extracts the 3D object intersection contour!
        let eik_iso = ffi::extract_field_isosurface(part.object_id, "eikonal", 10.0, 0.0);
        assert!(!eik_iso.vertices.is_empty());
        assert_eq!(eik_iso.vertices.len(), eik_iso.vertex_sdf.len());
        assert!(
            !eik_iso.interior_indices.is_empty(),
            "Expected non-empty interior_indices where eikonal=10 intersects the cube"
        );
        assert!(
            !eik_iso.exterior_indices.is_empty(),
            "Expected non-empty exterior_indices in the narrow band outside the cube"
        );
        assert!(
            eik_iso.intersection_contour.len() >= 6,
            "Expected 3D intersection boundary contour segments where eikonal=10 crosses phi_sdf=0"
        );
        // Verify every interior triangle vertex has phi_sdf <= 0
        for &idx in &eik_iso.interior_indices {
            let sdf_v = eik_iso.vertex_sdf[idx as usize];
            assert!(
                sdf_v <= 0.0,
                "Interior triangle vertex had positive SDF {}",
                sdf_v
            );
        }

        let slice = ffi::sample_field_slice(part.object_id, "eikonal", 2, 0.5, 32);
        assert!(slice.valid);
        assert_eq!(slice.width, 32);
        assert_eq!(slice.height, 32);
        assert_eq!(slice.values.len(), 32 * 32);
        assert_eq!(slice.sdf_values.len(), 32 * 32);

        let rgba = rasterize_slice_to_rgba(
            &slice,
            "eikonal",
            10.0,
            FieldColormap::CoolWarmSigned,
            true,
            true,
            &[5.0, 10.0, 15.0],
            Some(10.0),
        );
        assert_eq!(rgba.len(), 32 * 32 * 4);

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_soft_orientation_hints_respect_hard_tool_limits() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let positions: [f32; 24] = [
            -8.0, -8.0, 0.0,
             8.0, -8.0, 0.0,
             8.0,  8.0, 0.0,
            -8.0,  8.0, 0.0,
            -8.0, -8.0, 16.0,
             8.0, -8.0, 16.0,
             8.0,  8.0, 16.0,
            -8.0,  8.0, 16.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_hint_cube", &positions, &indices);
        assert!(part.object_id > 0);

        // Configure a 5/6-axis tool with a strict 25 deg max tilt limit
        let tools = vec![ffi::ToolSpec {
            id: 0,
            name: "5AxisTest".to_string(),
            nozzle_diameter_mm: 0.4,
            temperature_celsius: 215.0,
            has_rotational_axes: true,
            max_angle_from_bed_normal_deg: 25.0,
            clearance_profile_xz: vec![
                ffi::ClearanceXZ { x_radius_mm: 0.6, z_height_mm: 0.0 },
                ffi::ClearanceXZ { x_radius_mm: 4.0, z_height_mm: 5.0 },
            ],
            ..Default::default()
        }];
        ffi::set_machine_tools(&tools, 0);
        ffi::set_eikonal_solver_iterations(3);

        // 1. Baseline without hints
        ffi::set_spatial_orientation_hints(&[]);
        assert!(ffi::recompute_object_fields(part.object_id, 1.0, 3.0));
        let baseline_slice = ffi::sample_field_slice(part.object_id, "eikonal", 1, 0.5, 16);
        assert!(baseline_slice.valid);

        // 2. Apply strong soft hints (both surface tangency and an extreme 80-deg point hint)
        let hints = vec![
            ffi::SpatialOrientationHint {
                kind: 1, // Orthogonality point hint requesting ~80 deg tilt (exceeds 25 deg tool limit!)
                position_bed: [0.0, 0.0, 10.0],
                direction_bed: [0.9848, 0.0, 0.1736],
                influence_radius_mm: 14.0,
                weight: 1.0,
                apply_on_surface_only: false,
                skin_layer_count: 0,
            },
            ffi::SpatialOrientationHint {
                kind: 0, // Surface tangency hint on +X wall
                position_bed: [8.0, 0.0, 8.0],
                direction_bed: [1.0, 0.0, 0.0],
                influence_radius_mm: 8.0,
                weight: 0.9,
                apply_on_surface_only: true,
                skin_layer_count: 0,
            },
        ];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(part.object_id, 1.0, 3.0));

        let hinted_slice = ffi::sample_field_slice(part.object_id, "eikonal", 1, 0.5, 16);
        assert!(hinted_slice.valid);

        // Verify the soft hint genuinely tilted the Eikonal isosurfaces across the object (> 1.5 mm change across 16 mm span)
        let max_diff = baseline_slice
            .values
            .iter()
            .zip(hinted_slice.values.iter())
            .zip(hinted_slice.sdf_values.iter())
            .filter(|((_, _), &sdf)| sdf <= 0.0)
            .map(|((&a, &b), _)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1.5,
            "Expected soft orientation hints to strongly tilt interior Eikonal isosurfaces, max_diff={max_diff}"
        );

        // Verify hard tool angle limit (25 deg) was never exceeded despite 80-deg hint
        let tilt_stats = ffi::get_object_field_stats(part.object_id, "tool_tilt_deg");
        assert!(tilt_stats.valid);
        assert!(
            tilt_stats.max_value > 15.0 && tilt_stats.max_value <= 25.05,
            "Soft hint must tilt tool axis up to limit without exceeding max_angle_from_bed_normal_deg (25 deg), got max tilt {}",
            tilt_stats.max_value
        );

        // 3. Verify that the clearance constraint explicitly tracks the actual tool axis:
        // With a very narrow clearance cone (r = 5.0 at z = 1.0 -> alpha_max ≈ 11.3 deg):
        // - If has_rotational_axes = false (fixed vertical tool axis +Z), isosurface tilt is clamped near 11.3 deg.
        // - If has_rotational_axes = true (tilted tool axis up to 25 deg), the clearance cone tilts with the tool axis,
        //   allowing significantly steeper isosurfaces!
        let narrow_fixed_tool = vec![ffi::ToolSpec {
            id: 0,
            name: "FixedNarrowCone".to_string(),
            nozzle_diameter_mm: 0.4,
            temperature_celsius: 215.0,
            has_rotational_axes: false,
            max_angle_from_bed_normal_deg: 25.0,
            clearance_profile_xz: vec![
                ffi::ClearanceXZ { x_radius_mm: 0.6, z_height_mm: 0.0 },
                ffi::ClearanceXZ { x_radius_mm: 5.0, z_height_mm: 1.0 },
            ],
            ..Default::default()
        }];
        ffi::set_machine_tools(&narrow_fixed_tool, 0);
        assert!(ffi::recompute_object_fields(part.object_id, 1.0, 3.0));
        let fixed_tool_slice = ffi::sample_field_slice(part.object_id, "eikonal", 1, 0.5, 16);

        let mut narrow_rot_tool = narrow_fixed_tool.clone();
        narrow_rot_tool[0].has_rotational_axes = true;
        ffi::set_machine_tools(&narrow_rot_tool, 0);
        assert!(ffi::recompute_object_fields(part.object_id, 1.0, 3.0));
        let rot_tool_slice = ffi::sample_field_slice(part.object_id, "eikonal", 1, 0.5, 16);

        let tool_axis_clearance_diff = fixed_tool_slice
            .values
            .iter()
            .zip(rot_tool_slice.values.iter())
            .zip(rot_tool_slice.sdf_values.iter())
            .filter(|((_, _), &sdf)| sdf <= 0.0)
            .map(|((&a, &b), _)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            tool_axis_clearance_diff > 0.5,
            "Expected clearance constraint to follow the actual tilted tool axis vs fixed vertical axis, diff={tool_axis_clearance_diff}"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_bidirectional_branch_aware_isosurface_selection() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);

        // Build a two-tower (forked) mesh on a shared base trunk:
        // - Base trunk: [-14, 14] x [-6, 6] x [0, 8]
        // - Left tower: [-14, -4] x [-6, 6] x [8, 20] (peak Z = 20)
        // - Right tower: [4, 14] x [-6, 6] x [8, 16] (peak Z = 16)
        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut add_box = |x0: f32, x1: f32, y0: f32, y1: f32, z0: f32, z1: f32| {
            let base = (positions.len() / 3) as u32;
            positions.extend_from_slice(&[
                x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
                x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
            ]);
            indices.extend_from_slice(&[
                base, base + 2, base + 1, base, base + 3, base + 2,
                base + 4, base + 5, base + 6, base + 4, base + 6, base + 7,
                base, base + 1, base + 5, base, base + 5, base + 4,
                base + 1, base + 2, base + 6, base + 1, base + 6, base + 5,
                base + 2, base + 3, base + 7, base + 2, base + 7, base + 6,
                base + 3, base, base + 4, base + 3, base + 4, base + 7,
            ]);
        };
        add_box(-14.0, 14.0, -6.0, 6.0, 0.0, 8.2);
        add_box(-14.0, -4.0, -6.0, 6.0, 7.8, 20.0);
        add_box(4.0, 14.0, -6.0, 6.0, 7.8, 16.0);

        let part = ffi::register_triangle_mesh("test_two_tower_fork", &positions, &indices);
        assert!(part.object_id > 0);
        assert!(ffi::recompute_object_fields(part.object_id, 0.8, 4.0));

        // Configure distinct first_layer_mm = 0.30, top_layer_mm = 0.15,
        // and target_mm = 0.50 in [min_mm = 0.35, max_mm = 0.65].
        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 0.50,
            min_mm: 0.35,
            max_mm: 0.65,
            first_layer_mm: 0.30,
            top_layer_mm: 0.15,
        });

        let selected = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(selected.valid);
        assert!(
            selected.branch_count >= 2,
            "Expected multi-branch Reeb detection on two-tower forked geometry, got branch_count={}",
            selected.branch_count
        );
        assert!(!selected.layers.is_empty());

        // 1. Verify the first layer off the bed uses first_layer_mm (0.30 mm)
        let first_layers: Vec<_> = selected.layers.iter().filter(|l| l.is_first_layer).collect();
        assert!(!first_layers.is_empty(), "Expected at least one bed-anchored first layer");
        for fl in &first_layers {
            assert!(
                (fl.mean_thickness_mm - 0.30).abs() < 0.04,
                "Expected first layer thickness ~0.30 mm, got {}",
                fl.mean_thickness_mm
            );
        }

        // 2. Verify top layers exist for both terminal tower branches (Z=16 and Z=20)
        // and reach all the way to `peak_tau - top_layer_mm`!
        let mut top_layers: Vec<_> = selected.layers.iter().filter(|l| l.is_top_layer).collect();
        top_layers.sort_by(|a, b| a.isovalue.partial_cmp(&b.isovalue).unwrap());
        assert_eq!(
            top_layers.len(),
            2,
            "Expected exactly 2 terminal tower top layers, got {}",
            top_layers.len()
        );
        assert!(
            (top_layers[0].isovalue - (16.0 - 0.15)).abs() < 0.45,
            "Expected right tower top layer near 15.85 mm, got {}",
            top_layers[0].isovalue
        );
        assert!(
            (top_layers[1].isovalue - (20.0 - 0.15)).abs() < 0.45,
            "Expected left tower top layer near 19.85 mm, got {}",
            top_layers[1].isovalue
        );

        // 3. Verify intermediate layers respect [min_mm, max_mm] and EVERY layer (including top layers)
        // has non-empty extracted 3D interior geometry!
        for layer in &selected.layers {
            if !layer.is_first_layer {
                assert!(
                    layer.mean_thickness_mm >= 0.35 - 0.03 && layer.mean_thickness_mm <= 0.65 + 0.03,
                    "Layer {} on branch {} had mean_thickness_mm={} outside [0.35, 0.65]",
                    layer.layer_index,
                    layer.branch_id,
                    layer.mean_thickness_mm
                );
            }
            assert!(
                !layer.indices.is_empty(),
                "Layer {} (iso={}, is_first={}, is_top={}) produced an empty 3D mesh!",
                layer.layer_index,
                layer.isovalue,
                layer.is_first_layer,
                layer.is_top_layer
            );
        }

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_tilted_isosurface_selection_reaches_last_corner() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let positions: [f32; 24] = [
            -10.0, -10.0, 0.0,
             10.0, -10.0, 0.0,
             10.0,  10.0, 0.0,
            -10.0,  10.0, 0.0,
            -10.0, -10.0, 20.0,
             10.0, -10.0, 20.0,
             10.0,  10.0, 20.0,
            -10.0,  10.0, 20.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_tilted_corner_cube", &positions, &indices);
        assert!(part.object_id > 0);

        let tools = vec![ffi::ToolSpec {
            id: 0,
            name: "5AxisTest".to_string(),
            nozzle_diameter_mm: 0.4,
            temperature_celsius: 215.0,
            has_rotational_axes: true,
            max_angle_from_bed_normal_deg: 40.0,
            clearance_profile_xz: vec![
                ffi::ClearanceXZ { x_radius_mm: 0.5, z_height_mm: 0.0 },
                ffi::ClearanceXZ { x_radius_mm: 3.0, z_height_mm: 6.0 },
            ],
            ..Default::default()
        }];
        ffi::set_machine_tools(&tools, 0);
        ffi::set_eikonal_solver_iterations(4);

        // Apply a diagonal 3D orientation hint tilting the field toward the (+10, +10, 20) top corner
        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 0.0, 10.0],
            direction_bed: [0.50, 0.50, 0.7071],
            influence_radius_mm: 25.0,
            weight: 1.0,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(part.object_id, 0.6, 4.0));

        // Sample the XZ and YZ slices to find the true maximum interior Eikonal value near the corner
        let slice_xz = ffi::sample_field_slice(part.object_id, "eikonal", 1, 0.8, 48);
        let max_interior_tau = slice_xz
            .values
            .iter()
            .zip(slice_xz.sdf_values.iter())
            .filter(|(_, &sdf)| sdf <= 0.0)
            .map(|(&tau, _)| tau)
            .fold(0.0f32, f32::max);
        assert!(
            max_interior_tau > 20.5,
            "Expected tilted cube peak tau > 20.5 mm at top corner, got {max_interior_tau}"
        );

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 0.40,
            min_mm: 0.25,
            max_mm: 0.55,
            first_layer_mm: 0.30,
            top_layer_mm: 0.20,
        });

        let selected = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(selected.valid);
        assert_eq!(
            selected.branch_count, 1,
            "Single-peaked tilted cube must remain 1 unbroken branch, got {}",
            selected.branch_count
        );
        let last_layer = selected.layers.last().expect("Expected non-empty selected layers");
        assert!(last_layer.is_top_layer);
        assert!(
            last_layer.isovalue >= max_interior_tau - 0.55,
            "Expected last selected isosurface ({}) to reach the last corner (max_interior_tau={})",
            last_layer.isovalue,
            max_interior_tau
        );
        assert!(
            !last_layer.indices.is_empty(),
            "Expected non-empty 3D mesh on the final corner layer"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_shell_path_generation_monotonic_and_concentric() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);

        // Register a 16x16x8 mm block on the bed
        let positions: [f32; 24] = [
            -8.0, -8.0, 0.0,
             8.0, -8.0, 0.0,
             8.0,  8.0, 0.0,
            -8.0,  8.0, 0.0,
            -8.0, -8.0, 8.0,
             8.0, -8.0, 8.0,
             8.0,  8.0, 8.0,
            -8.0,  8.0, 8.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_shell_paths_block", &positions, &indices);
        assert!(part.object_id > 0);
        assert!(ffi::recompute_object_fields(part.object_id, 0.6, 4.0));

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 1.0,
            min_mm: 0.6,
            max_mm: 1.4,
            first_layer_mm: 0.5,
            top_layer_mm: 0.4,
        });

        let min_len = 0.80f32; // 2x 0.4mm nozzle
        let base_spec = ffi::ShellPathSpec {
            shell_thickness_mm: 2.0,
            inner_wall_count: 2,
            outer_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.28,
                max_mm: 0.58,
            },
            inner_wall_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.60,
            },
            solid_infill_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.62,
            },
            solid_infill_pattern: 0, // Monotonic
            min_extrusion_length_mm: min_len,
            sparse_infill_density: 0.0, // Tested separately in sparse infill test
            sparse_infill_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.60,
            },
            sparse_infill_pattern: 0,
            infill_angle_deg: 45.0,
            nozzle_diameter_mm: 0.4,
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 40.0,
            wave_overhang_overlap_fraction: 0.08,
            one_perimeter_on_conformal_patches: true,
            orient_outer_surfaces_to_normal: true,
            outer_surface_parallel_angle_deg: 15.0,
            brick_layer_mode: 0,
            enable_top_surface_lofting: true,
        };
        ffi::set_shell_path_spec(base_spec.clone());

        let mono_res = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(mono_res.valid);
        assert!(mono_res.outer_wall_count > 0, "Expected outer walls to be generated");
        assert!(mono_res.inner_wall_count > 0, "Expected inner walls to be generated");
        assert!(
            mono_res.solid_infill_count > 0,
            "Expected Monotonic solid infill to be generated in the remaining shell"
        );

        // Verify min extrusion length and line width bounds on every generated path,
        // and verify that individual segments within a path are shorter than min_extrusion_length_mm
        let mut found_sub_min_segment = false;
        for layer in &mono_res.layers {
            for path in &layer.paths {
                assert!(
                    path.total_length_mm >= min_len - 1.0e-4,
                    "Continuous extrusion path length {} was shorter than min_extrusion_length_mm {}",
                    path.total_length_mm,
                    min_len
                );
                assert!(path.points.len() >= 2);
                for w in path.points.windows(2) {
                    let dx = w[1].x - w[0].x;
                    let dy = w[1].y - w[0].y;
                    let dz = w[1].z - w[0].z;
                    let seg_len = (dx * dx + dy * dy + dz * dz).sqrt();
                    if seg_len < min_len {
                        found_sub_min_segment = true;
                    }
                }
                for pt in &path.points {
                    match path.feature_kind {
                        0 => assert!(
                            pt.width_mm >= 0.28 - 1.0e-3 && pt.width_mm <= 0.58 + 1.0e-3,
                            "Outer wall width {} outside [0.28, 0.58]",
                            pt.width_mm
                        ),
                        1 => assert!(
                            pt.width_mm >= 0.30 - 1.0e-3 && pt.width_mm <= 0.60 + 1.0e-3,
                            "Inner wall width {} outside [0.30, 0.60]",
                            pt.width_mm
                        ),
                        _ => assert!(
                            pt.width_mm >= 0.30 - 1.0e-3 && pt.width_mm <= 0.62 + 1.0e-3,
                            "Solid infill width {} outside [0.30, 0.62]",
                            pt.width_mm
                        ),
                    }
                }
            }
        }
        assert!(
            found_sub_min_segment,
            "Expected individual polyline segments shorter than min_extrusion_length_mm inside continuous paths"
        );

        // Verify mid-height horizontal layer restricts paths to the 2.0 mm outer shell
        // (leaving the central core [-6, 6] x [-6, 6] empty for future sparse infill)
        let mid_layer = mono_res
            .layers
            .iter()
            .find(|l| (l.isovalue - 4.0).abs() < 1.2)
            .expect("Expected a mid-height layer near Z=4.0");
        assert!(!mid_layer.paths.is_empty());
        for path in &mid_layer.paths {
            for pt in &path.points {
                let dist_from_outer_xy = (8.0 - pt.x.abs()).min(8.0 - pt.y.abs());
                assert!(
                    dist_from_outer_xy <= 2.65,
                    "Mid-height path point ({}, {}, {}) penetrated into sparse core beyond shell thickness (dist={})",
                    pt.x,
                    pt.y,
                    pt.z,
                    dist_from_outer_xy
                );
            }
        }

        // Now test Concentric solid infill pattern (pattern = 1)
        let mut conc_spec = base_spec;
        conc_spec.solid_infill_pattern = 1;
        ffi::set_shell_path_spec(conc_spec);

        let conc_res = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(conc_res.valid);
        assert!(conc_res.outer_wall_count > 0);
        assert!(conc_res.inner_wall_count > 0);
        assert!(
            conc_res.solid_infill_count > 0,
            "Expected Concentric solid infill to be generated in the remaining shell"
        );

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_tilted_isosurface_open_and_closed_walls() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Build a wedge/tapered block with a tilted orientation hint so non-planar isosurfaces
        // slice obliquely across the outer shell and produce both open and closed wall paths.
        let positions: [f32; 24] = [
            -8.0, -5.0, 0.0,
             8.0, -5.0, 0.0,
             8.0,  5.0, 0.0,
            -8.0,  5.0, 0.0,
            -8.0, -1.2, 10.0,
             8.0, -1.2, 10.0,
             8.0,  1.2, 10.0,
            -8.0,  1.2, 10.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_tapered_wedge_paths", &positions, &indices);
        assert!(part.object_id > 0);

        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 0.0, 5.0],
            direction_bed: [0.55, 0.0, 0.835],
            influence_radius_mm: 20.0,
            weight: 1.0,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 0.8,
            min_mm: 0.5,
            max_mm: 1.1,
            first_layer_mm: 0.4,
            top_layer_mm: 0.3,
        });

        ffi::set_shell_path_spec(ffi::ShellPathSpec {
            shell_thickness_mm: 1.8,
            inner_wall_count: 2,
            outer_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.26,
                max_mm: 0.58,
            },
            inner_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.26,
                max_mm: 0.58,
            },
            solid_infill_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.26,
                max_mm: 0.58,
            },
            solid_infill_pattern: 0,
            min_extrusion_length_mm: 0.8,
            sparse_infill_density: 0.20,
            sparse_infill_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.28,
                max_mm: 0.58,
            },
            sparse_infill_pattern: 0,
            infill_angle_deg: 45.0,
            nozzle_diameter_mm: 0.4,
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 40.0,
            wave_overhang_overlap_fraction: 0.08,
            one_perimeter_on_conformal_patches: true,
            orient_outer_surfaces_to_normal: true,
            outer_surface_parallel_angle_deg: 15.0,
            brick_layer_mode: 0,
            enable_top_surface_lofting: true,
        });

        let res = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(res.valid);
        assert!(res.total_path_count > 0);

        let any_open_wall = res
            .layers
            .iter()
            .flat_map(|l| l.paths.iter())
            .any(|p| (p.feature_kind == 0 || p.feature_kind == 1) && !p.is_closed);
        let any_closed_wall = res
            .layers
            .iter()
            .flat_map(|l| l.paths.iter())
            .any(|p| (p.feature_kind == 0 || p.feature_kind == 1) && p.is_closed);
        assert!(
            any_open_wall,
            "Expected non-planar tilted/tapered isosurfaces to produce open wall paths where walls terminate at the shell or clearance boundary"
        );
        assert!(
            any_closed_wall,
            "Expected closed wall loops on full-perimeter cross-sections"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_testobj1_top_layers_have_unbroken_isosurfaces_and_toolpaths() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let stl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/TestObj1.stl");
        let obj = crate::io::stl::load_stl_file(&stl_path).expect("Failed to load TestObj1.stl");

        let cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(layers.valid);
        assert!(tp.valid);
        assert_eq!(layers.layers.len(), tp.layers.len());

        let planned = crate::gcode::plan_and_emit_gcode_for_object(
            "TestObj1.stl",
            obj.object_id,
            &tp,
            Transform::IDENTITY,
            &cfg,
        );
        for (i, ((l, _t), pl)) in layers
            .layers
            .iter()
            .zip(tp.layers.iter())
            .zip(planned.layers.iter())
            .enumerate()
        {
            let prev_iso = layers.layers[..i]
                .iter()
                .rev()
                .find(|prev| prev.branch_id == l.branch_id || prev.branch_id as i32 == l.parent_branch_id)
                .map(|prev| prev.isovalue)
                .unwrap_or(0.0);
            let d_iso = l.isovalue - prev_iso;
            assert!(
                d_iso >= 0.10,
                "Layer #{i} (branch B{}, {}/{}) had overlapping/compressed dIso={d_iso:.3} (iso={:.3}, prev_iso={prev_iso:.3})",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.isovalue,
            );

            let ext_segs: Vec<_> = pl
                .segments
                .iter()
                .filter(|s| s.kind.is_extruding() && s.delta_e_mm > 0.0)
                .collect();
            assert!(
                !ext_segs.is_empty(),
                "Layer #{i} (branch B{}) must have extruding G-code segments",
                l.branch_id
            );
            let non_bridge_segs: Vec<_> = ext_segs
                .iter()
                .filter(|s| s.kind != crate::gcode::PlannedMoveKind::Bridge)
                .collect();
            if !non_bridge_segs.is_empty() {
                let mean_seg_h = non_bridge_segs.iter().map(|s| s.height_mm).sum::<f32>()
                    / (non_bridge_segs.len() as f32);
                if (l.is_top_layer || l.branch_layer_index + 2 >= l.branch_layer_count)
                    && cfg.slicing.enable_top_surface_lofting
                {
                    assert!(
                        mean_seg_h >= l.mean_thickness_mm - 0.03
                            && mean_seg_h <= cfg.slicing.layer_height.max_mm + 1.0e-4,
                        "Top Layer #{i} (branch B{}, {}/{}) mean planned segment height ({mean_seg_h:.3} mm) out of loft bounds [{:.3}, {:.3}]",
                        l.branch_id,
                        l.branch_layer_index + 1,
                        l.branch_layer_count,
                        l.mean_thickness_mm - 0.03,
                        cfg.slicing.layer_height.max_mm,
                    );
                } else {
                    assert!(
                        (mean_seg_h - l.mean_thickness_mm).abs() <= 0.03,
                        "Layer #{i} (branch B{}, {}/{}) mean planned segment height ({mean_seg_h:.3} mm) diverged from layer mean_thickness_mm ({:.3} mm)",
                        l.branch_id,
                        l.branch_layer_index + 1,
                        l.branch_layer_count,
                        l.mean_thickness_mm,
                    );
                }
            }
        }

        let q_pt = |v: &ffi::RenderVertex| -> (i32, i32, i32) {
            (
                (v.x * 1000.0).round() as i32,
                (v.y * 1000.0).round() as i32,
                (v.z * 1000.0).round() as i32,
            )
        };
        for (i, l) in layers.layers.iter().enumerate() {
            let mut edge_counts: std::collections::HashMap<((i32, i32, i32), (i32, i32, i32)), usize> =
                std::collections::HashMap::new();
            for tri in l.indices.chunks_exact(3) {
                let p0 = q_pt(&l.vertices[tri[0] as usize]);
                let p1 = q_pt(&l.vertices[tri[1] as usize]);
                let p2 = q_pt(&l.vertices[tri[2] as usize]);
                for &(a, b) in &[(p0, p1), (p1, p2), (p2, p0)] {
                    if a == b {
                        continue;
                    }
                    let key = if a < b { (a, b) } else { (b, a) };
                    *edge_counts.entry(key).or_insert(0) += 1;
                }
            }
            let boundary_edges = edge_counts.values().filter(|&&c| c == 1).count();
            let contour_segs = l.contour_segments.len() / 6;
            let hole_edges = boundary_edges.saturating_sub(contour_segs);
            assert!(
                !l.indices.is_empty(),
                "Layer #{i} (branch B{}, {}/{}) at iso={:.3} must not have an empty mesh",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.isovalue
            );
            assert_eq!(
                hole_edges,
                0,
                "Layer #{i} (branch B{}, {}/{}) at iso={:.3} has {hole_edges} spurious interior hole edges (bnd_edges={boundary_edges}, contour_segs={contour_segs})",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.isovalue
            );
        }

        // Also test with a tilted orientation hint on TestObj1.stl
        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 0.0, 5.0],
            direction_bed: [0.35, 0.35, 0.87],
            influence_radius_mm: 40.0,
            weight: 0.9,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));
        let tilted_layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        for (i, l) in tilted_layers.layers.iter().enumerate() {
            let mut edge_counts: std::collections::HashMap<((i32, i32, i32), (i32, i32, i32)), usize> =
                std::collections::HashMap::new();
            for tri in l.indices.chunks_exact(3) {
                let p0 = q_pt(&l.vertices[tri[0] as usize]);
                let p1 = q_pt(&l.vertices[tri[1] as usize]);
                let p2 = q_pt(&l.vertices[tri[2] as usize]);
                for &(a, b) in &[(p0, p1), (p1, p2), (p2, p0)] {
                    if a == b {
                        continue;
                    }
                    let key = if a < b { (a, b) } else { (b, a) };
                    *edge_counts.entry(key).or_insert(0) += 1;
                }
            }
            let boundary_edges = edge_counts.values().filter(|&&c| c == 1).count();
            let contour_segs = l.contour_segments.len() / 6;
            let hole_edges = boundary_edges.saturating_sub(contour_segs);
            assert!(
                !l.indices.is_empty(),
                "Tilted layer #{i} (branch B{}, {}/{}) at iso={:.3} must not have an empty mesh",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.isovalue
            );
            assert_eq!(
                hole_edges,
                0,
                "Tilted layer #{i} (branch B{}, {}/{}) at iso={:.3} has {hole_edges} spurious interior hole edges (bnd_edges={boundary_edges}, contour_segs={contour_segs})",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.isovalue
            );
        }
        ffi::set_spatial_orientation_hints(&[]);
        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        // 1. Every selected isosurface layer must have non-empty mesh triangles AND non-empty toolpaths,
        //    and no toolpath point may protrude beyond the physical object surface.
        let sdf_slice_eps = 0.15f32;
        let bmin = obj.local_min;
        let bmax = obj.local_max;
        for (i, (l, t)) in layers.layers.iter().zip(tp.layers.iter()).enumerate() {
            assert!(
                !l.indices.is_empty(),
                "Layer #{i} (branch {}, iso={}) has empty mesh!",
                l.branch_id,
                l.isovalue
            );
            assert!(
                !t.paths.is_empty(),
                "Layer #{i} (branch {}, iso={}, top={}) produced 0 toolpaths!",
                l.branch_id,
                l.isovalue,
                l.is_top_layer
            );
            for v in &l.vertices {
                assert!(
                    v.x >= bmin.x - sdf_slice_eps && v.x <= bmax.x + sdf_slice_eps &&
                    v.y >= bmin.y - sdf_slice_eps && v.y <= bmax.y + sdf_slice_eps &&
                    v.z >= bmin.z - sdf_slice_eps && v.z <= bmax.z + sdf_slice_eps,
                    "Layer #{i} (iso={:.3}) mesh vertex ({:.2}, {:.2}, {:.2}) protrudes outside object bbox [{bmin:?}..{bmax:?}]",
                    l.isovalue, v.x, v.y, v.z
                );
            }
            for p in &t.paths {
                for pt in &p.points {
                    assert!(
                        pt.x >= bmin.x - 0.02 && pt.x <= bmax.x + 0.02 &&
                        pt.y >= bmin.y - 0.02 && pt.y <= bmax.y + 0.02 &&
                        pt.z >= bmin.z - 0.05 && pt.z <= bmax.z + 0.05,
                        "Layer #{i} (iso={:.3}) toolpath point ({:.2}, {:.2}, {:.2}, kind={}) protrudes outside object bbox [{bmin:?}..{bmax:?}]",
                        l.isovalue, pt.x, pt.y, pt.z, p.feature_kind
                    );
                }
            }
        }

        // 2. Verify that the top layers on the single-span upper arch branches
        // (Y in [-6.5, -3.5] and Y in [17.5, 22.5]) have unbroken mesh coverage across all
        // 10 Y-bins (no missing middle sections) and toolpaths spanning their full Y extent.
        let mut checked_top_arch_layers = 0usize;
        for (i, (l, t)) in layers.layers.iter().zip(tp.layers.iter()).enumerate() {
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            let mut max_z = f32::MIN;
            for v in &l.vertices {
                min_y = min_y.min(v.y);
                max_y = max_y.max(v.y);
                max_z = max_z.max(v.z);
            }

            let is_b6_arch = min_y >= -6.7 && max_y <= -3.3 && max_z >= 13.5;
            let is_b7_arch = min_y >= 17.3 && max_y <= 22.7 && max_z >= 13.5;
            if is_b6_arch || is_b7_arch {
                checked_top_arch_layers += 1;
                let y_span = (max_y - min_y).max(0.1);
                let mut bin_counts = [0usize; 10];
                for tri in l.indices.chunks_exact(3) {
                    let v0 = &l.vertices[tri[0] as usize];
                    let v1 = &l.vertices[tri[1] as usize];
                    let v2 = &l.vertices[tri[2] as usize];
                    let cy = (v0.y + v1.y + v2.y) / 3.0;
                    let b = (((cy - min_y) / y_span) * 10.0).floor().clamp(0.0, 9.0) as usize;
                    bin_counts[b] += 1;
                }
                for (b, &cnt) in bin_counts.iter().enumerate() {
                    assert!(
                        cnt > 0,
                        "Layer #{i} (branch {}, iso={:.3}, Y=[{:.2}..{:.2}]) is missing triangles in interior Y-bin {b}: {:?}",
                        l.branch_id,
                        l.isovalue,
                        min_y,
                        max_y,
                        bin_counts
                    );
                }

                let mut tp_min_y = f32::MAX;
                let mut tp_max_y = f32::MIN;
                for p in &t.paths {
                    for pt in &p.points {
                        tp_min_y = tp_min_y.min(pt.y);
                        tp_max_y = tp_max_y.max(pt.y);
                    }
                }
                assert!(
                    (tp_max_y - tp_min_y) >= 0.75 * y_span,
                    "Layer #{i} (branch {}, iso={:.3}) toolpath Y-span [{:.2}..{:.2}] does not cover mesh Y-span [{:.2}..{:.2}]",
                    l.branch_id,
                    l.isovalue,
                    tp_min_y,
                    tp_max_y,
                    min_y,
                    max_y
                );
            }
        }
        assert!(
            checked_top_arch_layers >= 10,
            "Expected at least 10 top arch layers on B6/B7 above Z=13.5, checked {checked_top_arch_layers}"
        );

        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_non_planar_sparse_infill_patterns_and_endpoint_inset() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);

        // Register a 20x20x10 mm block on the bed (`[-10, 10] x [-10, 10] x [0, 10]`)
        // so the interior core (inside a 0.84 mm 2-wall shell) has diagonal > 25 mm (> 15 * 0.4 = 6.0 mm).
        let positions: [f32; 24] = [
            -10.0, -10.0,  0.0,
             10.0, -10.0,  0.0,
             10.0,  10.0,  0.0,
            -10.0,  10.0,  0.0,
            -10.0, -10.0, 10.0,
             10.0, -10.0, 10.0,
             10.0,  10.0, 10.0,
            -10.0,  10.0, 10.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_sparse_infill_block", &positions, &indices);
        assert!(part.object_id > 0);
        assert!(ffi::recompute_object_fields(part.object_id, 0.6, 4.0));

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 1.0,
            min_mm: 0.6,
            max_mm: 1.4,
            first_layer_mm: 0.5,
            top_layer_mm: 0.4,
        });

        let shell_thickness = 0.84f32; // 1 outer wall (0.42) + 1 inner wall (0.42)
        let sparse_w = 0.42f32;
        let mut spec = ffi::ShellPathSpec {
            shell_thickness_mm: shell_thickness,
            inner_wall_count: 1,
            outer_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.56,
            },
            inner_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.56,
            },
            solid_infill_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.56,
            },
            solid_infill_pattern: 0,
            min_extrusion_length_mm: 0.80,
            sparse_infill_density: 0.25,
            sparse_infill_width: ffi::LineWidthSpec {
                target_mm: sparse_w,
                min_mm: 0.30,
                max_mm: 0.60,
            },
            sparse_infill_pattern: 0,
            infill_angle_deg: 45.0,
            nozzle_diameter_mm: 0.40,
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 40.0,
            wave_overhang_overlap_fraction: 0.08,
            one_perimeter_on_conformal_patches: true,
            orient_outer_surfaces_to_normal: true,
            outer_surface_parallel_angle_deg: 15.0,
            brick_layer_mode: 0,
            enable_top_surface_lofting: true,
        };

        // Test all 7 sparse infill patterns (0..=6: Monotonic, Cubic, Gyroid, SchwarzD, SchwarzP, Concentric, AllWalls)
        for pattern in crate::config::SparseInfillPattern::ALL {
            spec.sparse_infill_pattern = pattern.as_u8();
            ffi::set_shell_path_spec(spec.clone());

            let res = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(res.valid, "Toolpath generation failed for {:?}", pattern);
            assert!(
                res.sparse_infill_count > 0,
                "Expected non-zero sparse_infill_count for pattern {:?}, got 0",
                pattern
            );

            // Inspect a mid-height layer (Z ~ 5.0 mm) to verify sparse infill endpoints
            // stay inside the shell + endpoint inset boundary and reach near the shell without off-by-one gaps
            let mid_layer = res
                .layers
                .iter()
                .find(|l| (l.isovalue - 5.0).abs() < 1.2)
                .expect("Expected mid-height layer near Z=5.0");

            let sparse_paths: Vec<_> = mid_layer
                .paths
                .iter()
                .filter(|p| p.feature_kind == 4)
                .collect();
            assert!(
                !sparse_paths.is_empty(),
                "Expected sparse infill paths on mid-height layer for {:?}",
                pattern
            );

            let mut min_dist_from_outer = f32::MAX;
            for path in &sparse_paths {
                for pt in &path.points {
                    let dist_xy = (10.0 - pt.x.abs()).min(10.0 - pt.y.abs());
                    min_dist_from_outer = min_dist_from_outer.min(dist_xy);
                    // No sparse centerline point may cross outside `shell_thickness - 0.5 * sparse_w`
                    assert!(
                        dist_xy >= shell_thickness - 0.25,
                        "Sparse infill point ({:.2}, {:.2}, {:.2}) for {:?} encroached into outer shell (dist_xy={:.3})",
                        pt.x,
                        pt.y,
                        pt.z,
                        pattern,
                        dist_xy
                    );
                }
            }

            // For Monotonic (0) and Concentric/AllWalls (5, 6), verify there is no off-by-one wall gap:
            // the closest sparse point to the shell must lie near `shell_thickness + 0.35 * sparse_w` (~0.99 mm),
            // well below `shell_thickness + 1.5 * sparse_w` (~1.47 mm).
            if matches!(
                pattern,
                crate::config::SparseInfillPattern::Monotonic
                    | crate::config::SparseInfillPattern::Concentric
                    | crate::config::SparseInfillPattern::AllWalls
            ) {
                assert!(
                    min_dist_from_outer <= shell_thickness + 0.85,
                    "Expected {:?} sparse infill to anchor adjacent to the shell without off-by-one gap, but min_dist_from_outer={:.3}",
                    pattern,
                    min_dist_from_outer
                );
            }
        }

        // Verify narrow-region promotion: increasing nozzle_diameter_mm so `15 * D_nozzle > island_diag`
        // promotes the interior core to solid infill (`sparse_infill_count == 0`, `solid_infill_count > 0`).
        spec.sparse_infill_pattern = 0;
        spec.nozzle_diameter_mm = 2.5; // 15 * 2.5 = 37.5 mm > 28 mm island diagonal
        ffi::set_shell_path_spec(spec);
        let promoted_res = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(promoted_res.valid);
        assert_eq!(
            promoted_res.sparse_infill_count, 0,
            "Expected narrow island (< 15 * D_nozzle) to be promoted from sparse to solid infill"
        );
        assert!(
            promoted_res.solid_infill_count > 0,
            "Expected promoted narrow island to generate solid infill paths"
        );

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_high_contrast_parameter_ramp_distinct_stops() {
        let stops = [0.0_f32, 0.25, 0.50, 0.75, 1.0];
        let colors: Vec<[f32; 4]> = stops
            .iter()
            .map(|&t| high_contrast_parameter_ramp(t))
            .collect();
        for c in &colors {
            assert!((0.0..=1.0).contains(&c[0]));
            assert!((0.0..=1.0).contains(&c[1]));
            assert!((0.0..=1.0).contains(&c[2]));
            assert!((c[3] - 1.0).abs() < 1.0e-5);
        }
        // Adjacent 25% stops must have strong Euclidean RGB separation (> 0.45)
        for pair in colors.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let dist = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                .sqrt();
            assert!(
                dist > 0.45,
                "Expected high RGB contrast between adjacent 25% stops ({a:?} vs {b:?}), got dist={dist:.3}"
            );
        }
    }

    #[test]
    fn test_dual_grid_multi_resolution_large_volume_voxel_reduction_and_slicing() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);

        // Register a large 80 x 80 x 60 mm solid block (384,000 mm^3 volume).
        // At 0.50 mm fine voxel size, a uniform interior grid requires >3,000,000 voxels,
        // whereas Dual-Grid Multi-Resolution (4x coarse core + fine boundary shell) uses
        // a small fraction of the active voxels while keeping fine 0.50 mm boundary resolution.
        let positions: [f32; 24] = [
            -40.0, -40.0, 0.0,
             40.0, -40.0, 0.0,
             40.0,  40.0, 0.0,
            -40.0,  40.0, 0.0,
            -40.0, -40.0, 60.0,
             40.0, -40.0, 60.0,
             40.0,  40.0, 60.0,
            -40.0,  40.0, 60.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_large_multires_block", &positions, &indices);
        assert!(part.object_id > 0);

        // 1. Single-resolution uniform fine grid baseline (0.50 mm voxel size)
        ffi::set_multi_resolution_grid_config(false, 4);
        assert!(ffi::recompute_object_fields(part.object_id, 0.50, 4.0));
        let single_res_stats = ffi::get_object_field_stats(part.object_id, "sdf");
        assert!(single_res_stats.valid);

        // 2. Dual-Grid Multi-Resolution (4x coarse interior + 0.50 mm fine boundary shell)
        ffi::set_multi_resolution_grid_config(true, 4);
        assert!(ffi::recompute_object_fields(part.object_id, 0.50, 4.0));
        let dual_res_stats = ffi::get_object_field_stats(part.object_id, "sdf");
        assert!(dual_res_stats.valid);
        assert!(
            (dual_res_stats.voxel_size - 0.50).abs() < 1.0e-4,
            "Expected fine boundary shell voxel size 0.50 mm in Dual-Grid mode, got {}",
            dual_res_stats.voxel_size
        );
        assert!(
            dual_res_stats.active_voxel_count * 2 < single_res_stats.active_voxel_count,
            "Expected Dual-Grid active_voxel_count ({}) to be < 50% of uniform single-grid ({})",
            dual_res_stats.active_voxel_count,
            single_res_stats.active_voxel_count
        );
        assert!(
            dual_res_stats.min_value < -20.0,
            "Expected Dual-Grid SDF stats to include deep interior core minimum (<-20 mm), got {}",
            dual_res_stats.min_value
        );

        // 3. Verify 2D cross-section slice sampling through the deep core is continuous and valid
        let mid_slice = ffi::sample_field_slice(part.object_id, "eikonal", 2, 0.5, 48);
        assert!(mid_slice.valid);
        let center_idx = (24 * 48 + 24) as usize;
        assert!(
            mid_slice.sdf_values[center_idx] < -15.0,
            "Expected deep interior core SDF < -15 mm at slice center, got {}",
            mid_slice.sdf_values[center_idx]
        );
        assert!(
            mid_slice.values[center_idx] > 10.0 && mid_slice.values[center_idx] < 50.0,
            "Expected valid solved Eikonal tau at deep core slice center, got {}",
            mid_slice.values[center_idx]
        );

        // 4. Verify non-planar layer selection and conformal shell-refined mesh extraction
        //    on the Dual-Grid object (use 6.0 mm layer thickness for fast test execution)
        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 6.0,
            min_mm: 4.0,
            max_mm: 8.0,
            first_layer_mm: 4.0,
            top_layer_mm: 4.0,
        });
        let selected = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(selected.valid);
        assert!(selected.layers.len() >= 8);

        for layer in &selected.layers {
            assert!(
                !layer.indices.is_empty() && !layer.contour_segments.is_empty(),
                "Layer {} (iso={}) in Dual-Grid mode had empty mesh or contour",
                layer.layer_index,
                layer.isovalue
            );
        }

        // 5. Verify shell & sparse infill toolpath generation succeeds on the Dual-Grid layers
        //    and that outer walls stay strictly inside the 80x80x60 mm block surface ([-40, 40] x [-40, 40])
        let cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_multi_resolution_grid_config(true, 4);
        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 6.0,
            min_mm: 4.0,
            max_mm: 8.0,
            first_layer_mm: 4.0,
            top_layer_mm: 4.0,
        });
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid);
        assert!(tp.outer_wall_count > 0, "Expected outer walls in Dual-Grid mode");
        assert!(tp.inner_wall_count > 0, "Expected inner walls in Dual-Grid mode");
        assert!(tp.sparse_infill_count > 0, "Expected sparse infill in Dual-Grid deep core");
        for l in &tp.layers {
            for p in &l.paths {
                for pt in &p.points {
                    assert!(
                        pt.x.abs() <= 40.02 && pt.y.abs() <= 40.02,
                        "Dual-Grid layer (iso={:.2}) toolpath point ({:.3}, {:.3}, {:.3}, kind={}) protrudes beyond [-40, 40] block surface!",
                        l.isovalue, pt.x, pt.y, pt.z, p.feature_kind
                    );
                }
            }
        }

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_thin_walled_curved_shell_complete_isosurfaces_with_default_auto_voxel() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);
        ffi::set_multi_resolution_grid_config(true, 4);

        // Build a large (~56 x 56 x 24 mm) watertight doubly-curved wavy thin-walled shell
        // with 0.90 mm radial wall thickness. Previously, both the 0.50 mm default voxel size
        // and the bounding-box Dual-Grid heuristic caused parts of many isosurfaces to go missing.
        let n_theta = 72usize;
        let n_z = 24usize;
        let height_z = 24.0f32;
        let wall_thickness = 0.90f32;

        let mut positions = Vec::with_capacity((n_z + 1) * n_theta * 2 * 3);
        let mut indices = Vec::new();

        for iz in 0..=n_z {
            let v = iz as f32 / n_z as f32;
            let z = v * height_z;
            let z_wave = 3.2 * (v * std::f32::consts::TAU).sin();
            for it in 0..n_theta {
                let theta = (it as f32 / n_theta as f32) * std::f32::consts::TAU;
                let r_mid = 22.0 + 3.5 * (3.0 * theta + 1.5 * v * std::f32::consts::PI).sin() + z_wave;
                let r_in = r_mid - 0.5 * wall_thickness;
                let r_out = r_mid + 0.5 * wall_thickness;
                let (s, c) = theta.sin_cos();
                // Outer ring vertex (even index)
                positions.extend_from_slice(&[r_out * c, r_out * s, z]);
                // Inner ring vertex (odd index)
                positions.extend_from_slice(&[r_in * c, r_in * s, z]);
            }
        }

        let ring_stride = (n_theta * 2) as u32;
        for iz in 0..n_z {
            let row0 = (iz as u32) * ring_stride;
            let row1 = ((iz + 1) as u32) * ring_stride;
            for it in 0..n_theta {
                let next_it = (it + 1) % n_theta;
                let o00 = row0 + (it as u32) * 2;
                let i00 = o00 + 1;
                let o01 = row0 + (next_it as u32) * 2;
                let i01 = o01 + 1;

                let o10 = row1 + (it as u32) * 2;
                let i10 = o10 + 1;
                let o11 = row1 + (next_it as u32) * 2;
                let i11 = o11 + 1;

                // Outer wall (outward normal)
                indices.extend_from_slice(&[o00, o01, o11, o00, o11, o10]);
                // Inner wall (inward normal)
                indices.extend_from_slice(&[i00, i11, i01, i00, i10, i11]);
            }
        }
        // Bottom cap (z = 0, downward normal) and top cap (z = height_z, upward normal)
        let top_row = (n_z as u32) * ring_stride;
        for it in 0..n_theta {
            let next_it = (it + 1) % n_theta;
            let o_b0 = (it as u32) * 2;
            let i_b0 = o_b0 + 1;
            let o_b1 = (next_it as u32) * 2;
            let i_b1 = o_b1 + 1;
            indices.extend_from_slice(&[o_b0, i_b1, o_b1, o_b0, i_b0, i_b1]);

            let o_t0 = top_row + (it as u32) * 2;
            let i_t0 = o_t0 + 1;
            let o_t1 = top_row + (next_it as u32) * 2;
            let i_t1 = o_t1 + 1;
            indices.extend_from_slice(&[o_t0, o_t1, i_t1, o_t0, i_t1, i_t0]);
        }

        let cfg = crate::config::SlicerConfig::default();
        assert!((cfg.eikonal.voxel_size_mm - 0.25).abs() < 1.0e-5);

        let part = ffi::register_triangle_mesh("test_thin_curved_shell", &positions, &indices);
        assert!(part.object_id > 0);
        assert!(ffi::recompute_object_fields(
            part.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 2.0,
            min_mm: 1.2,
            max_mm: 2.8,
            first_layer_mm: 1.5,
            top_layer_mm: 1.5,
        });

        let selected = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(selected.valid);
        assert!(
            selected.layers.len() >= 10,
            "Expected at least 10 layers on thin-walled curved shell, got {}",
            selected.layers.len()
        );

        // Verify that every layer has non-empty mesh & contours, and that all layers spanning
        // the body of the wavy shell (`iso <= 22.5 mm`, before the top flat rim where the 3 wave
        // lobes terminate at slightly different Eikonal arc lengths) have unbroken 360-degree
        // coverage across all 16 angular sectors.
        let mut full_ring_layers_checked = 0usize;
        for (idx, layer) in selected.layers.iter().enumerate() {
            assert!(
                !layer.indices.is_empty() && !layer.contour_segments.is_empty(),
                "Layer #{idx} (iso={:.3}) on thin curved shell has empty mesh or contour!",
                layer.isovalue
            );
            if layer.isovalue <= 22.5 {
                full_ring_layers_checked += 1;
                let mut sector_tri_counts = [0usize; 16];
                for tri in layer.indices.chunks_exact(3) {
                    let v0 = &layer.vertices[tri[0] as usize];
                    let v1 = &layer.vertices[tri[1] as usize];
                    let v2 = &layer.vertices[tri[2] as usize];
                    let cx = (v0.x + v1.x + v2.x) / 3.0;
                    let cy = (v0.y + v1.y + v2.y) / 3.0;
                    let angle = cy.atan2(cx).rem_euclid(std::f32::consts::TAU);
                    let sector = ((angle / std::f32::consts::TAU) * 16.0).floor().clamp(0.0, 15.0) as usize;
                    sector_tri_counts[sector] += 1;
                }
                for (s, &cnt) in sector_tri_counts.iter().enumerate() {
                    assert!(
                        cnt > 0,
                        "Thin-walled curved shell layer #{idx} (iso={:.3}) is missing isosurface section in angular sector {s}/16: {:?}",
                        layer.isovalue,
                        sector_tri_counts
                    );
                }
            }
        }
        assert!(
            full_ring_layers_checked >= 10,
            "Expected at least 10 full-ring body layers on thin-walled curved shell, got {full_ring_layers_checked}"
        );

        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: cfg.slicing.layer_height.target_mm,
            min_mm: cfg.slicing.layer_height.min_mm,
            max_mm: cfg.slicing.layer_height.max_mm,
            first_layer_mm: cfg.slicing.layer_height.first_layer_mm,
            top_layer_mm: cfg.slicing.layer_height.top_layer_mm,
        });
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_a4t_cowling_step_sdf_and_slicing() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let step_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/A4T Cowling - Dragon-Rapido UHF [xol-carriage].step");
        if !step_path.exists() {
            return;
        }
        let cfg = crate::config::SlicerConfig::default();
        ffi::set_spatial_orientation_hints(&[]);
        ffi::set_multi_resolution_grid_config(true, 4);
        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: cfg.slicing.layer_height.target_mm,
            min_mm: cfg.slicing.layer_height.min_mm,
            max_mm: cfg.slicing.layer_height.max_mm,
            first_layer_mm: cfg.slicing.layer_height.first_layer_mm,
            top_layer_mm: cfg.slicing.layer_height.top_layer_mm,
        });
        let parts = ffi::load_step_assembly(step_path.to_str().unwrap());
        println!("Loaded {} parts from STEP file", parts.len());
        assert!(!parts.is_empty(), "Expected at least one part from STEP file");

        for part in &parts {
            let verts = ffi::get_object_vertices(part.object_id);
            let indices = ffi::get_object_indices(part.object_id);
            let mut min_pt = [f32::INFINITY; 3];
            let mut max_pt = [f32::NEG_INFINITY; 3];
            for v in &verts {
                min_pt[0] = min_pt[0].min(v.x);
                min_pt[1] = min_pt[1].min(v.y);
                min_pt[2] = min_pt[2].min(v.z);
                max_pt[0] = max_pt[0].max(v.x);
                max_pt[1] = max_pt[1].max(v.y);
                max_pt[2] = max_pt[2].max(v.z);
            }
            println!(
                "  Bounds: X=[{:.2}, {:.2}] ({:.2}), Y=[{:.2}, {:.2}] ({:.2}), Z=[{:.2}, {:.2}] ({:.2})",
                min_pt[0], max_pt[0], max_pt[0] - min_pt[0],
                min_pt[1], max_pt[1], max_pt[1] - min_pt[1],
                min_pt[2], max_pt[2], max_pt[2] - min_pt[2],
            );

            // Check edge manifoldness before and after spatial vertex welding
            let mut unwelded_edges: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
            for tri in indices.chunks_exact(3) {
                for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                    let key = (a.min(b), a.max(b));
                    *unwelded_edges.entry(key).or_insert(0) += 1;
                }
            }
            let unwelded_open = unwelded_edges.values().filter(|&&c| c == 1).count();
            println!(
                "  Unwelded edges: total={} open(c==1)={} manifold(c==2)={}",
                unwelded_edges.len(),
                unwelded_open,
                unwelded_edges.values().filter(|&&c| c == 2).count()
            );

            let mut weld_map: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::new();
            let mut welded_idx = Vec::with_capacity(verts.len());
            let mut unique_count = 0u32;
            for v in &verts {
                let key = (
                    (v.x as f64 * 1000.0).round() as i64,
                    (v.y as f64 * 1000.0).round() as i64,
                    (v.z as f64 * 1000.0).round() as i64,
                );
                let id = *weld_map.entry(key).or_insert_with(|| {
                    let id = unique_count;
                    unique_count += 1;
                    id
                });
                welded_idx.push(id);
            }
            let mut welded_edges: std::collections::HashMap<(u32, u32), Vec<bool>> = std::collections::HashMap::new();
            for tri in indices.chunks_exact(3) {
                let w0 = welded_idx[tri[0] as usize];
                let w1 = welded_idx[tri[1] as usize];
                let w2 = welded_idx[tri[2] as usize];
                if w0 == w1 || w1 == w2 || w2 == w0 {
                    continue;
                }
                for &(a, b) in &[(w0, w1), (w1, w2), (w2, w0)] {
                    let key = (a.min(b), a.max(b));
                    welded_edges.entry(key).or_default().push(a < b);
                }
            }
            let welded_open = welded_edges.values().filter(|v| v.len() == 1).count();
            let welded_manifold_consistent = welded_edges
                .values()
                .filter(|v| v.len() == 2 && v[0] != v[1])
                .count();
            let welded_manifold_flipped = welded_edges
                .values()
                .filter(|v| v.len() == 2 && v[0] == v[1])
                .count();
            let welded_nonmanifold = welded_edges.values().filter(|v| v.len() > 2).count();
            println!(
                "  Welded (1um): unique_verts={} total_edges={} open={} consistent_2={} flipped_2={} nonmanifold(>2)={}",
                unique_count,
                welded_edges.len(),
                welded_open,
                welded_manifold_consistent,
                welded_manifold_flipped,
                welded_nonmanifold
            );
            assert_eq!(
                welded_open, 0,
                "Expected 0 open boundary edges after STEP tessellation hole-closing, got {welded_open}"
            );

            let mut signed_vol6 = 0.0f64;
            for tri in indices.chunks_exact(3) {
                let v0 = &verts[tri[0] as usize];
                let v1 = &verts[tri[1] as usize];
                let v2 = &verts[tri[2] as usize];
                let p0 = [v0.x as f64, v0.y as f64, v0.z as f64];
                let p1 = [v1.x as f64, v1.y as f64, v1.z as f64];
                let p2 = [v2.x as f64, v2.y as f64, v2.z as f64];
                let cx = p1[1] * p2[2] - p1[2] * p2[1];
                let cy = p1[2] * p2[0] - p1[0] * p2[2];
                let cz = p1[0] * p2[1] - p1[1] * p2[0];
                signed_vol6 += p0[0] * cx + p0[1] * cy + p0[2] * cz;
            }
            println!(
                "Part id={} name={} verts={} tris={} signed_volume={:.3} mm^3",
                part.object_id,
                part.name,
                verts.len(),
                indices.len() / 3,
                signed_vol6 / 6.0
            );
            assert!(signed_vol6 > 0.0, "Expected positive outward signed volume");

            let cfg = crate::config::SlicerConfig::default();
            assert!(ffi::recompute_object_fields(
                part.object_id,
                cfg.eikonal.voxel_size_mm,
                cfg.eikonal.narrow_band_width_voxels as f32
            ));

            let sdf_stats = ffi::get_object_field_stats(part.object_id, "sdf");
            let eik_stats = ffi::get_object_field_stats(part.object_id, "eikonal");
            let iso0 = ffi::extract_field_isosurface(part.object_id, "sdf", 0.0, 0.0);
            let iso1 = ffi::extract_field_isosurface(part.object_id, "sdf", 1.0, 0.0);
            println!(
                "  SDF stats: valid={} min={:.4} max={:.4} active_voxels={}",
                sdf_stats.valid,
                sdf_stats.min_value,
                sdf_stats.max_value,
                sdf_stats.active_voxel_count,
            );
            println!(
                "  Eikonal stats: valid={} min={:.4} max={:.4} active_voxels={}",
                eik_stats.valid,
                eik_stats.min_value,
                eik_stats.max_value,
                eik_stats.active_voxel_count
            );
            println!(
                "  SDF iso=0.0 tris={}, iso=1.0 tris={}",
                iso0.indices.len() / 3,
                iso1.indices.len() / 3
            );
            assert!(
                sdf_stats.valid && sdf_stats.min_value < -4.0,
                "Expected deep negative interior SDF (< -4.0 mm) on A4T Cowling STEP, got min={:.4}",
                sdf_stats.min_value
            );
            assert!(
                iso0.indices.len() / 3 > 500_000,
                "Expected complete 0.0 SDF isosurface (>500k tris), got {}",
                iso0.indices.len() / 3
            );

            let selected = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
            let non_empty_layers = selected
                .layers
                .iter()
                .filter(|l| !l.contour_segments.is_empty())
                .count();
            let toolpaths = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            println!(
                "  Selected layers: total={} non_empty={}, toolpath total_paths={} outer={} inner={} infill={}",
                selected.layers.len(),
                non_empty_layers,
                toolpaths.total_path_count,
                toolpaths.outer_wall_count,
                toolpaths.inner_wall_count,
                toolpaths.solid_infill_count
            );
            assert!(
                toolpaths.valid
                    && toolpaths.outer_wall_count > 1000
                    && toolpaths.inner_wall_count > 1000
                    && toolpaths.solid_infill_count > 1000,
                "Expected full outer wall, inner wall, and infill toolpaths on A4T Cowling STEP, got outer={} inner={} infill={}",
                toolpaths.outer_wall_count,
                toolpaths.inner_wall_count,
                toolpaths.solid_infill_count
            );
            let _ = ffi::remove_cad_object(part.object_id);
        }
    }

    #[test]
    fn test_object_surface_coloring_overhang_and_tool_angle() {
        let _guard = TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let test_verts = vec![
            // 0: Upward face (+Z)
            ffi::RenderVertex {
                x: 0.0,
                y: 0.0,
                z: 10.0,
                nx: 0.0,
                ny: 0.0,
                nz: 1.0,
            },
            // 1: Right face (+X)
            ffi::RenderVertex {
                x: 10.0,
                y: 0.0,
                z: 0.0,
                nx: 1.0,
                ny: 0.0,
                nz: 0.0,
            },
            // 2: Downward face (-Z)
            ffi::RenderVertex {
                x: 0.0,
                y: 0.0,
                z: -10.0,
                nx: 0.0,
                ny: 0.0,
                nz: -1.0,
            },
        ];

        let mut cfg = SlicerConfig::default();
        // 1. Default gravity (0, 0, -1) and identity transform:
        //    v0 (+Z) -> 0 deg overhang, v1 (+X) -> 90 deg, v2 (-Z) -> 180 deg.
        let (colors_ovh, min_ovh, max_ovh, mean_ovh, uses_solved) =
            compute_object_surface_vertex_colors(
                &test_verts,
                &Transform::IDENTITY,
                &cfg,
                ObjectSurfaceColorMode::Overhang,
                &[],
            );
        assert!(!uses_solved);
        assert!((min_ovh - 0.0).abs() < 1.0e-3);
        assert!((max_ovh - 180.0).abs() < 1.0e-3);
        assert!((mean_ovh - 90.0).abs() < 1.0e-3);
        assert_eq!(colors_ovh[0], srgba_to_linear_rgba(high_contrast_parameter_ramp(0.0)));
        assert_eq!(colors_ovh[1], srgba_to_linear_rgba(high_contrast_parameter_ramp(0.5)));
        assert_eq!(colors_ovh[2], srgba_to_linear_rgba(high_contrast_parameter_ramp(1.0)));

        // 2. Tilted gravity in bed coordinates (-X, 0, 0) via query_gravity_direction:
        //    v1 (+X) points directly opposite to gravity (-X), so v1 has 0 deg overhang!
        cfg.machine.gravity_direction_bed = [-1.0, 0.0, 0.0];
        let (colors_tilted_g, _, _, _, _) = compute_object_surface_vertex_colors(
            &test_verts,
            &Transform::IDENTITY,
            &cfg,
            ObjectSurfaceColorMode::Overhang,
            &[],
        );
        assert_eq!(
            colors_tilted_g[1],
            srgba_to_linear_rgba(high_contrast_parameter_ramp(0.0)),
            "+X normal must have 0° overhang when gravity is (-1, 0, 0)"
        );
        assert_eq!(
            colors_tilted_g[0],
            srgba_to_linear_rgba(high_contrast_parameter_ramp(0.5)),
            "+Z normal must have 90° overhang when gravity is (-1, 0, 0)"
        );

        // 3. ToolAngleFromNormal with tilted nozzle axis (1, 0, 0) via query_nozzle_frame:
        cfg.machine.nozzle_axis_bed = [1.0, 0.0, 0.0];
        let (colors_tool_nom, min_tn, max_tn, _, uses_solved_nom) =
            compute_object_surface_vertex_colors(
                &test_verts,
                &Transform::IDENTITY,
                &cfg,
                ObjectSurfaceColorMode::ToolAngleFromNormal,
                &[],
            );
        assert!(!uses_solved_nom);
        assert!((min_tn - 0.0).abs() < 1.0e-3);
        assert!((max_tn - 90.0).abs() < 1.0e-3);
        assert_eq!(
            colors_tool_nom[1],
            srgba_to_linear_rgba(high_contrast_parameter_ramp(0.0)),
            "+X normal aligned with +X nozzle axis must give 0° tool angle from normal"
        );

        // 4. Solved 3D toolAxisGrid via FFI (`get_object_vertex_tool_axes`) on a solid mesh:
        let positions: [f32; 24] = [
            -6.0, -6.0, -6.0,
             6.0, -6.0, -6.0,
             6.0,  6.0, -6.0,
            -6.0,  6.0, -6.0,
            -6.0, -6.0,  6.0,
             6.0, -6.0,  6.0,
             6.0,  6.0,  6.0,
            -6.0,  6.0,  6.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];
        let part = ffi::register_triangle_mesh("surface_color_box", &positions, &indices);
        assert!(part.object_id > 0);
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);
        assert!(ffi::recompute_object_fields(part.object_id, 0.8, 5.0));

        let obj_verts = ffi::get_object_vertices(part.object_id);
        let solved_axes = ffi::get_object_vertex_tool_axes(part.object_id);
        assert_eq!(
            solved_axes.len(),
            obj_verts.len() * 3,
            "Expected 3 solved tool axis components per object surface vertex"
        );
        let (colors_solved, min_s, max_s, _, uses_solved_field) =
            compute_object_surface_vertex_colors(
                &obj_verts,
                &Transform::IDENTITY,
                &SlicerConfig::default(),
                ObjectSurfaceColorMode::ToolAngleFromNormal,
                &solved_axes,
            );
        assert!(uses_solved_field);
        assert_eq!(colors_solved.len(), obj_verts.len());
        assert!(min_s >= 0.0 && max_s <= 180.0 && max_s > min_s);

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_bridge_and_wave_overhang_generation_from_overhang_field() {
        let _guard = TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Construct a watertight 3D bridge-arch + cantilever mesh:
        // - Left pillar:  X in [-10.0, -4.0], Y in [-4.0, 4.0], Z in [0.0, 6.0]
        // - Right pillar: X in [  4.0, 10.0], Y in [-4.0, 4.0], Z in [0.0, 6.0]
        // - Top bridge deck connecting both pillars AND cantilevered out to Y = +8.0:
        //   X in [-10.0, 10.0], Y in [-4.0, 8.0], Z in [4.0, 6.0]
        // We union three watertight boxes by registering their triangles via boolean-free
        // exact grid subdivision or separate overlapping boxes (OpenVDB meshToVolume handles
        // self-intersecting / overlapping closed shells cleanly via winding/unsigned distance,
        // or we can build the exact 2D profile extruded along Y and X).
        // Let's build a clean watertight mesh by subdividing the 3D grid of cubes:
        // X intervals: [-10, -4], [-4, 4], [4, 10]
        // Y intervals: [-4, 4], [4, 8]
        // Z intervals: [0, 4], [4, 6]
        let x_coords = [-10.0f32, -4.0, 4.0, 10.0];
        let y_coords = [-4.0f32, 4.0, 8.0];
        let z_coords = [0.0f32, 4.0, 6.0];
        // Active cells (ix, iy, iz):
        // - Left pillar bottom:  (0, 0, 0)
        // - Right pillar bottom: (2, 0, 0)
        // - Top deck over pillars & bridge span: (0, 0, 1), (1, 0, 1), (2, 0, 1)
        // - Top deck cantilevered in +Y:         (0, 1, 1), (1, 1, 1), (2, 1, 1)
        let is_occupied = |ix: i32, iy: i32, iz: i32| -> bool {
            if ix < 0 || ix >= 3 || iy < 0 || iy >= 2 || iz < 0 || iz >= 2 {
                return false;
            }
            match (ix, iy, iz) {
                (0, 0, 0) | (2, 0, 0) => true,
                (_, _, 1) => true,
                _ => false,
            }
        };

        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut add_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
            let base = (positions.len() / 3) as u32;
            for p in [p0, p1, p2, p3] {
                positions.extend_from_slice(&p);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };

        for ix in 0..3 {
            for iy in 0..2 {
                for iz in 0..2 {
                    if !is_occupied(ix, iy, iz) {
                        continue;
                    }
                    let x0 = x_coords[ix as usize];
                    let x1 = x_coords[(ix + 1) as usize];
                    let y0 = y_coords[iy as usize];
                    let y1 = y_coords[(iy + 1) as usize];
                    let z0 = z_coords[iz as usize];
                    let z1 = z_coords[(iz + 1) as usize];

                    // -X face
                    if !is_occupied(ix - 1, iy, iz) {
                        add_quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]);
                    }
                    // +X face
                    if !is_occupied(ix + 1, iy, iz) {
                        add_quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]);
                    }
                    // -Y face
                    if !is_occupied(ix, iy - 1, iz) {
                        add_quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]);
                    }
                    // +Y face
                    if !is_occupied(ix, iy + 1, iz) {
                        add_quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]);
                    }
                    // -Z face
                    if !is_occupied(ix, iy, iz - 1) {
                        add_quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]);
                    }
                    // +Z face
                    if !is_occupied(ix, iy, iz + 1) {
                        add_quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]);
                    }
                }
            }
        }

        let part = ffi::register_triangle_mesh("bridge_and_wave_arch", &positions, &indices);
        assert!(part.object_id > 0);

        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);
        // Align layers horizontally with the bridge ceiling Z = 4.0 so the first bridge
        // deck layer spans the full 8.0 mm gap between the left pillar (X=-4) and right pillar (X=+4).
        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 2.0, 4.5],
            direction_bed: [0.0, 0.0, 1.0],
            influence_radius_mm: 30.0,
            weight: 1.0,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        ffi::set_slice_layer_height(ffi::LayerHeightSpec {
            target_mm: 0.40,
            min_mm: 0.20,
            max_mm: 0.50,
            first_layer_mm: 0.40,
            top_layer_mm: 0.40,
        });
        assert!(ffi::recompute_object_fields(part.object_id, 0.45, 5.0));

        // Case A: max_bridge_length_mm = 40.0 mm (spans the 8.0 mm gap between X=-4 and X=+4 with Bridges,
        // while the one-sided +Y cantilever uses Wave Overhangs).
        ffi::set_shell_path_spec(ffi::ShellPathSpec {
            shell_thickness_mm: 1.6,
            inner_wall_count: 3,
            outer_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.60,
            },
            inner_wall_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.65,
            },
            solid_infill_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.65,
            },
            solid_infill_pattern: 0,
            min_extrusion_length_mm: 1.0,
            sparse_infill_density: 0.20,
            sparse_infill_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.32,
                max_mm: 0.68,
            },
            sparse_infill_pattern: 0,
            infill_angle_deg: 45.0,
            nozzle_diameter_mm: 0.40,
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 40.0,
            wave_overhang_overlap_fraction: 0.08,
            one_perimeter_on_conformal_patches: true,
            orient_outer_surfaces_to_normal: true,
            outer_surface_parallel_angle_deg: 15.0,
            brick_layer_mode: 0,
            enable_top_surface_lofting: true,
        });

        let tp_bridge = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_bridge.valid);
        assert!(
            tp_bridge.bridge_count > 0,
            "Expected straight circular Bridge extrusions across the 8mm two-pillar span when max_bridge_length_mm = 40.0"
        );
        assert!(
            tp_bridge.wave_overhang_count > 0,
            "Expected inside-out WaveOverhang extrusions on the cantilevered overhang region"
        );

        // Verify every Bridge path is a 2-point straight line with circular cross-section (width == height == 0.40mm)
        for layer in &tp_bridge.layers {
            for p in &layer.paths {
                if p.feature_kind == 5 {
                    assert_eq!(
                        p.points.len(),
                        2,
                        "Bridge path must be a 2-point straight-line extrusion"
                    );
                    for pt in &p.points {
                        assert!(
                            (pt.width_mm - 0.40).abs() < 1.0e-3
                                && (pt.height_mm - 0.40).abs() < 1.0e-3,
                            "Bridge point must have circular cross-section (0.40 x 0.40 mm), got ({}, {})",
                            pt.width_mm,
                            pt.height_mm
                        );
                    }
                }
            }
        }

        // Case B: Set max_bridge_length_mm = 2.0 mm (shorter than the 8.0 mm gap).
        // Now no bridges may exceed 2.0 mm, and the unsupported span falls back to Wave Overhangs!
        ffi::set_shell_path_spec(ffi::ShellPathSpec {
            shell_thickness_mm: 1.6,
            inner_wall_count: 3,
            outer_wall_width: ffi::LineWidthSpec {
                target_mm: 0.42,
                min_mm: 0.30,
                max_mm: 0.60,
            },
            inner_wall_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.65,
            },
            solid_infill_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.30,
                max_mm: 0.65,
            },
            solid_infill_pattern: 0,
            min_extrusion_length_mm: 1.0,
            sparse_infill_density: 0.20,
            sparse_infill_width: ffi::LineWidthSpec {
                target_mm: 0.45,
                min_mm: 0.32,
                max_mm: 0.68,
            },
            sparse_infill_pattern: 0,
            infill_angle_deg: 45.0,
            nozzle_diameter_mm: 0.40,
            overhang_angle_threshold_deg: 45.0,
            max_bridge_length_mm: 2.0,
            wave_overhang_overlap_fraction: 0.08,
            one_perimeter_on_conformal_patches: true,
            orient_outer_surfaces_to_normal: true,
            outer_surface_parallel_angle_deg: 15.0,
            brick_layer_mode: 0,
            enable_top_surface_lofting: true,
        });

        let tp_wave_only = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_wave_only.valid);
        assert_eq!(
            tp_wave_only.bridge_count, 0,
            "Expected 0 bridges across the 8.0 mm gap when max_bridge_length_mm is reduced to 2.0 mm"
        );
        let wave_len_with_bridges: f32 = tp_bridge
            .layers
            .iter()
            .flat_map(|l| l.paths.iter())
            .filter(|p| p.feature_kind == 6)
            .map(|p| p.total_length_mm)
            .sum();
        let wave_len_without_bridges: f32 = tp_wave_only
            .layers
            .iter()
            .flat_map(|l| l.paths.iter())
            .filter(|p| p.feature_kind == 6)
            .map(|p| p.total_length_mm)
            .sum();
        assert!(
            wave_len_without_bridges > wave_len_with_bridges + 20.0,
            "Expected significantly more WaveOverhang extrusion length ({wave_len_without_bridges:.1} mm) when max_bridge_length_mm is 2.0 mm vs 40.0 mm ({wave_len_with_bridges:.1} mm)"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_overhang_field_conformal_and_orthogonal_surface_bias() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::ui::sync_config_to_backend(&crate::config::SlicerConfig::default());
        ffi::set_spatial_orientation_hints(&[]);

        // 1. Verify default configuration & FFI round-trip
        let default_cfg = crate::config::EikonalFieldConfig::default();
        assert!(default_cfg.enable_overhang_orientation_bias);
        assert!((default_cfg.orthogonal_supported_angle_deg - 30.0).abs() < 1.0e-5);
        assert!((default_cfg.orthogonal_overhang_angle_deg - 60.0).abs() < 1.0e-5);

        let custom_spec = ffi::OverhangBiasSpec {
            enabled: true,
            weight: 0.85,
            orthogonal_supported_angle_deg: 30.0,
            orthogonal_overhang_angle_deg: 60.0,
            transition_width_deg: 10.0,
        };
        ffi::set_overhang_orientation_bias_spec(custom_spec);
        let readback = ffi::get_overhang_orientation_bias_spec();
        assert!(readback.enabled);
        assert!((readback.weight - 0.85).abs() < 1.0e-5);
        assert!((readback.orthogonal_supported_angle_deg - 30.0).abs() < 1.0e-5);
        assert!((readback.orthogonal_overhang_angle_deg - 60.0).abs() < 1.0e-5);
        assert!((readback.transition_width_deg - 10.0).abs() < 1.0e-5);

        // Configure a 5/6-axis tool with generous clearance so we can observe the surface bias
        // across all overhang regimes, while still enforcing a hard 45 deg max tool tilt.
        let tools = vec![ffi::ToolSpec {
            id: 0,
            name: "OverhangBiasTestTool".to_string(),
            nozzle_diameter_mm: 0.4,
            temperature_celsius: 215.0,
            has_rotational_axes: true,
            max_angle_from_bed_normal_deg: 45.0,
            clearance_profile_xz: vec![
                ffi::ClearanceXZ { x_radius_mm: 0.6, z_height_mm: 0.0 },
                ffi::ClearanceXZ { x_radius_mm: 12.0, z_height_mm: 6.0 },
            ],
            ..Default::default()
        }];
        ffi::set_machine_tools(&tools, 0);
        ffi::set_eikonal_solver_iterations(3);

        // Helper that builds a prism spanning Y in [-8, 8], Z in [0, 12], with a vertical left
        // wall at X = -10 and a sloped +X wall from x_bot at Z = 0 to x_top at Z = 12, then
        // measures the average X-component of the `eikonal` field isosurface normal near the +X wall.
        let measure_right_wall_iso_nx =
            |x_bot: f32, x_top: f32, enable_bias: bool, supp_deg: f32, ovh_deg: f32| -> f32 {
                let positions: [f32; 24] = [
                    -10.0, -8.0,  0.0,
                    x_bot, -8.0,  0.0,
                    x_bot,  8.0,  0.0,
                    -10.0,  8.0,  0.0,
                    -10.0, -8.0, 12.0,
                    x_top, -8.0, 12.0,
                    x_top,  8.0, 12.0,
                    -10.0,  8.0, 12.0,
                ];
                let indices: [u32; 36] = [
                    0, 2, 1, 0, 3, 2, // bottom
                    4, 5, 6, 4, 6, 7, // top
                    0, 1, 5, 0, 5, 4, // -Y
                    1, 2, 6, 1, 6, 5, // +X sloped wall
                    2, 3, 7, 2, 7, 6, // +Y
                    3, 0, 4, 3, 4, 7, // -X vertical wall
                ];
                let part =
                    ffi::register_triangle_mesh("test_overhang_bias_prism", &positions, &indices);
                assert!(part.object_id > 0);

                ffi::set_overhang_orientation_bias_spec(ffi::OverhangBiasSpec {
                    enabled: enable_bias,
                    weight: 0.90,
                    orthogonal_supported_angle_deg: supp_deg,
                    orthogonal_overhang_angle_deg: ovh_deg,
                    transition_width_deg: 6.0,
                });
                assert!(ffi::recompute_object_fields(part.object_id, 0.5, 3.0));

                let mut nx_sum = 0.0f32;
                let mut count = 0usize;
                for &iso in &[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 18.0] {
                    let mesh = ffi::extract_field_isosurface(part.object_id, "eikonal", iso, 0.0);
                    for (v, &sdf_val) in mesh.vertices.iter().zip(mesh.vertex_sdf.iter()) {
                        if sdf_val <= 0.0 && v.y.abs() <= 5.0 && v.z >= 3.5 && v.z <= 9.5 {
                            let t_z = (v.z / 12.0).clamp(0.0, 1.0);
                            let x_wall = (x_bot + (x_top - x_bot) * t_z) - part.origin_x;
                            if v.x >= x_wall - 2.5 && v.x <= x_wall + 0.25 {
                                nx_sum += v.nx;
                                count += 1;
                            }
                        }
                    }
                }

                let tilt_stats = ffi::get_object_field_stats(part.object_id, "tool_tilt_deg");
                assert!(tilt_stats.valid);
                assert!(
                    tilt_stats.max_value <= 45.05,
                    "Overhang orientation bias must never violate hard max_angle_from_bed_normal_deg (45 deg), got {}",
                    tilt_stats.max_value
                );

                let _ = ffi::remove_cad_object(part.object_id);
                assert!(
                    count > 10,
                    "Expected sufficient isosurface vertices near sloped wall (x_bot={x_bot}, x_top={x_top}, enable={enable_bias}, supp={supp_deg}, ovh={ovh_deg}), got {count}"
                );
                nx_sum / (count as f32)
            };

        // Case A: Steep supported slope (theta_ovh = -18.4 deg, inside [-30 deg, +60 deg] orthogonal band).
        // x_bot = 10.0, x_top = 6.0 over dz = 12.0 -> outward normal n_surf = (+0.9487, 0, +0.3162).
        // Orthogonal upward tangent t_up = (-0.3162, 0, +0.9487) has NEGATIVE X component (nx < 0).
        let nx_supp_ortho_off = measure_right_wall_iso_nx(10.0, 6.0, false, 30.0, 60.0);
        let nx_supp_ortho_on = measure_right_wall_iso_nx(10.0, 6.0, true, 30.0, 60.0);
        assert!(
            nx_supp_ortho_on < nx_supp_ortho_off - 0.04 && nx_supp_ortho_on < -0.04,
            "Expected steep supported wall (-18.4 deg in [-30, +60] deg band) to bias toward ORTHOGONAL (nx < 0), got off={nx_supp_ortho_off:.4}, on={nx_supp_ortho_on:.4}"
        );

        // Case B: Shallow supported roof (theta_ovh = -60.0 deg, below -30 deg -> CONFORMAL regime).
        // x_bot = 18.0, x_top = -2.785 over dz = 12.0 -> outward normal n_surf = (+0.5, 0, +0.866).
        // Conformal target +n_surf = (+0.5, 0, +0.866) has POSITIVE X component (nx > 0), whereas
        // expanding orthogonal_supported_angle_deg to 75 deg flips the exact same roof to ORTHOGONAL (nx < 0)!
        let nx_supp_conf_off = measure_right_wall_iso_nx(18.0, -2.785, false, 30.0, 60.0);
        let nx_supp_conf_on = measure_right_wall_iso_nx(18.0, -2.785, true, 30.0, 60.0);
        let nx_supp_forced_ortho = measure_right_wall_iso_nx(18.0, -2.785, true, 75.0, 60.0);
        assert!(
            nx_supp_conf_on > nx_supp_conf_off + 0.025 && nx_supp_conf_on > 0.025,
            "Expected shallow supported roof (-60 deg < -30 deg) to bias toward CONFORMAL (nx > 0), got off={nx_supp_conf_off:.4}, on={nx_supp_conf_on:.4}"
        );
        assert!(
            nx_supp_forced_ortho < 0.0 && nx_supp_conf_on > nx_supp_forced_ortho + 0.045,
            "Expected expanding orthogonal_supported_angle_deg to 75 deg to flip -60 deg roof from CONFORMAL ({nx_supp_conf_on:.4}) to ORTHOGONAL ({nx_supp_forced_ortho:.4})"
        );

        // Case C: Moderate overhang wall (theta_ovh = +40.0 deg, inside [-30 deg, +60 deg] orthogonal band).
        // x_bot = 6.0, x_top = 16.07 over dz = 12.0 -> outward normal n_surf = (+0.766, 0, -0.643).
        // Orthogonal upward tangent t_up = (+0.643, 0, +0.766) has strong POSITIVE X component (nx > 0.50),
        // whereas reducing orthogonal_overhang_angle_deg to 20 deg switches +40 deg to CONFORMAL (-n_surf),
        // pulling nx downward!
        let nx_ovh_ortho_on = measure_right_wall_iso_nx(6.0, 16.07, true, 30.0, 60.0);
        let nx_ovh_forced_conf = measure_right_wall_iso_nx(6.0, 16.07, true, 30.0, 20.0);
        assert!(
            nx_ovh_ortho_on > 0.45 && nx_ovh_ortho_on > nx_ovh_forced_conf + 0.25,
            "Expected 40 deg overhang (inside [-30, +60] deg band) to maintain ORTHOGONAL alignment (on={nx_ovh_ortho_on:.4} > forced_conf={nx_ovh_forced_conf:.4})"
        );

        // Case D: Steep overhang ceiling (theta_ovh = +63.4 deg > +60 deg -> CONFORMAL regime).
        // x_bot = 4.0, x_top = 28.0 over dz = 12.0 -> outward normal n_surf = (+0.4472, 0, -0.8944).
        // Conformal upward target -n_surf = (-0.4472, 0, +0.8944) pulls nx strongly downward
        // relative to both the unbiased field (nx_ovh_conf_off) and the orthogonal overhang regime
        // when orthogonal_overhang_angle_deg is expanded to 80 deg (nx_ovh_forced_ortho)!
        let nx_ovh_conf_off = measure_right_wall_iso_nx(4.0, 28.0, false, 30.0, 55.0);
        let nx_ovh_conf_on = measure_right_wall_iso_nx(4.0, 28.0, true, 30.0, 55.0);
        let nx_ovh_forced_ortho = measure_right_wall_iso_nx(4.0, 28.0, true, 30.0, 80.0);
        assert!(
            nx_ovh_conf_on < nx_ovh_conf_off - 0.05 && nx_ovh_forced_ortho > nx_ovh_conf_on + 0.25,
            "Expected steep 63.4 deg overhang (> +60 deg) to bias toward CONFORMAL (on={nx_ovh_conf_on:.4} < off={nx_ovh_conf_off:.4} and << forced_ortho={nx_ovh_forced_ortho:.4})"
        );

        // Restore default overhang bias spec
        ffi::set_overhang_orientation_bias_spec(ffi::OverhangBiasSpec {
            enabled: default_cfg.enable_overhang_orientation_bias,
            weight: default_cfg.overhang_bias_weight,
            orthogonal_supported_angle_deg: default_cfg.orthogonal_supported_angle_deg,
            orthogonal_overhang_angle_deg: default_cfg.orthogonal_overhang_angle_deg,
            transition_width_deg: default_cfg.overhang_bias_transition_deg,
        });
    }

    #[test]
    fn test_unreachable_order_field_detection_and_support_generation() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ffi::set_spatial_orientation_hints(&[]);
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);

        // 1. Build a watertight L-shaped cantilever + two-pillar bridge mesh:
        //    - Left pillar:  X in [-10.0, -4.0], Y in [-4.0, 4.0], Z in [0.0, 6.0]
        //    - Right pillar: X in [  4.0, 10.0], Y in [-4.0, 4.0], Z in [0.0, 6.0]
        //    - Top deck spanning the 8.0 mm gap between pillars AND cantilevered out in +Y to Y = +10.0:
        //      X in [-10.0, 10.0], Y in [-4.0, 10.0], Z in [4.0, 6.0]
        let x_coords = [-10.0f32, -4.0, 4.0, 10.0];
        let y_coords = [-4.0f32, 4.0, 10.0];
        let z_coords = [0.0f32, 4.0, 6.0];
        let is_occupied = |ix: i32, iy: i32, iz: i32| -> bool {
            if ix < 0 || ix >= 3 || iy < 0 || iy >= 2 || iz < 0 || iz >= 2 {
                return false;
            }
            match (ix, iy, iz) {
                (0, 0, 0) | (2, 0, 0) => true,
                (_, _, 1) => true,
                _ => false,
            }
        };

        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut add_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
            let base = (positions.len() / 3) as u32;
            for p in [p0, p1, p2, p3] {
                positions.extend_from_slice(&p);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };

        for ix in 0..3 {
            for iy in 0..2 {
                for iz in 0..2 {
                    if !is_occupied(ix, iy, iz) {
                        continue;
                    }
                    let x0 = x_coords[ix as usize];
                    let x1 = x_coords[(ix + 1) as usize];
                    let y0 = y_coords[iy as usize];
                    let y1 = y_coords[(iy + 1) as usize];
                    let z0 = z_coords[iz as usize];
                    let z1 = z_coords[(iz + 1) as usize];

                    if !is_occupied(ix - 1, iy, iz) {
                        add_quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]);
                    }
                    if !is_occupied(ix + 1, iy, iz) {
                        add_quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]);
                    }
                    if !is_occupied(ix, iy - 1, iz) {
                        add_quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]);
                    }
                    if !is_occupied(ix, iy + 1, iz) {
                        add_quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]);
                    }
                    if !is_occupied(ix, iy, iz - 1) {
                        add_quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]);
                    }
                    if !is_occupied(ix, iy, iz + 1) {
                        add_quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]);
                    }
                }
            }
        }

        let part = ffi::register_triangle_mesh("support_detection_test_arch", &positions, &indices);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.slicing.enable_support_generation = true;
        cfg.slicing.seed_order_field_from_supports = true;
        cfg.slicing.max_bridge_length_mm = 40.0;
        cfg.slicing.overhang_angle_threshold_deg = 45.0;
        crate::ui::sync_config_to_backend(&cfg);

        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        // 2. Verify `"support_needed"` field is exposed in `list_object_fields` and has positive range
        let fields = ffi::list_object_fields(part.object_id);
        assert!(
            fields.iter().any(|f| f == "support_needed"),
            "Expected 'support_needed' in list_object_fields, got {fields:?}"
        );
        let sn_stats = ffi::get_object_field_stats(part.object_id, "support_needed");
        assert!(sn_stats.valid);
        assert!(
            sn_stats.max_value > 0.5,
            "Expected support_needed max_value > 0.5 on cantilevered overhang, got {}",
            sn_stats.max_value
        );

        // 3. Verify vertex sampling (`get_object_vertex_support_needed`) and `ObjectSurfaceColorMode::SupportNeeded`:
        //    - Supported pillar base / top deck vertices have support_needed == 0.0
        //    - Two-pillar bridge underside (X in [-3, 3], Y in [-2, 2], Z = 4.0) is spanned by an 8 mm bridge
        //      (<= max_bridge_length_mm = 40 mm), so it does NOT require support!
        //    - One-sided cantilevered overhang underside (Y in [6.5, 9.5], Z = 4.0) cannot be bridged on two piers
        //      and cannot be reached causally by the constrained order field, so it has support_needed > 0.5!
        let verts = ffi::get_object_vertices(part.object_id);
        let vert_support = ffi::get_object_vertex_support_needed(part.object_id);
        assert_eq!(vert_support.len(), verts.len());

        let (colors_sn, min_sn, max_sn, _mean_sn, uses_sn_field) =
            compute_object_surface_vertex_colors(
                &verts,
                &Transform::IDENTITY,
                &cfg,
                ObjectSurfaceColorMode::SupportNeeded,
                &vert_support,
            );
        assert!(uses_sn_field);
        assert_eq!(colors_sn.len(), verts.len());
        assert!(min_sn < 1.0 && max_sn > 50.0);

        let mut max_cantilever_support = 0.0f32;
        let mut max_top_deck_support = 0.0f32;
        for (v, &s) in verts.iter().zip(vert_support.iter()) {
            let wy = v.y + part.origin_y;
            let wz = v.z + part.origin_z;
            if (wz - 4.0).abs() < 0.25 && v.nz < -0.5 && wy >= 6.0 {
                max_cantilever_support = max_cantilever_support.max(s);
            }
            if (wz - 6.0).abs() < 0.25 && v.nz > 0.5 {
                max_top_deck_support = max_top_deck_support.max(s);
            }
        }
        assert!(
            max_cantilever_support > 0.5,
            "Expected cantilevered underside (Z=4, Y>=6) to require support (> 0.5), got {max_cantilever_support}"
        );
        assert!(
            max_top_deck_support < 0.1,
            "Expected upward top deck (Z=6) to need 0 support, got {max_top_deck_support}"
        );

        // Also check the 2D slice of `"support_needed"` at Z = 4.0 mm (normalized pos = 4.0 / 6.0 = 0.667):
        // The cantilevered region (Y > 5) has positive support_needed whereas the bridged arch center
        // (X in [-2, 2], Y in [-2, 2]) has 0.0 support_needed when max_bridge_length_mm = 40.0 mm!
        let slice_z4 = ffi::sample_field_slice(part.object_id, "support_needed", 2, 4.0 / 6.0, 40);
        assert!(slice_z4.valid);

        // 4. Generate toolpaths and G-code with support generation enabled:
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid);
        assert!(
            tp.support_count > 0,
            "Expected Support Base toolpaths (feature_kind == 7) under the unreachable cantilever, got 0"
        );
        assert!(
            tp.support_interface_count > 0,
            "Expected Support Interface toolpaths (feature_kind == 8) beneath the cantilever contact ceiling, got 0"
        );
        // Verify that layer 0 (the build plate) includes the bottom bed adhesion floor
        // (SupportInterface, feature_kind == 8) when support_bed_interface_layers >= 1:
        let first_layer_has_bed_intf = tp
            .layers
            .first()
            .is_some_and(|l| l.paths.iter().any(|p| p.feature_kind == 8));
        assert!(
            first_layer_has_bed_intf,
            "Expected first layer on the build plate to include a bottom SupportInterface bed adhesion floor (8)"
        );

        let planned = crate::gcode::plan_and_emit_gcode_for_object(
            "support_detection_test_arch",
            part.object_id,
            &tp,
            Transform::IDENTITY,
            &cfg,
        );
        let gcode_str = &planned.gcode_text;
        assert!(
            gcode_str.contains("; TYPE:Support Base") && gcode_str.contains("; TYPE:Support Interface"),
            "Expected emitted G-code to contain '; TYPE:Support Base' and '; TYPE:Support Interface' moves"
        );

        // 5. Verify the `support_only_unreachable_and_bridges` toggle:
        //    (a) With `support_only_unreachable_and_bridges = true` and `max_bridge_length_mm = 40.0` (> 8 mm bridge):
        //        - The one-sided cantilever (Y in [4, 10]) is causally reachable from the pillars and is NOT
        //          a two-pier bridge, so it is left unsupported (`support_count == 0`).
        //        - The 8 mm two-pillar bridge (X in [-4, 4], Y in [-4, 4]) is <= 40 mm, so it is also unsupported.
        cfg.slicing.support_only_unreachable_and_bridges = true;
        cfg.slicing.max_bridge_length_mm = 40.0;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));
        let tp_only_unreach_40 = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_only_unreach_40.valid);
        assert_eq!(
            tp_only_unreach_40.support_count, 0,
            "Expected 0 supports when support_only_unreachable_and_bridges=true and bridge span (8 mm) <= max_bridge_length_mm (40 mm)"
        );

        //    (b) With `support_only_unreachable_and_bridges = true` and `max_bridge_length_mm = 5.0` (< 8 mm bridge):
        //        - The 8 mm two-pillar bridge (X in [-4, 4], Y in [-4, 4]) now exceeds `max_bridge_length_mm = 5.0 mm`,
        //          so support IS generated beneath the bridge span even with `support_only_unreachable_and_bridges = true`!
        cfg.slicing.support_only_unreachable_and_bridges = true;
        cfg.slicing.max_bridge_length_mm = 5.0;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));
        let tp_only_unreach_5 = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_only_unreach_5.valid);
        assert!(
            tp_only_unreach_5.support_count > 0,
            "Expected support beneath the 8 mm two-pillar bridge when support_only_unreachable_and_bridges=true and max_bridge_length_mm=5.0 mm"
        );

        let _ = ffi::remove_cad_object(part.object_id);

        //    (c) Verify an actually unreachable mid-air island (Z in [2.5, 4.5] alongside a grounded pillar at X in [-8, -2], Z in [0, 5])
        //        still receives support when `support_only_unreachable_and_bridges = true` and `max_bridge_length_mm = 40.0`:
        let mut island_pos: Vec<f32> = Vec::new();
        let mut island_idx: Vec<u32> = Vec::new();
        let mut push_box = |bx0: f32, bx1: f32, by0: f32, by1: f32, bz0: f32, bz1: f32| {
            let mut add_q = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
                let base = (island_pos.len() / 3) as u32;
                for p in [p0, p1, p2, p3] {
                    island_pos.extend_from_slice(&p);
                }
                island_idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            };
            add_q([bx0, by0, bz0], [bx0, by0, bz1], [bx0, by1, bz1], [bx0, by1, bz0]);
            add_q([bx1, by0, bz0], [bx1, by1, bz0], [bx1, by1, bz1], [bx1, by0, bz1]);
            add_q([bx0, by0, bz0], [bx1, by0, bz0], [bx1, by0, bz1], [bx0, by0, bz1]);
            add_q([bx0, by1, bz0], [bx0, by1, bz1], [bx1, by1, bz1], [bx1, by1, bz0]);
            add_q([bx0, by0, bz0], [bx0, by1, bz0], [bx1, by1, bz0], [bx1, by0, bz0]);
            add_q([bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]);
        };
        push_box(-8.0, -2.0, -3.0, 3.0, 0.0, 5.0);
        push_box(2.0, 8.0, -3.0, 3.0, 2.5, 4.5);
        let island_part = ffi::register_triangle_mesh("unreachable_island_test", &island_pos, &island_idx);
        assert!(island_part.object_id > 0);
        cfg.slicing.support_only_unreachable_and_bridges = true;
        cfg.slicing.max_bridge_length_mm = 40.0;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(island_part.object_id, 0.5, 4.0));
        let tp_island = ffi::generate_object_shell_toolpaths(island_part.object_id, 0.0);
        assert!(tp_island.valid);
        assert!(
            tp_island.support_count > 0,
            "Expected support beneath the causally unreachable floating island when support_only_unreachable_and_bridges=true"
        );
        let _ = ffi::remove_cad_object(island_part.object_id);
    }

    #[test]
    fn test_rectilinear_cone_clearance_clamps_anisotropic_slopes() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let positions: [f32; 24] = [
            -6.0, -6.0, 0.0,
             6.0, -6.0, 0.0,
             6.0,  6.0, 0.0,
            -6.0,  6.0, 0.0,
            -6.0, -6.0, 12.0,
             6.0, -6.0, 12.0,
             6.0,  6.0, 12.0,
            -6.0,  6.0, 12.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ];
        let part = ffi::register_triangle_mesh("rect_cone_cube", &positions, &indices);
        assert!(part.object_id > 0);

        // Configure a 3-axis tool (has_rotational_axes = false) with an asymmetric RectilinearCone:
        // - In X: narrow cone (rect_half_angle_x_deg = 25° => max layer slope α_max,X = 65°)
        // - In Y: wide shroud (rect_half_angle_y_deg = 78° => max layer slope α_max,Y = 12°)
        let mut cfg = crate::config::SlicerConfig::default();
        cfg.tools[0].has_rotational_axes = false;
        cfg.tools[0].clearance_mode = crate::config::ClearanceProfileMode::RectilinearCone;
        cfg.tools[0].rect_tip_half_width_x_mm = 0.5;
        cfg.tools[0].rect_tip_half_width_y_mm = 0.5;
        cfg.tools[0].rect_half_angle_x_deg = 25.0;
        cfg.tools[0].rect_half_angle_y_deg = 78.0;
        cfg.tools[0].rect_height_mm = 45.0;
        crate::ui::sync_config_to_backend(&cfg);

        // Apply an equal diagonal 3D orientation hint in +X and +Y: (0.60, 0.60, 0.529)
        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 0.0, 6.0],
            direction_bed: [0.60, 0.60, 0.52915],
            influence_radius_mm: 25.0,
            weight: 1.0,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(part.object_id, 0.8, 3.0));

        // Sample the Eikonal field on a mid-height Z slice (axis=2, normalizedPos=0.5)
        // Because α_max,X = 65° >> α_max,Y = 12°, the Eikonal arrival time τ must vary much
        // more strongly across X (cols) than across Y (rows).
        let z_slice = ffi::sample_field_slice(part.object_id, "eikonal", 2, 0.5, 16);
        assert!(z_slice.valid);
        let mid = 8usize;
        let tau_x_neg = z_slice.values[mid * 16 + 4];
        let tau_x_pos = z_slice.values[mid * 16 + 11];
        let tau_y_neg = z_slice.values[4 * 16 + mid];
        let tau_y_pos = z_slice.values[11 * 16 + mid];
        let delta_x = (tau_x_pos - tau_x_neg).abs();
        let delta_y = (tau_y_pos - tau_y_neg).abs();
        assert!(
            delta_x > 1.8 * delta_y,
            "Expected asymmetric RectilinearCone to allow much steeper slope in X than Y (delta_x={delta_x}, delta_y={delta_y})"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_height_only_planar_order_field_mode_respects_queried_nozzle_axis() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let positions: [f32; 24] = [
            -6.0, -6.0, 0.0,
             6.0, -6.0, 0.0,
             6.0,  6.0, 0.0,
            -6.0,  6.0, 0.0,
            -6.0, -6.0, 12.0,
             6.0, -6.0, 12.0,
             6.0,  6.0, 12.0,
            -6.0,  6.0, 12.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            2, 3, 7, 2, 7, 6,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
        ];
        let part = ffi::register_triangle_mesh("planar_mode_cube", &positions, &indices);
        assert!(part.object_id > 0);

        // 1. Vertical nozzle axis (0, 0, 1) with HeightOnlyPlanar mode:
        // Even with a strong diagonal 3D hint, τ on a constant-Z slice must be constant
        // and tool_tilt_deg must be identically 0.
        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        crate::ui::sync_config_to_backend(&cfg);
        assert_eq!(ffi::get_order_field_mode(), 1);

        let hints = vec![ffi::SpatialOrientationHint {
            kind: 1,
            position_bed: [0.0, 0.0, 6.0],
            direction_bed: [0.70, 0.0, 0.714],
            influence_radius_mm: 25.0,
            weight: 1.0,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        }];
        ffi::set_spatial_orientation_hints(&hints);
        assert!(ffi::recompute_object_fields(part.object_id, 0.8, 3.0));

        let z_slice = ffi::sample_field_slice(part.object_id, "eikonal", 2, 0.5, 16);
        assert!(z_slice.valid);
        let mid = 8usize;
        let tau_x_neg = z_slice.values[mid * 16 + 4];
        let tau_x_pos = z_slice.values[mid * 16 + 11];
        assert!(
            (tau_x_pos - tau_x_neg).abs() < 1.0e-4,
            "Expected zero X slope on constant-Z slice in HeightOnlyPlanar mode with vertical nozzle, got tau_x_neg={tau_x_neg}, tau_x_pos={tau_x_pos}"
        );

        let tilt_stats = ffi::get_object_field_stats(part.object_id, "tool_tilt_deg");
        assert!(tilt_stats.valid && tilt_stats.active_voxel_count > 0);
        assert!(
            tilt_stats.max_value.abs() < 1.0e-4,
            "Expected 0 deg tool tilt in HeightOnlyPlanar mode, got max={}",
            tilt_stats.max_value
        );

        let all_layers = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(all_layers.valid);
        assert!(all_layers.layers.len() >= 10);

        // 2. Tilted 6-axis nozzle axis (0.6, 0.0, 0.8) in bed coordinates:
        // HeightOnlyPlanar must query nozzle_axis_bed rather than hardcoding +Z, so τ on a
        // constant-Z slice must vary linearly with X (Δτ / Δx = 0.6).
        cfg.machine.nozzle_axis_bed = [0.6, 0.0, 0.8];
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.8, 3.0));

        let tilted_z_slice = ffi::sample_field_slice(part.object_id, "eikonal", 2, 0.5, 16);
        assert!(tilted_z_slice.valid);
        let tau_tilted_neg = tilted_z_slice.values[mid * 16 + 4];
        let tau_tilted_pos = tilted_z_slice.values[mid * 16 + 11];
        assert!(
            tau_tilted_pos - tau_tilted_neg > 1.5,
            "Expected HeightOnlyPlanar to project along queried tilted nozzle_axis_bed [0.6, 0, 0.8], got tau_neg={tau_tilted_neg}, tau_pos={tau_tilted_pos}"
        );

        // Reset backend state to defaults
        ffi::set_spatial_orientation_hints(&[]);
        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_nonplanar_tree_support_routes_around_part_in_bed_only_mode() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Build a C-shaped bracket where an upper overhang (X in [0, 4.5], Y in [-1.5, 1.5], Z in [7, 9])
        // sits completely inside the horizontal footprint of a wider lower shelf (X in [-4, 6.5], Y in [-3.0, 3.0], Z in [0, 2.5]):
        // - Vertical spine: X in [-4, 0], Y in [-3.0, 3.0], Z in [0, 9]
        // - Wider lower base shelf: X in [0, 6.5], Y in [-3.0, 3.0], Z in [0, 2.5]
        // - Narrower upper overhang arm: X in [0, 4.5], Y in [-1.5, 1.5], Z in [7, 9]
        let mut c_pos: Vec<f32> = Vec::new();
        let mut c_idx: Vec<u32> = Vec::new();
        let mut push_box = |bx0: f32, bx1: f32, by0: f32, by1: f32, bz0: f32, bz1: f32| {
            let mut add_q = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
                let base = (c_pos.len() / 3) as u32;
                for p in [p0, p1, p2, p3] {
                    c_pos.extend_from_slice(&p);
                }
                c_idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            };
            add_q([bx0, by0, bz0], [bx0, by0, bz1], [bx0, by1, bz1], [bx0, by1, bz0]);
            add_q([bx1, by0, bz0], [bx1, by1, bz0], [bx1, by1, bz1], [bx1, by0, bz1]);
            add_q([bx0, by0, bz0], [bx1, by0, bz0], [bx1, by0, bz1], [bx0, by0, bz1]);
            add_q([bx0, by1, bz0], [bx0, by1, bz1], [bx1, by1, bz1], [bx1, by1, bz0]);
            add_q([bx0, by0, bz0], [bx0, by1, bz0], [bx1, by1, bz0], [bx1, by0, bz0]);
            add_q([bx0, by0, bz1], [bx1, by0, bz1], [bx1, by1, bz1], [bx0, by1, bz1]);
        };
        push_box(-4.0, 0.0, -3.0, 3.0, 0.0, 9.0);
        push_box(0.0, 6.5, -3.0, 3.0, 0.0, 2.5);
        push_box(0.0, 4.5, -1.5, 1.5, 7.0, 9.0);

        let part = ffi::register_triangle_mesh("c_bracket_tree_test", &c_pos, &c_idx);
        assert!(part.object_id > 0);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.slicing.enable_support_generation = true;
        cfg.slicing.support_only_unreachable_and_bridges = false;

        // 1. ColumnarGrid with `support_on_bed_only = true`:
        // Because the lower shelf (Z in [0, 2.5], X in [0, 6], Y in [-2, 2]) lies directly beneath
        // the upper overhang (X in [0, 5], Y in [-2, 2]), straight vertical columns are blocked!
        cfg.slicing.support_style = crate::config::SupportStyle::ColumnarGrid;
        cfg.slicing.support_on_bed_only = true;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        let tp_col_bed_only = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_col_bed_only.valid);
        assert_eq!(
            tp_col_bed_only.support_count, 0,
            "Expected 0 columnar supports when support_on_bed_only=true and overhang is directly over the lower shelf"
        );

        // 2. NonPlanarTree with `support_on_bed_only = true`:
        // Tree branches start under the upper overhang (Z ~ 6.8 mm), steer laterally in Y/X around
        // the narrow lower shelf (Y in [-2, 2]), and anchor on the build plate (Z = 0) with 5/6-axis
        // branch-tangent nozzle tilt!
        cfg.slicing.support_style = crate::config::SupportStyle::NonPlanarTree;
        cfg.slicing.support_on_bed_only = true;
        cfg.slicing.tree_branch_angle_deg = 50.0;
        cfg.tools[0].has_rotational_axes = true;
        cfg.tools[0].max_angle_from_bed_normal_deg = 45.0;
        crate::ui::sync_config_to_backend(&cfg);

        let tp_tree_bed_only = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_tree_bed_only.valid);
        assert!(
            tp_tree_bed_only.support_count > 0 && tp_tree_bed_only.support_interface_count > 0,
            "Expected NonPlanarTree support to route around the lower shelf to the build plate (got base={}, intf={})",
            tp_tree_bed_only.support_count,
            tp_tree_bed_only.support_interface_count
        );

        // Verify that the first layer on the build plate has a grounded tree trunk bed adhesion
        // floor pad (feature_kind == 8), and that leaning mid-height tree branches
        // tilt the 5/6-axis nozzle axis along the branch tangent!
        let first_layer_has_floor_intf = tp_tree_bed_only
            .layers
            .first()
            .is_some_and(|l| l.paths.iter().any(|p| p.feature_kind == 8));
        assert!(
            first_layer_has_floor_intf,
            "Expected NonPlanarTree trunk to reach the build plate (layer 0) with a bed adhesion floor pad (8)"
        );

        let mut max_lateral_nozzle_tilt = 0.0f32;
        for lyr in &tp_tree_bed_only.layers {
            for path in &lyr.paths {
                if path.feature_kind == 7 || path.feature_kind == 8 {
                    for pt in &path.points {
                        let lat = (pt.nx * pt.nx + pt.ny * pt.ny).sqrt();
                        max_lateral_nozzle_tilt = max_lateral_nozzle_tilt.max(lat);
                    }
                }
            }
        }
        assert!(
            max_lateral_nozzle_tilt > 0.10,
            "Expected 5/6-axis nozzle axis on leaning tree support branch to tilt along 3D branch tangent, got max lateral component={max_lateral_nozzle_tilt}"
        );

        // Verify that every closed tree branch perimeter ring (feature_kind == 7, is_closed == true)
        // on layer k > 0 is continuously supported by the layer immediately below it (no unsupported
        // branch overhang jumps where a tree support forks):
        let mut prev_layer_sup_pts: Vec<[f32; 3]> = Vec::new();
        for (lyr_idx, lyr) in tp_tree_bed_only.layers.iter().enumerate() {
            let mut curr_sup_pts: Vec<[f32; 3]> = Vec::new();
            for path in &lyr.paths {
                if path.feature_kind == 7 || path.feature_kind == 8 {
                    for pt in &path.points {
                        curr_sup_pts.push([pt.x, pt.y, pt.z]);
                    }
                }
            }
            if lyr_idx > 0 && !prev_layer_sup_pts.is_empty() {
                for path in &lyr.paths {
                    if path.feature_kind == 7 && path.is_closed {
                        for pt in &path.points {
                            let min_d_xy = prev_layer_sup_pts
                                .iter()
                                .map(|p| ((pt.x - p[0]).powi(2) + (pt.y - p[1]).powi(2)).sqrt())
                                .fold(f32::INFINITY, f32::min);
                            assert!(
                                min_d_xy <= 0.95,
                                "Unsupported tree branch wall point at layer {} ({:.2}, {:.2}, {:.2}): nearest lower-layer support point is {:.3} mm away horizontally",
                                lyr_idx,
                                pt.x,
                                pt.y,
                                pt.z,
                                min_d_xy
                            );
                        }
                    }
                }
            }
            if !curr_sup_pts.is_empty() {
                prev_layer_sup_pts = curr_sup_pts;
            }
        }

        // Verify that the tree support trunk expands toward the build plate (lower trunk outer
        // wall radius on layer 1 is significantly larger than the upper branch tip radius):
        let ring_radius = |p: &ffi::ExtrusionPath| -> f32 {
            let n = p.points.len().max(1) as f32;
            let cx = p.points.iter().map(|pt| pt.x).sum::<f32>() / n;
            let cy = p.points.iter().map(|pt| pt.y).sum::<f32>() / n;
            p.points
                .iter()
                .map(|pt| ((pt.x - cx).powi(2) + (pt.y - cy).powi(2)).sqrt())
                .fold(0.0f32, f32::max)
        };
        let lower_trunk_max_r = tp_tree_bed_only
            .layers
            .iter()
            .skip(1)
            .take(5)
            .flat_map(|l| l.paths.iter())
            .filter(|p| p.feature_kind == 7)
            .map(&ring_radius)
            .fold(0.0f32, f32::max);
        let upper_branch_max_r = tp_tree_bed_only
            .layers
            .iter()
            .rev()
            .filter(|l| l.paths.iter().any(|p| p.feature_kind == 7 && p.is_closed))
            .take(3)
            .flat_map(|l| l.paths.iter())
            .filter(|p| p.feature_kind == 7 && p.is_closed)
            .map(&ring_radius)
            .fold(0.0f32, f32::max);
        assert!(
            lower_trunk_max_r > upper_branch_max_r + 0.35,
            "Expected tree support trunk to expand toward the bed (lower_trunk_max_r={lower_trunk_max_r:.3} mm vs upper_branch_max_r={upper_branch_max_r:.3} mm)"
        );

        // Verify default `tree_branch_wall_count == 1` produces hollow single-wall trunks above a
        // multi-ring closed bed adhesion base pad (`feature_kind == 8` on layer 0), and that
        // configuring `tree_branch_wall_count = 2` increases the trunk wall count accordingly:
        assert_eq!(cfg.slicing.tree_branch_wall_count, 1);
        let l0_base_pad_rings = tp_tree_bed_only.layers[0]
            .paths
            .iter()
            .filter(|p| p.feature_kind == 8)
            .count();
        assert!(
            l0_base_pad_rings >= 3,
            "Expected closed multi-ring bed adhesion base pad on layer 0, got {l0_base_pad_rings} rings"
        );
        let l2_1wall_rings = tp_tree_bed_only.layers[2]
            .paths
            .iter()
            .filter(|p| p.feature_kind == 7)
            .count();

        cfg.slicing.tree_branch_wall_count = 2;
        crate::ui::sync_config_to_backend(&cfg);
        let tp_tree_2wall = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        let l2_2wall_rings = tp_tree_2wall.layers[2]
            .paths
            .iter()
            .filter(|p| p.feature_kind == 7)
            .count();
        assert!(
            l2_2wall_rings >= 2 * l2_1wall_rings,
            "Expected 2-wall hollow trunk on layer 2 to emit at least 2x as many wall loops as 1-wall hollow trunk (1-wall={l2_1wall_rings}, 2-wall={l2_2wall_rings})"
        );

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_pvm_l_series_c_board_bracket_support_coverage() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let stl_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/PVM_L_Series_C_Board_Bracket.stl");
        let obj = crate::io::stl::load_stl_file(&stl_path)
            .expect("Failed to load PVM_L_Series_C_Board_Bracket.stl");

        // Extract elevated downward-facing overhang triangle centroids from the mesh
        let min_z = obj.local_min.z;
        let mut overhang_centroids: Vec<[f32; 3]> = Vec::new();
        for tri in obj.indices.chunks_exact(3) {
            let v0 = &obj.vertices[tri[0] as usize];
            let v1 = &obj.vertices[tri[1] as usize];
            let v2 = &obj.vertices[tri[2] as usize];
            let p0 = [v0.x, v0.y, v0.z];
            let p1 = [v1.x, v1.y, v1.z];
            let p2 = [v2.x, v2.y, v2.z];
            let ux = p1[0] - p0[0];
            let uy = p1[1] - p0[1];
            let uz = p1[2] - p0[2];
            let vx = p2[0] - p0[0];
            let vy = p2[1] - p0[1];
            let vz = p2[2] - p0[2];
            let nx = uy * vz - uz * vy;
            let ny = uz * vx - ux * vz;
            let nz = ux * vy - uy * vx;
            let area2 = (nx * nx + ny * ny + nz * nz).sqrt();
            if area2 < 0.20 {
                continue;
            }
            let unit_nz = nz / area2;
            let cz = (p0[2] + p1[2] + p2[2]) / 3.0;
            // Steep downward-facing overhang triangles elevated > 3 mm above the build plate
            if unit_nz < -0.80 && (cz - min_z) > 3.0 {
                let cx = (p0[0] + p1[0] + p2[0]) / 3.0;
                let cy = (p0[1] + p1[1] + p2[1]) / 3.0;
                overhang_centroids.push([cx, cy, cz]);
            }
        }
        assert!(
            !overhang_centroids.is_empty(),
            "Expected PVM_L_Series_C_Board_Bracket.stl to have elevated downward-facing overhang triangles"
        );

        for (style, on_bed_only) in [
            (crate::config::SupportStyle::ColumnarGrid, false),
            (crate::config::SupportStyle::NonPlanarTree, false),
            (crate::config::SupportStyle::NonPlanarTree, true),
        ] {
            let mut cfg = crate::config::SlicerConfig::default();
            cfg.slicing.enable_support_generation = true;
            cfg.slicing.support_style = style;
            cfg.slicing.support_on_bed_only = on_bed_only;
            cfg.slicing.support_only_unreachable_and_bridges = false;
            crate::ui::sync_config_to_backend(&cfg);

            assert!(ffi::recompute_object_fields(
                obj.object_id,
                cfg.eikonal.voxel_size_mm,
                cfg.eikonal.narrow_band_width_voxels as f32
            ));

            let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
            assert!(tp.valid);
            assert!(
                tp.support_count > 0 && tp.support_interface_count > 0,
                "Expected support & support interface paths for {:?} (on_bed_only={}) on PVM_L_Series_C_Board_Bracket.stl (got support={}, intf={})",
                style,
                on_bed_only,
                tp.support_count,
                tp.support_interface_count
            );

            // Collect all support points (7 or 8)
            let mut support_pts: Vec<[f32; 3]> = Vec::new();
            for lyr in &tp.layers {
                for path in &lyr.paths {
                    if path.feature_kind == 7 || path.feature_kind == 8 {
                        for pt in &path.points {
                            support_pts.push([pt.x, pt.y, pt.z]);
                        }
                    }
                }
            }

            // Verify that at least 85% of downward-facing overhang centroids have a support point
            // directly underneath them (within 1.6 mm horizontally and 0.05..1.6 mm vertically below)
            let mut covered_count = 0usize;
            for &[cx, cy, cz] in &overhang_centroids {
                let has_contact_below = support_pts.iter().any(|&[sx, sy, sz]| {
                    let dx = sx - cx;
                    let dy = sy - cy;
                    let dz = cz - sz;
                    (dx * dx + dy * dy) <= 1.6 * 1.6 && dz >= 0.05 && dz <= 1.6
                });
                if has_contact_below {
                    covered_count += 1;
                }
            }
            let coverage_ratio = covered_count as f32 / overhang_centroids.len() as f32;
            assert!(
                coverage_ratio >= 0.85,
                "Expected {:?} (on_bed_only={}) to support the interior of PVM_L_Series_C_Board_Bracket.stl overhangs (>= 85% coverage), got {:.1}% ({}/{})",
                style,
                on_bed_only,
                coverage_ratio * 100.0,
                covered_count,
                overhang_centroids.len()
            );

            // Verify 3D collision avoidance between support paths (7 or 8) and object extrusion paths (<= 6)
            // on every layer: no support bead may laterally interpenetrate a same-layer object wall/infill bead!
            for lyr in &tp.layers {
                let mut obj_pts: Vec<([f32; 3], f32)> = Vec::new();
                let mut sup_pts: Vec<([f32; 3], f32)> = Vec::new();
                for path in &lyr.paths {
                    if path.feature_kind <= 6 {
                        for pt in &path.points {
                            obj_pts.push(([pt.x, pt.y, pt.z], 0.5 * pt.width_mm.max(0.15)));
                        }
                    } else if path.feature_kind == 7 || path.feature_kind == 8 {
                        for pt in &path.points {
                            sup_pts.push(([pt.x, pt.y, pt.z], 0.5 * pt.width_mm.max(0.15)));
                        }
                    }
                }
                for &(sp, sw) in &sup_pts {
                    for &(op, ow) in &obj_pts {
                        let dz = (sp[2] - op[2]).abs();
                        if dz < 0.15 {
                            let d_xy = ((sp[0] - op[0]).powi(2) + (sp[1] - op[1]).powi(2)).sqrt();
                            assert!(
                                d_xy >= sw + ow + 0.05,
                                "Support point {:?} collided laterally with same-layer object extrusion {:?} (d_xy={:.3} < min {:.3})",
                                sp,
                                op,
                                d_xy,
                                sw + ow + 0.05
                            );
                        }
                    }
                }
            }

            // Verify on every layer (including the bed adhesion floor and support base layers)
            // that no two support paths intersect each other in 2D:
            for (lyr_idx, lyr) in tp.layers.iter().enumerate() {
                let mut layer_segs: Vec<(usize, usize, [f32; 2], [f32; 2])> = Vec::new();
                for (p_idx, path) in lyr.paths.iter().enumerate() {
                    if path.feature_kind == 7 || path.feature_kind == 8 {
                        for (s_idx, w) in path.points.windows(2).enumerate() {
                            let a = [w[0].x, w[0].y];
                            let b = [w[1].x, w[1].y];
                            if ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)) > 1.0e-6 {
                                layer_segs.push((p_idx, s_idx, a, b));
                            }
                        }
                    }
                }
                for i in 0..layer_segs.len() {
                    let (p1, _s1, a, b) = layer_segs[i];
                    for j in (i + 1)..layer_segs.len() {
                        let (p2, _s2, c, d) = layer_segs[j];
                        if p1 == p2 {
                            continue;
                        }
                        let ccw = |u: [f32; 2], v: [f32; 2], w: [f32; 2]| -> f32 {
                            (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
                        };
                        let d1 = ccw(a, b, c);
                        let d2 = ccw(a, b, d);
                        let d3 = ccw(c, d, a);
                        let d4 = ccw(c, d, b);
                        if ((d1 > 1.0e-5 && d2 < -1.0e-5) || (d1 < -1.0e-5 && d2 > 1.0e-5))
                            && ((d3 > 1.0e-5 && d4 < -1.0e-5) || (d3 < -1.0e-5 && d4 > 1.0e-5))
                        {
                            panic!(
                                "Intersecting {:?} (on_bed_only={}) support segments on layer {} between path {} ({:?}->{:?}) and path {} ({:?}->{:?})",
                                style, on_bed_only, lyr_idx, p1, a, b, p2, c, d
                            );
                        }
                    }
                }
            }

            // In NonPlanarTree + on_bed_only mode, verify that every closed tree branch perimeter
            // ring on layer k > 0 is supported from below without unsupported fork overhangs:
            if style == crate::config::SupportStyle::NonPlanarTree && on_bed_only {
                let mut prev_sup_pts: Vec<[f32; 3]> = Vec::new();
                for (lyr_idx, lyr) in tp.layers.iter().enumerate() {
                    let mut curr_sup_pts: Vec<[f32; 3]> = Vec::new();
                    for path in &lyr.paths {
                        if path.feature_kind == 7 || path.feature_kind == 8 {
                            for pt in &path.points {
                                curr_sup_pts.push([pt.x, pt.y, pt.z]);
                            }
                        }
                    }

                    if lyr_idx > 0 && !prev_sup_pts.is_empty() {
                        for path in &lyr.paths {
                            if path.feature_kind == 7 && path.is_closed {
                                for pt in &path.points {
                                    let min_d_xy = prev_sup_pts
                                        .iter()
                                        .map(|p| ((pt.x - p[0]).powi(2) + (pt.y - p[1]).powi(2)).sqrt())
                                        .fold(f32::INFINITY, f32::min);
                                    assert!(
                                        min_d_xy <= 0.95,
                                        "Unsupported tree branch fork/wall point on PVM bracket at layer {} ({:.2}, {:.2}, {:.2}): nearest lower-layer support point is {:.3} mm away horizontally",
                                        lyr_idx,
                                        pt.x,
                                        pt.y,
                                        pt.z,
                                        min_d_xy
                                    );
                                }
                            }
                        }
                    }
                    if !curr_sup_pts.is_empty() {
                        prev_sup_pts = curr_sup_pts;
                    }
                }

                // Verify that tree support trunks on PVM_L_Series_C_Board_Bracket.stl expand
                // significantly toward the build plate for strong bed adhesion:
                let ring_radius = |p: &ffi::ExtrusionPath| -> f32 {
                    let n = p.points.len().max(1) as f32;
                    let cx = p.points.iter().map(|pt| pt.x).sum::<f32>() / n;
                    let cy = p.points.iter().map(|pt| pt.y).sum::<f32>() / n;
                    p.points
                        .iter()
                        .map(|pt| ((pt.x - cx).powi(2) + (pt.y - cy).powi(2)).sqrt())
                        .fold(0.0f32, f32::max)
                };
                let base_trunk_max_r = tp
                    .layers
                    .iter()
                    .skip(1)
                    .take(4)
                    .flat_map(|l| l.paths.iter())
                    .filter(|p| p.feature_kind == 7 && p.is_closed)
                    .map(&ring_radius)
                    .fold(0.0f32, f32::max);
                assert!(
                    base_trunk_max_r >= 1.80,
                    "Expected PVM bracket tree support trunks to expand toward the build plate (got base_trunk_max_r={base_trunk_max_r:.3} mm, tip outer wall is ~0.99 mm)"
                );
            }
        }

        // Verify zero support interface spacing (`0.0 mm` = 100% solid touching contact roof)
        // and multi-tool support (`support_tool_index = Some(1)` -> `T1` toolchange in G-code):
        let mut cfg_zero = crate::config::SlicerConfig::default();
        cfg_zero.slicing.enable_support_generation = true;
        cfg_zero.slicing.support_style = crate::config::SupportStyle::NonPlanarTree;
        cfg_zero.slicing.support_interface_spacing_mm = 0.0; // Zero gap!
        cfg_zero.slicing.support_line_width_mm = 0.0; // Auto from support tool nozzle diameter
        cfg_zero.tools[1].id = 1;
        cfg_zero.tools[1].nozzle_diameter_mm = 0.50;
        cfg_zero.tools[1].temperature_celsius = 240.0;
        cfg_zero.slicing.support_tool_index = Some(1);
        cfg_zero.slicing.sanitize();
        assert_eq!(
            cfg_zero.slicing.support_interface_spacing_mm, 0.0,
            "Zero support_interface_spacing_mm must not be clamped above 0.0"
        );
        crate::ui::sync_config_to_backend(&cfg_zero);

        let tp_zero = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp_zero.valid && tp_zero.support_interface_count > 0);
        // Support paths must inherit tool 1's 0.50 mm nozzle diameter when support_line_width_mm == 0.0
        let has_t1_width = tp_zero.layers.iter().any(|l| {
            l.paths
                .iter()
                .any(|p| (p.feature_kind == 7 || p.feature_kind == 8) && (p.mean_width_mm - 0.50).abs() < 1.0e-3)
        });
        assert!(
            has_t1_width,
            "Expected support paths to use selected Support Tool T1's nozzle diameter (0.50 mm)"
        );

        let planned_multi = crate::gcode::plan_and_emit_gcode_for_object(
            "PVM_Bracket_MultiTool",
            obj.object_id,
            &tp_zero,
            Transform::IDENTITY,
            &cfg_zero,
        );
        assert!(
            planned_multi.gcode_text.contains("M104 S240 T1")
                && planned_multi.gcode_text.contains("T1 ; Switch tool")
                && planned_multi.gcode_text.contains("T0 ; Switch tool"),
            "Expected multi-tool G-code to preheat and switch between T1 (support) and T0 (object)"
        );

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_benchy_like_hull_infill_never_extends_outside_object() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct a Benchy-like overhanging boat hull with a recessed bottom plaque cavity
        // (`z = 0..1.2 mm`) and flaring bow/chine transition (`z = 0..5.0 mm`) in Dual-Grid mode.
        // Previously, `pl.sdfClipAllowance` (~0.1125 mm around z = 1.6..1.8 mm) collided with the
        // `0.45 * fineVoxelSize` outer-boundary filter in `generateShellToolpathsForLayers`,
        // dropping part of `\Gamma_outer` and causing `SolidInfill` to spill outside the object.
        let mut stl = String::from("solid benchy_hull\n");
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                p00[0], p00[1], p00[2], p10[0], p10[1], p10[2], p11[0], p11[1], p11[2]
            ));
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p01[0], p01[1], p01[2]
            ));
        };
        let mut add_box = |x0: f32, x1: f32, y0: f32, y1: f32, z0: f32, z1: f32| {
            add_quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]);
            add_quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]);
            add_quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]);
            add_quad([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1]);
            add_quad([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]);
            add_quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]);
        };

        // Lower hull ring surrounding a recessed bottom plaque cavity (`x in [-5, 5], y in [-3, 3], z in [0, 1.2]`)
        add_box(-12.0, 12.0, -8.0, -3.0, 0.0, 1.2);
        add_box(-12.0, 12.0, 3.0, 8.0, 0.0, 1.2);
        add_box(-12.0, -5.0, -3.0, 3.0, 0.0, 1.2);
        add_box(5.0, 12.0, -3.0, 3.0, 0.0, 1.2);
        // Internal plaque letter/rib island inside the bottom recess
        add_box(-2.0, 2.0, -1.2, 1.2, 0.0, 1.2);
        // Upper hull slab bridging over the plaque recess from z = 1.2 to z = 4.2 mm
        add_box(-12.0, 12.0, -8.0, 8.0, 1.2, 4.2);
        stl.push_str("endsolid benchy_hull\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_benchy_hull_infill_bounds.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write temporary STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected synthetic Benchy hull STL import to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = true;
        cfg.eikonal.coarse_grid_ratio = 4;
        cfg.slicing.infill_density_percent = 20.0;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));
        let layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(layers.valid && layers.layers.len() >= 10);

        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);

        // Verify on every layer that:
        // 1. All emitted object extrusion points and segment midpoints lie strictly inside the
        //    [-12, 12] x [-8, 8] outer hull footprint (and below z = 1.2 mm, never inside the
        //    empty bottom plaque recess cavities).
        // 2. All infill paths (`GapFill = 2`, `SolidInfill = 3`, `SparseInfill = 4`) stay strictly
        //    inside the outer wall perimeter on every layer, including the transition layers ~1.2..2.2 mm up.
        for layer in &tp.layers {
            for path in &layer.paths {
                if path.feature_kind == 7 || path.feature_kind == 8 {
                    continue;
                }
                let is_infill = matches!(path.feature_kind, 2 | 3 | 4);
                let max_x = if is_infill { 11.85 } else { 12.15 };
                let max_y = if is_infill { 7.85 } else { 8.15 };

                for (idx, pt) in path.points.iter().enumerate() {
                    assert!(
                        pt.x.abs() <= max_x && pt.y.abs() <= max_y,
                        "Layer {} (iso={:.2} mm) feature_kind={} point ({:.3}, {:.3}, {:.3}) extended outside object hull bounds (±{:.2}, ±{:.2})",
                        layer.layer_index,
                        layer.isovalue,
                        path.feature_kind,
                        pt.x,
                        pt.y,
                        pt.z,
                        max_x,
                        max_y
                    );
                    if idx + 1 < path.points.len() {
                        let nxt = &path.points[idx + 1];
                        let mx = 0.5 * (pt.x + nxt.x);
                        let my = 0.5 * (pt.y + nxt.y);
                        let mz = 0.5 * (pt.z + nxt.z);
                        assert!(
                            mx.abs() <= max_x && my.abs() <= max_y,
                            "Layer {} (iso={:.2} mm) feature_kind={} segment midpoint ({:.3}, {:.3}, {:.3}) extended outside object hull bounds",
                            layer.layer_index,
                            layer.isovalue,
                            path.feature_kind,
                            mx,
                            my,
                            mz
                        );
                        // Below z = 0.95 mm, no solid/sparse infill turnaround may jump across the recessed
                        // plaque cavity (`x in (-4.6, -2.4)` or `(2.4, 4.6)` with `y in (-2.6, 2.6)`).
                        if is_infill && mz < 0.95 {
                            let in_left_void = mx > -4.55 && mx < -2.45 && my.abs() < 2.55;
                            let in_right_void = mx > 2.45 && mx < 4.55 && my.abs() < 2.55;
                            assert!(
                                !in_left_void && !in_right_void,
                                "Layer {} (iso={:.2} mm) feature_kind={} turnaround midpoint ({:.3}, {:.3}, {:.3}) jumped across bottom plaque recess cavity!",
                                layer.layer_index,
                                layer.isovalue,
                                path.feature_kind,
                                mx,
                                my,
                                mz
                            );
                        }
                    }
                }
            }
        }

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_surface_skin_sloped_deck_with_protrusion_auto_and_manual() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct a Benchy-like sloped deck (`x in [-10, 10], y in [-6, 6]`, top deck sloped at ~18 deg:
        // `z_deck(x) = 3.0 + 0.15 * (x + 10.0)` from `z = 3.0 mm` at `x = -10` to `z = 6.0 mm` at `x = +10`)
        // with a central cabin/coaming protrusion (`x in [-2.5, 2.5], y in [-2.5, 2.5]`) rising up to `z = 9.0 mm`.
        // Under planar or weakly-tilted order fields, the isosurfaces arrive at the 18° sloped deck at an
        // oblique angle (producing stair-stepping unless conformal SDF skinning is active).
        let mut stl = String::from("solid sloped_deck_with_cabin\n");
        let mut add_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(p00, p10, p11);
            add_tri(p00, p11, p01);
        };

        let z_deck = |x: f32| -> f32 { 3.0 + 0.15 * (x + 10.0) };
        let x0 = -10.0_f32;
        let x1 = 10.0_f32;
        let y0 = -6.0_f32;
        let y1 = 6.0_f32;
        let z00 = z_deck(x0);
        let z10 = z_deck(x1);

        // Sloped hull block (`[-10, 10] x [-6, 6]`, bottom `z = 0`, sloped top `z = z_deck(x)`)
        add_quad([x0, y0, 0.0], [x0, y1, 0.0], [x1, y1, 0.0], [x1, y0, 0.0]); // bottom
        add_quad([x0, y0, z00], [x1, y0, z10], [x1, y1, z10], [x0, y1, z00]); // sloped deck
        add_quad([x0, y0, 0.0], [x1, y0, 0.0], [x1, y0, z10], [x0, y0, z00]); // front (-Y)
        add_quad([x1, y1, 0.0], [x0, y1, 0.0], [x0, y1, z00], [x1, y1, z10]); // back (+Y)
        add_quad([x0, y1, 0.0], [x0, y0, 0.0], [x0, y0, z00], [x0, y1, z00]); // left (-X)
        add_quad([x1, y0, 0.0], [x1, y1, 0.0], [x1, y1, z10], [x1, y0, z10]); // right (+X)

        // Central cabin protrusion (`[-2.5, 2.5] x [-2.5, 2.5]`, `z = 0..9.0 mm`)
        let cx0 = -2.5_f32;
        let cx1 = 2.5_f32;
        let cy0 = -2.5_f32;
        let cy1 = 2.5_f32;
        let cz1 = 9.0_f32;
        add_quad([cx0, cy0, 0.0], [cx0, cy1, 0.0], [cx1, cy1, 0.0], [cx1, cy0, 0.0]);
        add_quad([cx0, cy0, cz1], [cx1, cy0, cz1], [cx1, cy1, cz1], [cx0, cy1, cz1]);
        add_quad([cx0, cy0, 0.0], [cx1, cy0, 0.0], [cx1, cy0, cz1], [cx0, cy0, cz1]);
        add_quad([cx1, cy1, 0.0], [cx0, cy1, 0.0], [cx0, cy1, cz1], [cx1, cy1, cz1]);
        add_quad([cx0, cy1, 0.0], [cx0, cy0, 0.0], [cx0, cy0, cz1], [cx0, cy1, cz1]);
        add_quad([cx1, cy0, 0.0], [cx1, cy1, 0.0], [cx1, cy1, cz1], [cx1, cy0, cz1]);
        stl.push_str("endsolid sloped_deck_with_cabin\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_conformal_skin_deck_cabin.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write temporary STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected synthetic sloped deck + cabin STL import to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.25;
        cfg.slicing.layer_height.min_mm = 0.15;
        cfg.slicing.layer_height.max_mm = 0.30;
        cfg.slicing.layer_height.first_layer_mm = 0.25;
        cfg.slicing.layer_height.top_layer_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = false;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        // 1. Baseline without conformal skin
        let base_layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(base_layers.valid && !base_layers.layers.is_empty());

        // 2. Automatic Conformal SDF Skin mode (`enable_auto_conformal_skin = true`, 3 skin layers)
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.conformal_skin_min_oblique_deg = 8.0;
        cfg.slicing.conformal_skin_max_slope_deg = 60.0;
        crate::ui::sync_config_to_backend(&cfg);

        let auto_layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(
            auto_layers.valid && auto_layers.layers.len() >= base_layers.layers.len() + 3,
            "Expected automatic conformal skin mode to append at least 3 conformal SDF skin layers (base={}, auto={})",
            base_layers.layers.len(),
            auto_layers.layers.len()
        );

        // Identify the extracted conformal skin layers (`is_top_layer && min_thickness_mm == max_thickness_mm == top_layer_mm`)
        let skin_layers: Vec<&ffi::SelectedLayerInfo> = auto_layers
            .layers
            .iter()
            .filter(|l| {
                l.is_top_layer
                    && (l.max_thickness_mm - l.min_thickness_mm).abs() < 1.0e-5
                    && (l.mean_thickness_mm - cfg.slicing.layer_height.top_layer_mm).abs() < 1.0e-4
            })
            .collect();
        assert!(
            skin_layers.len() >= 3,
            "Expected at least 3 uniform-thickness conformal SDF skin layers, got {}",
            skin_layers.len()
        );

        // Verify that the sloped deck conformal skin layers are tilted (spanning z ~ 3.0..6.0 mm)
        // AND stand off around the central cabin protrusion (`|x| < 2.2 && |y| < 2.2` below `z = 7.0 mm`),
        // AND are scheduled causally BEFORE the upper cabin layers (`z > 7.5 mm`).
        let deck_skin_layer = skin_layers
            .iter()
            .copied()
            .find(|l| {
                let mut z_min = f32::INFINITY;
                let mut z_max = f32::NEG_INFINITY;
                for v in &l.vertices {
                    z_min = z_min.min(v.z);
                    z_max = z_max.max(v.z);
                }
                (z_max - z_min) > 1.2 && z_max < 7.0
            })
            .expect("Expected a sloped conformal skin layer spanning the 18° deck");

        for v in &deck_skin_layer.vertices {
            assert!(
                !(v.x.abs() < 2.2 && v.y.abs() < 2.2),
                "Expected sloped deck conformal skin layer to stand off around the rising cabin protrusion base, but found vertex at ({:.2}, {:.2}, {:.2})",
                v.x,
                v.y,
                v.z
            );
        }

        let upper_cabin_layer = auto_layers
            .layers
            .iter()
            .find(|l| {
                !l.vertices.is_empty()
                    && l.vertices.iter().all(|v| v.z > 7.2)
            })
            .expect("Expected upper cabin layers above z = 7.2 mm");
        assert!(
            deck_skin_layer.layer_index < upper_cabin_layer.layer_index,
            "Expected sloped deck conformal skin layer (#{}, iso={:.2}) to be scheduled causally BEFORE upper cabin protrusion layer (#{}, iso={:.2})",
            deck_skin_layer.layer_index,
            deck_skin_layer.isovalue,
            upper_cabin_layer.layer_index,
            upper_cabin_layer.isovalue
        );

        // 3. Manual ConformalSurfaceSkin hint (`kind = 2`, `skin_layer_count = 4`) with auto mode disabled
        cfg.slicing.enable_auto_conformal_skin = false;
        crate::ui::sync_config_to_backend(&cfg);
        let deck_normal = Vec3::new(-0.15, 0.0, 1.0).normalize();
        let manual_hints = vec![
            ffi::SpatialOrientationHint {
                kind: 2, // ConformalSurfaceSkin
                position_bed: [-6.0, 0.0, z_deck(-6.0)],
                direction_bed: [deck_normal.x, deck_normal.y, deck_normal.z],
                influence_radius_mm: 8.0,
                weight: 1.0,
                apply_on_surface_only: true,
                skin_layer_count: 4,
            },
            ffi::SpatialOrientationHint {
                kind: 2, // ConformalSurfaceSkin
                position_bed: [6.0, 0.0, z_deck(6.0)],
                direction_bed: [deck_normal.x, deck_normal.y, deck_normal.z],
                influence_radius_mm: 8.0,
                weight: 1.0,
                apply_on_surface_only: true,
                skin_layer_count: 4,
            },
        ];
        ffi::set_spatial_orientation_hints(&manual_hints);

        let manual_layers = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        let manual_skin_count = manual_layers
            .layers
            .iter()
            .filter(|l| {
                l.is_top_layer
                    && (l.max_thickness_mm - l.min_thickness_mm).abs() < 1.0e-5
                    && (l.mean_thickness_mm - cfg.slicing.layer_height.top_layer_mm).abs() < 1.0e-4
            })
            .count();
        assert_eq!(
            manual_skin_count, 4,
            "Expected manual ConformalSurfaceSkin hint with skin_layer_count=4 to extract exactly 4 conformal SDF skin layers"
        );

        // Verify that full 3D toolpaths (`OuterWall = 0`, `InnerWall = 1`, `SolidInfill = 2`) are generated
        // on the conformal SDF skin layers and never extend outside the object footprint.
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);
        let mut skin_has_outer_wall = false;
        let mut skin_has_solid_infill = false;
        for l_mesh in &manual_layers.layers {
            let is_skin = l_mesh.is_top_layer
                && (l_mesh.max_thickness_mm - l_mesh.min_thickness_mm).abs() < 1.0e-5
                && (l_mesh.mean_thickness_mm - cfg.slicing.layer_height.top_layer_mm).abs() < 1.0e-4;
            if is_skin {
                if let Some(tp_layer) = tp
                    .layers
                    .iter()
                    .find(|tl| tl.layer_index == l_mesh.layer_index)
                {
                    for p in &tp_layer.paths {
                        if p.feature_kind == 0 {
                            skin_has_outer_wall = true;
                        }
                        if p.feature_kind == 2 {
                            skin_has_solid_infill = true;
                        }
                        for pt in &p.points {
                            assert!(
                                pt.x.abs() <= 10.2 && pt.y.abs() <= 6.2,
                                "Conformal skin toolpath point ({:.2}, {:.2}, {:.2}) exceeded deck footprint",
                                pt.x,
                                pt.y,
                                pt.z
                            );
                        }
                    }
                }
            }
        }
        assert!(
            skin_has_outer_wall && skin_has_solid_infill,
            "Expected conformal SDF skin layers to produce both OuterWall and SolidInfill toolpaths"
        );

        ffi::set_spatial_orientation_hints(&[]);
        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_isolated_solid_layer_over_sparse_infill_anchoring_and_subskin_contour() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct a stepped block (`[-10, 10] x [-8, 8]`):
        // - Left deck (`x in [-10, 2]`): horizontal top roof at `z = 4.0 mm`
        // - Right tower (`x in [2, 10]`): continues up to `z = 8.0 mm`
        // With `shell_thickness_mm = 0.84` and `layer_height = 0.40 mm`:
        // - At `z ~ 2.0 mm`: pure sparse infill across `[-9.16, 9.16] x [-7.16, 7.16]`
        // - At `z ~ 2.8 mm` (1 layer beneath the `z = 3.2 mm` solid deck shell):
        //   Part 1 emits a Sub-Skin Boundary Anchor Contour in the sparse core beneath the upcoming deck.
        // - At `z ~ 3.2..3.6 mm` (first solid layer of the left deck over sparse infill, meeting the
        //   right tower's sparse core across the internal boundary at `x ~ 2.0 mm`):
        //   Part 2 suppresses floating inner loops and extends solid infill scanline endpoints across
        //   the internal solid/sparse boundary `x ~ 2.0 mm` to anchor onto underlying sparse infill ribs.
        let mut stl = String::from("solid stepped_deck_sparse_anchor\n");
        let mut add_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(p00, p10, p11);
            add_tri(p00, p11, p01);
        };

        // Watertight stepped block `[-10, 10] x [-8, 8]` split at `x = 2.0, z = 4.0` (zero T-junctions)
        // Bottom faces (z = 0)
        add_quad([-10.0, -8.0, 0.0], [-10.0, 8.0, 0.0], [2.0, 8.0, 0.0], [2.0, -8.0, 0.0]);
        add_quad([2.0, -8.0, 0.0], [2.0, 8.0, 0.0], [10.0, 8.0, 0.0], [10.0, -8.0, 0.0]);
        // Left deck top face (z = 4, x in [-10, 2]) and right tower top face (z = 8, x in [2, 10])
        add_quad([-10.0, -8.0, 4.0], [2.0, -8.0, 4.0], [2.0, 8.0, 4.0], [-10.0, 8.0, 4.0]);
        add_quad([2.0, -8.0, 8.0], [10.0, -8.0, 8.0], [10.0, 8.0, 8.0], [2.0, 8.0, 8.0]);
        // Front (-Y = -8) and Back (+Y = +8) faces
        add_quad([-10.0, -8.0, 0.0], [2.0, -8.0, 0.0], [2.0, -8.0, 4.0], [-10.0, -8.0, 4.0]);
        add_quad([2.0, -8.0, 0.0], [10.0, -8.0, 0.0], [10.0, -8.0, 4.0], [2.0, -8.0, 4.0]);
        add_quad([2.0, -8.0, 4.0], [10.0, -8.0, 4.0], [10.0, -8.0, 8.0], [2.0, -8.0, 8.0]);
        add_quad([2.0, 8.0, 0.0], [-10.0, 8.0, 0.0], [-10.0, 8.0, 4.0], [2.0, 8.0, 4.0]);
        add_quad([10.0, 8.0, 0.0], [2.0, 8.0, 0.0], [2.0, 8.0, 4.0], [10.0, 8.0, 4.0]);
        add_quad([10.0, 8.0, 4.0], [2.0, 8.0, 4.0], [2.0, 8.0, 8.0], [10.0, 8.0, 8.0]);
        // Left (-X = -10), Step Riser (X = 2, z in [4, 8]), and Right (+X = 10) faces
        add_quad([-10.0, 8.0, 0.0], [-10.0, -8.0, 0.0], [-10.0, -8.0, 4.0], [-10.0, 8.0, 4.0]);
        add_quad([2.0, 8.0, 4.0], [2.0, -8.0, 4.0], [2.0, -8.0, 8.0], [2.0, 8.0, 8.0]);
        add_quad([10.0, -8.0, 0.0], [10.0, 8.0, 0.0], [10.0, 8.0, 4.0], [10.0, -8.0, 4.0]);
        add_quad([10.0, -8.0, 4.0], [10.0, 8.0, 4.0], [10.0, 8.0, 8.0], [10.0, -8.0, 8.0]);
        stl.push_str("endsolid stepped_deck_sparse_anchor\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_stepped_deck_sparse_anchor.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write temporary STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected synthetic stepped deck STL import to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.35;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.40;
        cfg.slicing.layer_height.min_mm = 0.25;
        cfg.slicing.layer_height.max_mm = 0.50;
        cfg.slicing.layer_height.first_layer_mm = 0.40;
        cfg.slicing.layer_height.top_layer_mm = 0.40;
        cfg.slicing.enable_auto_conformal_skin = false;
        cfg.slicing.shell_thickness_mm = 0.84;
        cfg.slicing.inner_wall_count = 1;
        cfg.slicing.infill_density_percent = 20.0;
        cfg.slicing.sparse_infill_pattern = crate::config::SparseInfillPattern::Monotonic;
        cfg.slicing.solid_infill_pattern = crate::config::SolidInfillPattern::Concentric;
        cfg.slicing.infill_angle_deg = 0.0; // Scanlines along X/Y so deck scanlines cross x = 2.0 boundary
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let selected = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(selected.valid && !selected.layers.is_empty());

        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);

        // 1. Verify that on the layer immediately beneath the `z = 3.2 mm` solid deck shell
        //    (`z ~ 2.6..3.0 mm`), the 1-layer sub-skin boundary anchor contour increases total
        //    sparse infill extrusion length compared to the lower pure-sparse layer (`z ~ 1.6..2.0 mm`).
        let mid_sparse_layer = tp
            .layers
            .iter()
            .find(|l| (l.isovalue - 1.8).abs() < 0.25)
            .expect("Expected pure sparse layer near z = 1.8 mm");
        let sub_skin_layer = tp
            .layers
            .iter()
            .rev()
            .find(|l| l.isovalue <= 3.15 && l.isovalue >= 2.65)
            .expect("Expected sub-skin sparse layer immediately beneath z = 3.16 mm");

        let mid_sparse_len: f32 = mid_sparse_layer
            .paths
            .iter()
            .filter(|p| p.feature_kind == 4)
            .map(|p| p.total_length_mm)
            .sum();
        let sub_skin_sparse_len: f32 = sub_skin_layer
            .paths
            .iter()
            .filter(|p| p.feature_kind == 4)
            .map(|p| p.total_length_mm)
            .sum();
        assert!(
            sub_skin_sparse_len > mid_sparse_len + 4.0,
            "Expected 1-layer sub-skin boundary anchor contour to add sparse anchor length on sub-skin layer (mid={:.2} mm, sub_skin={:.2} mm)",
            mid_sparse_len,
            sub_skin_sparse_len
        );

        // 2. Verify that on the first solid deck transition layer (`z ~ 3.2..3.7 mm`), even though
        //    `Concentric` solid infill was configured, the over-sparse deck region uses open
        //    scanlines (`!p.is_closed`) rather than floating concentric loops in mid-air, and
        //    extends scanline endpoints across the inner shell boundary (`x = 2.0 - 0.84 = 1.16 mm`)
        //    into the right tower's sparse core to anchor onto underlying sparse infill ribs!
        let transition_layer = tp
            .layers
            .iter()
            .find(|l| {
                l.isovalue >= 3.15
                    && l.isovalue <= 3.75
                    && l.paths.iter().any(|p| p.feature_kind == 2)
            })
            .expect("Expected first solid deck transition layer with SolidInfill in z = 3.15..3.75 mm");
        let deck_solid_paths: Vec<_> = transition_layer
            .paths
            .iter()
            .filter(|p| p.feature_kind == 2)
            .collect();
        assert!(
            deck_solid_paths.iter().all(|p| !p.is_closed),
            "Expected solid infill over sparse infill on transition layer (iso={:.2}) to use open anchored scanlines instead of floating closed concentric loops",
            transition_layer.isovalue
        );
        let max_solid_x = deck_solid_paths
            .iter()
            .flat_map(|p| p.points.iter())
            .map(|pt| pt.x)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_solid_x > 1.20,
            "Expected deck solid infill scanlines on transition layer (iso={:.2}) to extend across the unanchored x=1.16 mm inner boundary to anchor on sparse ribs, got max_solid_x={:.2} mm",
            transition_layer.isovalue,
            max_solid_x
        );

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_surface_skin_3d_tool_clearance_collision_prevention_on_recessed_bow_deck() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct a Benchy-like recessed sloped bow deck (`x in [-2.0, 9.0], y in [-5.5, 5.5]`):
        // - Sloped bow deck: `z_deck(x) = 3.2 + 0.16 * (x + 2.0)` (from `z = 3.2 mm` at the cabin
        //   bulkhead `x = -2.0 mm` up to `z = 4.96 mm` at the bow `x = 9.0 mm`).
        // - Tall aft cabin block (`x in [-9.0, -2.0], y in [-5.5, 5.5], z in [0.0, 9.5 mm]`) rising
        //   directly above the bottom edge of the sloped deck (`x = -2.0 mm`).
        // - Port (`y in [-7.5, -5.5]`) and Starboard (`y in [5.5, 7.5]`) raised hull gunwale walls
        //   (`x in [-9.0, 9.0]`) rising `2.4 mm` above `z_deck(x)` (`z_gunwale(x) = z_deck(x) + 2.4`).
        //
        // When the conformal skin patch on the sloped bow deck is printed at `tau_completion` (~4.96 mm),
        // the cabin wall and both hull gunwale walls have ALREADY been printed up to `z = tau_completion`.
        // Reaching down to the bottom edge of the sloped deck (`x ~ -2.0 mm, z ~ 2.6..3.2 mm`) or close
        // to the port/starboard hull walls (`|y| ~ 5.5 mm`) would cause the nozzle/heater-block cone to
        // collide with the cabin and hull sides unless pruned by the 3D Tool Clearance Envelope check.
        let mut stl = String::from("solid benchy_recessed_bow_deck\n");
        let mut add_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(p00, p10, p11);
            add_tri(p00, p11, p01);
        };
        let mut add_sloped_box =
            |xa: f32, xb: f32, ya: f32, yb: f32, za0: f32, zb0: f32, za1: f32, zb1: f32| {
                add_quad([xa, ya, za0], [xa, yb, za0], [xb, yb, zb0], [xb, ya, zb0]); // bottom
                add_quad([xa, ya, za1], [xb, ya, zb1], [xb, yb, zb1], [xa, yb, za1]); // top
                add_quad([xa, ya, za0], [xb, ya, zb0], [xb, ya, zb1], [xa, ya, za1]); // -Y
                add_quad([xb, yb, zb0], [xa, yb, za0], [xa, yb, za1], [xb, yb, zb1]); // +Y
                add_quad([xa, yb, za0], [xa, ya, za0], [xa, ya, za1], [xa, yb, za1]); // -X
                add_quad([xb, ya, zb0], [xb, yb, zb0], [xb, yb, zb1], [xb, ya, zb1]); // +X
            };

        let z_deck = |x: f32| -> f32 { 3.2 + 0.16 * (x + 2.0) };
        // 1. Recessed sloped bow deck block (`x in [-2, 9], y in [-5.5, 5.5], z in [0, z_deck(x)]`)
        add_sloped_box(-2.0, 9.0, -5.5, 5.5, 0.0, 0.0, z_deck(-2.0), z_deck(9.0));
        // 2. Tall aft cabin block (`x in [-9, -2], y in [-5.5, 5.5], z in [0, 9.5]`)
        add_sloped_box(-9.0, -2.0, -5.5, 5.5, 0.0, 0.0, 9.5, 9.5);
        // 3. Port hull gunwale wall (`x in [-9, 9], y in [-7.5, -5.5], z in [0, z_deck(x) + 2.4]`)
        add_sloped_box(
            -9.0,
            9.0,
            -7.5,
            -5.5,
            0.0,
            0.0,
            z_deck(-9.0) + 2.4,
            z_deck(9.0) + 2.4,
        );
        // 4. Starboard hull gunwale wall (`x in [-9, 9], y in [5.5, 7.5], z in [0, z_deck(x) + 2.4]`)
        add_sloped_box(
            -9.0,
            9.0,
            5.5,
            7.5,
            0.0,
            0.0,
            z_deck(-9.0) + 2.4,
            z_deck(9.0) + 2.4,
        );
        stl.push_str("endsolid benchy_recessed_bow_deck\n");

        let tmp_path =
            std::env::temp_dir().join("manifold_test_benchy_recessed_bow_deck_clearance.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write temporary STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected synthetic Benchy recessed bow deck STL import to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.28;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.25;
        cfg.slicing.layer_height.min_mm = 0.15;
        cfg.slicing.layer_height.max_mm = 0.30;
        cfg.slicing.layer_height.first_layer_mm = 0.25;
        cfg.slicing.layer_height.top_layer_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.conformal_skin_min_oblique_deg = 6.0;
        cfg.slicing.conformal_skin_max_slope_deg = 60.0;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let selected = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(selected.valid && !selected.layers.is_empty());

        // Find conformal SDF skin layers located on the recessed bow deck (`x > -2.0, |y| < 5.5, z < 5.2`)
        let bow_deck_skin_layers: Vec<&ffi::SelectedLayerInfo> = selected
            .layers
            .iter()
            .filter(|l| {
                l.is_top_layer
                    && (l.max_thickness_mm - l.min_thickness_mm).abs() < 1.0e-5
                    && (l.mean_thickness_mm - cfg.slicing.layer_height.top_layer_mm).abs() < 1.0e-4
                    && l.vertices
                        .iter()
                        .any(|v| v.x > -1.5 && v.y.abs() < 4.5 && v.z < 5.1)
            })
            .collect();
        assert!(
            bow_deck_skin_layers.len() >= 3,
            "Expected at least 3 conformal SDF skin layers on the safe center of the sloped bow deck, got {}",
            bow_deck_skin_layers.len()
        );

        // Verify that every vertex on the recessed bow deck conformal skin layers stands off from:
        // 1. The aft cabin wall at `x = -2.0 mm` (which is already printed up to `tau_completion ~ 4.96 mm`,
        //    towering ~1.7..2.3 mm above the bottom edge of the deck where the tool clearance radius is >= 2.0 mm), and
        // 2. The port and starboard hull gunwale walls at `|y| = 5.5 mm` (which rise up to `min(z_deck(x)+2.4, tau_completion)`
        //    above the recessed deck).
        for layer in &bow_deck_skin_layers {
            for v in &layer.vertices {
                if v.y.abs() < 5.2 && v.z < 5.1 {
                    let dist_to_cabin_wall = v.x - (-2.0);
                    let dist_to_hull_gunwale = 5.5 - v.y.abs();
                    assert!(
                        dist_to_cabin_wall >= 1.65,
                        "Bow deck conformal skin vertex ({:.2}, {:.2}, {:.2}) approached within {:.2} mm of the tall cabin wall (x=-2.0), risking nozzle/cone collision!",
                        v.x,
                        v.y,
                        v.z,
                        dist_to_cabin_wall
                    );
                    assert!(
                        dist_to_hull_gunwale >= 1.25,
                        "Bow deck conformal skin vertex ({:.2}, {:.2}, {:.2}) approached within {:.2} mm of the raised hull gunwale wall (|y|=5.5), risking nozzle/cone collision!",
                        v.x,
                        v.y,
                        v.z,
                        dist_to_hull_gunwale
                    );
                }
            }
        }

        // Also generate full 3D shell toolpaths and verify zero conformal skin toolpath points collide
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);
        for layer in &bow_deck_skin_layers {
            if let Some(tp_layer) = tp
                .layers
                .iter()
                .find(|tl| tl.layer_index == layer.layer_index)
            {
                for path in &tp_layer.paths {
                    for pt in &path.points {
                        if pt.y.abs() < 5.2 && pt.z < 5.1 {
                            let dist_to_cabin_wall = pt.x - (-2.0);
                            let dist_to_hull_gunwale = 5.5 - pt.y.abs();
                            assert!(
                                dist_to_cabin_wall >= 1.65 && dist_to_hull_gunwale >= 1.25,
                                "Bow deck conformal skin toolpath point ({:.2}, {:.2}, {:.2}) violated 3D nozzle clearance from cabin (dx={:.2}) or hull sides (dy={:.2})!",
                                pt.x,
                                pt.y,
                                pt.z,
                                dist_to_cabin_wall,
                                dist_to_hull_gunwale
                            );
                        }
                    }
                }
            }
        }

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_skin_no_missing_perimeters_below_outermost_surface() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Watertight 24 x 24 crowned block: vertical side walls with domed top z(x, y)
        let nx = 24usize;
        let ny = 24usize;
        let z_top = |x: f32, y: f32| -> f32 {
            12.0 + 1.8 * (std::f32::consts::PI * x / 24.0).cos() * (std::f32::consts::PI * y / 24.0).cos()
        };
        let mut stl = String::from("solid crowned_block\n");
        let mut add_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(p00, p10, p11);
            add_tri(p00, p11, p01);
        };

        for i in 0..nx {
            let x0 = -12.0 + 24.0 * (i as f32 / nx as f32);
            let x1 = -12.0 + 24.0 * ((i + 1) as f32 / nx as f32);
            for j in 0..ny {
                let y0 = -12.0 + 24.0 * (j as f32 / ny as f32);
                let y1 = -12.0 + 24.0 * ((j + 1) as f32 / ny as f32);
                add_quad([x0, y0, 0.0], [x0, y1, 0.0], [x1, y1, 0.0], [x1, y0, 0.0]);
                add_quad(
                    [x0, y0, z_top(x0, y0)],
                    [x1, y0, z_top(x1, y0)],
                    [x1, y1, z_top(x1, y1)],
                    [x0, y1, z_top(x0, y1)],
                );
            }
        }
        for i in 0..nx {
            let x0 = -12.0 + 24.0 * (i as f32 / nx as f32);
            let x1 = -12.0 + 24.0 * ((i + 1) as f32 / nx as f32);
            add_quad([x0, -12.0, 0.0], [x1, -12.0, 0.0], [x1, -12.0, z_top(x1, -12.0)], [x0, -12.0, z_top(x0, -12.0)]);
            add_quad([x1, 12.0, 0.0], [x0, 12.0, 0.0], [x0, 12.0, z_top(x0, 12.0)], [x1, 12.0, z_top(x1, 12.0)]);
        }
        for j in 0..ny {
            let y0 = -12.0 + 24.0 * (j as f32 / ny as f32);
            let y1 = -12.0 + 24.0 * ((j + 1) as f32 / ny as f32);
            add_quad([-12.0, y1, 0.0], [-12.0, y0, 0.0], [-12.0, y0, z_top(-12.0, y0)], [-12.0, y1, z_top(-12.0, y1)]);
            add_quad([12.0, y0, 0.0], [12.0, y1, 0.0], [12.0, y1, z_top(12.0, y1)], [12.0, y0, z_top(12.0, y0)]);
        }
        stl.push_str("endsolid crowned_block\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_conformal_crowned_block.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write crowned block STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path).expect("Expected crowned block STL load");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::ConstrainedNonPlanar;
        cfg.eikonal.voxel_size_mm = 0.35;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.25;
        cfg.slicing.layer_height.min_mm = 0.14;
        cfg.slicing.layer_height.max_mm = 0.36;
        cfg.slicing.layer_height.first_layer_mm = 0.25;
        cfg.slicing.layer_height.top_layer_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid);
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);

        let mut skin_layers: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                l.is_top_layer
                    && !l.is_first_layer
                    && (l.max_thickness_mm - l.min_thickness_mm).abs() < 1.0e-4
                    && (l.mean_thickness_mm - 0.20).abs() < 1.0e-3
                    && l.branch_layer_count == 3
                    && l.vertices.len() >= 2000
            })
            .collect();
        skin_layers.sort_by_key(|l| l.branch_layer_index);
        assert_eq!(
            skin_layers.len(),
            3,
            "Expected 3 conformal skin layers on crowned block, got {}",
            skin_layers.len()
        );

        let mut outer_bounds = [(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32); 3];
        for (idx, layer) in skin_layers.iter().enumerate() {
            assert_eq!(layer.branch_layer_index, idx as u32);
            let tp_layer = tp
                .layers
                .iter()
                .find(|tl| tl.layer_index == layer.layer_index)
                .expect("Missing toolpath layer for conformal skin layer");

            let outer_walls: Vec<&ffi::ExtrusionPath> = tp_layer
                .paths
                .iter()
                .filter(|p| p.feature_kind == 0 && p.is_closed)
                .collect();
            assert!(
                !outer_walls.is_empty(),
                "Conformal skin layer m={} missing closed OuterWall perimeter",
                idx + 1
            );
            if idx < 2 {
                let inner_walls_cnt = tp_layer.paths.iter().filter(|p| p.feature_kind == 1).count();
                assert!(
                    inner_walls_cnt > 0,
                    "Deeper conformal skin layer m={} missing InnerWall perimeters",
                    idx + 1
                );
            }

            let mut min_x = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for p in outer_walls {
                for pt in &p.points {
                    min_x = min_x.min(pt.x);
                    max_x = max_x.max(pt.x);
                    min_y = min_y.min(pt.y);
                    max_y = max_y.max(pt.y);
                }
            }
            outer_bounds[idx] = (min_x, max_x, min_y, max_y);
        }

        let (ref_min_x, ref_max_x, ref_min_y, ref_max_y) = outer_bounds[2];
        for m in 0..2 {
            let (mx0, mx1, my0, my1) = outer_bounds[m];
            assert!(
                (mx0 - ref_min_x).abs() <= 0.35
                    && (mx1 - ref_max_x).abs() <= 0.35
                    && (my0 - ref_min_y).abs() <= 0.35
                    && (my1 - ref_max_y).abs() <= 0.35,
                "Inner skin layer m={} OuterWall bounds {:?} retreated away from outermost m=3 bounds {:?}",
                m + 1,
                outer_bounds[m],
                outer_bounds[2]
            );
        }

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_skin_sloped_roof_with_hollow_smokestack_boss_smooth_and_deferred() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Single watertight non-self-intersecting polar + corner-snapped roof with hollow smokestack
        let cx = -4.0_f32;
        let cy = 0.0_f32;
        let r_in = 2.0_f32;
        let r_out = 3.5_f32;
        let z_bore_floor = 6.0_f32;
        let z_stack_top = 20.0_f32;
        let z_roof = |x: f32, y: f32| -> f32 { 10.0 + 0.12 * (x + 12.0) - 0.015 * y * y };

        let mut raw_angles = Vec::with_capacity(52);
        for k in 0..48 {
            raw_angles.push((k as f32) * (2.0 * std::f32::consts::PI / 48.0));
        }
        for &(bx, by) in &[(-12.0_f32, -12.0_f32), (12.0, -12.0), (12.0, 12.0), (-12.0, 12.0)] {
            raw_angles.push((by - cy).atan2(bx - cx).rem_euclid(2.0 * std::f32::consts::PI));
        }
        raw_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut thetas = Vec::with_capacity(raw_angles.len());
        for th in raw_angles {
            if th < 2.0 * std::f32::consts::PI - 1.0e-4
                && thetas
                    .last()
                    .map_or(true, |&prev: &f32| (th - prev).abs() > 1.0e-4)
            {
                thetas.push(th);
            }
        }
        let n_ang = thetas.len();
        let n_rings = 8usize;

        let mut stl = String::from("solid sloped_roof_smokestack\n");
        let add_tri = |stl: &mut String, a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let add_quad = |stl: &mut String, p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(stl, p00, p10, p11);
            add_tri(stl, p00, p11, p01);
        };

        let box_ray_dist = |dx: f32, dy: f32| -> f32 {
            let tx = if dx > 1.0e-6 {
                (12.0 - cx) / dx
            } else if dx < -1.0e-6 {
                (-12.0 - cx) / dx
            } else {
                f32::INFINITY
            };
            let ty = if dy > 1.0e-6 {
                (12.0 - cy) / dy
            } else if dy < -1.0e-6 {
                (-12.0 - cy) / dy
            } else {
                f32::INFINITY
            };
            tx.min(ty)
        };

        let ring_xy = |theta: f32, r_frac: f32| -> (f32, f32) {
            let dx = theta.cos();
            let dy = theta.sin();
            let r_box = box_ray_dist(dx, dy);
            let r = r_out + r_frac * (r_box - r_out);
            (
                (cx + r * dx).clamp(-12.0, 12.0),
                (cy + r * dy).clamp(-12.0, 12.0),
            )
        };

        for a in 0..n_ang {
            let th0 = thetas[a];
            let th1 = thetas[(a + 1) % n_ang];
            let (dx0, dy0) = (th0.cos(), th0.sin());
            let (dx1, dy1) = (th1.cos(), th1.sin());

            // Roof radial rings
            for ri in 0..n_rings {
                let f0 = ri as f32 / n_rings as f32;
                let f1 = (ri + 1) as f32 / n_rings as f32;
                let (x00, y00) = ring_xy(th0, f0);
                let (x10, y10) = ring_xy(th0, f1);
                let (x11, y11) = ring_xy(th1, f1);
                let (x01, y01) = ring_xy(th1, f0);
                add_quad(
                    &mut stl,
                    [x00, y00, z_roof(x00, y00)],
                    [x10, y10, z_roof(x10, y10)],
                    [x11, y11, z_roof(x11, y11)],
                    [x01, y01, z_roof(x01, y01)],
                );
            }

            // Outer box side walls and bottom fan
            let (bx0, by0) = ring_xy(th0, 1.0);
            let (bx1, by1) = ring_xy(th1, 1.0);
            add_tri(&mut stl, [cx, cy, 0.0], [bx1, by1, 0.0], [bx0, by0, 0.0]);
            add_quad(
                &mut stl,
                [bx0, by0, 0.0],
                [bx1, by1, 0.0],
                [bx1, by1, z_roof(bx1, by1)],
                [bx0, by0, z_roof(bx0, by0)],
            );

            // Outer smokestack wall, top annulus, inner bore wall, and bore floor
            let (sx0, sy0) = (cx + r_out * dx0, cy + r_out * dy0);
            let (sx1, sy1) = (cx + r_out * dx1, cy + r_out * dy1);
            let (ix0, iy0) = (cx + r_in * dx0, cy + r_in * dy0);
            let (ix1, iy1) = (cx + r_in * dx1, cy + r_in * dy1);
            add_quad(
                &mut stl,
                [sx0, sy0, z_roof(sx0, sy0)],
                [sx1, sy1, z_roof(sx1, sy1)],
                [sx1, sy1, z_stack_top],
                [sx0, sy0, z_stack_top],
            );
            add_quad(
                &mut stl,
                [ix0, iy0, z_stack_top],
                [sx0, sy0, z_stack_top],
                [sx1, sy1, z_stack_top],
                [ix1, iy1, z_stack_top],
            );
            add_quad(
                &mut stl,
                [ix1, iy1, z_bore_floor],
                [ix0, iy0, z_bore_floor],
                [ix0, iy0, z_stack_top],
                [ix1, iy1, z_stack_top],
            );
            add_tri(&mut stl, [cx, cy, z_bore_floor], [ix0, iy0, z_bore_floor], [ix1, iy1, z_bore_floor]);
        }
        stl.push_str("endsolid sloped_roof_smokestack\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_sloped_roof_smokestack.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write smokestack roof STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected sloped roof smokestack STL load");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::ConstrainedNonPlanar;
        cfg.eikonal.voxel_size_mm = 0.32;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.25;
        cfg.slicing.layer_height.min_mm = 0.14;
        cfg.slicing.layer_height.max_mm = 0.36;
        cfg.slicing.layer_height.first_layer_mm = 0.25;
        cfg.slicing.layer_height.top_layer_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid);
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);

        let z_centroid_of = |l: &ffi::SelectedLayerInfo| -> f32 {
            if l.vertices.is_empty() {
                return 0.0;
            }
            let sum: f32 = l.vertices.iter().map(|v| v.z).sum();
            sum / (l.vertices.len() as f32)
        };

        let mut roof_skin_layers: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                let zc = z_centroid_of(l);
                l.is_top_layer
                    && !l.is_first_layer
                    && (l.mean_thickness_mm - 0.20).abs() < 1.0e-3
                    && (9.0..=13.5).contains(&zc)
            })
            .collect();
        assert!(
            roof_skin_layers.len() >= 3,
            "Expected >= 3 roof conformal skin layers, got {}",
            roof_skin_layers.len()
        );
        roof_skin_layers.sort_by_key(|l| (l.branch_layer_index, l.layer_index));
        let max_roof_skin_layer_index = roof_skin_layers
            .iter()
            .map(|l| l.layer_index)
            .max()
            .unwrap();

        let upper_stack_layers: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| z_centroid_of(l) > 14.0)
            .collect();
        assert!(
            !upper_stack_layers.is_empty(),
            "Expected upper smokestack layers above z = 14.0 mm"
        );
        for stack_layer in &upper_stack_layers {
            assert!(
                stack_layer.layer_index > max_roof_skin_layer_index,
                "Expected upper smokestack layer (#{}, zc={:.2}) to be causally deferred after roof skin layer (#{max_roof_skin_layer_index})",
                stack_layer.layer_index,
                z_centroid_of(stack_layer)
            );
        }

        let outermost_roof_skin = roof_skin_layers.last().copied().unwrap();
        let tp_outer = tp
            .layers
            .iter()
            .find(|tl| tl.layer_index == outermost_roof_skin.layer_index)
            .expect("Missing toolpath layer for outermost roof conformal skin");
        let outer_walls: Vec<&ffi::ExtrusionPath> = tp_outer
            .paths
            .iter()
            .filter(|p| p.feature_kind == 0)
            .collect();
        let outer_wall_len: f32 = outer_walls.iter().map(|p| p.total_length_mm).sum();
        assert!(
            (1..=2).contains(&outer_walls.len()) && outer_wall_len < 140.0,
            "Expected smooth non-fragmented collar on outermost roof skin layer (count={}, len={:.2} mm)",
            outer_walls.len(),
            outer_wall_len
        );

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    fn count_benchy_roof_side_bins(
        verts: &[ffi::RenderVertex],
        y_lo: f32,
        y_hi: f32,
    ) -> usize {
        (0..17usize)
            .filter(|&bx| {
                let x_lo = -7.0_f32 + bx as f32;
                let x_hi = x_lo + 1.0_f32;
                verts
                    .iter()
                    .any(|v| v.x >= x_lo && v.x < x_hi && v.y >= y_lo && v.y <= y_hi)
            })
            .count()
    }

    #[test]
    fn test_conformal_skin_benchy_5axis_roof_smooth_and_smokestack_deferred() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path_3mf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/benchy-5axis.3mf");
        if !path_3mf.exists() {
            return;
        }

        let report = crate::io::load_files_into_backend(&[path_3mf]);
        let mut cfg = report.restored_config.expect("Expected config in benchy-5axis.3mf");
        let obj = &report.objects[0];
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid && !iso.layers.is_empty());
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);
        assert_eq!(iso.layers.len(), tp.layers.len());

        let z_stats = |l: &ffi::SelectedLayerInfo| -> (f32, f32, f32) {
            let mut zmin = f32::INFINITY;
            let mut zmax = f32::NEG_INFINITY;
            let mut sum = 0.0_f32;
            for v in &l.vertices {
                zmin = zmin.min(v.z);
                zmax = zmax.max(v.z);
                sum += v.z;
            }
            let mean = if l.vertices.is_empty() { 0.0 } else { sum / l.vertices.len() as f32 };
            (zmin, zmax, mean)
        };

        let top_h = cfg.slicing.layer_height.top_layer_mm;
        let mut roof_skins: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                let (_, _, zc) = z_stats(l);
                l.is_top_layer
                    && !l.is_first_layer
                    && (l.mean_thickness_mm - top_h).abs() < 1.0e-3
                    && (36.5..=40.8).contains(&zc)
            })
            .collect();
        assert!(
            roof_skins.len() >= 3,
            "Expected >= 3 cabin roof conformal skin layers on benchy-5axis.3mf, got {}",
            roof_skins.len()
        );
        roof_skins.sort_by_key(|l| (l.branch_layer_index, l.layer_index));
        let max_roof_idx = roof_skins.iter().map(|l| l.layer_index).max().unwrap();

        let upper_stack: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| !l.vertices.is_empty() && z_stats(l).0 > 41.5)
            .collect();
        assert!(
            !upper_stack.is_empty(),
            "Expected upper smokestack layers with z_min > 41.5 mm on benchy-5axis.3mf"
        );
        for l in &upper_stack {
            assert!(
                l.layer_index > max_roof_idx,
                "Benchy smokestack layer #{} (z_min={:.2}) was not deferred after cabin roof skin #{}",
                l.layer_index,
                z_stats(l).0,
                max_roof_idx
            );
        }

        let outermost_roof = roof_skins.last().copied().unwrap();
        let tp_roof = tp
            .layers
            .iter()
            .find(|tl| tl.layer_index == outermost_roof.layer_index)
            .unwrap();
        let ow_cnt = tp_roof.paths.iter().filter(|p| p.feature_kind == 0).count();

        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let mut has_behind_chimney = false;
        let mut has_left_of_chimney = false;
        let mut has_right_of_chimney = false;
        let mut has_front_of_chimney = false;
        let mut inside_chimney_bore = 0usize;
        let mut min_r_to_chimney = f32::INFINITY;
        for v in &outermost_roof.vertices {
            x_min = x_min.min(v.x);
            x_max = x_max.max(v.x);
            y_min = y_min.min(v.y);
            y_max = y_max.max(v.y);
            let r_chim = (v.x + 4.82_f32).hypot(v.y);
            min_r_to_chimney = min_r_to_chimney.min(r_chim);
            if r_chim < 2.5_f32 {
                inside_chimney_bore += 1;
            }
            if v.x < -8.0 && v.y.abs() < 4.5 {
                has_behind_chimney = true;
            }
            if (-7.8..=-1.8).contains(&v.x) && (3.3..=5.0).contains(&v.y) {
                has_left_of_chimney = true;
            }
            if (-7.8..=-1.8).contains(&v.x) && (-5.0..=-3.3).contains(&v.y) {
                has_right_of_chimney = true;
            }
            if (-1.4..=0.5).contains(&v.x) && v.y.abs() < 2.5 {
                has_front_of_chimney = true;
            }
        }
        println!(
            "Benchy 5-axis roof skin: layers={}, max_roof_idx={}, stack_min_idx={}, outermost_roof OW={}, X=[{:.2},{:.2}], Y=[{:.2},{:.2}], min_r_chim={:.2}, bore_verts={}, quadrants(rear={}, L={}, R={}, front={})",
            roof_skins.len(),
            max_roof_idx,
            upper_stack.iter().map(|l| l.layer_index).min().unwrap(),
            ow_cnt,
            x_min,
            x_max,
            y_min,
            y_max,
            min_r_to_chimney,
            inside_chimney_bore,
            has_behind_chimney,
            has_left_of_chimney,
            has_right_of_chimney,
            has_front_of_chimney
        );

        // Log other conformal patches (rear deck, bow deck) for diagnostic inspection
        for l in &iso.layers {
            if l.is_top_layer && !l.is_first_layer && (l.mean_thickness_mm - top_h).abs() < 1.0e-3 {
                let (_, _, zc) = z_stats(l);
                if !(36.5..=40.8).contains(&zc) && !(46.5..=48.5).contains(&zc) {
                    let xmin = l.vertices.iter().map(|v| v.x).fold(f32::INFINITY, f32::min);
                    let xmax = l.vertices.iter().map(|v| v.x).fold(f32::NEG_INFINITY, f32::max);
                    println!(
                        "  other skin layer #{} (b={}, m={}/{}): zc={:.2}, X=[{:.2},{:.2}], verts={}",
                        l.layer_index,
                        l.branch_id,
                        l.branch_layer_index + 1,
                        l.branch_layer_count,
                        zc,
                        xmin,
                        xmax,
                        l.vertices.len()
                    );
                }
            }
        }

        assert!(
            (1..=4).contains(&ow_cnt),
            "Expected smooth non-fragmented OuterWall loops (1..=4) on Benchy cabin roof skin, got {ow_cnt}"
        );
        assert!(
            x_min <= -8.5 && x_max >= 12.0,
            "Expected full Benchy cabin roof conformal cap spanning [-8.5, 12.0], got [{x_min:.2}, {x_max:.2}]"
        );
        assert!(
            has_behind_chimney && has_left_of_chimney && has_right_of_chimney && has_front_of_chimney,
            "Expected Benchy cabin roof conformal cap to wrap completely around the chimney (rear={has_behind_chimney}, L={has_left_of_chimney}, R={has_right_of_chimney}, front={has_front_of_chimney})"
        );
        assert!(
            min_r_to_chimney <= 4.15,
            "Expected Benchy roof conformal cap to reach right up to the r=3.23 mm chimney wall, got min_r={min_r_to_chimney:.2}"
        );
        assert_eq!(
            inside_chimney_bore, 0,
            "Expected a clean hole for the chimney (zero roof skin vertices inside r < 2.5 mm), got {inside_chimney_bore}"
        );

        let all_chimney_above_roof: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                !l.vertices.is_empty()
                    && z_stats(l).0 > 39.2
                    && l.vertices.iter().all(|v| (v.x + 4.82_f32).hypot(v.y) < 4.3)
            })
            .collect();
        assert!(!all_chimney_above_roof.is_empty());
        for l in &all_chimney_above_roof {
            assert!(
                l.layer_index > max_roof_idx,
                "Benchy smokestack base/shaft layer #{} (z_min={:.2}) was not deferred after cabin roof skin #{}",
                l.layer_index,
                z_stats(l).0,
                max_roof_idx
            );
        }

        let left_bins = count_benchy_roof_side_bins(&outermost_roof.vertices, -8.5, -7.2);
        let right_bins = count_benchy_roof_side_bins(&outermost_roof.vertices, 7.2, 8.5);
        assert!(
            left_bins >= 16 && right_bins >= 16,
            "Expected >= 16/17 (95%) 1mm X bins covered on both Benchy roof side strips, got L={left_bins}/17, R={right_bins}/17"
        );

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_skin_benchy_3axis_a4t_full_roof_cap_no_side_notches_or_rear_cutout() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let path_3mf = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/benchy-5axis.3mf");
        if !path_3mf.exists() {
            return;
        }

        let report = crate::io::load_files_into_backend(&[path_3mf]);
        let mut cfg = report.restored_config.expect("Expected config in benchy-5axis.3mf");
        let obj = &report.objects[0];
        cfg.machine.kinematics_mode = crate::config::FiveAxisKinematicsMode::ThreeAxis;
        cfg.tools[0].has_rotational_axes = false;
        cfg.tools[0].clearance_mode = crate::config::ClearanceProfileMode::PiecewiseXZ;
        cfg.tools[0].clearance_profile_xz = vec![
            crate::config::ClearancePointXZ::new(0.5, 0.0),
            crate::config::ClearancePointXZ::new(4.0, 1.4),
            crate::config::ClearancePointXZ::new(9.0, 6.0),
            crate::config::ClearancePointXZ::new(32.0, 9.0),
            crate::config::ClearancePointXZ::new(42.0, 40.0),
        ];
        cfg.eikonal.auto_voxel_size = false;
        cfg.eikonal.voxel_size_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.one_perimeter_on_conformal_patches = true;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid && !iso.layers.is_empty());
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);
        assert_eq!(iso.layers.len(), tp.layers.len());

        let z_min_of = |l: &ffi::SelectedLayerInfo| -> f32 {
            l.vertices.iter().map(|v| v.z).fold(f32::INFINITY, f32::min)
        };
        let z_mean_of = |l: &ffi::SelectedLayerInfo| -> f32 {
            if l.vertices.is_empty() {
                0.0
            } else {
                l.vertices.iter().map(|v| v.z).sum::<f32>() / l.vertices.len() as f32
            }
        };

        let top_h = cfg.slicing.layer_height.top_layer_mm;
        let mut roof_skins: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                let zc = z_mean_of(l);
                l.is_top_layer
                    && !l.is_first_layer
                    && (l.mean_thickness_mm - top_h).abs() < 1.0e-3
                    && (36.5..=40.8).contains(&zc)
            })
            .collect();
        assert!(
            roof_skins.len() >= 3,
            "Expected >= 3 cabin roof conformal skin layers on 3-axis A4T Benchy, got {}",
            roof_skins.len()
        );
        roof_skins.sort_by_key(|l| (l.branch_layer_index, l.layer_index));
        let max_roof_idx = roof_skins.iter().map(|l| l.layer_index).max().unwrap();

        let outermost_roof = roof_skins.last().copied().unwrap();
        let tp_roof = tp
            .layers
            .iter()
            .find(|tl| tl.layer_index == outermost_roof.layer_index)
            .unwrap();
        let ow_cnt = tp_roof.paths.iter().filter(|p| p.feature_kind == 0).count();
        assert!(
            (1..=4).contains(&ow_cnt),
            "Expected 1..=4 OuterWall loops on 3-axis Benchy roof skin, got {ow_cnt}"
        );

        let mut x_min = f32::INFINITY;
        let mut x_max = f32::NEG_INFINITY;
        let mut y_min = f32::INFINITY;
        let mut y_max = f32::NEG_INFINITY;
        let mut min_r_quad = [f32::INFINITY; 4]; // 0=rear, 1=left(+Y), 2=right(-Y), 3=front
        let mut has_quad = [false; 4];
        let mut bore_verts = 0usize;

        for v in &outermost_roof.vertices {
            x_min = x_min.min(v.x);
            x_max = x_max.max(v.x);
            y_min = y_min.min(v.y);
            y_max = y_max.max(v.y);
            let r_chim = (v.x + 4.82_f32).hypot(v.y);
            if r_chim < 2.4 {
                bore_verts += 1;
            }
            if v.x < -8.0 && v.y.abs() < 2.5 {
                has_quad[0] = true;
                min_r_quad[0] = min_r_quad[0].min(r_chim);
            }
            if (-7.5..=-2.0).contains(&v.x) && (2.9..=4.2).contains(&v.y) {
                has_quad[1] = true;
                min_r_quad[1] = min_r_quad[1].min(r_chim);
            }
            if (-7.5..=-2.0).contains(&v.x) && (-4.2..=-2.9).contains(&v.y) {
                has_quad[2] = true;
                min_r_quad[2] = min_r_quad[2].min(r_chim);
            }
            if (-1.6..=0.0).contains(&v.x) && v.y.abs() < 2.5 {
                has_quad[3] = true;
                min_r_quad[3] = min_r_quad[3].min(r_chim);
            }
        }
        println!(
            "Benchy 3-axis A4T roof skin: layers={}, OW={}, X=[{:.2},{:.2}], Y=[{:.2},{:.2}], quad_min_r={:?}, bore_verts={}",
            roof_skins.len(),
            ow_cnt,
            x_min,
            x_max,
            y_min,
            y_max,
            min_r_quad,
            bore_verts
        );
        assert!(
            x_min <= -8.8 && x_max >= 12.2 && y_min <= -9.0 && y_max >= 9.0,
            "Expected full 3-axis Benchy roof extents X<=-8.8, X>=12.2, Y<=-9.0, Y>=9.0, got X=[{x_min:.2},{x_max:.2}], Y=[{y_min:.2},{y_max:.2}]"
        );
        for (q_idx, q_name) in ["rear", "left", "right", "front"].iter().enumerate() {
            assert!(
                has_quad[q_idx] && min_r_quad[q_idx] <= 3.9,
                "Expected 3-axis Benchy roof skin to wrap tightly around chimney in {q_name} quadrant (min_r <= 3.9), got has={}, min_r={:.2}",
                has_quad[q_idx],
                min_r_quad[q_idx]
            );
        }
        assert_eq!(bore_verts, 0, "Expected 0 roof skin vertices inside chimney bore r < 2.4 mm");

        let left_bins = count_benchy_roof_side_bins(&outermost_roof.vertices, -8.5, -7.2);
        let right_bins = count_benchy_roof_side_bins(&outermost_roof.vertices, 7.2, 8.5);
        assert!(
            left_bins >= 16 && right_bins >= 16,
            "Expected >= 16/17 (95%) 1mm X bins covered on both 3-axis Benchy roof side strips, got L={left_bins}/17, R={right_bins}/17"
        );

        let all_chimney_above_roof: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                !l.vertices.is_empty()
                    && z_min_of(l) > 39.2
                    && l.vertices.iter().all(|v| (v.x + 4.82_f32).hypot(v.y) < 4.3)
            })
            .collect();
        assert!(!all_chimney_above_roof.is_empty());
        for l in &all_chimney_above_roof {
            assert!(
                l.layer_index > max_roof_idx,
                "3-axis Benchy smokestack layer #{} (z_min={:.2}) was not deferred after cabin roof skin #{}",
                l.layer_index,
                z_min_of(l),
                max_roof_idx
            );
        }

        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_conformal_skin_flat_surface_with_single_boss_full_cap_and_hole() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Watertight flat deck [-12, 12] x [-12, 12], z in [0, 10] with a central vertical
        // cylindrical boss (r = 3.5 mm, z in [10, 20]), guaranteeing 1 single Reeb-graph branch.
        let r_boss = 3.5_f32;
        let z_deck = 10.0_f32;
        let z_boss_top = 20.0_f32;
        let half_xy = 12.0_f32;

        let mut raw_angles = Vec::with_capacity(52);
        for k in 0..48 {
            raw_angles.push((k as f32) * (2.0 * std::f32::consts::PI / 48.0));
        }
        for &(bx, by) in &[
            (half_xy, half_xy),
            (-half_xy, half_xy),
            (-half_xy, -half_xy),
            (half_xy, -half_xy),
        ] {
            raw_angles.push(by.atan2(bx).rem_euclid(2.0 * std::f32::consts::PI));
        }
        raw_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut thetas = Vec::with_capacity(raw_angles.len());
        for th in raw_angles {
            if th < 2.0 * std::f32::consts::PI - 1.0e-4
                && thetas
                    .last()
                    .map_or(true, |&prev: &f32| (th - prev).abs() > 1.0e-4)
            {
                thetas.push(th);
            }
        }
        let n_ang = thetas.len();
        let mut stl = String::from("solid flat_surface_with_single_boss\n");
        let add_tri = |stl: &mut String, a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let add_quad = |stl: &mut String, p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(stl, p00, p10, p11);
            add_tri(stl, p00, p11, p01);
        };
        let box_edge_xy = |theta: f32| -> (f32, f32) {
            let c = theta.cos();
            let s = theta.sin();
            let t = (half_xy / c.abs().max(1.0e-6)).min(half_xy / s.abs().max(1.0e-6));
            ((c * t).clamp(-half_xy, half_xy), (s * t).clamp(-half_xy, half_xy))
        };

        for a in 0..n_ang {
            let th0 = thetas[a];
            let th1 = thetas[(a + 1) % n_ang];
            let (bx0, by0) = box_edge_xy(th0);
            let (bx1, by1) = box_edge_xy(th1);
            let (rx0, ry0) = (r_boss * th0.cos(), r_boss * th0.sin());
            let (rx1, ry1) = (r_boss * th1.cos(), r_boss * th1.sin());

            // Bottom disk (z = 0), outer perimeter walls (z = 0..z_deck), and flat top annulus
            add_tri(&mut stl, [0.0, 0.0, 0.0], [bx1, by1, 0.0], [bx0, by0, 0.0]);
            add_quad(
                &mut stl,
                [bx0, by0, 0.0],
                [bx1, by1, 0.0],
                [bx1, by1, z_deck],
                [bx0, by0, z_deck],
            );
            add_quad(
                &mut stl,
                [rx0, ry0, z_deck],
                [bx0, by0, z_deck],
                [bx1, by1, z_deck],
                [rx1, ry1, z_deck],
            );
            // Cylindrical boss side walls and top cap
            add_quad(
                &mut stl,
                [rx0, ry0, z_deck],
                [rx1, ry1, z_deck],
                [rx1, ry1, z_boss_top],
                [rx0, ry0, z_boss_top],
            );
            add_tri(
                &mut stl,
                [0.0, 0.0, z_boss_top],
                [rx0, ry0, z_boss_top],
                [rx1, ry1, z_boss_top],
            );
        }
        stl.push_str("endsolid flat_surface_with_single_boss\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_flat_surface_single_boss.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write flat+boss STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected flat+boss STL load to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::ConstrainedNonPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.min_mm = 0.12;
        cfg.slicing.layer_height.max_mm = 0.30;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.layer_height.top_layer_mm = 0.18;
        cfg.slicing.enable_auto_conformal_skin = false;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.one_perimeter_on_conformal_patches = true;
        cfg.tools[0].clearance_mode = crate::config::ClearanceProfileMode::PiecewiseXZ;
        cfg.tools[0].clearance_profile_xz = vec![
            crate::config::ClearancePointXZ::new(0.5, 0.0),
            crate::config::ClearancePointXZ::new(4.0, 1.4),
            crate::config::ClearancePointXZ::new(9.0, 6.0),
        ];
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);

        // Paint manual ConformalSurfaceSkin hints on the flat deck quadrants around the boss
        let hints: Vec<ffi::SpatialOrientationHint> =
            [(-6.0_f32, -6.0_f32), (6.0, -6.0), (6.0, 6.0), (-6.0, 6.0)]
            .into_iter()
            .map(|(x, y)| ffi::SpatialOrientationHint {
                kind: 2,
                position_bed: [x, y, z_deck],
                direction_bed: [0.0, 0.0, 1.0],
                influence_radius_mm: 12.0,
                weight: 1.0,
                apply_on_surface_only: true,
                skin_layer_count: 3,
            })
            .collect();
        ffi::set_spatial_orientation_hints(&hints);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid && !iso.layers.is_empty());
        assert_eq!(iso.branch_count, 1, "Expected 1 Reeb branch on flat surface + single boss");
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && tp.total_path_count > 0);
        assert_eq!(iso.layers.len(), tp.layers.len());

        let top_h = cfg.slicing.layer_height.top_layer_mm;
        let mut cap_skins: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                if !l.is_top_layer || l.is_first_layer || (l.mean_thickness_mm - top_h).abs() >= 1.0e-3 {
                    return false;
                }
                let mean_z = if l.vertices.is_empty() {
                    0.0
                } else {
                    l.vertices.iter().map(|v| v.z).sum::<f32>() / l.vertices.len() as f32
                };
                (z_deck - 1.0..=z_deck + 0.5).contains(&mean_z)
            })
            .collect();
        assert!(
            cap_skins.len() >= 3,
            "Expected >= 3 conformal cap layers on flat deck around boss, got {}",
            cap_skins.len()
        );
        cap_skins.sort_by_key(|l| (l.branch_layer_index, l.layer_index));
        let max_cap_idx = cap_skins.iter().map(|l| l.layer_index).max().unwrap();

        let outermost_cap = cap_skins.last().copied().unwrap();
        let mut octant_min_r = [f32::INFINITY; 8];
        let mut hole_verts = 0usize;
        for v in &outermost_cap.vertices {
            let r = v.x.hypot(v.y);
            if r < 2.9 {
                hole_verts += 1;
            }
            let ang = v.y.atan2(v.x).rem_euclid(2.0 * std::f32::consts::PI);
            let oct = ((ang / (std::f32::consts::FRAC_PI_4)).floor() as usize).min(7);
            octant_min_r[oct] = octant_min_r[oct].min(r);
        }
        let worst_octant_r = octant_min_r.iter().copied().fold(0.0_f32, f32::max);
        let tp_outer = tp
            .layers
            .iter()
            .find(|tl| tl.layer_index == outermost_cap.layer_index)
            .unwrap();
        let ow_cnt = tp_outer.paths.iter().filter(|p| p.feature_kind == 0).count();
        println!(
            "Flat+Boss conformal cap: layers={}, max_cap_idx={}, worst_octant_r={:.2}, hole_verts={}, OW={}",
            cap_skins.len(),
            max_cap_idx,
            worst_octant_r,
            hole_verts,
            ow_cnt
        );
        assert!(
            worst_octant_r <= r_boss + 0.90,
            "Expected flat conformal cap to reach right up to r={r_boss:.2} mm boss in all 8 octants, got max min_r={worst_octant_r:.2}"
        );
        assert_eq!(hole_verts, 0, "Expected clean circular hole (0 verts inside r < 2.9 mm)");
        assert_eq!(ow_cnt, 2, "Expected 2 OuterWall loops (outer square + inner boss hole), got {ow_cnt}");

        let boss_layers: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                !l.vertices.is_empty()
                    && l.vertices.iter().map(|v| v.z).fold(f32::INFINITY, f32::min) > z_deck + 0.3
            })
            .collect();
        assert!(!boss_layers.is_empty());
        for bl in &boss_layers {
            assert!(
                bl.layer_index > max_cap_idx,
                "Boss layer #{} printed before conformal cap #{}",
                bl.layer_index,
                max_cap_idx
            );
        }

        ffi::set_spatial_orientation_hints(&[]);
        let default_cfg = crate::config::SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(obj.object_id);
    }

    #[test]
    fn test_layer_and_within_layer_scrub_tracks_tool_and_gravity() {
        let mut cfg = SlicerConfig::default();
        cfg.machine.kinematics_mode = crate::config::FiveAxisKinematicsMode::FiveAxisBC;
        cfg.machine.bed_tilts_with_b = true;

        let tilted_nozzle = Vec3::new(0.5, 0.0, 0.8660254).normalize();
        let make_seg = |p0: Vec3, p1: Vec3, b_deg: f32, c_deg: f32, noz: Vec3, tool_id: u32| {
            crate::gcode::PlannedSegment {
                p0_local: p0,
                p1_local: p1,
                p0_bed: p0,
                p1_bed: p1,
                nozzle_local_0: noz,
                nozzle_local_1: noz,
                nozzle_bed_0: noz,
                nozzle_bed_1: noz,
                b0_deg: b_deg,
                c0_deg: c_deg,
                b1_deg: b_deg,
                c1_deg: c_deg,
                kind: crate::gcode::PlannedMoveKind::OuterWall,
                tool_id,
                filament_slot: 0,
                wall_index: 0,
                width_mm: 0.42,
                height_mm: 0.20,
                length_mm: (p1 - p0).length(),
                nominal_speed_mms: 80.0,
                entry_speed_mms: 80.0,
                cruise_speed_mms: 80.0,
                exit_speed_mms: 80.0,
                accel_mms2: 2000.0,
                duration_s: 0.05,
                delta_e_mm: 0.10,
                volumetric_flow_mm3s: 6.7,
                extrusion_multiplier: 1.0,
                pressure_advance_s: 0.03,
                melt_pressure_mm3s: 6.7,
                retract_before_mm: 0.0,
                unretract_after_mm: 0.0,
                tool_iso_angle_deg: 0.0,
                uses_surface_normal: false,
                sdf_mm_0: 0.0,
                sdf_mm: 0.0,
                sdf_mm_1: 0.0,
            }
        };

        let planned = crate::gcode::PlannedGcodeResult {
            layers: vec![
                crate::gcode::PlannedLayerGcode {
                    layer_index: 0,
                    branch_id: 0,
                    isovalue: 0.2,
                    segments: vec![
                        make_seg(
                            Vec3::new(0.0, 0.0, 0.2),
                            Vec3::new(10.0, 0.0, 0.2),
                            0.0,
                            0.0,
                            Vec3::Z,
                            0,
                        ),
                        make_seg(
                            Vec3::new(10.0, 0.0, 0.2),
                            Vec3::new(10.0, 10.0, 0.2),
                            0.0,
                            0.0,
                            Vec3::Z,
                            0,
                        ),
                    ],
                },
                crate::gcode::PlannedLayerGcode {
                    layer_index: 1,
                    branch_id: 0,
                    isovalue: 0.4,
                    segments: vec![
                        make_seg(
                            Vec3::new(0.0, 0.0, 0.4),
                            Vec3::new(5.0, 0.0, 0.4),
                            0.0,
                            0.0,
                            Vec3::Z,
                            0,
                        ),
                        make_seg(
                            Vec3::new(5.0, 0.0, 0.4),
                            Vec3::new(12.0, 4.0, 0.8),
                            30.0,
                            0.0,
                            tilted_nozzle,
                            1,
                        ),
                        make_seg(
                            Vec3::new(12.0, 4.0, 0.8),
                            Vec3::new(20.0, 4.0, 1.2),
                            0.0,
                            0.0,
                            Vec3::Z,
                            0,
                        ),
                    ],
                },
            ],
            gcode_text: String::new(),
            total_time_s: 1.0,
            total_filament_mm: 10.0,
            total_filament_volume_mm3: 24.0,
            total_filament_grams: 0.03,
            bbox_min_bed: Vec3::ZERO,
            bbox_max_bed: Vec3::splat(20.0),
            exclude_object_name: "test".to_string(),
            exclude_object_center_bed: [10.0, 5.0],
            exclude_object_polygon_bed: Vec::new(),
            extruding_move_count: 5,
            travel_move_count: 0,
            retract_count: 0,
            toolchange_count: 0,
            min_speed_mms: 80.0,
            max_speed_mms: 80.0,
            min_accel_mms2: -2000.0,
            max_accel_mms2: 2000.0,
            min_flow_mm3s: 6.7,
            max_flow_mm3s: 6.7,
            min_width_mm: 0.42,
            max_width_mm: 0.42,
            min_height_mm: 0.20,
            max_height_mm: 0.20,
            min_flow_mult: 1.0,
            max_flow_mult: 1.0,
            min_pa_s: 0.03,
            max_pa_s: 0.03,
            min_pressure_mm3s: 6.7,
            max_pressure_mm3s: 6.7,
            min_tool_iso_angle_deg: 0.0,
            max_tool_iso_angle_deg: 30.0,
            min_sdf_mm: -2.0,
            max_sdf_mm: 0.5,
        };

        let mut vis = FieldVisualizationState::default();
        vis.show_toolpaths = true;
        vis.selected_layer_min_idx = 0;
        vis.selected_layer_max_idx = 1;
        vis.cached_planned_gcode = Some(planned);

        assert_eq!(vis.total_layer_count(), 2);
        assert_eq!(vis.active_layer_total_moves(), 3);

        // 1. Scrubbing to move 2 of layer 1 should place the tool at (12, 4, 0.8) with tilted nozzle & tilted gravity
        vis.layer_move_scrub_limit = Some(2);
        let pose = vis
            .compute_active_scrub_pose(&cfg, Transform::IDENTITY)
            .expect("Expected valid scrub pose");
        assert_eq!(pose.layer_index, 1);
        assert_eq!(pose.move_index, 2);
        assert_eq!(pose.total_layer_moves, 3);
        assert_eq!(pose.tool_id, 1);
        assert!((pose.tip_bed - Vec3::new(12.0, 4.0, 0.8)).length() < 1.0e-4);
        assert!((pose.frame_bed.nozzle_axis - tilted_nozzle).length() < 1.0e-4);
        assert!(
            pose.gravity_bed.x.abs() > 0.4,
            "Expected bed-frame gravity to tilt with B=30 deg on move 2, got {:?}",
            pose.gravity_bed
        );

        // 2. Verify that building the 3D bead mesh with `layer_move_scrub_limit = Some(2)` emits 2 moves on layer 0 + 2 moves on layer 1 = 4 prisms (32 vertices)
        let mut pos = Vec::new();
        let mut nrm = Vec::new();
        let mut col = Vec::new();
        let mut ind = Vec::new();
        build_planned_toolpaths_bead_mesh_parallel(
            vis.cached_planned_gcode.as_ref().unwrap(),
            &vis.toolpath_mesh_filter(),
            &cfg,
            0,
            1,
            1,
            None,
            ToolpathColorMode::ByFeatureType,
            &mut pos,
            &mut nrm,
            &mut col,
            &mut ind,
        );
        assert_eq!(pos.len(), 4 * 8);

        // 3. Scrubbing layer slider back to layer 0 (`layer_move_scrub_limit = None`) tracks layer 0's end move
        vis.selected_layer_max_idx = 0;
        vis.layer_move_scrub_limit = None;
        let pose_l0 = vis
            .compute_active_scrub_pose(&cfg, Transform::IDENTITY)
            .expect("Expected valid scrub pose on layer 0");
        assert_eq!(pose_l0.layer_index, 0);
        assert_eq!(pose_l0.move_index, 2);
        assert!((pose_l0.tip_bed - Vec3::new(10.0, 10.0, 0.2)).length() < 1.0e-4);
        assert!((pose_l0.gravity_bed - (-Vec3::Z)).length() < 1.0e-4);
    }

    #[test]
    fn test_orient_outer_surfaces_to_normal_or_isosurface_normal() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct an asymmetric prism with:
        // - +X wall sloped at 40° from bed normal +Z:
        //   At Z = 0, X = +12.0; at Z = 10.0, X = +12.0 - 10.0 * tan(50°)? Wait:
        //   Let outward normal on +X wall be tilted by 40° from +Z -> (sin 40°, 0, cos 40°) = (0.6428, 0, 0.7660).
        //   A face with outward normal (sin 40°, 0, cos 40°) has dx/dz = -cos(40°)/sin(40°) = -1.19175.
        //   Wait: let's make +X wall have outward normal at 50° from +Z:
        //   dx = -8.4 mm over dz = 10.0 mm -> outward normal = normalize(10.0, 0, 8.4) = (0.7656, 0, 0.6433),
        //   which is arccos(0.6433) = 49.96° (~50°) from bed normal +Z!
        // - -X wall is vertical (outward normal = (-1, 0, 0), 90° from +Z).
        //
        // Now configure a 5/6-axis tool with `max_angle_from_bed_normal_deg = 40.0°` (and narrow clearance so bed clearance doesn't restrict at Z >= 4):
        // - On the +X sloped wall (normal at ~50° from +Z), the closest reachable tool axis `surfAxis`
        //   clamps to 40° from +Z -> the residual angle between `surfAxis` and `n_surf` is ~10°!
        //   * When `orient_outer_surfaces_to_normal = true` and `outer_surface_parallel_angle_deg = 15.0°`:
        //     residual ~10° <= 15.0°, so the +X OuterWall orients to `surfAxis` (~40° tilt toward +X, `nx > 0.50`)!
        //   * When `outer_surface_parallel_angle_deg = 5.0°`:
        //     residual ~10° > 5.0°, so the +X OuterWall CANNOT reach within 5° of `n_surf` and falls back
        //     to the isosurface normal (nearly vertical in HeightOnlyPlanar mode, `nx < 0.15`)!
        //   * When `orient_outer_surfaces_to_normal = false`:
        //     toggle is off, so the +X OuterWall uses the isosurface normal (`nx < 0.15`)!
        // - On the -X vertical wall (normal at 90° from +Z), residual angle is 90° - 40° = 50° > 15°,
        //   so it ALWAYS falls back to the isosurface normal (`|nx| < 0.15`)!
        let positions: [f32; 24] = [
            -10.0, -8.0,  0.0,
             12.0, -8.0,  0.0,
             12.0,  8.0,  0.0,
            -10.0,  8.0,  0.0,
            -10.0, -8.0, 10.0,
              3.6, -8.0, 10.0,
              3.6,  8.0, 10.0,
            -10.0,  8.0, 10.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, // bottom
            4, 5, 6, 4, 6, 7, // top
            0, 1, 5, 0, 5, 4, // -Y
            1, 2, 6, 1, 6, 5, // +X sloped wall (normal ~50° from +Z)
            2, 3, 7, 2, 7, 6, // +Y
            3, 0, 4, 3, 4, 7, // -X vertical wall (normal 90° from +Z)
        ];
        let part = ffi::register_triangle_mesh("outer_surf_orient_prism", &positions, &indices);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.slicing.enable_support_generation = false;
        cfg.tools[0].has_rotational_axes = true;
        cfg.tools[0].max_angle_from_bed_normal_deg = 40.0;
        cfg.tools[0].clearance_profile_xz = vec![
            crate::config::ClearancePointXZ {
                x_radius_mm: 0.4,
                z_height_mm: 0.0,
            },
            crate::config::ClearancePointXZ {
                x_radius_mm: 1.5,
                z_height_mm: 3.0,
            },
        ];
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        let measure_wall_nx = |orient_enabled: bool, parallel_deg: f32| -> (f32, f32, f32, usize) {
            let mut c = cfg.clone();
            c.slicing.orient_outer_surfaces_to_normal = orient_enabled;
            c.slicing.outer_surface_parallel_angle_deg = parallel_deg;
            crate::ui::sync_config_to_backend(&c);

            let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp.valid);

            let mut pos_x_nx_sum = 0.0f32;
            let mut pos_x_iso_ang_sum = 0.0f32;
            let mut pos_x_surf_norm_cnt = 0usize;
            let mut pos_x_cnt = 0usize;
            let mut neg_x_nx_abs_sum = 0.0f32;
            let mut neg_x_cnt = 0usize;

            for layer in &tp.layers {
                for path in &layer.paths {
                    if path.feature_kind != 0 {
                        continue; // OuterWall only
                    }
                    for pt in &path.points {
                        if pt.z >= 4.0 && pt.z <= 8.0 && pt.y.abs() <= 4.0 {
                            if pt.x > 2.0 {
                                pos_x_nx_sum += pt.nx;
                                pos_x_iso_ang_sum += pt.tool_iso_angle_deg;
                                if pt.uses_surface_normal {
                                    pos_x_surf_norm_cnt += 1;
                                }
                                pos_x_cnt += 1;
                            } else if pt.x < -6.0 {
                                neg_x_nx_abs_sum += pt.nx.abs();
                                assert!(
                                    !pt.uses_surface_normal,
                                    "Expected -X vertical wall not to report uses_surface_normal"
                                );
                                neg_x_cnt += 1;
                            }
                        }
                    }
                }
            }
            assert!(pos_x_cnt > 5 && neg_x_cnt > 5);
            (
                pos_x_nx_sum / (pos_x_cnt as f32),
                neg_x_nx_abs_sum / (neg_x_cnt as f32),
                pos_x_iso_ang_sum / (pos_x_cnt as f32),
                pos_x_surf_norm_cnt,
            )
        };

        // 1. Toggle ON, threshold = 15° (residual angle ~10° <= 15° on +X wall -> tilts to ~40°, sin(40°)=0.643;
        //    on -X vertical wall, residual angle 50° > 15° -> falls back to isosurface normal +Z, nx ≈ 0)
        let (nx_on_15, neg_nx_on_15, iso_ang_on_15, surf_cnt_on_15) = measure_wall_nx(true, 15.0);
        assert!(
            nx_on_15 > 0.55,
            "Expected +X sloped wall (within 15° of reachable 40° tilt) to orient almost parallel to surface normal (nx > 0.55), got {nx_on_15:.3}"
        );
        assert!(
            iso_ang_on_15 > 32.0,
            "Expected +X sloped wall tool_iso_angle_deg > 32° when tilted ~40° from planar isosurface normal, got {iso_ang_on_15:.2}°"
        );
        assert!(
            surf_cnt_on_15 > 5,
            "Expected +X sloped wall points to set uses_surface_normal == true, got count={surf_cnt_on_15}"
        );
        assert!(
            neg_nx_on_15 < 0.08,
            "Expected -X vertical wall (50° beyond reachable tilt > 15°) to fall back to isosurface normal (|nx| < 0.08), got {neg_nx_on_15:.3}"
        );

        // 2. Toggle ON, threshold = 4° (residual angle ~10° > 4° on +X wall -> falls back to isosurface normal!)
        let (nx_on_4, _, iso_ang_on_4, surf_cnt_on_4) = measure_wall_nx(true, 4.0);
        assert!(
            nx_on_4 < 0.08,
            "Expected +X sloped wall (~10° from reachable 40° tilt) to fall back to isosurface normal when threshold is 4°, got {nx_on_4:.3}"
        );
        assert!(
            iso_ang_on_4 < 4.0 && surf_cnt_on_4 == 0,
            "Expected fallback to isosurface normal to have tool_iso_angle_deg < 4° and uses_surface_normal == false, got ang={iso_ang_on_4:.2}°, cnt={surf_cnt_on_4}"
        );

        // 3. Toggle OFF -> always uses isosurface normal even when threshold is 30°
        let (nx_off, _, iso_ang_off, surf_cnt_off) = measure_wall_nx(false, 30.0);
        assert!(
            nx_off < 0.08 && iso_ang_off < 4.0 && surf_cnt_off == 0,
            "Expected +X sloped wall to use isosurface normal when orient_outer_surfaces_to_normal is false, got {nx_off:.3}"
        );

        let default_cfg = SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_multicolor_3mf_filament_slot_field_toolpath_split_and_toolchange_gcode() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Construct a 16x16x8 mm rectangular box split along X = 0 so triangles on the -X half
        // use filament_slot = 0 and triangles on the +X half use filament_slot = 1.
        let positions: [f32; 24] = [
            -8.0, -8.0, 0.0, // 0 (-X)
             8.0, -8.0, 0.0, // 1 (+X)
             8.0,  8.0, 0.0, // 2 (+X)
            -8.0,  8.0, 0.0, // 3 (-X)
            -8.0, -8.0, 8.0, // 4 (-X)
             8.0, -8.0, 8.0, // 5 (+X)
             8.0,  8.0, 8.0, // 6 (+X)
            -8.0,  8.0, 8.0, // 7 (-X)
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2, // bottom
            4, 5, 6, 4, 6, 7, // top
            0, 1, 5, 0, 5, 4, // -Y front
            1, 2, 6, 1, 6, 5, // +X right (slot 1)
            2, 3, 7, 2, 7, 6, // +Y back
            3, 0, 4, 3, 4, 7, // -X left (slot 0)
        ];
        // Tag triangles: right (+X) and +X half of top/bottom/front/back get slot 1; -X half get slot 0
        let tri_slots: [u8; 12] = [
            1, 0, // bottom
            1, 0, // top
            1, 0, // -Y front
            1, 1, // +X right
            0, 1, // +Y back
            0, 0, // -X left
        ];

        let part = ffi::register_triangle_mesh("test_bicolor_box", &positions, &indices);
        assert!(part.object_id > 0);
        ffi::set_object_triangle_filament_slots(part.object_id, &tri_slots);

        let mut cfg = SlicerConfig::default();
        cfg.apply_imported_filament_slots(vec![
            crate::config::ThreeMfFilamentSlot::default_for_slot(0, 0),
            crate::config::ThreeMfFilamentSlot::default_for_slot(1, 1),
        ]);
        cfg.tools[0].id = 0;
        cfg.tools[0].temperature_celsius = 210.0;
        cfg.tools[1].id = 1;
        cfg.tools[1].temperature_celsius = 225.0;
        // Map Slot 0 -> Tool 1 (T1) and Slot 1 -> Tool 0 (T0) to verify arbitrary user-configured mapping!
        cfg.filament_slots[0].tool_index = 1;
        cfg.filament_slots[1].tool_index = 0;

        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        let fields = ffi::list_object_fields(part.object_id);
        assert!(
            fields.iter().any(|f| f == "filament_slot"),
            "Expected 'filament_slot' OpenVDB scalar field on multi-colour object, got {fields:?}"
        );

        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid && !tp.layers.is_empty());

        let mut count_slot_0 = 0usize;
        let mut count_slot_1 = 0usize;
        for layer in &tp.layers {
            for path in &layer.paths {
                if path.filament_slot == 0 {
                    count_slot_0 += 1;
                } else if path.filament_slot == 1 {
                    count_slot_1 += 1;
                }
            }
        }
        assert!(
            count_slot_0 > 0 && count_slot_1 > 0,
            "Expected multi-colour toolpath splitting to emit both filament_slot 0 ({count_slot_0}) and filament_slot 1 ({count_slot_1}) paths"
        );

        let planned = crate::gcode::plan_and_emit_gcode_for_object(
            "test_bicolor_box",
            part.object_id,
            &tp,
            Transform::IDENTITY,
            &cfg,
        );
        // Because Slot 0 is mapped to T1 (225°C) and Slot 1 is mapped to T0 (210°C),
        // planned segments with filament_slot == 0 must have tool_id == 1, and filament_slot == 1 must have tool_id == 0!
        let mut verified_slot0_t1 = false;
        let mut verified_slot1_t0 = false;
        for layer in &planned.layers {
            for seg in &layer.segments {
                if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-5 {
                    if seg.filament_slot == 0 {
                        assert_eq!(seg.tool_id, 1);
                        verified_slot0_t1 = true;
                    } else if seg.filament_slot == 1 {
                        assert_eq!(seg.tool_id, 0);
                        verified_slot1_t0 = true;
                    }
                }
            }
        }
        assert!(verified_slot0_t1 && verified_slot1_t0);
        assert!(
            planned.gcode_text.contains("M104 S225 T1 ; Preheat tool T1"),
            "Expected G-code header to preheat mapped auxiliary tool T1 at 225°C"
        );
        assert!(
            planned.gcode_text.contains("T1 ; Switch tool")
                && planned.gcode_text.contains("T0 ; Switch tool"),
            "Expected G-code to emit toolchanges between T0 and T1"
        );
        assert!(
            planned.toolchange_count > 0,
            "Expected toolchange_count > 0 for multi-colour print, got {}",
            planned.toolchange_count
        );

        // Verify that increasing `tool_change_time_s` by +10.0s increases `total_time_s`
        // by exactly `toolchange_count * 10.0` seconds.
        let mut cfg_slow_tc = cfg.clone();
        cfg_slow_tc.gcode.tool_change_time_s = cfg.gcode.tool_change_time_s + 10.0;
        let planned_slow_tc = crate::gcode::plan_and_emit_gcode_for_object(
            "test_bicolor_box",
            part.object_id,
            &tp,
            Transform::IDENTITY,
            &cfg_slow_tc,
        );
        assert_eq!(planned_slow_tc.toolchange_count, planned.toolchange_count);
        let expected_extra_s = (planned.toolchange_count as f32) * 10.0;
        let actual_extra_s = planned_slow_tc.total_time_s - planned.total_time_s;
        assert!(
            (actual_extra_s - expected_extra_s).abs() < 0.05,
            "Expected +10s per toolchange ({} toolchanges => +{expected_extra_s}s), got +{actual_extra_s}s",
            planned.toolchange_count
        );

        let default_cfg = SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_cmykw_optical_color_mixing_end_to_end() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let positions: [f32; 24] = [
            -8.0, -8.0, 0.0,
             8.0, -8.0, 0.0,
             8.0,  8.0, 0.0,
            -8.0,  8.0, 0.0,
            -8.0, -8.0, 4.0,
             8.0, -8.0, 4.0,
             8.0,  8.0, 4.0,
            -8.0,  8.0, 4.0,
        ];
        // 12 triangles: 0..2 bottom, 2..4 top, 4..6 front (-Y), 6..8 right (+X), 8..10 back (+Y), 10..12 left (-X)
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];

        let part = ffi::register_triangle_mesh("test_cmykw_box", &positions, &indices);
        assert!(part.object_id > 0);

        let green = crate::color_optics::pack_rgba8([20, 185, 45, 255]);   // secondary: Cyan + Yellow
        let orange = crate::color_optics::pack_rgba8([250, 110, 15, 255]); // secondary: Magenta + Yellow
        let purple = crate::color_optics::pack_rgba8([115, 25, 210, 255]); // secondary: Cyan + Magenta
        let white = crate::color_optics::pack_rgba8([244, 244, 244, 255]);
        let tri_rgba = vec![
            white, white,   // bottom
            green, green,   // top
            green, green,   // front (-Y)
            orange, orange, // right (+X)
            purple, purple, // back (+Y)
            orange, purple, // left (-X)
        ];
        ffi::set_object_triangle_colors_rgba(part.object_id, &tri_rgba);
        assert_eq!(ffi::get_object_triangle_colors_rgba(part.object_id), tri_rgba);

        let mut cfg = SlicerConfig::default();
        cfg.apply_optical_palette_to_tools(&crate::color_optics::bambu_pla_cmykw_palette());
        cfg.slicing.color_mixing.enabled = true;
        cfg.slicing.color_mixing.use_adjacent_layers = true;
        cfg.slicing.color_mixing.use_short_extrusions = true;
        cfg.slicing.color_mixing.min_short_extrusion_mm = 2.5;
        cfg.slicing.color_mixing.optical_shell_layers = 3;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.shell_thickness_mm = 1.2;
        cfg.slicing.infill_density_percent = 20.0;
        cfg.eikonal.auto_voxel_size = false;
        cfg.eikonal.voxel_size_mm = 0.5;
        crate::ui::sync_config_to_backend(&cfg);

        assert!(ffi::recompute_object_fields(part.object_id, 0.5, 4.0));

        // Verify C++ `sample_object_surface_optics` projects interior shell points onto nearest coloured face
        let query_pts: [f32; 9] = [
            0.0, -7.6, 2.0, // near front (-Y) green wall
            7.6,  0.0, 2.0, // near right (+X) orange wall
            0.0,  7.6, 2.0, // near back (+Y) purple wall
        ];
        let samples = ffi::sample_object_surface_optics(part.object_id, &query_pts);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].rgba_packed, green);
        assert!(samples[0].surf_ny < -0.7, "Expected -Y outward normal, got {:?}", samples[0]);
        assert_eq!(samples[1].rgba_packed, orange);
        assert!(samples[1].surf_nx > 0.7, "Expected +X outward normal, got {:?}", samples[1]);
        assert_eq!(samples[2].rgba_packed, purple);
        assert!(samples[2].surf_ny > 0.7, "Expected +Y outward normal, got {:?}", samples[2]);

        let raw_tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(raw_tp.valid && !raw_tp.layers.is_empty());

        let mixed_tp = build_mixed_toolpaths_for_object(part.object_id, &raw_tp, &cfg);
        assert!(mixed_tp.valid);
        assert!(
            mixed_tp.outer_wall_count > raw_tp.outer_wall_count,
            "Short-extrusion optical mixing should subdivide multi-colour outer perimeters ({} vs {})",
            mixed_tp.outer_wall_count,
            raw_tp.outer_wall_count
        );

        let mut used_tools = std::collections::BTreeSet::new();
        for layer in &mixed_tp.layers {
            for p in &layer.paths {
                let t_idx = cfg.tool_index_for_filament_slot(p.filament_slot);
                used_tools.insert(t_idx);
                if p.feature_kind == 4 {
                    // SparseInfill must use the White core backing filament (T4)
                    assert_eq!(t_idx, 4, "Sparse infill should use White backing tool T4");
                }
            }
        }
        // Green, Orange, and Purple require mixing Cyan (T0), Magenta (T1), Yellow (T2) over White (T4)!
        assert!(
            used_tools.len() >= 4,
            "Expected CMYKW solver to mix >= 4 tools across Green/Orange/Purple faces, got {used_tools:?}"
        );

        let planned = crate::gcode::plan_and_emit_gcode_for_object(
            "test_cmykw_box",
            part.object_id,
            &mixed_tp,
            Transform::IDENTITY,
            &cfg,
        );
        assert!(planned.toolchange_count >= 3);
        assert!(planned.gcode_text.contains("T0"));
        assert!(planned.gcode_text.contains("T1"));
        assert!(planned.gcode_text.contains("T2"));

        // Also verify Back-Lit Color Lithophane mode (`BackLitTransmittance`, open backlight without core backing)
        cfg.slicing.color_mixing.lighting_mode =
            crate::color_optics::OpticalLightingMode::BackLitTransmittance;
        let backlit_tp = build_mixed_toolpaths_for_object(part.object_id, &raw_tp, &cfg);
        assert!(backlit_tp.valid && !backlit_tp.layers.is_empty());

        let default_cfg = SlicerConfig::default();
        crate::ui::sync_config_to_backend(&default_cfg);
        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_hq_stadium_pbr_bucketing_outward_normals_and_translucency() {
        // 1. Verify closed-loop welded stadium ring count & outward-pointing FrontFace::Ccw normals
        let mut closed_acc = HqBucketAccum::default();
        let square_loop = vec![
            HqChainNode {
                pos: Vec3::new(0.0, 0.0, 0.2),
                nozzle_axis: Vec3::new(0.0, 0.2588, 0.9659), // tilted 15° nozzle axis (AGENTS.md Rule 1)
                width_mm: 0.44,
                height_mm: 0.20,
                linear_rgba: [0.1, 0.6, 0.9, 1.0],
                is_circular: false,
            },
            HqChainNode {
                pos: Vec3::new(10.0, 0.0, 0.2),
                nozzle_axis: Vec3::new(0.0, 0.2588, 0.9659),
                width_mm: 0.44,
                height_mm: 0.20,
                linear_rgba: [0.1, 0.6, 0.9, 1.0],
                is_circular: false,
            },
            HqChainNode {
                pos: Vec3::new(10.0, 10.0, 0.2),
                nozzle_axis: Vec3::new(0.0, 0.2588, 0.9659),
                width_mm: 0.44,
                height_mm: 0.20,
                linear_rgba: [0.1, 0.6, 0.9, 1.0],
                is_circular: false,
            },
            HqChainNode {
                pos: Vec3::new(0.0, 10.0, 0.2),
                nozzle_axis: Vec3::new(0.0, 0.2588, 0.9659),
                width_mm: 0.44,
                height_mm: 0.20,
                linear_rgba: [0.1, 0.6, 0.9, 1.0],
                is_circular: false,
            },
            HqChainNode {
                pos: Vec3::new(0.0, 0.0, 0.2),
                nozzle_axis: Vec3::new(0.0, 0.2588, 0.9659),
                width_mm: 0.44,
                height_mm: 0.20,
                linear_rgba: [0.1, 0.6, 0.9, 1.0],
                is_circular: false,
            },
        ];
        append_hq_welded_chain_stadium_mesh(&square_loop, Vec3::Z, &mut closed_acc);
        // Closed 4-corner loop must have exactly 4 * 12 = 48 vertices (zero internal endcap apices!)
        assert_eq!(closed_acc.positions.len(), 4 * HQ_RING_VERTS);
        assert_eq!(closed_acc.indices.len(), 4 * HQ_RING_VERTS * 6);

        for tri in closed_acc.indices.chunks_exact(3) {
            let p0 = Vec3::from_array(closed_acc.positions[tri[0] as usize]);
            let p1 = Vec3::from_array(closed_acc.positions[tri[1] as usize]);
            let p2 = Vec3::from_array(closed_acc.positions[tri[2] as usize]);
            let face_n = (p1 - p0).cross(p2 - p0);
            assert!(face_n.length() > 1.0e-7, "Degenerate triangle in stadium tube");
            let avg_vn = Vec3::from_array(closed_acc.normals[tri[0] as usize])
                + Vec3::from_array(closed_acc.normals[tri[1] as usize])
                + Vec3::from_array(closed_acc.normals[tri[2] as usize]);
            assert!(
                face_n.dot(avg_vn) > 0.0,
                "Triangle winding must match outward analytic normal for backface culling"
            );
        }

        // Open chain must include 2 convex bullnose endcap apices (`2 * 12 + 2 = 26` vertices) with outward normals
        let mut open_acc = HqBucketAccum::default();
        append_hq_welded_chain_stadium_mesh(&square_loop[..2], Vec3::Z, &mut open_acc);
        assert_eq!(open_acc.positions.len(), 2 * HQ_RING_VERTS + 2);
        for tri in open_acc.indices.chunks_exact(3) {
            let p0 = Vec3::from_array(open_acc.positions[tri[0] as usize]);
            let p1 = Vec3::from_array(open_acc.positions[tri[1] as usize]);
            let p2 = Vec3::from_array(open_acc.positions[tri[2] as usize]);
            let face_n = (p1 - p0).cross(p2 - p0);
            let avg_vn = Vec3::from_array(open_acc.normals[tri[0] as usize])
                + Vec3::from_array(open_acc.normals[tri[1] as usize])
                + Vec3::from_array(open_acc.normals[tri[2] as usize]);
            assert!(
                face_n.dot(avg_vn) > 0.0,
                "Bullnose endcap winding must face outward"
            );
        }

        // 2. Verify TD-derived filament opacity, tie-free depth_bias ordering, and X-ray multiplier
        let mut cfg = SlicerConfig::default();
        cfg.apply_optical_palette_to_tools(&crate::color_optics::bambu_pla_cmykw_palette());
        // In bambu_pla_cmykw_palette: T0 = Cyan (TD 3.5 mm), T2 = Yellow (TD 6.0 mm), T3 = Black (TD 0.6 mm), T4 = White (TD 5.0 mm)
        let mut vis = FieldVisualizationState::default();
        vis.toolpath_quality_mode = ToolpathQualityMode::HighQualityPbr;
        vis.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
        vis.enable_pbr_translucency = true;
        vis.toolpath_opacity = 1.0;

        let mat_core_white = build_hq_bucket_pbr_material(0, 4, 0.42, &vis, &cfg);
        let mat_inner_yel = build_hq_bucket_pbr_material(1, 2, 0.42, &vis, &cfg);
        let mat_outer_cyan = build_hq_bucket_pbr_material(2, 0, 0.42, &vis, &cfg);
        let mat_outer_black = build_hq_bucket_pbr_material(2, 3, 0.42, &vis, &cfg);

        // Screen-space specular_transmission is 0.0 (noise-free), while alpha opacity strictly tracks TD!
        assert_eq!(mat_outer_cyan.specular_transmission, 0.0);
        assert!(matches!(mat_core_white.alpha_mode, AlphaMode::Opaque));
        assert!(matches!(mat_outer_cyan.alpha_mode, AlphaMode::Blend));
        let alpha_yel = mat_inner_yel.base_color.alpha();
        let alpha_cyan = mat_outer_cyan.base_color.alpha();
        let alpha_black = mat_outer_black.base_color.alpha();
        assert!(
            alpha_yel < alpha_cyan && alpha_cyan < alpha_black && alpha_black > 0.75,
            "Higher TD must yield lower single-bead opacity (Yellow TD 6.0={alpha_yel} < Cyan TD 3.5={alpha_cyan} < Black TD 0.6={alpha_black})"
        );
        // Verify Beer-Lambert multi-layer additivity: 5 stacked 0.12mm layers == 1 single 0.60mm slab!
        let a_thin = crate::color_optics::td_opacity_for_thickness_mm(3.5, 0.12);
        let a_stack5 = 1.0 - (1.0 - a_thin).powi(5);
        let a_thick = crate::color_optics::td_opacity_for_thickness_mm(3.5, 0.60);
        assert!((a_stack5 - a_thick).abs() < 1.0e-5);
        // Positive Z-bias (> 10.0) prevents CAD/isosurface Z-fighting, and unique per-slot offsets prevent sort ties!
        assert!(mat_core_white.depth_bias >= 12.0);
        assert!(mat_core_white.depth_bias < mat_inner_yel.depth_bias);
        assert!(mat_inner_yel.depth_bias < mat_outer_cyan.depth_bias);
        assert_ne!(mat_outer_cyan.depth_bias, mat_outer_black.depth_bias);

        // Lowering `toolpath_opacity` scales core & shell opacities cleanly in back-to-front order
        vis.toolpath_opacity = 0.45;
        let mat_xray_core = build_hq_bucket_pbr_material(0, 4, 0.30, &vis, &cfg);
        let mat_xray_outer = build_hq_bucket_pbr_material(2, 0, 0.30, &vis, &cfg);
        assert!(matches!(mat_xray_core.alpha_mode, AlphaMode::Blend));
        assert!(matches!(mat_xray_outer.alpha_mode, AlphaMode::Blend));
        assert!(mat_xray_core.depth_bias < mat_xray_outer.depth_bias);
    }

    #[test]
    fn test_brick_layer_modes_lateral_vertical_and_combined() {
        let _guard = crate::bridge::TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // 16 x 16 x 2.0 mm box (5 layers of 0.40 mm)
        let positions: [f32; 24] = [
            -8.0, -8.0, 0.0,  8.0, -8.0, 0.0,  8.0, 8.0, 0.0, -8.0, 8.0, 0.0,
            -8.0, -8.0, 2.0,  8.0, -8.0, 2.0,  8.0, 8.0, 2.0, -8.0, 8.0, 2.0,
        ];
        let indices: [u32; 36] = [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ];
        let part = ffi::register_triangle_mesh("brick_test_box", &positions, &indices);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = false;
        cfg.slicing.layer_height = crate::config::LayerHeightConfig {
            target_mm: 0.40,
            min_mm: 0.30,
            max_mm: 0.50,
            first_layer_mm: 0.40,
            top_layer_mm: 0.40,
        };
        cfg.slicing.shell_thickness_mm = 2.0;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.outer_wall_width = crate::config::LineWidthConfig::new(0.40, 0.28, 0.60);
        cfg.slicing.inner_wall_width = crate::config::LineWidthConfig::new(0.40, 0.28, 0.65);

        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.35, 3.0));

        // 1. Standard mode (brick_layer_mode = Standard)
        cfg.slicing.brick_layer_mode = crate::config::BrickLayerMode::Standard;
        crate::ui::sync_config_to_backend(&cfg);
        let tp_std = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_std.valid && tp_std.layers.len() >= 3);

        let std_l0_w1 = tp_std.layers[0]
            .paths
            .iter()
            .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            .expect("Standard Layer 0 InnerWall 1");
        let std_l1_w1 = tp_std.layers[1]
            .paths
            .iter()
            .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            .expect("Standard Layer 1 InnerWall 1");
        assert!(
            (std_l0_w1.mean_width_mm - std_l1_w1.mean_width_mm).abs() < 0.03,
            "Standard mode should have uniform InnerWall 1 width across even/odd layers ({} vs {})",
            std_l0_w1.mean_width_mm,
            std_l1_w1.mean_width_mm
        );

        // 2. LateralSeamStagger mode (brick_layer_mode = LateralSeamStagger)
        cfg.slicing.brick_layer_mode = crate::config::BrickLayerMode::LateralSeamStagger;
        crate::ui::sync_config_to_backend(&cfg);
        let tp_lat = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_lat.valid);
        let lat_l1_w1 = tp_lat.layers[1]
            .paths
            .iter()
            .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            .expect("Lateral Layer 1 InnerWall 1");
        assert!(
            lat_l1_w1.mean_width_mm > std_l1_w1.mean_width_mm + 0.14,
            "LateralSeamStagger odd layer InnerWall 1 should widen by +0.5*w_inner (~0.60mm vs {}), got {}",
            std_l1_w1.mean_width_mm,
            lat_l1_w1.mean_width_mm
        );

        // 3. VerticalHalfLayer mode (brick_layer_mode = VerticalHalfLayer)
        cfg.slicing.brick_layer_mode = crate::config::BrickLayerMode::VerticalHalfLayer;
        crate::ui::sync_config_to_backend(&cfg);
        let tp_vert = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_vert.valid);

        // Layer 0 (first layer) must contain both a 0.5*h starter pass and a +0.5*h raised pass for InnerWall 1
        let l0_w1_paths: Vec<_> = tp_vert.layers[0]
            .paths
            .iter()
            .filter(|p| p.feature_kind == 1 && p.wall_index == 1)
            .collect();
        assert!(
            l0_w1_paths.len() >= 2,
            "VerticalHalfLayer Layer 0 should have starter + raised passes for InnerWall 1, got {}",
            l0_w1_paths.len()
        );
        let has_starter = l0_w1_paths
            .iter()
            .any(|p| p.points.iter().all(|pt| pt.height_mm < 0.25));
        let has_raised = l0_w1_paths
            .iter()
            .any(|p| p.points.iter().all(|pt| pt.height_mm > 0.30));
        assert!(has_starter && has_raised, "Expected both half-height starter and full-height raised pass on Layer 0");

        // Interior layer 1: InnerWall 1 (odd) should sit +0.5*h higher in Z than OuterWall 0 (even)
        let l1_outer_z = tp_vert.layers[1]
            .paths
            .iter()
            .find(|p| p.feature_kind == 0)
            .unwrap()
            .points[0]
            .z;
        let l1_inner1_z = tp_vert.layers[1]
            .paths
            .iter()
            .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            .unwrap()
            .points[0]
            .z;
        assert!(
            l1_inner1_z > l1_outer_z + 0.14,
            "VerticalHalfLayer interior odd wall should be shifted +0.5*h above even wall ({l1_inner1_z} vs {l1_outer_z})"
        );

        // Topmost layer containing inner walls: InnerWall 1 should have half-height closing cap pass (height_mm ~ 0.20mm)
        let top_w1 = tp_vert
            .layers
            .iter()
            .rev()
            .find_map(|l| {
                l.paths
                    .iter()
                    .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            })
            .expect("Topmost layer with InnerWall 1");
        assert!(
            top_w1.points[0].height_mm < 0.25,
            "Topmost inner-wall layer odd wall should use half-height closing cap pass, got {}",
            top_w1.points[0].height_mm
        );

        // 4. CombinedLateralAndVertical mode (brick_layer_mode = CombinedLateralAndVertical)
        cfg.slicing.brick_layer_mode = crate::config::BrickLayerMode::CombinedLateralAndVertical;
        crate::ui::sync_config_to_backend(&cfg);
        let tp_comb = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_comb.valid);
        let comb_l1_w1 = tp_comb.layers[1]
            .paths
            .iter()
            .find(|p| p.feature_kind == 1 && p.wall_index == 1)
            .expect("Combined Layer 1 InnerWall 1");
        assert!(
            comb_l1_w1.mean_width_mm > std_l1_w1.mean_width_mm + 0.14
                && comb_l1_w1.points[0].z > l1_outer_z + 0.14,
            "CombinedLateralAndVertical must apply both lateral width stagger and vertical +0.5*h shift"
        );

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_brick_layers_overhang_and_wave_overhang_contact_and_categorisation() {
        let _guard = crate::bridge::TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        // Build an L-shaped cantilevered overhang block:
        // - Lower column (z in [0.0, 1.6] mm): x in [-5.0, +5.0], y in [-5.0, +5.0]
        // - Upper cantilevered shelf (z in [1.6, 3.2] mm): x in [-5.0, +9.0], y in [-5.0, +5.0]
        //   (so x in [+5.0, +9.0] at z = 1.6..2.0 mm is a 4.0 mm free-air cantilevered overhang
        //   requiring multiple inside-out WaveOverhang passes!).
        let v = [
            // Base z = 0.0 (0..4)
            [-5.0f32, -5.0, 0.0],
            [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0],
            [-5.0, 5.0, 0.0],
            // Step z = 1.6 (4..10)
            [-5.0, -5.0, 1.6],
            [5.0, -5.0, 1.6],
            [5.0, 5.0, 1.6],
            [-5.0, 5.0, 1.6],
            [9.0, -5.0, 1.6],
            [9.0, 5.0, 1.6],
            // Top z = 3.2 (10..14)
            [-5.0, -5.0, 3.2],
            [9.0, -5.0, 3.2],
            [9.0, 5.0, 3.2],
            [-5.0, 5.0, 3.2],
        ];
        let tris: &[[u32; 3]] = &[
            // Bottom z = 0
            [0, 2, 1],
            [0, 3, 2],
            // Top z = 3.2
            [10, 11, 12],
            [10, 12, 13],
            // Left x = -5 (full height 0..3.2 via 0,3,7,4 and 4,7,13,10)
            [0, 7, 3],
            [0, 4, 7],
            [4, 13, 7],
            [4, 10, 13],
            // Lower right x = +5 (z = 0..1.6)
            [1, 2, 6],
            [1, 6, 5],
            // Cantilevered overhang underside z = 1.6 (x = +5..+9, facing -Z!)
            [5, 6, 9],
            [5, 9, 8],
            // Upper right x = +9 (z = 1.6..3.2)
            [8, 9, 12],
            [8, 12, 11],
            // Front y = -5
            [0, 1, 5],
            [0, 5, 4],
            [4, 5, 11],
            [4, 11, 10],
            [5, 8, 11],
            // Back y = +5
            [3, 6, 2],
            [3, 7, 6],
            [7, 12, 6],
            [7, 13, 12],
            [6, 12, 9],
        ];
        let mut positions = Vec::with_capacity(v.len() * 3);
        for p in &v {
            positions.extend_from_slice(p);
        }
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for tri in tris {
            indices.extend_from_slice(tri);
        }

        let part = ffi::register_triangle_mesh("CantileverOverhangBlock", &positions, &indices);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.35;
        cfg.slicing.layer_height.min_mm = 0.20;
        cfg.slicing.layer_height.max_mm = 0.45;
        cfg.slicing.layer_height.first_layer_mm = 0.35;
        cfg.slicing.shell_thickness_mm = 1.60;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = false;

        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));
        let layers_res = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(layers_res.valid);

        for mode in crate::config::BrickLayerMode::ALL {
            cfg.slicing.brick_layer_mode = mode;
            crate::ui::sync_config_to_backend(&cfg);
            let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp.valid, "Toolpaths should be valid for mode {:?}", mode);
            assert!(
                tp.wave_overhang_count > 0,
                "Expected WaveOverhang extrusions on cantilever shelf for mode {:?}",
                mode
            );

            for layer in &tp.layers {
                let wave_paths: Vec<_> = layer
                    .paths
                    .iter()
                    .filter(|p| p.feature_kind == 6 && !p.points.is_empty())
                    .collect();
                if wave_paths.is_empty() {
                    continue;
                }
                let non_wave_paths: Vec<_> = layer
                    .paths
                    .iter()
                    .filter(|p| (p.feature_kind == 0 || p.feature_kind == 1) && !p.points.is_empty())
                    .collect();
                assert!(
                    !non_wave_paths.is_empty(),
                    "WaveOverhang layer {} must also have supported walls in mode {:?}",
                    layer.layer_index,
                    mode
                );

                // Every WaveOverhang pass `p` must be in lateral & vertical contact with at least one
                // earlier extrusion on the same layer (a supported wall for p = 1, or pass p - 1 for p > 1)
                for wp in &wave_paths {
                    let mut min_center_dist = f32::MAX;
                    let mut min_z_diff = f32::MAX;
                    for other in &layer.paths {
                        let is_valid_neighbor = (other.feature_kind == 0 || other.feature_kind == 1)
                            || (other.feature_kind == 6 && other.wall_index < wp.wall_index);
                        if !is_valid_neighbor {
                            continue;
                        }
                        for p_w in &wp.points {
                            let pw = Vec3::new(p_w.x, p_w.y, p_w.z);
                            for p_o in &other.points {
                                let po = Vec3::new(p_o.x, p_o.y, p_o.z);
                                let d = pw.distance(po);
                                if d < min_center_dist {
                                    min_center_dist = d;
                                    min_z_diff = (pw.z - po.z).abs();
                                }
                            }
                        }
                    }
                    assert!(
                        min_center_dist <= 0.85,
                        "WaveOverhang pass {} on layer {} is out of contact (min_dist = {:.3} mm) in mode {:?}",
                        wp.wall_index,
                        layer.layer_index,
                        min_center_dist,
                        mode
                    );
                    assert!(
                        min_z_diff <= 0.15,
                        "WaveOverhang pass {} on layer {} has vertical brick Z-mismatch ({:.3} mm) with adjacent extrusion in mode {:?}",
                        wp.wall_index,
                        layer.layer_index,
                        min_z_diff,
                        mode
                    );
                }
            }
        }

        let _ = ffi::remove_cad_object(part.object_id);
    }

    #[test]
    fn test_nozzle_sdf_extrusion_visualization_and_sampling() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // 1. Verify layer-height reference, signed transfer curve, non-finite guards & near-zero contrast
        let mut ref_cfg = SlicerConfig::default();
        ref_cfg.slicing.layer_height.min_mm = 0.08;
        ref_cfg.slicing.layer_height.max_mm = 0.35;
        ref_cfg.slicing.layer_height.target_mm = 0.20;
        let h_ref = nozzle_sdf_reference_layer_height_mm(&ref_cfg);
        assert!((h_ref - 0.20).abs() < 1.0e-6);
        ref_cfg.slicing.layer_height.target_mm = 0.50;
        assert!((nozzle_sdf_reference_layer_height_mm(&ref_cfg) - 0.35).abs() < 1.0e-6);
        ref_cfg.slicing.layer_height.target_mm = f32::NAN;
        assert!(nozzle_sdf_reference_layer_height_mm(&ref_cfg).is_finite());

        assert_eq!(nozzle_sdf_signed_t(f32::NAN, 0.20), 0.0);
        assert_eq!(nozzle_sdf_signed_t(0.0, 0.20), 0.0);
        assert!((nozzle_sdf_signed_t(0.05, 0.20) - (1.0 / 3.0)).abs() < 1.0e-5);
        assert!((nozzle_sdf_signed_t(-0.05, 0.20) + (1.0 / 3.0)).abs() < 1.0e-5);
        assert!((nozzle_sdf_signed_t(0.10, 0.20) - 0.5).abs() < 1.0e-5);
        assert!((nozzle_sdf_signed_t(-0.10, 0.20) + 0.5).abs() < 1.0e-5);
        assert!((nozzle_sdf_signed_t(0.20, 0.20) - (2.0 / 3.0)).abs() < 1.0e-5);
        assert!((nozzle_sdf_signed_t(-0.20, 0.20) + (2.0 / 3.0)).abs() < 1.0e-5);
        assert!(nozzle_sdf_signed_t(-0.40, 0.20) < nozzle_sdf_signed_t(-0.20, 0.20));
        assert!(nozzle_sdf_signed_t(0.40, 0.20) > nozzle_sdf_signed_t(0.20, 0.20));

        let c_nan = sample_nozzle_sdf_rgba(f32::NAN, 0.20);
        assert!(c_nan.iter().all(|c| c.is_finite()));
        let c_t_nan = sample_nozzle_sdf_rgba_for_signed_t(f32::NAN);
        assert!(c_t_nan.iter().all(|c| c.is_finite()));

        let rgb_dist = |a: [f32; 4], b: [f32; 4]| -> f32 {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };

        let c_surf = sample_nozzle_sdf_rgba(0.0, 0.20);
        assert!(
            c_surf[0] > 0.90 && c_surf[1] > 0.90 && c_surf[2] > 0.90,
            "Surface (sdf=0) should be crisp white, got {:?}",
            c_surf
        );
        let c_qtr_in = sample_nozzle_sdf_rgba(-0.05, 0.20);
        let c_qtr_out = sample_nozzle_sdf_rgba(0.05, 0.20);
        let c_one_in = sample_nozzle_sdf_rgba(-0.20, 0.20);
        let c_one_out = sample_nozzle_sdf_rgba(0.20, 0.20);
        let c_deep_in = sample_nozzle_sdf_rgba(-2.0, 0.20);
        let c_deep_out = sample_nozzle_sdf_rgba(2.0, 0.20);

        assert!(
            rgb_dist(c_qtr_in, c_surf) > 0.80,
            "-h/4 (-0.05 mm) should strongly contrast with surface white: {:?} vs {:?}",
            c_qtr_in,
            c_surf
        );
        assert!(
            rgb_dist(c_qtr_out, c_surf) > 0.80,
            "+h/4 (+0.05 mm) should strongly contrast with surface white: {:?} vs {:?}",
            c_qtr_out,
            c_surf
        );
        assert!(rgb_dist(c_qtr_in, c_one_in) > 0.35);
        assert!(rgb_dist(c_qtr_in, c_deep_in) > 0.80);
        assert!(rgb_dist(c_qtr_out, c_one_out) > 0.35);
        assert!(rgb_dist(c_qtr_out, c_deep_out) > 0.80);
        assert!(
            c_one_in[2] > c_one_in[0] + 0.25,
            "Inside solid (sdf<0) should be cool cyan/blue/indigo, got {:?}",
            c_one_in
        );
        assert!(
            c_one_out[0] > c_one_out[2] + 0.50,
            "Outside solid (sdf>0) should be warm amber/crimson, got {:?}",
            c_one_out
        );

        // 2. Register a 12x12x3 mm closed box, build SDF, and verify backend + toolpath + planned SDF
        let hx = 6.0f32;
        let hy = 6.0f32;
        let z0 = 0.0f32;
        let z1 = 3.0f32;
        let verts = [
            [-hx, -hy, z0],
            [hx, -hy, z0],
            [hx, hy, z0],
            [-hx, hy, z0],
            [-hx, -hy, z1],
            [hx, -hy, z1],
            [hx, hy, z1],
            [-hx, hy, z1],
        ];
        let tris: &[[u32; 3]] = &[
            [0, 2, 1],
            [0, 3, 2], // bottom -Z
            [4, 5, 6],
            [4, 6, 7], // top +Z
            [0, 1, 5],
            [0, 5, 4], // front -Y
            [1, 2, 6],
            [1, 6, 5], // right +X
            [2, 3, 7],
            [2, 7, 6], // back +Y
            [3, 0, 4],
            [3, 4, 7], // left -X
        ];
        let mut pos_flat = Vec::with_capacity(verts.len() * 3);
        for v in &verts {
            pos_flat.extend_from_slice(v);
        }
        let mut idx_flat = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            idx_flat.extend_from_slice(t);
        }

        let part = ffi::register_triangle_mesh("NozzleSdfBox", &pos_flat, &idx_flat);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.30;
        cfg.slicing.layer_height.min_mm = 0.20;
        cfg.slicing.layer_height.max_mm = 0.40;
        cfg.slicing.layer_height.first_layer_mm = 0.30;
        cfg.slicing.shell_thickness_mm = 1.60;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = false;

        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

        // Direct SDF batch query: interior vs surface vs outside
        let sampled = ffi::sample_object_sdf(
            part.object_id,
            &[0.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0],
        );
        assert_eq!(sampled.len(), 3);
        assert!(
            sampled[0] < -1.0,
            "Box interior SDF expected < -1.0, got {}",
            sampled[0]
        );
        assert!(
            sampled[1].abs() < 0.15,
            "Box top surface SDF expected near 0, got {}",
            sampled[1]
        );
        assert!(
            sampled[2] > 0.85,
            "Outside point SDF expected > 0.85, got {}",
            sampled[2]
        );

        let layers_res = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(layers_res.valid);
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid && !tp.layers.is_empty());

        let mut outer_sdfs = Vec::new();
        let mut deep_sdfs = Vec::new();
        for layer in &tp.layers {
            for path in &layer.paths {
                for pt in &path.points {
                    assert!(pt.sdf_mm.is_finite());
                    if path.feature_kind == 0 {
                        outer_sdfs.push(pt.sdf_mm);
                    } else if path.feature_kind == 1 || path.feature_kind == 4 {
                        deep_sdfs.push(pt.sdf_mm);
                    }
                }
            }
        }
        assert!(!outer_sdfs.is_empty() && !deep_sdfs.is_empty());
        let mean_outer = outer_sdfs.iter().sum::<f32>() / outer_sdfs.len() as f32;
        let min_deep = deep_sdfs.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            mean_outer < -0.05 && mean_outer > -0.60,
            "OuterWall mean SDF should sit near -0.5*W inside CAD, got {}",
            mean_outer
        );
        assert!(
            min_deep < mean_outer - 0.15,
            "InnerWall/Infill should sample deeper negative SDF ({}) than OuterWall ({})",
            min_deep,
            mean_outer
        );

        // 3. Plan G-code, refresh exact OpenVDB nozzle SDF, and test Fast + HQ + Scrubbing
        let mut planned = crate::gcode::plan_and_emit_gcode_for_object_with_progress(
            "NozzleSdfBox",
            part.object_id,
            &tp,
            Transform::IDENTITY,
            &cfg,
            None,
        );
        crate::gcode::refresh_planned_gcode_nozzle_sdf(&mut planned, |xyz| {
            ffi::sample_object_sdf(part.object_id, xyz)
        });
        assert!(planned.min_sdf_mm.is_finite() && planned.max_sdf_mm.is_finite());
        assert!(
            planned.min_sdf_mm < -0.4,
            "Expected deep negative planned min_sdf_mm, got {}",
            planned.min_sdf_mm
        );

        let mut vis = FieldVisualizationState::default();
        vis.show_toolpaths = true;
        vis.toolpath_color_mode = ToolpathColorMode::ByNozzleSdf;
        vis.selected_layer_min_idx = 0;
        vis.selected_layer_max_idx = planned.layers.len().saturating_sub(1);
        vis.cached_planned_gcode = Some(planned);

        // Build Fast and HQ meshes
        let BuiltToolpathGeometry::Fast {
            positions: fast_pos,
            colors: fast_colors,
            ..
        } = vis.toolpath_mesh_filter().build_geometry(
            vis.cached_planned_gcode.as_ref(),
            None,
            &cfg,
        )
        else {
            panic!("Expected BuiltToolpathGeometry::Fast");
        };
        assert!(!fast_pos.is_empty() && fast_colors.len() == fast_pos.len());
        let first_c = fast_colors[0];
        assert!(
            fast_colors
                .iter()
                .any(|c| (c[0] - first_c[0]).abs() > 1.0e-3 || (c[2] - first_c[2]).abs() > 1.0e-3),
            "Fast ByNozzleSdf mesh should exhibit varying per-vertex SDF colors"
        );

        vis.toolpath_quality_mode = ToolpathQualityMode::HighQualityPbr;
        let BuiltToolpathGeometry::HighQualityPbr(hq_buckets) = vis
            .toolpath_mesh_filter()
            .build_geometry(vis.cached_planned_gcode.as_ref(), None, &cfg)
        else {
            panic!("Expected BuiltToolpathGeometry::HighQualityPbr");
        };
        assert!(!hq_buckets.is_empty() && !hq_buckets[0].positions.is_empty());

        // Zero-crossing midpoint ring split test (-0.5 -> 0.0 -> +0.5 should emit 3 rings = 12 verts)
        let cross_seg = crate::gcode::PlannedSegment {
            p0_local: Vec3::new(0.0, 0.0, 1.0),
            p1_local: Vec3::new(4.0, 0.0, 1.0),
            p0_bed: Vec3::new(0.0, 0.0, 1.0),
            p1_bed: Vec3::new(4.0, 0.0, 1.0),
            nozzle_local_0: Vec3::Z,
            nozzle_local_1: Vec3::Z,
            nozzle_bed_0: Vec3::Z,
            nozzle_bed_1: Vec3::Z,
            b0_deg: 0.0,
            c0_deg: 0.0,
            b1_deg: 0.0,
            c1_deg: 0.0,
            kind: crate::gcode::PlannedMoveKind::OuterWall,
            tool_id: 0,
            filament_slot: 0,
            wall_index: 0,
            width_mm: 0.42,
            height_mm: 0.30,
            length_mm: 4.0,
            nominal_speed_mms: 60.0,
            entry_speed_mms: 60.0,
            cruise_speed_mms: 60.0,
            exit_speed_mms: 60.0,
            accel_mms2: 1500.0,
            duration_s: 0.06,
            delta_e_mm: 0.2,
            volumetric_flow_mm3s: 7.5,
            extrusion_multiplier: 1.0,
            pressure_advance_s: 0.02,
            melt_pressure_mm3s: 7.5,
            retract_before_mm: 0.0,
            unretract_after_mm: 0.0,
            tool_iso_angle_deg: 0.0,
            uses_surface_normal: false,
            sdf_mm_0: -0.5,
            sdf_mm: 0.0,
            sdf_mm_1: 0.5,
        };
        let mut seg_pos = Vec::new();
        let mut seg_nrm = Vec::new();
        let mut seg_col = Vec::new();
        let mut seg_ind = Vec::new();
        append_planned_segment_bead_mesh(
            &cross_seg,
            vis.cached_planned_gcode.as_ref().unwrap(),
            ToolpathColorMode::ByNozzleSdf,
            None,
            &cfg,
            1.0,
            &mut seg_pos,
            &mut seg_nrm,
            &mut seg_col,
            &mut seg_ind,
        );
        assert_eq!(
            seg_pos.len(),
            12,
            "Zero-crossing segment should insert a midpoint ring (3 rings * 4 = 12 verts)"
        );

        // Same-side non-linear near-zero transition (-h -> -h/2 -> 0) also splits (3 rings = 12 verts)
        let mut same_side_seg = cross_seg.clone();
        same_side_seg.sdf_mm_0 = -0.30;
        same_side_seg.sdf_mm = -0.15;
        same_side_seg.sdf_mm_1 = 0.0;
        seg_pos.clear();
        seg_nrm.clear();
        seg_col.clear();
        seg_ind.clear();
        append_planned_segment_bead_mesh(
            &same_side_seg,
            vis.cached_planned_gcode.as_ref().unwrap(),
            ToolpathColorMode::ByNozzleSdf,
            None,
            &cfg,
            1.0,
            &mut seg_pos,
            &mut seg_nrm,
            &mut seg_col,
            &mut seg_ind,
        );
        assert_eq!(
            seg_pos.len(),
            12,
            "Single-sided non-linear gradient (-h -> -h/2 -> 0) should split with midpoint ring"
        );

        // Constant deep-interior SDF segment does not need midpoint ring (2 rings = 8 verts)
        let mut flat_seg = cross_seg;
        flat_seg.sdf_mm_0 = -1.5;
        flat_seg.sdf_mm = -1.5;
        flat_seg.sdf_mm_1 = -1.5;
        seg_pos.clear();
        seg_nrm.clear();
        seg_col.clear();
        seg_ind.clear();
        append_planned_segment_bead_mesh(
            &flat_seg,
            vis.cached_planned_gcode.as_ref().unwrap(),
            ToolpathColorMode::ByNozzleSdf,
            None,
            &cfg,
            1.0,
            &mut seg_pos,
            &mut seg_nrm,
            &mut seg_col,
            &mut seg_ind,
        );
        assert_eq!(
            seg_pos.len(),
            8,
            "Constant-SDF segment should not add an unnecessary midpoint ring"
        );

        // Verify scrubbed toolhead pose reports endpoint sdf_mm
        let pose = vis
            .compute_active_scrub_pose(&cfg, Transform::IDENTITY)
            .expect("Expected valid scrub pose");
        assert!(pose.sdf_mm.is_finite() && pose.sdf_mm <= 0.1);

        let _ = ffi::remove_cad_object(part.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_hq_stadium_acute_hairpin_corner_splits_without_fold() {
        // Use a non-Z tilted local nozzle frame (AGENTS.md Rule 1) and short acute segments
        let z_noz = Vec3::new(0.25, -0.15, 0.95).normalize();
        let frame = NozzleFrame::from_nozzle_and_travel(z_noz, Vec3::X);
        let x_ax = frame.travel_axis;
        let y_ax = frame.lateral_axis;
        let origin = Vec3::new(1.0, 2.0, 3.0);

        let p0 = origin;
        let p1 = p0 + 1.5 * x_ax;
        let p2 = p1 + 0.09 * x_ax;
        let dir_back = (-0.7071 * x_ax + 0.7071 * y_ax).normalize();
        let p3 = p2 + 0.15 * dir_back;
        let p4 = p3 + 1.5 * dir_back;

        let make_node = |pos: Vec3| HqChainNode {
            pos,
            nozzle_axis: z_noz,
            width_mm: 0.40,
            height_mm: 0.20,
            linear_rgba: [0.2, 0.7, 0.95, 1.0],
            is_circular: false,
        };
        let nodes = vec![
            make_node(p0),
            make_node(p1),
            make_node(p2),
            make_node(p3),
            make_node(p4),
        ];

        let mut acc = HqBucketAccum::default();
        append_hq_welded_chain_stadium_mesh(&nodes, z_noz, &mut acc);

        assert!(!acc.positions.is_empty() && !acc.indices.is_empty());
        assert!(
            acc.positions.len() > nodes.len() * HQ_RING_VERTS + 2,
            "Acute ~135° hairpin on short segments must split into capped sub-chains (got {} verts)",
            acc.positions.len()
        );
        for p in &acc.positions {
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
        }
        for n in &acc.normals {
            assert!(n[0].is_finite() && n[1].is_finite() && n[2].is_finite());
        }
        for tri in acc.indices.chunks_exact(3) {
            let v0 = Vec3::from_array(acc.positions[tri[0] as usize]);
            let v1 = Vec3::from_array(acc.positions[tri[1] as usize]);
            let v2 = Vec3::from_array(acc.positions[tri[2] as usize]);
            let face_n = (v1 - v0).cross(v2 - v0);
            assert!(face_n.length() > 1.0e-7, "Degenerate triangle in split hairpin mesh");
            let avg_n = Vec3::from_array(acc.normals[tri[0] as usize])
                + Vec3::from_array(acc.normals[tri[1] as usize])
                + Vec3::from_array(acc.normals[tri[2] as usize]);
            assert!(
                face_n.dot(avg_n) > 0.0,
                "Hairpin miter split must produce strictly outward-wound triangles"
            );
        }
    }

    #[test]
    fn test_concave_corner_sdf_not_filled_by_renormalization() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Watertight L-step bracket pre-centered at X=0, Y=0 so `recenterVertexBuffer` is identity:
        // [-6,6]x[-4,4]x[0,4.10] U [-6,-0.10]x[-4,4]x[4.10,10], concave corner at x=-0.10, z=4.10
        // (intentionally off the 0.25 mm voxel planes so voxel (0.0, 0.0, 4.25) has true SDF +0.10 mm).
        let verts: &[[f32; 3]] = &[
            [-6.0, -4.0, 0.0],  // 0
            [6.0, -4.0, 0.0],   // 1
            [6.0, -4.0, 4.10],  // 2
            [-0.10, -4.0, 4.10], // 3 (concave corner)
            [-0.10, -4.0, 10.0], // 4
            [-6.0, -4.0, 10.0], // 5
            [-6.0, 4.0, 0.0],   // 6
            [6.0, 4.0, 0.0],    // 7
            [6.0, 4.0, 4.10],   // 8
            [-0.10, 4.0, 4.10], // 9 (concave corner)
            [-0.10, 4.0, 10.0], // 10
            [-6.0, 4.0, 10.0],  // 11
        ];
        let tris: &[[u32; 3]] = &[
            // Front -Y
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 5],
            // Back +Y
            [6, 8, 7],
            [6, 9, 8],
            [6, 10, 9],
            [6, 11, 10],
            // Bottom -Z
            [0, 6, 7],
            [0, 7, 1],
            // Right lower +X
            [1, 7, 8],
            [1, 8, 2],
            // Deck +Z
            [2, 8, 9],
            [2, 9, 3],
            // Riser +X
            [3, 9, 10],
            [3, 10, 4],
            // Top +Z
            [4, 10, 11],
            [4, 11, 5],
            // Left -X
            [5, 11, 6],
            [5, 6, 0],
        ];
        let mut pos_flat = Vec::with_capacity(verts.len() * 3);
        for v in verts {
            pos_flat.extend_from_slice(v);
        }
        let mut idx_flat = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            idx_flat.extend_from_slice(t);
        }

        let part = ffi::register_triangle_mesh("ConcaveLStep", &pos_flat, &idx_flat);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.min_mm = 0.16;
        cfg.slicing.layer_height.max_mm = 0.24;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = false;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

        // 1. Exterior concave corner + adjacent interior near-surface voxels must all have exact signs
        let sdfs = ffi::sample_object_sdf(
            part.object_id,
            &[
                0.0, 0.0, 4.25,   // exterior concave-corner voxel center (true SDF +0.10 mm)
                -0.15, 0.0, 4.25, // interior riser face (true SDF -0.05 mm)
                0.0, 0.0, 4.00,   // interior deck face near corner (true SDF -0.10 mm)
                0.25, 0.0, 4.00,  // interior deck face (true SDF -0.10 mm)
            ],
        );
        assert_eq!(sdfs.len(), 4);
        assert!(
            sdfs[0] > 0.08,
            "Exterior concave pocket (0.0, 0.0, 4.25) must remain positive (> +0.08 mm), got {}",
            sdfs[0]
        );
        assert!(
            sdfs[1] < -0.01 && sdfs[2] < -0.01 && sdfs[3] < -0.01,
            "Interior near-surface voxels around concave corner must remain strictly negative, got {:?}",
            &sdfs[1..]
        );

        // 2. OuterWall paths above the deck must not bulge outside the x=-0.10 riser face
        let layers_res = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(layers_res.valid);
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid && !tp.layers.is_empty());

        let mut checked_riser_pts = 0usize;
        for layer in &tp.layers {
            for path in &layer.paths {
                if path.feature_kind != 0 {
                    continue;
                }
                for pt in &path.points {
                    if pt.z > 4.35 && pt.z < 9.6 && pt.x > -1.0 && pt.y > -2.5 && pt.y < 2.5 {
                        checked_riser_pts += 1;
                        assert!(
                            pt.x <= -0.065,
                            "Concave-corner OuterWall must not bulge outside x=-0.10 at z={:.2}: x={:.4}",
                            pt.z,
                            pt.x
                        );
                    }
                }
            }
        }
        assert!(checked_riser_pts > 0, "Expected OuterWall points along riser x=-0.10");

        let _ = ffi::remove_cad_object(part.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_slope_compensation_conformal_top_and_underside_containment() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Watertight cantilevered ramp prism pre-centered at X=0, Y=0 (Y in [-6, 6]):
        // Pillar [-8,0]x[0,4], cantilevered ceiling [0,8] at Z=4, tip wall X=8 Z in [4,6],
        // ~20 deg ramp (8,6)->(2,8.184), flat plateau [-8,2] at Z=8.184.
        let z_top = 8.184f32;
        let verts: &[[f32; 3]] = &[
            [-8.0, -6.0, 0.0],   // 0
            [0.0, -6.0, 0.0],    // 1
            [0.0, -6.0, 4.0],    // 2 (reflex)
            [8.0, -6.0, 4.0],    // 3
            [8.0, -6.0, 6.0],    // 4
            [2.0, -6.0, z_top],  // 5
            [-8.0, -6.0, z_top], // 6
            [-8.0, 6.0, 0.0],    // 7
            [0.0, 6.0, 0.0],     // 8
            [0.0, 6.0, 4.0],     // 9 (reflex)
            [8.0, 6.0, 4.0],     // 10
            [8.0, 6.0, 6.0],     // 11
            [2.0, 6.0, z_top],   // 12
            [-8.0, 6.0, z_top],  // 13
        ];
        let tris: &[[u32; 3]] = &[
            // Front -Y
            [2, 3, 4],
            [2, 4, 5],
            [2, 5, 6],
            [2, 6, 0],
            [2, 0, 1],
            // Back +Y
            [9, 11, 10],
            [9, 12, 11],
            [9, 13, 12],
            [9, 7, 13],
            [9, 8, 7],
            // Sides
            [0, 7, 8],
            [0, 8, 1],
            [1, 8, 9],
            [1, 9, 2],
            [2, 9, 10],
            [2, 10, 3],
            [3, 10, 11],
            [3, 11, 4],
            [4, 11, 12],
            [4, 12, 5],
            [5, 12, 13],
            [5, 13, 6],
            [6, 13, 7],
            [6, 7, 0],
        ];
        let mut pos_flat = Vec::with_capacity(verts.len() * 3);
        for v in verts {
            pos_flat.extend_from_slice(v);
        }
        let mut idx_flat = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            idx_flat.extend_from_slice(t);
        }

        let part = ffi::register_triangle_mesh("SlopeAndCeilingPrism", &pos_flat, &idx_flat);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.min_mm = 0.16;
        cfg.slicing.layer_height.max_mm = 0.24;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = true;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

        // Phase 1: conformal skin enabled -> outermost flat top skin near 0.0 mm & ceiling containment
        let layers_res = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(layers_res.valid);
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid && !tp.layers.is_empty());

        let mut top_z_errs = Vec::new();
        let mut max_top_layer = 0usize;
        let mut top_candidates: Vec<(usize, f32)> = Vec::new();
        let mut ceil_min_z = f32::INFINITY;
        let mut ceil_xyz = Vec::new();

        for (li, layer) in tp.layers.iter().enumerate() {
            for path in &layer.paths {
                for pt in &path.points {
                    if path.feature_kind == 2
                        && pt.x >= -6.5
                        && pt.x <= 0.5
                        && pt.y >= -4.0
                        && pt.y <= 4.0
                        && pt.z >= z_top - 0.12
                    {
                        max_top_layer = max_top_layer.max(li);
                        top_candidates.push((li, pt.z - z_top));
                    }
                    if pt.x >= 1.0
                        && pt.x <= 7.0
                        && pt.y >= -4.0
                        && pt.y <= 4.0
                        && pt.z >= 3.7
                        && pt.z <= 4.45
                    {
                        ceil_min_z = ceil_min_z.min(pt.z);
                        ceil_xyz.push(pt.x);
                        ceil_xyz.push(pt.y);
                        ceil_xyz.push(pt.z);
                    }
                }
            }
        }
        for (li, dz) in top_candidates {
            if li == max_top_layer {
                top_z_errs.push(dz);
            }
        }
        assert!(!top_z_errs.is_empty(), "Expected outermost top plateau conformal skin points");
        top_z_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let top_med = top_z_errs[top_z_errs.len() / 2];
        assert!(
            top_med.abs() < 0.025,
            "Outermost conformal top skin should sit near CAD surface (0.0 mm), got median {} mm",
            top_med
        );

        assert!(!ceil_xyz.is_empty(), "Expected underside ceiling overhang paths");
        assert!(
            ceil_min_z >= 4.0 + 0.70 * 0.20,
            "Cantilevered underside Z=4.0 must keep nozzle >= 0.70*h inside, got min z={}",
            ceil_min_z
        );
        let ceil_sdfs = ffi::sample_object_sdf(part.object_id, &ceil_xyz);
        let max_ceil_sdf = ceil_sdfs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_ceil_sdf <= 0.02,
            "Cantilevered underside paths must stay contained in CAD SDF, got max {}",
            max_ceil_sdf
        );

        // Phase 2: conformal skin disabled (reusing existing SDF) -> planar OuterWall on 20 deg ramp
        cfg.slicing.enable_auto_conformal_skin = false;
        crate::ui::sync_config_to_backend(&cfg);
        let layers_planar = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(layers_planar.valid);
        let tp_planar = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp_planar.valid && !tp_planar.layers.is_empty());

        let slope_angle = (z_top - 6.0f32).atan2(6.0);
        let sin_s = slope_angle.sin();
        let cos_s = slope_angle.cos();
        let mut slope_sdfs = Vec::new();
        for layer in &tp_planar.layers {
            for path in &layer.paths {
                if path.feature_kind != 0 {
                    continue;
                }
                for pt in &path.points {
                    if pt.x >= 2.8
                        && pt.x <= 7.4
                        && pt.y >= -4.0
                        && pt.y <= 4.0
                        && pt.z >= 6.2
                        && pt.z <= z_top - 0.15
                    {
                        let d_slope = (pt.x - 2.0) * sin_s + (pt.z - z_top) * cos_s;
                        slope_sdfs.push(d_slope);
                    }
                }
            }
        }
        assert!(!slope_sdfs.is_empty(), "Expected ~20 deg sloped OuterWall points");
        slope_sdfs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let slope_med = slope_sdfs[slope_sdfs.len() / 2];
        assert!(
            slope_med > -0.10 && slope_med < 0.02,
            "20 deg upward sloped OuterWall should follow -H_tip (~-0.04 mm, not -0.20 mm), got median {}",
            slope_med
        );

        let _ = ffi::remove_cad_object(part.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_top_surface_lofting_smooths_shallow_slope_and_scales_extrusion() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Helper: builds a watertight 10-deg sloped wedge (`lx in [-12, 12]`, `ly in [-8, 8]`,
        // bottom `lz = 0`, top `lz = 4.0 + (lx + 12) * tan(10 deg)`), optionally rigidly rotated
        // around +Y by `tilt_deg` and shifted in +Z by `z_offset` to verify NozzleFrame invariance.
        let run_case = |tilt_deg: f32, z_offset: f32| {
            let slope_rad = 10.0f32.to_radians();
            let z_slope_max = 4.0 + 24.0 * slope_rad.tan();
            let local_verts: [[f32; 3]; 8] = [
                [-12.0, -8.0, 0.0],
                [12.0, -8.0, 0.0],
                [12.0, 8.0, 0.0],
                [-12.0, 8.0, 0.0],
                [-12.0, -8.0, 4.0],
                [12.0, -8.0, z_slope_max],
                [12.0, 8.0, z_slope_max],
                [-12.0, 8.0, 4.0],
            ];
            let tris: [u32; 36] = [
                0, 2, 1, 0, 3, 2, // bottom
                4, 5, 6, 4, 6, 7, // sloped top (10 deg)
                0, 1, 5, 0, 5, 4, // -Y
                1, 2, 6, 1, 6, 5, // +X
                2, 3, 7, 2, 7, 6, // +Y
                3, 0, 4, 3, 4, 7, // -X
            ];

            let th = tilt_deg.to_radians();
            let (st, ct) = th.sin_cos();
            let u_x = Vec3::new(ct, 0.0, -st);
            let u_y = Vec3::Y;
            let u_z = Vec3::new(st, 0.0, ct);
            let origin = Vec3::new(0.0, 0.0, z_offset);

            let mut positions = Vec::with_capacity(24);
            for lv in &local_verts {
                let p = origin + u_x * lv[0] + u_y * lv[1] + u_z * lv[2];
                positions.extend_from_slice(&[p.x, p.y, p.z]);
            }

            let part = ffi::register_triangle_mesh(
                &format!("loft_wedge_{:.0}", tilt_deg),
                &positions,
                &tris,
            );
            assert!(part.object_id > 0);

            let mut cfg = SlicerConfig::default();
            cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
            cfg.eikonal.voxel_size_mm = 0.25;
            cfg.eikonal.enable_multi_resolution_grid = false;
            cfg.machine.nozzle_axis_bed = [u_z.x, u_z.y, u_z.z];
            cfg.machine.gravity_direction_bed = [-u_z.x, -u_z.y, -u_z.z];
            cfg.slicing.layer_height.target_mm = 0.20;
            cfg.slicing.layer_height.min_mm = 0.08;
            cfg.slicing.layer_height.max_mm = 0.36;
            cfg.slicing.layer_height.first_layer_mm = 0.20;
            cfg.slicing.inner_wall_count = 1;
            cfg.slicing.shell_thickness_mm = 2.0;
            cfg.slicing.infill_density_percent = 20.0;
            cfg.slicing.enable_support_generation = false;
            cfg.slicing.enable_auto_conformal_skin = false;
            cfg.slicing.enable_top_surface_lofting = false;
            crate::ui::sync_config_to_backend(&cfg);
            assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

            let layers_off = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
            assert!(layers_off.valid && !layers_off.layers.is_empty());
            let tp_off = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp_off.valid);

            cfg.slicing.enable_top_surface_lofting = true;
            crate::ui::sync_config_to_backend(&cfg);
            let layers_on = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
            assert!(layers_on.valid && !layers_on.layers.is_empty());
            let tp_on = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp_on.valid);

            let effective_origin = origin - Vec3::new(part.origin_x, part.origin_y, part.origin_z);
            let to_local = |x: f32, y: f32, z: f32| -> (f32, f32, f32) {
                let rel = Vec3::new(x, y, z) - effective_origin;
                (rel.dot(u_x), rel.dot(u_y), rel.dot(u_z))
            };
            let z_top_local = |lx: f32| -> f32 { 4.0 + (lx + 12.0) * slope_rad.tan() };
            let cos_s = slope_rad.cos();

            let collect_stats = |tp: &ffi::ObjectToolpathsResult| {
                let mut iw_exp_sdfs = Vec::new();
                let mut iw_exp_heights = Vec::new();
                let mut si_exp_sdfs = Vec::new();
                let mut si_exp_heights = Vec::new();
                let mut covered_heights = Vec::new();
                let mut max_slope_sdf = f32::NEG_INFINITY;
                let mut max_height = 0.0f32;

                for layer in &tp.layers {
                    for path in &layer.paths {
                        if path.feature_kind != 1 && path.feature_kind != 2 {
                            continue;
                        }
                        for pt in &path.points {
                            let (lx, ly, lz) = to_local(pt.x, pt.y, pt.z);
                            if lx.abs() > 10.0 || ly.abs() > 6.0 || lz < 4.1 {
                                continue;
                            }
                            let z_top = z_top_local(lx);
                            let dz_below = z_top - lz;
                            let sdf_plane = (lz - z_top) * cos_s;
                            max_slope_sdf = max_slope_sdf.max(sdf_plane);
                            max_height = max_height.max(pt.height_mm);
                            assert!(
                                pt.height_mm <= cfg.slicing.layer_height.max_mm + 1.0e-4,
                                "Lofted height_mm ({}) must not exceed layer_height.max_mm ({})",
                                pt.height_mm,
                                cfg.slicing.layer_height.max_mm
                            );
                            if tilt_deg > 1.0 {
                                let noz_dot = Vec3::new(pt.nx, pt.ny, pt.nz).dot(u_z);
                                assert!(
                                    noz_dot > 0.995,
                                    "Expected local nozzle axis aligned with tilted u_z, got dot={noz_dot}"
                                );
                            }
                            // Interior covered points well beneath Layer k+1:
                            if dz_below >= 0.35 && lz >= 4.4 && lx.abs() <= 9.0 && ly.abs() <= 5.0 {
                                covered_heights.push(pt.height_mm);
                            }
                        }
                    }
                }

                // Collect exposed terrace points by scanning layers where the unlofted layer sits
                // inside `0.05 < z_top(lx) - lz_layer < 0.155` (strictly outside Layer k+1).
                for (_li, layer) in tp.layers.iter().enumerate() {
                    let mut ow_lzs: Vec<f32> = layer
                        .paths
                        .iter()
                        .filter(|p| p.feature_kind == 0)
                        .flat_map(|p| p.points.iter())
                        .map(|pt| to_local(pt.x, pt.y, pt.z).2)
                        .collect();
                    if ow_lzs.is_empty() {
                        continue;
                    }
                    ow_lzs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let layer_lz = ow_lzs[ow_lzs.len() / 2];
                    if layer_lz < 4.5 || layer_lz > z_slope_max - 0.6 {
                        continue;
                    }
                    for path in &layer.paths {
                        if path.feature_kind != 1 && path.feature_kind != 2 {
                            continue;
                        }
                        for pt in &path.points {
                            let (lx, ly, lz) = to_local(pt.x, pt.y, pt.z);
                            if lx.abs() > 10.0 || ly.abs() > 6.0 {
                                continue;
                            }
                            let z_top = z_top_local(lx);
                            let base_dz_below = z_top - layer_lz;
                            // Inside the unlofted exposed terrace band where wSlope = wExp = 1.0
                            if base_dz_below > 0.05 && base_dz_below < 0.155 {
                                let sdf_plane = (lz - z_top) * cos_s;
                                if path.feature_kind == 1 {
                                    iw_exp_sdfs.push(sdf_plane);
                                    iw_exp_heights.push(pt.height_mm);
                                } else {
                                    si_exp_sdfs.push(sdf_plane);
                                    si_exp_heights.push(pt.height_mm);
                                }
                            }
                        }
                    }
                }
                (
                    iw_exp_sdfs,
                    iw_exp_heights,
                    si_exp_sdfs,
                    si_exp_heights,
                    covered_heights,
                    max_slope_sdf,
                    max_height,
                )
            };

            let (iw_s_off, iw_h_off, si_s_off, si_h_off, _, _, _) = collect_stats(&tp_off);
            let (iw_s_on, iw_h_on, si_s_on, si_h_on, cov_h_on, max_sdf_on, _max_h_on) =
                collect_stats(&tp_on);

            assert!(
                !iw_s_off.is_empty() && !iw_s_on.is_empty(),
                "Expected exposed InnerWall samples at tilt={tilt_deg}°"
            );
            assert!(
                !si_s_off.is_empty() && !si_s_on.is_empty(),
                "Expected exposed SolidInfill samples at tilt={tilt_deg}°"
            );
            assert!(!cov_h_on.is_empty(), "Expected covered interior points at tilt={tilt_deg}°");

            let mean = |v: &[f32]| -> f32 { v.iter().sum::<f32>() / (v.len() as f32) };
            let iw_sdf_off = mean(&iw_s_off);
            let iw_sdf_on = mean(&iw_s_on);
            let si_sdf_off = mean(&si_s_off);
            let si_sdf_on = mean(&si_s_on);
            let iw_h_mean_on = mean(&iw_h_on);
            let si_h_mean_on = mean(&si_h_on);
            let cov_h_mean_on = mean(&cov_h_on);
            let _ = (mean(&iw_h_off), mean(&si_h_off));

            assert!(
                iw_sdf_on - iw_sdf_off >= 0.04,
                "Exposed InnerWall SDF should improve by >= 0.04 mm (tilt={tilt_deg}°): off={iw_sdf_off:.4}, on={iw_sdf_on:.4}"
            );
            assert!(
                si_sdf_on - si_sdf_off >= 0.05,
                "Exposed SolidInfill SDF should improve by >= 0.05 mm (tilt={tilt_deg}°): off={si_sdf_off:.4}, on={si_sdf_on:.4}"
            );
            assert!(
                iw_h_mean_on > 0.24 && si_h_mean_on > 0.26,
                "Expected lofted bead height > target_mm (tilt={tilt_deg}°): iw={iw_h_mean_on:.4}, si={si_h_mean_on:.4}"
            );
            assert!(
                max_sdf_on <= 0.025,
                "Lofted vertices must not protrude above CAD surface (tilt={tilt_deg}°): max_sdf={max_sdf_on:.4}"
            );
            assert!(
                (cov_h_mean_on - 0.20).abs() < 0.015,
                "Covered interior vertices must remain unlofted near 0.20 mm (tilt={tilt_deg}°): got {cov_h_mean_on:.4}"
            );

            // Plan G-code and verify extruded filament volume (`delta_e_mm`) increases cleanly
            // for both InnerWall and SolidInfill when top-surface lofting is enabled.
            cfg.slicing.enable_top_surface_lofting = false;
            let planned_off = crate::gcode::plan_and_emit_gcode_for_object(
                "loft_wedge",
                part.object_id,
                &tp_off,
                Transform::IDENTITY,
                &cfg,
            );
            cfg.slicing.enable_top_surface_lofting = true;
            let planned_on = crate::gcode::plan_and_emit_gcode_for_object(
                "loft_wedge",
                part.object_id,
                &tp_on,
                Transform::IDENTITY,
                &cfg,
            );

            let sum_e = |p: &crate::gcode::PlannedGcodeResult, kind: crate::gcode::PlannedMoveKind| -> f32 {
                p.layers
                    .iter()
                    .flat_map(|l| l.segments.iter())
                    .filter(|s| s.kind == kind && s.delta_e_mm > 0.0)
                    .map(|s| s.delta_e_mm)
                    .sum()
            };
            let iw_e_off = sum_e(&planned_off, crate::gcode::PlannedMoveKind::InnerWall);
            let iw_e_on = sum_e(&planned_on, crate::gcode::PlannedMoveKind::InnerWall);
            let si_e_off = sum_e(&planned_off, crate::gcode::PlannedMoveKind::SolidInfill);
            let si_e_on = sum_e(&planned_on, crate::gcode::PlannedMoveKind::SolidInfill);
            assert!(
                iw_e_on > iw_e_off && si_e_on > si_e_off,
                "Expected lofted G-code extruded filament delta_e to increase (tilt={tilt_deg}°): InnerWall {iw_e_off:.3}->{iw_e_on:.3}, SolidInfill {si_e_off:.3}->{si_e_on:.3}"
            );

            let _ = ffi::remove_cad_object(part.object_id);
        };

        run_case(0.0, 0.0);
        run_case(20.0, 10.0);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_top_surface_lofting_respects_conformal_skin_caps() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // 18-deg sloped wedge (`18° > 12°` auto conformal skin trigger) with both conformal skin
        // and top-surface lofting enabled. Core layers beneath the conformal skin cap must not
        // loft up into the conformal skin patch.
        let slope_rad = 18.0f32.to_radians();
        let z_slope_max = 4.0 + 24.0 * slope_rad.tan();
        let verts: &[[f32; 3]] = &[
            [-12.0, -8.0, 0.0],
            [12.0, -8.0, 0.0],
            [12.0, 8.0, 0.0],
            [-12.0, 8.0, 0.0],
            [-12.0, -8.0, 4.0],
            [12.0, -8.0, z_slope_max],
            [12.0, 8.0, z_slope_max],
            [-12.0, 8.0, 4.0],
        ];
        let tris: &[[u32; 3]] = &[
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let mut pos_flat = Vec::with_capacity(verts.len() * 3);
        for v in verts {
            pos_flat.extend_from_slice(v);
        }
        let mut idx_flat = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            idx_flat.extend_from_slice(t);
        }

        let part = ffi::register_triangle_mesh("ConformalLoftCoexistWedge", &pos_flat, &idx_flat);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.min_mm = 0.08;
        cfg.slicing.layer_height.max_mm = 0.36;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.inner_wall_count = 2;
        cfg.slicing.shell_thickness_mm = 2.0;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.enable_top_surface_lofting = true;
        crate::ui::sync_config_to_backend(&cfg);
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

        let sel = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
        assert!(sel.valid);
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid && sel.layers.len() == tp.layers.len());

        let z_top = |x: f32| -> f32 { 4.0 + (x + 12.0) * slope_rad.tan() };
        let cos_s = slope_rad.cos();
        let mut skin_layer_count = 0usize;
        let mut checked_core_pts = 0usize;
        let mut max_core_sdf = f32::NEG_INFINITY;

        for (li, layer) in tp.layers.iter().enumerate() {
            let info = &sel.layers[li];
            let is_uniform = (info.max_thickness_mm - info.min_thickness_mm).abs() <= 1.0e-4;
            let is_skin = is_uniform && info.is_top_layer && !info.is_first_layer;
            if is_skin {
                skin_layer_count += 1;
                for path in &layer.paths {
                    for pt in &path.points {
                        assert!(
                            pt.height_mm <= 0.23,
                            "Conformal skin layers should keep conformal thickness <= 0.23 mm, got {}",
                            pt.height_mm
                        );
                    }
                }
                continue;
            }
            if info.is_first_layer {
                continue;
            }
            for path in &layer.paths {
                if path.feature_kind != 1 && path.feature_kind != 2 && path.feature_kind != 3 {
                    continue;
                }
                for pt in &path.points {
                    if pt.x.abs() <= 8.0 && pt.y.abs() <= 4.0 && pt.z >= 4.0 {
                        let sdf_plane = (pt.z - z_top(pt.x)) * cos_s;
                        max_core_sdf = max_core_sdf.max(sdf_plane);
                        checked_core_pts += 1;
                    }
                }
            }
        }

        assert!(
            skin_layer_count >= 3,
            "Expected >= 3 conformal skin layers on 18-deg wedge, got {skin_layer_count}"
        );
        assert!(checked_core_pts > 0, "Expected core InnerWall/SolidInfill points beneath skin cap");
        assert!(
            max_core_sdf <= -0.35,
            "Core layers beneath conformal skin cap must not loft into the skin cap, got max_core_sdf={max_core_sdf:.4}"
        );

        let _ = ffi::remove_cad_object(part.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_plateau_pinning_and_flat_deck_solid_infill() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Exact signed distance helper for a closed watertight triangle mesh (`local` frame)
        let tri_mesh_signed_dist = |verts: &[[f32; 3]], tris: &[[u32; 3]], q: Vec3| -> f32 {
            let closest_pt_tri = |p: Vec3, a: Vec3, b: Vec3, c: Vec3| -> (f32, Vec3) {
                let ab = b - a;
                let ac = c - a;
                let ap = p - a;
                let d1 = ab.dot(ap);
                let d2 = ac.dot(ap);
                if d1 <= 0.0 && d2 <= 0.0 {
                    return (p.distance_squared(a), a);
                }
                let bp = p - b;
                let d3 = ab.dot(bp);
                let d4 = ac.dot(bp);
                if d3 >= 0.0 && d4 <= d3 {
                    return (p.distance_squared(b), b);
                }
                let vc = d1 * d4 - d3 * d2;
                if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                    let v = d1 / (d1 - d3);
                    let cp = a + ab * v;
                    return (p.distance_squared(cp), cp);
                }
                let cp_v = p - c;
                let d5 = ab.dot(cp_v);
                let d6 = ac.dot(cp_v);
                if d6 >= 0.0 && d5 <= d6 {
                    return (p.distance_squared(c), c);
                }
                let vb = d5 * d2 - d1 * d6;
                if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                    let w = d2 / (d2 - d6);
                    let cp = a + ac * w;
                    return (p.distance_squared(cp), cp);
                }
                let va = d3 * d6 - d5 * d4;
                if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                    let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    let cp = b + (c - b) * w;
                    return (p.distance_squared(cp), cp);
                }
                let denom = 1.0 / (va + vb + vc);
                let v = vb * denom;
                let w = vc * denom;
                let cp = a + ab * v + ac * w;
                (p.distance_squared(cp), cp)
            };

            let mut min_sq = f32::MAX;
            for tri in tris {
                let a = Vec3::from_array(verts[tri[0] as usize]);
                let b = Vec3::from_array(verts[tri[1] as usize]);
                let c = Vec3::from_array(verts[tri[2] as usize]);
                let (d_sq, _) = closest_pt_tri(q, a, b, c);
                if d_sq < min_sq {
                    min_sq = d_sq;
                }
            }
            let unsigned_d = min_sq.sqrt();
            // Stepped block interior check: base [-12,12]^2 x [0,8.5], boss [-5,5]^2 x [8.5,15.5]
            let in_base = q.x.abs() <= 12.0 && q.y.abs() <= 12.0 && q.z >= 0.0 && q.z <= 8.50;
            let in_boss = q.x.abs() <= 5.0 && q.y.abs() <= 5.0 && q.z >= 8.50 && q.z <= 15.50;
            if in_base || in_boss {
                -unsigned_d
            } else {
                unsigned_d
            }
        };

        let local_verts: [[f32; 3]; 16] = [
            // Base bottom (z = 0.0)
            [-12.0, -12.0, 0.0],
            [12.0, -12.0, 0.0],
            [12.0, 12.0, 0.0],
            [-12.0, 12.0, 0.0],
            // Base deck outer ring (z = 8.50)
            [-12.0, -12.0, 8.50],
            [12.0, -12.0, 8.50],
            [12.0, 12.0, 8.50],
            [-12.0, 12.0, 8.50],
            // Boss base inner ring (z = 8.50)
            [-5.0, -5.0, 8.50],
            [5.0, -5.0, 8.50],
            [5.0, 5.0, 8.50],
            [-5.0, 5.0, 8.50],
            // Boss top (z = 15.50)
            [-5.0, -5.0, 15.50],
            [5.0, -5.0, 15.50],
            [5.0, 5.0, 15.50],
            [-5.0, 5.0, 15.50],
        ];
        let local_tris: &[[u32; 3]] = &[
            // Bottom
            [0, 2, 1],
            [0, 3, 2],
            // Base outer walls
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            // Flat deck annulus at z = 8.50
            [4, 5, 9],
            [4, 9, 8],
            [5, 6, 10],
            [5, 10, 9],
            [6, 7, 11],
            [6, 11, 10],
            [7, 4, 8],
            [7, 8, 11],
            // Boss vertical walls
            [8, 9, 13],
            [8, 13, 12],
            [9, 10, 14],
            [9, 14, 13],
            [10, 11, 15],
            [10, 15, 14],
            [11, 8, 12],
            [11, 12, 15],
            // Boss top at z = 15.50
            [12, 13, 14],
            [12, 14, 15],
        ];

        let run_case = |tilt_deg: f32, order_mode: crate::config::OrderFieldMode| {
            let th = tilt_deg.to_radians();
            let (st, ct) = th.sin_cos();
            let u_x = Vec3::new(ct, 0.0, -st);
            let u_y = Vec3::Y;
            let u_z = Vec3::new(st, 0.0, ct);
            let origin = Vec3::new(0.0, 0.0, 4.0);

            let mut positions = Vec::with_capacity(local_verts.len() * 3);
            for lv in &local_verts {
                let p = origin + u_x * lv[0] + u_y * lv[1] + u_z * lv[2];
                positions.extend_from_slice(&[p.x, p.y, p.z]);
            }
            let mut indices = Vec::with_capacity(local_tris.len() * 3);
            for t in local_tris {
                indices.extend_from_slice(t);
            }

            let part = ffi::register_triangle_mesh(
                &format!("plateau_stepped_block_{:.0}", tilt_deg),
                &positions,
                &indices,
            );
            assert!(part.object_id > 0);

            let mut cfg = SlicerConfig::default();
            cfg.eikonal.order_field_mode = order_mode;
            cfg.eikonal.voxel_size_mm = 0.25;
            cfg.eikonal.enable_multi_resolution_grid = false;
            cfg.machine.nozzle_axis_bed = [u_z.x, u_z.y, u_z.z];
            cfg.machine.gravity_direction_bed = [-u_z.x, -u_z.y, -u_z.z];
            cfg.slicing.layer_height.target_mm = 0.20;
            cfg.slicing.layer_height.min_mm = 0.08;
            cfg.slicing.layer_height.max_mm = 0.36;
            cfg.slicing.layer_height.first_layer_mm = 0.20;
            cfg.slicing.layer_height.top_layer_mm = 0.20;
            cfg.slicing.inner_wall_count = 1;
            cfg.slicing.shell_thickness_mm = 2.0;
            cfg.slicing.infill_density_percent = 20.0;
            cfg.slicing.enable_support_generation = false;
            cfg.slicing.enable_auto_conformal_skin = false;
            cfg.slicing.enable_top_surface_lofting = true;
            crate::ui::sync_config_to_backend(&cfg);
            assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));

            let sel = ffi::select_slicing_isosurfaces(part.object_id, 0.0);
            assert!(sel.valid && !sel.layers.is_empty());
            let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp.valid && sel.layers.len() == tp.layers.len());

            let effective_origin = origin - Vec3::new(part.origin_x, part.origin_y, part.origin_z);
            let to_local = |x: f32, y: f32, z: f32| -> Vec3 {
                let rel = Vec3::new(x, y, z) - effective_origin;
                Vec3::new(rel.dot(u_x), rel.dot(u_y), rel.dot(u_z))
            };

            // 1. Check all layer thicknesses obey [min_mm, max_mm]
            for l in &sel.layers {
                if !l.is_first_layer {
                    assert!(
                        l.mean_thickness_mm >= 0.08 - 0.01 && l.mean_thickness_mm <= 0.36 + 0.01,
                        "Layer {} (tilt={tilt_deg}°) thickness {} out of [0.08, 0.36]",
                        l.layer_index,
                        l.mean_thickness_mm
                    );
                }
            }

            // 2. Verify a plateau-pinned intermediate layer lands at lz ≈ 8.48 (within 0.04 mm of 8.50)
            //    with is_top_layer = false, AND a flat leaf-top layer lands at lz ≈ 15.48.
            let mut deck_layer_idx: Option<usize> = None;
            let mut best_deck_err = f32::MAX;
            let mut boss_top_layer_idx: Option<usize> = None;
            let mut best_boss_err = f32::MAX;

            for (li, layer) in tp.layers.iter().enumerate() {
                let info = &sel.layers[li];
                let mut deck_zs = Vec::new();
                let mut boss_zs = Vec::new();
                for path in &layer.paths {
                    for pt in &path.points {
                        let lp = to_local(pt.x, pt.y, pt.z);
                        if lp.x.abs().max(lp.y.abs()) > 5.8 && lp.x.abs().max(lp.y.abs()) < 11.4 {
                            deck_zs.push(lp.z);
                        } else if lp.x.abs() < 4.4 && lp.y.abs() < 4.4 && lp.z > 14.5 {
                            boss_zs.push(lp.z);
                        }
                    }
                }
                if !deck_zs.is_empty() && !info.is_top_layer {
                    deck_zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med_z = deck_zs[deck_zs.len() / 2];
                    let err = (med_z - 8.48).abs();
                    if med_z <= 8.51 && err < best_deck_err {
                        best_deck_err = err;
                        deck_layer_idx = Some(li);
                    }
                }
                if !boss_zs.is_empty() && info.is_top_layer {
                    boss_zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med_z = boss_zs[boss_zs.len() / 2];
                    let err = (med_z - 15.48).abs();
                    if med_z <= 15.51 && err < best_boss_err {
                        best_boss_err = err;
                        boss_top_layer_idx = Some(li);
                    }
                }
            }

            let d_idx = deck_layer_idx.expect("Expected plateau-pinned deck layer near z=8.50");
            assert!(
                best_deck_err <= 0.045,
                "Expected plateau-pinned deck layer within 0.045 mm of 8.48 (tilt={tilt_deg}°), got err={best_deck_err:.4}"
            );
            let b_idx = boss_top_layer_idx.expect("Expected flat leaf-top layer near z=15.50");
            assert!(
                best_boss_err <= 0.045,
                "Expected flat leaf-top layer within 0.045 mm of 15.48 (tilt={tilt_deg}°), got err={best_boss_err:.4}"
            );

            // 3. Verify the plateau-pinned deck layer has FULL SolidInfill (not erased by maxSolidInfillSdf!)
            //    and OuterWall / InnerWall / SolidInfill all sit flush near lz ≈ 8.48..8.50 with no protrusion.
            let mut deck_ow_zs = Vec::new();
            let mut deck_iw_zs = Vec::new();
            let mut deck_si_zs = Vec::new();
            let mut deck_si_len = 0.0f32;
            let mut deck_max_sdf = f32::NEG_INFINITY;
            for path in &tp.layers[d_idx].paths {
                for pt in &path.points {
                    let lp = to_local(pt.x, pt.y, pt.z);
                    let sdf = tri_mesh_signed_dist(&local_verts, local_tris, lp);
                    deck_max_sdf = deck_max_sdf.max(sdf);
                    let in_deck_annulus =
                        lp.x.abs().max(lp.y.abs()) > 5.8 && lp.x.abs().max(lp.y.abs()) < 11.4;
                    let on_outer_rim =
                        lp.x.abs().max(lp.y.abs()) > 11.4 && lp.z > 8.1;
                    if path.feature_kind == 0 && (in_deck_annulus || on_outer_rim) {
                        deck_ow_zs.push(lp.z);
                    } else if path.feature_kind == 1 && (in_deck_annulus || on_outer_rim) {
                        deck_iw_zs.push(lp.z);
                    } else if path.feature_kind == 2 && in_deck_annulus {
                        deck_si_zs.push(lp.z);
                    }
                }
                if path.feature_kind == 2 {
                    let any_annulus = path.points.iter().any(|pt| {
                        let lp = to_local(pt.x, pt.y, pt.z);
                        lp.x.abs().max(lp.y.abs()) > 5.8 && lp.x.abs().max(lp.y.abs()) < 11.4
                    });
                    if any_annulus {
                        deck_si_len += path.total_length_mm;
                    }
                }
            }

            assert!(
                !deck_si_zs.is_empty() && deck_si_len >= 60.0,
                "Flat deck layer must retain full SolidInfill (tilt={tilt_deg}°): n_pts={}, len={deck_si_len:.1} mm",
                deck_si_zs.len()
            );
            assert!(
                !deck_ow_zs.is_empty() && !deck_iw_zs.is_empty(),
                "Expected OuterWall and InnerWall on plateau-pinned deck layer (tilt={tilt_deg}°)"
            );
            let median = |v: &mut Vec<f32>| -> f32 {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let ow_med = median(&mut deck_ow_zs);
            let iw_med = median(&mut deck_iw_zs);
            let si_med = median(&mut deck_si_zs);
            assert!(
                (ow_med - 8.48).abs() <= 0.045
                    && (iw_med - 8.48).abs() <= 0.045
                    && (si_med - 8.48).abs() <= 0.045,
                "Deck OuterWall/InnerWall/SolidInfill should all sit flush near z=8.48 (tilt={tilt_deg}°): ow={ow_med:.3}, iw={iw_med:.3}, si={si_med:.3}"
            );
            assert!(
                deck_max_sdf <= 0.025,
                "Deck layer must not protrude outside CAD (tilt={tilt_deg}°): max_sdf={deck_max_sdf:.4}"
            );

            // 4. Verify boss top layer also has SolidInfill and flush walls without protrusion
            let mut boss_max_sdf = f32::NEG_INFINITY;
            let mut boss_si_pts = 0usize;
            for path in &tp.layers[b_idx].paths {
                for pt in &path.points {
                    let lp = to_local(pt.x, pt.y, pt.z);
                    boss_max_sdf = boss_max_sdf.max(tri_mesh_signed_dist(&local_verts, local_tris, lp));
                    if path.feature_kind == 2 && lp.x.abs() < 4.5 && lp.y.abs() < 4.5 {
                        boss_si_pts += 1;
                    }
                }
            }
            assert!(
                boss_si_pts > 0,
                "Boss top leaf layer must retain SolidInfill (tilt={tilt_deg}°)"
            );
            assert!(
                boss_max_sdf <= 0.025,
                "Boss top layer must not protrude outside CAD (tilt={tilt_deg}°): max_sdf={boss_max_sdf:.4}"
            );

            let _ = ffi::remove_cad_object(part.object_id);
        };

        run_case(0.0, crate::config::OrderFieldMode::ConstrainedNonPlanar);
        run_case(20.0, crate::config::OrderFieldMode::HeightOnlyPlanar);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_outer_wall_rim_lofting_and_tool_clearance_slope_limit() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Part A: Stepped base with a narrow 6-deg sloped top rim (`> 5°`, so NOT plateau-pinned)
        // where exposed top-facing OuterWall beads sit ~0.10 mm below the top face and must loft
        // upward with InnerWall while obeying strict side-wall containment (`max_sdf <= 0.025`).
        {
            let slope_rad = 6.0f32.to_radians();
            let z_hi = 4.6 + 24.0 * slope_rad.tan();
            let local_verts: [[f32; 3]; 16] = [
                // Base bottom (z = 0.0)
                [-14.0, -5.0, 0.0],
                [14.0, -5.0, 0.0],
                [14.0, 5.0, 0.0],
                [-14.0, 5.0, 0.0],
                // Base top outer ring (z = 4.0)
                [-14.0, -5.0, 4.0],
                [14.0, -5.0, 4.0],
                [14.0, 5.0, 4.0],
                [-14.0, 5.0, 4.0],
                // Rail base inner ring (z = 4.0)
                [-12.0, -1.0, 4.0],
                [12.0, -1.0, 4.0],
                [12.0, 1.0, 4.0],
                [-12.0, 1.0, 4.0],
                // Sloped rail top (6 deg along +X)
                [-12.0, -1.0, 4.6],
                [12.0, -1.0, z_hi],
                [12.0, 1.0, z_hi],
                [-12.0, 1.0, 4.6],
            ];
            let local_tris: &[[u32; 3]] = &[
                [0, 2, 1],
                [0, 3, 2],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [4, 5, 9],
                [4, 9, 8],
                [5, 6, 10],
                [5, 10, 9],
                [6, 7, 11],
                [6, 11, 10],
                [7, 4, 8],
                [7, 8, 11],
                [8, 9, 13],
                [8, 13, 12],
                [9, 10, 14],
                [9, 14, 13],
                [10, 11, 15],
                [10, 15, 14],
                [11, 8, 12],
                [11, 12, 15],
                [12, 13, 14],
                [12, 14, 15],
            ];

            let mut positions = Vec::with_capacity(local_verts.len() * 3);
            for lv in &local_verts {
                positions.extend_from_slice(lv);
            }
            let mut indices = Vec::with_capacity(local_tris.len() * 3);
            for t in local_tris {
                indices.extend_from_slice(t);
            }
            let part = ffi::register_triangle_mesh("outer_rim_sloped_rail", &positions, &indices);
            assert!(part.object_id > 0);
            let origin_shift = Vec3::new(part.origin_x, part.origin_y, part.origin_z);

            let mut cfg = SlicerConfig::default();
            cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
            cfg.eikonal.voxel_size_mm = 0.25;
            cfg.eikonal.enable_multi_resolution_grid = false;
            cfg.slicing.layer_height.target_mm = 0.20;
            cfg.slicing.layer_height.min_mm = 0.08;
            cfg.slicing.layer_height.max_mm = 0.36;
            cfg.slicing.layer_height.first_layer_mm = 0.20;
            cfg.slicing.inner_wall_count = 1;
            cfg.slicing.shell_thickness_mm = 2.0;
            cfg.slicing.enable_support_generation = false;
            cfg.slicing.enable_auto_conformal_skin = false;
            cfg.slicing.enable_top_surface_lofting = false;
            crate::ui::sync_config_to_backend(&cfg);
            assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));
            assert!(ffi::select_slicing_isosurfaces(part.object_id, 0.0).valid);
            let tp_off = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp_off.valid);

            cfg.slicing.enable_top_surface_lofting = true;
            crate::ui::sync_config_to_backend(&cfg);
            let tp_on = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp_on.valid);

            let z_top_at = |lx: f32| -> f32 { 4.6 + (lx + 12.0) * slope_rad.tan() };
            let cos_s = slope_rad.cos();
            let collect_ow_stats = |tp: &ffi::ObjectToolpathsResult| {
                let mut exp_sdfs = Vec::new();
                let mut exp_heights = Vec::new();
                let mut max_side_x = f32::NEG_INFINITY;
                let mut max_side_y = f32::NEG_INFINITY;
                for layer in &tp.layers {
                    let mut ys_z = Vec::new();
                    for path in &layer.paths {
                        if path.feature_kind == 0 {
                            for pt in &path.points {
                                let lp = Vec3::new(pt.x, pt.y, pt.z) + origin_shift;
                                if lp.z > 4.2 {
                                    ys_z.push(lp.z);
                                }
                            }
                        }
                    }
                    if ys_z.is_empty() {
                        continue;
                    }
                    ys_z.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let base_lz = ys_z[0];
                    for path in &layer.paths {
                        if path.feature_kind != 0 {
                            continue;
                        }
                        for pt in &path.points {
                            let lp = Vec3::new(pt.x, pt.y, pt.z) + origin_shift;
                            if lp.z < 4.5 || lp.x < -10.5 || lp.x > 10.5 {
                                continue;
                            }
                            if lp.y.abs() <= 1.15 {
                                // Check lateral side-wall bounds of the rail ([-12, 12] x [-1, 1])
                                max_side_y = max_side_y.max(lp.y.abs() - 1.0);
                                max_side_x = max_side_x.max(lp.x.abs() - 12.0);
                            }
                            let dz_base = z_top_at(lp.x) - base_lz;
                            if dz_base > 0.06 && dz_base < 0.145 && lp.y.abs() <= 1.0 {
                                exp_sdfs.push((lp.z - z_top_at(lp.x)) * cos_s);
                                exp_heights.push(pt.height_mm);
                            }
                        }
                    }
                }
                (exp_sdfs, exp_heights, max_side_x, max_side_y)
            };

            let (s_off, h_off, _, _) = collect_ow_stats(&tp_off);
            let (s_on, h_on, max_sx, max_sy) = collect_ow_stats(&tp_on);
            assert!(
                !s_off.is_empty() && !s_on.is_empty(),
                "Expected exposed top-facing OuterWall rim samples on 6-deg rail"
            );
            let mean = |v: &[f32]| v.iter().sum::<f32>() / (v.len() as f32);
            let m_s_off = mean(&s_off);
            let m_s_on = mean(&s_on);
            let m_h_off = mean(&h_off);
            let m_h_on = mean(&h_on);
            assert!(
                m_s_on - m_s_off >= 0.03,
                "Exposed OuterWall rim should loft closer to top CAD surface: off={m_s_off:.4}, on={m_s_on:.4}"
            );
            assert!(
                m_h_on > m_h_off + 0.02,
                "Lofted OuterWall rim should scale height_mm: off={m_h_off:.4}, on={m_h_on:.4}"
            );
            assert!(
                max_sx.is_finite() && max_sy.is_finite() && max_sx <= -0.05 && max_sy <= -0.05,
                "Lofted OuterWall must stay inside rail side walls: max_sx={max_sx:.4}, max_sy={max_sy:.4}"
            );
            let _ = ffi::remove_cad_object(part.object_id);
        }

        // Part B: Verify the loft ramp climb slope obeys the active tool's physical clearance slope.
        // Configure a 3-axis tool with ParametricCone half-angle = 70 deg => max climb = 20 deg (tan ≈ 0.364).
        {
            let slope_rad = 10.0f32.to_radians();
            let z_slope_max = 4.0 + 24.0 * slope_rad.tan();
            let local_verts: [f32; 24] = [
                -12.0, -8.0, 0.0,
                12.0, -8.0, 0.0,
                12.0, 8.0, 0.0,
                -12.0, 8.0, 0.0,
                -12.0, -8.0, 4.0,
                12.0, -8.0, z_slope_max,
                12.0, 8.0, z_slope_max,
                -12.0, 8.0, 4.0,
            ];
            let tris: [u32; 36] = [
                0, 2, 1, 0, 3, 2,
                4, 5, 6, 4, 6, 7,
                0, 1, 5, 0, 5, 4,
                1, 2, 6, 1, 6, 5,
                2, 3, 7, 2, 7, 6,
                3, 0, 4, 3, 4, 7,
            ];
            let part = ffi::register_triangle_mesh("clearance_limited_loft_wedge", &local_verts, &tris);
            assert!(part.object_id > 0);

            let mut cfg = SlicerConfig::default();
            cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
            cfg.eikonal.voxel_size_mm = 0.25;
            cfg.eikonal.enable_multi_resolution_grid = false;
            cfg.slicing.layer_height.target_mm = 0.20;
            cfg.slicing.layer_height.min_mm = 0.08;
            cfg.slicing.layer_height.max_mm = 0.36;
            cfg.slicing.layer_height.first_layer_mm = 0.20;
            cfg.slicing.inner_wall_count = 1;
            cfg.slicing.shell_thickness_mm = 2.0;
            cfg.slicing.enable_support_generation = false;
            cfg.slicing.enable_auto_conformal_skin = false;
            cfg.slicing.enable_top_surface_lofting = true;

            let mut tool = cfg.tools[0].clone();
            tool.has_rotational_axes = false;
            tool.clearance_mode = crate::config::ClearanceProfileMode::ParametricCone;
            tool.cone_half_angle_deg = 70.0; // max clearance slope = 20 deg (tan = 0.36397)
            cfg.machine.kinematics_mode = crate::config::FiveAxisKinematicsMode::ThreeAxis;
            cfg.tools = vec![tool];
            cfg.active_tool_index = 0;

            crate::ui::sync_config_to_backend(&cfg);
            assert!(ffi::recompute_object_fields(part.object_id, 0.25, 4.0));
            assert!(ffi::select_slicing_isosurfaces(part.object_id, 0.0).valid);
            let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
            assert!(tp.valid);

            let max_allowed_slope = 20.0f32.to_radians().tan() + 0.025;
            let mut max_observed_slope = 0.0f32;
            let mut max_height_mm = 0.0f32;
            for layer in &tp.layers {
                for path in &layer.paths {
                    if path.feature_kind > 3 {
                        continue;
                    }
                    for pt in &path.points {
                        max_height_mm = max_height_mm.max(pt.height_mm);
                    }
                    let n = path.points.len();
                    let edges = if path.is_closed { n } else { n.saturating_sub(1) };
                    for e in 0..edges {
                        let a = &path.points[e];
                        let b = &path.points[(e + 1) % n];
                        if a.height_mm <= 0.205 && b.height_mm <= 0.205 {
                            continue;
                        }
                        let dxy = (b.x - a.x).hypot(b.y - a.y);
                        let dz = (b.z - a.z).abs();
                        if dxy > 0.02 {
                            let slope = dz / dxy;
                            max_observed_slope = max_observed_slope.max(slope);
                            assert!(
                                slope <= max_allowed_slope,
                                "Loft ramp edge slope {slope:.4} exceeded tool clearance limit {max_allowed_slope:.4} (dz={dz:.4}, dxy={dxy:.4})"
                            );
                        }
                    }
                }
            }
            assert!(
                max_height_mm > 0.24,
                "Expected active lofting under clearance limit test, got max_height_mm={max_height_mm:.4}"
            );
            let _ = max_observed_slope;
            let _ = ffi::remove_cad_object(part.object_id);
        }

        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_window_lintel_and_ceiling_bridge_leveling_and_direction() {
        let _guard = TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Construct a watertight cabin mesh with:
        // - Front window opening: X in [3.6, 6.0], Y in [-5.0, 5.0], Z in [0.0, 6.0]
        // - Front window lintel:  X in [3.6, 6.0], Y in [-5.0, 5.0], Z in [6.0, 9.0] (2.4 mm thick)
        // - Flat cabin ceiling:   X in [-6.0, 6.0], Y in [-7.0, 7.0], Z in [9.0, 11.0]
        // - One-sided cantilever: X in [-6.0, 6.0], Y in [7.0, 11.0], Z in [9.0, 11.0]
        let x_coords = [-6.0f32, -3.6, 3.6, 6.0];
        let y_coords = [-7.0f32, -5.0, 5.0, 7.0, 11.0];
        let z_coords = [0.0f32, 6.0, 9.0, 11.0];
        let is_occupied = |ix: i32, iy: i32, iz: i32| -> bool {
            if ix < 0 || ix >= 3 || iy < 0 || iy >= 4 || iz < 0 || iz >= 3 {
                return false;
            }
            match iz {
                0 => (iy == 0 || iy == 2) || (ix == 0 && iy == 1),
                1 => (iy == 0 || iy == 2) || ((ix == 0 || ix == 2) && iy == 1),
                2 => true,
                _ => false,
            }
        };

        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut add_quad = |p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]| {
            let base = (positions.len() / 3) as u32;
            for p in [p0, p1, p2, p3] {
                positions.extend_from_slice(&p);
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };

        for ix in 0..3 {
            for iy in 0..4 {
                for iz in 0..3 {
                    if !is_occupied(ix, iy, iz) {
                        continue;
                    }
                    let x0 = x_coords[ix as usize];
                    let x1 = x_coords[(ix + 1) as usize];
                    let y0 = y_coords[iy as usize];
                    let y1 = y_coords[(iy + 1) as usize];
                    let z0 = z_coords[iz as usize];
                    let z1 = z_coords[(iz + 1) as usize];

                    if !is_occupied(ix - 1, iy, iz) {
                        add_quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]);
                    }
                    if !is_occupied(ix + 1, iy, iz) {
                        add_quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]);
                    }
                    if !is_occupied(ix, iy - 1, iz) {
                        add_quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]);
                    }
                    if !is_occupied(ix, iy + 1, iz) {
                        add_quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]);
                    }
                    if !is_occupied(ix, iy, iz - 1) {
                        add_quad([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]);
                    }
                    if !is_occupied(ix, iy, iz + 1) {
                        add_quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]);
                    }
                }
            }
        }

        let part = ffi::register_triangle_mesh("cabin_window_lintel_and_ceiling", &positions, &indices);
        assert!(part.object_id > 0);

        let mut cfg = SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::ConstrainedNonPlanar;
        cfg.eikonal.enable_overhang_orientation_bias = true;
        cfg.eikonal.auto_voxel_size = false;
        cfg.eikonal.voxel_size_mm = 0.4;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.max_bridge_length_mm = 15.0;
        cfg.slicing.overhang_angle_threshold_deg = 45.0;
        cfg.slicing.enable_support_generation = false;
        cfg.machine.kinematics_mode = crate::config::FiveAxisKinematicsMode::ThreeAxis;
        let mut tool = cfg.tools[0].clone();
        tool.has_rotational_axes = false;
        cfg.tools = vec![tool];
        cfg.active_tool_index = 0;

        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);

        assert!(ffi::recompute_object_fields(part.object_id, 0.4, 5.0));
        assert!(ffi::select_slicing_isosurfaces(part.object_id, 0.0).valid);
        let tp = ffi::generate_object_shell_toolpaths(part.object_id, 0.0);
        assert!(tp.valid);

        #[derive(Default)]
        struct LayerBridgeStats {
            bridge_len: f32,
            weighted_abs_dy: f32,
            max_path_len: f32,
            min_z: f32,
            max_z: f32,
        }

        let mut lintel_by_layer: std::collections::BTreeMap<usize, LayerBridgeStats> =
            std::collections::BTreeMap::new();
        let mut ceiling_by_layer: std::collections::BTreeMap<usize, LayerBridgeStats> =
            std::collections::BTreeMap::new();
        let mut lintel_wave_len = 0.0f32;
        let mut cantilever_wave_count = 0usize;
        let mut cantilever_wave_len = 0.0f32;

        for (l_idx, layer) in tp.layers.iter().enumerate() {
            for path in &layer.paths {
                if path.points.len() < 2 {
                    continue;
                }
                let n = path.points.len();
                let mut path_lintel_bridge_len = 0.0f32;
                let mut path_ceiling_bridge_len = 0.0f32;

                for k in 0..(n - 1) {
                    let a = &path.points[k];
                    let b = &path.points[k + 1];
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let dz = b.z - a.z;
                    let seg_len = (dx * dx + dy * dy + dz * dz).sqrt();
                    if seg_len <= 1.0e-5 {
                        continue;
                    }
                    let mx = 0.5 * (a.x + b.x);
                    let my = 0.5 * (a.y + b.y);
                    let mz = 0.5 * (a.z + b.z);

                    if path.feature_kind == 5 {
                        if (3.6..=6.0).contains(&mx)
                            && (-4.5..=4.5).contains(&my)
                            && (5.5..=7.2).contains(&mz)
                        {
                            let e = lintel_by_layer.entry(l_idx).or_insert(LayerBridgeStats {
                                bridge_len: 0.0,
                                weighted_abs_dy: 0.0,
                                max_path_len: 0.0,
                                min_z: f32::MAX,
                                max_z: f32::MIN,
                            });
                            e.bridge_len += seg_len;
                            e.weighted_abs_dy += dy.abs();
                            e.min_z = e.min_z.min(a.z.min(b.z));
                            e.max_z = e.max_z.max(a.z.max(b.z));
                            path_lintel_bridge_len += seg_len;
                        }
                        if (-3.4..=3.4).contains(&mx)
                            && (-4.8..=4.8).contains(&my)
                            && (8.5..=9.8).contains(&mz)
                        {
                            let e = ceiling_by_layer.entry(l_idx).or_insert(LayerBridgeStats {
                                bridge_len: 0.0,
                                weighted_abs_dy: 0.0,
                                max_path_len: 0.0,
                                min_z: f32::MAX,
                                max_z: f32::MIN,
                            });
                            e.bridge_len += seg_len;
                            e.weighted_abs_dy += dy.abs();
                            e.min_z = e.min_z.min(a.z.min(b.z));
                            e.max_z = e.max_z.max(a.z.max(b.z));
                            path_ceiling_bridge_len += seg_len;
                        }
                    } else if path.feature_kind == 6 {
                        if (3.6..=6.0).contains(&mx)
                            && (-4.0..=4.0).contains(&my)
                            && (5.5..=7.2).contains(&mz)
                        {
                            lintel_wave_len += seg_len;
                        }
                        if my > 7.5 && (8.5..=10.5).contains(&mz) {
                            cantilever_wave_count += 1;
                            cantilever_wave_len += seg_len;
                        }
                    }
                }

                if path_lintel_bridge_len > 0.0 {
                    if let Some(e) = lintel_by_layer.get_mut(&l_idx) {
                        e.max_path_len = e.max_path_len.max(path_lintel_bridge_len);
                    }
                }
                if path_ceiling_bridge_len > 0.0 {
                    if let Some(e) = ceiling_by_layer.get_mut(&l_idx) {
                        e.max_path_len = e.max_path_len.max(path_ceiling_bridge_len);
                    }
                }
            }
        }

        let lintel_total_bridge_len: f32 =
            lintel_by_layer.values().map(|s| s.bridge_len).sum();
        let lintel_total_abs_dy: f32 =
            lintel_by_layer.values().map(|s| s.weighted_abs_dy).sum();
        let lintel_mean_abs_dy_ratio = lintel_total_abs_dy / lintel_total_bridge_len.max(1.0e-5);
        let max_lintel_bridge_len = lintel_by_layer
            .values()
            .map(|s| s.max_path_len)
            .fold(0.0f32, f32::max);
        let primary_lintel = lintel_by_layer
            .values()
            .max_by(|a, b| a.bridge_len.total_cmp(&b.bridge_len));

        assert!(
            lintel_total_bridge_len > 0.0,
            "Expected bridge extrusions under the front window lintel"
        );
        assert!(
            lintel_mean_abs_dy_ratio >= 0.85,
            "Expected window lintel bridges to span predominantly along Y (|dy|/len = {lintel_mean_abs_dy_ratio:.3})"
        );
        assert!(
            max_lintel_bridge_len >= 9.0,
            "Expected window lintel bridge to span >= 9.0 mm across opening, got {max_lintel_bridge_len:.3} mm"
        );
        assert!(
            lintel_by_layer.len() <= 2,
            "Expected <= 2 bridging layers on window lintel, got {}",
            lintel_by_layer.len()
        );
        if let Some(prim) = primary_lintel {
            let z_spread = prim.max_z - prim.min_z;
            assert!(
                z_spread <= 0.30,
                "Expected flat window lintel primary bridge layer (z spread <= 0.30 mm), got {z_spread:.3} mm"
            );
        }
        assert!(
            lintel_wave_len < 0.15 * lintel_total_bridge_len,
            "Expected minimal WaveOverhang inside window lintel opening (wave={lintel_wave_len:.3}, bridge={lintel_total_bridge_len:.3})"
        );

        let primary_ceiling = ceiling_by_layer
            .values()
            .max_by(|a, b| a.bridge_len.total_cmp(&b.bridge_len));
        assert!(
            primary_ceiling.is_some(),
            "Expected bridge extrusions on the cabin ceiling"
        );
        if let Some(prim) = primary_ceiling {
            let z_spread = prim.max_z - prim.min_z;
            assert!(
                z_spread <= 0.35,
                "Expected flat cabin ceiling primary bridge layer (z spread <= 0.35 mm), got {z_spread:.3} mm"
            );
        }

        assert!(
            cantilever_wave_count > 0 && cantilever_wave_len > 2.0,
            "Expected WaveOverhang extrusions on the one-sided cantilever (count={cantilever_wave_count}, len={cantilever_wave_len:.3})"
        );

        let _ = ffi::remove_cad_object(part.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    #[test]
    fn test_conformal_cap_adjacent_collar_no_gap_or_deep_sdf_trench() {
        let _guard = TEST_BACKEND_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        let mut stl = String::from("solid conformal_cap_collar\n");
        let mut add_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            stl.push_str(&format!(
                "  facet normal 0 0 0\n    outer loop\n      vertex {} {} {}\n      vertex {} {} {}\n      vertex {} {} {}\n    endloop\n  endfacet\n",
                a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]
            ));
        };
        let mut add_quad = |p00: [f32; 3], p10: [f32; 3], p11: [f32; 3], p01: [f32; 3]| {
            add_tri(p00, p10, p11);
            add_tri(p00, p11, p01);
        };
        let mut add_sloped_box =
            |xa: f32, xb: f32, ya: f32, yb: f32, za0: f32, zb0: f32, za1: f32, zb1: f32| {
                add_quad([xa, ya, za0], [xa, yb, za0], [xb, yb, zb0], [xb, ya, zb0]);
                add_quad([xa, ya, za1], [xb, ya, zb1], [xb, yb, zb1], [xa, yb, za1]);
                add_quad([xa, ya, za0], [xb, ya, zb0], [xb, ya, zb1], [xa, ya, za1]);
                add_quad([xb, yb, zb0], [xa, yb, za0], [xa, yb, za1], [xb, yb, zb1]);
                add_quad([xa, yb, za0], [xa, ya, za0], [xa, ya, za1], [xa, yb, za1]);
                add_quad([xb, ya, zb0], [xb, yb, zb0], [xb, yb, zb1], [xb, ya, zb1]);
            };

        let z_deck = |x: f32| -> f32 { 3.2 + 0.16 * (x + 2.0) };
        add_sloped_box(-2.0, 9.0, -5.5, 5.5, 0.0, 0.0, z_deck(-2.0), z_deck(9.0));
        add_sloped_box(-9.0, -2.0, -5.5, 5.5, 0.0, 0.0, 9.5, 9.5);
        add_sloped_box(
            -9.0,
            9.0,
            -7.5,
            -5.5,
            0.0,
            0.0,
            z_deck(-9.0) + 2.4,
            z_deck(9.0) + 2.4,
        );
        add_sloped_box(
            -9.0,
            9.0,
            5.5,
            7.5,
            0.0,
            0.0,
            z_deck(-9.0) + 2.4,
            z_deck(9.0) + 2.4,
        );
        stl.push_str("endsolid conformal_cap_collar\n");

        let tmp_path = std::env::temp_dir().join("manifold_test_conformal_cap_collar.stl");
        std::fs::write(&tmp_path, stl.as_bytes()).expect("Failed to write temporary STL");
        let obj = crate::io::stl::load_stl_file(&tmp_path)
            .expect("Expected synthetic conformal cap collar STL import to succeed");
        let _ = std::fs::remove_file(&tmp_path);

        let mut cfg = crate::config::SlicerConfig::default();
        cfg.eikonal.order_field_mode = crate::config::OrderFieldMode::HeightOnlyPlanar;
        cfg.eikonal.auto_voxel_size = false;
        cfg.eikonal.voxel_size_mm = 0.25;
        cfg.eikonal.enable_multi_resolution_grid = false;
        cfg.slicing.layer_height.target_mm = 0.20;
        cfg.slicing.layer_height.min_mm = 0.10;
        cfg.slicing.layer_height.max_mm = 0.32;
        cfg.slicing.layer_height.first_layer_mm = 0.20;
        cfg.slicing.layer_height.top_layer_mm = 0.20;
        cfg.slicing.enable_auto_conformal_skin = true;
        cfg.slicing.conformal_skin_layers = 3;
        cfg.slicing.conformal_skin_min_oblique_deg = 6.0;
        cfg.slicing.conformal_skin_max_slope_deg = 60.0;
        cfg.slicing.enable_top_surface_lofting = true;
        cfg.slicing.enable_support_generation = false;
        cfg.slicing.sanitize();
        crate::ui::sync_config_to_backend(&cfg);
        ffi::set_spatial_orientation_hints(&[]);
        ffi::set_nozzle_axis_bed([0.0, 0.0, 1.0]);
        ffi::set_gravity_direction_bed([0.0, 0.0, -1.0]);

        assert!(ffi::recompute_object_fields(
            obj.object_id,
            cfg.eikonal.voxel_size_mm,
            cfg.eikonal.narrow_band_width_voxels as f32
        ));

        let iso = ffi::select_slicing_isosurfaces(obj.object_id, 0.0);
        assert!(iso.valid && !iso.layers.is_empty());
        let tp = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
        assert!(tp.valid && iso.layers.len() == tp.layers.len());

        let top_h = cfg.slicing.layer_height.top_layer_mm;
        let mut bow_skin_layers: Vec<&ffi::SelectedLayerInfo> = iso
            .layers
            .iter()
            .filter(|l| {
                l.is_top_layer
                    && !l.is_first_layer
                    && (l.max_thickness_mm - l.min_thickness_mm).abs() <= 1.0e-4
                    && (l.mean_thickness_mm - top_h).abs() <= 1.0e-3
                    && l.vertices
                        .iter()
                        .any(|v| v.x > -1.5 && v.y.abs() < 4.5 && v.z < 5.1)
            })
            .collect();
        assert!(
            bow_skin_layers.len() >= 3,
            "Expected >= 3 conformal skin cap layers on bow deck, got {}",
            bow_skin_layers.len()
        );
        bow_skin_layers.sort_by_key(|l| (l.branch_layer_index, l.layer_index));

        // 1. Conformal skin layers remain collision-clear of aft cabin (x=-2) & gunwales (|y|=5.5)
        for layer in &bow_skin_layers {
            for v in &layer.vertices {
                if v.y.abs() < 5.2 && v.z < 5.1 {
                    assert!(
                        v.x - (-2.0) >= 1.65 && 5.5 - v.y.abs() >= 1.25,
                        "Conformal skin vertex ({:.2}, {:.2}, {:.2}) violated clearance",
                        v.x,
                        v.y,
                        v.z
                    );
                }
            }
        }

        let outermost_skin = bow_skin_layers.last().copied().unwrap();
        let outermost_skin_tp = tp
            .layers
            .iter()
            .find(|tl| tl.layer_index == outermost_skin.layer_index)
            .expect("Expected toolpath layer for outermost bow deck skin");
        let skin_layer_indices: std::collections::HashSet<u32> =
            bow_skin_layers.iter().map(|l| l.layer_index).collect();

        let cap_outer_wall_pts: Vec<[f32; 3]> = outermost_skin_tp
            .paths
            .iter()
            .filter(|p| p.feature_kind == 0)
            .flat_map(|p| p.points.iter().map(|pt| [pt.x, pt.y, pt.z]))
            .collect();
        assert!(
            !cap_outer_wall_pts.is_empty(),
            "Expected OuterWall points on outermost conformal skin cap"
        );
        let cap_all_pts: Vec<[f32; 3]> = outermost_skin_tp
            .paths
            .iter()
            .flat_map(|p| p.points.iter().map(|pt| [pt.x, pt.y, pt.z]))
            .collect();

        let mut core_pts: Vec<(u8, [f32; 3])> = Vec::new();
        for (li, layer) in tp.layers.iter().enumerate() {
            let info = &iso.layers[li];
            if info.is_first_layer || skin_layer_indices.contains(&info.layer_index) {
                continue;
            }
            for path in &layer.paths {
                if path.feature_kind > 3 {
                    continue;
                }
                for pt in &path.points {
                    if pt.x > -1.3
                        && pt.x < 8.5
                        && pt.y.abs() < 4.8
                        && pt.z >= 2.5
                        && pt.z <= z_deck(pt.x) + 0.5
                    {
                        core_pts.push((path.feature_kind, [pt.x, pt.y, pt.z]));
                    }
                }
            }
        }

        let cos_deck = 1.0f32 / (1.0f32 + 0.16 * 0.16).sqrt();
        let mut max_under_cap_sdf = f32::NEG_INFINITY;
        let mut under_cap_count = 0usize;
        let mut collar_sdfs: Vec<f32> = Vec::new();
        let mut min_d3_to_cap_ow = f32::MAX;

        for &(fk, p) in &core_pts {
            let mut min_dxy_cap = f32::MAX;
            for &cp in &cap_all_pts {
                let dxy = (p[0] - cp[0]).hypot(p[1] - cp[1]);
                if dxy < min_dxy_cap {
                    min_dxy_cap = dxy;
                }
            }
            let mut min_dxy_cap_ow = f32::MAX;
            let mut min_d3_cap_ow_pt = f32::MAX;
            for &cp in &cap_outer_wall_pts {
                let dxy = (p[0] - cp[0]).hypot(p[1] - cp[1]);
                if dxy < min_dxy_cap_ow {
                    min_dxy_cap_ow = dxy;
                }
                let d3 = ((p[0] - cp[0]).powi(2)
                    + (p[1] - cp[1]).powi(2)
                    + (p[2] - cp[2]).powi(2))
                .sqrt();
                if d3 < min_d3_cap_ow_pt {
                    min_d3_cap_ow_pt = d3;
                }
            }

            let sdf_deck = (p[2] - z_deck(p[0])) * cos_deck;

            // 2. Core directly under the interior of the cap remains eroded below the skin slab
            if min_dxy_cap <= 0.15 && min_dxy_cap_ow >= 0.55 {
                max_under_cap_sdf = max_under_cap_sdf.max(sdf_deck);
                under_cap_count += 1;
            }

            // 3. Exposed top core OuterWalls in the collar outside the cap footprint
            if fk == 0 && (0.25..=0.90).contains(&min_dxy_cap) {
                let has_higher_core_above = core_pts.iter().any(|&(_, q)| {
                    (q[0] - p[0]).hypot(q[1] - p[1]) < 0.22 && q[2] - p[2] > 0.10
                });
                if !has_higher_core_above {
                    collar_sdfs.push(sdf_deck);
                    min_d3_to_cap_ow = min_d3_to_cap_ow.min(min_d3_cap_ow_pt);
                }
            }
        }

        assert!(under_cap_count > 0, "Expected core points beneath cap interior");
        assert!(
            max_under_cap_sdf <= -0.35,
            "Expected core under cap interior to stay eroded <= -0.35 mm, got {max_under_cap_sdf:.4}"
        );
        assert!(
            !collar_sdfs.is_empty(),
            "Expected exposed top core OuterWall points in the adjacent cap collar [0.25, 0.90] mm"
        );

        collar_sdfs.sort_by(|a, b| a.total_cmp(b));
        let p10_idx = (collar_sdfs.len() * 10) / 100;
        let med_idx = collar_sdfs.len() / 2;
        let p10_sdf = collar_sdfs[p10_idx];
        let med_sdf = collar_sdfs[med_idx];
        let max_sdf = *collar_sdfs.last().unwrap();

        assert!(
            p10_sdf >= -0.22 && med_sdf >= -0.10 && max_sdf <= 0.02,
            "Expected collar OuterWall without deep SDF trench (p10 >= -0.22, med >= -0.10, max <= 0.02), got p10={p10_sdf:.3}, med={med_sdf:.3}, max={max_sdf:.3}"
        );
        assert!(
            min_d3_to_cap_ow <= 0.55,
            "Expected close seam between collar OuterWall and cap OuterWall (<= 0.55 mm), got {min_d3_to_cap_ow:.3} mm"
        );

        let _ = ffi::remove_cad_object(obj.object_id);
        crate::ui::sync_config_to_backend(&SlicerConfig::default());
    }

    fn dummy_valid_toolpaths() -> ObjectToolpathsResult {
        ObjectToolpathsResult {
            valid: true,
            total_path_count: 1,
            outer_wall_count: 1,
            layers: vec![ffi::LayerToolpaths {
                layer_index: 0,
                branch_id: 0,
                isovalue: 0.2,
                paths: Vec::new(),
            }],
            ..Default::default()
        }
    }

    fn dummy_planned_gcode() -> crate::gcode::PlannedGcodeResult {
        crate::gcode::PlannedGcodeResult {
            layers: vec![crate::gcode::PlannedLayerGcode {
                layer_index: 0,
                branch_id: 0,
                isovalue: 0.2,
                segments: Vec::new(),
            }],
            gcode_text: String::new(),
            total_time_s: 1.0,
            total_filament_mm: 10.0,
            total_filament_volume_mm3: 24.0,
            total_filament_grams: 0.03,
            bbox_min_bed: Vec3::ZERO,
            bbox_max_bed: Vec3::splat(10.0),
            exclude_object_name: "plate_obj".to_string(),
            exclude_object_center_bed: [5.0, 5.0],
            exclude_object_polygon_bed: Vec::new(),
            extruding_move_count: 1,
            travel_move_count: 0,
            retract_count: 0,
            toolchange_count: 0,
            min_speed_mms: 60.0,
            max_speed_mms: 60.0,
            min_accel_mms2: -1000.0,
            max_accel_mms2: 1000.0,
            min_flow_mm3s: 5.0,
            max_flow_mm3s: 5.0,
            min_width_mm: 0.4,
            max_width_mm: 0.4,
            min_height_mm: 0.2,
            max_height_mm: 0.2,
            min_flow_mult: 1.0,
            max_flow_mult: 1.0,
            min_pa_s: 0.02,
            max_pa_s: 0.02,
            min_pressure_mm3s: 5.0,
            max_pressure_mm3s: 5.0,
            min_tool_iso_angle_deg: 0.0,
            max_tool_iso_angle_deg: 0.0,
            min_sdf_mm: 0.0,
            max_sdf_mm: 0.0,
        }
    }

    #[test]
    fn test_multiplate_switch_to_unsliced_plate_does_not_loop_post_slice() {
        let mut vis = FieldVisualizationState::default();
        vis.show_toolpaths = true;

        // Plate 1 (object 1) has already completed "Slice & Plan All"
        let tp1 = dummy_valid_toolpaths();
        vis.plate_unmixed_toolpaths.insert(1, tp1.clone());
        vis.plate_mixed_toolpaths.insert(1, tp1.clone());
        vis.cached_planned_gcode = Some(dummy_planned_gcode());

        // Simulate switching to unsliced Plate 2 (object 2)
        vis.cached_unmixed_toolpaths = None;
        vis.cached_toolpaths = None;
        vis.dirty_planned_gcode = true;
        vis.dirty_toolpaths_mesh = true;

        // Frame 1 on unsliced Plate 2: must NOT spawn mix/plan/heavy_mesh; must request synchronous clear
        let d1 = compute_post_slice_work_decision(&vis, &[2], Some(2), true);
        assert!(!d1.active_plate_has_plannable_tp);
        assert!(!d1.want_mix);
        assert!(!d1.want_plan);
        assert!(!d1.want_heavy_mesh);
        assert!(d1.clear_stale_planned_gcode);

        // Apply the synchronous clear performed in sync_field_visualization_system
        vis.dirty_planned_gcode = false;
        vis.cached_planned_gcode = None;
        vis.planned_gcode_is_world_space = false;
        vis.dirty_toolpaths_mesh = true;

        // Frame 2 on unsliced Plate 2 (even with HighQualityPbr enabled): every flag must remain false
        vis.toolpath_quality_mode = ToolpathQualityMode::HighQualityPbr;
        let d2 = compute_post_slice_work_decision(&vis, &[2], Some(2), true);
        assert_eq!(d2, PostSliceWorkDecision::default());
        assert!(!vis.is_busy());

        // Switching back to Plate 1 (object 1) restores toolpaths and triggers planning once
        vis.cached_unmixed_toolpaths = vis.plate_unmixed_toolpaths.get(&1).cloned();
        vis.cached_toolpaths = vis.plate_mixed_toolpaths.get(&1).cloned();
        vis.dirty_planned_gcode = true;
        let d_back = compute_post_slice_work_decision(&vis, &[1], Some(1), true);
        assert!(d_back.active_plate_has_plannable_tp);
        assert!(!d_back.want_mix);
        assert!(d_back.want_plan);
        assert!(!d_back.clear_stale_planned_gcode);
    }

    #[test]
    fn test_selecting_unsliced_nonprintable_object_on_sliced_plate_does_not_loop_mix() {
        let mut vis = FieldVisualizationState::default();
        vis.show_toolpaths = true;

        let tp1 = dummy_valid_toolpaths();
        vis.plate_unmixed_toolpaths.insert(1, tp1.clone());
        vis.plate_mixed_toolpaths.insert(1, tp1);
        vis.cached_planned_gcode = Some(dummy_planned_gcode());
        vis.dirty_color_mixing = false;
        vis.dirty_planned_gcode = false;

        // Object 2 on Plate 1 is non-printable and unsliced (`cached_toolpaths == None`)
        vis.cached_unmixed_toolpaths = None;
        vis.cached_toolpaths = None;

        let decision = compute_post_slice_work_decision(&vis, &[1], Some(2), false);
        assert!(decision.active_plate_has_plannable_tp);
        assert!(!decision.want_mix);
        assert!(!decision.want_plan);
        assert!(!decision.clear_stale_planned_gcode);
    }

    #[test]
    fn test_mark_objects_dirty_and_lazy_color_remix_preserve_other_plates() {
        let mut vis = FieldVisualizationState::default();
        vis.show_toolpaths = true;

        // Plate 1 object (id=1) is already sliced & mixed
        let tp1 = dummy_valid_toolpaths();
        vis.plate_unmixed_toolpaths.insert(1, tp1.clone());
        vis.plate_mixed_toolpaths.insert(1, tp1);

        // Recomputing fields / slicing Plate 2 object (id=2) must preserve Plate 1 caches
        vis.mark_objects_dirty(&[2]);
        assert!(vis.plate_unmixed_toolpaths.contains_key(&1));
        assert!(vis.plate_mixed_toolpaths.contains_key(&1));

        // Changing global colour mixing while viewing unsliced Plate 2 clears stale mixed entries
        // cleanly without starting a post-slice task on Plate 2
        vis.dirty_color_mixing = true;
        vis.cached_unmixed_toolpaths = None;
        vis.cached_toolpaths = None;
        let d_unsliced = compute_post_slice_work_decision(&vis, &[2], Some(2), true);
        assert!(d_unsliced.clear_stale_mixed_toolpaths);
        assert!(!d_unsliced.want_mix);
        assert!(!d_unsliced.want_plan);

        // Apply the synchronous stale mix clear
        vis.dirty_color_mixing = false;
        vis.plate_mixed_toolpaths.clear();

        // Returning to Plate 1 detects missing mixed cache and lazily re-mixes & plans Plate 1
        vis.cached_unmixed_toolpaths = vis.plate_unmixed_toolpaths.get(&1).cloned();
        vis.cached_toolpaths = vis.plate_mixed_toolpaths.get(&1).cloned();
        vis.dirty_planned_gcode = true;
        let d_plate1 = compute_post_slice_work_decision(&vis, &[1], Some(1), true);
        assert!(d_plate1.active_plate_has_plannable_tp);
        assert!(d_plate1.want_mix);
        assert!(d_plate1.want_plan);
        assert!(!d_plate1.clear_stale_mixed_toolpaths);
    }
}


