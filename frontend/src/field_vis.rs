use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::tasks::IoTaskPool;

use crate::bridge::ffi::{
    self, ExtrusionPath, FieldSlice2D, FieldStats, LayerHeightSpec, LineWidthSpec,
    ObjectToolpathsResult, SelectedIsosurfacesResult, ShellPathSpec, SupportSpec,
};
use crate::config::{NozzleFrame, SlicerConfig};
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
}

impl ToolpathColorMode {
    pub const ALL: [Self; 9] = [
        Self::ByFeatureType,
        Self::ByLineWidth,
        Self::ByLayerHeight,
        Self::BySpeed,
        Self::ByAcceleration,
        Self::ByVolumetricFlow,
        Self::ByExtrusionMultiplier,
        Self::ByPressureAdvance,
        Self::ByMeltZonePressure,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ByFeatureType => "Feature Type (Outer / Inner / Solid / Sparse / Scarf / Wipe)",
            Self::ByLineWidth => "Line Width W (mm)",
            Self::ByLayerHeight => "Layer Height H (mm)",
            Self::BySpeed => "Planned Speed v (mm/s)",
            Self::ByAcceleration => "Acceleration a (mm/s²)",
            Self::ByVolumetricFlow => "Actual Volumetric Flow Q (mm³/s)",
            Self::ByExtrusionMultiplier => "Flow Multiplier M (Corner + Transient)",
            Self::ByPressureAdvance => "Dynamic Pressure Advance k_PA (s)",
            Self::ByMeltZonePressure => "Melt-Zone Pressure P_melt (MPa)",
        }
    }

    pub fn short_label(self) -> &'static str {
        match self {
            Self::ByFeatureType => "Feature / Line Type",
            Self::ByLineWidth => "Line Width W (mm)",
            Self::ByLayerHeight => "Layer Height H (mm)",
            Self::BySpeed => "Planned Speed v (mm/s)",
            Self::ByAcceleration => "Acceleration a (mm/s²)",
            Self::ByVolumetricFlow => "Volumetric Flow Q (mm³/s)",
            Self::ByExtrusionMultiplier => "Flow Multiplier M (×)",
            Self::ByPressureAdvance => "Dynamic PA k_PA (s)",
            Self::ByMeltZonePressure => "Melt-Zone Pressure P_melt",
        }
    }
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
    pub show_scarf_seams: bool,
    pub show_wipes: bool,
    pub show_travel_moves: bool,
    pub show_z_hops: bool,
    pub show_toolpath_centerlines: bool,
    pub toolpath_color_mode: ToolpathColorMode,
    pub cached_toolpaths: Option<ObjectToolpathsResult>,
    pub cached_planned_gcode: Option<crate::gcode::PlannedGcodeResult>,
    pub toolpaths_entity: Option<Entity>,
    pub toolpaths_mesh_handle: Option<Handle<Mesh>>,
    pub toolpaths_mat_handle: Option<Handle<StandardMaterial>>,
    pub dirty_toolpaths: bool,
    pub dirty_planned_gcode: bool,
    pub dirty_toolpaths_mesh: bool,

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

    /// Background task computing OpenVDB scalar fields for `(object_id, object_name)`.
    pub pending_field_task: Option<(u64, String, bevy::tasks::Task<bool>)>,
    /// Background task computing selected slicing isosurfaces and/or non-planar shell toolpaths.
    pub pending_slicing_task: Option<(
        u64,
        bevy::tasks::Task<(
            Option<SelectedIsosurfacesResult>,
            Option<ObjectToolpathsResult>,
        )>,
    )>,
    /// True while the user is actively holding down the pointer on a UI slider/drag widget.
    pub ui_pointer_down: bool,
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
            show_scarf_seams: true,
            show_wipes: true,
            show_travel_moves: false,
            show_z_hops: false,
            show_toolpath_centerlines: false,
            toolpath_color_mode: ToolpathColorMode::ByFeatureType,
            cached_toolpaths: None,
            cached_planned_gcode: None,
            toolpaths_entity: None,
            toolpaths_mesh_handle: None,
            toolpaths_mat_handle: None,
            dirty_toolpaths: true,
            dirty_planned_gcode: true,
            dirty_toolpaths_mesh: true,
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
            ui_pointer_down: false,
        }
    }
}

impl FieldVisualizationState {
    pub fn is_busy(&self) -> bool {
        self.pending_field_task.is_some() || self.pending_slicing_task.is_some()
    }

    pub fn mark_all_dirty(&mut self) {
        self.cached_stats = None;
        self.cached_selected_layers = None;
        self.cached_toolpaths = None;
        self.cached_planned_gcode = None;
        self.dirty_isosurface = true;
        self.dirty_selected_layers = true;
        self.dirty_selected_layers_mesh = true;
        self.dirty_toolpaths = true;
        self.dirty_planned_gcode = true;
        self.dirty_toolpaths_mesh = true;
        self.dirty_slice = true;
        self.dirty_surface_colors = true;
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
            let tp_res = if need_toolpaths {
                Some(ffi::generate_object_shell_toolpaths(object_id, adaptivity))
            } else {
                None
            };
            (layers_res, tp_res)
        });

        self.pending_slicing_task = Some((object_id, task));
    }

    pub fn apply_slicing_task_result(
        &mut self,
        layers_opt: Option<SelectedIsosurfacesResult>,
        toolpaths_opt: Option<ObjectToolpathsResult>,
    ) {
        if let Some(res) = layers_opt {
            if res.valid && !res.layers.is_empty() {
                let max_idx = res.layers.len() - 1;
                self.selected_layer_min_idx = 0;
                self.selected_layer_max_idx = max_idx;
                self.cached_selected_layers = Some(res);
            } else {
                self.selected_layer_min_idx = 0;
                self.selected_layer_max_idx = 0;
                self.cached_selected_layers = None;
            }
            self.dirty_selected_layers_mesh = true;
            self.dirty_slice = true;
        }

        if let Some(res) = toolpaths_opt {
            if res.valid && !res.layers.is_empty() {
                if self.selected_layer_max_idx == 0 && res.layers.len() > 1 {
                    self.selected_layer_max_idx = res.layers.len() - 1;
                }
                self.cached_toolpaths = Some(res);
            } else {
                self.cached_toolpaths = None;
                self.cached_planned_gcode = None;
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

/// Synchronizes available OpenVDB scalar fields, updates 3D isosurfaces (`volumeToMesh`)
/// with exact object-intersection clipping/highlighting ($\phi_{\text{sdf}} \le 0$) and 3D
/// intersection contour curves ($\tau = c \cap \phi_{\text{sdf}} = 0$), renders the selected
/// slicing isosurfaces stack, and updates 2D cross-section slices (`GridSampler`) in-place.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn sync_field_visualization_system(
    mut commands: Commands,
    plate: Res<BuildPlateState>,
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

    // 1. Detect object selection change
    if vis_state.last_object_id != selected_id {
        if let Some(prev_id) = vis_state.last_object_id {
            if let Some(prev_obj) = plate.objects.iter().find(|o| o.object_id == prev_id) {
                if let Some(mesh) = meshes.get_mut(&prev_obj.mesh_handle) {
                    mesh.insert_attribute(
                        Mesh::ATTRIBUTE_COLOR,
                        vec![[1.0f32, 1.0, 1.0, 1.0]; prev_obj.vertices.len()],
                    );
                }
            }
        }
        vis_state.last_object_id = selected_id;
        vis_state.available_fields.clear();
        vis_state.cached_stats = None;
        vis_state.cached_selected_layers = None;
        vis_state.cached_toolpaths = None;
        vis_state.cached_planned_gcode = None;
        vis_state.contour_segments_local.clear();
        vis_state.dirty_isosurface = true;
        vis_state.dirty_selected_layers = true;
        vis_state.dirty_selected_layers_mesh = true;
        vis_state.dirty_toolpaths = true;
        vis_state.dirty_planned_gcode = true;
        vis_state.dirty_toolpaths_mesh = true;
        vis_state.dirty_slice = true;
        vis_state.dirty_surface_colors = true;
        vis_state.last_surface_color_transform = None;
        vis_state.last_solved_field_rotation = None;

        if let Some(id) = selected_id {
            if !vis_state.is_busy() {
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
        if !vis_state.show_toolpaths || vis_state.cached_planned_gcode.is_none() {
            if let Some(ent) = vis_state.toolpaths_entity {
                if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
                    *vis = Visibility::Hidden;
                }
            }
            return;
        }

        // Render standalone planned G-code toolpaths (e.g., Calibration Patterns) at Transform::IDENTITY
        if vis_state.dirty_toolpaths_mesh {
            vis_state.dirty_toolpaths_mesh = false;
            let stride = vis_state.selected_layer_stride.max(1);
            let min_idx = vis_state.selected_layer_min_idx;
            let max_idx = vis_state.selected_layer_max_idx;
            let highlighted = vis_state.highlighted_selected_layer;
            let color_mode = vis_state.toolpath_color_mode;

            let mut positions: Vec<[f32; 3]> = Vec::new();
            let mut normals: Vec<[f32; 3]> = Vec::new();
            let mut colors: Vec<[f32; 4]> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            if let Some(ref planned) = vis_state.cached_planned_gcode {
                build_planned_toolpaths_bead_mesh_parallel(
                    planned,
                    &vis_state,
                    &config,
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
                    vis_state.toolpaths_entity,
                    &vis_state.toolpaths_mesh_handle,
                    &vis_state.toolpaths_mat_handle,
                ) {
                    meshes.insert(mesh_h, bevy_mesh);
                    if let Some(mat) = materials.get_mut(mat_h) {
                        mat.base_color = Color::WHITE;
                    }
                    if let Ok((mut tf, mut vis)) = toolpaths_query.get_mut(ent) {
                        *tf = Transform::IDENTITY;
                        *vis = Visibility::Inherited;
                    }
                } else {
                    let mesh_h = meshes.add(bevy_mesh);
                    let mat_h = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        metallic: 0.0,
                        perceptual_roughness: 0.56,
                        reflectance: 0.16,
                        alpha_mode: AlphaMode::Opaque,
                        depth_bias: 14.0,
                        double_sided: true,
                        cull_mode: None,
                        ..default()
                    });
                    let ent = commands
                        .spawn((
                            PbrBundle {
                                mesh: mesh_h.clone(),
                                material: mat_h.clone(),
                                transform: Transform::IDENTITY,
                                ..default()
                            },
                            ToolpathsMeshEntity,
                        ))
                        .id();
                    vis_state.toolpaths_entity = Some(ent);
                    vis_state.toolpaths_mesh_handle = Some(mesh_h);
                    vis_state.toolpaths_mat_handle = Some(mat_h);
                }
            }
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

    if vis_state.cached_stats.is_none() && !vis_state.is_busy() {
        vis_state.refresh_stats_for(obj_id);
    }

    // 1b. Update CAD object surface vertex colors (Default CAD / Overhang / Tool Angle from Normal)
    let tf_changed = vis_state.last_surface_color_transform != Some(obj_world_tf);
    let grav_changed = vis_state.last_surface_color_gravity_bed != config.machine.gravity_direction_bed;
    let nozzle_changed = vis_state.last_surface_color_nozzle_bed != config.machine.nozzle_axis_bed;
    if vis_state.dirty_surface_colors
        || (vis_state.surface_color_mode != ObjectSurfaceColorMode::DefaultCad
            && (tf_changed || grav_changed || nozzle_changed))
    {
        vis_state.dirty_surface_colors = false;
        vis_state.last_surface_color_transform = Some(obj_world_tf);
        vis_state.last_surface_color_gravity_bed = config.machine.gravity_direction_bed;
        vis_state.last_surface_color_nozzle_bed = config.machine.nozzle_axis_bed;

        let solved_axes = if vis_state.surface_color_mode == ObjectSurfaceColorMode::ToolAngleFromNormal
            && !vis_state.is_busy()
            && vis_state.available_fields.iter().any(|f| f == "tool_tilt_deg" || f == "eikonal")
            && vis_state
                .last_solved_field_rotation
                .map_or(true, |r| r.angle_between(obj_world_tf.rotation) < 1.0e-3)
        {
            ffi::get_object_vertex_tool_axes(obj_id)
        } else if vis_state.surface_color_mode == ObjectSurfaceColorMode::SupportNeeded
            && !vis_state.is_busy()
            && vis_state.available_fields.iter().any(|f| f == "support_needed" || f == "eikonal")
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
        vis_state.spawn_slicing_task(obj_id, &config, need_layers, want_toolpaths);
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

                    for seg in layer.contour_segments.chunks_exact(6) {
                        let p0 = obj_world_tf.transform_point(Vec3::new(seg[0], seg[1], seg[2]));
                        let p1 = obj_world_tf.transform_point(Vec3::new(seg[3], seg[4], seg[5]));
                        gizmos.line(p0, p1, line_color);
                    }
                }
            }
        }
    }

    // 2d. Plan G-code & Render Non-Planar Toolpaths (3D Variable-Width Extruded Bead Mesh)
    if vis_state.cached_toolpaths.is_some()
        && (vis_state.dirty_planned_gcode || vis_state.cached_planned_gcode.is_none())
        && !vis_state.ui_pointer_down
    {
        vis_state.dirty_planned_gcode = false;
        if let Some(ref tp_res) = vis_state.cached_toolpaths {
            let planned = crate::gcode::plan_and_emit_gcode_for_object(
                &scene_obj.name,
                scene_obj.object_id,
                tp_res,
                obj_world_tf,
                &config,
            );
            vis_state.cached_planned_gcode = Some(planned);
            vis_state.dirty_toolpaths_mesh = true;
        }
    }

    if !vis_state.show_toolpaths {
        if let Some(ent) = vis_state.toolpaths_entity {
            if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    } else if vis_state.dirty_toolpaths_mesh {
        vis_state.dirty_toolpaths_mesh = false;

        let stride = vis_state.selected_layer_stride.max(1);
        let min_idx = vis_state.selected_layer_min_idx;
        let max_idx = vis_state.selected_layer_max_idx;
        let highlighted = vis_state.highlighted_selected_layer;
        let color_mode = vis_state.toolpath_color_mode;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        if let Some(ref planned) = vis_state.cached_planned_gcode {
            build_planned_toolpaths_bead_mesh_parallel(
                planned,
                &vis_state,
                &config,
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
        } else if let Some(ref tp_res) = vis_state.cached_toolpaths {
            build_raw_toolpaths_bead_mesh_parallel(
                tp_res,
                &vis_state,
                &config,
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
                vis_state.toolpaths_entity,
                &vis_state.toolpaths_mesh_handle,
                &vis_state.toolpaths_mat_handle,
            ) {
                meshes.insert(mesh_h, bevy_mesh);
                if let Some(mat) = materials.get_mut(mat_h) {
                    mat.base_color = Color::WHITE;
                }
                if let Ok((mut tf, mut vis)) = toolpaths_query.get_mut(ent) {
                    *tf = obj_world_tf;
                    *vis = Visibility::Inherited;
                }
            } else {
                let mesh_h = meshes.add(bevy_mesh);
                let mat_h = materials.add(StandardMaterial {
                    base_color: Color::WHITE,
                    metallic: 0.0,
                    perceptual_roughness: 0.56,
                    reflectance: 0.16,
                    alpha_mode: AlphaMode::Opaque,
                    depth_bias: 14.0,
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

                for (idx, layer_p) in planned.layers.iter().enumerate() {
                    let is_hovered = highlighted == Some(idx);
                    let in_range = idx >= lo && idx <= hi;
                    let matches_stride =
                        (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
                    if !is_hovered && (!in_range || !matches_stride) {
                        continue;
                    }
                    for seg in &layer_p.segments {
                        let is_travel = matches!(
                            seg.kind,
                            crate::gcode::PlannedMoveKind::Travel
                                | crate::gcode::PlannedMoveKind::ZHop
                        );
                        if !is_travel && !vis_state.show_toolpath_centerlines {
                            continue;
                        }
                        if !vis_state.is_planned_segment_visible(seg) {
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
                        let p0 = obj_world_tf.transform_point(seg.p0_local);
                        let p1 = obj_world_tf.transform_point(seg.p1_local);
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
        if let Some(ent) = vis_state.toolpaths_entity {
            if let Ok((mut tf, _)) = toolpaths_query.get_mut(ent) {
                if *tf != obj_world_tf {
                    *tf = obj_world_tf;
                    vis_state.dirty_planned_gcode = true;
                }
            }
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
}

/// Builds 3D oriented bead prism geometry across visible layers of a `PlannedGcodeResult`
/// in parallel across CPU worker threads using `std::thread::scope`.
#[allow(clippy::too_many_arguments)]
fn build_planned_toolpaths_bead_mesh_parallel(
    planned: &crate::gcode::PlannedGcodeResult,
    vis_state: &FieldVisualizationState,
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
            if vis_state.is_planned_segment_visible(seg) {
                layer_segs += 1;
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

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(visible_layers.len());

    if num_threads <= 1 || total_est_segs < 256 {
        for &(idx, dim_factor, _) in &visible_layers {
            for seg in &planned.layers[idx].segments {
                if !vis_state.is_planned_segment_visible(seg) {
                    continue;
                }
                append_planned_segment_bead_mesh(
                    seg,
                    planned,
                    color_mode,
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
                for &(idx, dim_factor, _) in chunk {
                    for seg in &planned.layers[idx].segments {
                        if !vis_state.is_planned_segment_visible(seg) {
                            continue;
                        }
                        append_planned_segment_bead_mesh(
                            seg,
                            planned,
                            color_mode,
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
    vis_state: &FieldVisualizationState,
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

    let mut visible_layers: Vec<(usize, f32, usize)> = Vec::new();
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
            if vis_state.is_raw_path_visible(path.feature_kind) && path.points.len() >= 2 {
                layer_pts += path.points.len();
            }
        }
        if layer_pts > 0 {
            visible_layers.push((idx, dim_factor, layer_pts));
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
        for &(idx, dim_factor, _) in &visible_layers {
            for path in &tp_res.layers[idx].paths {
                if !vis_state.is_raw_path_visible(path.feature_kind) || path.points.len() < 2 {
                    continue;
                }
                append_extrusion_path_bead_mesh(
                    path,
                    color_mode,
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
                let mut c_pos = Vec::with_capacity(est * 4);
                let mut c_nrm = Vec::with_capacity(est * 4);
                let mut c_col = Vec::with_capacity(est * 4);
                let mut c_ind = Vec::with_capacity(est * 24);
                for &(idx, dim_factor, _) in chunk {
                    for path in &tp_res.layers[idx].paths {
                        if !vis_state.is_raw_path_visible(path.feature_kind)
                            || path.points.len() < 2
                        {
                            continue;
                        }
                        append_extrusion_path_bead_mesh(
                            path,
                            color_mode,
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

/// Appends a single planned segment (`PlannedSegment`) as a 3D oriented bead prism
/// in its local `NozzleFrame` $(\hat{\mathbf{x}}_{\text{nozzle}}, \hat{\mathbf{y}}_{\text{nozzle}}, \hat{\mathbf{z}}_{\text{nozzle}})$.
#[allow(clippy::too_many_arguments)]
fn append_planned_segment_bead_mesh(
    seg: &crate::gcode::PlannedSegment,
    planned: &crate::gcode::PlannedGcodeResult,
    color_mode: ToolpathColorMode,
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

    let mut rgba = compute_planned_segment_color(seg, planned, color_mode, config);
    rgba[0] = (rgba[0] * brightness_scale).clamp(0.0, 1.0);
    rgba[1] = (rgba[1] * brightness_scale).clamp(0.0, 1.0);
    rgba[2] = (rgba[2] * brightness_scale).clamp(0.0, 1.0);
    let linear_rgba = srgba_to_linear_rgba(rgba);

    let base_ring_start = positions.len() as u32;

    for (p, frame) in [(seg.p0_local, frame0), (seg.p1_local, frame1)] {
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

    let r0 = base_ring_start;
    let r1 = base_ring_start + 4;
    for k in 0..4u32 {
        let k_next = (k + 1) % 4;
        let a0 = r0 + k;
        let a1 = r0 + k_next;
        let b0 = r1 + k;
        let b1 = r1 + k_next;
        indices.extend_from_slice(&[a0, b0, b1, a0, b1, a1]);
    }
    // End caps
    indices.extend_from_slice(&[r0, r0 + 2, r0 + 1, r0, r0 + 3, r0 + 2]);
    indices.extend_from_slice(&[r1, r1 + 1, r1 + 2, r1, r1 + 2, r1 + 3]);
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
            _ => compute_toolpath_point_color(
                seg.kind as u8,
                seg.wall_index,
                seg.width_mm,
                ToolpathColorMode::ByFeatureType,
                config,
            ),
        },
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
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
) {
    let n = path.points.len();
    if n < 2 {
        return;
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

        // 4-sided cross-section ring in local (lateral_axis, nozzle_axis):
        // 0: top (+z_nozzle), 1: right (+y_nozzle), 2: bottom (-z_nozzle), 3: left (-y_nozzle)
        let v_top = p_curr + frame.nozzle_axis * half_h;
        let v_right = p_curr + frame.lateral_axis * half_w;
        let v_bottom = p_curr - frame.nozzle_axis * half_h;
        let v_left = p_curr - frame.lateral_axis * half_w;

        let mut rgba = compute_toolpath_point_color(
            path.feature_kind,
            path.wall_index,
            pt.width_mm,
            color_mode,
            config,
        );
        rgba[0] = (rgba[0] * brightness_scale).clamp(0.0, 1.0);
        rgba[1] = (rgba[1] * brightness_scale).clamp(0.0, 1.0);
        rgba[2] = (rgba[2] * brightness_scale).clamp(0.0, 1.0);
        let linear_rgba = srgba_to_linear_rgba(rgba);

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

    let seg_count = if path.is_closed && n >= 3 { n } else { n - 1 };
    for s in 0..seg_count {
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

    // End caps for open extrusion polylines
    if !path.is_closed {
        let r_start = base_ring_start;
        indices.extend_from_slice(&[r_start, r_start + 2, r_start + 1, r_start, r_start + 3, r_start + 2]);
        let r_end = base_ring_start + ((n - 1) as u32) * 4;
        indices.extend_from_slice(&[r_end, r_end + 1, r_end + 2, r_end, r_end + 2, r_end + 3]);
    }
}

/// Computes the RGBA color for a toolpath point according to `ToolpathColorMode`.
fn compute_toolpath_point_color(
    feature_kind: u8,
    wall_index: u32,
    width_mm: f32,
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
}


