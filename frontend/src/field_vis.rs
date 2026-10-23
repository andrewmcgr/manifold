use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::tasks::IoTaskPool;

use crate::bridge::ffi::{
    self, ExtrusionPath, FieldSlice2D, FieldStats, LayerHeightSpec, LineWidthSpec,
    ObjectToolpathsResult, SelectedIsosurfacesResult, ShellPathSpec,
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
    pub show_solid_infill: bool,
    pub show_sparse_infill: bool,
    pub show_travel_moves: bool,
    pub show_scarf_and_wipes: bool,
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
            show_solid_infill: true,
            show_sparse_infill: true,
            show_travel_moves: false,
            show_scarf_and_wipes: true,
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
        };
        let adaptivity = self.adaptivity;

        let task = IoTaskPool::get().spawn(async move {
            ffi::set_slice_layer_height(lh_spec);
            ffi::set_shell_path_spec(shell_spec);

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

    // If no object is selected, hide active field visualization entities
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
        if let Some(ent) = vis_state.toolpaths_entity {
            if let Ok((_, mut vis)) = toolpaths_query.get_mut(ent) {
                *vis = Visibility::Hidden;
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
        let show_outer = vis_state.show_outer_walls;
        let show_inner = vis_state.show_inner_walls;
        let show_infill = vis_state.show_solid_infill;
        let show_sparse = vis_state.show_sparse_infill;
        let show_scarf_wipes = vis_state.show_scarf_and_wipes;
        let show_travels = vis_state.show_travel_moves;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        if let Some(ref planned) = vis_state.cached_planned_gcode {
            let total = planned.layers.len();
            if total > 0 {
                let lo = min_idx.min(total - 1);
                let hi = max_idx.min(total - 1).max(lo);

                let mut est_segs = 0usize;
                for (idx, layer_p) in planned.layers.iter().enumerate() {
                    let is_hovered = highlighted == Some(idx);
                    let in_range = idx >= lo && idx <= hi;
                    let matches_stride =
                        (idx.saturating_sub(lo) % stride == 0) || idx == 0 || idx + 1 == total;
                    if !is_hovered && (!in_range || !matches_stride) {
                        continue;
                    }
                    for seg in &layer_p.segments {
                        if is_planned_segment_visible(
                            seg,
                            show_outer,
                            show_inner,
                            show_infill,
                            show_sparse,
                            show_scarf_wipes,
                            show_travels,
                        ) {
                            est_segs += 1;
                        }
                    }
                }
                positions.reserve(est_segs * 8);
                normals.reserve(est_segs * 8);
                colors.reserve(est_segs * 8);
                indices.reserve(est_segs * 36);

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

                    for seg in &layer_p.segments {
                        if !is_planned_segment_visible(
                            seg,
                            show_outer,
                            show_inner,
                            show_infill,
                            show_sparse,
                            show_scarf_wipes,
                            show_travels,
                        ) {
                            continue;
                        }
                        append_planned_segment_bead_mesh(
                            seg,
                            planned,
                            color_mode,
                            &config,
                            dim_factor,
                            &mut positions,
                            &mut normals,
                            &mut colors,
                            &mut indices,
                        );
                    }
                }
            }
        } else if let Some(ref tp_res) = vis_state.cached_toolpaths {
            let total = tp_res.layers.len();
            if total > 0 {
                let lo = min_idx.min(total - 1);
                let hi = max_idx.min(total - 1).max(lo);

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

                    for path in &layer_tp.paths {
                        let visible = match path.feature_kind {
                            0 => show_outer,
                            1 => show_inner,
                            4 => show_sparse,
                            _ => show_infill,
                        };
                        if !visible || path.points.len() < 2 {
                            continue;
                        }
                        append_extrusion_path_bead_mesh(
                            path,
                            color_mode,
                            &config,
                            dim_factor,
                            &mut positions,
                            &mut normals,
                            &mut colors,
                            &mut indices,
                        );
                    }
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
        && (vis_state.show_toolpath_centerlines || vis_state.show_travel_moves)
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
                        if is_travel && !vis_state.show_travel_moves {
                            continue;
                        }
                        if !is_travel && !vis_state.show_toolpath_centerlines {
                            continue;
                        }
                        if !is_planned_segment_visible(
                            seg,
                            vis_state.show_outer_walls,
                            vis_state.show_inner_walls,
                            vis_state.show_solid_infill,
                            vis_state.show_sparse_infill,
                            vis_state.show_scarf_and_wipes,
                            vis_state.show_travel_moves,
                        ) {
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

/// Returns whether a `PlannedSegment` should be rendered given the current UI visibility toggles.
fn is_planned_segment_visible(
    seg: &crate::gcode::PlannedSegment,
    show_outer: bool,
    show_inner: bool,
    show_infill: bool,
    show_sparse: bool,
    show_scarf_wipes: bool,
    show_travels: bool,
) -> bool {
    match seg.kind {
        crate::gcode::PlannedMoveKind::Travel | crate::gcode::PlannedMoveKind::ZHop => show_travels,
        crate::gcode::PlannedMoveKind::ScarfSeam | crate::gcode::PlannedMoveKind::Wipe => {
            show_scarf_wipes && (show_outer || show_inner)
        }
        crate::gcode::PlannedMoveKind::OuterWall => show_outer,
        crate::gcode::PlannedMoveKind::InnerWall => show_inner,
        crate::gcode::PlannedMoveKind::SparseInfill => show_sparse,
        crate::gcode::PlannedMoveKind::SolidInfill | crate::gcode::PlannedMoveKind::GapFill => {
            show_infill
        }
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

    let half_w = (0.48 * w_eff).max(0.03);
    let half_h = (0.42 * h_eff).max(0.02);

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

        let half_w = (0.48 * pt.width_mm).max(0.03);
        let half_h = (0.42 * pt.height_mm).max(0.02);

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
            _ => [0.98, 0.32, 0.88, 1.0], // Gap Fill: vivid magenta
        },
        ToolpathColorMode::ByLineWidth | _ => {
            let spec = match feature_kind {
                0 => config.slicing.outer_wall_width,
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
            },
            ffi::SpatialOrientationHint {
                kind: 0, // Surface tangency hint on +X wall
                position_bed: [8.0, 0.0, 8.0],
                direction_bed: [1.0, 0.0, 0.0],
                influence_radius_mm: 8.0,
                weight: 0.9,
                apply_on_surface_only: true,
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
            let mean_seg_h =
                ext_segs.iter().map(|s| s.height_mm).sum::<f32>() / (ext_segs.len() as f32);
            assert!(
                (mean_seg_h - l.mean_thickness_mm).abs() <= 0.03,
                "Layer #{i} (branch B{}, {}/{}) mean planned segment height ({mean_seg_h:.3} mm) diverged from layer mean_thickness_mm ({:.3} mm)",
                l.branch_id,
                l.branch_layer_index + 1,
                l.branch_layer_count,
                l.mean_thickness_mm,
            );
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

        let _ = ffi::remove_cad_object(part.object_id);
    }
}


