use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::bridge::ffi::{self, FieldSlice2D, FieldStats, LayerHeightSpec, SelectedIsosurfacesResult};
use crate::config::SlicerConfig;
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
    #[default]
    HighlightInsideDimOutside,
    /// Clips away all exterior triangles ($\phi_{\text{sdf}} > 0$), rendering only the exact
    /// portion of the isosurface that intersects the solid object ($\phi_{\text{sdf}} \le 0$).
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

/// Marker component for the 3D isosurface mesh entity extracted via `openvdb::tools::volumeToMesh`.
#[derive(Component)]
pub struct FieldIsosurfaceEntity;

/// Marker component for the 3D cross-section slice plane entity sampled via `openvdb::tools::GridSampler`.
#[derive(Component)]
pub struct FieldSliceEntity;

/// Marker component for the 3D combined mesh entity of selected slicing isosurfaces.
#[derive(Component)]
pub struct SelectedLayersMeshEntity;

/// Generic scalar field visualization state resource supporting any named `openvdb::FloatGrid`
/// on the currently selected object (`"sdf"`, `"eikonal"`, `"overhang"`, and future fields),
/// plus 3D/2D visualization of the selected non-planar slicing isosurfaces.
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
}

impl Default for FieldVisualizationState {
    fn default() -> Self {
        Self {
            active_field_name: "sdf".to_string(),
            available_fields: Vec::new(),
            cached_stats: None,
            last_object_id: None,
            solid_opacity: 1.0,
            show_isosurface: false,
            isovalue: 0.0,
            adaptivity: 0.0,
            intersection_mode: IsoIntersectionMode::HighlightInsideDimOutside,
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
        }
    }
}

impl FieldVisualizationState {
    pub fn mark_all_dirty(&mut self) {
        self.cached_stats = None;
        self.cached_selected_layers = None;
        self.dirty_isosurface = true;
        self.dirty_selected_layers = true;
        self.dirty_selected_layers_mesh = true;
        self.dirty_slice = true;
    }

    pub fn recompute_selected_layers(&mut self, object_id: u64, config: &SlicerConfig) {
        let lh = config.slicing.layer_height;
        ffi::set_slice_layer_height(LayerHeightSpec {
            target_mm: lh.target_mm,
            min_mm: lh.min_mm,
            max_mm: lh.max_mm,
            first_layer_mm: lh.first_layer_mm,
            top_layer_mm: lh.top_layer_mm,
        });
        let res = ffi::select_slicing_isosurfaces(object_id, self.adaptivity);
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
        self.dirty_selected_layers = false;
        self.dirty_selected_layers_mesh = true;
        self.dirty_slice = true;
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
        ),
    >,
    mut iso_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldIsosurfaceEntity>,
            Without<CadObjectEntity>,
            Without<FieldSliceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
    mut slice_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldSliceEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<SelectedLayersMeshEntity>,
        ),
    >,
    mut selected_layers_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<SelectedLayersMeshEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
            Without<FieldSliceEntity>,
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
        vis_state.contour_segments_local.clear();
        vis_state.dirty_isosurface = true;
        vis_state.dirty_selected_layers = true;
        vis_state.dirty_selected_layers_mesh = true;
        vis_state.dirty_slice = true;

        if let Some(id) = selected_id {
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
        return;
    };

    let Some(scene_obj) = plate.objects.iter().find(|o| o.object_id == obj_id) else {
        return;
    };

    let obj_world_tf = object_transforms
        .get(scene_obj.entity)
        .copied()
        .unwrap_or(Transform::IDENTITY);

    if vis_state.cached_stats.is_none() {
        vis_state.refresh_stats_for(obj_id);
    }

    // Check if hover highlight on the selected layers list changed
    if vis_state.last_highlighted_selected_layer != vis_state.highlighted_selected_layer {
        vis_state.last_highlighted_selected_layer = vis_state.highlighted_selected_layer;
        vis_state.dirty_selected_layers_mesh = true;
        if vis_state.show_slice_plane && vis_state.show_selected_layers {
            vis_state.dirty_slice = true;
        }
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
    } else if vis_state.dirty_isosurface {
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
                colors.push(rgba);
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
                    metallic: 0.12,
                    perceptual_roughness: 0.36,
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

    // 2c. Update & Render Selected Slicing Isosurfaces Stack (3D Clipped Surfaces + Contours)
    if vis_state.show_selected_layers
        && vis_state.dirty_selected_layers
        && vis_state.available_fields.iter().any(|f| f == "eikonal")
    {
        vis_state.recompute_selected_layers(obj_id, &config);
    }

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
                let matches_stride =
                    ((idx - lo) % stride == 0) || layer.is_first_layer || layer.is_top_layer;
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

                for v in &layer.vertices {
                    positions.push([v.x, v.y, v.z]);
                    normals.push([v.nx, v.ny, v.nz]);
                    colors.push(rgba);
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
                    metallic: 0.10,
                    perceptual_roughness: 0.38,
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
                    let matches_stride =
                        ((idx - lo) % stride == 0) || layer.is_first_layer || layer.is_top_layer;
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

    // 3. Update 2D Cross-Section Slice Plane in-place using AlphaMode::Mask(0.5)
    //    so hardware depth testing is active and there is zero transparency sort flicker!
    if !vis_state.show_slice_plane {
        if let Some(ent) = vis_state.slice_entity {
            if let Ok((_, mut vis)) = slice_query.get_mut(ent) {
                *vis = Visibility::Hidden;
            }
        }
    } else if vis_state.dirty_slice {
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
                                && (((idx - lo) % stride == 0) || l.is_first_layer || l.is_top_layer)
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

    // 4. Keep active isosurface, selected layers, & slice entities locked to the selected object's Transform
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

/// Computes the RGBA color for a selected slicing isosurface layer according to `SelectedLayersColorMode`.
fn compute_selected_layer_color(
    layer: &ffi::SelectedLayerInfo,
    mode: SelectedLayersColorMode,
    min_mm: f32,
    target_mm: f32,
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
            if h <= target_mm {
                let denom = (target_mm - min_mm).max(1.0e-4);
                let t = ((h - min_mm) / denom).clamp(0.0, 1.0);
                // Cool Blue (min_mm) -> Emerald Green (target_mm)
                let r = 0.16 * (1.0 - t) + 0.18 * t;
                let g = 0.52 * (1.0 - t) + 0.88 * t;
                let b = 0.96 * (1.0 - t) + 0.42 * t;
                [r, g, b, 1.0]
            } else {
                let denom = (max_mm - target_mm).max(1.0e-4);
                let t = ((h - target_mm) / denom).clamp(0.0, 1.0);
                // Emerald Green (target_mm) -> Warm Amber/Coral (max_mm)
                let r = 0.18 * (1.0 - t) + 0.98 * t;
                let g = 0.88 * (1.0 - t) + 0.50 * t;
                let b = 0.42 * (1.0 - t) + 0.16 * t;
                [r, g, b, 1.0]
            }
        }
        SelectedLayersColorMode::ByBranch => {
            const PALETTE: [[f32; 3]; 8] = [
                [0.20, 0.86, 0.62],
                [0.96, 0.62, 0.20],
                [0.36, 0.68, 0.98],
                [0.94, 0.38, 0.76],
                [0.72, 0.90, 0.24],
                [0.98, 0.42, 0.38],
                [0.58, 0.48, 0.96],
                [0.24, 0.90, 0.92],
            ];
            let rgb = PALETTE[(layer.branch_id as usize) % PALETTE.len()];
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
}

