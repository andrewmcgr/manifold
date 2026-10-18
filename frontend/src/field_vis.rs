use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::bridge::ffi::{self, FieldSlice2D, FieldStats};
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

/// Marker component for the 3D isosurface mesh entity extracted via `openvdb::tools::volumeToMesh`.
#[derive(Component)]
pub struct FieldIsosurfaceEntity;

/// Marker component for the 3D cross-section slice plane entity sampled via `openvdb::tools::GridSampler`.
#[derive(Component)]
pub struct FieldSliceEntity;

/// Generic scalar field visualization state resource supporting any named `openvdb::FloatGrid`
/// on the currently selected object (`"sdf"`, `"eikonal"`, `"overhang"`, and future fields).
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
        }
    }
}

impl FieldVisualizationState {
    pub fn mark_all_dirty(&mut self) {
        self.cached_stats = None;
        self.dirty_isosurface = true;
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
/// intersection contour curves ($\tau = c \cap \phi_{\text{sdf}} = 0$), and updates 2D
/// cross-section slices (`GridSampler`) in-place without entity despawn/spawn flicker.
#[allow(clippy::too_many_arguments)]
pub fn sync_field_visualization_system(
    mut commands: Commands,
    plate: Res<BuildPlateState>,
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
        ),
    >,
    mut iso_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldIsosurfaceEntity>,
            Without<CadObjectEntity>,
            Without<FieldSliceEntity>,
        ),
    >,
    mut slice_query: Query<
        (&mut Transform, &mut Visibility),
        (
            With<FieldSliceEntity>,
            Without<CadObjectEntity>,
            Without<FieldIsosurfaceEntity>,
        ),
    >,
) {
    let selected_id = plate.selected_object_id;

    // 1. Detect object selection change
    if vis_state.last_object_id != selected_id {
        vis_state.last_object_id = selected_id;
        vis_state.available_fields.clear();
        vis_state.cached_stats = None;
        vis_state.contour_segments_local.clear();
        vis_state.dirty_isosurface = true;
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
            let rgba_bytes = rasterize_slice_to_rgba(
                &slice,
                &vis_state.active_field_name,
                vis_state.isovalue,
                vis_state.colormap,
                vis_state.show_iso_contours,
                vis_state.fade_exterior_background,
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

    // 4. Keep active isosurface & slice entities locked to the selected object's Transform
    if vis_state.show_isosurface {
        if let Some(ent) = vis_state.isosurface_entity {
            if let Ok((mut tf, _)) = iso_query.get_mut(ent) {
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
/// using the selected `FieldColormap`, highlighting the object boundary ($\phi_{\text{sdf}} = 0$)
/// and the active isosurface intersection contour ($v = \text{isovalue}$ inside $\phi_{\text{sdf}} \le 0$).
fn rasterize_slice_to_rgba(
    slice: &FieldSlice2D,
    field_name: &str,
    active_isovalue: f32,
    colormap: FieldColormap,
    show_contours: bool,
    fade_background: bool,
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
            if is_non_sdf && sdf_v.abs() < 0.22 {
                // Crisp white outline along the object boundary (\phi_sdf = 0)
                r = 0.96;
                g = 0.98;
                b = 1.0;
            } else if is_non_sdf && (val - active_isovalue).abs() < iso_band && sdf_v <= 0.0 {
                // Bright cyan highlight where the active non-SDF isosurface intersects the object!
                r = 0.15;
                g = 1.0;
                b = 0.92;
            } else if !is_non_sdf && min_v < 0.0 && max_v > 0.0 && val.abs() < iso_band {
                r = 1.0;
                g = 1.0;
                b = 1.0;
            } else {
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
    use crate::bridge::ffi;

    #[test]
    fn test_openvdb_sdf_and_generic_field_visualization_pipeline() {
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

        let fields = ffi::list_object_fields(part.object_id);
        assert!(fields.contains(&"sdf".to_string()));
        assert!(fields.contains(&"eikonal".to_string()));
        assert!(fields.contains(&"overhang".to_string()));

        // Verify that changing voxel resolution preserves full interior & exterior millimeter range
        assert!(ffi::recompute_object_fields(part.object_id, 0.25, 6.0));
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
        );
        assert_eq!(rgba.len(), 32 * 32 * 4);

        let _ = ffi::remove_cad_object(part.object_id);
    }
}
