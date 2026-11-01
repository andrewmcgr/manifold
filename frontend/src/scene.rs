use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::{Task};
use bevy::window::PrimaryWindow;
use bevy_egui::EguiContexts;
use wgpu::util::DeviceExt;

use crate::arrange::{auto_arrange_spiral, ArrangeItem};
use crate::bridge::ffi::{self, RenderVertex};
use crate::config::{
    BedShape, PointNormalHint, SlicerConfig, SurfaceFaceHint, SurfaceHintMode,
};
use crate::field_vis::SliceAxis;
use crate::hull::{
    ray_intersect_triangle, rotation_to_place_face_on_bed, SimplifiedConvexHull, SurfacePatch,
};
use crate::io::{LoadedBatchReport, LoadedObjectData};

/// Active user editing tool mode selectable from the top/side toolbar.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolMode {
    #[default]
    Select,
    MoveOnBed,
    Rotate,
    PlaceOnFace,
    /// Select STEP B-Rep faces or contiguous planar mesh patches to suggest $\nabla \tau \parallel \pm \hat{\mathbf{n}}_{\text{face}}$ (isosurfaces conformal/tangent to the face).
    SurfaceHintOrthogonal,
    /// Select STEP B-Rep faces or contiguous planar mesh patches to suggest $\nabla \tau \perp \hat{\mathbf{n}}_{\text{face}}$ (isosurfaces orthogonal to the face).
    SurfaceHintTangent,
    /// Place & orient 3D point normal hints in space (on the active 2D slice plane or object surface).
    PointHint,
}

impl ToolMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select / Orbit",
            Self::MoveOnBed => "Move on Bed (XY)",
            Self::Rotate => "Rotate / Orient",
            Self::PlaceOnFace => "Place on Face (Hull)",
            Self::SurfaceHintOrthogonal => "Hint: Face ⊥ Normal (Conformal)",
            Self::SurfaceHintTangent => "Hint: Face ∥ Normal (⊥ Layers)",
            Self::PointHint => "Hint: 3D Point Normal",
        }
    }
}

/// Marker component linking a Bevy 3D mesh `Entity` back to its backend `object_id`.
#[derive(Component, Clone, Copy, Debug)]
pub struct CadObjectEntity {
    pub object_id: u64,
}

/// Interactive 3D orbit camera controller for the build-plate viewport.
#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::new(0.0, 0.0, 25.0),
            radius: 340.0,
            yaw: -std::f32::consts::FRAC_PI_4,
            pitch: 0.52,
        }
    }
}

/// Complete state of a single loaded object on the build plate.
pub struct SceneObject {
    pub object_id: u64,
    pub name: String,
    pub source_file: String,
    pub has_brep: bool,
    pub entity: Entity,
    pub mesh_handle: Handle<Mesh>,
    pub material_handle: Handle<StandardMaterial>,
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub hull: SimplifiedConvexHull,
    pub surface_patches: Vec<SurfacePatch>,
    pub triangle_to_patch: Vec<u32>,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub visible: bool,
}

impl SceneObject {
    /// Computes the exact world-space AABB `(world_min, world_max)` for the object
    /// under `transform` by transforming its convex hull polygon vertices (plus local bounds).
    pub fn compute_world_aabb(&self, transform: &Transform) -> (Vec3, Vec3) {
        let mut min_pt = Vec3::splat(f32::INFINITY);
        let mut max_pt = Vec3::splat(f32::NEG_INFINITY);

        if !self.hull.faces.is_empty() {
            for face in &self.hull.faces {
                for &p in &face.polygon {
                    let wp = transform.transform_point(p);
                    min_pt = min_pt.min(wp);
                    max_pt = max_pt.max(wp);
                }
            }
        }

        // Also sample a strided subset of mesh vertices to guarantee exact Z-min
        let step = (self.vertices.len() / 512).max(1);
        for v in self.vertices.iter().step_by(step) {
            let wp = transform.transform_point(Vec3::new(v.x, v.y, v.z));
            min_pt = min_pt.min(wp);
            max_pt = max_pt.max(wp);
        }

        if min_pt.x.is_infinite() {
            (transform.translation, transform.translation)
        } else {
            (min_pt, max_pt)
        }
    }

    /// Raycasts a world-space ray against this object's triangle mesh.
    pub fn raycast_mesh(&self, transform: &Transform, ray: Ray3d) -> Option<f32> {
        self.raycast_surface_patch(transform, ray)
            .map(|(_, _, _, world_dist)| world_dist)
    }

    /// Raycasts a world-space ray against this object's triangle mesh, returning
    /// `(patch_id, local_hit_point, local_hit_normal, world_distance)`.
    pub fn raycast_surface_patch(
        &self,
        transform: &Transform,
        ray: Ray3d,
    ) -> Option<(u32, Vec3, Vec3, f32)> {
        let (aabb_min, aabb_max) = self.compute_world_aabb(transform);
        if !ray_intersects_aabb(ray, aabb_min, aabb_max) {
            return None;
        }

        // Transform the ray into the object's local coordinate space for fast testing
        let inv_mat = transform.compute_matrix().inverse();
        let local_origin = inv_mat.transform_point3(ray.origin);
        let local_dir = inv_mat.transform_vector3(*ray.direction).normalize_or_zero();
        if local_dir.length_squared() < 1e-6 {
            return None;
        }

        let mut best_hit: Option<(usize, f32, Vec3)> = None;
        for (tri_idx, tri) in self.indices.chunks_exact(3).enumerate() {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];
            let p0 = Vec3::new(v0.x, v0.y, v0.z);
            let p1 = Vec3::new(v1.x, v1.y, v1.z);
            let p2 = Vec3::new(v2.x, v2.y, v2.z);
            if let Some(t) = ray_intersect_triangle(local_origin, local_dir, p0, p1, p2) {
                if best_hit.map_or(true, |(_, best_t, _)| t < best_t) {
                    let n = (p1 - p0).cross(p2 - p0).normalize_or_zero();
                    best_hit = Some((tri_idx, t, n));
                }
            }
        }

        best_hit.map(|(tri_idx, t, tri_normal)| {
            let patch_id = self
                .triangle_to_patch
                .get(tri_idx)
                .copied()
                .unwrap_or(0);
            let patch_normal = self
                .surface_patches
                .get(patch_id as usize)
                .map(|p| p.normal)
                .unwrap_or(tri_normal);
            let local_hit = local_origin + local_dir * t;
            let world_hit = transform.transform_point(local_hit);
            let world_dist = ray.origin.distance(world_hit);
            (patch_id, local_hit, patch_normal, world_dist)
        })
    }
}

/// Adjusts `transform.translation.z` so that the rotated object's lowest point ($z_{\min}$)
/// rests flush on the print bed at $z = 0.0$.
pub fn snap_object_to_bed(obj: &SceneObject, transform: &mut Transform) {
    let (world_min, _) = obj.compute_world_aabb(transform);
    transform.translation.z -= world_min.z;
}

/// Central resource managing all objects on the build plate, active editing tool,
/// selection, hover state, and background I/O tasks.
#[derive(Resource, Default)]
pub struct BuildPlateState {
    pub objects: Vec<SceneObject>,
    pub selected_object_id: Option<u64>,
    pub tool_mode: ToolMode,
    pub hovered_hull_face: Option<usize>,
    pub hovered_surface_patch: Option<u32>,
    pub hovered_point_preview: Option<(Vec3, Vec3)>,
    pub selected_surface_hint_patch: Option<u32>,
    pub selected_point_hint_id: Option<u32>,
    pub ui_hovered_surface_hint_patch: Option<u32>,
    pub ui_hovered_point_hint_id: Option<u32>,
    pub is_dragging_point_hint_dir: bool,
    pub is_dragging_object: bool,
    pub drag_offset_xy: Vec2,
    pub clone_copies_count: u32,
    /// Remembered `.3mf` file name when a Manifold `.3mf` project was loaded (used as the default on Save 3MF).
    pub project_3mf_filename: Option<String>,
    pub status_message: String,
    pub import_notices: Vec<String>,
    pub import_errors: Vec<String>,
    pub render_pipeline: Option<wgpu::RenderPipeline>,
    pub pending_load_task: Option<Task<Option<LoadedBatchReport>>>,
    pub pending_save_task: Option<Task<Option<Result<String, String>>>>,
}

impl BuildPlateState {
    pub fn selected_object(&self) -> Option<&SceneObject> {
        let id = self.selected_object_id?;
        self.objects.iter().find(|o| o.object_id == id)
    }

    pub fn total_vertices(&self) -> usize {
        self.objects.iter().map(|o| o.vertices.len()).sum()
    }

    pub fn total_triangles(&self) -> usize {
        self.objects.iter().map(|o| o.indices.len() / 3).sum()
    }
}

/// Spawns newly loaded `LoadedObjectData` items into the Bevy 3D scene and uploads
/// their `RenderVertex` and `u32` buffers to `wgpu`.
#[allow(clippy::too_many_arguments)]
pub fn spawn_loaded_objects(
    commands: &mut Commands,
    render_device: Option<&RenderDevice>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    plate: &mut BuildPlateState,
    loaded_objects: Vec<LoadedObjectData>,
) {
    for data in loaded_objects {
        let mut vertex_buffer = None;
        let mut index_buffer = None;

        if let Some(device) = render_device {
            let wgpu_device = device.wgpu_device();
            let vertex_bytes: &[u8] = bytemuck::cast_slice(&data.vertices);
            let index_bytes: &[u8] = bytemuck::cast_slice(&data.indices);

            vertex_buffer = Some(wgpu_device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("occt_tessellated_vertex_buffer"),
                    contents: vertex_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                },
            ));

            index_buffer = Some(wgpu_device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("occt_tessellated_index_buffer"),
                    contents: index_bytes,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                },
            ));
        }

        let mut positions = Vec::with_capacity(data.vertices.len());
        let mut normals = Vec::with_capacity(data.vertices.len());
        let colors = vec![[1.0f32, 1.0, 1.0, 1.0]; data.vertices.len()];
        for v in &data.vertices {
            positions.push([v.x, v.y, v.z]);
            normals.push([v.nx, v.ny, v.nz]);
        }

        let bevy_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(data.indices.clone()));

        let mesh_handle = meshes.add(bevy_mesh);

        let mat_handle = materials.add(StandardMaterial {
            base_color: Color::srgb(0.24, 0.62, 0.92),
            metallic: 0.22,
            perceptual_roughness: 0.45,
            double_sided: true,
            cull_mode: None,
            ..default()
        });

        let mut initial_transform = Transform {
            translation: data.initial_translation,
            rotation: data.initial_rotation,
            scale: Vec3::ONE,
        };

        let entity = commands
            .spawn((
                PbrBundle {
                    mesh: mesh_handle.clone(),
                    material: mat_handle.clone(),
                    transform: initial_transform,
                    ..default()
                },
                CadObjectEntity {
                    object_id: data.object_id,
                },
            ))
            .id();

        let scene_obj = SceneObject {
            object_id: data.object_id,
            name: data.name,
            source_file: data.source_file,
            has_brep: data.has_brep,
            entity,
            mesh_handle,
            material_handle: mat_handle,
            vertices: data.vertices,
            indices: data.indices,
            local_min: data.local_min,
            local_max: data.local_max,
            hull: data.hull,
            surface_patches: data.surface_patches,
            triangle_to_patch: data.triangle_to_patch,
            vertex_buffer,
            index_buffer,
            visible: true,
        };

        snap_object_to_bed(&scene_obj, &mut initial_transform);
        commands.entity(entity).insert(initial_transform);

        plate.selected_object_id = Some(scene_obj.object_id);
        plate.objects.push(scene_obj);
    }
}

/// Removes a single object by `object_id` from both the Bevy 3D viewport and the
/// C++ backend without disturbing any other objects (including sibling parts from
/// a decomposed STEP assembly).
pub fn remove_object_from_plate(
    commands: &mut Commands,
    plate: &mut BuildPlateState,
    object_id: u64,
) {
    if let Some(pos) = plate.objects.iter().position(|o| o.object_id == object_id) {
        let removed = plate.objects.remove(pos);
        commands.entity(removed.entity).despawn_recursive();
        let _ = ffi::remove_cad_object(object_id);

        if plate.selected_object_id == Some(object_id) {
            plate.selected_object_id = plate.objects.last().map(|o| o.object_id);
        }
        plate.hovered_hull_face = None;
        plate.hovered_surface_patch = None;
        plate.hovered_point_preview = None;
        plate.status_message = format!("Removed object '{}' (ID {object_id}).", removed.name);
    }
}

fn strip_clone_number_suffix(name: &str) -> &str {
    if let Some(open_idx) = name.rfind(" (") {
        if name.ends_with(')') {
            let inside = &name[open_idx + 2..name.len() - 1];
            if !inside.is_empty() && inside.chars().all(|c| c.is_ascii_digit()) {
                return &name[..open_idx];
            }
        }
    }
    name
}

/// Clones an already-imported object `copies` times in both the C++ backend and the
/// Bevy 3D viewport, placing each clone at a non-overlapping XY position on the build plate
/// and copying any per-object orientation hints.
#[allow(clippy::too_many_arguments)]
pub fn clone_object_on_plate(
    commands: &mut Commands,
    render_device: Option<&RenderDevice>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    plate: &mut BuildPlateState,
    config: &mut SlicerConfig,
    transform_query: &Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    source_object_id: u64,
    copies: usize,
) {
    let copies = copies.clamp(1, 64);
    let Some(src_idx) = plate
        .objects
        .iter()
        .position(|o| o.object_id == source_object_id)
    else {
        return;
    };

    let src_tf = transform_query
        .get(plate.objects[src_idx].entity)
        .map(|(t, _)| *t)
        .unwrap_or(Transform::IDENTITY);
    let (src_min, src_max) = plate.objects[src_idx].compute_world_aabb(&src_tf);
    let half_extents_xy = 0.5 * Vec2::new(src_max.x - src_min.x, src_max.y - src_min.y);
    let src_center_xy = 0.5 * Vec2::new(src_min.x + src_max.x, src_min.y + src_max.y);
    let center_offset_xy = src_center_xy - src_tf.translation.xy();

    let half_clearance = 0.5 * config.arrangement.part_clearance_mm.max(0.0);
    let dtheta = config.arrangement.spiral_angle_step_rad.max(0.05);
    let b_coeff = config.arrangement.spiral_radial_pitch_mm.max(1.0) / std::f32::consts::TAU;

    let mut placed_boxes: Vec<(Vec2, Vec2)> = Vec::with_capacity(plate.objects.len() + copies);
    for obj in &plate.objects {
        let tf = transform_query
            .get(obj.entity)
            .map(|(t, _)| *t)
            .unwrap_or(Transform::IDENTITY);
        let (w_min, w_max) = obj.compute_world_aabb(&tf);
        placed_boxes.push((
            w_min.xy() - Vec2::splat(half_clearance),
            w_max.xy() + Vec2::splat(half_clearance),
        ));
    }

    let base_name = strip_clone_number_suffix(&plate.objects[src_idx].name).to_string();
    let src_source_file = plate.objects[src_idx].source_file.clone();
    let src_has_brep = plate.objects[src_idx].has_brep;
    let src_vertices = plate.objects[src_idx].vertices.clone();
    let src_indices = plate.objects[src_idx].indices.clone();
    let src_local_min = plate.objects[src_idx].local_min;
    let src_local_max = plate.objects[src_idx].local_max;
    let src_hull = plate.objects[src_idx].hull.clone();
    let src_patches = plate.objects[src_idx].surface_patches.clone();
    let src_tri_to_patch = plate.objects[src_idx].triangle_to_patch.clone();

    let src_surface_hints: Vec<SurfaceFaceHint> = config
        .eikonal
        .surface_hints
        .iter()
        .filter(|h| h.object_id == source_object_id)
        .cloned()
        .collect();
    let src_point_hints: Vec<PointNormalHint> = config
        .eikonal
        .point_hints
        .iter()
        .filter(|h| h.object_id == source_object_id)
        .cloned()
        .collect();

    let mut next_point_hint_id = config
        .eikonal
        .point_hints
        .iter()
        .map(|h| h.id)
        .max()
        .unwrap_or(0)
        + 1;

    let mut new_loaded_objects = Vec::with_capacity(copies);
    let mut theta = dtheta;
    let padded_half = half_extents_xy + Vec2::splat(half_clearance);

    for _ in 0..copies {
        let mut suffix_num = 2usize;
        let clone_name = loop {
            let candidate = format!("{base_name} ({suffix_num})");
            let used_on_plate = plate.objects.iter().any(|o| o.name == candidate);
            let used_in_batch = new_loaded_objects
                .iter()
                .any(|o: &LoadedObjectData| o.name == candidate);
            if !used_on_plate && !used_in_batch {
                break candidate;
            }
            suffix_num += 1;
        };

        let new_id = ffi::clone_cad_object(source_object_id, &clone_name);
        if new_id == 0 {
            continue;
        }

        // Find the next non-overlapping XY position along an Archimedean spiral around src_center_xy
        let chosen_center = loop {
            let r = b_coeff * theta;
            let candidate_center = src_center_xy + Vec2::new(r * theta.cos(), r * theta.sin());
            let cand_min = candidate_center - padded_half;
            let cand_max = candidate_center + padded_half;

            let overlaps = placed_boxes.iter().any(|&(other_min, other_max)| {
                cand_min.x < other_max.x
                    && cand_max.x > other_min.x
                    && cand_min.y < other_max.y
                    && cand_max.y > other_min.y
            });

            theta += dtheta;
            if r > 20.0 {
                let adaptive_dtheta =
                    (config.arrangement.spiral_radial_pitch_mm * 0.35 / r).clamp(0.02, dtheta);
                theta += adaptive_dtheta - dtheta;
            }

            if !overlaps {
                placed_boxes.push((cand_min, cand_max));
                break candidate_center;
            }
        };

        for sh in &src_surface_hints {
            let mut cloned_sh = sh.clone();
            cloned_sh.object_id = new_id;
            config.eikonal.surface_hints.push(cloned_sh);
        }
        for ph in &src_point_hints {
            let mut cloned_ph = ph.clone();
            cloned_ph.id = next_point_hint_id;
            next_point_hint_id += 1;
            cloned_ph.object_id = new_id;
            config.eikonal.point_hints.push(cloned_ph);
        }

        let target_translation_xy = chosen_center - center_offset_xy;
        new_loaded_objects.push(LoadedObjectData {
            object_id: new_id,
            name: clone_name,
            source_file: src_source_file.clone(),
            has_brep: src_has_brep,
            initial_translation: Vec3::new(target_translation_xy.x, target_translation_xy.y, 0.0),
            initial_rotation: src_tf.rotation,
            vertices: src_vertices.clone(),
            indices: src_indices.clone(),
            local_min: src_local_min,
            local_max: src_local_max,
            hull: src_hull.clone(),
            surface_patches: src_patches.clone(),
            triangle_to_patch: src_tri_to_patch.clone(),
        });
    }

    let spawned_count = new_loaded_objects.len();
    if spawned_count > 0 {
        spawn_loaded_objects(
            commands,
            render_device,
            meshes,
            materials,
            plate,
            new_loaded_objects,
        );
        plate.status_message = format!(
            "Added {spawned_count} clone(s) of '{base_name}'; {} object(s) on build plate.",
            plate.objects.len()
        );
    }
}

/// Runs the descending-Z-height Archimedean spiral auto-arrangement across all
/// objects currently on the build plate.
pub fn run_auto_arrange_on_plate(
    plate: &mut BuildPlateState,
    config: &SlicerConfig,
    transform_query: &mut Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
) {
    if plate.objects.is_empty() {
        plate.status_message = "No objects on the build plate to arrange.".to_string();
        return;
    }

    let mut items = Vec::with_capacity(plate.objects.len());
    for obj in &plate.objects {
        let transform = transform_query
            .get(obj.entity)
            .map(|(t, _)| *t)
            .unwrap_or(Transform::IDENTITY);
        let (w_min, w_max) = obj.compute_world_aabb(&transform);
        let half_extents_xy = 0.5 * Vec2::new(w_max.x - w_min.x, w_max.y - w_min.y);
        let center_xy = 0.5 * Vec2::new(w_min.x + w_max.x, w_min.y + w_max.y);
        let center_offset_xy = center_xy - transform.translation.xy();
        let height_z = (w_max.z - w_min.z).max(0.1);

        items.push(ArrangeItem {
            object_id: obj.object_id,
            half_extents_xy,
            center_offset_xy,
            height_z,
        });
    }

    let placements = auto_arrange_spiral(&items, &config.bed, &config.arrangement);
    let mut out_of_bounds_count = 0;

    for placement in &placements {
        if !placement.fits_on_bed {
            out_of_bounds_count += 1;
        }
        if let Some(obj) = plate
            .objects
            .iter()
            .find(|o| o.object_id == placement.object_id)
        {
            if let Ok((mut transform, _)) = transform_query.get_mut(obj.entity) {
                transform.translation.x = placement.target_x;
                transform.translation.y = placement.target_y;
                snap_object_to_bed(obj, &mut transform);
            }
        }
    }

    if out_of_bounds_count == 0 {
        plate.status_message = format!(
            "Auto-arranged {} object(s) in spiral order by descending Z height.",
            placements.len()
        );
    } else {
        plate.status_message = format!(
            "Auto-arranged {} object(s); warning: {} object(s) exceed current bed dimensions.",
            placements.len(),
            out_of_bounds_count
        );
    }
}

/// Updates object material highlights so the selected object is visually distinct,
/// while preserving translucency (with backface culling) when any visualization is active.
pub fn update_selection_materials_system(
    plate: Res<BuildPlateState>,
    vis_state: Res<crate::field_vis::FieldVisualizationState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !plate.is_changed() && !vis_state.is_changed() {
        return;
    }

    for obj in &plate.objects {
        if let Some(mat) = materials.get_mut(&obj.material_handle) {
            let is_selected = plate.selected_object_id == Some(obj.object_id);
            if is_selected {
                let any_vis_active = vis_state.show_isosurface
                    || vis_state.show_slice_plane
                    || vis_state.show_selected_layers
                    || vis_state.show_toolpaths;
                let alpha = if any_vis_active {
                    vis_state.solid_opacity.clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let use_vertex_colormap = vis_state.surface_color_mode
                    != crate::field_vis::ObjectSurfaceColorMode::DefaultCad;
                if alpha < 0.02 {
                    mat.base_color = Color::srgba(0.55, 0.60, 0.68, 0.0);
                    mat.alpha_mode = AlphaMode::Mask(1.0);
                    mat.cull_mode = Some(wgpu::Face::Back);
                } else if alpha < 0.99 {
                    if use_vertex_colormap {
                        mat.base_color = Color::srgba(1.0, 1.0, 1.0, alpha);
                        mat.metallic = 0.0;
                        mat.perceptual_roughness = 0.78;
                        mat.reflectance = 0.08;
                    } else {
                        // Neutral, low-glare ghost shell so it does not cast a colored or specular
                        // veil over internal toolpaths, isosurfaces, or scalar slices.
                        mat.base_color = Color::srgba(0.52, 0.56, 0.64, alpha);
                        mat.metallic = 0.0;
                        mat.perceptual_roughness = 0.90;
                        mat.reflectance = 0.06;
                    }
                    mat.alpha_mode = AlphaMode::Blend;
                    // Backface culling prevents the translucent CAD shell from self-interfering
                    mat.cull_mode = Some(wgpu::Face::Back);
                    mat.double_sided = false;
                } else {
                    if use_vertex_colormap {
                        // Pure white base_color so per-vertex ATTRIBUTE_COLOR directly drives the surface colormap
                        mat.base_color = Color::srgba(1.0, 1.0, 1.0, 1.0);
                        mat.metallic = 0.04;
                        mat.perceptual_roughness = 0.62;
                        mat.reflectance = 0.18;
                    } else {
                        mat.base_color = Color::srgba(0.32, 0.76, 0.98, 1.0);
                        mat.metallic = 0.22;
                        mat.perceptual_roughness = 0.45;
                        mat.reflectance = 0.50;
                    }
                    mat.alpha_mode = AlphaMode::Opaque;
                    mat.cull_mode = None;
                    mat.double_sided = true;
                }
            } else {
                mat.base_color = Color::srgb(0.22, 0.52, 0.78);
                mat.metallic = 0.22;
                mat.perceptual_roughness = 0.45;
                mat.reflectance = 0.50;
                mat.alpha_mode = AlphaMode::Opaque;
                mat.cull_mode = None;
                mat.double_sided = true;
            }
        }
    }
}

/// Handles viewport mouse interactions:
/// - Raycast object selection
/// - Dragging selected object on the XY bed (`ToolMode::MoveOnBed`)
/// - Interactive object rotation (`ToolMode::Rotate`)
/// - Hovering and clicking a $\le 27$-face convex hull facet (`ToolMode::PlaceOnFace`)
/// - Hovering and toggling surface-face hints (`ToolMode::SurfaceHintOrthogonal`, `ToolMode::SurfaceHintTangent`)
/// - Placing and interactively aiming 3D point normal hints (`ToolMode::PointHint`)
/// - Camera orbit / pan / zoom when not dragging an object or gizmo
#[allow(clippy::too_many_arguments)]
pub fn viewport_interaction_system(
    mut contexts: EguiContexts,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
    vis_state: Res<crate::field_vis::FieldVisualizationState>,
    mut camera_query: Query<(&Camera, &GlobalTransform, &mut OrbitCamera, &mut Transform), Without<CadObjectEntity>>,
    mut object_transforms: Query<&mut Transform, With<CadObjectEntity>>,
) {
    let Some(ctx) = contexts.try_ctx_mut() else {
        return;
    };
    if ctx.wants_pointer_input() || ctx.is_pointer_over_area() {
        mouse_motion.clear();
        mouse_wheel.clear();
        plate.hovered_hull_face = None;
        plate.hovered_surface_patch = None;
        plate.hovered_point_preview = None;
        if mouse_buttons.just_released(MouseButton::Left) {
            plate.is_dragging_object = false;
            plate.is_dragging_point_hint_dir = false;
        }
        return;
    }

    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok((camera, cam_global, mut orbit, mut cam_transform)) = camera_query.get_single_mut() else {
        return;
    };

    let cursor_ray = window
        .cursor_position()
        .and_then(|pos| camera.viewport_to_world(cam_global, pos));

    // 1. Update hover states for `PlaceOnFace`, `SurfaceHint*`, and `PointHint`
    plate.hovered_hull_face = None;
    plate.hovered_surface_patch = None;
    plate.hovered_point_preview = None;

    if let (Some(ray), Some(sel_id)) = (cursor_ray, plate.selected_object_id) {
        if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
            if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                match plate.tool_mode {
                    ToolMode::PlaceOnFace => {
                        if let Some((face_idx, _)) = obj.hull.raycast(obj_tf, ray) {
                            plate.hovered_hull_face = Some(face_idx);
                        }
                    }
                    ToolMode::SurfaceHintOrthogonal | ToolMode::SurfaceHintTangent => {
                        if let Some((patch_id, _, _, _)) = obj.raycast_surface_patch(obj_tf, ray) {
                            plate.hovered_surface_patch = Some(patch_id);
                        }
                    }
                    ToolMode::PointHint => {
                        if !plate.is_dragging_point_hint_dir {
                            if let Some((local_pt, local_dir)) =
                                compute_point_hint_placement(obj, obj_tf, &vis_state, ray)
                            {
                                let world_pt = obj_tf.transform_point(local_pt);
                                let world_dir =
                                    (obj_tf.rotation * local_dir).normalize_or_zero();
                                plate.hovered_point_preview = Some((world_pt, world_dir));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // 2. Handle Left Mouse Button press
    if mouse_buttons.just_pressed(MouseButton::Left) {
        match plate.tool_mode {
            ToolMode::PlaceOnFace => {
                if let (Some(face_idx), Some(sel_id)) =
                    (plate.hovered_hull_face, plate.selected_object_id)
                {
                    if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                        if let Some(face) = obj.hull.faces.get(face_idx) {
                            let face_normal = face.normal;
                            let face_area = face.area;
                            let obj_name = obj.name.clone();
                            if let Ok(mut obj_tf) = object_transforms.get_mut(obj.entity) {
                                obj_tf.rotation =
                                    rotation_to_place_face_on_bed(obj_tf.rotation, face_normal);
                                snap_object_to_bed(obj, &mut obj_tf);
                                plate.status_message = format!(
                                    "Placed '{}' on hull face #{} (area {:.1} mm²).",
                                    obj_name,
                                    face_idx + 1,
                                    face_area
                                );
                                mouse_motion.clear();
                                return;
                            }
                        }
                    }
                }
            }
            ToolMode::SurfaceHintOrthogonal | ToolMode::SurfaceHintTangent => {
                if let (Some(patch_id), Some(sel_id)) =
                    (plate.hovered_surface_patch, plate.selected_object_id)
                {
                    let target_mode = if plate.tool_mode == ToolMode::SurfaceHintOrthogonal {
                        SurfaceHintMode::NormalOrthogonalToSurface
                    } else {
                        SurfaceHintMode::NormalTangentToSurface
                    };
                    let depth_mm = config.eikonal.default_surface_hint_depth_mm;
                    let weight = config.eikonal.default_hint_weight;

                    if let Some(existing_pos) = config
                        .eikonal
                        .surface_hints
                        .iter()
                        .position(|h| h.object_id == sel_id && h.patch_id == patch_id)
                    {
                        if config.eikonal.surface_hints[existing_pos].mode == target_mode {
                            config.eikonal.surface_hints.remove(existing_pos);
                            if plate.selected_surface_hint_patch == Some(patch_id) {
                                plate.selected_surface_hint_patch = None;
                            }
                            plate.status_message = format!(
                                "Removed surface orientation hint on patch #{patch_id}."
                            );
                        } else {
                            config.eikonal.surface_hints[existing_pos].mode = target_mode;
                            plate.selected_surface_hint_patch = Some(patch_id);
                            plate.status_message = format!(
                                "Switched patch #{patch_id} hint to '{}'.",
                                target_mode.label()
                            );
                        }
                    } else {
                        config.eikonal.surface_hints.push(SurfaceFaceHint {
                            object_id: sel_id,
                            patch_id,
                            mode: target_mode,
                            influence_depth_mm: depth_mm,
                            weight,
                        });
                        plate.selected_surface_hint_patch = Some(patch_id);
                        plate.status_message = format!(
                            "Added '{}' hint on patch #{patch_id} (depth {:.1} mm).",
                            target_mode.label(),
                            depth_mm
                        );
                    }
                    mouse_motion.clear();
                    return;
                }
            }
            ToolMode::PointHint => {
                if let (Some(ray), Some(sel_id)) = (cursor_ray, plate.selected_object_id) {
                    if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                        if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                            // 2a. First check if the user clicked near an existing 3D point hint's base sphere
                            // (or arrow tip if that point hint is currently selected and its arrow is visible)
                            const ARROW_LEN_MM: f32 = 16.0;
                            let mut hit_existing_id: Option<u32> = None;
                            let mut best_ray_dist = 4.5_f32;
                            for h in config
                                .eikonal
                                .point_hints
                                .iter()
                                .filter(|h| h.object_id == sel_id)
                            {
                                let w_base = obj_tf.transform_point(Vec3::from(h.local_position));
                                let d_base = ray_point_distance(ray, w_base);
                                let d = if plate.selected_point_hint_id == Some(h.id) {
                                    let w_dir = (obj_tf.rotation * Vec3::from(h.local_direction))
                                        .normalize_or_zero();
                                    let w_tip = w_base + w_dir * ARROW_LEN_MM;
                                    d_base.min(ray_point_distance(ray, w_tip))
                                } else {
                                    d_base
                                };
                                if d < best_ray_dist {
                                    best_ray_dist = d;
                                    hit_existing_id = Some(h.id);
                                }
                            }

                            if let Some(hid) = hit_existing_id {
                                plate.selected_point_hint_id = Some(hid);
                                plate.is_dragging_point_hint_dir = true;
                                plate.status_message = format!(
                                    "Selected 3D point hint #{hid} — drag in viewport to aim normal direction."
                                );
                                mouse_motion.clear();
                                return;
                            }

                            // 2b. Otherwise, place a new 3D point hint on the active 2D slice plane or object surface
                            if let Some((local_pt, local_dir)) =
                                compute_point_hint_placement(obj, obj_tf, &vis_state, ray)
                            {
                                let next_id = config
                                    .eikonal
                                    .point_hints
                                    .iter()
                                    .map(|h| h.id)
                                    .max()
                                    .unwrap_or(0)
                                    + 1;
                                let radius = config.eikonal.default_point_hint_radius_mm;
                                let weight = config.eikonal.default_hint_weight;
                                config.eikonal.point_hints.push(PointNormalHint {
                                    id: next_id,
                                    object_id: sel_id,
                                    local_position: local_pt.to_array(),
                                    local_direction: local_dir.to_array(),
                                    influence_radius_mm: radius,
                                    weight,
                                });
                                plate.selected_point_hint_id = Some(next_id);
                                plate.is_dragging_point_hint_dir = true;
                                plate.status_message = format!(
                                    "Placed 3D point hint #{next_id} at ({:.1}, {:.1}, {:.1}) — drag to aim normal vector.",
                                    local_pt.x, local_pt.y, local_pt.z
                                );
                                mouse_motion.clear();
                                return;
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        // Raycast against all visible objects to select the closest hit
        if let Some(ray) = cursor_ray {
            let mut closest_hit: Option<(u64, f32)> = None;
            for obj in &plate.objects {
                if !obj.visible {
                    continue;
                }
                if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                    if let Some(dist) = obj.raycast_mesh(obj_tf, ray) {
                        if closest_hit.map_or(true, |(_, best_d)| dist < best_d) {
                            closest_hit = Some((obj.object_id, dist));
                        }
                    }
                }
            }

            if let Some((hit_id, _)) = closest_hit {
                plate.selected_object_id = Some(hit_id);
                if plate.tool_mode == ToolMode::MoveOnBed {
                    if let Some(plane_pt) = intersect_ray_z_plane(ray, 0.0) {
                        if let Some(obj) = plate.objects.iter().find(|o| o.object_id == hit_id) {
                            if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                                plate.is_dragging_object = true;
                                plate.drag_offset_xy = obj_tf.translation.xy() - plane_pt.xy();
                            }
                        }
                    }
                } else if plate.tool_mode == ToolMode::Rotate {
                    plate.is_dragging_object = true;
                }
            }
        }
    }

    if mouse_buttons.just_released(MouseButton::Left) {
        plate.is_dragging_object = false;
        plate.is_dragging_point_hint_dir = false;
    }

    let mut delta_drag = Vec2::ZERO;
    for ev in mouse_motion.read() {
        delta_drag += ev.delta;
    }

    let mut scroll = 0.0_f32;
    for ev in mouse_wheel.read() {
        scroll += ev.y;
    }

    // 3a. Interactive aiming of the selected 3D point hint's normal direction vector
    if plate.is_dragging_point_hint_dir && mouse_buttons.pressed(MouseButton::Left) {
        if delta_drag != Vec2::ZERO {
            if let (Some(sel_id), Some(hid)) =
                (plate.selected_object_id, plate.selected_point_hint_id)
            {
                if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                    if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                        if let Some(hint) = config
                            .eikonal
                            .point_hints
                            .iter_mut()
                            .find(|h| h.object_id == sel_id && h.id == hid)
                        {
                            // First try aiming the 3D arrow directly toward the cursor ray's projection
                            // on the camera-parallel plane passing through the hint's world position
                            let w_base = obj_tf.transform_point(Vec3::from(hint.local_position));
                            let cam_fwd = *cam_transform.forward();
                            let mut updated = false;
                            if let Some(ray) = cursor_ray {
                                if let Some(plane_hit) =
                                    intersect_ray_plane(ray, w_base, cam_fwd)
                                {
                                    let offset = plane_hit - w_base;
                                    if offset.length() > 1.5 {
                                        let new_world_dir = offset.normalize();
                                        let new_local_dir = (obj_tf.rotation.inverse()
                                            * new_world_dir)
                                            .normalize_or_zero();
                                        if new_local_dir.length_squared() > 0.5 {
                                            hint.local_direction = new_local_dir.to_array();
                                            updated = true;
                                        }
                                    }
                                }
                            }
                            if !updated {
                                let right = *cam_transform.right();
                                let up = *cam_transform.up();
                                let w_dir = (obj_tf.rotation * Vec3::from(hint.local_direction))
                                    .normalize_or_zero();
                                let rot_yaw = Quat::from_axis_angle(up, delta_drag.x * 0.012);
                                let rot_pitch = Quat::from_axis_angle(right, -delta_drag.y * 0.012);
                                let new_world_dir = (rot_yaw * rot_pitch * w_dir).normalize_or_zero();
                                let new_local_dir =
                                    (obj_tf.rotation.inverse() * new_world_dir).normalize_or_zero();
                                if new_local_dir.length_squared() > 0.5 {
                                    hint.local_direction = new_local_dir.to_array();
                                }
                            }
                        }
                    }
                }
            }
        }
        return;
    }

    // 3b. Active object manipulation when dragging in MoveOnBed or Rotate mode
    if plate.is_dragging_object && mouse_buttons.pressed(MouseButton::Left) {
        if let Some(sel_id) = plate.selected_object_id {
            match plate.tool_mode {
                ToolMode::MoveOnBed => {
                    if let Some(ray) = cursor_ray {
                        if let Some(plane_pt) = intersect_ray_z_plane(ray, 0.0) {
                            let target_xy = plane_pt.xy() + plate.drag_offset_xy;
                            if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id)
                            {
                                if let Ok(mut obj_tf) = object_transforms.get_mut(obj.entity) {
                                    obj_tf.translation.x = target_xy.x;
                                    obj_tf.translation.y = target_xy.y;
                                    snap_object_to_bed(obj, &mut obj_tf);
                                }
                            }
                        }
                    }
                    return;
                }
                ToolMode::Rotate => {
                    if delta_drag != Vec2::ZERO {
                        if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                            if let Ok(mut obj_tf) = object_transforms.get_mut(obj.entity) {
                                let yaw_rot = Quat::from_rotation_z(-delta_drag.x * 0.01);
                                let pitch_rot = Quat::from_rotation_x(delta_drag.y * 0.01);
                                obj_tf.rotation = (yaw_rot * obj_tf.rotation * pitch_rot).normalize();
                                snap_object_to_bed(obj, &mut obj_tf);
                            }
                        }
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    // 4. Camera orbit / pan / zoom
    if delta_drag == Vec2::ZERO && scroll == 0.0 {
        return;
    }

    if mouse_buttons.pressed(MouseButton::Left) {
        orbit.yaw -= delta_drag.x * 0.008;
        orbit.pitch = (orbit.pitch + delta_drag.y * 0.008).clamp(-1.50, 1.50);
    } else if mouse_buttons.pressed(MouseButton::Right)
        || mouse_buttons.pressed(MouseButton::Middle)
    {
        let right = cam_transform.right();
        let up = cam_transform.up();
        let pan_scale = orbit.radius * 0.0015;
        orbit.focus += (-right * delta_drag.x + up * delta_drag.y) * pan_scale;
    }

    if scroll != 0.0 {
        let zoom_factor = (1.0 - scroll * 0.1).clamp(0.5, 2.0);
        orbit.radius = (orbit.radius * zoom_factor).clamp(5.0, 50_000.0);
    }

    let eye = compute_orbit_eye(&orbit);
    *cam_transform = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z);
}

/// Computes the local-space `(position, initial_normal)` for placing a 3D point hint:
/// 1. If the 2D cross-section slice plane is active and the cursor ray intersects it inside
///    the object's local bounding box, places the point directly on the 2D slice plane.
/// 2. Otherwise, raycasts the object's surface mesh and places the point slightly inside
///    the surface (`1.5 mm` inward along `-normal`).
fn compute_point_hint_placement(
    obj: &SceneObject,
    obj_tf: &Transform,
    vis_state: &crate::field_vis::FieldVisualizationState,
    ray: Ray3d,
) -> Option<(Vec3, Vec3)> {
    let inv_mat = obj_tf.compute_matrix().inverse();
    let local_origin = inv_mat.transform_point3(ray.origin);
    let local_dir = inv_mat.transform_vector3(*ray.direction).normalize_or_zero();
    if local_dir.length_squared() < 1e-6 {
        return None;
    }

    // Priority 1: Active 2D Cross-Section Slice Plane
    if vis_state.show_slice_plane {
        let t = vis_state.slice_normalized_pos.clamp(0.0, 1.0);
        let (plane_normal, plane_coord) = match vis_state.slice_axis {
            SliceAxis::X => (Vec3::X, obj.local_min.x + t * (obj.local_max.x - obj.local_min.x)),
            SliceAxis::Y => (Vec3::Y, obj.local_min.y + t * (obj.local_max.y - obj.local_min.y)),
            SliceAxis::Z => (Vec3::Z, obj.local_min.z + t * (obj.local_max.z - obj.local_min.z)),
        };
        let denom = local_dir.dot(plane_normal);
        if denom.abs() > 1e-5 {
            let dist = (plane_coord - local_origin.dot(plane_normal)) / denom;
            if dist > 0.0 {
                let hit = local_origin + local_dir * dist;
                let margin = 2.0_f32;
                if hit.x >= obj.local_min.x - margin
                    && hit.x <= obj.local_max.x + margin
                    && hit.y >= obj.local_min.y - margin
                    && hit.y <= obj.local_max.y + margin
                    && hit.z >= obj.local_min.z - margin
                    && hit.z <= obj.local_max.z + margin
                {
                    let up_local = (obj_tf.rotation.inverse() * Vec3::Z).normalize_or_zero();
                    let default_dir = if up_local.length_squared() > 0.5 {
                        up_local
                    } else {
                        Vec3::Z
                    };
                    return Some((hit, default_dir));
                }
            }
        }
    }

    // Priority 2: Object surface mesh hit (inset slightly into the interior)
    if let Some((_, local_hit, local_normal, _)) = obj.raycast_surface_patch(obj_tf, ray) {
        let inset_pt = local_hit - local_normal * 1.5;
        let up_local = (obj_tf.rotation.inverse() * Vec3::Z).normalize_or_zero();
        let dir = if local_normal.dot(up_local) >= -0.1 {
            local_normal
        } else {
            -local_normal
        };
        return Some((inset_pt, dir.normalize_or_zero()));
    }

    None
}

pub fn compute_orbit_eye(orbit: &OrbitCamera) -> Vec3 {
    let cos_pitch = orbit.pitch.cos();
    let offset = Vec3::new(
        orbit.radius * cos_pitch * orbit.yaw.cos(),
        orbit.radius * cos_pitch * orbit.yaw.sin(),
        orbit.radius * orbit.pitch.sin(),
    );
    orbit.focus + offset
}

/// Draws the configured print bed grid, build volume box, selection bounding box,
/// $\le 27$-face simplified convex hull overlay (`PlaceOnFace`), selectable B-Rep / planar
/// mesh surface patches (`SurfaceHint*`), and 3D point normal hint gizmos (`PointHint`).
pub fn draw_viewport_gizmos_system(
    mut gizmos: Gizmos,
    config: Res<SlicerConfig>,
    plate: Res<BuildPlateState>,
    object_transforms: Query<&Transform, With<CadObjectEntity>>,
) {
    let bed = &config.bed;
    let grid_color = Color::srgba(0.35, 0.38, 0.44, 0.45);
    let border_color = Color::srgba(0.55, 0.62, 0.72, 0.85);
    let volume_color = Color::srgba(0.38, 0.44, 0.54, 0.25);

    match bed.shape {
        BedShape::Rectangular => {
            let hx = 0.5 * bed.width_x;
            let hy = 0.5 * bed.depth_y;
            let step = bed.grid_spacing.max(5.0);

            let mut x = -hx;
            while x <= hx + 1e-3 {
                gizmos.line(Vec3::new(x, -hy, 0.0), Vec3::new(x, hy, 0.0), grid_color);
                x += step;
            }
            let mut y = -hy;
            while y <= hy + 1e-3 {
                gizmos.line(Vec3::new(-hx, y, 0.0), Vec3::new(hx, y, 0.0), grid_color);
                y += step;
            }

            // Outer bed rectangle
            gizmos.linestrip(
                [
                    Vec3::new(-hx, -hy, 0.0),
                    Vec3::new(hx, -hy, 0.0),
                    Vec3::new(hx, hy, 0.0),
                    Vec3::new(-hx, hy, 0.0),
                    Vec3::new(-hx, -hy, 0.0),
                ],
                border_color,
            );

            if bed.show_volume_box {
                let hz = bed.height_z;
                gizmos.linestrip(
                    [
                        Vec3::new(-hx, -hy, hz),
                        Vec3::new(hx, -hy, hz),
                        Vec3::new(hx, hy, hz),
                        Vec3::new(-hx, hy, hz),
                        Vec3::new(-hx, -hy, hz),
                    ],
                    volume_color,
                );
                for &(cx, cy) in &[(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)] {
                    gizmos.line(
                        Vec3::new(cx, cy, 0.0),
                        Vec3::new(cx, cy, hz),
                        volume_color,
                    );
                }
            }
        }
        BedShape::Circular => {
            let radius = 0.5 * bed.width_x;
            let segments = 64;
            let mut ring_pts = Vec::with_capacity(segments + 1);
            for i in 0..=segments {
                let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
                ring_pts.push(Vec3::new(radius * angle.cos(), radius * angle.sin(), 0.0));
            }
            gizmos.linestrip(ring_pts.clone(), border_color);

            if bed.show_volume_box {
                for pt in &mut ring_pts {
                    pt.z = bed.height_z;
                }
                gizmos.linestrip(ring_pts, volume_color);
            }

            let step = bed.grid_spacing.max(5.0);
            let mut x = -radius;
            while x <= radius {
                let y_span = (radius * radius - x * x).max(0.0).sqrt();
                gizmos.line(
                    Vec3::new(x, -y_span, 0.0),
                    Vec3::new(x, y_span, 0.0),
                    grid_color,
                );
                gizmos.line(
                    Vec3::new(-y_span, x, 0.0),
                    Vec3::new(y_span, x, 0.0),
                    grid_color,
                );
                x += step;
            }
        }
    }

    // Coordinate origin axes placed at the printer's G-code origin:
    // - Front-left corner `(-width_x/2, -depth_y/2, 0)` for rectangular beds with corner origin
    // - Bed center `(0, 0, 0)` for circular beds or when `gcode_origin_at_bed_center` is enabled
    let origin_pos = config.gcode_origin_scene_pos();
    gizmos.line(
        origin_pos,
        origin_pos + Vec3::new(25.0, 0.0, 0.0),
        Color::srgb(0.95, 0.25, 0.25),
    );
    gizmos.line(
        origin_pos,
        origin_pos + Vec3::new(0.0, 25.0, 0.0),
        Color::srgb(0.25, 0.90, 0.35),
    );
    gizmos.line(
        origin_pos,
        origin_pos + Vec3::new(0.0, 0.0, 25.0),
        Color::srgb(0.30, 0.55, 0.98),
    );

    // When the printer's G-code origin is at the front-left corner, draw a subtle reticle at the
    // bed center `(0, 0, 0)` so the default object location / auto-arrangement origin remains clear.
    if origin_pos.length_squared() > 1.0e-3 {
        let center_mark_color = Color::srgba(0.62, 0.70, 0.82, 0.65);
        gizmos.line(
            Vec3::new(-7.0, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 0.0),
            center_mark_color,
        );
        gizmos.line(
            Vec3::new(0.0, -7.0, 0.0),
            Vec3::new(0.0, 7.0, 0.0),
            center_mark_color,
        );
    }

    // Optional 3D visualization of the queried Nozzle Local Frame, active tool XZ clearance
    // profile, and Gravity direction vector in bed coordinates
    if config.machine.show_nozzle_clearance_gizmo {
        let tip_bed = if let Some(sel_id) = plate.selected_object_id {
            if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                    let (min_p, max_p) = obj.compute_world_aabb(obj_tf);
                    Vec3::new(0.5 * (min_p.x + max_p.x), 0.5 * (min_p.y + max_p.y), max_p.z + 2.0)
                } else {
                    Vec3::new(0.0, 0.0, 35.0)
                }
            } else {
                Vec3::new(0.0, 0.0, 35.0)
            }
        } else {
            Vec3::new(0.0, 0.0, 35.0)
        };

        let frame = config.query_nozzle_frame(tip_bed, Vec3::X);
        let gravity = config.query_gravity_bed(tip_bed);

        // Draw Nozzle Local Orthonormal Basis (X_travel = Red, Y_lateral = Green, Z_nozzle = Cyan)
        gizmos.arrow(
            tip_bed,
            tip_bed + frame.travel_axis * 24.0,
            Color::srgb(1.0, 0.30, 0.25),
        );
        gizmos.arrow(
            tip_bed,
            tip_bed + frame.lateral_axis * 24.0,
            Color::srgb(0.25, 0.95, 0.40),
        );
        gizmos.arrow(
            tip_bed,
            tip_bed + frame.nozzle_axis * 32.0,
            Color::srgb(0.20, 0.85, 1.0),
        );

        // Draw Gravity Vector in Bed Coordinates (Magenta arrow)
        gizmos.arrow(
            tip_bed,
            tip_bed + gravity * 28.0,
            Color::srgb(0.98, 0.32, 0.88),
        );

        // Draw the active tool's revolved XZ clearance profile in the nozzle's local frame
        if let Some(tool) = config.active_tool() {
            let profile_color = Color::srgba(1.0, 0.72, 0.18, 0.85);
            let ring_segments = 24usize;
            for pt in &tool.clearance_profile_xz {
                let mut ring = Vec::with_capacity(ring_segments + 1);
                for k in 0..=ring_segments {
                    let theta = (k as f32 / ring_segments as f32) * std::f32::consts::TAU;
                    let local_p = Vec3::new(
                        pt.x_radius_mm * theta.cos(),
                        pt.x_radius_mm * theta.sin(),
                        pt.z_height_mm,
                    );
                    ring.push(frame.nozzle_local_to_bed(tip_bed, local_p));
                }
                gizmos.linestrip(ring, profile_color);
            }
            for dir_2d in [
                Vec2::new(1.0, 0.0),
                Vec2::new(-1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(0.0, -1.0),
            ] {
                let meridian: Vec<Vec3> = tool
                    .clearance_profile_xz
                    .iter()
                    .map(|pt| {
                        frame.nozzle_local_to_bed(
                            tip_bed,
                            Vec3::new(
                                dir_2d.x * pt.x_radius_mm,
                                dir_2d.y * pt.x_radius_mm,
                                pt.z_height_mm,
                            ),
                        )
                    })
                    .collect();
                if meridian.len() >= 2 {
                    gizmos.linestrip(meridian, profile_color);
                }
            }
        }
    }

    // Selected object gizmos & overlays
    let Some(sel_id) = plate.selected_object_id else {
        return;
    };
    let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) else {
        return;
    };
    let Ok(obj_tf) = object_transforms.get(obj.entity) else {
        return;
    };

    if plate.tool_mode == ToolMode::PlaceOnFace {
        // Render all <= 27 simplified convex hull faces, highlighting the hovered face
        for face in &obj.hull.faces {
            if face.polygon.len() < 3 {
                continue;
            }
            let is_hovered = plate.hovered_hull_face == Some(face.face_index);
            let edge_color = if is_hovered {
                Color::srgb(1.0, 0.82, 0.15)
            } else {
                Color::srgba(0.20, 0.92, 0.78, 0.70)
            };

            let mut world_loop: Vec<Vec3> = face
                .polygon
                .iter()
                .map(|&p| obj_tf.transform_point(p))
                .collect();
            world_loop.push(world_loop[0]);
            gizmos.linestrip(world_loop, edge_color);

            if is_hovered {
                let world_centroid = obj_tf.transform_point(face.centroid);
                let world_normal = (obj_tf.rotation * face.normal).normalize_or_zero();
                gizmos.arrow(
                    world_centroid,
                    world_centroid + world_normal * 18.0,
                    Color::srgb(1.0, 0.45, 0.10),
                );
            }
        }
    } else {
        // Draw world-space AABB selection wireframe around the active object
        let (min_p, max_p) = obj.compute_world_aabb(obj_tf);
        let sel_color = Color::srgba(0.98, 0.78, 0.22, 0.75);
        draw_aabb_wireframe(&mut gizmos, min_p, max_p, sel_color);
    }

    // --- Render Active & Hovered Surface-Face Orientation Hints ---
    for sh in config
        .eikonal
        .surface_hints
        .iter()
        .filter(|h| h.object_id == sel_id)
    {
        if let Some(patch) = obj.surface_patches.get(sh.patch_id as usize) {
            let is_selected = plate.selected_surface_hint_patch == Some(sh.patch_id);
            let is_ui_hovered = plate.ui_hovered_surface_hint_patch == Some(sh.patch_id);
            let is_vp_hovered = matches!(
                plate.tool_mode,
                ToolMode::SurfaceHintOrthogonal | ToolMode::SurfaceHintTangent
            ) && plate.hovered_surface_patch == Some(sh.patch_id);
            let show_arrows = is_selected || is_ui_hovered || is_vp_hovered;

            let color = if is_ui_hovered {
                Color::srgb(1.0, 0.92, 0.20)
            } else if is_selected {
                match sh.mode {
                    SurfaceHintMode::NormalOrthogonalToSurface => Color::srgb(0.15, 0.95, 0.98),
                    SurfaceHintMode::NormalTangentToSurface => Color::srgb(1.0, 0.42, 0.85),
                }
            } else {
                match sh.mode {
                    SurfaceHintMode::NormalOrthogonalToSurface => {
                        Color::srgba(0.15, 0.95, 0.98, 0.55)
                    }
                    SurfaceHintMode::NormalTangentToSurface => Color::srgba(1.0, 0.42, 0.85, 0.55),
                }
            };

            draw_surface_patch_gizmo(
                &mut gizmos,
                patch,
                obj_tf,
                color,
                if show_arrows { Some(sh.mode) } else { None },
            );
            if is_ui_hovered {
                let w_center = obj_tf.transform_point(patch.centroid);
                gizmos.sphere(w_center, Quat::IDENTITY, 1.8, color);
            }
        }
    }

    if matches!(
        plate.tool_mode,
        ToolMode::SurfaceHintOrthogonal | ToolMode::SurfaceHintTangent
    ) {
        if let Some(hovered_pid) = plate.hovered_surface_patch {
            if let Some(patch) = obj.surface_patches.get(hovered_pid as usize) {
                let preview_mode = if plate.tool_mode == ToolMode::SurfaceHintOrthogonal {
                    SurfaceHintMode::NormalOrthogonalToSurface
                } else {
                    SurfaceHintMode::NormalTangentToSurface
                };
                draw_surface_patch_gizmo(
                    &mut gizmos,
                    patch,
                    obj_tf,
                    Color::srgb(1.0, 0.88, 0.18),
                    Some(preview_mode),
                );
            }
        }
    }

    // --- Render Active & Preview 3D Point Normal Hints ---
    const ARROW_LEN_MM: f32 = 16.0;
    for ph in config
        .eikonal
        .point_hints
        .iter()
        .filter(|h| h.object_id == sel_id)
    {
        let is_selected = plate.selected_point_hint_id == Some(ph.id);
        let is_ui_hovered = plate.ui_hovered_point_hint_id == Some(ph.id);
        let show_arrow = is_selected || is_ui_hovered;

        let w_base = obj_tf.transform_point(Vec3::from(ph.local_position));
        if !show_arrow {
            // Suppress arrow and tip handle for non-selected/non-hovered point hints to prevent clutter
            gizmos.sphere(
                w_base,
                Quat::IDENTITY,
                1.3,
                Color::srgba(0.25, 0.98, 0.55, 0.65),
            );
            continue;
        }

        let w_dir = (obj_tf.rotation * Vec3::from(ph.local_direction)).normalize_or_zero();
        let w_tip = w_base + w_dir * ARROW_LEN_MM;

        let arrow_color = if is_ui_hovered {
            Color::srgb(1.0, 0.95, 0.25)
        } else {
            Color::srgb(1.0, 0.86, 0.15)
        };
        let base_radius = if is_ui_hovered { 2.0 } else { 1.6 };
        gizmos.sphere(w_base, Quat::IDENTITY, base_radius, arrow_color);
        gizmos.arrow(w_base, w_tip, arrow_color);
        gizmos.sphere(w_tip, Quat::IDENTITY, 1.3, Color::srgb(1.0, 0.48, 0.15));

        // Draw influence radius wireframe circle around the selected or hovered point hint
        let ring_color = if is_ui_hovered {
            Color::srgba(1.0, 0.95, 0.25, 0.70)
        } else {
            Color::srgba(0.25, 0.98, 0.55, 0.45)
        };
        let r = ph.influence_radius_mm.max(1.0);
        let up_ref = if w_dir.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
        let t1 = w_dir.cross(up_ref).normalize_or_zero();
        let t2 = w_dir.cross(t1).normalize_or_zero();
        let segs = 24usize;
        let mut ring = Vec::with_capacity(segs + 1);
        for k in 0..=segs {
            let a = (k as f32 / segs as f32) * std::f32::consts::TAU;
            ring.push(w_base + r * (a.cos() * t1 + a.sin() * t2));
        }
        gizmos.linestrip(ring, ring_color);
    }

    if plate.tool_mode == ToolMode::PointHint {
        if let Some((w_pos, w_dir)) = plate.hovered_point_preview {
            let preview_color = Color::srgba(1.0, 0.90, 0.25, 0.75);
            gizmos.sphere(w_pos, Quat::IDENTITY, 1.4, preview_color);
            gizmos.arrow(w_pos, w_pos + w_dir * ARROW_LEN_MM, preview_color);
        }
    }
}

fn draw_surface_patch_gizmo(
    gizmos: &mut Gizmos,
    patch: &SurfacePatch,
    obj_tf: &Transform,
    color: Color,
    mode: Option<SurfaceHintMode>,
) {
    for &[a, b] in &patch.boundary_edges {
        gizmos.line(obj_tf.transform_point(a), obj_tf.transform_point(b), color);
    }

    let w_center = obj_tf.transform_point(patch.centroid);
    let w_normal = (obj_tf.rotation * patch.normal).normalize_or_zero();
    match mode {
        Some(SurfaceHintMode::NormalOrthogonalToSurface) => {
            // Arrow along surface normal (\nabla \tau \parallel \pm n_face -> conformal layers)
            gizmos.arrow(w_center, w_center + w_normal * 14.0, color);
        }
        Some(SurfaceHintMode::NormalTangentToSurface) => {
            // Tangent cross-vectors on the surface (\nabla \tau \perp n_face -> layers meet face at 90°)
            let up = if w_normal.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
            let t1 = w_normal.cross(up).normalize_or_zero();
            let t2 = w_normal.cross(t1).normalize_or_zero();
            gizmos.arrow(w_center - t1 * 8.0, w_center + t1 * 8.0, color);
            gizmos.arrow(w_center - t2 * 8.0, w_center + t2 * 8.0, color);
        }
        None => {}
    }
}

fn draw_aabb_wireframe(gizmos: &mut Gizmos, min: Vec3, max: Vec3, color: Color) {
    let b0 = Vec3::new(min.x, min.y, min.z);
    let b1 = Vec3::new(max.x, min.y, min.z);
    let b2 = Vec3::new(max.x, max.y, min.z);
    let b3 = Vec3::new(min.x, max.y, min.z);

    let t0 = Vec3::new(min.x, min.y, max.z);
    let t1 = Vec3::new(max.x, min.y, max.z);
    let t2 = Vec3::new(max.x, max.y, max.z);
    let t3 = Vec3::new(min.x, max.y, max.z);

    gizmos.linestrip([b0, b1, b2, b3, b0], color);
    gizmos.linestrip([t0, t1, t2, t3, t0], color);
    gizmos.line(b0, t0, color);
    gizmos.line(b1, t1, color);
    gizmos.line(b2, t2, color);
    gizmos.line(b3, t3, color);
}

fn intersect_ray_z_plane(ray: Ray3d, z_plane: f32) -> Option<Vec3> {
    let dir_z = ray.direction.z;
    if dir_z.abs() < 1e-6 {
        return None;
    }
    let t = (z_plane - ray.origin.z) / dir_z;
    if t > 0.0 {
        Some(ray.origin + *ray.direction * t)
    } else {
        None
    }
}

fn intersect_ray_plane(ray: Ray3d, plane_origin: Vec3, plane_normal: Vec3) -> Option<Vec3> {
    let denom = ray.direction.dot(plane_normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_origin - ray.origin).dot(plane_normal) / denom;
    if t > 0.0 {
        Some(ray.origin + *ray.direction * t)
    } else {
        None
    }
}

fn ray_point_distance(ray: Ray3d, pt: Vec3) -> f32 {
    let v = pt - ray.origin;
    let proj = v.dot(*ray.direction).max(0.0);
    let closest = ray.origin + *ray.direction * proj;
    closest.distance(pt)
}

fn ray_intersects_aabb(ray: Ray3d, min: Vec3, max: Vec3) -> bool {
    let inv_dir = Vec3::new(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z,
    );
    let t1 = (min - ray.origin) * inv_dir;
    let t2 = (max - ray.origin) * inv_dir;
    let tmin = t1.min(t2).max_element();
    let tmax = t1.max(t2).min_element();
    tmax >= tmin.max(0.0)
}
