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
use crate::config::{BedShape, SlicerConfig};
use crate::hull::{
    ray_intersect_triangle, rotation_to_place_face_on_bed, SimplifiedConvexHull,
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
}

impl ToolMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select / Orbit",
            Self::MoveOnBed => "Move on Bed (XY)",
            Self::Rotate => "Rotate / Orient",
            Self::PlaceOnFace => "Place on Face (Hull)",
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
    pub material_handle: Handle<StandardMaterial>,
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub hull: SimplifiedConvexHull,
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

        let mut best_local_t: Option<f32> = None;
        for tri in self.indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];
            if let Some(t) = ray_intersect_triangle(
                local_origin,
                local_dir,
                Vec3::new(v0.x, v0.y, v0.z),
                Vec3::new(v1.x, v1.y, v1.z),
                Vec3::new(v2.x, v2.y, v2.z),
            ) {
                if best_local_t.map_or(true, |best| t < best) {
                    best_local_t = Some(t);
                }
            }
        }

        best_local_t.map(|t| {
            let local_hit = local_origin + local_dir * t;
            let world_hit = transform.transform_point(local_hit);
            ray.origin.distance(world_hit)
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
    pub is_dragging_object: bool,
    pub drag_offset_xy: Vec2,
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
        .with_inserted_indices(Indices::U32(data.indices.clone()));

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
                    mesh: meshes.add(bevy_mesh),
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
            material_handle: mat_handle,
            vertices: data.vertices,
            indices: data.indices,
            local_min: data.local_min,
            local_max: data.local_max,
            hull: data.hull,
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
        plate.status_message = format!("Removed object '{}' (ID {object_id}).", removed.name);
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

/// Updates object material highlights so the selected object is visually distinct.
pub fn update_selection_materials_system(
    plate: Res<BuildPlateState>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !plate.is_changed() {
        return;
    }

    for obj in &plate.objects {
        if let Some(mat) = materials.get_mut(&obj.material_handle) {
            let is_selected = plate.selected_object_id == Some(obj.object_id);
            mat.base_color = if is_selected {
                Color::srgb(0.32, 0.76, 0.98)
            } else {
                Color::srgb(0.22, 0.52, 0.78)
            };
        }
    }
}

/// Handles viewport mouse interactions:
/// - Raycast object selection
/// - Dragging selected object on the XY bed (`ToolMode::MoveOnBed`)
/// - Interactive object rotation (`ToolMode::Rotate`)
/// - Hovering and clicking a $\le 27$-face convex hull facet (`ToolMode::PlaceOnFace`)
/// - Camera orbit / pan / zoom when not dragging an object
#[allow(clippy::too_many_arguments)]
pub fn viewport_interaction_system(
    mut contexts: EguiContexts,
    windows: Query<&Window, With<PrimaryWindow>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut plate: ResMut<BuildPlateState>,
    mut camera_query: Query<(&Camera, &GlobalTransform, &mut OrbitCamera, &mut Transform), Without<CadObjectEntity>>,
    mut object_transforms: Query<&mut Transform, With<CadObjectEntity>>,
) {
    let ctx = contexts.ctx_mut();
    if ctx.wants_pointer_input() || ctx.is_pointer_over_area() {
        mouse_motion.clear();
        mouse_wheel.clear();
        plate.hovered_hull_face = None;
        if mouse_buttons.just_released(MouseButton::Left) {
            plate.is_dragging_object = false;
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

    // 1. Update hovered convex hull face when in `PlaceOnFace` mode
    plate.hovered_hull_face = None;
    if plate.tool_mode == ToolMode::PlaceOnFace {
        if let (Some(ray), Some(sel_id)) = (cursor_ray, plate.selected_object_id) {
            if let Some(obj) = plate.objects.iter().find(|o| o.object_id == sel_id) {
                if let Ok(obj_tf) = object_transforms.get(obj.entity) {
                    if let Some((face_idx, _)) = obj.hull.raycast(obj_tf, ray) {
                        plate.hovered_hull_face = Some(face_idx);
                    }
                }
            }
        }
    }

    // 2. Handle Left Mouse Button press: hull face placement, object picking, or starting bed drag
    if mouse_buttons.just_pressed(MouseButton::Left) {
        if plate.tool_mode == ToolMode::PlaceOnFace {
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
    }

    let mut delta_drag = Vec2::ZERO;
    for ev in mouse_motion.read() {
        delta_drag += ev.delta;
    }

    let mut scroll = 0.0_f32;
    for ev in mouse_wheel.read() {
        scroll += ev.y;
    }

    // 3. Active object manipulation when dragging in MoveOnBed or Rotate mode
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
/// and (when in `PlaceOnFace` mode) the $\le 27$-face simplified convex hull overlay.
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

    // Coordinate origin axes on the bed corner/center
    gizmos.line(Vec3::ZERO, Vec3::new(25.0, 0.0, 0.0), Color::srgb(0.95, 0.25, 0.25));
    gizmos.line(Vec3::ZERO, Vec3::new(0.0, 25.0, 0.0), Color::srgb(0.25, 0.90, 0.35));
    gizmos.line(Vec3::ZERO, Vec3::new(0.0, 0.0, 25.0), Color::srgb(0.30, 0.55, 0.98));

    // Selected object gizmos & <=27-face simplified convex hull overlay
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
