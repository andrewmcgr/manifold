//! `ManifoldApp`: GUI shell — left settings panel, central 3D viewport with
//! an in-panel import toolbar (Phase 4, see ROADMAP.md).

use crate::camera::OrbitCamera;
use crate::profile::Profile;
use crate::render::{
    MeshOverlayMode, MeshRenderResources, UploadedMesh, UploadedScene, UploadedToolpaths,
    Viewport3dCallback,
};
use crate::scene;
use crate::toolpath_view::{self, ToolpathDataView};
use eframe::egui;
use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::infill::InfillPatternKind;
use manifold_core::machine::Machine;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::tool::Tool;
use manifold_core::transform::Transform;
use manifold_core::{ids::ObjectId, ids::ToolId, mesh::Mesh, object, object::Object, stl, threemf};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use transform_gizmo_egui::math::Transform as GizmoTransform;
use transform_gizmo_egui::prelude::*;

/// Axis-aligned plane the SDF slice view samples over (subtask 09). Basis
/// vectors and the world-space origin for a given `offset` are derived in
/// [`ManifoldApp::recompute_slice`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlicePlane {
    Xy,
    Xz,
    Yz,
}

impl SlicePlane {
    /// `(basis1, basis2, normal)` for this plane, all orthonormal.
    fn basis(self) -> (glam::DVec3, glam::DVec3, glam::DVec3) {
        match self {
            SlicePlane::Xy => (glam::DVec3::X, glam::DVec3::Y, glam::DVec3::Z),
            SlicePlane::Xz => (glam::DVec3::X, glam::DVec3::Z, glam::DVec3::Y),
            SlicePlane::Yz => (glam::DVec3::Y, glam::DVec3::Z, glam::DVec3::X),
        }
    }
}

/// Message sent from the background slicing thread spawned by
/// `ManifoldApp::start_slice` back to the main/UI thread, polled once per
/// frame in `update()` via `ManifoldApp::drain_slice_messages`.
enum SliceMessage {
    /// `0.0..=1.0` fraction of how far through the order-field domain
    /// slicing currently is, from `manifold_core::plan_toolpaths_with_progress`.
    Progress(f64),
    /// The final result, converted to `String` so the message doesn't need
    /// to carry a non-`'static`-bound error type across the thread
    /// boundary.
    Done(Result<Vec<manifold_core::toolpath::Path>, String>),
}

pub struct ManifoldApp {
    config: manifold_core::SlicerConfig,
    /// The machine (bed/build-volume/tools) objects are sliced and
    /// centered against, editable from the settings panel (Phase 3, see
    /// ROADMAP.md).
    machine: Machine,
    objects: Vec<Object>,
    /// GPU-uploaded copies of `objects`, rebuilt whenever `objects` changes.
    uploaded_meshes: Arc<Vec<UploadedMesh>>,
    /// Scene dressing (origin axes, bed grid/quad, toolhead markers),
    /// rebuilt via `Self::build_scene` whenever `machine`'s bed/tool
    /// geometry changes in the settings panel (Phase 3/6, see ROADMAP.md).
    uploaded_scene: Arc<UploadedScene>,
    camera: OrbitCamera,
    next_object_id: u32,
    import_error: Option<String>,
    /// Index into `objects` of the currently selected object, if any —
    /// drives which object the move/rotate/scale gizmo manipulates
    /// (Phase 7, see ROADMAP.md).
    selected: Option<usize>,
    /// The move/rotate/scale gizmo, reused across frames so drag state
    /// persists between `interact()` calls.
    gizmo: Gizmo,
    /// 3D pivot point anchored on drag start for orbit rotation.
    drag_pivot: Option<DVec3>,
    /// Depth along the camera forward axis anchored on drag start for 1:1 pan.
    drag_depth: f64,
    /// Active drag interaction mode for the viewport canvas.
    active_drag: Option<DragMode>,
    /// Path to the currently loaded or saved profile file.
    profile_path: Option<std::path::PathBuf>,
    /// Whether "Lay on Face" facet inspection mode is active.
    lay_on_face_active: bool,
    /// Cached simplified convex hull for the currently selected object `(object_index, hull)`.
    cached_hull: Option<(usize, manifold_core::convex_hull::SimplifiedHull)>,
    /// Gcode from the last successful "Slice" action (Phase 8, see
    /// ROADMAP.md), previewed in the settings panel and written out by
    /// "Export…".
    gcode: Option<String>,
    /// Planned toolpaths from the last successful "Slice" action (Phase 13,
    /// see ROADMAP.md), previewed in the 3D viewport when `show_toolpaths`
    /// is enabled.
    toolpaths: Option<Vec<manifold_core::toolpath::Path>>,
    /// Selected data view color-coding mode for toolpath visualization (e.g. Line Type).
    toolpath_data_view: ToolpathDataView,
    /// GPU-uploaded copy of `toolpaths`, rebuilt whenever `toolpaths`
    /// changes, same pattern as `uploaded_meshes`/`uploaded_scene`.
    uploaded_toolpaths: Option<Arc<UploadedToolpaths>>,
    /// Whether the toolpath preview line geometry is drawn in the viewport
    /// (Phase 13, see ROADMAP.md).
    show_toolpaths: bool,
    /// Mesh visualization mode (None, Conformal Regions, or Surface Order Gradient).
    mesh_overlay_mode: MeshOverlayMode,
    /// Order-based scrub slider value (Phase 13 subtask 05): segments with
    /// `order <= scrub_order` are drawn, others hidden ("up to and
    /// including" semantics). `f64::INFINITY` (the default) shows every
    /// segment. Reset to the max order of the newly planned toolpaths each
    /// time `slice()` succeeds.
    scrub_order: f64,
    /// `(min, max)` order value across all segments in `toolpaths`, sizing
    /// the scrub slider's range — recomputed in `slice()` whenever
    /// `toolpaths` changes. `None` when `toolpaths` is `None` or contains no
    /// segments.
    toolpath_order_range: Option<(f64, f64)>,
    slice_error: Option<String>,
    /// Receiver for the background slicing thread spawned by `start_slice`,
    /// polled once per frame in `update()` via `drain_slice_messages`.
    /// `None` when no slice is currently in progress.
    slicing: Option<std::sync::mpsc::Receiver<SliceMessage>>,
    /// `JoinHandle` for the background slicing thread spawned by
    /// `start_slice`, kept so `drain_slice_messages` can `join()` it and
    /// recover the panic payload if the thread dies without ever sending
    /// `SliceMessage::Done` (e.g. an `unwrap()`/`expect()`/indexing panic
    /// somewhere in the slicing/toolpath pipeline) — otherwise a silently
    /// dropped `Sender` leaves `self.slicing` stuck `Some` forever and the
    /// UI spins indefinitely with no error surfaced. `None` when no slice
    /// is currently in progress.
    slicing_handle: Option<std::thread::JoinHandle<()>>,
    /// `0.0..=1.0` progress of the in-progress slice, reported by
    /// `manifold_core::plan_toolpaths_with_progress`. Only meaningful while
    /// `slicing` is `Some`.
    slice_progress: f64,
    /// Set by a failed "Save Profile…"/"Load Profile…" action (Phase 10, see
    /// ROADMAP.md).
    profile_error: Option<String>,
    next_tool_id: u32,
    /// Identifiers of line types currently toggled off/hidden in the viewport.
    hidden_line_types: std::collections::HashSet<crate::toolpath_view::LineTypeKey>,
    /// Summary statistics for the last successful slice (estimated time, filament volume/mass).
    print_statistics: Option<manifold_core::PrintStatistics>,
    /// Whether the SDF debug panel (Phase D, see MESH_SDF_VISUALIZATION.md)
    /// is shown as an additional right-hand side panel.
    show_sdf_panel: bool,
    /// Sign method the SDF panel will use when constructing a `MeshSdf`
    /// (subtask 08 wires the actual construction).
    sdf_sign_method: manifold_fidget::mesh_sdf::SignMethod,
    /// Iso-level (mm) the SDF panel will pass to isosurface extraction
    /// (subtask 08).
    sdf_iso_level: f64,
    /// Set by a failed SDF recompute/extraction action (subtask 08/09 will
    /// populate this; wired here so the display path exists already).
    sdf_error: Option<String>,
    /// Isosurface triangle soup from the last successful recompute
    /// (subtask 08 populates this; `None` until then).
    sdf_isosurface: Option<Vec<manifold_fidget::marching_cubes::Vertex>>,
    /// Slice heatmap grid from the last successful recompute (subtask 09
    /// populates this; `None` until then).
    sdf_slice: Option<manifold_fidget::slice::SliceGrid>,
    /// GPU-uploaded copy of `sdf_isosurface`, rebuilt whenever a recompute
    /// succeeds; rendered as a semi-transparent overlay alongside
    /// `uploaded_meshes` in `viewport()`. `None` until the first successful
    /// recompute.
    sdf_overlay_mesh: Option<Arc<UploadedMesh>>,
    /// Which axis-aligned plane the slice view samples over (subtask 09).
    sdf_slice_plane: SlicePlane,
    /// Offset (mm) along the plane's normal axis at which the slice is
    /// sampled (subtask 09).
    sdf_slice_offset: f64,
    /// Uploaded heatmap texture for the last `sdf_slice`, rebuilt whenever
    /// the slice is recomputed (recompute-on-demand only, never rebuilt
    /// per-frame; subtask 09).
    sdf_slice_texture: Option<egui::TextureHandle>,
    /// Commands from the Phase 9 MCP automation server, drained once per
    /// frame in `update()`. `None` if the `mcp-server` feature is off or
    /// the server thread failed to start.
    #[cfg(feature = "mcp-server")]
    mcp_rx: Option<std::sync::mpsc::Receiver<crate::mcp::Command>>,
}

impl ManifoldApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let wgpu_render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("manifold-gui requires the eframe wgpu renderer backend");

        wgpu_render_state
            .renderer
            .write()
            .callback_resources
            .insert(MeshRenderResources::new(
                &wgpu_render_state.device,
                wgpu_render_state.target_format,
            ));

        let machine = default_machine();
        let uploaded_scene = Arc::new(Self::build_scene(&wgpu_render_state.device, &machine));

        let mut camera = OrbitCamera::default();
        let (min, max) = machine.build_volume.bounding_box();
        camera.frame(min, max);

        #[cfg(feature = "mcp-server")]
        let mcp_rx = match crate::mcp::spawn(crate::mcp::ADDR) {
            Ok(rx) => Some(rx),
            Err(error) => {
                tracing::warn!(?error, "failed to start MCP automation server");
                None
            }
        };

        Self {
            config: manifold_core::SlicerConfig::default(),
            machine,
            objects: Vec::new(),
            uploaded_meshes: Arc::new(Vec::new()),
            uploaded_scene,
            camera,
            next_object_id: 0,
            import_error: None,
            selected: None,
            gizmo: Gizmo::default(),
            drag_pivot: None,
            drag_depth: 0.0,
            active_drag: None,
            profile_path: None,
            lay_on_face_active: false,
            cached_hull: None,
            gcode: None,
            toolpaths: None,
            toolpath_data_view: ToolpathDataView::default(),
            hidden_line_types: std::collections::HashSet::new(),
            uploaded_toolpaths: None,
            show_toolpaths: true,
            mesh_overlay_mode: MeshOverlayMode::default(),
            scrub_order: f64::INFINITY,
            toolpath_order_range: None,
            slice_error: None,
            slicing: None,
            slicing_handle: None,
            slice_progress: 0.0,
            profile_error: None,
            print_statistics: None,
            next_tool_id: 1,
            show_sdf_panel: false,
            sdf_sign_method: manifold_fidget::mesh_sdf::SignMethod::Pseudonormal,
            sdf_iso_level: 0.0,
            sdf_error: None,
            sdf_isosurface: None,
            sdf_slice: None,
            sdf_overlay_mesh: None,
            sdf_slice_plane: SlicePlane::Xy,
            sdf_slice_offset: 0.0,
            sdf_slice_texture: None,
            #[cfg(feature = "mcp-server")]
            mcp_rx,
        }
    }

    /// Drain pending automation commands from the MCP server thread and
    /// apply them against scene state, same as any other UI mutation.
    #[cfg(feature = "mcp-server")]
    fn drain_mcp_commands(&mut self, frame: &mut eframe::Frame) {
        let Some(rx) = &self.mcp_rx else { return };
        let commands: Vec<_> = rx.try_iter().collect();
        for command in commands {
            match command {
                crate::mcp::Command::SelectObject(index) => {
                    self.selected = if index < self.objects.len() {
                        Some(index)
                    } else {
                        None
                    };
                }
                crate::mcp::Command::RemoveObject(index) => {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.remove_object(index, &device);
                }
                crate::mcp::Command::ClearObjects => {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.clear_objects(&device);
                }
                crate::mcp::Command::SetTransform { index, x, y, z } => {
                    if let Some(object) = self.objects.get_mut(index) {
                        let (scale, rotation, _) =
                            object.transform.0.to_scale_rotation_translation();
                        object.transform = Transform::from_scale_rotation_translation(
                            scale,
                            rotation,
                            glam::DVec3::new(x, y, z),
                        );
                        self.update_camera_bounds();
                        let device = frame
                            .wgpu_render_state()
                            .expect("wgpu renderer is required")
                            .device
                            .clone();
                        self.reupload(&device);
                    }
                }
                crate::mcp::Command::ImportFile(path) => {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.import(&path, &device);
                }
                crate::mcp::Command::ListObjects(reply) => {
                    let json = serde_json::to_string(
                        &self
                            .objects
                            .iter()
                            .enumerate()
                            .map(|(index, object)| {
                                serde_json::json!({
                                    "index": index,
                                    "id": object.id.0,
                                    "triangle_count": object.mesh.triangle_count(),
                                })
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap_or_else(|_| "[]".to_string());
                    let _ = reply.send(json);
                }
                crate::mcp::Command::GetSelected(reply) => {
                    let _ = reply.send(self.selected);
                }
            }
        }
    }

    /// Build the scene dressing (origin axes, bed grid/quad, toolhead
    /// markers) for the given `machine` and upload it to the GPU. Called at
    /// startup and whenever `machine`'s bed/tool geometry changes in the
    /// settings panel.
    fn build_scene(device: &eframe::egui_wgpu::wgpu::Device, machine: &Machine) -> UploadedScene {
        let mut lines = scene::build_origin_axes(50.0);
        lines.extend(scene::build_grid(machine, 10.0));
        let mut triangles = scene::build_bed_quad(machine);
        triangles.extend(scene::build_toolhead_markers(machine, 8.0));
        UploadedScene::upload(device, &lines, &triangles)
    }

    /// Combined axis-aligned bounding box enclosing the machine build volume
    /// and every loaded object (transformed to world space).
    fn scene_bounding_box(&self) -> (DVec3, DVec3) {
        let (mut min, mut max) = self.machine.build_volume.bounding_box();
        for object in &self.objects {
            if let Some((local_min, local_max)) = object.mesh.bounding_box() {
                for corner in manifold_core::mesh::bounding_box_corners(local_min, local_max) {
                    let world = object.transform.transform_point(corner);
                    min = min.min(world);
                    max = max.max(world);
                }
            }
        }
        (min, max)
    }

    /// Update the camera's zoom limits ([`OrbitCamera::max_distance`]) so
    /// that everything in the scene occupies approximately 1/3 of the screen
    /// height at maximum zoom-out.
    fn update_camera_bounds(&mut self) {
        let (min, max) = self.scene_bounding_box();
        self.camera.update_max_distance(min, max);
    }

    /// Cast a ray from the camera through `cursor_pos` into the scene.
    ///
    /// Intersects against:
    /// 1. Loaded object meshes (closest forward triangle hit).
    /// 2. Print bed plane (within bed dimensions).
    /// 3. Invisible skybox sphere enclosing the scene, if no surface was hit.
    fn cast_scene_ray(&self, rect: egui::Rect, cursor_pos: egui::Pos2) -> SceneRayHit {
        let (ray_orig, ray_dir) = self.camera.unproject_ray(rect, cursor_pos);

        let mut closest_t = f64::INFINITY;
        let mut hit_surface = false;
        let mut hit_point = DVec3::ZERO;
        let mut hit_index = 0;

        // 1. Check loaded objects
        for (idx, object) in self.objects.iter().enumerate() {
            let Some((local_min, local_max)) = object.mesh.bounding_box() else {
                continue;
            };

            let inv = object.transform.0.inverse();
            let local_ray_orig = inv.transform_point3(ray_orig);
            let local_ray_dir = inv.transform_vector3(ray_dir);
            let local_dir_len = local_ray_dir.length();
            if local_dir_len < 1e-12 {
                continue;
            }
            let local_dir_norm = local_ray_dir / local_dir_len;

            // Fast AABB rejection
            if ray_aabb_intersect(local_ray_orig, local_dir_norm, local_min, local_max).is_none() {
                continue;
            }

            // Triangle intersection
            for chunk in object.mesh.indices.chunks_exact(3) {
                let v0 = object.mesh.vertices[chunk[0] as usize];
                let v1 = object.mesh.vertices[chunk[1] as usize];
                let v2 = object.mesh.vertices[chunk[2] as usize];

                if let Some(t_local) =
                    ray_triangle_intersect(local_ray_orig, local_dir_norm, v0, v1, v2)
                {
                    let t_world = t_local / local_dir_len;
                    if t_world > 1e-4 && t_world < closest_t {
                        closest_t = t_world;
                        hit_point = ray_orig + ray_dir * t_world;
                        hit_surface = true;
                        hit_index = idx;
                    }
                }
            }
        }

        if hit_surface {
            return SceneRayHit::Object {
                index: hit_index,
                point: hit_point,
            };
        }

        // 2. Check print bed
        let (bed_min, bed_max) = self.machine.build_volume.bounding_box();
        let bed_z = bed_min.z;
        if ray_dir.z.abs() > 1e-6 {
            let t_bed = (bed_z - ray_orig.z) / ray_dir.z;
            if t_bed > 1e-4 {
                let p = ray_orig + ray_dir * t_bed;
                if p.x >= bed_min.x && p.x <= bed_max.x && p.y >= bed_min.y && p.y <= bed_max.y {
                    return SceneRayHit::Bed(p);
                }
            }
        }

        // 3. Invisible skybox sphere enclosing the scene
        let (scene_min, scene_max) = self.scene_bounding_box();
        let scene_center = (scene_min + scene_max) * 0.5;
        let scene_radius = (scene_max - scene_min).length() * 0.5;
        let skybox_radius = (scene_radius * 2.5).max(self.camera.max_distance);

        let m = ray_orig - scene_center;
        let b = m.dot(ray_dir);
        let c = m.length_squared() - skybox_radius * skybox_radius;
        let disc = b * b - c;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            let t1 = -b - sqrt_disc;
            let t2 = -b + sqrt_disc;
            let t = if t1 > 1e-4 {
                t1
            } else if t2 > 1e-4 {
                t2
            } else {
                self.camera.distance
            };
            SceneRayHit::Skybox(ray_orig + ray_dir * t)
        } else {
            SceneRayHit::Skybox(ray_orig + ray_dir * self.camera.distance)
        }
    }

    /// Load every object from `path`, dispatching on its file extension
    /// (mirrors `manifold-cli`'s `load_objects`).
    fn import(&mut self, path: &Path, device: &eframe::egui_wgpu::wgpu::Device) {
        match load_objects(path, &mut self.next_object_id) {
            Ok(mut new_objects) => {
                object::center_on_bed(&mut new_objects, &self.machine.build_volume);
                self.objects.append(&mut new_objects);
                self.update_camera_bounds();
                self.reupload(device);
                self.import_error = None;
            }
            Err(err) => self.import_error = Some(err.to_string()),
        }
    }

    /// Removes the object at `index` from the scene and refreshes GPU buffers.
    fn remove_object(&mut self, index: usize, device: &eframe::egui_wgpu::wgpu::Device) {
        if index < self.objects.len() {
            self.objects.remove(index);
            self.update_camera_bounds();
            if let Some(selected) = self.selected {
                if selected == index {
                    self.selected = None;
                    self.lay_on_face_active = false;
                    self.cached_hull = None;
                } else if selected > index {
                    self.selected = Some(selected - 1);
                }
            }

            // Invalidate slicing outputs since geometry changed
            self.gcode = None;
            self.toolpaths = None;
            self.uploaded_toolpaths = None;
            self.toolpath_order_range = None;
            self.print_statistics = None;
            self.slice_error = None;

            // Clear SDF previews if no object is selected
            if self.selected.is_none() {
                self.sdf_slice = None;
                self.sdf_slice_texture = None;
                self.sdf_isosurface = None;
                self.sdf_overlay_mesh = None;
                self.sdf_error = None;
            }

            self.reupload(device);
        }
    }

    /// Removes all objects from the scene and resets selection.
    fn clear_objects(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        self.objects.clear();
        self.selected = None;
        self.update_camera_bounds();
        self.gcode = None;
        self.toolpaths = None;
        self.uploaded_toolpaths = None;
        self.toolpath_order_range = None;
        self.print_statistics = None;
        self.slice_error = None;
        self.sdf_slice = None;
        self.sdf_slice_texture = None;
        self.sdf_isosurface = None;
        self.sdf_overlay_mesh = None;
        self.sdf_error = None;
        self.reupload(device);
    }

    fn reupload(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        let uploaded = self
            .objects
            .iter()
            .map(|object| {
                UploadedMesh::upload(
                    device,
                    &object.mesh,
                    &object.transform,
                    self.mesh_overlay_mode,
                    Some(&self.config),
                )
            })
            .collect();
        self.uploaded_meshes = Arc::new(uploaded);
    }

    /// Rebuilds and re-uploads `uploaded_toolpaths` from `toolpaths`,
    /// mirroring `reupload`'s pattern for `uploaded_meshes`. No-op (clears
    /// the uploaded copy) if `toolpaths` is `None`.
    ///
    /// Filters segments via `self.scrub_order` using `toolpath_view`'s
    /// CPU-side rebuild-on-change approach (see that function's doc
    /// comment for the tradeoff versus a shader-side discard) — called
    /// both after a fresh `slice()` and whenever the scrub slider value
    /// changes.
    fn reupload_toolpaths(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        self.uploaded_toolpaths = self.toolpaths.as_ref().map(|paths| {
            let instances = toolpath_view::build_toolpath_lines_filtered(
                paths,
                self.scrub_order,
                self.toolpath_data_view,
                &self.config,
                Some(&self.machine),
                &self.hidden_line_types,
            );
            Arc::new(UploadedToolpaths::upload(device, &instances))
        });
    }

    /// Interact with the transform gizmo, but only let it capture pointer
    /// input (and thus contend with the orbit-camera drag on `viewport`'s
    /// canvas response) when the cursor is actually near the gizmo, or the
    /// gizmo is already mid-drag.
    ///
    /// `Gizmo::interact` (from `transform_gizmo_egui`) registers its own
    /// tiny probe widget at the cursor position *every frame it is called*,
    /// unconditionally reporting `hovered: true` regardless of proximity to
    /// the actual handles (real hit-testing happens afterward, internally).
    /// Because that probe widget is registered after — and thus takes
    /// pointer-interaction priority over — the canvas's own
    /// `Sense::click_and_drag` response, calling it every frame while any
    /// object is selected silently steals every orbit/pan drag anywhere in
    /// the viewport, not just drags that start on a handle. This method
    /// reimplements the crate's `GizmoExt::interact` convenience wrapper
    /// (see its source) but computes `hovered` from screen-space proximity
    /// to the gizmo's origin instead of an unconditional probe widget, so
    /// camera orbiting away from the selected object's gizmo works again.
    fn gizmo_interact(
        &mut self,
        ui: &egui::Ui,
        rect: egui::Rect,
        view_proj: glam::Mat4,
        origin: glam::DVec3,
        targets: &[GizmoTransform],
        allow_interaction: bool,
    ) -> Option<(GizmoResult, Vec<GizmoTransform>)> {
        const HOVER_RADIUS_PX: f32 = 220.0;

        let cursor_pos = ui.input(|i| i.pointer.hover_pos()).unwrap_or_default();

        // Only require screen-space proximity to *start* a new gizmo
        // interaction; once a subgizmo is already active (`is_focused`,
        // reflecting last frame's result), keep tracking the drag
        // regardless of how far the cursor has since moved — normal for
        // e.g. a long rotation drag.
        let near_gizmo = self.gizmo.is_focused()
            || world_to_screen(view_proj, rect, origin)
                .is_some_and(|screen_pos| screen_pos.distance(cursor_pos) < HOVER_RADIUS_PX);
        let hovered = allow_interaction && ui.rect_contains_pointer(rect) && near_gizmo;

        let gizmo_result = self.gizmo.update(
            GizmoInteraction {
                cursor_pos: (cursor_pos.x, cursor_pos.y),
                hovered,
                drag_started: hovered
                    && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary)),
                dragging: hovered
                    && ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary)),
            },
            targets,
        );

        let draw_data = self.gizmo.draw();
        ui.painter().add(egui::Mesh {
            indices: draw_data.indices,
            vertices: draw_data
                .vertices
                .into_iter()
                .zip(draw_data.colors)
                .map(|(pos, [r, g, b, a])| egui::epaint::Vertex {
                    pos: pos.into(),
                    uv: egui::Pos2::default(),
                    color: egui::Rgba::from_rgba_premultiplied(r, g, b, a).into(),
                })
                .collect(),
            ..Default::default()
        });

        gizmo_result
    }

    /// Kick off the slicing pipeline over the current `objects`/`machine`/
    /// `config` on a background thread, so the UI stays responsive while a
    /// slow slice runs. Progress and the final result arrive via
    /// `drain_slice_messages`, polled once per frame from `update()`.
    fn start_slice(&mut self) {
        let workspace = manifold_core::Workspace::new(
            self.objects.clone(),
            self.machine.clone(),
            self.config.clone(),
        );
        let (tx, rx) = std::sync::mpsc::channel();
        let progress_tx = tx.clone();
        let handle = std::thread::spawn(move || {
            let mut on_progress = move |fraction: f64| {
                let _ = progress_tx.send(SliceMessage::Progress(fraction));
            };
            let result = manifold_core::plan_toolpaths_with_progress(&workspace, &mut on_progress)
                .map_err(|error| error.to_string());
            let _ = tx.send(SliceMessage::Done(result));
        });
        self.slicing = Some(rx);
        self.slicing_handle = Some(handle);
        self.slice_progress = 0.0;
        self.slice_error = None;
    }

    /// Drains any pending messages from the background slicing thread
    /// started by `start_slice`, updating `slice_progress` and finalizing
    /// the result via `finish_slice` once `Done` arrives. Returns `true` if
    /// slicing just finished this call (so the caller can perform any
    /// follow-up GPU work that needs a `wgpu::Device`, e.g.
    /// `reupload_toolpaths`).
    ///
    /// If the channel disconnects without ever sending `Done` (the
    /// background thread panicked — an unhandled `unwrap()`/`expect()`/
    /// index-out-of-bounds somewhere in the slicing/toolpath pipeline, most
    /// likely triggered by degenerate/edge-case mesh geometry), this joins
    /// the thread to recover the panic payload and surfaces it via
    /// `slice_error` instead of leaving `self.slicing` stuck `Some` forever
    /// (which would otherwise spin the progress bar/spinner indefinitely
    /// with no feedback — see the "slice progress bar spins forever" bug).
    fn drain_slice_messages(&mut self) -> bool {
        let mut finished_result = None;
        let mut disconnected = false;
        if let Some(rx) = &self.slicing {
            loop {
                match rx.try_recv() {
                    Ok(SliceMessage::Progress(fraction)) => self.slice_progress = fraction,
                    Ok(SliceMessage::Done(result)) => {
                        finished_result = Some(result);
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }
        if let Some(result) = finished_result {
            self.slicing = None;
            self.slicing_handle = None;
            self.finish_slice(result);
            return true;
        }
        if disconnected {
            self.slicing = None;
            let panic_message = self
                .slicing_handle
                .take()
                .and_then(|handle| handle.join().err())
                .map(|payload| panic_payload_message(&payload))
                .unwrap_or_else(|| {
                    "background slicing thread ended without a result (no panic \
                     payload captured)"
                        .to_string()
                });
            self.slice_error = Some(format!("slicing thread panicked: {panic_message}"));
        }
        false
    }

    /// Store the result of a finished slice (or error) for preview/export
    /// (Phase 8, see ROADMAP.md). Shared by the synchronous finalization
    /// path in `drain_slice_messages`.
    fn finish_slice(&mut self, result: Result<Vec<manifold_core::toolpath::Path>, String>) {
        match result {
            Ok(paths) => {
                let stats = manifold_core::compute_print_statistics_with_machine(
                    &paths,
                    &self.config,
                    Some(&self.machine),
                    None,
                );
                let gcode = manifold_core::gcode::emit_with_machine(
                    &paths,
                    &self.config,
                    Some(&self.machine),
                );
                self.toolpath_order_range = toolpath_view::order_range(&paths);
                // Default the scrub slider to the max order so a fresh
                // slice shows every segment ("up to and including" the
                // top of the range).
                self.scrub_order = self
                    .toolpath_order_range
                    .map_or(f64::INFINITY, |(_, max)| max);
                self.uploaded_toolpaths = None;
                self.toolpaths = Some(paths);
                self.print_statistics = Some(stats);
                self.gcode = Some(gcode);
                self.slice_error = None;
            }
            Err(error) => {
                self.toolpaths = None;
                self.uploaded_toolpaths = None;
                self.toolpath_order_range = None;
                self.print_statistics = None;
                self.gcode = None;
                self.slice_error = Some(error);
            }
        }
    }
}

/// Helper widget for numeric input fields replacing fixed sliders: supports both
/// dragging and direct numeric entry, without imposing artificial upper bounds.
fn drag_num<T: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    value: &mut T,
    speed: f64,
    range: std::ops::RangeInclusive<T>,
    label: impl Into<egui::WidgetText>,
) -> egui::Response {
    ui.horizontal(|ui| {
        let drag_resp = ui.add(egui::DragValue::new(value).speed(speed).range(range));
        let label_resp = ui.label(label);
        drag_resp | label_resp
    })
    .inner
}

impl ManifoldApp {
    fn settings_panel(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let config_before = self.config.clone();
        ui.heading("Settings");

        egui::CollapsingHeader::new("Profile")
            .default_open(true)
            .show(ui, |ui| {
                let filename = self
                    .profile_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .and_then(|f| f.to_str())
                    .unwrap_or("(unsaved profile)");
                ui.horizontal(|ui| {
                    ui.label("File:");
                    ui.strong(filename);
                });

                ui.horizontal(|ui| {
                    ui.label("Description:");
                    ui.text_edit_singleline(&mut self.config.description);
                });

                ui.horizontal(|ui| {
                    if ui.button("Save Profile…").clicked() {
                        let mut dialog = rfd::FileDialog::new()
                            .add_filter("Profile", &["json"])
                            .set_file_name(
                                self.profile_path
                                    .as_ref()
                                    .and_then(|p| p.file_name())
                                    .and_then(|f| f.to_str())
                                    .unwrap_or("profile.json"),
                            );
                        if let Some(parent) = self.profile_path.as_ref().and_then(|p| p.parent()) {
                            dialog = dialog.set_directory(parent);
                        }
                        if let Some(path) = dialog.save_file() {
                            let profile = Profile {
                                machine: self.machine.clone(),
                                config: self.config.clone(),
                            };
                            match profile.save(&path) {
                                Ok(()) => {
                                    self.profile_path = Some(path);
                                    self.profile_error = None;
                                }
                                Err(error) => {
                                    self.profile_error = Some(error.to_string());
                                }
                            }
                        }
                    }

                    if ui.button("Load Profile…").clicked() {
                        let mut dialog = rfd::FileDialog::new().add_filter("Profile", &["json"]);
                        if let Some(parent) = self.profile_path.as_ref().and_then(|p| p.parent()) {
                            dialog = dialog.set_directory(parent);
                        }
                        if let Some(path) = dialog.pick_file() {
                            match Profile::load(&path) {
                                Ok(profile) => {
                                    self.machine = profile.machine;
                                    self.config = profile.config;
                                    self.next_tool_id = self
                                        .machine
                                        .tools
                                        .iter()
                                        .map(|tool| tool.id.0)
                                        .max()
                                        .map_or(0, |max_id| max_id + 1);
                                    self.profile_path = Some(path);
                                    self.profile_error = None;

                                    let device = frame
                                        .wgpu_render_state()
                                        .expect("wgpu renderer is required")
                                        .device
                                        .clone();
                                    self.update_camera_bounds();
                                    self.uploaded_scene =
                                        Arc::new(Self::build_scene(&device, &self.machine));
                                }
                                Err(error) => {
                                    self.profile_error = Some(error.to_string());
                                }
                            }
                        }
                    }
                });

                if let Some(err) = &self.profile_error {
                    ui.colored_label(egui::Color32::RED, format!("Profile failed: {err}"));
                }
            });

        ui.separator();

        ui.collapsing("Layering", |ui| {
            drag_num(
                ui,
                &mut self.config.layer_height,
                0.01,
                0.001..=f64::INFINITY,
                "Layer height (mm)",
            );
            let mut first_layer_h = self.config.first_layer_height();
            if drag_num(
                ui,
                &mut first_layer_h,
                0.01,
                0.001..=f64::INFINITY,
                "First layer height (mm)",
            )
            .changed()
            {
                self.config.first_layer_height = Some(first_layer_h);
            }
            drag_num(
                ui,
                &mut self.config.top_layers,
                1.0,
                0..=usize::MAX,
                "Top solid layers",
            );
            drag_num(
                ui,
                &mut self.config.bottom_layers,
                1.0,
                0..=usize::MAX,
                "Bottom solid layers",
            );
        });

        ui.collapsing("Extrusion & Walls", |ui| {
            drag_num(
                ui,
                &mut self.config.wall_line_width,
                0.01,
                0.01..=f64::INFINITY,
                "Wall line width (mm)",
            );
            let mut bead_clearance_compensation = self.config.bead_clearance_compensation_enabled();
            if ui
                .checkbox(
                    &mut bead_clearance_compensation,
                    "Bead clearance compensation",
                )
                .on_hover_text(
                    "Clamps bead width/height down to measured available room \
                     (nearby wall/channel width, flat-nozzle-land clearance) \
                     so extruded material never has nowhere to go.",
                )
                .changed()
            {
                self.config.bead_clearance_compensation_enabled = Some(bead_clearance_compensation);
            }
            let mut slope_mode = self.config.slope_compensation_mode();
            egui::ComboBox::from_label("Slope compensation mode")
                .selected_text(match slope_mode {
                    manifold_core::SlopeCompensationMode::GeometricOffset => {
                        "Geometric Offset (+Z)"
                    }
                    manifold_core::SlopeCompensationMode::VolumetricModulation => {
                        "Volumetric Modulation (Flow)"
                    }
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut slope_mode,
                        manifold_core::SlopeCompensationMode::GeometricOffset,
                        "Geometric Offset (+Z)",
                    );
                    ui.selectable_value(
                        &mut slope_mode,
                        manifold_core::SlopeCompensationMode::VolumetricModulation,
                        "Volumetric Modulation (Flow)",
                    );
                });
            self.config.slope_compensation_mode = Some(slope_mode);
            let mut wall_order = self.config.wall_order();
            egui::ComboBox::from_label("Wall order")
                .selected_text(match wall_order {
                    manifold_core::WallOrder::InnerOuterInner => "Inner / Outer / Inner",
                    manifold_core::WallOrder::OutsideIn => "Outside-In",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut wall_order,
                        manifold_core::WallOrder::InnerOuterInner,
                        "Inner / Outer / Inner",
                    );
                    ui.selectable_value(
                        &mut wall_order,
                        manifold_core::WallOrder::OutsideIn,
                        "Outside-In",
                    );
                });
            self.config.wall_order = Some(wall_order);
            drag_num(
                ui,
                &mut self.config.shell_thickness,
                0.01,
                0.0..=f64::INFINITY,
                "Shell thickness (mm)",
            );
            drag_num(
                ui,
                &mut self.config.wall_offset,
                0.01,
                0.0..=f64::INFINITY,
                "Wall offset (mm)",
            );
            let mut first_layer_w = self.config.first_layer_line_width();
            if drag_num(
                ui,
                &mut first_layer_w,
                0.01,
                0.01..=f64::INFINITY,
                "First layer line width (mm)",
            )
            .changed()
            {
                self.config.first_layer_line_width = Some(first_layer_w);
            }
            let mut first_layer_mult = self.config.first_layer_extrusion_multiplier();
            if drag_num(
                ui,
                &mut first_layer_mult,
                0.01,
                0.01..=f64::INFINITY,
                "First layer flow multiplier",
            )
            .changed()
            {
                self.config.first_layer_extrusion_multiplier = Some(first_layer_mult);
            }
            drag_num(
                ui,
                &mut self.config.filament_diameter,
                0.01,
                0.1..=f64::INFINITY,
                "Filament diameter (mm)",
            );
            let mut density = self.config.filament_density();
            if drag_num(
                ui,
                &mut density,
                0.01,
                0.01..=f64::INFINITY,
                "Filament density (g/cm³)",
            )
            .changed()
            {
                self.config.filament_density_g_cm3 = Some(density);
            }
        });

        ui.collapsing("Temperatures", |ui| {
            let mut def_nozzle = self.config.default_nozzle_temperature();
            if drag_num(
                ui,
                &mut def_nozzle,
                1.0,
                0.0..=f64::INFINITY,
                "Default nozzle temp (°C)",
            )
            .changed()
            {
                self.config.default_nozzle_temperature = Some(def_nozzle);
            }
            let mut bed_temp = self.config.bed_temperature();
            if drag_num(
                ui,
                &mut bed_temp,
                1.0,
                0.0..=f64::INFINITY,
                "Bed temperature (°C)",
            )
            .changed()
            {
                self.config.bed_temperature = Some(bed_temp);
            }
            let mut chamber_temp = self.config.chamber_temperature();
            if drag_num(
                ui,
                &mut chamber_temp,
                1.0,
                0.0..=f64::INFINITY,
                "Chamber temperature (°C)",
            )
            .changed()
            {
                self.config.chamber_temperature = Some(chamber_temp);
            }
        });

        ui.collapsing("Wave Overhangs", |ui| {
            ui.checkbox(
                &mut self.config.wave_overhangs_enabled,
                "Enable wave overhangs",
            );
            if self.config.wave_overhangs_enabled {
                let mut overlap = self.config.wave_overhang_overlap();
                if drag_num(
                    ui,
                    &mut overlap,
                    0.005,
                    0.0..=f64::INFINITY,
                    "Track overlap (mm)",
                )
                .changed()
                {
                    self.config.wave_overhang_overlap = Some(overlap);
                }
                let mut speed_mms = (self.config.wave_overhang_speed() / 60.0).round();
                if drag_num(
                    ui,
                    &mut speed_mms,
                    1.0,
                    0.1..=f64::INFINITY,
                    "Overhang speed (mm/s)",
                )
                .changed()
                {
                    self.config.wave_overhang_speed = Some(speed_mms * 60.0);
                }
                let mut flow = self.config.wave_overhang_flow();
                if drag_num(ui, &mut flow, 0.01, 0.01..=f64::INFINITY, "Flow multiplier").changed()
                {
                    self.config.wave_overhang_flow = Some(flow);
                }
                let mut fan_pct = self.config.overhang_fan_speed_percent();
                if drag_num(ui, &mut fan_pct, 1.0, 0.0..=100.0, "Overhang fan speed (%)").changed()
                {
                    self.config.overhang_fan_speed_percent = Some(fan_pct);
                }
            }
        });

        ui.collapsing("Retraction & Seams", |ui| {
            let mut use_fluid = self.config.use_fluid_dynamics();
            if ui
                .checkbox(
                    &mut use_fluid,
                    "Use dynamic fluid model (adaptive PA & retraction)",
                )
                .changed()
            {
                if use_fluid {
                    self.config.fluid_dynamics =
                        Some(manifold_core::fluid_dynamics::FluidDynamicsConfig::default());
                    self.config.use_firmware_retraction = false;
                } else {
                    self.config.fluid_dynamics = None;
                }
            }

            if let Some(ref mut fluid_cfg) = self.config.fluid_dynamics {
                ui.collapsing("Fluid Dynamics Parameters", |ui| {
                    ui.label("2-Point Pressure Advance Calibration:");
                    let mut pa_low = fluid_cfg.pa_calibration_low.0;
                    let mut q_low = fluid_cfg.pa_calibration_low.1;
                    if drag_num(
                        ui,
                        &mut pa_low,
                        0.001,
                        0.0..=f64::INFINITY,
                        "Low-flow PA (s)",
                    )
                    .changed()
                    {
                        fluid_cfg.pa_calibration_low.0 = pa_low;
                    }
                    if drag_num(
                        ui,
                        &mut q_low,
                        0.1,
                        0.0..=f64::INFINITY,
                        "Low-flow Q (mm³/s)",
                    )
                    .changed()
                    {
                        fluid_cfg.pa_calibration_low.1 = q_low;
                    }

                    let mut pa_high = fluid_cfg.pa_calibration_high.0;
                    let mut q_high = fluid_cfg.pa_calibration_high.1;
                    if drag_num(
                        ui,
                        &mut pa_high,
                        0.001,
                        0.0..=f64::INFINITY,
                        "High-flow PA (s)",
                    )
                    .changed()
                    {
                        fluid_cfg.pa_calibration_high.0 = pa_high;
                    }
                    if drag_num(
                        ui,
                        &mut q_high,
                        0.5,
                        0.0..=f64::INFINITY,
                        "High-flow Q (mm³/s)",
                    )
                    .changed()
                    {
                        fluid_cfg.pa_calibration_high.1 = q_high;
                    }

                    ui.separator();
                    ui.label("Thermal & Ooze Parameters:");
                    drag_num(
                        ui,
                        &mut fluid_cfg.static_retraction_mm,
                        0.01,
                        0.0..=f64::INFINITY,
                        "Static break distance (mm)",
                    );
                    drag_num(
                        ui,
                        &mut fluid_cfg.max_fan_temp_drop_c,
                        0.5,
                        0.0..=f64::INFINITY,
                        "Max fan temp drop (°C)",
                    );
                    drag_num(
                        ui,
                        &mut fluid_cfg.ooze_time_constant_ref_s,
                        0.05,
                        0.001..=f64::INFINITY,
                        "Ooze time constant τ (s)",
                    );
                    drag_num(
                        ui,
                        &mut fluid_cfg.ooze_max_length_ref_mm,
                        0.01,
                        -f64::INFINITY..=f64::INFINITY,
                        "Max ooze prime / swell (mm)",
                    );
                    let mut b_low = fluid_cfg.swell_ratio_low();
                    if drag_num(
                        ui,
                        &mut b_low,
                        0.01,
                        1.0..=f64::INFINITY,
                        "Swell ratio @ low flow (B_low)",
                    )
                    .changed()
                    {
                        fluid_cfg.swell_ratio_low = Some(b_low);
                    }
                    let mut b_high = fluid_cfg.swell_ratio_high();
                    if drag_num(
                        ui,
                        &mut b_high,
                        0.01,
                        1.0..=f64::INFINITY,
                        "Swell ratio @ high flow (B_high)",
                    )
                    .changed()
                    {
                        fluid_cfg.swell_ratio_high = Some(b_high);
                    }
                });
            } else {
                let mut r_len = self.config.retraction_length();
                if drag_num(
                    ui,
                    &mut r_len,
                    0.1,
                    0.0..=f64::INFINITY,
                    "Retraction distance (mm)",
                )
                .changed()
                {
                    self.config.retraction_length = Some(r_len);
                }
                let mut pa_val = self.config.pressure_advance.unwrap_or(0.0);
                if drag_num(
                    ui,
                    &mut pa_val,
                    0.001,
                    0.0..=f64::INFINITY,
                    "Pressure advance (s)",
                )
                .changed()
                {
                    self.config.pressure_advance = if pa_val > 0.0 { Some(pa_val) } else { None };
                }
                ui.checkbox(
                    &mut self.config.use_firmware_retraction,
                    "Use firmware retraction (G10/G11)",
                );
            }

            ui.checkbox(
                &mut self.config.enable_slicer_pressure_advance,
                "Slicer-side adaptive pressure advance",
            );
            if self.config.enable_slicer_pressure_advance {
                let mut tol = self.config.slicer_pa_tolerance_mm();
                if drag_num(
                    ui,
                    &mut tol,
                    0.0005,
                    0.0001..=f64::INFINITY,
                    "PA tolerance (mm)",
                )
                .changed()
                {
                    self.config.slicer_pa_tolerance_mm = Some(tol);
                }

                let mut min_seg = self.config.slicer_pa_min_segment_length();
                if drag_num(
                    ui,
                    &mut min_seg,
                    0.05,
                    0.01..=f64::INFINITY,
                    "PA min segment length (mm)",
                )
                .changed()
                {
                    self.config.slicer_pa_min_segment_length = Some(min_seg);
                }

                let mut max_freq = self.config.slicer_pa_max_frequency_hz();
                if drag_num(
                    ui,
                    &mut max_freq,
                    10.0,
                    10.0..=f64::INFINITY,
                    "PA max frequency (Hz)",
                )
                .changed()
                {
                    self.config.slicer_pa_max_frequency_hz = Some(max_freq);
                }
            }

            let mut r_spd_mms = (self.config.retraction_speed() / 60.0).round();
            if drag_num(
                ui,
                &mut r_spd_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Retraction speed (mm/s)",
            )
            .changed()
            {
                self.config.retraction_speed = Some(r_spd_mms * 60.0);
            }

            let mut u_spd_mms = (self.config.unretract_speed() / 60.0).round();
            if drag_num(
                ui,
                &mut u_spd_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Unretract speed (mm/s)",
            )
            .changed()
            {
                self.config.unretract_speed = Some(u_spd_mms * 60.0);
            }

            let mut u_extra = self.config.unretract_extra_length();
            if drag_num(
                ui,
                &mut u_extra,
                0.01,
                -f64::INFINITY..=f64::INFINITY,
                "Unretract extra length (mm)",
            )
            .changed()
            {
                self.config.unretract_extra_length = Some(u_extra);
            }

            let mut min_travel_retract = self.config.min_travel_for_retract();
            if drag_num(
                ui,
                &mut min_travel_retract,
                0.1,
                0.0..=f64::INFINITY,
                "Minimum travel for retract (mm)",
            )
            .changed()
            {
                self.config.min_travel_for_retract = Some(min_travel_retract);
            }

            let mut taper_dist = self.config.pre_retract_taper_distance.unwrap_or(0.0);
            if drag_num(
                ui,
                &mut taper_dist,
                0.1,
                0.0..=f64::INFINITY,
                "Pre-retract taper distance (mm)",
            )
            .changed()
            {
                self.config.pre_retract_taper_distance = if taper_dist > 0.0 {
                    Some(taper_dist)
                } else {
                    None
                };
            }

            ui.separator();
            ui.label("Seams & Surface Transitions:");

            let mut s_gap = self.config.seam_gap();
            if drag_num(
                ui,
                &mut s_gap,
                0.05,
                0.0..=f64::INFINITY,
                "Seam gap / coasting distance (mm)",
            )
            .changed()
            {
                self.config.seam_gap = if s_gap > 1e-4 { Some(s_gap) } else { None };
            }

            ui.checkbox(&mut self.config.wipe_enabled, "Wipe on retraction");
            if self.config.wipe_enabled {
                let mut wipe_dist = self.config.wipe_distance();
                if drag_num(
                    ui,
                    &mut wipe_dist,
                    0.1,
                    0.0..=f64::INFINITY,
                    "Wipe distance (mm)",
                )
                .changed()
                {
                    self.config.wipe_distance = Some(wipe_dist);
                }
            }

            ui.checkbox(&mut self.config.scarf_joint_enabled, "Scarf joint seams");
            if self.config.scarf_joint_enabled {
                let mut scarf_len = self.config.scarf_joint_length();
                if drag_num(
                    ui,
                    &mut scarf_len,
                    0.5,
                    0.0..=f64::INFINITY,
                    "Scarf overlap length (mm)",
                )
                .changed()
                {
                    self.config.scarf_joint_length = Some(scarf_len);
                }

                let mut scarf_steps = self.config.scarf_joint_steps();
                if drag_num(ui, &mut scarf_steps, 1.0, 2..=usize::MAX, "Scarf steps").changed() {
                    self.config.scarf_joint_steps = Some(scarf_steps);
                }

                let mut scarf_start_h = self.config.scarf_joint_start_height_fraction() * 100.0;
                if drag_num(
                    ui,
                    &mut scarf_start_h,
                    1.0,
                    0.0..=100.0,
                    "Scarf start height (%)",
                )
                .changed()
                {
                    self.config.scarf_joint_start_height_fraction = Some(scarf_start_h / 100.0);
                }

                let mut scarf_flow = self.config.scarf_joint_flow_ratio();
                if drag_num(
                    ui,
                    &mut scarf_flow,
                    0.01,
                    0.01..=f64::INFINITY,
                    "Scarf joint flow multiplier",
                )
                .changed()
                {
                    self.config.scarf_joint_flow_ratio = Some(scarf_flow);
                }
            }
        });

        ui.collapsing("Travel & Simplification", |ui| {
            ui.checkbox(&mut self.config.z_hop_enabled, "Z-hop on travel");
            if self.config.z_hop_enabled {
                drag_num(
                    ui,
                    &mut self.config.z_hop_height,
                    0.05,
                    0.0..=f64::INFINITY,
                    "Z-hop height (mm)",
                );
            }
            ui.checkbox(
                &mut self.config.path_simplify_enabled,
                "Simplify wall toolpaths",
            );
            if self.config.path_simplify_enabled {
                drag_num(
                    ui,
                    &mut self.config.path_simplify_tolerance,
                    0.005,
                    0.0..=f64::INFINITY,
                    "Simplification tolerance (mm)",
                );
            }
        });

        ui.separator();
        ui.heading("Infill");
        let mut sparse_pattern = self.config.sparse_infill_pattern();
        if egui::ComboBox::from_label("Sparse pattern")
            .selected_text(infill_pattern_label(sparse_pattern))
            .show_ui(ui, |ui| {
                let mut changed = false;
                changed |= ui
                    .selectable_value(&mut sparse_pattern, InfillPatternKind::Cubic, "Cubic")
                    .changed();
                changed |= ui
                    .selectable_value(&mut sparse_pattern, InfillPatternKind::Gyroid, "Gyroid")
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut sparse_pattern,
                        InfillPatternKind::SchwarzD,
                        "Schwarz Diamond (D)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut sparse_pattern,
                        InfillPatternKind::SchwarzP,
                        "Schwarz Primitive (P)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut sparse_pattern,
                        InfillPatternKind::Monotonic,
                        "Monotonic",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut sparse_pattern,
                        InfillPatternKind::Concentric,
                        "Concentric",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut sparse_pattern,
                        InfillPatternKind::AllWalls,
                        "All Walls",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(&mut sparse_pattern, InfillPatternKind::None, "None")
                    .changed();
                changed
            })
            .inner
            .unwrap_or(false)
        {
            self.config.sparse_infill_pattern = Some(sparse_pattern);
        }

        let mut solid_pattern = self.config.solid_infill_pattern();
        if egui::ComboBox::from_label("Solid pattern")
            .selected_text(infill_pattern_label(solid_pattern))
            .show_ui(ui, |ui| {
                let mut changed = false;
                changed |= ui
                    .selectable_value(&mut solid_pattern, InfillPatternKind::AllWalls, "All Walls")
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut solid_pattern,
                        InfillPatternKind::Concentric,
                        "Concentric",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut solid_pattern,
                        InfillPatternKind::Monotonic,
                        "Monotonic",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(&mut solid_pattern, InfillPatternKind::Cubic, "Cubic")
                    .changed();
                changed |= ui
                    .selectable_value(&mut solid_pattern, InfillPatternKind::Gyroid, "Gyroid")
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut solid_pattern,
                        InfillPatternKind::SchwarzD,
                        "Schwarz Diamond (D)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut solid_pattern,
                        InfillPatternKind::SchwarzP,
                        "Schwarz Primitive (P)",
                    )
                    .changed();
                changed |= ui
                    .selectable_value(&mut solid_pattern, InfillPatternKind::None, "None")
                    .changed();
                changed
            })
            .inner
            .unwrap_or(false)
        {
            self.config.solid_infill_pattern = Some(solid_pattern);
        }
        drag_num(
            ui,
            &mut self.config.infill_line_width,
            0.01,
            0.01..=f64::INFINITY,
            "Infill line width (mm)",
        );
        drag_num(
            ui,
            &mut self.config.infill_angle_deg,
            1.0,
            0.0..=360.0,
            "Infill angle (deg)",
        );
        drag_num(
            ui,
            &mut self.config.infill_density,
            0.01,
            0.0..=1.0,
            "Infill density",
        );

        ui.separator();
        ui.heading("Order field");
        let previous_order_field = self.config.order_field;
        egui::ComboBox::from_label("Kind")
            .selected_text(format!("{:?}", self.config.order_field))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.config.order_field,
                    OrderFieldKind::Height,
                    "Height",
                );
                ui.selectable_value(
                    &mut self.config.order_field,
                    OrderFieldKind::Conical,
                    "Conical",
                );
                ui.selectable_value(
                    &mut self.config.order_field,
                    OrderFieldKind::Eikonal,
                    "Eikonal",
                );
                ui.selectable_value(
                    &mut self.config.order_field,
                    OrderFieldKind::DualIso,
                    "DualIso",
                );
            });
        if previous_order_field != OrderFieldKind::Conical
            && self.config.order_field == OrderFieldKind::Conical
        {
            // Default the apex to the selected (else first) object's world-
            // space bounding-box center, and the axis to the same
            // "vertically up" direction `SlicerConfig::default` already uses
            // for `order_field_axis` (matching `Height`'s direction, so
            // switching kinds is a smooth transition) — a reasonable
            // starting cone for whatever is loaded, for now.
            let object = self
                .selected
                .and_then(|index| self.objects.get(index))
                .or_else(|| self.objects.first());
            if let Some(object) = object {
                if let Some((min, max)) = object.mesh.bounding_box() {
                    let local_center = (min + max) * 0.5;
                    self.config.order_field_apex = object.transform.transform_point(local_center);
                }
            }
            self.config.order_field_axis = manifold_core::SlicerConfig::default().order_field_axis;
        }
        if self.config.order_field == OrderFieldKind::Conical {
            ui.horizontal(|ui| {
                ui.label("Apex");
                ui.add(egui::DragValue::new(&mut self.config.order_field_apex.x).prefix("x: "));
                ui.add(egui::DragValue::new(&mut self.config.order_field_apex.y).prefix("y: "));
                ui.add(egui::DragValue::new(&mut self.config.order_field_apex.z).prefix("z: "));
            });
            drag_num(
                ui,
                &mut self.config.order_field_slope,
                0.01,
                0.0..=f64::INFINITY,
                "Cone slope",
            );
        }
        if self.config.order_field == OrderFieldKind::Eikonal
            || self.config.order_field == OrderFieldKind::DualIso
        {
            let mut surface_weight = self.config.eikonal_surface_order_weight();
            if drag_num(
                ui,
                &mut surface_weight,
                0.05,
                0.0..=f64::INFINITY,
                "Surface order weight",
            )
            .on_hover_text(
                "Weight multiplier for the geodesic SurfaceEikonal lower bound on the model skin. 1.0 enforces the exact surface arrival time to eliminate surface local minima; 0.0 disables it.",
            )
            .changed()
            {
                self.config.eikonal_surface_order_weight = Some(surface_weight);
            }
            ui.checkbox(
                &mut self.config.eikonal_conform_top_surfaces,
                "Conform to top surfaces",
            );
            if self.config.eikonal_conform_top_surfaces {
                let mut max_angle = self.config.eikonal_conformal_max_angle_deg();
                if drag_num(
                    ui,
                    &mut max_angle,
                    0.5,
                    0.0..=90.0,
                    "Top conform detach angle (°)",
                )
                .changed()
                {
                    self.config.eikonal_conformal_max_angle_deg = Some(max_angle);
                }
            }
            ui.checkbox(
                &mut self.config.eikonal_conform_bottom_surfaces,
                "Conform to bottom surfaces",
            );
            if self.config.eikonal_conform_bottom_surfaces {
                let mut bottom_angle = self.config.eikonal_conformal_bottom_max_angle_deg();
                if drag_num(
                    ui,
                    &mut bottom_angle,
                    0.5,
                    0.0..=90.0,
                    "Bottom conform detach angle (°)",
                )
                .changed()
                {
                    self.config.eikonal_conformal_bottom_max_angle_deg = Some(bottom_angle);
                }
            }
            if self.config.eikonal_conform_top_surfaces
                || self.config.eikonal_conform_bottom_surfaces
            {
                let mut skin_depth = self.config.eikonal_conformal_skin_depth_mm();
                if drag_num(
                    ui,
                    &mut skin_depth,
                    0.1,
                    0.0..=f64::INFINITY,
                    "Conformal skin depth (mm)",
                )
                .changed()
                {
                    self.config.eikonal_conformal_skin_depth_mm = Some(skin_depth);
                }
            }
            ui.checkbox(
                &mut self.config.eikonal_enforce_monotonic_growth,
                "Enforce vertical monotonicity",
            )
            .on_hover_text(
                "Enforces strictly increasing layer order along vertical columns (dOrder/dz >= 0.15) to prevent downward stalls or mid-air floating loops.",
            );
            ui.collapsing("Toolhead clearance profile (XZ)", |ui| {
                ui.label("Radial distance (X) and height (Z) from nozzle tip");
                let mut remove_index: Option<usize> = None;
                for (i, (x, z)) in self.machine.eikonal_slope_profile.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(x)
                                .prefix("x: ")
                                .suffix(" mm")
                                .range(0.001..=f64::INFINITY)
                                .speed(0.1),
                        );
                        ui.add(
                            egui::DragValue::new(z)
                                .prefix("z: ")
                                .suffix(" mm")
                                .range(0.0..=f64::INFINITY)
                                .speed(0.1),
                        );
                        let angle = if *z > 0.0 && *x > 0.0 {
                            (*z / *x).atan().to_degrees()
                        } else {
                            0.0
                        };
                        ui.label(
                            egui::RichText::new(format!("({angle:.1}°)"))
                                .size(11.0)
                                .color(egui::Color32::from_gray(160)),
                        );
                        if ui.button("Remove").clicked() {
                            remove_index = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_index {
                    self.machine.eikonal_slope_profile.remove(i);
                }
                if ui.button("Add point").clicked() {
                    let (next_x, next_z) = self
                        .machine
                        .eikonal_slope_profile
                        .last()
                        .map(|&(x, z)| (x + 5.0, z + 5.0))
                        .unwrap_or((0.8, 0.0));
                    self.machine.eikonal_slope_profile.push((next_x, next_z));
                }
            });
        }

        ui.separator();
        ui.heading("Speeds & Accelerations");

        ui.collapsing("Speeds (mm/s)", |ui| {
            let mut print_speed_mms = (self.config.print_speed / 60.0).round();
            if drag_num(
                ui,
                &mut print_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Print speed (mm/s)",
            )
            .changed()
            {
                self.config.print_speed = print_speed_mms * 60.0;
            }
            let mut outer_wall_speed_mms =
                (self.config.outer_wall_speed.unwrap_or_else(|| {
                    (self.config.print_speed * 0.6).min(self.config.print_speed)
                }) / 60.0)
                    .round();
            if drag_num(
                ui,
                &mut outer_wall_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Outer wall speed (mm/s)",
            )
            .changed()
            {
                self.config.outer_wall_speed = Some(outer_wall_speed_mms * 60.0);
            }
            let mut inner_wall_speed_mms = (self
                .config
                .inner_wall_speed
                .unwrap_or(self.config.print_speed)
                / 60.0)
                .round();
            if drag_num(
                ui,
                &mut inner_wall_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Inner wall speed (mm/s)",
            )
            .changed()
            {
                self.config.inner_wall_speed = Some(inner_wall_speed_mms * 60.0);
            }
            let mut infill_speed_mms =
                (self.config.infill_speed.unwrap_or(self.config.print_speed) / 60.0).round();
            if drag_num(
                ui,
                &mut infill_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Infill speed (mm/s)",
            )
            .changed()
            {
                self.config.infill_speed = Some(infill_speed_mms * 60.0);
            }
            let mut solid_infill_speed_mms =
                (self.config.solid_infill_speed.unwrap_or_else(|| {
                    (self.config.print_speed * 0.8).min(self.config.print_speed)
                }) / 60.0)
                    .round();
            if drag_num(
                ui,
                &mut solid_infill_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Solid infill speed (mm/s)",
            )
            .changed()
            {
                self.config.solid_infill_speed = Some(solid_infill_speed_mms * 60.0);
            }
            let mut bridge_speed_mms =
                (self.config.bridge_speed.unwrap_or_else(|| {
                    (self.config.print_speed * 0.5).min(self.config.print_speed)
                }) / 60.0)
                    .round();
            if drag_num(
                ui,
                &mut bridge_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Bridge speed (mm/s)",
            )
            .changed()
            {
                self.config.bridge_speed = Some(bridge_speed_mms * 60.0);
            }
            let mut first_layer_speed_mms = (self.config.first_layer_print_speed() / 60.0).round();
            if drag_num(
                ui,
                &mut first_layer_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "First layer speed (mm/s)",
            )
            .changed()
            {
                self.config.first_layer_print_speed = Some(first_layer_speed_mms * 60.0);
            }
            let mut travel_speed_mms = (self.config.travel_speed / 60.0).round();
            if drag_num(
                ui,
                &mut travel_speed_mms,
                1.0,
                0.1..=f64::INFINITY,
                "Travel speed (mm/s)",
            )
            .changed()
            {
                self.config.travel_speed = travel_speed_mms * 60.0;
            }

            let mut max_vol_speed = self.config.max_volumetric_speed.unwrap_or(0.0);
            if drag_num(
                ui,
                &mut max_vol_speed,
                0.5,
                0.0..=f64::INFINITY,
                "Max volumetric speed (mm³/s) (0=off)",
            )
            .changed()
            {
                self.config.max_volumetric_speed = if max_vol_speed > 0.0 {
                    Some(max_vol_speed)
                } else {
                    None
                };
            }
            let mut spd_deadband = self.config.speed_deadband_percent();
            if drag_num(
                ui,
                &mut spd_deadband,
                0.5,
                0.0..=100.0,
                "Speed deadband (%)",
            )
            .changed()
            {
                self.config.speed_deadband_percent = Some(spd_deadband);
            }
        });

        ui.collapsing("Accelerations (mm/s²)", |ui| {
            let mut def_accel = self.config.default_acceleration.unwrap_or(5000.0);
            if drag_num(
                ui,
                &mut def_accel,
                50.0,
                1.0..=f64::INFINITY,
                "Default acceleration",
            )
            .changed()
            {
                self.config.default_acceleration = Some(def_accel);
            }
            let mut outer_wall_accel = self.config.outer_wall_acceleration.unwrap_or(2500.0);
            if drag_num(
                ui,
                &mut outer_wall_accel,
                50.0,
                1.0..=f64::INFINITY,
                "Outer wall acceleration",
            )
            .changed()
            {
                self.config.outer_wall_acceleration = Some(outer_wall_accel);
            }
            let mut inner_wall_accel = self.config.inner_wall_acceleration.unwrap_or(5000.0);
            if drag_num(
                ui,
                &mut inner_wall_accel,
                50.0,
                1.0..=f64::INFINITY,
                "Inner wall acceleration",
            )
            .changed()
            {
                self.config.inner_wall_acceleration = Some(inner_wall_accel);
            }
            let mut infill_accel = self.config.infill_acceleration.unwrap_or(7000.0);
            if drag_num(
                ui,
                &mut infill_accel,
                50.0,
                1.0..=f64::INFINITY,
                "Infill acceleration",
            )
            .changed()
            {
                self.config.infill_acceleration = Some(infill_accel);
            }
            let mut travel_accel = self.config.travel_acceleration.unwrap_or(10000.0);
            if drag_num(
                ui,
                &mut travel_accel,
                50.0,
                1.0..=f64::INFINITY,
                "Travel acceleration",
            )
            .changed()
            {
                self.config.travel_acceleration = Some(travel_accel);
            }
            let mut first_layer_accel = self.config.first_layer_acceleration.unwrap_or(2000.0);
            if drag_num(
                ui,
                &mut first_layer_accel,
                50.0,
                1.0..=f64::INFINITY,
                "First layer acceleration",
            )
            .changed()
            {
                self.config.first_layer_acceleration = Some(first_layer_accel);
            }
            let mut scv = self.config.square_corner_velocity();
            if drag_num(
                ui,
                &mut scv,
                0.5,
                0.1..=f64::INFINITY,
                "Square corner velocity (mm/s)",
            )
            .changed()
            {
                self.config.square_corner_velocity = Some(scv);
            }
            let mut accel_deadband = self.config.acceleration_deadband_percent();
            if drag_num(
                ui,
                &mut accel_deadband,
                0.5,
                0.0..=100.0,
                "Acceleration deadband (%)",
            )
            .changed()
            {
                self.config.acceleration_deadband_percent = Some(accel_deadband);
            }
            let mut cruise_ratio = self.config.minimum_cruise_ratio();
            if drag_num(
                ui,
                &mut cruise_ratio,
                0.05,
                0.0..=1.0,
                "Minimum cruise ratio",
            )
            .changed()
            {
                self.config.minimum_cruise_ratio = Some(cruise_ratio);
            }
        });

        ui.separator();
        ui.heading("Cooling & Fan");
        let mut fan_pct = self.config.fan_speed_percent();
        if drag_num(ui, &mut fan_pct, 1.0, 0.0..=100.0, "Fan speed (%)").changed() {
            self.config.fan_speed_percent = Some(fan_pct);
        }
        let mut overhang_fan_pct = self.config.overhang_fan_speed_percent();
        if drag_num(
            ui,
            &mut overhang_fan_pct,
            1.0,
            0.0..=100.0,
            "Overhang fan speed (%)",
        )
        .changed()
        {
            self.config.overhang_fan_speed_percent = Some(overhang_fan_pct);
        }
        let mut fan_delay = self.config.fan_layer_delay();
        if drag_num(
            ui,
            &mut fan_delay,
            1.0,
            0..=u32::MAX,
            "Fan disabled initial layers",
        )
        .changed()
        {
            self.config.fan_layer_delay = Some(fan_delay);
        }

        ui.separator();
        ui.heading("Machine");
        let (min, mut max) = self.machine.build_volume.bounding_box();
        let mut bed_changed = false;
        bed_changed |= drag_num(ui, &mut max.x, 1.0, 1.0..=f64::INFINITY, "Bed X (mm)").changed();
        bed_changed |= drag_num(ui, &mut max.y, 1.0, 1.0..=f64::INFINITY, "Bed Y (mm)").changed();
        bed_changed |= drag_num(
            ui,
            &mut max.z,
            1.0,
            1.0..=f64::INFINITY,
            "Build height (mm)",
        )
        .changed();
        if bed_changed {
            self.machine.build_volume = BoundingVolume::Aabb { min, max };
            self.update_camera_bounds();
            let device = frame
                .wgpu_render_state()
                .expect("wgpu renderer is required")
                .device
                .clone();
            self.uploaded_scene = Arc::new(Self::build_scene(&device, &self.machine));
        }
        ui.collapsing("Tools & Nozzles", |ui| {
            let mut remove_idx = None;
            let num_tools = self.machine.tools.len();
            for (i, tool) in self.machine.tools.iter_mut().enumerate() {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(format!("Tool {}", tool.id.0));
                        if num_tools > 1 && ui.button("Remove").clicked() {
                            remove_idx = Some(i);
                        }
                    });
                    drag_num(
                        ui,
                        &mut tool.nozzle_diameter,
                        0.01,
                        0.01..=f64::INFINITY,
                        "Nozzle diameter (mm)",
                    );
                    let mut flat_diam = tool.nozzle_flat_diameter();
                    if drag_num(
                        ui,
                        &mut flat_diam,
                        0.05,
                        0.0..=f64::INFINITY,
                        "Nozzle flat land diameter (mm)",
                    )
                    .changed()
                    {
                        tool.nozzle_flat_diameter = Some(flat_diam);
                    }
                    drag_num(
                        ui,
                        &mut tool.extrusion_multiplier,
                        0.01,
                        0.01..=f64::INFINITY,
                        "Extrusion multiplier",
                    );
                    let mut temp = tool.nozzle_temperature();
                    if drag_num(ui, &mut temp, 1.0, 0.0..=f64::INFINITY, "Nozzle temp (°C)")
                        .changed()
                    {
                        tool.nozzle_temperature = Some(temp);
                    }
                });
            }
            if let Some(idx) = remove_idx {
                self.machine.tools.remove(idx);
            }
            if ui.button("Add tool").clicked() {
                let mut new_tool = Tool::new(ToolId(self.next_tool_id), 0.4);
                new_tool.nozzle_temperature = Some(self.config.default_nozzle_temperature());
                self.machine.tools.push(new_tool);
                self.next_tool_id += 1;
            }
        });

        ui.checkbox(
            &mut self.machine.use_stepper_dynamics,
            "Use global stepper dynamics model",
        );
        if self.machine.use_stepper_dynamics {
            ui.collapsing("Global Stepper Dynamics", |ui| {
                let mut a0 = self.machine.zero_speed_acceleration();
                if drag_num(
                    ui,
                    &mut a0,
                    50.0,
                    1.0..=f64::INFINITY,
                    "Zero-speed acceleration (a₀, mm/s²)",
                )
                .changed()
                {
                    self.machine.zero_speed_acceleration = Some(a0);
                }

                let mut vmax = self.machine.max_available_speed();
                if drag_num(
                    ui,
                    &mut vmax,
                    5.0,
                    1.0..=f64::INFINITY,
                    "Max available speed (v_max, mm/s)",
                )
                .changed()
                {
                    self.machine.max_available_speed = Some(vmax);
                }

                let mut a_limit = self.machine.acceleration_limit();
                if drag_num(
                    ui,
                    &mut a_limit,
                    50.0,
                    1.0..=f64::INFINITY,
                    "Acceleration limit (mm/s²)",
                )
                .changed()
                {
                    self.machine.acceleration_limit = Some(a_limit);
                }

                let mut v_limit = self.machine.speed_limit();
                if drag_num(
                    ui,
                    &mut v_limit,
                    5.0,
                    1.0..=f64::INFINITY,
                    "Speed limit (mm/s)",
                )
                .changed()
                {
                    self.machine.speed_limit = Some(v_limit);
                }

                let mut mach_cruise = self.machine.minimum_cruise_ratio();
                if drag_num(
                    ui,
                    &mut mach_cruise,
                    0.05,
                    0.0..=1.0,
                    "Minimum cruise ratio",
                )
                .changed()
                {
                    self.machine.minimum_cruise_ratio = Some(mach_cruise);
                }
            });
        }

        ui.collapsing("Per-Axis Kinematics & Stepper Dynamics", |ui| {
            let global_speed = self.machine.speed_limit();
            let global_accel = self.machine.acceleration_limit();
            let global_a0 = self.machine.zero_speed_acceleration();
            let global_vmax = self.machine.max_available_speed();

            for axis in [
                manifold_core::kinematics::Axis::X,
                manifold_core::kinematics::Axis::Y,
                manifold_core::kinematics::Axis::Z,
            ] {
                let axis_name = match axis {
                    manifold_core::kinematics::Axis::X => "X Axis",
                    manifold_core::kinematics::Axis::Y => "Y Axis",
                    manifold_core::kinematics::Axis::Z => "Z Axis",
                };
                let mut has_override = self.machine.axis_limits.contains_key(&axis);
                ui.collapsing(axis_name, |ui| {
                    if ui
                        .checkbox(&mut has_override, "Override global limits")
                        .changed()
                    {
                        if has_override {
                            self.machine
                                .axis_limits
                                .insert(axis, manifold_core::kinematics::AxisLimits::default());
                        } else {
                            self.machine.axis_limits.remove(&axis);
                        }
                    }
                    if let Some(limits) = self.machine.axis_limits.get_mut(&axis) {
                        let mut spd = limits.speed_limit.unwrap_or(match axis {
                            manifold_core::kinematics::Axis::Z => 40.0,
                            _ => global_speed,
                        });
                        if drag_num(
                            ui,
                            &mut spd,
                            1.0,
                            0.1..=f64::INFINITY,
                            "Max speed limit (mm/s)",
                        )
                        .changed()
                        {
                            limits.speed_limit = Some(spd);
                        }

                        let mut accel = limits.acceleration_limit.unwrap_or(match axis {
                            manifold_core::kinematics::Axis::Z => 1500.0,
                            _ => global_accel,
                        });
                        if drag_num(
                            ui,
                            &mut accel,
                            50.0,
                            1.0..=f64::INFINITY,
                            "Max acceleration (mm/s²)",
                        )
                        .changed()
                        {
                            limits.acceleration_limit = Some(accel);
                        }

                        ui.checkbox(
                            &mut limits.use_stepper_dynamics,
                            "Use stepper dynamics model",
                        );
                        if limits.use_stepper_dynamics {
                            let mut a0 = limits.zero_speed_acceleration.unwrap_or(match axis {
                                manifold_core::kinematics::Axis::Z => 3000.0,
                                _ => global_a0,
                            });
                            if drag_num(
                                ui,
                                &mut a0,
                                50.0,
                                1.0..=f64::INFINITY,
                                "Zero-speed accel (a₀, mm/s²)",
                            )
                            .changed()
                            {
                                limits.zero_speed_acceleration = Some(a0);
                            }

                            let mut vmax = limits.max_available_speed.unwrap_or(match axis {
                                manifold_core::kinematics::Axis::Z => 60.0,
                                _ => global_vmax,
                            });
                            if drag_num(
                                ui,
                                &mut vmax,
                                5.0,
                                1.0..=f64::INFINITY,
                                "Max available speed (v_max, mm/s)",
                            )
                            .changed()
                            {
                                limits.max_available_speed = Some(vmax);
                            }
                        }
                    }
                });
            }
        });

        ui.collapsing("Custom Gcode Macros", |ui| {
            ui.label("Start Gcode");
            ui.add(
                egui::TextEdit::multiline(&mut self.config.start_gcode)
                    .desired_rows(3)
                    .code_editor()
                    .hint_text("e.g. PRINT_START EXTRUDER={first_used_tool_temperature} BED={bed_temperature} CHAMBER={chamber_temperature} PRINT_MIN={print_min_x},{print_min_y} PRINT_MAX={print_max_x},{print_max_y}"),
            );
            ui.label("End Gcode");
            ui.add(
                egui::TextEdit::multiline(&mut self.config.end_gcode)
                    .desired_rows(2)
                    .code_editor()
                    .hint_text("e.g. PRINT_END"),
            );
            ui.label(
                "Placeholders:\n\
                 • Bounding Box: {print_min_x} {print_min_y} {print_max_x} {print_max_y}\n\
                 • Temperatures: {bed_temperature} {chamber_temperature}\n\
                 • First Tool: {first_used_tool} {first_used_tool_temperature} (or {nozzle_temperature})\n\
                 • Per-Tool: {temperature_0} {temperature_1} {nozzle_temperature_0}",
            );
        });

        ui.separator();
        ui.heading("Objects");
        if self.objects.is_empty() {
            ui.label("No objects loaded. Use Import to load an STL or 3MF file.");
        } else {
            let tool_ids: Vec<ToolId> = self.machine.tools.iter().map(|tool| tool.id).collect();
            let mut remove_index = None;
            for (index, object) in self.objects.iter_mut().enumerate() {
                let selected = self.selected == Some(index);
                let label = format!(
                    "{} — {} triangles",
                    object.display_name(),
                    object.mesh.triangle_count()
                );
                ui.horizontal(|ui| {
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected = if selected { None } else { Some(index) };
                    }
                    egui::ComboBox::from_id_salt(("object_tool", object.id.0))
                        .selected_text(format!("Tool {}", object.tool.0))
                        .show_ui(ui, |ui| {
                            for &tool_id in &tool_ids {
                                ui.selectable_value(
                                    &mut object.tool,
                                    tool_id,
                                    format!("Tool {}", tool_id.0),
                                );
                            }
                        });
                    if ui.button("Remove").clicked() {
                        remove_index = Some(index);
                    }
                });
            }
            if let Some(index) = remove_index {
                let device = frame
                    .wgpu_render_state()
                    .expect("wgpu renderer is required")
                    .device
                    .clone();
                self.remove_object(index, &device);
            }
            if self.objects.len() > 1 && ui.button("Clear all objects").clicked() {
                let device = frame
                    .wgpu_render_state()
                    .expect("wgpu renderer is required")
                    .device
                    .clone();
                self.clear_objects(&device);
            }
        }

        if let Some(err) = &self.import_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, err);
        }

        if let Some(err) = &self.slice_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("Slice failed: {err}"));
        }

        ui.separator();
        ui.checkbox(&mut self.show_sdf_panel, "Show SDF debug panel");

        if let Some(gcode) = &self.gcode {
            ui.separator();
            ui.heading("Gcode");
            if let Some(stats) = &self.print_statistics {
                ui.label(format!("Estimated time: {}", stats.formatted_time()));
                ui.label(format!(
                    "Filament: {:.2} m ({:.1} g / {:.2} cm³)",
                    stats.filament_length_meters,
                    stats.filament_weight_grams,
                    stats.filament_volume_cm3
                ));
            }
            ui.label(format!("{} line(s) generated", gcode.lines().count()));
            egui::ScrollArea::both()
                .max_height(150.0)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.monospace(gcode);
                });
        }

        if self.mesh_overlay_mode != MeshOverlayMode::None
            && (self.config.layer_height != config_before.layer_height
                || self.config.nozzle_diameter != config_before.nozzle_diameter
                || self.config.eikonal_surface_order_weight
                    != config_before.eikonal_surface_order_weight
                || self.config.eikonal_enforce_monotonic_growth
                    != config_before.eikonal_enforce_monotonic_growth
                || self.config.eikonal_conform_top_surfaces
                    != config_before.eikonal_conform_top_surfaces
                || self.config.eikonal_conformal_max_angle_deg
                    != config_before.eikonal_conformal_max_angle_deg
                || self.config.eikonal_conform_bottom_surfaces
                    != config_before.eikonal_conform_bottom_surfaces
                || self.config.eikonal_conformal_bottom_max_angle_deg
                    != config_before.eikonal_conformal_bottom_max_angle_deg
                || self.config.order_field != config_before.order_field)
        {
            if let Some(render_state) = frame.wgpu_render_state() {
                self.reupload(&render_state.device);
            }
        }
    }

    /// Builds a `MeshSdf` from the selected object's mesh, samples it over
    /// the current `sdf_slice_plane`/`sdf_slice_offset`, and uploads the
    /// resulting grid as a heatmap texture into `sdf_slice_texture`.
    ///
    /// Recompute-on-demand only (called from the "Recompute Slice" button),
    /// never per-frame — matches `MESH_SDF_VISUALIZATION.md` Phase D.
    fn recompute_slice(&mut self, ctx: &egui::Context) {
        let Some(index) = self.selected else {
            self.sdf_error = Some("no object selected".to_string());
            return;
        };
        let Some(object) = self.objects.get(index) else {
            self.sdf_error = Some("selected object no longer exists".to_string());
            return;
        };

        let mesh = &object.mesh;
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
            .collect();
        let mut sdf = manifold_fidget::mesh_sdf::MeshSdf::new(mesh.vertices.clone(), faces);
        sdf.set_sign_method(self.sdf_sign_method);

        if sdf.is_empty() {
            self.sdf_error = Some("selected object's mesh has no triangles".to_string());
            return;
        }

        let (basis1, basis2, _normal) = self.sdf_slice_plane.basis();
        let origin = self.sdf_slice_plane.basis().2 * self.sdf_slice_offset;

        // Fixed extent/resolution: covers the mesh's bounding box
        // generously with a simple default rather than exposing more
        // controls in this pass.
        let (min, max) = mesh
            .bounding_box()
            .unwrap_or((glam::DVec3::ZERO, glam::DVec3::ONE));
        let extent = (max - min).max_element().max(1.0) * 1.5;
        const RESOLUTION: usize = 96;

        let grid = manifold_fidget::slice::sample_plane(
            &sdf, origin, basis1, basis2, extent, extent, RESOLUTION, RESOLUTION,
        );

        let color_image = slice_grid_to_color_image(&grid);
        let texture = ctx.load_texture("sdf_slice", color_image, egui::TextureOptions::LINEAR);

        self.sdf_slice = Some(grid);
        self.sdf_slice_texture = Some(texture);
        self.sdf_error = None;
    }

    /// SDF debug panel (Phase D, see MESH_SDF_VISUALIZATION.md): object
    /// picker (reflects `self.selected`), sign-method toggle, iso-level
    /// control, and a recompute trigger. Isosurface extraction is wired
    /// (subtask 08); slice sampling wiring (subtask 09) is still stubbed.
    fn sdf_panel(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        ui.heading("SDF");

        match self.selected {
            Some(index) => {
                let object = &self.objects[index];
                ui.label(format!(
                    "Object {} — {} triangles",
                    object.id.0,
                    object.mesh.triangle_count()
                ));
            }
            None => {
                ui.label("No object selected — select one in the Objects list.");
            }
        }

        ui.separator();
        ui.label("Sign method");
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.sdf_sign_method,
                manifold_fidget::mesh_sdf::SignMethod::Pseudonormal,
                "Pseudonormal",
            );
            ui.add_enabled(false, egui::RadioButton::new(false, "Winding number"))
                .on_disabled_hover_text("not yet implemented");
        });

        ui.separator();
        drag_num(
            ui,
            &mut self.sdf_iso_level,
            0.05,
            -f64::INFINITY..=f64::INFINITY,
            "Iso level (mm)",
        );

        ui.separator();
        if ui
            .add_enabled(self.selected.is_some(), egui::Button::new("Recompute"))
            .clicked()
        {
            let device = frame
                .wgpu_render_state()
                .expect("wgpu renderer is required")
                .device
                .clone();
            self.recompute_sdf(&device);
        }

        ui.separator();
        ui.heading("Slice view");
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("sdf_slice_plane")
                .selected_text(format!("{:?}", self.sdf_slice_plane))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sdf_slice_plane, SlicePlane::Xy, "XY");
                    ui.selectable_value(&mut self.sdf_slice_plane, SlicePlane::Xz, "XZ");
                    ui.selectable_value(&mut self.sdf_slice_plane, SlicePlane::Yz, "YZ");
                });
            drag_num(
                ui,
                &mut self.sdf_slice_offset,
                0.5,
                -f64::INFINITY..=f64::INFINITY,
                "Offset (mm)",
            );
        });
        if ui
            .add_enabled(
                self.selected.is_some(),
                egui::Button::new("Recompute Slice"),
            )
            .clicked()
        {
            self.recompute_slice(ui.ctx());
        }
        if let Some(texture) = &self.sdf_slice_texture {
            ui.add(egui::Image::new(texture).max_width(240.0));
        }

        if let Some(err) = &self.sdf_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("SDF failed: {err}"));
        }
    }

    /// Builds a `MeshSdf` from the selected object's mesh (in world space,
    /// with `object.transform` baked into the vertex positions so the
    /// extracted isosurface lines up with the already-transformed mesh
    /// rendered by `viewport()`), extracts the isosurface at
    /// `self.sdf_iso_level` via marching cubes, and uploads the result as a
    /// semi-transparent overlay. Recompute-on-demand only — never called
    /// per-frame (see `MESH_SDF_VISUALIZATION.md` Phase D). Sets
    /// `self.sdf_error` and clears any stale overlay/isosurface on failure
    /// instead of panicking.
    fn recompute_sdf(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        let Some(index) = self.selected else {
            self.sdf_error = Some("no object selected".to_string());
            return;
        };
        let object = &self.objects[index];
        let mesh = &object.mesh;

        let Some((local_min, local_max)) = mesh.bounding_box() else {
            self.sdf_error = Some("selected object has an empty mesh".to_string());
            self.sdf_isosurface = None;
            self.sdf_overlay_mesh = None;
            return;
        };

        let vertices: Vec<glam::DVec3> = mesh
            .vertices
            .iter()
            .map(|&v| object.transform.transform_point(v))
            .collect();
        let faces: Vec<[usize; 3]> = mesh
            .indices
            .chunks_exact(3)
            .map(|tri| [tri[0] as usize, tri[1] as usize, tri[2] as usize])
            .collect();

        let mut sdf = manifold_fidget::mesh_sdf::MeshSdf::new(vertices, faces);
        sdf.set_sign_method(self.sdf_sign_method);

        // Extraction box: the (transformed) mesh's bounding box, padded so
        // an iso-level offset outward from the surface is still captured.
        let (min, max) = (
            object.transform.transform_point(local_min),
            object.transform.transform_point(local_max),
        );
        let (min, max) = (min.min(max), min.max(max));
        let padding = glam::DVec3::splat(self.sdf_iso_level.abs() + 1.0);
        let (min, max) = (min - padding, max + padding);

        const RESOLUTION: usize = 48;
        let isosurface = manifold_fidget::marching_cubes::extract_isosurface(
            &sdf,
            min,
            max,
            RESOLUTION,
            self.sdf_iso_level,
        );

        if isosurface.is_empty() {
            self.sdf_error =
                Some("isosurface extraction produced no triangles at this iso level".to_string());
            self.sdf_isosurface = None;
            self.sdf_overlay_mesh = None;
            return;
        }

        self.sdf_overlay_mesh = Some(Arc::new(UploadedMesh::upload_from_vertices(
            device,
            &isosurface,
        )));
        self.sdf_isosurface = Some(isosurface);
        self.sdf_error = None;
    }

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("Import…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Mesh files", &["stl", "3mf"])
                    .pick_file()
                {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.import(&path, &device);
                }
            }
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("Remove"))
                .on_hover_text("Remove selected object (or press Delete)")
                .clicked()
            {
                if let Some(index) = self.selected {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.remove_object(index, &device);
                }
            }
            ui.label(format!("{} object(s) loaded", self.objects.len()));

            ensure_row_space(ui, 95.0);
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new("Drop to Bed"))
                .on_hover_text("Drop the selected object flush to the print bed")
                .clicked()
            {
                if let Some(index) = self.selected {
                    if let Some(object) = self.objects.get_mut(index) {
                        let (bed_min, _) = self.machine.build_volume.bounding_box();
                        object.transform =
                            crate::lay_on_face::drop_object_to_bed(object, bed_min.z);
                        let device = frame
                            .wgpu_render_state()
                            .expect("wgpu renderer is required")
                            .device
                            .clone();
                        self.update_camera_bounds();
                        self.reupload(&device);
                    }
                }
            }

            let lay_text = if self.lay_on_face_active {
                "Done Lay on Face"
            } else {
                "Lay on Face"
            };
            ensure_row_space(ui, 130.0);
            if ui
                .add_enabled(self.selected.is_some(), egui::Button::new(lay_text))
                .on_hover_text("Click a facet on the convex hull overlay to orient that face flat against the bed")
                .clicked()
            {
                self.lay_on_face_active = !self.lay_on_face_active;
                if !self.lay_on_face_active {
                    self.cached_hull = None;
                }
            }

            ui.separator();
            let slicing_in_progress = self.slicing.is_some();
            ensure_row_space(ui, 80.0);
            if ui
                .add_enabled(
                    !self.objects.is_empty() && !slicing_in_progress,
                    egui::Button::new("Slice"),
                )
                .clicked()
            {
                self.start_slice();
            }
            if slicing_in_progress {
                ensure_row_space(ui, 150.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    // `plan_toolpaths_with_progress` splits `0.0..=1.0` evenly
                    // between slicing (first half) and toolpath planning
                    // (second half, see `toolpath::plan_with_progress`) — swap
                    // the label at the midpoint so the bar doesn't look stuck
                    // once slicing itself finishes.
                    let stage = if self.slice_progress < 0.5 {
                        "Slicing"
                    } else {
                        "Planning toolpaths"
                    };
                    ui.add(
                        egui::ProgressBar::new(self.slice_progress as f32)
                            .desired_width(120.0)
                            .text(format!("{stage}… {:.0}%", self.slice_progress * 100.0)),
                    );
                });
            }
            ensure_row_space(ui, 80.0);
            if ui
                .add_enabled(self.gcode.is_some(), egui::Button::new("Export…"))
                .clicked()
            {
                let default_gcode_name = self
                    .objects
                    .first()
                    .and_then(|obj| obj.name.as_ref())
                    .map(|name| format!("{name}.gcode"))
                    .unwrap_or_else(|| "out.gcode".to_string());

                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Gcode", &["gcode"])
                    .set_file_name(&default_gcode_name)
                    .save_file()
                {
                    if let Some(gcode) = &self.gcode {
                        if let Err(error) = std::fs::write(&path, gcode) {
                            self.slice_error = Some(error.to_string());
                        }
                    }
                }
            }
            ensure_row_space(ui, 130.0);
            ui.checkbox(&mut self.show_toolpaths, "Show toolpaths");
            ensure_row_space(ui, 190.0);
            let mode_before = self.mesh_overlay_mode;
            egui::ComboBox::from_label("Mesh Overlay")
                .selected_text(match self.mesh_overlay_mode {
                    MeshOverlayMode::None => "None (Shaded)",
                    MeshOverlayMode::ConformalRegions => "Conformal & Seed Regions",
                    MeshOverlayMode::SurfaceOrder => "Surface Order Gradient",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.mesh_overlay_mode,
                        MeshOverlayMode::None,
                        "None (Shaded)",
                    );
                    ui.selectable_value(
                        &mut self.mesh_overlay_mode,
                        MeshOverlayMode::ConformalRegions,
                        "Conformal & Seed Regions",
                    );
                    ui.selectable_value(
                        &mut self.mesh_overlay_mode,
                        MeshOverlayMode::SurfaceOrder,
                        "Surface Order Gradient",
                    );
                });
            if self.mesh_overlay_mode != mode_before {
                let device = frame
                    .wgpu_render_state()
                    .expect("wgpu renderer is required")
                    .device
                    .clone();
                self.reupload(&device);
            }
            if let Some(stats) = &self.print_statistics {
                ensure_row_space(ui, 250.0);
                ui.separator();
                ui.label(format!(
                    "⏱ {}  |  🧵 {:.2} m ({:.1} g)  |  📦 {} layers",
                    stats.formatted_time(),
                    stats.filament_length_meters,
                    stats.filament_weight_grams,
                    stats.total_layers,
                ));
            }

            // Order-based scrub slider (Phase 13 subtask 05): ranged over
            // the min/max `order` value across all segments in the current
            // `toolpaths`, disabled when there's nothing to scrub. Dragging
            // triggers a CPU-side rebuild-on-change re-upload (see
            // `toolpath_view::build_toolpath_lines`'s doc comment for the
            // tradeoff versus a shader-side discard).
            let (slider_min, slider_max) = self.toolpath_order_range.unwrap_or((0.0, 0.0));
            let mut slider_value = self.scrub_order.min(slider_max).max(slider_min);
            ensure_row_space(ui, 240.0);
            let slider_response = ui.add_enabled(
                self.toolpaths.is_some(),
                egui::Slider::new(&mut slider_value, slider_min..=slider_max).text("Scrub order"),
            );
            if slider_response.changed() {
                self.scrub_order = slider_value;
                let device = frame
                    .wgpu_render_state()
                    .expect("wgpu renderer is required")
                    .device
                    .clone();
                self.reupload_toolpaths(&device);
            }
        });

        if self.selected.is_some()
            && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        {
            if let Some(index) = self.selected {
                let device = frame
                    .wgpu_render_state()
                    .expect("wgpu renderer is required")
                    .device
                    .clone();
                self.remove_object(index, &device);
            }
        }

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            // Single click handling (select or deselect without dragging)
            if response.clicked() && !self.gizmo.is_focused() && !self.lay_on_face_active {
                if let Some(cursor_pos) = response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.latest_pos()))
                {
                    let hit = self.cast_scene_ray(rect, cursor_pos);
                    match hit {
                        SceneRayHit::Object { index, .. } => {
                            self.selected = Some(index);
                        }
                        SceneRayHit::Bed(_) | SceneRayHit::Skybox(_) => {
                            // Rule 4: If an object is selected and it's a single click not on the gizmo rather than a drag, deselect the object.
                            self.selected = None;
                            self.lay_on_face_active = false;
                            self.cached_hull = None;
                        }
                    }
                }
            }

            if response.drag_started() && !self.gizmo.is_focused() {
                if let Some(cursor_pos) = response
                    .interact_pointer_pos()
                    .or_else(|| ui.input(|i| i.pointer.latest_pos()))
                {
                    let hit = self.cast_scene_ray(rect, cursor_pos);
                    let is_secondary = ui.input(|i| i.pointer.secondary_down());
                    let is_middle = ui.input(|i| i.pointer.middle_down());

                    if is_middle || ui.input(|i| i.modifiers.shift) {
                        let pivot = match hit {
                            SceneRayHit::Object { point, .. } => point,
                            SceneRayHit::Bed(_) | SceneRayHit::Skybox(_) => self.camera.eye(),
                        };
                        self.drag_pivot = Some(pivot);
                        self.active_drag = None;
                    } else if is_secondary {
                        let point = match hit {
                            SceneRayHit::Object { point, .. } => point,
                            SceneRayHit::Bed(point) => point,
                            SceneRayHit::Skybox(_) => self.camera.target,
                        };
                        let forward = (self.camera.target - self.camera.eye()).normalize_or_zero();
                        let depth = (point - self.camera.eye()).dot(forward).abs();
                        self.drag_depth =
                            depth.clamp(self.camera.min_distance, self.camera.max_distance);
                        self.active_drag = Some(DragMode::ViewPan);
                    } else {
                        // Left click (Primary button)
                        match hit {
                            SceneRayHit::Object { index, .. } => {
                                // Rule 2: If the click is on an object, select it and drag it in the XY plane.
                                self.selected = Some(index);
                                if let Some(object) = self.objects.get(index) {
                                    let (_, _, trans) =
                                        object.transform.0.to_scale_rotation_translation();
                                    let start_plane = intersect_horizontal_plane(
                                        &self.camera,
                                        rect,
                                        cursor_pos,
                                        trans.z,
                                    );
                                    self.active_drag = Some(DragMode::ObjectXy {
                                        index,
                                        init_translation: trans,
                                        start_plane_pos: start_plane,
                                    });
                                }
                            }
                            SceneRayHit::Bed(p) | SceneRayHit::Skybox(p) => {
                                if let Some(selected_index) = self.selected {
                                    if let Some(object) = self.objects.get(selected_index) {
                                        // Rule 3: If an object is selected and not on the gizmo, drag it in the XY plane.
                                        let (_, _, trans) =
                                            object.transform.0.to_scale_rotation_translation();
                                        let start_plane = intersect_horizontal_plane(
                                            &self.camera,
                                            rect,
                                            cursor_pos,
                                            trans.z,
                                        );
                                        self.active_drag = Some(DragMode::ObjectXy {
                                            index: selected_index,
                                            init_translation: trans,
                                            start_plane_pos: start_plane,
                                        });
                                    } else {
                                        let forward = (self.camera.target - self.camera.eye())
                                            .normalize_or_zero();
                                        let depth = (p - self.camera.eye()).dot(forward).abs();
                                        self.drag_depth = depth.clamp(
                                            self.camera.min_distance,
                                            self.camera.max_distance,
                                        );
                                        self.active_drag = Some(DragMode::ViewPan);
                                    }
                                } else {
                                    // Rule 1: If no object selected and the click is not on an object, same as right click (i.e. drag the view).
                                    let forward = (self.camera.target - self.camera.eye())
                                        .normalize_or_zero();
                                    let depth = (p - self.camera.eye()).dot(forward).abs();
                                    self.drag_depth = depth
                                        .clamp(self.camera.min_distance, self.camera.max_distance);
                                    self.active_drag = Some(DragMode::ViewPan);
                                }
                            }
                        }
                    }
                }
            }

            if response.dragged() {
                let delta = response.drag_delta();
                if ui.input(|i| i.pointer.middle_down() || i.modifiers.shift) {
                    let pivot = self.drag_pivot.unwrap_or(self.camera.target);
                    self.camera.orbit_around(pivot, delta.x, delta.y);
                } else if let Some(active) = self.active_drag {
                    match active {
                        DragMode::ViewPan => {
                            let depth = if self.drag_depth > 0.0 {
                                self.drag_depth
                            } else {
                                self.camera.distance
                            };
                            self.camera
                                .pan_with_depth(delta.x, delta.y, rect.height(), depth);
                        }
                        DragMode::ObjectXy {
                            index,
                            init_translation,
                            start_plane_pos,
                        } => {
                            if let Some(cursor_pos) = ui.input(|i| i.pointer.latest_pos()) {
                                if let Some(object) = self.objects.get_mut(index) {
                                    let curr_plane = intersect_horizontal_plane(
                                        &self.camera,
                                        rect,
                                        cursor_pos,
                                        init_translation.z,
                                    );
                                    let delta_x = curr_plane.x - start_plane_pos.x;
                                    let delta_y = curr_plane.y - start_plane_pos.y;
                                    let (scale, rotation, _) =
                                        object.transform.0.to_scale_rotation_translation();
                                    object.transform = Transform::from_scale_rotation_translation(
                                        scale,
                                        rotation,
                                        DVec3::new(
                                            init_translation.x + delta_x,
                                            init_translation.y + delta_y,
                                            init_translation.z,
                                        ),
                                    );
                                    let device = frame
                                        .wgpu_render_state()
                                        .expect("wgpu renderer is required")
                                        .device
                                        .clone();
                                    self.update_camera_bounds();
                                    self.reupload(&device);
                                }
                            }
                        }
                    }
                }
            }

            if response.drag_stopped() {
                self.drag_pivot = None;
                self.drag_depth = 0.0;
                self.active_drag = None;
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.zoom(scroll);
                }
            }

            let aspect_ratio = rect.width() / rect.height().max(1.0);
            let view_proj = self.camera.view_proj(aspect_ratio);

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    Viewport3dCallback {
                        rect,
                        view_proj,
                        scene: self.uploaded_scene.clone(),
                        meshes: self.uploaded_meshes.clone(),
                        overlay: self.sdf_overlay_mesh.clone(),
                        toolpaths: if self.show_toolpaths {
                            self.uploaded_toolpaths.clone()
                        } else {
                            None
                        },
                    },
                ));

            // Hover tooltip (Phase 13 subtask 06): CPU-side O(n) nearest-
            // segment picking over the currently visible (scrub-filtered)
            // segment set, reusing `world_to_screen` (documented after this
            // impl block) to project each segment's endpoints into the same
            // screen space as the cursor. Only active when toolpaths are
            // shown and present.
            if self.show_toolpaths {
                if let Some(toolpaths) = &self.toolpaths {
                    if let Some(hover_pos) = response.hover_pos() {
                        const PICK_THRESHOLD_PX: f32 = 8.0;
                        let mut nearest: Option<(
                            f32,
                            &manifold_core::toolpath::Segment,
                            glam::DVec3,
                            glam::DVec3,
                            usize,
                            usize,
                        )> = None;
                        for (path_idx, path) in toolpaths.iter().enumerate() {
                            let count = path.points.len();
                            for i in 0..path.segments.len() {
                                let segment = &path.segments[i];
                                if segment.order > self.scrub_order {
                                    continue;
                                }
                                let a = path.points[i];
                                let b = path.points[(i + 1) % count];
                                let (Some(screen_a), Some(screen_b)) = (
                                    world_to_screen(view_proj, rect, a),
                                    world_to_screen(view_proj, rect, b),
                                ) else {
                                    continue;
                                };
                                let dist = point_segment_distance(hover_pos, screen_a, screen_b);
                                if nearest
                                    .as_ref()
                                    .is_none_or(|(best_dist, _, _, _, _, _)| dist < *best_dist)
                                {
                                    nearest = Some((dist, segment, a, b, path_idx, i));
                                }
                            }
                        }
                        if let Some((dist, segment, a, b, path_idx, seg_idx)) = nearest {
                            if dist <= PICK_THRESHOLD_PX {
                                response.clone().show_tooltip_ui(|ui| {
                                    ui.heading(format!(
                                        "Extrusion #{} (Path #{}, Seg #{})",
                                        segment.id, path_idx, seg_idx
                                    ));
                                    if segment.kind
                                        == manifold_core::toolpath::MoveKind::DebugExcluded
                                    {
                                        ui.colored_label(
                                            egui::Color32::from_rgb(255, 30, 230),
                                            "DEBUG / EXCLUDED (Not in G-code)",
                                        );
                                    }
                                    ui.label(format!("kind: {:?}", segment.kind));
                                    ui.label(format!(
                                        "speed (cmd): {:.1} mm/s",
                                        segment.speed / 60.0
                                    ));
                                    let actual_speed = toolpath_view::segment_scalar_value(
                                        segment,
                                        a,
                                        b,
                                        ToolpathDataView::ActualSpeed,
                                        &self.config,
                                        Some(&self.machine),
                                    );
                                    ui.label(format!("speed (actual): {:.1} mm/s", actual_speed));

                                    let length = (b - a).length();
                                    let duration = if actual_speed > 1e-4 {
                                        length / actual_speed
                                    } else {
                                        let speed_mm_s = (segment.speed / 60.0).max(1e-3);
                                        length / speed_mm_s
                                    };
                                    if duration < 0.1 {
                                        ui.label(format!("duration: {:.1} ms", duration * 1000.0));
                                    } else {
                                        ui.label(format!("duration: {:.3} s", duration));
                                    }

                                    let flow = toolpath_view::segment_scalar_value(
                                        segment,
                                        a,
                                        b,
                                        ToolpathDataView::FlowRate,
                                        &self.config,
                                        Some(&self.machine),
                                    );
                                    if segment.kind != manifold_core::toolpath::MoveKind::Travel {
                                        ui.label(format!("flow rate: {:.2} mm³/s", flow));
                                    }
                                    let accel = toolpath_view::segment_scalar_value(
                                        segment,
                                        a,
                                        b,
                                        ToolpathDataView::Acceleration,
                                        &self.config,
                                        Some(&self.machine),
                                    );
                                    ui.label(format!("accel (cmd): {:.0} mm/s²", accel));
                                    let actual_accel = toolpath_view::segment_scalar_value(
                                        segment,
                                        a,
                                        b,
                                        ToolpathDataView::ActualAcceleration,
                                        &self.config,
                                        Some(&self.machine),
                                    );
                                    ui.label(format!("accel (actual): {:.0} mm/s²", actual_accel));
                                    ui.label(format!(
                                        "extrusion_rate: {:.3}",
                                        segment.extrusion_rate
                                    ));
                                    ui.label(format!("order: {:.3}", segment.order));
                                });
                            }
                        }
                    }
                }
            }

            // Gizmo paints as plain egui geometry into this `Ui`'s layer, so
            // it must run after the wgpu scene/mesh paint callbacks above to
            // composite on top of them (see ROADMAP.md Phase 7).
            if let Some(index) = self.selected {
                if !self.lay_on_face_active {
                    if let Some(object) = self.objects.get(index) {
                        self.gizmo.update_config(GizmoConfig {
                            view_matrix: self.camera.view_matrix_f64().into(),
                            projection_matrix: self
                                .camera
                                .projection_matrix_f64(aspect_ratio)
                                .into(),
                            viewport: rect,
                            modes: GizmoMode::all(),
                            orientation: GizmoOrientation::Local,
                            ..Default::default()
                        });

                        let (scale, rotation, translation) =
                            object.transform.0.to_scale_rotation_translation();
                        let gizmo_transform = GizmoTransform::from_scale_rotation_translation(
                            scale,
                            rotation,
                            translation,
                        );

                        let allow_gizmo = self.active_drag.is_none();
                        if let Some((_, mut new_transforms)) = self.gizmo_interact(
                            ui,
                            rect,
                            view_proj,
                            translation,
                            &[gizmo_transform],
                            allow_gizmo,
                        ) {
                            if let Some(new_transform) = new_transforms.pop() {
                                let scale: mint::Vector3<f64> = new_transform.scale;
                                let rotation: mint::Quaternion<f64> = new_transform.rotation;
                                let translation: mint::Vector3<f64> = new_transform.translation;
                                self.objects[index].transform =
                                    Transform::from_scale_rotation_translation(
                                        scale.into(),
                                        rotation.into(),
                                        translation.into(),
                                    );

                                let device = frame
                                    .wgpu_render_state()
                                    .expect("wgpu renderer is required")
                                    .device
                                    .clone();
                                self.update_camera_bounds();
                                self.reupload(&device);
                            }
                        }
                    }
                }
            }

            if self.lay_on_face_active {
                if let Some(index) = self.selected {
                    if self.cached_hull.as_ref().map(|(idx, _)| *idx) != Some(index) {
                        if let Some(object) = self.objects.get(index) {
                            if let Some(hull) =
                                manifold_core::convex_hull::compute_simplified_convex_hull(
                                    &object.mesh.vertices,
                                    manifold_core::convex_hull::DEFAULT_MAX_FACETS,
                                    manifold_core::convex_hull::DEFAULT_VOLUME_THRESHOLD_RATIO,
                                )
                            {
                                self.cached_hull = Some((index, hull));
                            }
                        }
                    }

                    let hovered_facet = if let Some((_, hull)) = &self.cached_hull {
                        if let Some(cursor_pos) = response.hover_pos() {
                            let (ray_orig, ray_dir) = self.camera.unproject_ray(rect, cursor_pos);
                            if let Some(object) = self.objects.get(index) {
                                crate::lay_on_face::pick_hull_facet(hull, object, ray_orig, ray_dir)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if response.clicked() {
                        if let Some(facet_idx) = hovered_facet {
                            if let Some((_, hull)) = &self.cached_hull {
                                if let Some(object) = self.objects.get_mut(index) {
                                    let (bed_min, _) = self.machine.build_volume.bounding_box();
                                    let facet = &hull.facets[facet_idx];
                                    object.transform = crate::lay_on_face::orient_facet_to_bed(
                                        object, facet, bed_min.z,
                                    );
                                    let device = frame
                                        .wgpu_render_state()
                                        .expect("wgpu renderer is required")
                                        .device
                                        .clone();
                                    self.update_camera_bounds();
                                    self.reupload(&device);
                                    self.lay_on_face_active = false;
                                    self.cached_hull = None;
                                }
                            }
                        } else {
                            // Clicked off the object: exit lay on face
                            self.lay_on_face_active = false;
                            self.cached_hull = None;
                        }
                    }

                    if let Some((_, hull)) = &self.cached_hull {
                        if let Some(object) = self.objects.get(index) {
                            crate::lay_on_face::render_hull_overlay(
                                ui.painter(),
                                hull,
                                object,
                                view_proj,
                                rect,
                                self.camera.eye(),
                                hovered_facet,
                            );
                        }
                    }
                } else {
                    self.lay_on_face_active = false;
                    self.cached_hull = None;
                }

                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.lay_on_face_active = false;
                    self.cached_hull = None;
                }
            }

            // Floating top-right viewport legend overlay (extensible for future data views)
            if self.show_toolpaths && self.toolpaths.is_some() {
                let margin = 12.0;
                let legend_width = 155.0;
                let legend_pos =
                    egui::pos2(rect.max.x - legend_width - margin, rect.min.y + margin);

                egui::Area::new(egui::Id::new("toolpath_viewport_legend"))
                    .fixed_pos(legend_pos)
                    .order(egui::Order::Foreground)
                    .show(ui.ctx(), |ui| {
                        egui::Frame::window(ui.style())
                            .fill(egui::Color32::from_black_alpha(205))
                            .stroke(egui::Stroke::new(
                                1.0_f32,
                                egui::Color32::from_white_alpha(35),
                            ))
                            .rounding(egui::Rounding::same(6.0))
                            .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                            .show(ui, |ui| {
                                ui.set_width(legend_width - 16.0);
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("Data:").strong().size(11.0));
                                    let prev_view = self.toolpath_data_view;
                                    egui::ComboBox::from_id_salt("toolpath_data_view_combo")
                                        .selected_text(
                                            egui::RichText::new(self.toolpath_data_view.name())
                                                .size(11.0),
                                        )
                                        .width(92.0)
                                        .show_ui(ui, |ui| {
                                            for view in [
                                                ToolpathDataView::LineType,
                                                ToolpathDataView::Speed,
                                                ToolpathDataView::ActualSpeed,
                                                ToolpathDataView::FlowRate,
                                                ToolpathDataView::Acceleration,
                                                ToolpathDataView::ActualAcceleration,
                                                ToolpathDataView::TravelDurations,
                                            ] {
                                                ui.selectable_value(
                                                    &mut self.toolpath_data_view,
                                                    view,
                                                    view.name(),
                                                );
                                            }
                                        });
                                    if self.toolpath_data_view != prev_view {
                                        let device = frame
                                            .wgpu_render_state()
                                            .expect("wgpu renderer is required")
                                            .device
                                            .clone();
                                        self.reupload_toolpaths(&device);
                                    }
                                });
                                ui.add_space(3.0);
                                match self.toolpath_data_view {
                                    ToolpathDataView::LineType => {
                                        let mut toggled = false;
                                        for entry in toolpath_view::line_type_legend() {
                                            let is_hidden =
                                                self.hidden_line_types.contains(&entry.key);
                                            ui.horizontal(|ui| {
                                                let (badge_rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                let [r, g, b, a] = entry.color;
                                                let color = if is_hidden {
                                                    egui::Color32::from_rgba_unmultiplied(
                                                        (r * 80.0).round() as u8,
                                                        (g * 80.0).round() as u8,
                                                        (b * 80.0).round() as u8,
                                                        70,
                                                    )
                                                } else {
                                                    egui::Color32::from_rgba_unmultiplied(
                                                        (r * 255.0).round() as u8,
                                                        (g * 255.0).round() as u8,
                                                        (b * 255.0).round() as u8,
                                                        (a * 255.0).round() as u8,
                                                    )
                                                };
                                                ui.painter().rect_filled(badge_rect, 2.0, color);
                                                let text = if is_hidden {
                                                    egui::RichText::new(entry.label)
                                                        .size(11.0)
                                                        .color(egui::Color32::from_gray(120))
                                                        .strikethrough()
                                                } else {
                                                    egui::RichText::new(entry.label).size(11.0)
                                                };
                                                if ui
                                                    .selectable_label(!is_hidden, text)
                                                    .on_hover_text(if is_hidden {
                                                        "Click to show this line type"
                                                    } else {
                                                        "Click to hide this line type"
                                                    })
                                                    .clicked()
                                                {
                                                    if is_hidden {
                                                        self.hidden_line_types.remove(&entry.key);
                                                    } else {
                                                        self.hidden_line_types.insert(entry.key);
                                                    }
                                                    toggled = true;
                                                }
                                            });
                                        }
                                        if toggled {
                                            if let Some(render_state) = frame.wgpu_render_state() {
                                                let device = render_state.device.clone();
                                                self.reupload_toolpaths(&device);
                                            }
                                        }
                                    }
                                    view => {
                                        let unit = view.unit();
                                        let paths = self.toolpaths.as_deref().unwrap_or_default();
                                        let (min_val, max_val) = toolpath_view::data_view_range(
                                            paths,
                                            view,
                                            &self.config,
                                            Some(&self.machine),
                                        )
                                        .unwrap_or((0.0, 1.0));

                                        let stops = [1.00, 0.75, 0.50, 0.25, 0.00];
                                        for &t in &stops {
                                            let val = min_val + t * (max_val - min_val);
                                            let [r, g, b, a] = toolpath_view::scalar_to_color(t);
                                            let color = egui::Color32::from_rgba_unmultiplied(
                                                (r * 255.0).round() as u8,
                                                (g * 255.0).round() as u8,
                                                (b * 255.0).round() as u8,
                                                (a * 255.0).round() as u8,
                                            );
                                            ui.horizontal(|ui| {
                                                let (badge_rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                ui.painter().rect_filled(badge_rect, 2.0, color);
                                                let label_str =
                                                    if view == ToolpathDataView::TravelDurations {
                                                        if max_val < 0.1 {
                                                            format!("{:.1} ms", val * 1000.0)
                                                        } else {
                                                            format!("{val:.3} s")
                                                        }
                                                    } else if (max_val - min_val).abs() > 10.0 {
                                                        format!("{val:.0} {unit}")
                                                    } else {
                                                        format!("{val:.2} {unit}")
                                                    };
                                                ui.label(egui::RichText::new(label_str).size(11.0));
                                            });
                                        }

                                        if view == ToolpathDataView::FlowRate
                                            || view == ToolpathDataView::Speed
                                            || view == ToolpathDataView::ActualSpeed
                                            || view == ToolpathDataView::Acceleration
                                            || view == ToolpathDataView::ActualAcceleration
                                        {
                                            ui.horizontal(|ui| {
                                                let (badge_rect, _) = ui.allocate_exact_size(
                                                    egui::vec2(10.0, 10.0),
                                                    egui::Sense::hover(),
                                                );
                                                let [r, g, b, a] = toolpath_view::COLOR_TRAVEL;
                                                let color = egui::Color32::from_rgba_unmultiplied(
                                                    (r * 255.0).round() as u8,
                                                    (g * 255.0).round() as u8,
                                                    (b * 255.0).round() as u8,
                                                    (a * 255.0).round() as u8,
                                                );
                                                ui.painter().rect_filled(badge_rect, 2.0, color);
                                                ui.label(egui::RichText::new("Travel").size(11.0));
                                            });
                                        }
                                    }
                                }
                            });
                    });
            }
        });
    }
}

/// Wrap to the next line in a horizontal layout if the remaining available width
/// is less than `needed_width` and at least one item has already been placed on this row.
fn ensure_row_space(ui: &mut egui::Ui, needed_width: f32) {
    let row_used = ui.cursor().min.x - ui.max_rect().min.x;
    if row_used > 5.0 && ui.available_width() < needed_width {
        ui.end_row();
    }
}

/// Ray intersection result against the 3D scene.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SceneRayHit {
    /// Hit a loaded object mesh in the scene at `index`.
    Object { index: usize, point: DVec3 },
    /// Hit the print bed.
    Bed(DVec3),
    /// Missed all surfaces and intersected the invisible enclosing skybox.
    Skybox(DVec3),
}

/// Active drag interaction mode for the viewport canvas.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DragMode {
    /// Dragging the camera view (1:1 pan).
    ViewPan,
    /// Dragging an object constrained to the horizontal XY plane.
    ObjectXy {
        index: usize,
        init_translation: DVec3,
        start_plane_pos: DVec3,
    },
}

/// Intersects a camera ray through `cursor_pos` with a horizontal plane $Z = \text{plane\_z}$.
fn intersect_horizontal_plane(
    camera: &OrbitCamera,
    rect: egui::Rect,
    cursor_pos: egui::Pos2,
    plane_z: f64,
) -> DVec3 {
    let (ray_orig, ray_dir) = camera.unproject_ray(rect, cursor_pos);
    if ray_dir.z.abs() > 1e-9 {
        let t = (plane_z - ray_orig.z) / ray_dir.z;
        if t > 0.0 {
            return ray_orig + ray_dir * t;
        }
    }
    let p = ray_orig + ray_dir * camera.distance;
    DVec3::new(p.x, p.y, plane_z)
}

/// Slab-based ray-AABB intersection test returning the nearest $t \ge 0$.
fn ray_aabb_intersect(orig: DVec3, dir: DVec3, min: DVec3, max: DVec3) -> Option<f64> {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;

    for i in 0..3 {
        let (o, d, bmin, bmax) = match i {
            0 => (orig.x, dir.x, min.x, max.x),
            1 => (orig.y, dir.y, min.y, max.y),
            _ => (orig.z, dir.z, min.z, max.z),
        };
        if d.abs() < 1e-12 {
            if o < bmin || o > bmax {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (bmin - o) * inv_d;
            let mut t2 = (bmax - o) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }
    if tmax < 1e-6 {
        return None;
    }
    Some(tmin.max(0.0))
}

/// Möller–Trumbore ray-triangle intersection test returning $t > 0$.
fn ray_triangle_intersect(orig: DVec3, dir: DVec3, v0: DVec3, v1: DVec3, v2: DVec3) -> Option<f64> {
    const EPS: f64 = 1e-9;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let det = edge1.dot(h);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = orig - v0;
    let u = s.dot(h) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(edge1);
    let v = dir.dot(q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = edge2.dot(q) * inv_det;
    if t > EPS {
        Some(t)
    } else {
        None
    }
}

/// Projects a world-space point to screen-space pixel coordinates within
/// `rect`, given a camera `view_proj` matrix. Returns `None` if the point
/// projects behind the camera (`w <= 0`).
fn world_to_screen(
    view_proj: glam::Mat4,
    rect: egui::Rect,
    point: glam::DVec3,
) -> Option<egui::Pos2> {
    let clip = view_proj * point.as_vec3().extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(egui::Pos2::new(
        rect.min.x + (ndc.x * 0.5 + 0.5) * rect.width(),
        rect.min.y + (1.0 - (ndc.y * 0.5 + 0.5)) * rect.height(),
    ))
}

/// Distance in screen-space pixels from `point` to the line segment
/// `a`-`b`, used by the hover-tooltip nearest-segment scan in `viewport()`.
fn point_segment_distance(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= f32::EPSILON {
        return (point - a).length();
    }
    let t = ((point - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let closest = a + ab * t;
    (point - closest).length()
}

/// Converts a slice heatmap grid to an `egui::ColorImage` using a simple
/// blue-white-red diverging colormap centered at zero: negative values
/// (inside the surface) shade toward blue, positive (outside) toward red,
/// and values near zero (the surface itself) are white. Scaled by the
/// grid's own max absolute value so the colormap always spans the full
/// range of the current slice.
fn slice_grid_to_color_image(grid: &manifold_fidget::slice::SliceGrid) -> egui::ColorImage {
    let max_abs = grid
        .values
        .iter()
        .fold(0.0_f32, |acc, v| acc.max(v.abs()))
        .max(f32::EPSILON);

    let pixels: Vec<egui::Color32> = grid
        .values
        .iter()
        .map(|&v| {
            let t = (v / max_abs).clamp(-1.0, 1.0);
            if t < 0.0 {
                // Inside: blend white -> blue.
                let f = -t;
                let c = (255.0 * (1.0 - f)) as u8;
                egui::Color32::from_rgb(c, c, 255)
            } else {
                // Outside: blend white -> red.
                let f = t;
                let c = (255.0 * (1.0 - f)) as u8;
                egui::Color32::from_rgb(255, c, c)
            }
        })
        .collect();

    egui::ColorImage {
        size: [grid.width, grid.height],
        pixels,
    }
}

/// Extracts a human-readable message from a thread panic payload (as
/// returned by `std::thread::JoinHandle::join`'s `Err` variant), used by
/// `ManifoldApp::drain_slice_messages` to surface *why* the background
/// slicing thread died instead of just reporting that it did. Panic
/// payloads are almost always a `&'static str` (from `panic!("literal")`)
/// or a `String` (from `panic!("{}", ...)`/`.expect(...)`/`.unwrap()`'s
/// formatted message) — anything else falls back to a generic message
/// rather than failing to report at all.
fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// A placeholder 200x200x200mm three-axis machine with a single tool at
/// the origin, used until machine configuration is loaded from a project
/// file (see ROADMAP.md).
fn default_machine() -> Machine {
    Machine::new(
        BoundingVolume::Aabb {
            min: glam::DVec3::ZERO,
            max: glam::DVec3::new(200.0, 200.0, 200.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    )
}

/// Human-readable display label for an [`InfillPatternKind`].
fn infill_pattern_label(kind: InfillPatternKind) -> &'static str {
    match kind {
        InfillPatternKind::Cubic => "Cubic",
        InfillPatternKind::Gyroid => "Gyroid",
        InfillPatternKind::SchwarzD => "Schwarz Diamond (D)",
        InfillPatternKind::SchwarzP => "Schwarz Primitive (P)",
        InfillPatternKind::Monotonic => "Monotonic",
        InfillPatternKind::Concentric => "Concentric",
        InfillPatternKind::AllWalls => "All Walls",
        InfillPatternKind::None => "None",
    }
}

impl eframe::App for ManifoldApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(feature = "mcp-server")]
        self.drain_mcp_commands(frame);

        if self.slicing.is_some() {
            if self.drain_slice_messages() {
                if self.toolpaths.is_some() {
                    let device = frame
                        .wgpu_render_state()
                        .expect("wgpu renderer is required")
                        .device
                        .clone();
                    self.reupload_toolpaths(&device);
                }
            } else {
                // Still in progress: keep polling every frame rather than
                // waiting for the next input-driven repaint.
                ctx.request_repaint();
            }
        }

        egui::SidePanel::left("settings_panel")
            .default_width(260.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.settings_panel(ui, frame));
            });

        if self.show_sdf_panel {
            egui::SidePanel::right("sdf_panel")
                .default_width(260.0)
                .show(ctx, |ui| self.sdf_panel(ui, frame));
        }

        egui::CentralPanel::default().show(ctx, |ui| self.viewport(ui, frame));
    }
}

/// Load every object from `path`, dispatching on its file extension.
///
/// All loaded objects are assigned to [`ToolId(0)`] and IDs allocated from
/// `next_object_id`, mirroring `manifold-cli`'s `load_objects` until
/// per-file tool assignment (Phase 8, see ROADMAP.md) is wired up.
fn load_objects(path: &Path, next_object_id: &mut u32) -> anyhow::Result<Vec<Object>> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("object")
        .to_string();

    match extension.as_str() {
        "3mf" => {
            let file = File::open(path)?;
            let mut objects = threemf::load_3mf(file, ToolId(0))?;
            let multiple = objects.len() > 1;
            for (idx, object) in objects.iter_mut().enumerate() {
                object.id = ObjectId(*next_object_id);
                *next_object_id += 1;
                if object.name.is_none() {
                    object.name = Some(if multiple {
                        format!("{} #{}", stem, idx + 1)
                    } else {
                        stem.clone()
                    });
                }
            }
            Ok(objects)
        }
        "stl" => {
            let file = File::open(path)?;
            let mesh: Mesh = stl::load_stl(BufReader::new(file))?;
            let id = ObjectId(*next_object_id);
            *next_object_id += 1;
            let mut obj = Object::new(id, mesh, ToolId(0));
            obj.name = Some(stem);
            Ok(vec![obj])
        }
        other => anyhow::bail!(
            "unsupported input format {:?} for {}: only .3mf and .stl are supported today",
            other,
            path.display()
        ),
    }
}
