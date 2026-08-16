//! `ManifoldApp`: GUI shell — left settings panel, central 3D viewport with
//! an in-panel import toolbar (Phase 4, see ROADMAP.md).

use crate::camera::OrbitCamera;
use crate::profile::Profile;
use crate::render::{
    MeshPaintCallback, MeshRenderResources, OverlayPaintCallback, ScenePaintCallback,
    ToolpathPaintCallback, UploadedMesh, UploadedScene, UploadedToolpaths,
};
use crate::scene;
use crate::toolpath_view;
use eframe::egui;
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
    /// Gcode from the last successful "Slice" action (Phase 8, see
    /// ROADMAP.md), previewed in the settings panel and written out by
    /// "Export…".
    gcode: Option<String>,
    /// Planned toolpaths from the last successful "Slice" action (Phase 13,
    /// see ROADMAP.md), previewed in the 3D viewport when `show_toolpaths`
    /// is enabled.
    toolpaths: Option<Vec<manifold_core::toolpath::Path>>,
    /// GPU-uploaded copy of `toolpaths`, rebuilt whenever `toolpaths`
    /// changes, same pattern as `uploaded_meshes`/`uploaded_scene`.
    uploaded_toolpaths: Option<Arc<UploadedToolpaths>>,
    /// Whether the toolpath preview line geometry is drawn in the viewport
    /// (Phase 13, see ROADMAP.md).
    show_toolpaths: bool,
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
    /// `0.0..=1.0` progress of the in-progress slice, reported by
    /// `manifold_core::plan_toolpaths_with_progress`. Only meaningful while
    /// `slicing` is `Some`.
    slice_progress: f64,
    /// Set by a failed "Save Profile…"/"Load Profile…" action (Phase 10, see
    /// ROADMAP.md).
    profile_error: Option<String>,
    next_tool_id: u32,
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
            gcode: None,
            toolpaths: None,
            uploaded_toolpaths: None,
            show_toolpaths: true,
            scrub_order: f64::INFINITY,
            toolpath_order_range: None,
            slice_error: None,
            slicing: None,
            slice_progress: 0.0,
            profile_error: None,
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
                crate::mcp::Command::SetTransform { index, x, y, z } => {
                    if let Some(object) = self.objects.get_mut(index) {
                        let (scale, rotation, _) =
                            object.transform.0.to_scale_rotation_translation();
                        object.transform = Transform::from_scale_rotation_translation(
                            scale,
                            rotation,
                            glam::DVec3::new(x, y, z),
                        );
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

    /// Load every object from `path`, dispatching on its file extension
    /// (mirrors `manifold-cli`'s `load_objects`).
    fn import(&mut self, path: &Path, device: &eframe::egui_wgpu::wgpu::Device) {
        match load_objects(path, &mut self.next_object_id) {
            Ok(mut new_objects) => {
                object::center_on_bed(&mut new_objects, &self.machine.build_volume);
                self.objects.append(&mut new_objects);
                self.reupload(device);
                self.import_error = None;
            }
            Err(err) => self.import_error = Some(err.to_string()),
        }
    }

    fn reupload(&mut self, device: &eframe::egui_wgpu::wgpu::Device) {
        let uploaded = self
            .objects
            .iter()
            .map(|object| UploadedMesh::upload(device, &object.mesh, &object.transform))
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
            let vertices = toolpath_view::build_toolpath_lines(paths, self.scrub_order);
            Arc::new(UploadedToolpaths::upload(device, &vertices))
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
        let hovered = ui.rect_contains_pointer(rect) && near_gizmo;

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
        std::thread::spawn(move || {
            let mut on_progress = move |fraction: f64| {
                let _ = progress_tx.send(SliceMessage::Progress(fraction));
            };
            let result = manifold_core::plan_toolpaths_with_progress(&workspace, &mut on_progress)
                .map_err(|error| error.to_string());
            let _ = tx.send(SliceMessage::Done(result));
        });
        self.slicing = Some(rx);
        self.slice_progress = 0.0;
        self.slice_error = None;
    }

    /// Drains any pending messages from the background slicing thread
    /// started by `start_slice`, updating `slice_progress` and finalizing
    /// the result via `finish_slice` once `Done` arrives. Returns `true` if
    /// slicing just finished this call (so the caller can perform any
    /// follow-up GPU work that needs a `wgpu::Device`, e.g.
    /// `reupload_toolpaths`).
    fn drain_slice_messages(&mut self) -> bool {
        let mut finished_result = None;
        if let Some(rx) = &self.slicing {
            while let Ok(message) = rx.try_recv() {
                match message {
                    SliceMessage::Progress(fraction) => self.slice_progress = fraction,
                    SliceMessage::Done(result) => finished_result = Some(result),
                }
            }
        }
        match finished_result {
            Some(result) => {
                self.slicing = None;
                self.finish_slice(result);
                true
            }
            None => false,
        }
    }

    /// Store the result of a finished slice (or error) for preview/export
    /// (Phase 8, see ROADMAP.md). Shared by the synchronous finalization
    /// path in `drain_slice_messages`.
    fn finish_slice(&mut self, result: Result<Vec<manifold_core::toolpath::Path>, String>) {
        match result {
            Ok(paths) => {
                let gcode = manifold_core::gcode::emit(&paths, &self.config);
                self.toolpath_order_range = toolpath_view::order_range(&paths);
                // Default the scrub slider to the max order so a fresh
                // slice shows every segment ("up to and including" the
                // top of the range).
                self.scrub_order = self
                    .toolpath_order_range
                    .map_or(f64::INFINITY, |(_, max)| max);
                self.uploaded_toolpaths = None;
                self.toolpaths = Some(paths);
                self.gcode = Some(gcode);
                self.slice_error = None;
            }
            Err(error) => {
                self.toolpaths = None;
                self.uploaded_toolpaths = None;
                self.toolpath_order_range = None;
                self.gcode = None;
                self.slice_error = Some(error);
            }
        }
    }

    fn settings_panel(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        ui.heading("Settings");
        ui.add(
            egui::Slider::new(&mut self.config.layer_height, 0.05..=1.0).text("Layer height (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.config.nozzle_diameter, 0.1..=1.5)
                .text("Nozzle diameter (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.config.wall_line_width, 0.05..=1.5)
                .text("Wall line width (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.config.shell_thickness, 0.0..=5.0)
                .text("Shell thickness (mm)"),
        );
        ui.add(egui::Slider::new(&mut self.config.wall_offset, 0.0..=1.0).text("Wall offset (mm)"));

        ui.separator();
        ui.heading("Infill");
        egui::ComboBox::from_label("Pattern")
            .selected_text(format!("{:?}", self.config.infill_pattern))
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.config.infill_pattern,
                    InfillPatternKind::Monotonic,
                    "Monotonic",
                );
            });
        ui.add(
            egui::Slider::new(&mut self.config.infill_line_width, 0.05..=1.5)
                .text("Infill line width (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.config.infill_angle_deg, 0.0..=180.0)
                .text("Infill angle (deg)"),
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
            ui.add(
                egui::Slider::new(&mut self.config.order_field_slope, 0.0..=2.0).text("Cone slope"),
            );
        }

        ui.separator();
        ui.heading("Print Gcode");
        ui.label("Start Gcode");
        ui.add(
            egui::TextEdit::multiline(&mut self.config.start_gcode)
                .desired_rows(3)
                .code_editor()
                .hint_text("e.g. PRINT_START T_TOOL=240 T_BED=105 T_CHAMBER=45 PRINT_MIN={print_min_x},{print_min_y} PRINT_MAX={print_max_x},{print_max_y}"),
        );
        ui.label("End Gcode");
        ui.add(
            egui::TextEdit::multiline(&mut self.config.end_gcode)
                .desired_rows(2)
                .code_editor()
                .hint_text("e.g. PRINT_END"),
        );
        ui.label(
            "Placeholders: {print_min_x} {print_min_y} {print_max_x} {print_max_y} \
             (first layer's XY bounding box, substituted at slice time).",
        );

        ui.separator();
        ui.heading("Machine");
        let (min, mut max) = self.machine.build_volume.bounding_box();
        let mut bed_changed = false;
        bed_changed |= ui
            .add(egui::Slider::new(&mut max.x, 50.0..=1000.0).text("Bed X (mm)"))
            .changed();
        bed_changed |= ui
            .add(egui::Slider::new(&mut max.y, 50.0..=1000.0).text("Bed Y (mm)"))
            .changed();
        bed_changed |= ui
            .add(egui::Slider::new(&mut max.z, 50.0..=1000.0).text("Build height (mm)"))
            .changed();
        if bed_changed {
            self.machine.build_volume = BoundingVolume::Aabb { min, max };
            let device = frame
                .wgpu_render_state()
                .expect("wgpu renderer is required")
                .device
                .clone();
            self.uploaded_scene = Arc::new(Self::build_scene(&device, &self.machine));
        }
        if let Some(tool) = self.machine.tools.first_mut() {
            ui.add(
                egui::Slider::new(&mut tool.nozzle_diameter, 0.1..=1.5)
                    .text("Tool 0 nozzle diameter (mm)"),
            );
            ui.add(
                egui::Slider::new(&mut tool.extrusion_multiplier, 0.5..=1.5)
                    .text("Tool 0 extrusion multiplier"),
            );
        }
        if ui.button("Add tool").clicked() {
            self.machine
                .tools
                .push(Tool::new(ToolId(self.next_tool_id), 0.4));
            self.next_tool_id += 1;
        }

        ui.horizontal(|ui| {
            if ui.button("Save Profile…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Profile", &["json"])
                    .set_file_name("profile.json")
                    .save_file()
                {
                    let profile = Profile {
                        machine: self.machine.clone(),
                        config: self.config.clone(),
                    };
                    match profile.save(&path) {
                        Ok(()) => self.profile_error = None,
                        Err(error) => self.profile_error = Some(error.to_string()),
                    }
                }
            }
            if ui.button("Load Profile…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Profile", &["json"])
                    .pick_file()
                {
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
                            self.profile_error = None;

                            let device = frame
                                .wgpu_render_state()
                                .expect("wgpu renderer is required")
                                .device
                                .clone();
                            self.uploaded_scene =
                                Arc::new(Self::build_scene(&device, &self.machine));
                        }
                        Err(error) => self.profile_error = Some(error.to_string()),
                    }
                }
            }
        });

        ui.separator();
        ui.heading("Objects");
        if self.objects.is_empty() {
            ui.label("No objects loaded. Use Import to load an STL or 3MF file.");
        } else {
            let tool_ids: Vec<ToolId> = self.machine.tools.iter().map(|tool| tool.id).collect();
            for (index, object) in self.objects.iter_mut().enumerate() {
                let selected = self.selected == Some(index);
                let label = format!(
                    "Object {} — {} triangles",
                    object.id.0,
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
                });
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
        if let Some(err) = &self.profile_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("Profile failed: {err}"));
        }

        ui.separator();
        ui.checkbox(&mut self.show_sdf_panel, "Show SDF debug panel");

        if let Some(gcode) = &self.gcode {
            ui.separator();
            ui.heading("Gcode");
            ui.label(format!("{} line(s) generated", gcode.lines().count()));
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    ui.monospace(gcode);
                });
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
        ui.add(egui::Slider::new(&mut self.sdf_iso_level, -2.0..=2.0).text("Iso level (mm)"));

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
            ui.add(egui::Slider::new(&mut self.sdf_slice_offset, -50.0..=50.0).text("Offset (mm)"));
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
        ui.horizontal(|ui| {
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
            ui.label(format!("{} object(s) loaded", self.objects.len()));

            ui.separator();
            let slicing_in_progress = self.slicing.is_some();
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
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add(
                        egui::ProgressBar::new(self.slice_progress as f32)
                            .text(format!("Slicing… {:.0}%", self.slice_progress * 100.0)),
                    );
                });
            }
            if ui
                .add_enabled(self.gcode.is_some(), egui::Button::new("Export…"))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Gcode", &["gcode"])
                    .set_file_name("out.gcode")
                    .save_file()
                {
                    if let Some(gcode) = &self.gcode {
                        if let Err(error) = std::fs::write(&path, gcode) {
                            self.slice_error = Some(error.to_string());
                        }
                    }
                }
            }
            ui.checkbox(&mut self.show_toolpaths, "Show toolpaths");

            // Order-based scrub slider (Phase 13 subtask 05): ranged over
            // the min/max `order` value across all segments in the current
            // `toolpaths`, disabled when there's nothing to scrub. Dragging
            // triggers a CPU-side rebuild-on-change re-upload (see
            // `toolpath_view::build_toolpath_lines`'s doc comment for the
            // tradeoff versus a shader-side discard).
            let (slider_min, slider_max) = self.toolpath_order_range.unwrap_or((0.0, 0.0));
            let mut slider_value = self.scrub_order.min(slider_max).max(slider_min);
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

        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

            if response.dragged() {
                let delta = response.drag_delta();
                if ui.input(|i| i.pointer.secondary_down()) {
                    self.camera.pan(delta.x, delta.y);
                } else {
                    self.camera.orbit(delta.x, delta.y);
                }
            }
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll);
            }

            let aspect_ratio = rect.width() / rect.height().max(1.0);
            let view_proj = self.camera.view_proj(aspect_ratio);

            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    ScenePaintCallback {
                        view_proj,
                        scene: self.uploaded_scene.clone(),
                    },
                ));
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    rect,
                    MeshPaintCallback {
                        view_proj,
                        meshes: self.uploaded_meshes.clone(),
                    },
                ));
            if let Some(overlay_mesh) = &self.sdf_overlay_mesh {
                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        OverlayPaintCallback {
                            view_proj,
                            mesh: overlay_mesh.clone(),
                        },
                    ));
            }
            if self.show_toolpaths {
                if let Some(toolpaths) = &self.uploaded_toolpaths {
                    ui.painter()
                        .add(eframe::egui_wgpu::Callback::new_paint_callback(
                            rect,
                            ToolpathPaintCallback {
                                view_proj,
                                toolpaths: toolpaths.clone(),
                            },
                        ));
                }
            }

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
                        let mut nearest: Option<(f32, &manifold_core::toolpath::Segment)> = None;
                        for path in toolpaths {
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
                                if nearest.is_none_or(|(best_dist, _)| dist < best_dist) {
                                    nearest = Some((dist, segment));
                                }
                            }
                        }
                        if let Some((dist, segment)) = nearest {
                            if dist <= PICK_THRESHOLD_PX {
                                response.clone().show_tooltip_ui(|ui| {
                                    ui.label(format!("kind: {:?}", segment.kind));
                                    ui.label(format!("speed: {:.3}", segment.speed));
                                    ui.label(format!(
                                        "extrusion_rate: {:.3}",
                                        segment.extrusion_rate
                                    ));
                                    ui.label(format!(
                                        "support_fraction: {:.3}",
                                        segment.support_fraction
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
                if let Some(object) = self.objects.get(index) {
                    self.gizmo.update_config(GizmoConfig {
                        view_matrix: self.camera.view_matrix_f64().into(),
                        projection_matrix: self.camera.projection_matrix_f64(aspect_ratio).into(),
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

                    if let Some((_, mut new_transforms)) =
                        self.gizmo_interact(ui, rect, view_proj, translation, &[gizmo_transform])
                    {
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
                            self.reupload(&device);
                        }
                    }
                }
            }
        });
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
            .show(ctx, |ui| self.settings_panel(ui, frame));

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

    match extension.as_str() {
        "3mf" => {
            let file = File::open(path)?;
            let mut objects = threemf::load_3mf(file, ToolId(0))?;
            for object in &mut objects {
                object.id = ObjectId(*next_object_id);
                *next_object_id += 1;
            }
            Ok(objects)
        }
        "stl" => {
            let file = File::open(path)?;
            let mesh: Mesh = stl::load_stl(BufReader::new(file))?;
            let id = ObjectId(*next_object_id);
            *next_object_id += 1;
            Ok(vec![Object::new(id, mesh, ToolId(0))])
        }
        other => anyhow::bail!(
            "unsupported input format {:?} for {}: only .3mf and .stl are supported today",
            other,
            path.display()
        ),
    }
}
