//! `ManifoldApp`: GUI shell — left settings panel, central 3D viewport with
//! an in-panel import toolbar (Phase 4, see ROADMAP.md).

use crate::camera::OrbitCamera;
use crate::render::{
    MeshPaintCallback, MeshRenderResources, ScenePaintCallback, UploadedMesh, UploadedScene,
};
use crate::scene;
use eframe::egui;
use manifold_core::bounds::BoundingVolume;
use manifold_core::machine::Machine;
use manifold_core::tool::Tool;
use manifold_core::transform::Transform;
use manifold_core::{ids::ObjectId, ids::ToolId, mesh::Mesh, object, object::Object, stl, threemf};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use transform_gizmo_egui::math::Transform as GizmoTransform;
use transform_gizmo_egui::prelude::*;

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
    slice_error: Option<String>,
    next_tool_id: u32,
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
            camera: OrbitCamera::default(),
            next_object_id: 0,
            import_error: None,
            selected: None,
            gizmo: Gizmo::default(),
            gcode: None,
            slice_error: None,
            next_tool_id: 1,
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

    /// Run the slicing pipeline over the current `objects`/`machine`/
    /// `config` and store the result (or error) for preview/export
    /// (Phase 8, see ROADMAP.md).
    fn slice(&mut self) {
        let workspace = manifold_core::Workspace::new(
            self.objects.clone(),
            self.machine.clone(),
            self.config.clone(),
        );
        match manifold_core::slice_to_gcode(&workspace) {
            Ok(gcode) => {
                self.gcode = Some(gcode);
                self.slice_error = None;
            }
            Err(error) => {
                self.gcode = None;
                self.slice_error = Some(error.to_string());
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
        }
        if ui.button("Add tool").clicked() {
            self.machine
                .tools
                .push(Tool::new(ToolId(self.next_tool_id), 0.4));
            self.next_tool_id += 1;
        }

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
            if ui
                .add_enabled(!self.objects.is_empty(), egui::Button::new("Slice"))
                .clicked()
            {
                self.slice();
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
                        self.gizmo.interact(ui, &[gizmo_transform])
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

        egui::SidePanel::left("settings_panel")
            .default_width(260.0)
            .show(ctx, |ui| self.settings_panel(ui, frame));

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
