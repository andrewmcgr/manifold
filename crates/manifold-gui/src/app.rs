//! `ManifoldApp`: GUI shell — left settings panel, central 3D viewport with
//! an in-panel import toolbar (Phase 4, see ROADMAP.md).

use crate::camera::OrbitCamera;
use crate::render::{MeshPaintCallback, MeshRenderResources, UploadedMesh};
use eframe::egui;
use manifold_core::{ids::ObjectId, ids::ToolId, mesh::Mesh, object::Object, stl, threemf};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

pub struct ManifoldApp {
    config: manifold_core::SlicerConfig,
    objects: Vec<Object>,
    /// GPU-uploaded copies of `objects`, rebuilt whenever `objects` changes.
    uploaded_meshes: Arc<Vec<UploadedMesh>>,
    camera: OrbitCamera,
    next_object_id: u32,
    import_error: Option<String>,
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

        Self {
            config: manifold_core::SlicerConfig::default(),
            objects: Vec::new(),
            uploaded_meshes: Arc::new(Vec::new()),
            camera: OrbitCamera::default(),
            next_object_id: 0,
            import_error: None,
        }
    }

    /// Load every object from `path`, dispatching on its file extension
    /// (mirrors `manifold-cli`'s `load_objects`).
    fn import(&mut self, path: &Path, device: &eframe::egui_wgpu::wgpu::Device) {
        match load_objects(path, &mut self.next_object_id) {
            Ok(mut new_objects) => {
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

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.add(
            egui::Slider::new(&mut self.config.layer_height, 0.05..=1.0).text("Layer height (mm)"),
        );
        ui.add(
            egui::Slider::new(&mut self.config.nozzle_diameter, 0.1..=1.5)
                .text("Nozzle diameter (mm)"),
        );

        ui.separator();
        ui.heading("Objects");
        if self.objects.is_empty() {
            ui.label("No objects loaded. Use Import to load an STL or 3MF file.");
        } else {
            for object in &self.objects {
                ui.label(format!(
                    "Object {} — {} triangles",
                    object.id.0,
                    object.mesh.triangle_count()
                ));
            }
        }

        if let Some(err) = &self.import_error {
            ui.separator();
            ui.colored_label(egui::Color32::RED, err);
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
                    MeshPaintCallback {
                        view_proj,
                        meshes: self.uploaded_meshes.clone(),
                    },
                ));
        });
    }
}

impl eframe::App for ManifoldApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::left("settings_panel")
            .default_width(260.0)
            .show(ctx, |ui| self.settings_panel(ui));

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
