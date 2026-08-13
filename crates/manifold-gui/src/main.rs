//! `manifold-gui`: egui/wgpu desktop front-end for the Manifold slicer.
//!
//! TODO(roadmap): Phases 4-7 (see ROADMAP.md) — grow this scaffold into the
//! full slicer layout: left `SidePanel` settings, `CentralPanel` 3D
//! viewport with an in-panel top toolbar (Phase 4); a wgpu render pipeline
//! embedded via `egui_wgpu::Callback` with an orbit camera (Phase 5);
//! in-scene origin/bed/toolhead visualization driven by `Machine` (Phase
//! 6); and `transform-gizmo-egui`-based per-object move/rotate/scale
//! gizmos (Phase 7, needs an egui-version compatibility check against the
//! `eframe = "0.29"` pin below).

use eframe::egui;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Manifold",
        native_options,
        Box::new(|cc| Ok(Box::new(ManifoldApp::new(cc)))),
    )
}

struct ManifoldApp {}

impl ManifoldApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {}
    }
}

impl eframe::App for ManifoldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Manifold — non-planar slicer");
            ui.label("GUI scaffold. Load a model to begin.");
        });
    }
}
