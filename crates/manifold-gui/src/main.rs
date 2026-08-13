//! `manifold-gui`: egui/wgpu desktop front-end for the Manifold slicer.

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
