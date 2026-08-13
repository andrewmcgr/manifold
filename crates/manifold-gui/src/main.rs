//! `manifold-gui`: egui/wgpu desktop front-end for the Manifold slicer.
//!
//! TODO(roadmap): Phases 6-7 (see ROADMAP.md) — grow this into in-scene
//! origin/bed/toolhead visualization driven by `Machine` (Phase 6), and
//! `transform-gizmo-egui`-based per-object move/rotate/scale gizmos
//! (Phase 7, needs an egui-version compatibility check against the
//! `eframe = "0.29"` pin below).

mod app;
mod camera;
mod render;

use app::ManifoldApp;

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
