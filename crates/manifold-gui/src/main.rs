//! `manifold-gui`: egui/wgpu desktop front-end for the Manifold slicer.
//!
//! TODO(roadmap): Phases 6-7 (see ROADMAP.md) — grow this into in-scene
//! origin/bed/toolhead visualization driven by `Machine` (Phase 6), and
//! `transform-gizmo-egui`-based per-object move/rotate/scale gizmos
//! (Phase 7, needs an egui-version compatibility check against the
//! `eframe = "0.29"` pin below).

mod app;
mod camera;
#[cfg(feature = "mcp-server")]
mod mcp;
mod profile;
mod render;
mod scene;
mod toolpath_view;

use app::ManifoldApp;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn")
            }),
        )
        .init();

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
