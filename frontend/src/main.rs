pub mod arrange;
pub mod bridge;
pub mod calibration;
pub mod cli;
pub mod config;
pub mod field_vis;
pub mod gcode;
pub mod gpu_compute;
pub mod hull;
pub mod io;
pub mod moonraker;
pub mod presets;
pub mod scene;
pub mod surface_ui;
pub mod ui;

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::IoTaskPool;
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bridge::ffi::RenderVertex;
use field_vis::{sync_field_visualization_system, FieldVisualizationState};
use scene::{
    compute_orbit_eye, draw_viewport_gizmos_system, update_selection_materials_system,
    viewport_interaction_system, BuildPlateState, OrbitCamera,
};
use ui::{poll_io_tasks_system, sync_config_to_backend, ui_workspace_system};

/// WGSL shader module matching the `RenderVertex` memory layout
/// (`@location(0) position: vec3<f32>`, `@location(1) normal: vec3<f32>`).
const VIEWPORT_SHADER_WGSL: &str = r#"
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position * 0.02, 1.0);
    out.world_normal = normalize(in.normal);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 0.6, 1.0));
    let ndotl = max(dot(normalize(in.world_normal), light_dir), 0.15);
    let base_color = vec3<f32>(0.22, 0.58, 0.88);
    return vec4<f32>(base_color * ndotl, 1.0);
}
"#;

#[derive(Resource, Default)]
struct StartupFiles(Vec<std::path::PathBuf>);

fn main() {
    let gui_opts = match cli::handle_cli_args_from_env() {
        cli::CliDispatch::LaunchGui(opts) => opts,
        cli::CliDispatch::Exit(code) => std::process::exit(code),
    };

    let initial_cfg = gui_opts.initial_config.unwrap_or_default();
    sync_config_to_backend(&initial_cfg);
    let mut preset_lib = presets::PresetLibrary::default();
    preset_lib.selected_printer = initial_cfg.printer_preset_name.clone();
    preset_lib.selected_tool = initial_cfg.tool_preset_name.clone();
    preset_lib.selected_filament = initial_cfg.filament_preset_name.clone();
    preset_lib.selected_process = initial_cfg.process_preset_name.clone();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Manifold Slicer".into(),
                resolution: (1440.0_f32, 900.0_f32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::srgb(0.11, 0.12, 0.15)))
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 450.0,
        })
        .insert_resource(initial_cfg)
        .insert_resource(preset_lib)
        .insert_resource(StartupFiles(gui_opts.initial_files))
        .init_resource::<BuildPlateState>()
        .init_resource::<FieldVisualizationState>()
        .init_resource::<calibration::CalibrationUiState>()
        .init_resource::<moonraker::MoonrakerUiState>()
        .init_resource::<surface_ui::SurfaceUiState>()
        .add_systems(
            Startup,
            (
                setup_viewport_scene_and_pipeline,
                set_window_icon_startup_system,
            ),
        )
        .add_systems(
            Update,
            (
                ui_workspace_system,
                poll_io_tasks_system,
                surface_ui::sync_and_interact_surface_ui_system,
                viewport_interaction_system,
                sync_field_visualization_system,
                update_selection_materials_system,
                draw_viewport_gizmos_system,
            )
                .chain(),
        )
        .run();
}

/// Registers system mathematical and technical symbol fallback fonts into `egui`
/// so glyphs such as `⊥`, `∥`, `✕`, `∇`, `φ`, `τ`, and `Δ` render cleanly instead of placeholder boxes.
fn configure_egui_symbol_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let candidate_paths = [
        ("apple_symbols", "/System/Library/Fonts/Apple Symbols.ttf"),
        (
            "arial_unicode",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ),
        (
            "dejavu_sans",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ),
        (
            "noto_symbols2",
            "/usr/share/fonts/truetype/noto/NotoSansSymbols2-Regular.ttf",
        ),
        (
            "noto_math",
            "/usr/share/fonts/opentype/noto/NotoSansMath-Regular.ttf",
        ),
        ("segoe_ui_symbol", "C:/Windows/Fonts/seguisym.ttf"),
        ("cambria_math", "C:/Windows/Fonts/cambria.ttc"),
        ("arial_win", "C:/Windows/Fonts/arial.ttf"),
    ];

    let mut loaded_any = false;
    for (name, path) in candidate_paths {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert(name.to_string(), egui::FontData::from_owned(bytes));
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                family.push(name.to_string());
            }
            if let Some(family) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                family.push(name.to_string());
            }
            loaded_any = true;
        }
    }

    if loaded_any {
        ctx.set_fonts(fonts);
    }
}

/// Initializes the 3D viewport camera/lighting and constructs the raw `wgpu::RenderPipeline`
/// configured with `RenderVertex::desc()` (24-byte stride via `bytemuck`).
fn setup_viewport_scene_and_pipeline(
    mut commands: Commands,
    mut egui_contexts: EguiContexts,
    render_device: Option<Res<RenderDevice>>,
    mut plate: ResMut<BuildPlateState>,
    mut startup_files: ResMut<StartupFiles>,
) {
    if let Some(ctx) = egui_contexts.try_ctx_mut() {
        configure_egui_symbol_fonts(ctx);
    }

    let orbit = OrbitCamera::default();
    let eye = compute_orbit_eye(&orbit);

    // Spawn Bevy 3D viewport camera with Z-up CAD convention
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z),
            ..default()
        },
        orbit,
    ));

    // Key & fill directional lights
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(250.0, -220.0, 350.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 5_500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(-240.0, 260.0, -140.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    if !startup_files.0.is_empty() {
        let paths = std::mem::take(&mut startup_files.0);
        plate.status_message = format!("Loading {} startup file(s)...", paths.len());
        plate.pending_load_task =
            Some(IoTaskPool::get().spawn(async move { Some(io::load_files_into_backend(&paths)) }));
    } else {
        plate.status_message =
            "Ready — import STEP assemblies, STL meshes, or 3MF projects.".to_string();
    }

    // Initialize the custom wgpu render pipeline using the RenderVertex layout
    if let Some(device) = render_device {
        let wgpu_device = device.wgpu_device();
        let shader = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold_cad_viewport_shader"),
            source: wgpu::ShaderSource::Wgsl(VIEWPORT_SHADER_WGSL.into()),
        });

        let pipeline_layout = wgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold_cad_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let pipeline = wgpu_device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold_cad_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[RenderVertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        plate.render_pipeline = Some(pipeline);
    }
}

const EMBEDDED_APP_ICON_PNG: &[u8] = include_bytes!("../assets/app_icon.png");

/// Decodes the embedded 256×256 RGBA application icon PNG.
pub fn decode_embedded_app_icon_rgba() -> Option<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(EMBEDDED_APP_ICON_PNG));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let bytes = &buf[..info.buffer_size()];

    let rgba = match info.color_type {
        png::ColorType::Rgba => bytes.to_vec(),
        png::ColorType::Rgb => {
            let mut out = Vec::with_capacity((info.width * info.height * 4) as usize);
            for chunk in bytes.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        _ => return None,
    };
    Some((info.width, info.height, rgba))
}

fn set_window_icon_startup_system(_winit_windows: NonSend<bevy::winit::WinitWindows>) {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::{c_char, c_void};
        #[link(name = "objc")]
        extern "C" {
            fn objc_getClass(name: *const c_char) -> *mut c_void;
            fn sel_registerName(name: *const c_char) -> *mut c_void;
            fn objc_msgSend();
        }
        unsafe {
            let msg_id_sel: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(objc_msgSend as *const ());
            let msg_data: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const u8,
                usize,
            ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
            let msg_id_arg: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
            ) -> *mut c_void = std::mem::transmute(objc_msgSend as *const ());
            let msg_set_icon: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                std::mem::transmute(objc_msgSend as *const ());

            let ns_data_cls = objc_getClass(c"NSData".as_ptr());
            let ns_image_cls = objc_getClass(c"NSImage".as_ptr());
            let ns_app_cls = objc_getClass(c"NSApplication".as_ptr());
            if !ns_data_cls.is_null() && !ns_image_cls.is_null() && !ns_app_cls.is_null() {
                let data_sel = sel_registerName(c"dataWithBytes:length:".as_ptr());
                let alloc_sel = sel_registerName(c"alloc".as_ptr());
                let init_data_sel = sel_registerName(c"initWithData:".as_ptr());
                let shared_app_sel = sel_registerName(c"sharedApplication".as_ptr());
                let set_icon_sel = sel_registerName(c"setApplicationIconImage:".as_ptr());

                let ns_data = msg_data(
                    ns_data_cls,
                    data_sel,
                    EMBEDDED_APP_ICON_PNG.as_ptr(),
                    EMBEDDED_APP_ICON_PNG.len(),
                );
                if !ns_data.is_null() {
                    let img_alloc = msg_id_sel(ns_image_cls, alloc_sel);
                    let ns_image = msg_id_arg(img_alloc, init_data_sel, ns_data);
                    let ns_app = msg_id_sel(ns_app_cls, shared_app_sel);
                    if !ns_image.is_null() && !ns_app.is_null() {
                        msg_set_icon(ns_app, set_icon_sel, ns_image);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedded_app_icon_decodes_cleanly() {
        let (w, h, rgba) =
            decode_embedded_app_icon_rgba().expect("Embedded app_icon.png should decode");
        assert_eq!(w, 256);
        assert_eq!(h, 256);
        assert_eq!(rgba.len(), (256 * 256 * 4) as usize);
        // Verify outer corner (0, 0) is transparent and center (128, 128) is opaque
        assert_eq!(rgba[3], 0);
        assert_eq!(rgba[(128 * 256 + 128) * 4 + 3], 255);
    }
}
