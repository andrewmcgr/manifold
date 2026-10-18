pub mod bridge;

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bridge::ffi::{self, RenderVertex};
use wgpu::util::DeviceExt;

/// Holds the active `wgpu` GPU buffers uploaded from the C++ OCCT tessellator,
/// along with metadata displayed in the `egui` control panel.
#[derive(Resource, Default)]
pub struct CadGpuMeshState {
    pub loaded_file_path: Option<String>,
    pub status_message: String,
    pub vertex_count: usize,
    pub index_count: usize,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub render_pipeline: Option<wgpu::RenderPipeline>,
}

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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Manifold2 — Non-Planar 3D Printing Slicer".into(),
                resolution: (1440.0, 900.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin)
        .init_resource::<CadGpuMeshState>()
        .add_systems(Startup, setup_viewport_scene_and_pipeline)
        .add_systems(Update, ui_side_panel_system)
        .run();
}

/// Initializes the 3D viewport camera/lighting and constructs the raw `wgpu::RenderPipeline`
/// configured with `RenderVertex::desc()` (24-byte stride via `bytemuck`).
fn setup_viewport_scene_and_pipeline(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut mesh_state: ResMut<CadGpuMeshState>,
) {
    // Spawn Bevy 3D viewport camera and key light
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(120.0, 120.0, 120.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(150.0, 200.0, 250.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    mesh_state.status_message = "Ready — select a STEP (.stp / .step) model to slice.".to_string();

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
            cache: None,
        });

        mesh_state.render_pipeline = Some(pipeline);
    }
}

/// Renders the `egui` side panel with a "Load STEP file" button that opens a
/// native file dialog, invokes the C++ OCCT/OpenVDB backend across the `cxx` FFI
/// bridge, and uploads the resulting vertex/index buffers directly to `wgpu`.
fn ui_side_panel_system(
    mut contexts: EguiContexts,
    render_device: Option<Res<RenderDevice>>,
    mut mesh_state: ResMut<CadGpuMeshState>,
) {
    egui::SidePanel::left("slicer_control_panel")
        .default_width(320.0)
        .resizable(true)
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("Manifold Non-Planar Slicer");
            ui.separator();

            ui.label("CAD Ingestion & Volumetric Field (OCCT + OpenVDB)");
            ui.add_space(8.0);

            if ui.button("Load STEP file").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("STEP CAD Model", &["step", "stp", "STEP", "STP"])
                    .pick_file()
                {
                    let path_str = path.to_string_lossy().into_owned();
                    load_step_and_upload_to_gpu(&path_str, render_device.as_deref(), &mut mesh_state);
                }
            }

            ui.add_space(12.0);
            ui.separator();
            ui.heading("Telemetry");

            if let Some(ref path) = mesh_state.loaded_file_path {
                ui.label(format!("Active Model: {path}"));
            } else {
                ui.label("Active Model: <none>");
            }

            ui.label(format!("Vertices: {}", mesh_state.vertex_count));
            ui.label(format!("Triangles: {}", mesh_state.index_count / 3));
            ui.label(format!(
                "GPU Vertex Buffer: {}",
                if mesh_state.vertex_buffer.is_some() {
                    "Uploaded (wgpu::BufferUsage::VERTEX)"
                } else {
                    "Unallocated"
                }
            ));

            ui.add_space(8.0);
            ui.small(&mesh_state.status_message);
        });
}

/// Invokes the `cxx` FFI bridge functions (`load_cad_model`, `get_tessellated_vertices`,
/// `get_tessellated_indices`) and maps the resulting `Vec<RenderVertex>` zero-copy
/// into a `wgpu::Buffer` via `bytemuck::cast_slice`.
fn load_step_and_upload_to_gpu(
    path: &str,
    render_device: Option<&RenderDevice>,
    mesh_state: &mut CadGpuMeshState,
) {
    let loaded = ffi::load_cad_model(path);
    if !loaded {
        mesh_state.status_message = format!("Failed to load or tessellate STEP file: {path}");
        return;
    }

    let vertices: Vec<RenderVertex> = ffi::get_tessellated_vertices();
    let indices: Vec<u32> = ffi::get_tessellated_indices();

    mesh_state.loaded_file_path = Some(path.to_string());
    mesh_state.vertex_count = vertices.len();
    mesh_state.index_count = indices.len();

    if let Some(device) = render_device {
        let wgpu_device = device.wgpu_device();

        // Zero-copy slice cast from &[RenderVertex] (24-byte C-compatible struct) to &[u8]
        let vertex_bytes: &[u8] = bytemuck::cast_slice(&vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&indices);

        let vertex_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occt_tessellated_vertex_buffer"),
            contents: vertex_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("occt_tessellated_index_buffer"),
            contents: index_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        mesh_state.vertex_buffer = Some(vertex_buffer);
        mesh_state.index_buffer = Some(index_buffer);
    }

    mesh_state.status_message = format!(
        "Loaded {} vertices and {} triangles; OpenVDB Eikonal FloatGrid initialized.",
        mesh_state.vertex_count,
        mesh_state.index_count / 3
    );
}
