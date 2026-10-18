pub mod arrange;
pub mod bridge;
pub mod config;
pub mod hull;
pub mod io;
pub mod scene;
pub mod ui;

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy_egui::EguiPlugin;
use bridge::ffi::RenderVertex;
use config::SlicerConfig;
use scene::{
    compute_orbit_eye, draw_viewport_gizmos_system, update_selection_materials_system,
    viewport_interaction_system, BuildPlateState, OrbitCamera,
};
use ui::{poll_io_tasks_system, ui_workspace_system};

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
        .init_resource::<SlicerConfig>()
        .init_resource::<BuildPlateState>()
        .add_systems(Startup, setup_viewport_scene_and_pipeline)
        .add_systems(
            Update,
            (
                ui_workspace_system,
                poll_io_tasks_system,
                viewport_interaction_system,
                update_selection_materials_system,
                draw_viewport_gizmos_system,
            ),
        )
        .run();
}

/// Initializes the 3D viewport camera/lighting and constructs the raw `wgpu::RenderPipeline`
/// configured with `RenderVertex::desc()` (24-byte stride via `bytemuck`).
fn setup_viewport_scene_and_pipeline(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut plate: ResMut<BuildPlateState>,
) {
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

    plate.status_message =
        "Ready — import STEP assemblies, STL meshes, or 3MF projects.".to_string();

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
