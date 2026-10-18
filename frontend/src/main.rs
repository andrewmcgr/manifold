pub mod bridge;

use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, IoTaskPool, Task};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bridge::ffi::{self, RenderVertex};
use wgpu::util::DeviceExt;

/// Result of asynchronous file picking + C++ OCCT/OpenVDB processing.
pub struct LoadedCadPayload {
    pub path: String,
    pub loaded: bool,
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
}

/// Marker component for the active CAD model entity in the 3D viewport.
#[derive(Component)]
pub struct CadModelEntity;

/// Interactive 3D orbit camera controller for inspecting loaded STEP solids.
#[derive(Component)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            radius: 150.0,
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: std::f32::consts::FRAC_PI_6,
        }
    }
}

/// Holds the active `wgpu` GPU buffers uploaded from the C++ OCCT tessellator,
/// along with metadata displayed in the `egui` control panel.
#[derive(Resource, Default)]
pub struct CadGpuMeshState {
    pub loaded_file_path: Option<String>,
    pub status_message: String,
    pub vertex_count: usize,
    pub index_count: usize,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    pub render_pipeline: Option<wgpu::RenderPipeline>,
    pub pending_load_task: Option<Task<Option<LoadedCadPayload>>>,
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
        .init_resource::<CadGpuMeshState>()
        .add_systems(Startup, setup_viewport_scene_and_pipeline)
        .add_systems(
            Update,
            (
                ui_side_panel_system,
                poll_cad_load_task_system,
                orbit_camera_control_system,
                draw_build_plate_gizmos_system,
            ),
        )
        .run();
}

/// Initializes the 3D viewport camera/lighting and constructs the raw `wgpu::RenderPipeline`
/// configured with `RenderVertex::desc()` (24-byte stride via `bytemuck`).
fn setup_viewport_scene_and_pipeline(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut mesh_state: ResMut<CadGpuMeshState>,
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
        transform: Transform::from_xyz(200.0, -180.0, 300.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 5_000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(-200.0, 220.0, -120.0).looking_at(Vec3::ZERO, Vec3::Z),
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
        });

        mesh_state.render_pipeline = Some(pipeline);
    }
}

/// Renders the `egui` side panel with a "Load STEP file" button that opens an
/// asynchronous native file dialog (`rfd::AsyncFileDialog`) so macOS `NSOpenPanel`
/// can dispatch on the main thread without deadlocking Bevy's worker pool.
fn ui_side_panel_system(
    mut contexts: EguiContexts,
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

            let is_busy = mesh_state.pending_load_task.is_some();
            if ui
                .add_enabled(!is_busy, egui::Button::new("Load STEP file"))
                .clicked()
            {
                mesh_state.status_message = "Selecting STEP file...".to_string();
                let task = IoTaskPool::get().spawn(async move {
                    let file_handle = rfd::AsyncFileDialog::new()
                        .add_filter("STEP CAD Model", &["step", "stp", "STEP", "STP"])
                        .pick_file()
                        .await?;

                    let path = file_handle.path().to_string_lossy().into_owned();
                    let loaded = ffi::load_cad_model(&path);
                    let (vertices, indices) = if loaded {
                        (
                            ffi::get_tessellated_vertices(),
                            ffi::get_tessellated_indices(),
                        )
                    } else {
                        (Vec::new(), Vec::new())
                    };

                    Some(LoadedCadPayload {
                        path,
                        loaded,
                        vertices,
                        indices,
                    })
                });
                mesh_state.pending_load_task = Some(task);
            }

            ui.add_space(12.0);
            ui.separator();
            ui.heading("Telemetry");

            if let Some(ref path) = mesh_state.loaded_file_path {
                ui.label(format!("Active Model: {path}"));
                let extents = mesh_state.bounds_max - mesh_state.bounds_min;
                ui.label(format!(
                    "Dimensions: {:.2} × {:.2} × {:.2} mm",
                    extents.x, extents.y, extents.z
                ));
            } else {
                ui.label("Active Model: <none>");
            }

            ui.label(format!("Vertices: {}", mesh_state.vertex_count));
            ui.label(format!("Triangles: {}", mesh_state.index_count / 3));
            ui.label(format!(
                "GPU Vertex Buffer: {}",
                if mesh_state.vertex_buffer.is_some() {
                    "Uploaded (wgpu::BufferUsages::VERTEX)"
                } else {
                    "Unallocated"
                }
            ));

            ui.add_space(12.0);
            ui.separator();
            ui.small("Viewport Controls: Left-drag = Orbit | Right-drag = Pan | Scroll = Zoom");
            ui.add_space(4.0);
            ui.small(&mesh_state.status_message);
        });
}

/// Polls the background STEP load task, uploads the resulting `Vec<RenderVertex>`
/// and index arrays zero-copy into `wgpu::Buffer` instances via `bytemuck::cast_slice`,
/// and spawns the rendered 3D mesh in the Bevy viewport.
fn poll_cad_load_task_system(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut mesh_state: ResMut<CadGpuMeshState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    existing_models: Query<Entity, With<CadModelEntity>>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let Some(ref mut task) = mesh_state.pending_load_task else {
        return;
    };

    let Some(result) = block_on(future::poll_once(task)) else {
        return;
    };

    mesh_state.pending_load_task = None;

    let Some(payload) = result else {
        mesh_state.status_message = "File selection cancelled.".to_string();
        return;
    };

    if !payload.loaded || payload.vertices.is_empty() {
        mesh_state.status_message =
            format!("Failed to load or tessellate STEP file: {}", payload.path);
        return;
    }

    mesh_state.loaded_file_path = Some(payload.path);
    mesh_state.vertex_count = payload.vertices.len();
    mesh_state.index_count = payload.indices.len();

    // 1. Upload raw C++ RenderVertex & index slices directly to wgpu buffers via bytemuck
    if let Some(device) = render_device {
        let wgpu_device = device.wgpu_device();

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&payload.vertices);
        let index_bytes: &[u8] = bytemuck::cast_slice(&payload.indices);

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

    // 2. Compute bounding box and build the viewport Mesh for 3D scene rendering
    let mut min_pt = Vec3::splat(f32::INFINITY);
    let mut max_pt = Vec3::splat(f32::NEG_INFINITY);
    let mut positions = Vec::with_capacity(payload.vertices.len());
    let mut normals = Vec::with_capacity(payload.vertices.len());

    for v in &payload.vertices {
        let pos = Vec3::new(v.x, v.y, v.z);
        min_pt = min_pt.min(pos);
        max_pt = max_pt.max(pos);
        positions.push([v.x, v.y, v.z]);
        normals.push([v.nx, v.ny, v.nz]);
    }

    mesh_state.bounds_min = min_pt;
    mesh_state.bounds_max = max_pt;

    let center = 0.5 * (min_pt + max_pt);
    let half_extents = 0.5 * (max_pt - min_pt);
    let bounding_radius = half_extents.length().max(1.0);

    // Remove previously loaded CAD model entity, if any
    for entity in &existing_models {
        commands.entity(entity).despawn_recursive();
    }

    let bevy_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(payload.indices));

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(bevy_mesh),
            material: materials.add(StandardMaterial {
                base_color: Color::srgb(0.24, 0.62, 0.92),
                metallic: 0.25,
                perceptual_roughness: 0.45,
                double_sided: true,
                cull_mode: None,
                ..default()
            }),
            ..default()
        },
        CadModelEntity,
    ));

    // Auto-frame the 3D orbit camera onto the loaded CAD solid
    for (mut orbit, mut transform) in &mut camera_query {
        orbit.focus = center;
        orbit.radius = bounding_radius * 2.4;
        let eye = compute_orbit_eye(&orbit);
        *transform = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z);
    }

    mesh_state.status_message = format!(
        "Loaded {} vertices and {} triangles; OpenVDB Eikonal FloatGrid initialized.",
        mesh_state.vertex_count,
        mesh_state.index_count / 3
    );
}

fn compute_orbit_eye(orbit: &OrbitCamera) -> Vec3 {
    let cos_pitch = orbit.pitch.cos();
    let offset = Vec3::new(
        orbit.radius * cos_pitch * orbit.yaw.cos(),
        orbit.radius * cos_pitch * orbit.yaw.sin(),
        orbit.radius * orbit.pitch.sin(),
    );
    orbit.focus + offset
}

/// Interactive mouse orbit / pan / zoom camera system (ignores pointer events over `egui`).
fn orbit_camera_control_system(
    mut contexts: EguiContexts,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let ctx = contexts.ctx_mut();
    if ctx.wants_pointer_input() || ctx.is_pointer_over_area() {
        mouse_motion.clear();
        mouse_wheel.clear();
        return;
    }

    let mut delta_drag = Vec2::ZERO;
    for ev in mouse_motion.read() {
        delta_drag += ev.delta;
    }

    let mut scroll = 0.0_f32;
    for ev in mouse_wheel.read() {
        scroll += ev.y;
    }

    if delta_drag == Vec2::ZERO && scroll == 0.0 {
        return;
    }

    for (mut orbit, mut transform) in &mut query {
        if mouse_buttons.pressed(MouseButton::Left) {
            orbit.yaw -= delta_drag.x * 0.008;
            orbit.pitch = (orbit.pitch + delta_drag.y * 0.008).clamp(-1.50, 1.50);
        } else if mouse_buttons.pressed(MouseButton::Right)
            || mouse_buttons.pressed(MouseButton::Middle)
        {
            let right = transform.right();
            let up = transform.up();
            let pan_scale = orbit.radius * 0.0015;
            orbit.focus += (-right * delta_drag.x + up * delta_drag.y) * pan_scale;
        }

        if scroll != 0.0 {
            let zoom_factor = (1.0 - scroll * 0.1).clamp(0.5, 2.0);
            orbit.radius = (orbit.radius * zoom_factor).clamp(0.5, 50_000.0);
        }

        let eye = compute_orbit_eye(&orbit);
        *transform = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z);
    }
}

/// Draws a reference XY build-plate grid and coordinate axes under the active model.
fn draw_build_plate_gizmos_system(mut gizmos: Gizmos, mesh_state: Res<CadGpuMeshState>) {
    let (center, half_size, z_plane) = if mesh_state.vertex_count > 0 {
        let c = 0.5 * (mesh_state.bounds_min + mesh_state.bounds_max);
        let span = (mesh_state.bounds_max - mesh_state.bounds_min)
            .max_element()
            .max(20.0);
        (Vec2::new(c.x, c.y), span * 0.75, mesh_state.bounds_min.z)
    } else {
        (Vec2::ZERO, 100.0, 0.0)
    };

    let divisions = 10;
    let step = (half_size * 2.0) / divisions as f32;
    let grid_color = Color::srgba(0.35, 0.38, 0.44, 0.45);

    for i in 0..=divisions {
        let offset = -half_size + i as f32 * step;
        gizmos.line(
            Vec3::new(center.x - half_size, center.y + offset, z_plane),
            Vec3::new(center.x + half_size, center.y + offset, z_plane),
            grid_color,
        );
        gizmos.line(
            Vec3::new(center.x + offset, center.y - half_size, z_plane),
            Vec3::new(center.x + offset, center.y + half_size, z_plane),
            grid_color,
        );
    }
}
