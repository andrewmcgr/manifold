//! wgpu triangle-mesh render pipeline embedded via `egui_wgpu::Callback`
//! (Phase 5, see ROADMAP.md).
//!
//! `manifold_core::mesh::Mesh` carries no per-vertex normal data, so each
//! mesh is expanded here (CPU-side, at upload time) into a non-indexed
//! triangle list where every vertex is duplicated per-triangle and carries
//! that triangle's flat face normal.
//!
//! Rendering uses an offscreen 4x MSAA depth+color target in `prepare` with
//! full 32-bit hardware depth testing (`Depth32Float`), so multiple objects and
//! concave features properly occlude one another in correct 3D depth order with
//! smooth antialiasing on all lines and edges, before resolving and blitting
//! into egui's frame pass in `paint`.

use crate::scene::{SceneLineInstance, SceneVertex};
use crate::toolpath_view::ToolpathLineInstance;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use egui_wgpu::wgpu::util::DeviceExt as _;
use glam::{DVec3, Mat4};
use manifold_core::mesh::Mesh;
use manifold_fidget::marching_cubes::Vertex as FieldVertex;

/// Strategy for coloring the uploaded mesh in the 3D viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MeshOverlayMode {
    #[default]
    None,
    ConformalRegions,
    SurfaceOrder,
}

/// Number of multisample anti-aliasing samples used for 3D viewport rendering.
const MSAA_SAMPLES: u32 = 4;

/// Camera and viewport uniform passed to all 3D shaders.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [f32; 16],
    viewport_size: [f32; 2],
    line_width: f32,
    render_mode: f32,
}

/// One GPU vertex: position + flat face normal + RGBA color, all in world space.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

/// A mesh already uploaded to the GPU as a non-indexed vertex buffer.
pub struct UploadedMesh {
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
}

impl UploadedMesh {
    /// Expand `mesh` (with `object_transform` baked into vertex positions —
    /// see ROADMAP.md Phase 7 for interactive per-object transforms) into a
    /// flat-shaded, non-indexed vertex buffer and upload it.
    /// If `overlay_mode` is active, colors faces according to their Eikonal
    /// seed/conforming classifications or continuous surface arrival order gradient.
    pub fn upload(
        device: &wgpu::Device,
        mesh: &Mesh,
        object_transform: &manifold_core::transform::Transform,
        overlay_mode: MeshOverlayMode,
        config: Option<&manifold_core::SlicerConfig>,
    ) -> Self {
        let min_z = mesh
            .vertices
            .iter()
            .map(|v| object_transform.transform_point(*v).z)
            .fold(f64::INFINITY, f64::min);
        let min_z = if min_z.is_finite() { min_z } else { 0.0 };
        let seed_tolerance = config
            .map(|c| c.layer_height.min(c.nozzle_diameter) / 8.0)
            .unwrap_or(0.1);

        let surface_arrival_times = if overlay_mode == MeshOverlayMode::SurfaceOrder {
            let is_seed = |p: DVec3| p.z <= min_z + seed_tolerance;
            let world_verts: Vec<DVec3> = mesh
                .vertices
                .iter()
                .map(|v| object_transform.transform_point(*v))
                .collect();
            let faces: Vec<[usize; 3]> = mesh
                .indices
                .chunks_exact(3)
                .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
                .collect();
            Some(manifold_fidget::surface_eikonal::solve_surface_eikonal(
                &world_verts,
                &faces,
                is_seed,
            ))
        } else {
            None
        };

        let max_surface_time = surface_arrival_times
            .as_ref()
            .map(|times| {
                times
                    .iter()
                    .filter(|t| t.is_finite())
                    .copied()
                    .fold(0.0f64, f64::max)
                    .max(1e-6)
            })
            .unwrap_or(1.0);

        let mut vertices = Vec::with_capacity(mesh.indices.len());
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
            let world = |i: u32| object_transform.transform_point(mesh.vertices[i as usize]);
            let (pa, pb, pc) = (world(a), world(b), world(c));
            let normal = (pb - pa).cross(pc - pa).normalize_or_zero();

            let (ca, cb, cc) = match overlay_mode {
                MeshOverlayMode::SurfaceOrder => {
                    let map_color = |v_idx: u32| {
                        surface_arrival_times
                            .as_ref()
                            .map_or([0.65, 0.68, 0.72, 1.0], |times| {
                                let t = (times[v_idx as usize] / max_surface_time).clamp(0.0, 1.0);
                                crate::toolpath_view::scalar_to_color(t)
                            })
                    };
                    (map_color(a), map_color(b), map_color(c))
                }
                MeshOverlayMode::ConformalRegions => {
                    let color = match config {
                        Some(cfg) => {
                            let on_bed = pa.z <= min_z + seed_tolerance
                                && pb.z <= min_z + seed_tolerance
                                && pc.z <= min_z + seed_tolerance;

                            if on_bed {
                                // Bed contact seed: Bright Green
                                [0.2, 0.85, 0.3, 1.0]
                            } else if normal.z > 1e-3 {
                                // Upward-facing
                                if !cfg.eikonal_conform_top_surfaces {
                                    [0.65, 0.68, 0.72, 1.0]
                                } else {
                                    let beta_deg = normal.z.clamp(-1.0, 1.0).acos().to_degrees();
                                    let top_detach = cfg.eikonal_conformal_max_angle_deg();
                                    if beta_deg <= top_detach - 5.0 {
                                        // Top Conforming: Cyan
                                        [0.15, 0.65, 0.95, 1.0]
                                    } else if beta_deg <= top_detach {
                                        // Top Transition Band: Purple
                                        [0.6, 0.35, 0.9, 1.0]
                                    } else {
                                        // Steep/Detached: Default Slate Gray
                                        [0.65, 0.68, 0.72, 1.0]
                                    }
                                }
                            } else if normal.z < -1e-3 {
                                // Downward-facing
                                if !cfg.eikonal_conform_bottom_surfaces {
                                    [0.65, 0.68, 0.72, 1.0]
                                } else {
                                    let beta_deg = (-normal.z).clamp(-1.0, 1.0).acos().to_degrees();
                                    let bottom_detach =
                                        cfg.eikonal_conformal_bottom_max_angle_deg();
                                    if beta_deg <= bottom_detach - 5.0 {
                                        // Bottom Conforming: Orange
                                        [0.95, 0.55, 0.1, 1.0]
                                    } else if beta_deg <= bottom_detach {
                                        // Bottom Transition Band: Gold/Yellow
                                        [0.95, 0.75, 0.2, 1.0]
                                    } else {
                                        // Steep/Detached: Default Slate Gray
                                        [0.65, 0.68, 0.72, 1.0]
                                    }
                                }
                            } else {
                                // Vertical walls
                                [0.65, 0.68, 0.72, 1.0]
                            }
                        }
                        None => [0.65, 0.68, 0.72, 1.0],
                    };
                    (color, color, color)
                }
                MeshOverlayMode::None => {
                    let default_color = [0.65, 0.68, 0.72, 1.0];
                    (default_color, default_color, default_color)
                }
            };

            vertices.push(Vertex {
                position: pa.as_vec3().to_array(),
                normal: normal.as_vec3().to_array(),
                color: ca,
            });
            vertices.push(Vertex {
                position: pb.as_vec3().to_array(),
                normal: normal.as_vec3().to_array(),
                color: cb,
            });
            vertices.push(Vertex {
                position: pc.as_vec3().to_array(),
                normal: normal.as_vec3().to_array(),
                color: cc,
            });
        }

        Self::from_vertices(device, &vertices, "manifold mesh vertex buffer")
    }

    /// Upload a triangle soup that already carries its own per-vertex
    /// normals (e.g. field-gradient normals from marching-cubes isosurface
    /// extraction — see `MESH_SDF_VISUALIZATION.md` Phase D), without
    /// recomputing flat per-triangle normals from positions the way
    /// [`Self::upload`] does. Reuses the same GPU `Vertex` layout and
    /// upload path — no separate pipeline.
    pub fn upload_from_vertices(device: &wgpu::Device, vertices: &[FieldVertex]) -> Self {
        let vertices: Vec<Vertex> = vertices
            .iter()
            .map(|v| Vertex {
                position: v.position.as_vec3().to_array(),
                normal: v.normal.as_vec3().to_array(),
                color: [1.0, 0.55, 0.15, 0.45],
            })
            .collect();

        Self::from_vertices(device, &vertices, "manifold sdf overlay vertex buffer")
    }

    /// Shared buffer-creation tail for [`Self::upload`] and
    /// [`Self::upload_from_vertices`].
    fn from_vertices(device: &wgpu::Device, vertices: &[Vertex], label: &str) -> Self {
        let vertex_count = vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            vertex_count,
        }
    }
}

/// A NanoVDB sparse grid uploaded to the GPU as storage + uniform buffers for real-time volume raymarching.
#[allow(dead_code)]
pub struct UploadedNanoGrid {
    pub storage_buffer: wgpu::Buffer,
    pub header_buffer: wgpu::Buffer,
    pub leaf_count: u32,
}

#[allow(dead_code)]
impl UploadedNanoGrid {
    /// Uploads `grid` to the GPU as storage and uniform buffers.
    pub fn upload(device: &wgpu::Device, grid: &manifold_fidget::nanovdb::NanoGridBuffer) -> Self {
        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nanogrid_storage_buffer"),
            contents: bytemuck::cast_slice(&grid.leaves),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let header_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nanogrid_header_buffer"),
            contents: bytemuck::bytes_of(&grid.header),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        Self {
            storage_buffer,
            header_buffer,
            leaf_count: grid.leaves.len() as u32,
        }
    }
}

/// A scene-dressing buffer (origin axes, grid, bed quad, or toolhead
/// markers) already uploaded to the GPU — see `crate::scene`.
pub struct UploadedScene {
    line_buffer: wgpu::Buffer,
    line_instance_count: u32,
    tri_buffer: wgpu::Buffer,
    tri_vertex_count: u32,
}

impl UploadedScene {
    /// Upload line-instance geometry (origin axes + grid) and triangle-list
    /// geometry (bed quad + toolhead markers), already built by
    /// `crate::scene`'s builders.
    pub fn upload(
        device: &wgpu::Device,
        lines: &[SceneLineInstance],
        triangles: &[SceneVertex],
    ) -> Self {
        let fallback_line = [SceneLineInstance::default()];
        let line_contents = if lines.is_empty() {
            bytemuck::cast_slice(&fallback_line)
        } else {
            bytemuck::cast_slice(lines)
        };
        let fallback_tri = [SceneVertex::default()];
        let tri_contents = if triangles.is_empty() {
            bytemuck::cast_slice(&fallback_tri)
        } else {
            bytemuck::cast_slice(triangles)
        };

        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold scene line buffer"),
            contents: line_contents,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold scene triangle buffer"),
            contents: tri_contents,
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            line_buffer,
            line_instance_count: lines.len() as u32,
            tri_buffer,
            tri_vertex_count: triangles.len() as u32,
        }
    }
}

/// Toolpath preview line geometry (Phase 13, see ROADMAP.md) already
/// uploaded to the GPU as a line-instance buffer — mirrors
/// `UploadedScene`'s line-buffer half.
pub struct UploadedToolpaths {
    line_buffer: wgpu::Buffer,
    line_instance_count: u32,
}

impl UploadedToolpaths {
    /// Upload line-instance geometry already built by
    /// `crate::toolpath_view::build_toolpath_lines`.
    pub fn upload(device: &wgpu::Device, instances: &[ToolpathLineInstance]) -> Self {
        let fallback = [ToolpathLineInstance::default()];
        let line_contents = if instances.is_empty() {
            bytemuck::cast_slice(&fallback)
        } else {
            bytemuck::cast_slice(instances)
        };

        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold toolpath line buffer"),
            contents: line_contents,
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            line_buffer,
            line_instance_count: instances.len() as u32,
        }
    }
}

struct OffscreenTarget {
    _msaa_color_texture: wgpu::Texture,
    msaa_color_view: wgpu::TextureView,
    _color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    _depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// Resources kept alive alongside the egui render pass (installed into
/// `egui_wgpu::CallbackResources`), shared across frames.
pub struct MeshRenderResources {
    pipeline: wgpu::RenderPipeline,
    mesh_transparent_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    scene_line_pipeline: wgpu::RenderPipeline,
    scene_tri_pipeline: wgpu::RenderPipeline,
    toolpath_line_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    target_format: wgpu::TextureFormat,
    offscreen: Option<OffscreenTarget>,
}

impl MeshRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let depth_format = wgpu::TextureFormat::Depth32Float;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold mesh shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mesh_shader.wgsl").into()),
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("manifold camera bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold mesh pipeline layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
        }];

        let depth_stencil_state = |depth_write_enabled| wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let multisample_state = wgpu::MultisampleState {
            count: MSAA_SAMPLES,
            mask: !0,
            alpha_to_coverage_enabled: false,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold mesh pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state(true)),
            multisample: multisample_state,
            multiview: None,
            cache: None,
        });

        // Semi-transparent mesh variant used when toolpath display is active,
        // allowing internal toolpaths to remain visible through the model shell.
        let mesh_transparent_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("manifold mesh transparent pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_transparent",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil_state(false)),
                multisample: multisample_state,
                multiview: None,
                cache: None,
            });

        // Semi-transparent overlay variant (SDF isosurface debug overlay,
        // see `MESH_SDF_VISUALIZATION.md` Phase D): same shader module,
        // vertex layout, and pipeline layout as `pipeline` above — only the
        // fragment entry point (alpha < 1) and blend/cull state differ.
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold mesh overlay pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_overlay",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state(false)),
            multisample: multisample_state,
            multiview: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold camera uniform buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform {
                view_proj: Mat4::IDENTITY.to_cols_array(),
                viewport_size: [1.0, 1.0],
                line_width: 1.4,
                render_mode: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("manifold camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let scene_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold scene shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scene_shader.wgsl").into()),
        });

        let scene_tri_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        }];

        let scene_line_instance_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneLineInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
        }];

        let scene_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold scene line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_line",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &scene_line_instance_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state(true)),
            multisample: multisample_state,
            multiview: None,
            cache: None,
        });

        let scene_tri_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold scene triangle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_tri",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &scene_tri_vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(depth_stencil_state(true)),
            multisample: multisample_state,
            multiview: None,
            cache: None,
        });

        let toolpath_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold toolpath shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("toolpath_shader.wgsl").into()),
        });

        let toolpath_instance_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ToolpathLineInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x4,
                3 => Float32,
                4 => Float32,
                5 => Float32,
            ],
        }];

        let toolpath_line_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("manifold toolpath line pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &toolpath_shader,
                    entry_point: "vs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &toolpath_instance_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &toolpath_shader,
                    entry_point: "fs_main",
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(depth_stencil_state(true)),
                multisample: multisample_state,
                multiview: None,
                cache: None,
            });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("manifold blit shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(in_vertex_index & 1u) * 4 - 1);
    let y = f32(i32(in_vertex_index & 2u) * 2 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@group(0) @binding(0)
var t_color: texture_2d<f32>;
@group(0) @binding(1)
var s_color: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_color));
    let uv = in.clip_position.xy / dims;
    return textureSample(t_color, s_color, uv);
}
"#
                .into(),
            ),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("manifold blit bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("manifold blit pipeline layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("manifold blit sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            mesh_transparent_pipeline,
            overlay_pipeline,
            scene_line_pipeline,
            scene_tri_pipeline,
            toolpath_line_pipeline,
            blit_pipeline,
            blit_bind_group_layout,
            sampler,
            camera_buffer,
            camera_bind_group,
            target_format,
            offscreen: None,
        }
    }

    fn ensure_offscreen(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(target) = &self.offscreen {
            if target.width == width && target.height == height {
                return;
            }
        }

        let width = width.max(1);
        let height = height.max(1);

        let msaa_color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifold offscreen msaa color texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_color_view =
            msaa_color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifold offscreen color texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("manifold offscreen depth texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("manifold blit bind group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.offscreen = Some(OffscreenTarget {
            _msaa_color_texture: msaa_color_texture,
            msaa_color_view,
            _color_texture: color_texture,
            color_view,
            _depth_texture: depth_texture,
            depth_view,
            blit_bind_group,
            width,
            height,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        rect: egui::Rect,
        view_proj: Mat4,
        scene: &UploadedScene,
        meshes: &[UploadedMesh],
        overlay: Option<&UploadedMesh>,
        toolpaths: Option<&UploadedToolpaths>,
    ) {
        let ppp = screen_descriptor.pixels_per_point;
        let screen_w = screen_descriptor.size_in_pixels[0];
        let screen_h = screen_descriptor.size_in_pixels[1];

        let vp_x = (rect.min.x * ppp).max(0.0).round() as u32;
        let vp_y = (rect.min.y * ppp).max(0.0).round() as u32;
        let vp_w = ((rect.width() * ppp).round() as u32)
            .min(screen_w.saturating_sub(vp_x))
            .max(1);
        let vp_h = ((rect.height() * ppp).round() as u32)
            .min(screen_h.saturating_sub(vp_y))
            .max(1);

        self.ensure_offscreen(device, screen_w, screen_h);
        let line_width = 1.4 * ppp;
        let camera_uniform = CameraUniform {
            view_proj: view_proj.to_cols_array(),
            viewport_size: [vp_w as f32, vp_h as f32],
            line_width,
            render_mode: 0.0,
        };
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniform]),
        );

        let Some(target) = &self.offscreen else {
            return;
        };

        let mut rpass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("manifold 3d offscreen render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.msaa_color_view,
                resolve_target: Some(&target.color_view),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_viewport(vp_x as f32, vp_y as f32, vp_w as f32, vp_h as f32, 0.0, 1.0);
        rpass.set_scissor_rect(vp_x, vp_y, vp_w, vp_h);

        // 1. Draw scene dressing (bed quad + origin/grid lines)
        rpass.set_bind_group(0, &self.camera_bind_group, &[]);

        rpass.set_pipeline(&self.scene_tri_pipeline);
        rpass.set_vertex_buffer(0, scene.tri_buffer.slice(..));
        rpass.draw(0..scene.tri_vertex_count, 0..1);

        rpass.set_pipeline(&self.scene_line_pipeline);
        rpass.set_vertex_buffer(0, scene.line_buffer.slice(..));
        rpass.draw(0..6, 0..scene.line_instance_count);

        if let Some(tp) = toolpaths {
            // When toolpaths are visible: draw toolpaths first, then draw the
            // mesh semi-transparently so internal toolpaths remain clearly visible.
            rpass.set_pipeline(&self.toolpath_line_pipeline);
            rpass.set_vertex_buffer(0, tp.line_buffer.slice(..));
            rpass.draw(0..6, 0..tp.line_instance_count);

            rpass.set_pipeline(&self.mesh_transparent_pipeline);
            for mesh in meshes {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rpass.draw(0..mesh.vertex_count, 0..1);
            }
        } else {
            // Normal mode: draw opaque meshes with depth testing & depth writing
            rpass.set_pipeline(&self.pipeline);
            for mesh in meshes {
                rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                rpass.draw(0..mesh.vertex_count, 0..1);
            }
        }

        // Draw semi-transparent overlay (if any)
        if let Some(overlay_mesh) = overlay {
            rpass.set_pipeline(&self.overlay_pipeline);
            rpass.set_vertex_buffer(0, overlay_mesh.vertex_buffer.slice(..));
            rpass.draw(0..overlay_mesh.vertex_count, 0..1);
        }
    }

    pub fn paint_blit(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        let Some(target) = &self.offscreen else {
            return;
        };
        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, &target.blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// The unified per-frame paint callback: renders the 3D scene (scene dressing,
/// meshes, SDF overlay, and toolpaths) with a 32-bit floating point depth buffer
/// and 4x MSAA anti-aliasing offscreen in `prepare`, then blits the result into
/// egui's frame pass in `paint`.
pub struct Viewport3dCallback {
    pub rect: egui::Rect,
    pub view_proj: Mat4,
    pub scene: std::sync::Arc<UploadedScene>,
    pub meshes: std::sync::Arc<Vec<UploadedMesh>>,
    pub overlay: Option<std::sync::Arc<UploadedMesh>>,
    pub toolpaths: Option<std::sync::Arc<UploadedToolpaths>>,
}

impl egui_wgpu::CallbackTrait for Viewport3dCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &mut MeshRenderResources = callback_resources.get_mut().unwrap();
        resources.prepare(
            device,
            queue,
            screen_descriptor,
            egui_encoder,
            self.rect,
            self.view_proj,
            &self.scene,
            &self.meshes,
            self.overlay.as_deref(),
            self.toolpaths.as_deref(),
        );
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.paint_blit(render_pass);
    }
}
