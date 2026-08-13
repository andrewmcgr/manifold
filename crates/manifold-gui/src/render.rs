//! wgpu triangle-mesh render pipeline embedded via `egui_wgpu::Callback`
//! (Phase 5, see ROADMAP.md).
//!
//! `manifold_core::mesh::Mesh` carries no per-vertex normal data, so each
//! mesh is expanded here (CPU-side, at upload time) into a non-indexed
//! triangle list where every vertex is duplicated per-triangle and carries
//! that triangle's flat face normal.

use crate::scene::SceneVertex;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use egui_wgpu::wgpu::util::DeviceExt as _;
use glam::Mat4;
use manifold_core::mesh::Mesh;
use manifold_fidget::marching_cubes::Vertex as FieldVertex;

/// One GPU vertex: position + flat face normal, both in world space.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
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
    pub fn upload(
        device: &wgpu::Device,
        mesh: &Mesh,
        object_transform: &manifold_core::transform::Transform,
    ) -> Self {
        let mut vertices = Vec::with_capacity(mesh.indices.len());
        for triangle in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]];
            let world = |i: u32| object_transform.transform_point(mesh.vertices[i as usize]);
            let (pa, pb, pc) = (world(a), world(b), world(c));
            let normal = (pb - pa).cross(pc - pa).normalize_or_zero();
            for p in [pa, pb, pc] {
                vertices.push(Vertex {
                    position: p.as_vec3().to_array(),
                    normal: normal.as_vec3().to_array(),
                });
            }
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

/// A scene-dressing buffer (origin axes, grid, bed quad, or toolhead
/// markers) already uploaded to the GPU — see `crate::scene`.
pub struct UploadedScene {
    line_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    tri_buffer: wgpu::Buffer,
    tri_vertex_count: u32,
}

impl UploadedScene {
    /// Upload line-list geometry (origin axes + grid) and triangle-list
    /// geometry (bed quad + toolhead markers), already built by
    /// `crate::scene`'s builders.
    pub fn upload(device: &wgpu::Device, lines: &[SceneVertex], triangles: &[SceneVertex]) -> Self {
        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold scene line buffer"),
            contents: bytemuck::cast_slice(lines),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let tri_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold scene triangle buffer"),
            contents: bytemuck::cast_slice(triangles),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            line_buffer,
            line_vertex_count: lines.len() as u32,
            tri_buffer,
            tri_vertex_count: triangles.len() as u32,
        }
    }
}

/// Resources kept alive alongside the egui render pass (installed into
/// `egui_wgpu::CallbackResources`), shared across frames.
pub struct MeshRenderResources {
    pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    scene_line_pipeline: wgpu::RenderPipeline,
    scene_tri_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl MeshRenderResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
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
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }];

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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Semi-transparent overlay variant (SDF isosurface debug overlay,
        // see `MESH_SDF_VISUALIZATION.md` Phase D): same shader module,
        // vertex layout, and pipeline layout as `pipeline` above \u2014 only the
        // fragment entry point (alpha < 1) and blend/cull state differ, so
        // this reuses rather than duplicates the mesh rendering setup.
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold camera uniform buffer"),
            contents: bytemuck::cast_slice(&[Mat4::IDENTITY.to_cols_array()]),
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

        let scene_vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SceneVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
        }];

        let scene_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold scene line pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &scene_vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &scene_shader,
                entry_point: "fs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(target_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let scene_tri_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("manifold scene triangle pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &scene_shader,
                entry_point: "vs_main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &scene_vertex_buffers,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            overlay_pipeline,
            scene_line_pipeline,
            scene_tri_pipeline,
            camera_buffer,
            camera_bind_group,
        }
    }

    fn write_camera(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[view_proj.to_cols_array()]),
        );
    }

    fn paint(&self, render_pass: &mut wgpu::RenderPass<'_>, meshes: &[UploadedMesh]) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        for mesh in meshes {
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.draw(0..mesh.vertex_count, 0..1);
        }
    }

    /// Draws a single overlay mesh (e.g. the SDF isosurface debug overlay,
    /// `MESH_SDF_VISUALIZATION.md` Phase D) with the semi-transparent
    /// `overlay_pipeline` instead of the opaque `pipeline`, so it renders
    /// visually distinct from the real mesh(es) drawn by [`Self::paint`].
    fn paint_overlay(&self, render_pass: &mut wgpu::RenderPass<'_>, mesh: &UploadedMesh) {
        render_pass.set_pipeline(&self.overlay_pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        render_pass.draw(0..mesh.vertex_count, 0..1);
    }

    fn paint_scene(&self, render_pass: &mut wgpu::RenderPass<'_>, scene: &UploadedScene) {
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

        render_pass.set_pipeline(&self.scene_tri_pipeline);
        render_pass.set_vertex_buffer(0, scene.tri_buffer.slice(..));
        render_pass.draw(0..scene.tri_vertex_count, 0..1);

        render_pass.set_pipeline(&self.scene_line_pipeline);
        render_pass.set_vertex_buffer(0, scene.line_buffer.slice(..));
        render_pass.draw(0..scene.line_vertex_count, 0..1);
    }
}

/// The per-frame paint callback: carries the view-projection matrix and the
/// already-uploaded meshes to draw this frame.
pub struct MeshPaintCallback {
    pub view_proj: Mat4,
    pub meshes: std::sync::Arc<Vec<UploadedMesh>>,
}

impl egui_wgpu::CallbackTrait for MeshPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.write_camera(queue, self.view_proj);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.paint(render_pass, &self.meshes);
    }
}

/// The per-frame paint callback for scene dressing (origin axes, bed grid/
/// quad, toolhead markers) — drawn before `MeshPaintCallback` so imported
/// objects paint on top. There is no depth buffer available inside
/// `egui_wgpu::Callback`'s shared render pass (a known limitation of this
/// embedding approach, deferred rather than solved with a custom
/// multi-pass setup — see ROADMAP.md Phase 6), so draw order stands in for
/// depth testing.
pub struct ScenePaintCallback {
    pub view_proj: Mat4,
    pub scene: std::sync::Arc<UploadedScene>,
}

impl egui_wgpu::CallbackTrait for ScenePaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.write_camera(queue, self.view_proj);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.paint_scene(render_pass, &self.scene);
    }
}

/// The per-frame paint callback for a single semi-transparent overlay mesh
/// (currently the SDF isosurface debug overlay, see
/// `MESH_SDF_VISUALIZATION.md` Phase D). Drawn after `MeshPaintCallback` so
/// it composites on top of the real mesh(es).
pub struct OverlayPaintCallback {
    pub view_proj: Mat4,
    pub mesh: std::sync::Arc<UploadedMesh>,
}

impl egui_wgpu::CallbackTrait for OverlayPaintCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.write_camera(queue, self.view_proj);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources: &MeshRenderResources = callback_resources.get().unwrap();
        resources.paint_overlay(render_pass, &self.mesh);
    }
}
