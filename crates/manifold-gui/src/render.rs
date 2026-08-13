//! wgpu triangle-mesh render pipeline embedded via `egui_wgpu::Callback`
//! (Phase 5, see ROADMAP.md).
//!
//! `manifold_core::mesh::Mesh` carries no per-vertex normal data, so each
//! mesh is expanded here (CPU-side, at upload time) into a non-indexed
//! triangle list where every vertex is duplicated per-triangle and carries
//! that triangle's flat face normal.

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use egui_wgpu::wgpu::util::DeviceExt as _;
use glam::Mat4;
use manifold_core::mesh::Mesh;

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

        let vertex_count = vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("manifold mesh vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            vertex_count,
        }
    }
}

/// Resources kept alive alongside the egui render pass (installed into
/// `egui_wgpu::CallbackResources`), shared across frames.
pub struct MeshRenderResources {
    pipeline: wgpu::RenderPipeline,
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

        Self {
            pipeline,
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
