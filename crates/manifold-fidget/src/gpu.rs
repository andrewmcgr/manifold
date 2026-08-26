//! `wgpu`-accelerated GPU compute pipelines for NanoVDB sparse grids.
//!
//! Provides GPU kernels for:
//! 1. **Sparse Marching Cubes**: High-throughput isosurface extraction directly from NanoVDB storage buffers.
//! 2. **Dual-Field Toolpath Intersection**: Direct 3D non-planar toolpath curve generation from the intersection
//!    of a Mesh SDF grid and an Eikonal deposition order field.

use crate::marching_cubes::{EDGE_TABLE, TRI_TABLE};
use crate::nanovdb::NanoGridBuffer;
use glam::DVec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Shared GPU device and queue context.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GpuContext {
    /// Attempts to initialize a GPU context using high-performance or fallback adapters.
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .or_else(|| {
                    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: true,
                        compatible_surface: None,
                    }))
                })?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("manifold_gpu_compute"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
                .ok()?;

            Some(Self {
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
    }
}

/// GPU Sparse Marching Cubes compute pipeline.
pub struct GpuSparseMarchingCubes {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    edge_table_buf: wgpu::Buffer,
    tri_table_buf: wgpu::Buffer,
}

impl GpuSparseMarchingCubes {
    pub fn new(ctx: GpuContext) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sparse_marching_cubes_shader"),
            source: wgpu::ShaderSource::Wgsl(SPARSE_MC_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sparse_mc_bgl"),
            entries: &[
                // 0: Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: NanoVDB Leaves storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Edge table
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Tri table
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Atomic counter (u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: Output vertices array<vec4<f32>>
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sparse_mc_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sparse_mc_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let edge_table_u32: Vec<u32> = EDGE_TABLE.iter().map(|&x| x as u32).collect();
        let edge_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_edge_table"),
            contents: bytemuck::cast_slice(&edge_table_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut tri_table_flat = [0i32; 256 * 16];
        for i in 0..256 {
            for j in 0..16 {
                tri_table_flat[i * 16 + j] = TRI_TABLE[i][j] as i32;
            }
        }
        let tri_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_tri_table"),
            contents: bytemuck::cast_slice(&tri_table_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            ctx,
            pipeline,
            bind_group_layout,
            edge_table_buf,
            tri_table_buf,
        }
    }

    /// Extracts triangle vertices on the GPU and reads them back as a CPU `Vec<DVec3>`.
    pub fn extract_isosurface_positions(&self, grid: &NanoGridBuffer, iso: f32) -> Vec<DVec3> {
        if grid.leaves.is_empty() {
            return Vec::new();
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Max possible triangles: 5 triangles * 3 verts * 7^3 cells per leaf
        let max_verts = grid.leaves.len() * 343 * 15;
        let max_bytes = (max_verts * std::mem::size_of::<[f32; 4]>()) as u64;

        let leaves_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nanovdb_leaves"),
            contents: bytemuck::cast_slice(&grid.leaves),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            iso: f32,
            leaf_count: u32,
            voxel_dx: f32,
            voxel_dy: f32,
            voxel_dz: f32,
            pad0: f32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            iso,
            leaf_count: grid.leaves.len() as u32,
            voxel_dx: grid.header.voxel_size[0],
            voxel_dy: grid.header.voxel_size[1],
            voxel_dz: grid.header.voxel_size[2],
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mc_counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let out_verts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc_out_verts"),
            size: max_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sparse_mc_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: leaves_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.edge_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tri_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_verts_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_mc_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sparse_mc_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(grid.leaves.len() as u32, 1, 1);
        }

        let readback_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_verts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_verts"),
            size: max_bytes.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&counter_buf, 0, &readback_counter, 0, 4);
        encoder.copy_buffer_to_buffer(&out_verts_buf, 0, &readback_verts, 0, max_bytes.max(64));

        queue.submit(Some(encoder.finish()));

        // Readback
        let counter_slice = readback_counter.slice(..);
        let verts_slice = readback_verts.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        counter_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            return Vec::new();
        }

        let vert_count = {
            let data = counter_slice.get_mapped_range();
            let count: u32 = *bytemuck::from_bytes(&data[0..4]);
            count as usize
        };
        readback_counter.unmap();

        if vert_count == 0 {
            return Vec::new();
        }

        let (tx_v, rx_v) = std::sync::mpsc::channel();
        verts_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_v.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx_v.recv().ok().and_then(|r| r.ok()).is_none() {
            return Vec::new();
        }

        let positions = {
            let data = verts_slice.get_mapped_range();
            let raw_vecs: &[[f32; 4]] = bytemuck::cast_slice(&data[0..vert_count * 16]);
            raw_vecs
                .iter()
                .map(|v| DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
                .collect()
        };
        readback_verts.unmap();

        positions
    }
}

/// GPU Dual-Field Non-Planar Toolpath compute pipeline (Option 4).
///
/// Directly calculates the line segments of intersection between the Mesh SDF isosurface
/// and the Eikonal deposition order field without generating intermediate 3D triangle meshes.
pub struct GpuDualFieldToolpath {
    ctx: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    edge_table_buf: wgpu::Buffer,
    tri_table_buf: wgpu::Buffer,
}

impl GpuDualFieldToolpath {
    pub fn new(ctx: GpuContext) -> Self {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dual_field_toolpath_shader"),
            source: wgpu::ShaderSource::Wgsl(DUAL_FIELD_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dual_field_bgl"),
            entries: &[
                // 0: Params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: Mesh SDF NanoVDB Leaves storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: Order Field NanoVDB Leaves storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: Edge table
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Tri table
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: Atomic segment counter (u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: Output line endpoints array<vec4<f32>> (pairs of endpoints)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dual_field_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dual_field_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let edge_table_u32: Vec<u32> = EDGE_TABLE.iter().map(|&x| x as u32).collect();
        let edge_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_edge_table"),
            contents: bytemuck::cast_slice(&edge_table_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut tri_table_flat = [0i32; 256 * 16];
        for i in 0..256 {
            for j in 0..16 {
                tri_table_flat[i * 16 + j] = TRI_TABLE[i][j] as i32;
            }
        }
        let tri_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_tri_table"),
            contents: bytemuck::cast_slice(&tri_table_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            ctx,
            pipeline,
            bind_group_layout,
            edge_table_buf,
            tri_table_buf,
        }
    }

    /// Extracts 3D toolpath segments at $\{ p \mid \text{SDF}(p) = -\text{wall\_offset} \land \text{Order}(p) = \text{target\_order} \}$.
    pub fn extract_toolpath_segments(
        &self,
        sdf_grid: &NanoGridBuffer,
        order_grid: &NanoGridBuffer,
        wall_offset: f32,
        target_order: f32,
    ) -> Vec<(DVec3, DVec3)> {
        if sdf_grid.leaves.is_empty() {
            return Vec::new();
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        let max_segments = sdf_grid.leaves.len() * 343 * 5;
        let max_bytes = (max_segments * 2 * std::mem::size_of::<[f32; 4]>()) as u64;

        let sdf_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_sdf_leaves"),
            contents: bytemuck::cast_slice(&sdf_grid.leaves),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let order_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_order_leaves"),
            contents: bytemuck::cast_slice(&order_grid.leaves),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct DfParams {
            wall_offset: f32,
            target_order: f32,
            leaf_count: u32,
            voxel_dx: f32,
            voxel_dy: f32,
            voxel_dz: f32,
            pad0: f32,
            pad1: f32,
        }

        let params = DfParams {
            wall_offset,
            target_order,
            leaf_count: sdf_grid.leaves.len() as u32,
            voxel_dx: sdf_grid.header.voxel_size[0],
            voxel_dy: sdf_grid.header.voxel_size[1],
            voxel_dz: sdf_grid.header.voxel_size[2],
            pad0: 0.0,
            pad1: 0.0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("df_counter"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let out_lines_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("df_out_lines"),
            size: max_bytes.max(64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("df_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sdf_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: order_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.edge_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.tri_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counter_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out_lines_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("df_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("df_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(sdf_grid.leaves.len() as u32, 1, 1);
        }

        let readback_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("df_readback_counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let readback_lines = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("df_readback_lines"),
            size: max_bytes.max(64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&counter_buf, 0, &readback_counter, 0, 4);
        encoder.copy_buffer_to_buffer(&out_lines_buf, 0, &readback_lines, 0, max_bytes.max(64));

        queue.submit(Some(encoder.finish()));

        let counter_slice = readback_counter.slice(..);
        let lines_slice = readback_lines.slice(..);

        let (tx, rx) = std::sync::mpsc::channel();
        counter_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            return Vec::new();
        }

        let line_count = {
            let data = counter_slice.get_mapped_range();
            let count: u32 = *bytemuck::from_bytes(&data[0..4]);
            count as usize
        };
        readback_counter.unmap();

        if line_count == 0 {
            return Vec::new();
        }

        let (tx_v, rx_v) = std::sync::mpsc::channel();
        lines_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx_v.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if rx_v.recv().ok().and_then(|r| r.ok()).is_none() {
            return Vec::new();
        }

        let segments = {
            let data = lines_slice.get_mapped_range();
            let raw_vecs: &[[f32; 4]] = bytemuck::cast_slice(&data[0..line_count * 2 * 16]);
            raw_vecs
                .chunks_exact(2)
                .map(|pair| {
                    (
                        DVec3::new(pair[0][0] as f64, pair[0][1] as f64, pair[0][2] as f64),
                        DVec3::new(pair[1][0] as f64, pair[1][1] as f64, pair[1][2] as f64),
                    )
                })
                .collect()
        };
        readback_lines.unmap();

        segments
    }
}

/// WGSL shader source for Dual-Field Toolpath Curve Intersection.
const DUAL_FIELD_WGSL: &str = r#"
struct DfParams {
    wall_offset: f32,
    target_order: f32,
    leaf_count: u32,
    voxel_dx: f32,
    voxel_dy: f32,
    voxel_dz: f32,
    pad0: f32,
    pad1: f32,
};

struct NanoLeafBrick {
    origin: vec4<i32>,
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    value_min: f32,
    value_max: f32,
    pad: vec2<f32>,
    values: array<f32, 512>,
};

@group(0) @binding(0) var<uniform> params: DfParams;
@group(0) @binding(1) var<storage, read> sdf_leaves: array<NanoLeafBrick>;
@group(0) @binding(2) var<storage, read> order_leaves: array<NanoLeafBrick>;
@group(0) @binding(3) var<storage, read> edge_table: array<u32, 256>;
@group(0) @binding(4) var<storage, read> tri_table: array<i32>;
@group(0) @binding(5) var<storage, read_write> atomic_counter: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> out_lines: array<vec4<f32>>;

fn get_corner_offset(c: u32) -> vec3<u32> {
    switch (c) {
        case 0u: { return vec3<u32>(0u, 0u, 0u); }
        case 1u: { return vec3<u32>(1u, 0u, 0u); }
        case 2u: { return vec3<u32>(1u, 1u, 0u); }
        case 3u: { return vec3<u32>(0u, 1u, 0u); }
        case 4u: { return vec3<u32>(0u, 0u, 1u); }
        case 5u: { return vec3<u32>(1u, 0u, 1u); }
        case 6u: { return vec3<u32>(1u, 1u, 1u); }
        default: { return vec3<u32>(0u, 1u, 1u); }
    }
}

fn get_edge_corners(e: u32) -> vec2<u32> {
    switch (e) {
        case 0u: { return vec2<u32>(0u, 1u); }
        case 1u: { return vec2<u32>(1u, 2u); }
        case 2u: { return vec2<u32>(2u, 3u); }
        case 3u: { return vec2<u32>(3u, 0u); }
        case 4u: { return vec2<u32>(4u, 5u); }
        case 5u: { return vec2<u32>(5u, 6u); }
        case 6u: { return vec2<u32>(6u, 7u); }
        case 7u: { return vec2<u32>(7u, 4u); }
        case 8u: { return vec2<u32>(0u, 4u); }
        case 9u: { return vec2<u32>(1u, 5u); }
        case 10u: { return vec2<u32>(2u, 6u); }
        default: { return vec2<u32>(3u, 7u); }
    }
}

fn get_leaf_val(leaf_idx: u32, cx: u32, cy: u32, cz: u32, is_order: bool) -> f32 {
    let idx = (cz * 8u + cy) * 8u + cx;
    if (is_order) {
        if (leaf_idx < arrayLength(&order_leaves)) {
            return order_leaves[leaf_idx].values[idx];
        }
        return 0.0;
    }
    return sdf_leaves[leaf_idx].values[idx];
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let leaf_idx = workgroup_id.x;
    if (leaf_idx >= params.leaf_count) {
        return;
    }

    let iso = -params.wall_offset;
    if (sdf_leaves[leaf_idx].value_min > iso || sdf_leaves[leaf_idx].value_max < iso) {
        return;
    }

    let b_min = sdf_leaves[leaf_idx].bbox_min.xyz;
    let d = vec3<f32>(params.voxel_dx, params.voxel_dy, params.voxel_dz);

    for (var cell_idx = local_id.x; cell_idx < 343u; cell_idx = cell_idx + 64u) {
        let cx = cell_idx % 7u;
        let cy = (cell_idx / 7u) % 7u;
        let cz = cell_idx / 49u;

        var corner_sdf: array<f32, 8>;
        var corner_order: array<f32, 8>;
        var corner_pos: array<vec3<f32>, 8>;

        for (var i = 0u; i < 8u; i = i + 1u) {
            let co = get_corner_offset(i);
            let vx = cx + co.x;
            let vy = cy + co.y;
            let vz = cz + co.z;
            corner_sdf[i] = get_leaf_val(leaf_idx, vx, vy, vz, false);
            corner_order[i] = get_leaf_val(leaf_idx, vx, vy, vz, true);
            corner_pos[i] = b_min + vec3<f32>(f32(vx) * d.x, f32(vy) * d.y, f32(vz) * d.z);
        }

        var cube_index: u32 = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            if (corner_sdf[i] < iso) {
                cube_index = cube_index | (1u << i);
            }
        }

        let edge_flags = edge_table[cube_index];
        if (edge_flags == 0u) {
            continue;
        }

        var edge_vertex: array<vec3<f32>, 12>;
        var edge_order: array<f32, 12>;
        for (var e = 0u; e < 12u; e = e + 1u) {
            if ((edge_flags & (1u << e)) != 0u) {
                let corners = get_edge_corners(e);
                let c0 = corners.x;
                let c1 = corners.y;
                let v0 = corner_sdf[c0];
                let v1 = corner_sdf[c1];
                let denom = v1 - v0;
                var t: f32 = 0.5;
                if (abs(denom) > 0.0000001) {
                    t = clamp((iso - v0) / denom, 0.0, 1.0);
                }
                edge_vertex[e] = mix(corner_pos[c0], corner_pos[c1], t);
                edge_order[e] = mix(corner_order[c0], corner_order[c1], t);
            }
        }

        let tri_base = cube_index * 16u;
        for (var i = 0u; i < 15u; i = i + 3u) {
            let e0 = tri_table[tri_base + i];
            if (e0 < 0) {
                break;
            }
            let e1 = tri_table[tri_base + i + 1u];
            let e2 = tri_table[tri_base + i + 2u];

            let p0 = edge_vertex[u32(e0)];
            let p1 = edge_vertex[u32(e1)];
            let p2 = edge_vertex[u32(e2)];

            let ord0 = edge_order[u32(e0)];
            let ord1 = edge_order[u32(e1)];
            let ord2 = edge_order[u32(e2)];

            let target_val = params.target_order;
            var cross_pts: array<vec3<f32>, 2>;
            var cross_count: u32 = 0u;

            // Edge 0-1
            if ((ord0 <= target_val && ord1 >= target_val) || (ord1 <= target_val && ord0 >= target_val)) {
                let d_ord = ord1 - ord0;
                var t_ord: f32 = 0.5;
                if (abs(d_ord) > 0.0000001) {
                    t_ord = clamp((target_val - ord0) / d_ord, 0.0, 1.0);
                }
                cross_pts[cross_count] = mix(p0, p1, t_ord);
                cross_count = cross_count + 1u;
            }

            // Edge 1-2
            if (cross_count < 2u && ((ord1 <= target_val && ord2 >= target_val) || (ord2 <= target_val && ord1 >= target_val))) {
                let d_ord = ord2 - ord1;
                var t_ord: f32 = 0.5;
                if (abs(d_ord) > 0.0000001) {
                    t_ord = clamp((target_val - ord1) / d_ord, 0.0, 1.0);
                }
                cross_pts[cross_count] = mix(p1, p2, t_ord);
                cross_count = cross_count + 1u;
            }

            // Edge 2-0
            if (cross_count < 2u && ((ord2 <= target_val && ord0 >= target_val) || (ord0 <= target_val && ord2 >= target_val))) {
                let d_ord = ord0 - ord2;
                var t_ord: f32 = 0.5;
                if (abs(d_ord) > 0.0000001) {
                    t_ord = clamp((target_val - ord2) / d_ord, 0.0, 1.0);
                }
                cross_pts[cross_count] = mix(p2, p0, t_ord);
                cross_count = cross_count + 1u;
            }

            if (cross_count == 2u) {
                let out_idx = atomicAdd(&atomic_counter, 1u) * 2u;
                out_lines[out_idx] = vec4<f32>(cross_pts[0], 1.0);
                out_lines[out_idx + 1u] = vec4<f32>(cross_pts[1], 1.0);
            }
        }
    }
}
"#;

/// WGSL shader source for Sparse Marching Cubes on NanoVDB leaf bricks.
const SPARSE_MC_WGSL: &str = r#"
struct McParams {
    iso: f32,
    leaf_count: u32,
    voxel_dx: f32,
    voxel_dy: f32,
    voxel_dz: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

struct NanoLeafBrick {
    origin: vec4<i32>,
    bbox_min: vec4<f32>,
    bbox_max: vec4<f32>,
    value_min: f32,
    value_max: f32,
    pad: vec2<f32>,
    values: array<f32, 512>,
};

@group(0) @binding(0) var<uniform> params: McParams;
@group(0) @binding(1) var<storage, read> leaves: array<NanoLeafBrick>;
@group(0) @binding(2) var<storage, read> edge_table: array<u32, 256>;
@group(0) @binding(3) var<storage, read> tri_table: array<i32>;
@group(0) @binding(4) var<storage, read_write> atomic_counter: atomic<u32>;
@group(0) @binding(5) var<storage, read_write> out_vertices: array<vec4<f32>>;

fn get_corner_offset(c: u32) -> vec3<u32> {
    switch (c) {
        case 0u: { return vec3<u32>(0u, 0u, 0u); }
        case 1u: { return vec3<u32>(1u, 0u, 0u); }
        case 2u: { return vec3<u32>(1u, 1u, 0u); }
        case 3u: { return vec3<u32>(0u, 1u, 0u); }
        case 4u: { return vec3<u32>(0u, 0u, 1u); }
        case 5u: { return vec3<u32>(1u, 0u, 1u); }
        case 6u: { return vec3<u32>(1u, 1u, 1u); }
        default: { return vec3<u32>(0u, 1u, 1u); }
    }
}

fn get_edge_corners(e: u32) -> vec2<u32> {
    switch (e) {
        case 0u: { return vec2<u32>(0u, 1u); }
        case 1u: { return vec2<u32>(1u, 2u); }
        case 2u: { return vec2<u32>(2u, 3u); }
        case 3u: { return vec2<u32>(3u, 0u); }
        case 4u: { return vec2<u32>(4u, 5u); }
        case 5u: { return vec2<u32>(5u, 6u); }
        case 6u: { return vec2<u32>(6u, 7u); }
        case 7u: { return vec2<u32>(7u, 4u); }
        case 8u: { return vec2<u32>(0u, 4u); }
        case 9u: { return vec2<u32>(1u, 5u); }
        case 10u: { return vec2<u32>(2u, 6u); }
        default: { return vec2<u32>(3u, 7u); }
    }
}

fn get_voxel_val(leaf_idx: u32, cx: u32, cy: u32, cz: u32) -> f32 {
    let idx = (cz * 8u + cy) * 8u + cx;
    return leaves[leaf_idx].values[idx];
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let leaf_idx = workgroup_id.x;
    if (leaf_idx >= params.leaf_count) {
        return;
    }

    if (leaves[leaf_idx].value_min > params.iso || leaves[leaf_idx].value_max < params.iso) {
        return;
    }

    let b_min = leaves[leaf_idx].bbox_min.xyz;
    let d = vec3<f32>(params.voxel_dx, params.voxel_dy, params.voxel_dz);

    for (var cell_idx = local_id.x; cell_idx < 343u; cell_idx = cell_idx + 64u) {
        let cx = cell_idx % 7u;
        let cy = (cell_idx / 7u) % 7u;
        let cz = cell_idx / 49u;

        var corner_val: array<f32, 8>;
        var corner_pos: array<vec3<f32>, 8>;

        for (var i = 0u; i < 8u; i = i + 1u) {
            let co = get_corner_offset(i);
            let vx = cx + co.x;
            let vy = cy + co.y;
            let vz = cz + co.z;
            corner_val[i] = get_voxel_val(leaf_idx, vx, vy, vz);
            corner_pos[i] = b_min + vec3<f32>(f32(vx) * d.x, f32(vy) * d.y, f32(vz) * d.z);
        }

        var cube_index: u32 = 0u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            if (corner_val[i] < params.iso) {
                cube_index = cube_index | (1u << i);
            }
        }

        let edge_flags = edge_table[cube_index];
        if (edge_flags == 0u) {
            continue;
        }

        var edge_vertex: array<vec3<f32>, 12>;
        for (var e = 0u; e < 12u; e = e + 1u) {
            if ((edge_flags & (1u << e)) != 0u) {
                let corners = get_edge_corners(e);
                let c0 = corners.x;
                let c1 = corners.y;
                let v0 = corner_val[c0];
                let v1 = corner_val[c1];
                let denom = v1 - v0;
                var t: f32 = 0.5;
                if (abs(denom) > 0.0000001) {
                    t = clamp((params.iso - v0) / denom, 0.0, 1.0);
                }
                edge_vertex[e] = mix(corner_pos[c0], corner_pos[c1], t);
            }
        }

        let tri_base = cube_index * 16u;
        for (var i = 0u; i < 15u; i = i + 3u) {
            let e0 = tri_table[tri_base + i];
            if (e0 < 0) {
                break;
            }
            let e1 = tri_table[tri_base + i + 1u];
            let e2 = tri_table[tri_base + i + 2u];

            let out_idx = atomicAdd(&atomic_counter, 3u);
            out_vertices[out_idx] = vec4<f32>(edge_vertex[u32(e0)], 1.0);
            out_vertices[out_idx + 1u] = vec4<f32>(edge_vertex[u32(e1)], 1.0);
            out_vertices[out_idx + 2u] = vec4<f32>(edge_vertex[u32(e2)], 1.0);
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sphere_tree, TreeField};

    #[test]
    fn gpu_marching_cubes_extracts_sphere() {
        let Some(ctx) = GpuContext::new() else {
            eprintln!("Skipping GPU test: no compatible GPU adapter available.");
            return;
        };

        let tree = TreeField::new(sphere_tree(1.0));
        let grid = NanoGridBuffer::build_from_scalar_field(
            &tree,
            DVec3::splat(-1.5),
            DVec3::splat(1.5),
            0.1,
            0.0,
            0.2,
        );

        let mc = GpuSparseMarchingCubes::new(ctx);
        let verts = mc.extract_isosurface_positions(&grid, 0.0);

        assert!(!verts.is_empty(), "GPU MC produced no vertices");
        assert_eq!(verts.len() % 3, 0);

        for p in &verts {
            let r = p.length();
            assert!((r - 1.0).abs() < 0.15, "vertex radius {} off 1.0", r);
        }
    }

    #[test]
    fn gpu_dual_field_toolpath_extracts_circle_contour() {
        use crate::order::HeightOrderField;

        let Some(ctx) = GpuContext::new() else {
            eprintln!("Skipping GPU test: no compatible GPU adapter available.");
            return;
        };

        let tree = TreeField::new(sphere_tree(1.0));
        let order = HeightOrderField::new(DVec3::Z);

        let sdf_grid = NanoGridBuffer::build_from_scalar_field(
            &tree,
            DVec3::splat(-1.5),
            DVec3::splat(1.5),
            0.1,
            0.0,
            0.2,
        );

        let order_grid = NanoGridBuffer::build_from_scalar_field(
            &order,
            DVec3::splat(-1.5),
            DVec3::splat(1.5),
            0.1,
            0.0,
            1.5,
        );

        let df = GpuDualFieldToolpath::new(ctx);
        let segments = df.extract_toolpath_segments(&sdf_grid, &order_grid, 0.0, 0.0);

        assert!(
            !segments.is_empty(),
            "GPU Dual-Field produced no line segments"
        );
        // All segment endpoints on equatorial cut (z=0) of unit sphere should have z~0 and radius~1
        for (p0, p1) in &segments {
            assert!(p0.z.abs() < 0.15, "p0.z off 0: {}", p0.z);
            assert!(p1.z.abs() < 0.15, "p1.z off 0: {}", p1.z);
            let r0 = (p0.x * p0.x + p0.y * p0.y).sqrt();
            let r1 = (p1.x * p1.x + p1.y * p1.y).sqrt();
            assert!((r0 - 1.0).abs() < 0.15, "p0 radius off 1.0: {r0}");
            assert!((r1 - 1.0).abs() < 0.15, "p1 radius off 1.0: {r1}");
        }
    }
}
