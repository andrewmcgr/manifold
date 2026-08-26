// Fullscreen Volume Raymarching shader for NanoVDB sparse grids.

struct CameraUniform {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    _pad: f32,
};

struct RaymarchParams {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    iso: f32,
    leaf_count: u32,
    max_steps: u32,
    step_size: f32,
};

struct NanoGridHeader {
    magic: u32,
    grid_type: u32,
    leaf_count: u32,
    pad0: u32,
    world_bbox_min: vec4<f32>,
    world_bbox_max: vec4<f32>,
    voxel_size: vec4<f32>,
    index_bbox_min: vec4<i32>,
    index_bbox_max: vec4<i32>,
    background_value: f32,
    pad1: vec3<f32>,
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

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> params: RaymarchParams;
@group(0) @binding(2) var<uniform> header: NanoGridHeader;
@group(0) @binding(3) var<storage, read> leaves: array<NanoLeafBrick>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate fullscreen triangle: (0,0), (2,0), (0,2) in clip space [-1, 1]
    let x = f32((in_vertex_index << 1u) & 2u);
    let y = f32(in_vertex_index & 2u);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn sample_grid(world_p: vec3<f32>) -> f32 {
    let p_rel = world_p - header.world_bbox_min.xyz;
    let vx = p_rel.x / header.voxel_size.x;
    let vy = p_rel.y / header.voxel_size.y;
    let vz = p_rel.z / header.voxel_size.z;

    if (vx < 0.0 || vy < 0.0 || vz < 0.0) {
        return header.background_value;
    }

    let ix = i32(floor(vx));
    let iy = i32(floor(vy));
    let iz = i32(floor(vz));

    let bx = ix / 8;
    let by = iy / 8;
    let bz = iz / 8;

    // Search active leaf bricks
    for (var i = 0u; i < params.leaf_count; i = i + 1u) {
        let leaf = &leaves[i];
        if ((*leaf).origin.x / 8 == bx && (*leaf).origin.y / 8 == by && (*leaf).origin.z / 8 == bz) {
            let lx = u32(ix - (*leaf).origin.x);
            let ly = u32(iy - (*leaf).origin.y);
            let lz = u32(iz - (*leaf).origin.z);
            if (lx < 8u && ly < 8u && lz < 8u) {
                return (*leaf).values[(lz * 8u + ly) * 8u + lx];
            }
        }
    }

    return header.background_value;
}

fn compute_gradient(p: vec3<f32>) -> vec3<f32> {
    let h = max(header.voxel_size.x, 0.01);
    let gx = sample_grid(p + vec3<f32>(h, 0.0, 0.0)) - sample_grid(p - vec3<f32>(h, 0.0, 0.0));
    let gy = sample_grid(p + vec3<f32>(0.0, h, 0.0)) - sample_grid(p - vec3<f32>(0.0, h, 0.0));
    let gz = sample_grid(p + vec3<f32>(0.0, 0.0, h)) - sample_grid(p - vec3<f32>(0.0, 0.0, h));
    let g = vec3<f32>(gx, gy, gz) / (2.0 * h);
    if (length(g) > 0.0001) {
        return normalize(g);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world ray from inverse view-projection
    let ndc = vec4<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
    let world_target_h = params.inv_view_proj * ndc;
    let world_target = world_target_h.xyz / world_target_h.w;
    let ray_origin = params.camera_pos.xyz;
    let ray_dir = normalize(world_target - ray_origin);

    var t: f32 = 0.5;
    let step = max(params.step_size, 0.05);

    for (var i = 0u; i < params.max_steps; i = i + 1u) {
        let p = ray_origin + ray_dir * t;
        let d = sample_grid(p);

        if (d <= params.iso) {
            // Hit surface: compute lighting
            let normal = compute_gradient(p);
            let light_dir = normalize(vec3<f32>(0.5, 0.8, 1.0));
            let diff = max(dot(normal, light_dir), 0.15);
            let base_color = vec3<f32>(0.2, 0.6, 0.9);
            let color = base_color * diff + vec3<f32>(0.1, 0.1, 0.15);
            return vec4<f32>(color, 0.95);
        }

        t = t + max(step, d * 0.5);
        if (t > 500.0) {
            break;
        }
    }

    discard;
}
