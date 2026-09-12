// Unlit textured-quad shader for object name labels (see
// `scene::build_object_labels` / `render::UploadedScene`). The rasterized
// glyph atlas is bound as a read-only storage buffer of per-pixel coverage
// values (0.0..=1.0) rather than a texture, so uploading it needs only
// `wgpu::Device` (no `wgpu::Queue`), matching every other scene-dressing
// buffer's upload path in this crate.
//
// Shares the same camera uniform binding/layout as `scene_shader.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    render_mode: f32,
    camera_pos: vec3<f32>,
    bed_z: f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct AtlasInfo {
    width: u32,
    height: u32,
}

@group(1) @binding(0)
var<storage, read> atlas_pixels: array<f32>;
@group(1) @binding(1)
var<uniform> atlas_info: AtlasInfo;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let w = atlas_info.width;
    let h = atlas_info.height;
    let x = clamp(u32(in.uv.x * f32(w)), 0u, w - 1u);
    let y = clamp(u32(in.uv.y * f32(h)), 0u, h - 1u);
    let coverage = atlas_pixels[y * w + x];
    if (coverage < 0.02) {
        discard;
    }
    return vec4<f32>(1.0, 1.0, 1.0, coverage);
}
