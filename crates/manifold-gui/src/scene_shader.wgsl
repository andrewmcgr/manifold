// Minimal unlit, per-vertex-colored shader for scene dressing (origin
// axes, bed grid/quad, toolhead markers) — see ROADMAP.md Phase 6.
//
// Shares the same camera uniform binding/layout as `mesh_shader.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct TriVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct LineInstanceInput {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_tri(in: TriVertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@vertex
fn vs_line(
    @builtin(vertex_index) in_vertex_index: u32,
    in: LineInstanceInput,
) -> VertexOutput {
    var side = -1.0;
    if (in_vertex_index == 1u || in_vertex_index == 4u || in_vertex_index == 5u) {
        side = 1.0;
    }
    var u = 0.0;
    if (in_vertex_index == 2u || in_vertex_index == 3u || in_vertex_index == 5u) {
        u = 1.0;
    }

    var clip0 = camera.view_proj * vec4<f32>(in.start, 1.0);
    var clip1 = camera.view_proj * vec4<f32>(in.end, 1.0);

    // Near-plane clipping in homogeneous clip coordinates to prevent behind-camera distortion
    let w_near = 0.001;
    if (clip0.w < w_near && clip1.w > w_near) {
        let t = (w_near - clip0.w) / (clip1.w - clip0.w);
        clip0 = mix(clip0, clip1, t);
    } else if (clip1.w < w_near && clip0.w > w_near) {
        let t = (w_near - clip1.w) / (clip0.w - clip1.w);
        clip1 = mix(clip1, clip0, t);
    }

    let p0_ndc = clip0.xy / clip0.w;
    let p1_ndc = clip1.xy / clip1.w;

    // Screen-space direction in pixels
    let screen_delta = (p1_ndc - p0_ndc) * camera.viewport_size * 0.5;
    let len = length(screen_delta);
    var screen_norm = vec2<f32>(0.0, 0.0);
    if (len > 0.0001) {
        screen_norm = vec2<f32>(-screen_delta.y, screen_delta.x) / len;
    }

    // Offset in screen pixels converted to NDC
    let offset_ndc = screen_norm * (side * camera.line_width * 0.5) / (camera.viewport_size * 0.5);

    let p_clip = mix(clip0, clip1, u);
    var out: VertexOutput;
    out.clip_position = vec4<f32>(p_clip.xy + offset_ndc * p_clip.w, p_clip.z, p_clip.w);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
