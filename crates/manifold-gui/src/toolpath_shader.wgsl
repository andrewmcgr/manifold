// Minimal unlit, per-vertex-colored shader for toolpath preview rendering
// (Phase 13, see ROADMAP.md). Structurally identical to `scene_shader.wgsl`
// except the vertex layout carries an extra trailing `order` attribute
// (used for CPU-side scrub filtering today; reserved for a future
// shader-side discard against a scrub uniform, see toolpath_view.rs's
// `ToolpathVertex` doc comment) which this shader accepts but does not yet
// use in either stage.
//
// Shares the same camera uniform binding/layout as `mesh_shader.wgsl` /
// `scene_shader.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) order: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
