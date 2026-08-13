// Minimal flat-shaded triangle-mesh shader.
//
// Vertices are pre-expanded to a non-indexed triangle list on the CPU side
// (`render.rs`), each carrying a per-triangle face normal, since
// `manifold_core::mesh::Mesh` stores no normal data of its own.

struct Camera {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.4, 0.6, 0.8));
    let ambient = 0.25;
    let diffuse = max(dot(normalize(in.normal), light_dir), 0.0);
    let intensity = ambient + (1.0 - ambient) * diffuse;
    let base_color = vec3<f32>(0.65, 0.68, 0.72);
    return vec4<f32>(base_color * intensity, 1.0);
}
