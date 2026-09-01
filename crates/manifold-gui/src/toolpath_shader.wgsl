// Physically-scaled 3D volumetric extrusion ribbon shader with cylindrical
// bead cross-section lighting and screen-space fallback for travel lines.
//
// Shares the same camera uniform binding/layout as `mesh_shader.wgsl` /
// `scene_shader.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    render_mode: f32, // 0.0 = Physical 3D extrusion, 1.0 = Screen-space lines
}

@group(0) @binding(0)
var<uniform> camera: Camera;

struct ToolpathInstanceInput {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) order: f32,
    @location(4) width: f32,
    @location(5) height: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>, // u along move [0, 1], v across width [-1, +1]
    @location(2) normal: vec3<f32>,
    @location(3) @interpolate(flat) is_travel: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    in: ToolpathInstanceInput,
) -> VertexOutput {
    var side = -1.0;
    if (in_vertex_index == 1u || in_vertex_index == 4u || in_vertex_index == 5u) {
        side = 1.0;
    }
    var u = 0.0;
    if (in_vertex_index == 2u || in_vertex_index == 3u || in_vertex_index == 5u) {
        u = 1.0;
    }

    var out: VertexOutput;
    out.color = in.color;
    out.uv = vec2<f32>(u, side);

    // If segment has physical extrusion width and physical 3D mode is active:
    if (in.width > 0.001 && camera.render_mode < 0.5) {
        let delta = in.end - in.start;
        let len = length(delta);
        var dir = vec3<f32>(1.0, 0.0, 0.0);
        if (len > 0.0001) {
            dir = delta / len;
        }

        // Compute local coordinate frame: tangent, binormal (transverse across width), normal (up along layer height)
        var up = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(dir.z) > 0.95) {
            up = vec3<f32>(0.0, 1.0, 0.0);
        }
        let binormal = normalize(cross(dir, up));
        let normal = normalize(cross(binormal, dir));

        let p_center = mix(in.start, in.end, u);
        let half_w = in.width * 0.5;
        let half_h = in.height * 0.5;

        // Cylindrical/elliptical top surface arch
        let v_arch = sqrt(max(0.0, 1.0 - side * side * 0.75));
        let p_world = p_center + binormal * (side * half_w) + normal * (half_h * v_arch);

        out.clip_position = camera.view_proj * vec4<f32>(p_world, 1.0);
        out.normal = normalize(normal * 0.7 + binormal * (side * 0.7));
        out.is_travel = 0.0;
        return out;
    }

    // Screen-space constant pixel thickness fallback (for travel moves or screen-space line mode)
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

    let line_px = select(camera.line_width * 0.75, camera.line_width, in.width > 0.001);
    let offset_ndc = screen_norm * (side * line_px * 0.5) / (camera.viewport_size * 0.5);

    let p_clip = mix(clip0, clip1, u);
    out.clip_position = vec4<f32>(p_clip.xy + offset_ndc * p_clip.w, p_clip.z, p_clip.w);
    out.normal = vec3<f32>(0.0, 0.0, 1.0);
    out.is_travel = 1.0;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.is_travel > 0.5) {
        return in.color;
    }

    // 3D Extrusion bead shading: rounded cross-section lighting
    let v = in.uv.y;
    let roundness = clamp(1.0 - v * v * 0.35, 0.0, 1.0);

    let light_dir = normalize(vec3<f32>(0.35, 0.45, 0.82));
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    let lighting = 0.60 + 0.40 * diffuse;

    let rgb = in.color.rgb * lighting * roundness;
    return vec4<f32>(rgb, in.color.a);
}
