// Physically-scaled 3D volumetric extrusion shader: each bead segment is a
// rectangular prism (24 vertices, 4 faces), not a flat ribbon, with
// per-face lighting and a screen-space fallback for travel lines.
//
// Shares the same camera uniform binding/layout as `mesh_shader.wgsl` /
// `scene_shader.wgsl`.

struct Camera {
    view_proj: mat4x4<f32>,
    viewport_size: vec2<f32>,
    line_width: f32,
    render_mode: f32, // 0.0 = Physical 3D extrusion, 1.0 = Screen-space lines
    camera_pos: vec3<f32>,
    bed_z: f32,
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
    @location(2) normal: vec3<f32>,
    @location(3) @interpolate(flat) is_travel: f32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    in: ToolpathInstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = in.color;

    let is_physical = in.width > 0.001 && camera.render_mode < 0.5;

    if (is_physical) {
        let delta = in.end - in.start;
        let len = length(delta);
        var dir = vec3<f32>(1.0, 0.0, 0.0);
        if (len > 0.0001) {
            dir = delta / len;
        }

        // Local coordinate frame: tangent (dir), binormal (transverse across width), normal (up along layer height).
        var up = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(dir.z) > 0.95) {
            up = vec3<f32>(0.0, 1.0, 0.0);
        }
        let binormal = normalize(cross(dir, up));
        let normal = normalize(cross(binormal, dir));
        // This (dir, binormal, normal) triad is left-handed (normal ==
        // -(dir x binormal)). Harmless today: every face_normal below is
        // assigned explicitly by name, not derived from triangle winding,
        // and toolpath_line_pipeline uses cull_mode: None (render.rs) so
        // winding doesn't affect visibility either. If backface culling is
        // ever enabled here, this handedness must be fixed first or half
        // the box's faces will disappear.

        let half_w = in.width * 0.5;
        let height = in.height;

        // Rectangular prism, not a flat quad: `in.start`/`in.end` sit at
        // the bead's TOP surface (the layer's own order/Z value -- see
        // `extrusion::local_layer_geometry`'s doc comment for why), so
        // the box extends downward by the full `height` to the previous
        // layer. 4 faces (top, bottom, left wall, right wall) x 6
        // vertices (2 triangles) each = 24 vertices, built entirely from
        // `in_vertex_index` -- no new CPU-side buffers needed beyond the
        // width/height already carried per instance.
        let face = in_vertex_index / 6u;
        let local = in_vertex_index % 6u;

        // Standard 2-triangle quad from a local index 0..5: 0=u0/a,
        // 1=u1/a, 2=u0/b, 3=u1/a, 4=u1/b, 5=u0/b (a, b = the face's two rails).
        var vu = 0.0;
        if (local == 1u || local == 3u || local == 4u) {
            vu = 1.0;
        }
        var rail_b = false;
        if (local == 2u || local == 4u || local == 5u) {
            rail_b = true;
        }

        // Each face's two rails as (binormal offset, normal offset) pairs,
        // plus that face's outward normal for flat per-face shading.
        var bw_a = 0.0;
        var nh_a = 0.0;
        var bw_b = 0.0;
        var nh_b = 0.0;
        var face_normal = normal;
        if (face == 0u) {
            // Top: left rail -> right rail, at the bead's top surface.
            bw_a = -half_w;
            nh_a = 0.0;
            bw_b = half_w;
            nh_b = 0.0;
            face_normal = normal;
        } else if (face == 1u) {
            // Bottom: right rail -> left rail, one full layer height below the top.
            bw_a = half_w;
            nh_a = -height;
            bw_b = -half_w;
            nh_b = -height;
            face_normal = -normal;
        } else if (face == 2u) {
            // Left wall: bottom -> top.
            bw_a = -half_w;
            nh_a = -height;
            bw_b = -half_w;
            nh_b = 0.0;
            face_normal = -binormal;
        } else {
            // Right wall: top -> bottom.
            bw_a = half_w;
            nh_a = 0.0;
            bw_b = half_w;
            nh_b = -height;
            face_normal = binormal;
        }

        var bw = bw_a;
        var nh = nh_a;
        if (rail_b) {
            bw = bw_b;
            nh = nh_b;
        }

        let p_center = mix(in.start, in.end, vu);
        let p_world = p_center + binormal * bw + normal * nh;

        out.clip_position = camera.view_proj * vec4<f32>(p_world, 1.0);
        out.normal = face_normal;
        out.is_travel = 0.0;
        return out;
    }

    // Screen-space constant-pixel-thickness fallback (travel moves, or
    // screen-space line mode): only needs 6 vertices for a flat billboard
    // quad. The physical branch above may emit up to 24 vertices per
    // instance, so indices 6..23 here collapse every vertex of those extra
    // "triangles" onto the same point (`in.start`), giving zero screen
    // area instead of redrawing the same quad multiple times -- which
    // would visibly darken alpha-blended travel-move color (`COLOR_TRAVEL`
    // has alpha 0.65) each time it overlapped itself.
    if (in_vertex_index >= 6u) {
        out.clip_position = camera.view_proj * vec4<f32>(in.start, 1.0);
        out.normal = vec3<f32>(0.0, 0.0, 1.0);
        out.is_travel = 1.0;
        return out;
    }

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

    // Flat per-face lighting: each box face carries its own outward normal
    // (see `vs_main`'s physical-3D branch), so directional shading alone
    // gives each face a distinct, correct-looking shade.
    let light_dir = normalize(vec3<f32>(0.35, 0.45, 0.82));
    let diffuse = max(dot(in.normal, light_dir), 0.0);
    let lighting = 0.55 + 0.45 * diffuse;

    let rgb = in.color.rgb * lighting;
    return vec4<f32>(rgb, in.color.a);
}
