//! Scene dressing geometry: origin axes, print bed/grid, toolhead markers
//! (Phase 6, see ROADMAP.md). Pure geometry builders operating on
//! `manifold_core::machine::Machine` — no GPU/wgpu types here, kept
//! separate from `render.rs`'s GPU upload/pipeline concerns.

use glam::DVec3;
use manifold_core::machine::Machine;

/// One vertex for the unlit scene-dressing shader: position + RGBA color.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct SceneVertex {
    position: [f32; 3],
    color: [f32; 4],
}

impl SceneVertex {
    pub fn new(position: DVec3, color: [f32; 4]) -> Self {
        Self {
            position: position.as_vec3().to_array(),
            color,
        }
    }
}

/// One line segment instance for the unlit scene-dressing line shader.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct SceneLineInstance {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 4],
}

impl SceneLineInstance {
    pub fn new(start: DVec3, end: DVec3, color: [f32; 4]) -> Self {
        Self {
            start: start.as_vec3().to_array(),
            end: end.as_vec3().to_array(),
            color,
        }
    }
}

const AXIS_RED: [f32; 4] = [0.85, 0.15, 0.15, 1.0];
const AXIS_GREEN: [f32; 4] = [0.15, 0.75, 0.15, 1.0];
const AXIS_BLUE: [f32; 4] = [0.15, 0.35, 0.9, 1.0];
const GRID_COLOR: [f32; 4] = [0.55, 0.55, 0.55, 1.0];
const BED_COLOR: [f32; 4] = [0.3, 0.3, 0.32, 0.35];
const TOOLHEAD_COLOR: [f32; 4] = [0.95, 0.55, 0.1, 1.0];

/// A fixed-size RGB axis triad at the world origin (X=red, Y=green,
/// Z=blue), as a line-instance buffer.
pub fn build_origin_axes(length: f64) -> Vec<SceneLineInstance> {
    vec![
        SceneLineInstance::new(DVec3::ZERO, DVec3::new(length, 0.0, 0.0), AXIS_RED),
        SceneLineInstance::new(DVec3::ZERO, DVec3::new(0.0, length, 0.0), AXIS_GREEN),
        SceneLineInstance::new(DVec3::ZERO, DVec3::new(0.0, 0.0, length), AXIS_BLUE),
    ]
}

/// A ground-plane grid over the machine's build volume XY extent, at the
/// substrate's Z, as a line-instance buffer.
pub fn build_grid(machine: &Machine, spacing: f64) -> Vec<SceneLineInstance> {
    let (min, max) = machine.build_volume.bounding_box();
    let z = min.z;
    let mut lines = Vec::new();

    let mut x = min.x;
    while x <= max.x {
        lines.push(SceneLineInstance::new(
            DVec3::new(x, min.y, z),
            DVec3::new(x, max.y, z),
            GRID_COLOR,
        ));
        x += spacing;
    }
    let mut y = min.y;
    while y <= max.y {
        lines.push(SceneLineInstance::new(
            DVec3::new(min.x, y, z),
            DVec3::new(max.x, y, z),
            GRID_COLOR,
        ));
        y += spacing;
    }

    lines
}

/// A translucent quad filling the machine's build volume XY extent, at
/// the substrate's Z, as a triangle-list vertex buffer.
pub fn build_bed_quad(machine: &Machine) -> Vec<SceneVertex> {
    let (min, max) = machine.build_volume.bounding_box();
    let z = min.z;
    let corners = [
        DVec3::new(min.x, min.y, z),
        DVec3::new(max.x, min.y, z),
        DVec3::new(max.x, max.y, z),
        DVec3::new(min.x, max.y, z),
    ];

    [
        corners[0], corners[1], corners[2], // first triangle
        corners[0], corners[2], corners[3], // second triangle
    ]
    .into_iter()
    .map(|p| SceneVertex::new(p, BED_COLOR))
    .collect()
}

/// A small pyramid marker at each of the machine's tools' mount
/// translations, as a triangle-list vertex buffer.
pub fn build_toolhead_markers(machine: &Machine, size: f64) -> Vec<SceneVertex> {
    let mut vertices = Vec::new();
    for tool in &machine.tools {
        let base = tool.mount.transform_point(DVec3::ZERO);
        let apex = base + DVec3::new(0.0, 0.0, size);
        let half = size * 0.5;
        let corners = [
            base + DVec3::new(-half, -half, 0.0),
            base + DVec3::new(half, -half, 0.0),
            base + DVec3::new(half, half, 0.0),
            base + DVec3::new(-half, half, 0.0),
        ];

        // Four side faces of the pyramid, apex pointing up.
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            vertices.push(SceneVertex::new(a, TOOLHEAD_COLOR));
            vertices.push(SceneVertex::new(b, TOOLHEAD_COLOR));
            vertices.push(SceneVertex::new(apex, TOOLHEAD_COLOR));
        }
    }
    vertices
}
