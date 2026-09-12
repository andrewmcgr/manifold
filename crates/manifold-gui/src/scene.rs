//! Scene dressing geometry: origin axes, print bed/grid, toolhead markers
//! (Phase 6, see ROADMAP.md). Pure geometry builders operating on
//! `manifold_core::machine::Machine` — no GPU/wgpu types here, kept
//! separate from `render.rs`'s GPU upload/pipeline concerns.

use glam::DVec3;
use manifold_core::machine::Machine;
use manifold_core::object::Object;

use crate::text_raster::TextAtlas;

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
/// Translucent grey used for object footprint outlines on the bed — kept
/// low-alpha so the line doesn't visually compete with the toolpath/mesh
/// preview when viewed from below the bed.
const FOOTPRINT_COLOR: [f32; 4] = [0.6, 0.6, 0.6, 0.4];

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

/// A 1mm-thick closed outline (drawn as a quad ring, since line-instance
/// segments have no configurable width) tracing each object's convex XY
/// footprint at the bed's Z, as a triangle-list vertex buffer. Objects whose
/// mesh has fewer than 4 vertices (no valid hull) are silently skipped.
pub fn build_footprint_outlines(machine: &Machine, objects: &[Object]) -> Vec<SceneVertex> {
    const LINE_WIDTH: f64 = 1.0;
    let (bed_min, _) = machine.build_volume.bounding_box();
    let z = bed_min.z;

    let mut vertices = Vec::new();
    for object in objects {
        let Some(polygon) = object.footprint_polygon() else {
            continue;
        };
        let n = polygon.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = polygon[i];
            let b = polygon[(i + 1) % n];
            let edge = b - a;
            let len = edge.length();
            if len < 1e-9 {
                continue;
            }
            // Outward normal (polygon is CCW from the XY hull projection, so
            // rotating the edge direction -90 degrees points outward).
            let normal = glam::DVec2::new(edge.y, -edge.x) / len * (LINE_WIDTH * 0.5);

            let a_out = DVec3::new(a.x + normal.x, a.y + normal.y, z);
            let a_in = DVec3::new(a.x - normal.x, a.y - normal.y, z);
            let b_out = DVec3::new(b.x + normal.x, b.y + normal.y, z);
            let b_in = DVec3::new(b.x - normal.x, b.y - normal.y, z);

            vertices.push(SceneVertex::new(a_out, FOOTPRINT_COLOR));
            vertices.push(SceneVertex::new(b_out, FOOTPRINT_COLOR));
            vertices.push(SceneVertex::new(b_in, FOOTPRINT_COLOR));

            vertices.push(SceneVertex::new(a_out, FOOTPRINT_COLOR));
            vertices.push(SceneVertex::new(b_in, FOOTPRINT_COLOR));
            vertices.push(SceneVertex::new(a_in, FOOTPRINT_COLOR));
        }
    }
    vertices
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


/// One vertex for the textured object-label shader: position + atlas UV.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct SceneTextVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl SceneTextVertex {
    pub fn new(position: DVec3, uv: [f32; 2]) -> Self {
        Self { position: position.as_vec3().to_array(), uv }
    }
}

/// World-space cap height of a rendered object name label, in millimeters.
const LABEL_HEIGHT_MM: f64 = 5.0;
/// Gap between an object's footprint (at its minimum-Y edge) and the label
/// text placed just outside it, in millimeters.
const LABEL_CLEARANCE_MM: f64 = 1.5;
/// Small Z lift above the footprint outline's plane to prevent z-fighting
/// between the label quad and the footprint/bed quads.
const LABEL_Z_EPSILON: f64 = 0.02;

/// A textured quad per object, baseline-aligned to world X (not rotated with
/// the object), centered on the object footprint's X extent, and placed just
/// outside the footprint's minimum-Y edge — i.e. a name tag sitting in front
/// of each object on the bed. Objects with no footprint (or not present in
/// `atlas`, e.g. an empty mesh) are silently skipped.
pub fn build_object_labels(
    machine: &Machine,
    objects: &[Object],
    atlas: &TextAtlas,
) -> Vec<SceneTextVertex> {
    let (bed_min, _) = machine.build_volume.bounding_box();
    let z = bed_min.z + LABEL_Z_EPSILON;
    let atlas_width_px = atlas.width_px.max(1) as f64;
    let atlas_height_px = atlas.height_px.max(1) as f64;
    let mm_per_px = LABEL_HEIGHT_MM / atlas_height_px;

    let mut vertices = Vec::new();
    for (object, metrics) in objects.iter().zip(&atlas.labels) {
        let Some(polygon) = object.footprint_polygon() else {
            continue;
        };
        if polygon.len() < 3 {
            continue;
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        for p in &polygon {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
        }

        let width_mm = metrics.width_px as f64 * mm_per_px;
        let center_x = (min_x + max_x) * 0.5;
        let x0 = center_x - width_mm * 0.5;
        let x1 = x0 + width_mm;
        let y_top = min_y - LABEL_CLEARANCE_MM;
        let y_bottom = y_top - LABEL_HEIGHT_MM;

        let u0 = metrics.x_offset_px as f32 / atlas_width_px as f32;
        let u1 = (metrics.x_offset_px + metrics.width_px) as f32 / atlas_width_px as f32;

        let p00 = DVec3::new(x0, y_bottom, z);
        let p10 = DVec3::new(x1, y_bottom, z);
        let p11 = DVec3::new(x1, y_top, z);
        let p01 = DVec3::new(x0, y_top, z);

        // v=0 at the raster's top row (glyph ascent), v=1 at the bottom
        // (glyph descent), matching the atlas's row-major top-down layout so
        // the label reads upright when viewed from above the bed (+Y up).
        vertices.push(SceneTextVertex::new(p01, [u0, 0.0]));
        vertices.push(SceneTextVertex::new(p11, [u1, 0.0]));
        vertices.push(SceneTextVertex::new(p10, [u1, 1.0]));

        vertices.push(SceneTextVertex::new(p01, [u0, 0.0]));
        vertices.push(SceneTextVertex::new(p10, [u1, 1.0]));
        vertices.push(SceneTextVertex::new(p00, [u0, 1.0]));
    }
    vertices
}
