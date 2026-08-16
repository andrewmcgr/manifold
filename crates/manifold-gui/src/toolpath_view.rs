//! Toolpath preview geometry: converts `manifold_core::toolpath::Path`
//! data into line-list vertices colored by `MoveKind` (Phase 13, see
//! ROADMAP.md). Pure geometry builders — no GPU/wgpu types here, kept
//! separate from `render.rs`'s GPU upload/pipeline concerns, mirroring
//! `scene.rs`'s existing separation.
//!

use manifold_core::toolpath::{MoveKind, Path};

/// One vertex for the unlit toolpath line shader: position + RGBA color +
/// the source segment's `order` value (carried per-vertex so the scrub
/// filter can operate either CPU-side, before upload, or shader-side, via
/// a uniform threshold against this field).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ToolpathVertex {
    position: [f32; 3],
    color: [f32; 4],
    order: f32,
}

impl ToolpathVertex {
    fn new(position: glam::DVec3, color: [f32; 4], order: f64) -> Self {
        Self {
            position: position.as_vec3().to_array(),
            color,
            order: order as f32,
        }
    }
}

const COLOR_WALL_OUTER: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const COLOR_WALL_INNER: [f32; 4] = [0.6, 0.8, 1.0, 1.0];
const COLOR_INFILL: [f32; 4] = [0.95, 0.65, 0.15, 1.0];
const COLOR_BRIDGE: [f32; 4] = [0.9, 0.2, 0.75, 1.0];
const COLOR_OVERHANG: [f32; 4] = [0.9, 0.15, 0.15, 1.0];
const COLOR_TRAVEL: [f32; 4] = [0.4, 0.4, 0.4, 0.4];

/// Fixed `MoveKind` -> RGBA color palette used for toolpath preview
/// rendering.
fn palette_color(kind: MoveKind) -> [f32; 4] {
    match kind {
        MoveKind::WallOuter => COLOR_WALL_OUTER,
        MoveKind::WallInner => COLOR_WALL_INNER,
        MoveKind::Infill => COLOR_INFILL,
        MoveKind::Bridge => COLOR_BRIDGE,
        MoveKind::Overhang => COLOR_OVERHANG,
        MoveKind::Travel => COLOR_TRAVEL,
    }
}

/// Build a line-list vertex buffer from a set of planned toolpaths: one
/// line segment per `Segment` (`points[i] -> points[(i + 1) %
/// points.len()]`), colored by `segment.kind` via [`palette_color`], with
/// `segment.order` carried on both endpoint vertices.
///
/// `scrub_order` implements the order-based scrub slider's "up to and
/// including" semantics (Phase 13 subtask 05): only segments with
/// `segment.order <= scrub_order` are included. Pass `f64::INFINITY` to
/// include every segment (no filtering).
///
/// This is the **CPU-side rebuild-on-change** approach: the caller
/// (`app.rs`'s `reupload_toolpaths`) re-runs this filter-and-rebuild step
/// only when the slider value actually changes, then re-uploads the
/// resulting buffer. That is simpler than threading a scrub uniform
/// through `render.rs`'s bind-group setup and a shader-side `discard`, at
/// the cost of a full CPU rebuild + GPU re-upload per slider-drag frame
/// instead of a single per-frame uniform write — acceptable at the MVP
/// single-object scale this phase targets (see `toolpath_shader.wgsl`'s
/// doc comment, which reserves the per-vertex `order` attribute for a
/// possible future shader-side discard if drag interactivity ever becomes
/// a problem).
pub fn build_toolpath_lines(paths: &[Path], scrub_order: f64) -> Vec<ToolpathVertex> {
    let mut vertices = Vec::new();
    for path in paths {
        let count = path.points.len();
        for (index, segment) in path.segments.iter().enumerate() {
            if segment.order > scrub_order {
                continue;
            }
            let start = path.points[index];
            let end = path.points[(index + 1) % count];
            let color = palette_color(segment.kind);
            vertices.push(ToolpathVertex::new(start, color, segment.order));
            vertices.push(ToolpathVertex::new(end, color, segment.order));
        }
    }
    vertices
}

/// The `(min, max)` `order` value across all segments in `paths`, used to
/// size the scrub slider's range. `None` if `paths` contains no segments
/// at all.
pub fn order_range(paths: &[Path]) -> Option<(f64, f64)> {
    let mut range: Option<(f64, f64)> = None;
    for path in paths {
        for segment in &path.segments {
            range = Some(match range {
                None => (segment.order, segment.order),
                Some((min, max)) => (min.min(segment.order), max.max(segment.order)),
            });
        }
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use manifold_core::ids::ToolId;
    use manifold_core::toolpath::Segment;

    fn path_with_kinds(kinds: &[MoveKind], order: f64) -> Path {
        let points: Vec<DVec3> = (0..kinds.len())
            .map(|i| DVec3::new(i as f64, 0.0, 0.0))
            .collect();
        let segments = kinds
            .iter()
            .map(|&kind| Segment {
                kind,
                speed: 60.0,
                extrusion_rate: 1.0,
                support_fraction: 0.0,
                order,
                extrusion_length: 0.0,
            })
            .collect();
        Path {
            points,
            segments,
            tool: ToolId(0),
        }
    }

    #[test]
    fn vertex_count_matches_two_per_segment() {
        let path = path_with_kinds(
            &[MoveKind::WallOuter, MoveKind::WallInner, MoveKind::Infill],
            0.0,
        );
        let vertices = build_toolpath_lines(&[path], f64::INFINITY);
        assert_eq!(vertices.len(), 6);
    }

    #[test]
    fn vertex_count_sums_across_multiple_paths() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter, MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(
            &[MoveKind::Infill, MoveKind::Bridge, MoveKind::Overhang],
            0.0,
        );
        let vertices = build_toolpath_lines(&[path_a, path_b], f64::INFINITY);
        assert_eq!(vertices.len(), 2 * 2 + 3 * 2);
    }

    #[test]
    fn segment_endpoints_wrap_around_the_closing_edge() {
        let path = path_with_kinds(&[MoveKind::WallOuter, MoveKind::WallOuter], 0.0);
        let vertices = build_toolpath_lines(&[path], f64::INFINITY);
        // Second segment closes the loop: points[1] -> points[0].
        assert_eq!(vertices[2].position, [1.0, 0.0, 0.0]);
        assert_eq!(vertices[3].position, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn color_mapping_matches_palette_per_move_kind() {
        let cases = [
            (MoveKind::WallOuter, COLOR_WALL_OUTER),
            (MoveKind::WallInner, COLOR_WALL_INNER),
            (MoveKind::Infill, COLOR_INFILL),
            (MoveKind::Bridge, COLOR_BRIDGE),
            (MoveKind::Overhang, COLOR_OVERHANG),
            (MoveKind::Travel, COLOR_TRAVEL),
        ];
        for (kind, expected_color) in cases {
            let path = path_with_kinds(&[kind], 0.0);
            let vertices = build_toolpath_lines(&[path], f64::INFINITY);
            assert_eq!(vertices[0].color, expected_color);
            assert_eq!(vertices[1].color, expected_color);
        }
    }

    #[test]
    fn order_is_propagated_to_both_endpoint_vertices() {
        let path = path_with_kinds(&[MoveKind::WallOuter, MoveKind::Infill], 0.75);
        let vertices = build_toolpath_lines(&[path], f64::INFINITY);
        assert!(vertices.iter().all(|v| v.order == 0.75));
    }

    #[test]
    fn distinct_paths_can_carry_distinct_order_values() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::WallOuter], 1.0);
        let vertices = build_toolpath_lines(&[path_a, path_b], f64::INFINITY);
        assert_eq!(vertices[0].order, 0.0);
        assert_eq!(vertices[1].order, 0.0);
        assert_eq!(vertices[2].order, 1.0);
        assert_eq!(vertices[3].order, 1.0);
    }

    #[test]
    fn empty_paths_produce_no_vertices() {
        assert!(build_toolpath_lines(&[], f64::INFINITY).is_empty());
    }

    #[test]
    fn scrub_order_excludes_segments_above_cutoff() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::Infill], 1.0);
        let vertices = build_toolpath_lines(&[path_a, path_b], 0.0);
        // Only path_a's segment (order 0.0) survives the <= 0.0 cutoff.
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0].order, 0.0);
    }

    #[test]
    fn scrub_order_is_inclusive_of_the_cutoff_value() {
        let path = path_with_kinds(&[MoveKind::WallOuter], 0.5);
        let vertices = build_toolpath_lines(&[path], 0.5);
        assert_eq!(vertices.len(), 2);
    }

    #[test]
    fn order_range_spans_min_and_max_across_paths() {
        let path_a = path_with_kinds(&[MoveKind::WallOuter], 0.0);
        let path_b = path_with_kinds(&[MoveKind::Infill, MoveKind::Bridge], 2.0);
        let (min, max) = order_range(&[path_a, path_b]).unwrap();
        assert_eq!(min, 0.0);
        assert_eq!(max, 2.0);
    }

    #[test]
    fn order_range_is_none_for_empty_paths() {
        assert!(order_range(&[]).is_none());
    }
}
