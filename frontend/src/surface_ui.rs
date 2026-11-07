use std::collections::HashMap;

use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render:: render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::render::renderer::RenderDevice;
use bevy::window::PrimaryWindow;
use bevy_egui::{egui, EguiContexts};

use crate::config::{SlicerConfig, SurfaceHintMode};
use crate::field_vis::{srgba_to_linear_rgba, FieldVisualizationState};
use crate::scene::{
    clone_object_on_plate, remove_object_from_plate, snap_object_to_bed, BuildPlateState,
    CadObjectEntity, OrbitCamera, ToolMode,
};

/// Open-icon-set vector icons (based on the ISC-licensed **Lucide** 24×24 icon geometry)
/// for project setup, object manipulation, build plate management, and orientation hints.
///
/// Because every icon is constructed as `egui::Shape` vector primitives via [`build_icon_shapes`],
/// the exact same icons render crisply in both:
/// 1. 2D `egui` toolbars and panels (`render_icon_button`, `render_delineated_icon_pill_tab`), and
/// 2. 3D surface-mounted widgets (`SurfacePanel3d` / `tessellate_surface_panel_to_bevy_mesh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UiIcon {
    /// Pointer cursor (`Select / Orbit`).
    SelectCursor,
    /// 4-way cardinal translation arrows (`Move on Bed (XY)`).
    MoveObject,
    /// Clockwise rotation arc with arrowhead (`Rotate / Orient`, `+90°`).
    RotateObject,
    /// Counter-clockwise reset arrow (`Reset` rotation).
    ResetTransform,
    /// Faceted block with downward snap arrow onto build-plate line (`Place on Face (Hull)`).
    LayOnFace,
    /// Overlapping copy squares with `+` (`Clone`, `+1 Clone`, `Add Clones`).
    CloneObject,
    /// Trash can with lid and vertical ribs (`Remove`, `Delete`, `Clear Plate`).
    DeleteObject,
    /// Target reticle with 4 crosshair ticks (`Center on Bed (0, 0)`).
    CenterOnBed,
    /// 2×2 grid of arranged parts (`Auto-Arrange`).
    AutoArrange,
    /// 4 corner framing brackets (`Frame All`).
    FrameCamera,
    /// Label tag badge (`Object Labels`).
    ObjectLabels,
    /// Stacked isometric build plates (`Plate` switcher, `Move to Plate`).
    BuildPlate,
    /// Rounded square with `+` (`+ Plate`, `+ Add Plate`, `+ Add at Center`).
    AddPlate,
    /// Surface baseline with perpendicular normal arrow (`Hint: Face ⊥ Normal (Conformal)`).
    SurfaceHintOrthogonal,
    /// Surface baseline with parallel tangent arrow (`Hint: Face ∥ Normal (⊥ Layers)`).
    SurfaceHintTangent,
    /// Stacked conformal SDF offset layers with normal arrow (`Hint: Conformal Skin (SDF Layers)`).
    SurfaceHintConformalSkin,
    /// 3D point sphere with normal vector arrow (`Hint: 3D Point Normal`).
    PointHint3d,
    /// Tray with incoming downward arrow (`Import CAD / Mesh / 3MF...`).
    ImportFile,
    /// Floppy/archive box (`Save 3MF...`).
    SaveProject,
    /// Document with outgoing arrow (`Export G-code...`).
    ExportGcode,
}

impl UiIcon {
    /// All icon variants (used for exhaustive testing).
    pub const ALL: [Self; 20] = [
        Self::SelectCursor,
        Self::MoveObject,
        Self::RotateObject,
        Self::ResetTransform,
        Self::LayOnFace,
        Self::CloneObject,
        Self::DeleteObject,
        Self::CenterOnBed,
        Self::AutoArrange,
        Self::FrameCamera,
        Self::ObjectLabels,
        Self::BuildPlate,
        Self::AddPlate,
        Self::SurfaceHintOrthogonal,
        Self::SurfaceHintTangent,
        Self::SurfaceHintConformalSkin,
        Self::PointHint3d,
        Self::ImportFile,
        Self::SaveProject,
        Self::ExportGcode,
    ];

    /// Maps each viewport [`ToolMode`] to its corresponding [`UiIcon`].
    pub const fn for_tool_mode(mode: ToolMode) -> Self {
        match mode {
            ToolMode::Select => Self::SelectCursor,
            ToolMode::MoveOnBed => Self::MoveObject,
            ToolMode::Rotate => Self::RotateObject,
            ToolMode::PlaceOnFace => Self::LayOnFace,
            ToolMode::SurfaceHintOrthogonal => Self::SurfaceHintOrthogonal,
            ToolMode::SurfaceHintTangent => Self::SurfaceHintTangent,
            ToolMode::SurfaceHintConformalSkin => Self::SurfaceHintConformalSkin,
            ToolMode::PointHint => Self::PointHint3d,
        }
    }
}

/// Builds `egui::Shape` vector primitives for `icon` fitted inside `rect` using the
/// 24×24 Lucide coordinate system. Compatible with both 2D `egui::Painter` and 3D
/// `tessellate_surface_panel_to_bevy_mesh`.
pub fn build_icon_shapes(
    icon: UiIcon,
    rect: egui::Rect,
    color: egui::Color32,
) -> Vec<egui::Shape> {
    let side = rect.width().min(rect.height()).max(4.0);
    let sq = egui::Rect::from_center_size(rect.center(), egui::vec2(side, side));
    let map = |x: f32, y: f32| -> egui::Pos2 {
        egui::pos2(sq.min.x + (x / 24.0) * side, sq.min.y + (y / 24.0) * side)
    };
    let stroke_w = (side * 0.095).clamp(1.25, 2.4);
    let stroke = egui::Stroke::new(stroke_w, color);
    let thin_stroke = egui::Stroke::new((stroke_w * 0.82).max(1.05), color);

    let sample_arc = |cx: f32, cy: f32, r: f32, start_deg: f32, end_deg: f32, steps: usize| -> Vec<egui::Pos2> {
        let mut pts = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = i as f32 / (steps.max(1) as f32);
            let deg = start_deg + t * (end_deg - start_deg);
            let rad = deg.to_radians();
            pts.push(map(cx + r * rad.cos(), cy + r * rad.sin()));
        }
        pts
    };

    let mut shapes = Vec::new();

    match icon {
        UiIcon::SelectCursor => {
            let poly = vec![
                map(5.0, 3.0),
                map(5.0, 18.8),
                map(9.3, 14.8),
                map(12.4, 20.8),
                map(15.0, 19.4),
                map(11.9, 13.4),
                map(17.8, 13.4),
            ];
            shapes.push(egui::Shape::closed_line(poly, stroke));
        }
        UiIcon::MoveObject => {
            shapes.push(egui::Shape::line_segment([map(3.0, 12.0), map(21.0, 12.0)], stroke));
            shapes.push(egui::Shape::line_segment([map(12.0, 3.0), map(12.0, 21.0)], stroke));
            // 4 cardinal arrowheads
            shapes.push(egui::Shape::line(
                vec![map(9.0, 6.2), map(12.0, 3.0), map(15.0, 6.2)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(9.0, 17.8), map(12.0, 21.0), map(15.0, 17.8)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(6.2, 9.0), map(3.0, 12.0), map(6.2, 15.0)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(17.8, 9.0), map(21.0, 12.0), map(17.8, 15.0)],
                stroke,
            ));
        }
        UiIcon::RotateObject => {
            let arc = sample_arc(12.0, 12.0, 8.0, 42.0, 330.0, 18);
            shapes.push(egui::Shape::line(arc, stroke));
            shapes.push(egui::Shape::line(
                vec![map(19.2, 3.4), map(19.2, 8.2), map(14.4, 8.2)],
                stroke,
            ));
            shapes.push(egui::Shape::circle_filled(
                map(12.0, 12.0),
                (stroke_w * 0.72).max(1.0),
                color,
            ));
        }
        UiIcon::ResetTransform => {
            let arc = sample_arc(12.0, 12.0, 8.0, -150.0, 138.0, 18);
            shapes.push(egui::Shape::line(arc, stroke));
            shapes.push(egui::Shape::line(
                vec![map(4.8, 3.4), map(4.8, 8.2), map(9.6, 8.2)],
                stroke,
            ));
        }
        UiIcon::LayOnFace => {
            // Build plate base line
            shapes.push(egui::Shape::line_segment([map(3.0, 20.5), map(21.0, 20.5)], stroke));
            // Tilted part facet above the plate
            shapes.push(egui::Shape::closed_line(
                vec![
                    map(5.5, 7.2),
                    map(13.5, 3.2),
                    map(18.5, 8.5),
                    map(10.5, 12.5),
                ],
                thin_stroke,
            ));
            // Downward snap arrow onto the plate
            shapes.push(egui::Shape::line_segment([map(12.0, 12.2), map(12.0, 18.2)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(8.8, 15.2), map(12.0, 18.4), map(15.2, 15.2)],
                stroke,
            ));
        }
        UiIcon::CloneObject => {
            // Back square (top-left)
            shapes.push(egui::Shape::line(
                vec![
                    map(15.2, 8.5),
                    map(15.2, 3.8),
                    map(3.8, 3.8),
                    map(3.8, 15.2),
                    map(8.5, 15.2),
                ],
                thin_stroke,
            ));
            // Front square (bottom-right)
            let front_rect = egui::Rect::from_min_max(map(8.5, 8.5), map(20.2, 20.2));
            shapes.push(egui::Shape::rect_stroke(front_rect, 1.6, stroke));
            // Plus inside front square
            shapes.push(egui::Shape::line_segment(
                [map(11.8, 14.35), map(16.9, 14.35)],
                thin_stroke,
            ));
            shapes.push(egui::Shape::line_segment(
                [map(14.35, 11.8), map(14.35, 16.9)],
                thin_stroke,
            ));
        }
        UiIcon::DeleteObject => {
            // Lid & handle
            shapes.push(egui::Shape::line_segment([map(3.5, 6.2), map(20.5, 6.2)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(8.8, 6.2), map(8.8, 3.6), map(15.2, 3.6), map(15.2, 6.2)],
                thin_stroke,
            ));
            // Trash can body
            shapes.push(egui::Shape::line(
                vec![
                    map(5.6, 6.2),
                    map(6.6, 20.2),
                    map(17.4, 20.2),
                    map(18.4, 6.2),
                ],
                stroke,
            ));
            // Inner vertical ribs
            shapes.push(egui::Shape::line_segment([map(10.0, 10.2), map(10.0, 16.8)], thin_stroke));
            shapes.push(egui::Shape::line_segment([map(14.0, 10.2), map(14.0, 16.8)], thin_stroke));
        }
        UiIcon::CenterOnBed => {
            shapes.push(egui::Shape::circle_stroke(
                map(12.0, 12.0),
                side * (7.5 / 24.0),
                stroke,
            ));
            shapes.push(egui::Shape::line_segment([map(12.0, 2.0), map(12.0, 6.8)], stroke));
            shapes.push(egui::Shape::line_segment([map(12.0, 17.2), map(12.0, 22.0)], stroke));
            shapes.push(egui::Shape::line_segment([map(2.0, 12.0), map(6.8, 12.0)], stroke));
            shapes.push(egui::Shape::line_segment([map(17.2, 12.0), map(22.0, 12.0)], stroke));
            shapes.push(egui::Shape::circle_filled(
                map(12.0, 12.0),
                (stroke_w * 0.75).max(1.1),
                color,
            ));
        }
        UiIcon::AutoArrange => {
            for (x0, y0, x1, y1) in [
                (3.5, 3.5, 10.2, 10.2),
                (13.8, 3.5, 20.5, 10.2),
                (3.5, 13.8, 10.2, 20.5),
                (13.8, 13.8, 20.5, 20.5),
            ] {
                shapes.push(egui::Shape::rect_stroke(
                    egui::Rect::from_min_max(map(x0, y0), map(x1, y1)),
                    1.2,
                    stroke,
                ));
            }
        }
        UiIcon::FrameCamera => {
            shapes.push(egui::Shape::line(
                vec![map(3.5, 8.8), map(3.5, 3.5), map(8.8, 3.5)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(15.2, 3.5), map(20.5, 3.5), map(20.5, 8.8)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(20.5, 15.2), map(20.5, 20.5), map(15.2, 20.5)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(8.8, 20.5), map(3.5, 20.5), map(3.5, 15.2)],
                stroke,
            ));
            shapes.push(egui::Shape::circle_stroke(
                map(12.0, 12.0),
                side * (2.6 / 24.0),
                thin_stroke,
            ));
        }
        UiIcon::ObjectLabels => {
            shapes.push(egui::Shape::closed_line(
                vec![
                    map(3.5, 12.0),
                    map(3.5, 4.0),
                    map(11.5, 4.0),
                    map(20.2, 12.7),
                    map(12.7, 20.2),
                ],
                stroke,
            ));
            shapes.push(egui::Shape::circle_filled(
                map(7.6, 7.6),
                (stroke_w * 0.85).max(1.2),
                color,
            ));
        }
        UiIcon::BuildPlate => {
            shapes.push(egui::Shape::closed_line(
                vec![
                    map(12.0, 3.4),
                    map(21.0, 8.0),
                    map(12.0, 12.6),
                    map(3.0, 8.0),
                ],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(3.0, 12.5), map(12.0, 17.0), map(21.0, 12.5)],
                thin_stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(3.0, 16.6), map(12.0, 21.0), map(21.0, 16.6)],
                thin_stroke,
            ));
        }
        UiIcon::AddPlate => {
            shapes.push(egui::Shape::rect_stroke(
                egui::Rect::from_min_max(map(3.6, 3.6), map(20.4, 20.4)),
                2.0,
                thin_stroke,
            ));
            shapes.push(egui::Shape::line_segment([map(7.5, 12.0), map(16.5, 12.0)], stroke));
            shapes.push(egui::Shape::line_segment([map(12.0, 7.5), map(12.0, 16.5)], stroke));
        }
        UiIcon::SurfaceHintOrthogonal => {
            // Surface baseline
            shapes.push(egui::Shape::line(
                vec![map(3.0, 19.2), map(8.5, 17.5), map(15.5, 17.5), map(21.0, 19.2)],
                stroke,
            ));
            // Right-angle marker
            shapes.push(egui::Shape::line(
                vec![map(12.0, 14.0), map(15.4, 14.0), map(15.4, 17.5)],
                thin_stroke,
            ));
            // Perpendicular normal arrow
            shapes.push(egui::Shape::line_segment([map(12.0, 17.5), map(12.0, 3.8)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(8.2, 7.8), map(12.0, 3.8), map(15.8, 7.8)],
                stroke,
            ));
        }
        UiIcon::SurfaceHintTangent => {
            // Surface baseline
            shapes.push(egui::Shape::line_segment([map(3.0, 18.8), map(21.0, 18.8)], stroke));
            // Parallel guide line
            shapes.push(egui::Shape::line_segment([map(3.5, 14.6), map(15.5, 14.6)], thin_stroke));
            // Parallel tangent arrow
            shapes.push(egui::Shape::line_segment([map(3.5, 9.8), map(20.2, 9.8)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(15.8, 6.0), map(20.2, 9.8), map(15.8, 13.6)],
                stroke,
            ));
        }
        UiIcon::SurfaceHintConformalSkin => {
            // Three stacked conformal SDF offset curves + upward normal arrow
            shapes.push(egui::Shape::line(
                vec![map(3.0, 19.6), map(8.5, 18.0), map(15.5, 18.0), map(21.0, 19.6)],
                stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(3.5, 15.6), map(8.5, 14.0), map(15.5, 14.0), map(20.5, 15.6)],
                thin_stroke,
            ));
            shapes.push(egui::Shape::line(
                vec![map(4.0, 11.6), map(8.5, 10.0), map(15.5, 10.0), map(20.0, 11.6)],
                thin_stroke,
            ));
            shapes.push(egui::Shape::line_segment([map(12.0, 10.0), map(12.0, 3.2)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(8.8, 6.4), map(12.0, 3.2), map(15.2, 6.4)],
                stroke,
            ));
        }
        UiIcon::PointHint3d => {
            // 3D point sphere + aura arc
            shapes.push(egui::Shape::circle_filled(
                map(7.2, 16.8),
                side * (2.6 / 24.0),
                color,
            ));
            let aura = sample_arc(7.2, 16.8, 5.2, 110.0, 340.0, 12);
            shapes.push(egui::Shape::line(aura, thin_stroke));
            // Normal vector arrow aiming up-right
            shapes.push(egui::Shape::line_segment([map(7.2, 16.8), map(19.6, 4.4)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(13.8, 4.4), map(19.6, 4.4), map(19.6, 10.2)],
                stroke,
            ));
        }
        UiIcon::ImportFile => {
            // Tray at bottom
            shapes.push(egui::Shape::line(
                vec![
                    map(3.5, 14.2),
                    map(3.5, 19.8),
                    map(20.5, 19.8),
                    map(20.5, 14.2),
                ],
                stroke,
            ));
            // Incoming downward arrow
            shapes.push(egui::Shape::line_segment([map(12.0, 3.5), map(12.0, 15.0)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(8.0, 11.0), map(12.0, 15.2), map(16.0, 11.0)],
                stroke,
            ));
        }
        UiIcon::SaveProject => {
            // Floppy outline with chamfered top-right corner
            shapes.push(egui::Shape::closed_line(
                vec![
                    map(4.0, 4.0),
                    map(16.5, 4.0),
                    map(20.0, 7.5),
                    map(20.0, 20.0),
                    map(4.0, 20.0),
                ],
                stroke,
            ));
            // Top label slot
            shapes.push(egui::Shape::line(
                vec![map(8.0, 4.0), map(8.0, 9.2), map(15.2, 9.2), map(15.2, 4.0)],
                thin_stroke,
            ));
            // Bottom shutter
            shapes.push(egui::Shape::line(
                vec![
                    map(7.2, 20.0),
                    map(7.2, 13.5),
                    map(16.8, 13.5),
                    map(16.8, 20.0),
                ],
                thin_stroke,
            ));
        }
        UiIcon::ExportGcode => {
            // Document outline on left
            shapes.push(egui::Shape::line(
                vec![
                    map(14.2, 3.8),
                    map(4.8, 3.8),
                    map(4.8, 20.2),
                    map(14.2, 20.2),
                ],
                stroke,
            ));
            // Outgoing arrow to right
            shapes.push(egui::Shape::line_segment([map(9.8, 12.0), map(20.8, 12.0)], stroke));
            shapes.push(egui::Shape::line(
                vec![map(16.8, 8.0), map(20.8, 12.0), map(16.8, 16.0)],
                stroke,
            ));
        }
    }

    shapes
}

/// Paints `icon` inside `rect` on a 2D `egui::Painter`.
pub fn draw_icon(painter: &egui::Painter, icon: UiIcon, rect: egui::Rect, color: egui::Color32) {
    painter.extend(build_icon_shapes(icon, rect, color));
}

/// Renders a 2D `egui` button containing a vector [`UiIcon`] and optional text label.
pub fn render_icon_button(
    ui: &mut egui::Ui,
    icon: UiIcon,
    label: impl Into<String>,
    small: bool,
    enabled: bool,
) -> egui::Response {
    let label_str = label.into();
    let font_id = if small {
        egui::TextStyle::Small.resolve(ui.style())
    } else {
        egui::TextStyle::Button.resolve(ui.style())
    };
    let icon_side = if small { 12.5_f32 } else { 14.5_f32 };
    let pad = if small {
        egui::vec2(5.5, 2.5)
    } else {
        ui.spacing().button_padding
    };
    let gap = if label_str.is_empty() {
        0.0
    } else if small {
        4.2
    } else {
        5.5
    };

    let galley_opt = if label_str.is_empty() {
        None
    } else {
        Some(ui.painter().layout_no_wrap(
            label_str.clone(),
            font_id.clone(),
            egui::Color32::WHITE,
        ))
    };
    let text_size = galley_opt.as_ref().map_or(egui::Vec2::ZERO, |g| g.size());
    let inner_w = icon_side + gap + text_size.x;
    let inner_h = icon_side.max(text_size.y);
    let total_size = egui::vec2(inner_w + 2.0 * pad.x, inner_h + 2.0 * pad.y);

    let sense = if enabled {
        egui::Sense::click()
    } else {
        egui::Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(total_size, sense);
    if enabled && response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let visuals = if enabled {
            ui.style().interact(&response)
        } else {
            &ui.style().visuals.widgets.inactive
        };
        let bg_fill = if enabled {
            visuals.bg_fill
        } else {
            visuals.bg_fill.gamma_multiply(0.55)
        };
        let fg_color = if enabled {
            visuals.fg_stroke.color
        } else {
            visuals.fg_stroke.color.gamma_multiply(0.5)
        };
        let expanded = rect.expand(visuals.expansion);
        ui.painter()
            .rect(expanded, visuals.rounding, bg_fill, visuals.bg_stroke);

        let left_x = rect.center().x - 0.5 * inner_w;
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(left_x + 0.5 * icon_side, rect.center().y),
            egui::vec2(icon_side, icon_side),
        );
        draw_icon(ui.painter(), icon, icon_rect, fg_color);

        if !label_str.is_empty() {
            let galley = ui.painter().layout_no_wrap(label_str, font_id, fg_color);
            let text_pos = egui::pos2(
                left_x + icon_side + gap,
                rect.center().y - 0.5 * galley.size().y,
            );
            ui.painter().galley(text_pos, galley, fg_color);
        }
    }

    response
}

/// Renders a clearly delineated bordered pill/tab button with an optional vector [`UiIcon`]
/// for inline mode & selector strips (e.g., `ToolMode`, `Build Plate` switchers, and `Orientation Hint` tools).
pub fn render_delineated_icon_pill_tab(
    ui: &mut egui::Ui,
    is_active: bool,
    icon: Option<UiIcon>,
    label: impl Into<String>,
) -> egui::Response {
    let label_str = label.into();
    let font_id = if is_active {
        egui::FontId::proportional(12.8)
    } else {
        egui::FontId::proportional(12.4)
    };
    let fg_color = if is_active {
        egui::Color32::WHITE
    } else {
        egui::Color32::from_rgb(195, 203, 218)
    };
    let icon_side = 13.5_f32;
    let gap = if icon.is_some() && !label_str.is_empty() {
        5.0
    } else {
        0.0
    };
    let pad = egui::vec2(7.5, 4.0);

    let galley = ui
        .painter()
        .layout_no_wrap(label_str, font_id, fg_color);
    let icon_w = if icon.is_some() { icon_side } else { 0.0 };
    let inner_w = icon_w + gap + galley.size().x;
    let inner_h = icon_side.max(galley.size().y);
    let total_size = egui::vec2(inner_w + 2.0 * pad.x, inner_h + 2.0 * pad.y);

    let (rect, response) = ui.allocate_exact_size(total_size, egui::Sense::click());
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    if ui.is_rect_visible(rect) {
        let fill = if is_active {
            egui::Color32::from_rgb(44, 96, 166)
        } else if response.hovered() {
            egui::Color32::from_rgb(38, 44, 56)
        } else {
            egui::Color32::from_rgb(28, 32, 40)
        };
        let stroke = if is_active {
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(104, 178, 255))
        } else if response.hovered() {
            egui::Stroke::new(1.2_f32, egui::Color32::from_rgb(110, 125, 150))
        } else {
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(72, 80, 96))
        };
        ui.painter()
            .rect(rect, egui::Rounding::same(5.0), fill, stroke);

        let left_x = rect.center().x - 0.5 * inner_w;
        if let Some(ic) = icon {
            let icon_rect = egui::Rect::from_center_size(
                egui::pos2(left_x + 0.5 * icon_side, rect.center().y),
                egui::vec2(icon_side, icon_side),
            );
            draw_icon(ui.painter(), ic, icon_rect, fg_color);
        }
        let text_pos = egui::pos2(
            left_x + icon_w + gap,
            rect.center().y - 0.5 * galley.size().y,
        );
        ui.painter().galley(text_pos, galley, fg_color);
    }

    response
}


/// An orthonormal right-handed 3D coordinate frame anchored to a surface in 3D space
/// (such as the build plate at $z = 0$, a STEP B-Rep face, or a planar/curved mesh patch).
///
/// Local surface coordinates `(u_mm, v_mm)` are measured in millimeters along `u_axis` (horizontal
/// reading direction) and `v_axis` (upward surface direction), elevated by `elevation_mm` along
/// the outward surface `normal` ($\hat{\mathbf{n}} = \hat{\mathbf{u}} \times \hat{\mathbf{v}}$).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceFrame3d {
    pub origin: Vec3,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
    pub normal: Vec3,
    pub elevation_mm: f32,
}

impl Default for SurfaceFrame3d {
    fn default() -> Self {
        Self::build_plate_at(Vec2::ZERO)
    }
}

impl SurfaceFrame3d {
    /// Creates a 3D surface frame lying flat on the build plate ($z = 0$) at `origin_xy`,
    /// reading left-to-right along $+X$ (`u_axis = Vec3::X`) with $+Y$ up (`v_axis = Vec3::Y`)
    /// and outward normal $+Z$ (`normal = Vec3::Z`).
    pub const fn build_plate_at(origin_xy: Vec2) -> Self {
        Self {
            origin: Vec3::new(origin_xy.x, origin_xy.y, 0.0),
            u_axis: Vec3::X,
            v_axis: Vec3::Y,
            normal: Vec3::Z,
            elevation_mm: 0.06,
        }
    }

    /// Constructs a right-handed orthonormal 3D surface frame at `origin` with outward `normal`,
    /// projecting `preferred_u` into the tangent plane to form `u_axis` and setting
    /// `v_axis = normal.cross(u_axis)`.
    pub fn from_point_and_normal(origin: Vec3, normal: Vec3, preferred_u: Vec3) -> Self {
        let n = normal.try_normalize().unwrap_or(Vec3::Z);
        let proj_u = preferred_u - n * preferred_u.dot(n);
        let u = if let Some(unit_u) = proj_u.try_normalize() {
            unit_u
        } else {
            let fallback_ref = if n.z.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let fallback_u = fallback_ref - n * fallback_ref.dot(n);
            fallback_u.try_normalize().unwrap_or(Vec3::X)
        };
        let v = n.cross(u).try_normalize().unwrap_or(Vec3::Y);
        Self {
            origin,
            u_axis: u,
            v_axis: v,
            normal: n,
            elevation_mm: 0.08,
        }
    }

    /// Transforms an object-local `SurfaceFrame3d` into world/bed coordinates using `transform`.
    pub fn transformed(&self, transform: &Transform) -> Self {
        let world_origin = transform.transform_point(self.origin);
        let world_u = (transform.rotation * self.u_axis)
            .try_normalize()
            .unwrap_or(Vec3::X);
        let world_n = (transform.rotation * self.normal)
            .try_normalize()
            .unwrap_or(Vec3::Z);
        let world_v = world_n.cross(world_u).try_normalize().unwrap_or(Vec3::Y);
        Self {
            origin: world_origin,
            u_axis: world_u,
            v_axis: world_v,
            normal: world_n,
            elevation_mm: self.elevation_mm,
        }
    }

    /// Maps local 2D surface coordinates `(u_mm, v_mm)` (in millimeters) and an additional
    /// normal offset `extra_normal_offset_mm` to a 3D world-space point.
    #[inline]
    pub fn local_to_world(&self, u_mm: f32, v_mm: f32, extra_normal_offset_mm: f32) -> Vec3 {
        self.origin
            + self.u_axis * u_mm
            + self.v_axis * v_mm
            + self.normal * (self.elevation_mm + extra_normal_offset_mm)
    }

    /// Intersects a 3D ray against this surface frame's elevated tangent plane, returning
    /// `Some((ray_distance_t, Vec2::new(u_mm, v_mm)))` if the ray hits the plane in front of the camera.
    pub fn ray_intersect_local_uv(&self, ray: Ray3d) -> Option<(f32, Vec2)> {
        let dir = *ray.direction;
        let denom = dir.dot(self.normal);
        if denom.abs() < 1.0e-6 {
            return None;
        }
        let plane_origin = self.origin + self.normal * self.elevation_mm;
        let t = (plane_origin - ray.origin).dot(self.normal) / denom;
        if t <= 0.0 {
            return None;
        }
        let hit_pt = ray.origin + dir * t;
        let rel = hit_pt - plane_origin;
        let u_mm = rel.dot(self.u_axis);
        let v_mm = rel.dot(self.v_axis);
        Some((t, Vec2::new(u_mm, v_mm)))
    }
}

/// Specifies how a 3D surface widget panel is anchored in the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceAnchor3d {
    /// Mounted flat on the build plate ($z = 0$) at `(x_mm, y_mm)`, reading along $+X$.
    BuildPlate(Vec2),
    /// Mounted on an explicit 3D surface frame in world/bed coordinates.
    WorldSurface(SurfaceFrame3d),
    /// Mounted on an object-local 3D surface frame (e.g. a B-Rep face or mesh patch),
    /// automatically moving and rotating with `object_id`'s `Transform`.
    ObjectSurface {
        object_id: u64,
        local_frame: SurfaceFrame3d,
    },
}

impl SurfaceAnchor3d {
    /// Resolves this anchor into a world-space [`SurfaceFrame3d`].
    pub fn resolve_world_frame(
        &self,
        plate: &BuildPlateState,
        object_transforms: &Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    ) -> Option<SurfaceFrame3d> {
        match *self {
            Self::BuildPlate(xy) => Some(SurfaceFrame3d::build_plate_at(xy)),
            Self::WorldSurface(frame) => Some(frame),
            Self::ObjectSurface {
                object_id,
                local_frame,
            } => {
                let obj = plate
                    .objects
                    .iter()
                    .find(|o| o.object_id == object_id && o.visible)?;
                let (tf, _) = object_transforms.get(obj.entity).ok()?;
                Some(local_frame.transformed(tf))
            }
        }
    }
}

/// Actions emitted when a user clicks or drags a 3D surface widget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SurfaceWidgetAction {
    /// Selects `object_id` on the build plate (triggered by clicking an object's 3D build-plate label).
    SelectObject(u64),
    /// Selects `object_id` and activates or toggles a viewport [`ToolMode`] (`MoveOnBed`, `Rotate`, `PlaceOnFace`, etc.).
    ActivateToolForObject { object_id: u64, mode: ToolMode },
    /// Centers `object_id` at `(0, 0)` on the build plate.
    CenterObjectOnBed(u64),
    /// Rotates `object_id` by $+90^\circ$ around $+Z$ on the build plate.
    RotateObjectZ90(u64),
    /// Clones `object_id` on the active build plate.
    CloneObject(u64),
    /// Deletes `object_id` from the active build plate.
    DeleteObject(u64),
    /// Toggles a surface orientation hint on `(object_id, patch_id)` between Orthogonal ($\perp$) and Tangent ($\parallel$).
    CycleSurfaceHintMode { object_id: u64, patch_id: u32 },
    /// Updates the blending weight (`0.0..=1.0`) of a surface orientation hint on `(object_id, patch_id)`.
    SetSurfaceHintWeight { object_id: u64, patch_id: u32 },
    /// Removes the surface orientation hint on `(object_id, patch_id)`.
    RemoveSurfaceHint { object_id: u64, patch_id: u32 },
    /// General-purpose custom action ID for future 3D surface widgets.
    Custom(u64),
}

/// Interactive or display widget primitives that can be placed inside a [`SurfacePanel3d`].
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceWidgetKind {
    /// Static text label or badge on the 3D surface.
    Label {
        text: String,
        color: egui::Color32,
        monospace: bool,
    },
    /// Clickable button or selectable pill (with optional vector [`UiIcon`]) on the 3D surface.
    Button {
        icon: Option<UiIcon>,
        text: String,
        selected: bool,
        text_color: egui::Color32,
        fill_color: egui::Color32,
        border_color: egui::Color32,
        action: SurfaceWidgetAction,
    },
    /// Clickable boolean toggle / checkbox on the 3D surface.
    Toggle {
        label: String,
        value: bool,
        action: SurfaceWidgetAction,
    },
    /// Draggable scalar slider (`min..=max`) on the 3D surface.
    Slider {
        label: String,
        value: f32,
        min: f32,
        max: f32,
        suffix: String,
        action: SurfaceWidgetAction,
    },
}

/// A single widget instance laid out inside a [`SurfacePanel3d`].
#[derive(Clone, Debug)]
pub struct SurfaceWidget3d {
    pub id: u64,
    /// Bounding rectangle in local panel point coordinates (where `(0, 0)` is the panel center,
    /// $+X$ is right along `u_axis`, and $+Y$ is down along `-v_axis`).
    pub rect_pt: egui::Rect,
    pub kind: SurfaceWidgetKind,
}

/// A 3D surface-mounted UI panel containing one or more [`SurfaceWidget3d`] items,
/// rendered as a depth-tested 3D Bevy mesh on its host [`SurfaceFrame3d`].
#[derive(Clone, Debug)]
pub struct SurfacePanel3d {
    pub id: u64,
    pub anchor: SurfaceAnchor3d,
    /// Physical scale factor converting 2D layout points (`pt`) into 3D millimeters (`mm`) on the surface.
    pub mm_per_point: f32,
    pub bg_color: egui::Color32,
    pub border_color: egui::Color32,
    pub rounding_pt: f32,
    pub padding_pt: egui::Vec2,
    /// Total panel bounding box in local panel points, centered at `(0, 0)`.
    pub panel_rect_pt: egui::Rect,
    pub widgets: Vec<SurfaceWidget3d>,
}

impl SurfacePanel3d {
    /// Converts a local 3D surface hit coordinate `(u_mm, v_mm)` into 2D panel point coordinates `(x_pt, y_pt)`.
    #[inline]
    pub fn uv_mm_to_panel_pt(&self, uv_mm: Vec2) -> egui::Pos2 {
        let scale = self.mm_per_point.max(1.0e-4);
        egui::pos2(uv_mm.x / scale, -uv_mm.y / scale)
    }

    /// Converts a 2D panel point coordinate `(x_pt, y_pt)` into local 3D surface coordinates `(u_mm, v_mm)`.
    #[inline]
    pub fn panel_pt_to_uv_mm(&self, pt: egui::Pos2) -> Vec2 {
        Vec2::new(pt.x * self.mm_per_point, -pt.y * self.mm_per_point)
    }

    /// Returns the total 3D width and height of this panel in millimeters `(width_u_mm, height_v_mm)`.
    #[inline]
    pub fn size_mm(&self) -> Vec2 {
        Vec2::new(
            self.panel_rect_pt.width() * self.mm_per_point,
            self.panel_rect_pt.height() * self.mm_per_point,
        )
    }

    /// Hit-tests a local 3D surface point `uv_mm` against the interactive widgets in this panel.
    pub fn hit_test_widget(&self, uv_mm: Vec2) -> Option<&SurfaceWidget3d> {
        let pt = self.uv_mm_to_panel_pt(uv_mm);
        if !self.panel_rect_pt.expand(2.0).contains(pt) {
            return None;
        }
        self.widgets.iter().rev().find(|w| {
            matches!(
                w.kind,
                SurfaceWidgetKind::Button { .. }
                    | SurfaceWidgetKind::Toggle { .. }
                    | SurfaceWidgetKind::Slider { .. }
            ) && w.rect_pt.expand(1.5).contains(pt)
        })
    }
}

/// Marker component for a 3D Bevy mesh entity rendering a [`SurfacePanel3d`].
#[derive(Component)]
pub struct SurfacePanelMeshEntity {
    pub panel_id: u64,
}

/// Cached GPU handles for a spawned [`SurfacePanel3d`] entity.
struct CachedSurfacePanelGpu {
    entity: Entity,
    mesh_handle: Handle<Mesh>,
}

/// Resource managing 3D surface-mounted labels and interactive surface widgets,
/// their shared `egui` font-atlas texture/material, and 3D raycast hover/drag state.
#[derive(Resource, Default)]
pub struct SurfaceUiState {
    /// Active 3D surface panels (build-plate object labels + surface-mounted widgets) for the current frame.
    pub panels: Vec<SurfacePanel3d>,
    /// Currently hovered `(panel_id, widget_id)` from 3D raycasting.
    pub hovered_widget: Option<(u64, u64)>,
    /// Currently dragged slider `(panel_id, widget_id)` during a left-mouse drag on a 3D surface.
    pub active_slider_drag: Option<(u64, u64)>,
    /// Optional custom panels registered by external features/tools.
    pub custom_panels: Vec<SurfacePanel3d>,

    font_atlas_image: Option<Handle<Image>>,
    font_atlas_material: Option<Handle<StandardMaterial>>,
    font_atlas_signature: (usize, usize, u64),
    panel_gpu_cache: HashMap<u64, CachedSurfacePanelGpu>,
}

impl SurfaceUiState {
    /// Returns true if the cursor is currently hovering or dragging an interactive 3D surface widget.
    pub fn is_interacting(&self) -> bool {
        self.hovered_widget.is_some() || self.active_slider_drag.is_some()
    }
}

/// Builds a 3D build-plate label panel for `obj`, positioned flat on the build plate
/// ($z = 0$) just in front of ($-Y$) the object's `exclude_object` footprint outline and reading along $+X$.
/// When `is_selected` is true, also includes a compact 3D surface quick-action icon bar
/// (`Move`, `Rotate`, `Lay on Face`, `Center`, `+90°Z`, `Clone`, `Delete`) directly below the object label.
fn build_object_plate_label_panel(
    ctx: &egui::Context,
    object_id: u64,
    label_text: String,
    poly_xy: &[[f32; 2]],
    center_xy: [f32; 2],
    is_selected: bool,
    active_tool_mode: ToolMode,
) -> SurfacePanel3d {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for &[x, y] in poly_xy {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if !min_x.is_finite() {
        min_x = center_xy[0] - 10.0;
        max_x = center_xy[0] + 10.0;
        min_y = center_xy[1] - 10.0;
        max_y = center_xy[1] + 10.0;
    }

    let span_x = (max_x - min_x).max(12.0);
    let span_y = (max_y - min_y).max(12.0);
    let footprint_span = span_x.max(span_y);

    // Lay out text at 22pt in egui so glyphs in the font atlas are high-resolution,
    // and map points to millimeters on the build plate (clamped so labels remain legible
    // on both small and large parts).
    let font_pt = 20.0_f32;
    let mm_per_point = (footprint_span * 0.0038).clamp(0.16, 0.28);
    let font_id = egui::FontId::monospace(font_pt);

    let text_color = if is_selected {
        egui::Color32::from_rgb(255, 228, 120)
    } else {
        egui::Color32::from_rgb(185, 238, 255)
    };
    let fill_color = if is_selected {
        egui::Color32::from_rgba_unmultiplied(28, 24, 14, 235)
    } else {
        egui::Color32::from_rgba_unmultiplied(14, 20, 28, 225)
    };
    let border_color = if is_selected {
        egui::Color32::from_rgba_unmultiplied(252, 202, 58, 245)
    } else {
        egui::Color32::from_rgba_unmultiplied(82, 209, 245, 210)
    };

    let text_size_pt = ctx.fonts(|f| {
        f.layout_no_wrap(label_text.clone(), font_id, text_color)
            .size()
    });
    let pad_pt = egui::vec2(12.0, 5.5);
    let icon_extra_pt = 24.0_f32;
    let label_btn_w = text_size_pt.x + icon_extra_pt + 2.0 * pad_pt.x;
    let label_btn_h = (text_size_pt.y + 2.0 * pad_pt.y).max(30.0);

    let mut widgets = Vec::new();
    let total_size_pt = if is_selected {
        let icon_btn_w = 30.0_f32;
        let icon_btn_h = 28.0_f32;
        let btn_gap = 4.5_f32;
        let action_count = 7usize;
        let action_row_w =
            (action_count as f32) * icon_btn_w + ((action_count - 1) as f32) * btn_gap;
        let row_gap = 4.5_f32;
        let total_w = label_btn_w.max(action_row_w);
        let total_h = label_btn_h + row_gap + icon_btn_h;
        let panel_rect = egui::Rect::from_center_size(egui::Pos2::ZERO, egui::vec2(total_w, total_h));

        let label_rect = egui::Rect::from_center_size(
            egui::pos2(0.0, panel_rect.top() + 0.5 * label_btn_h),
            egui::vec2(label_btn_w, label_btn_h),
        );
        widgets.push(SurfaceWidget3d {
            id: 1,
            rect_pt: label_rect,
            kind: SurfaceWidgetKind::Button {
                icon: Some(UiIcon::ObjectLabels),
                text: label_text,
                selected: true,
                text_color,
                fill_color,
                border_color,
                action: SurfaceWidgetAction::SelectObject(object_id),
            },
        });

        let actions: [(UiIcon, bool, SurfaceWidgetAction, bool); 7] = [
            (
                UiIcon::MoveObject,
                active_tool_mode == ToolMode::MoveOnBed,
                SurfaceWidgetAction::ActivateToolForObject {
                    object_id,
                    mode: ToolMode::MoveOnBed,
                },
                false,
            ),
            (
                UiIcon::RotateObject,
                active_tool_mode == ToolMode::Rotate,
                SurfaceWidgetAction::ActivateToolForObject {
                    object_id,
                    mode: ToolMode::Rotate,
                },
                false,
            ),
            (
                UiIcon::LayOnFace,
                active_tool_mode == ToolMode::PlaceOnFace,
                SurfaceWidgetAction::ActivateToolForObject {
                    object_id,
                    mode: ToolMode::PlaceOnFace,
                },
                false,
            ),
            (
                UiIcon::CenterOnBed,
                false,
                SurfaceWidgetAction::CenterObjectOnBed(object_id),
                false,
            ),
            (
                UiIcon::RotateObject,
                false,
                SurfaceWidgetAction::RotateObjectZ90(object_id),
                false,
            ),
            (
                UiIcon::CloneObject,
                false,
                SurfaceWidgetAction::CloneObject(object_id),
                false,
            ),
            (
                UiIcon::DeleteObject,
                false,
                SurfaceWidgetAction::DeleteObject(object_id),
                true,
            ),
        ];

        let row_left_x = -0.5 * action_row_w;
        let row_top_y = panel_rect.top() + label_btn_h + row_gap;
        for (i, (ic, tool_active, act, is_danger)) in actions.into_iter().enumerate() {
            let bx = row_left_x + (i as f32) * (icon_btn_w + btn_gap);
            let btn_rect =
                egui::Rect::from_min_size(egui::pos2(bx, row_top_y), egui::vec2(icon_btn_w, icon_btn_h));
            let (fg, bg, bd) = if is_danger {
                (
                    egui::Color32::from_rgb(255, 165, 165),
                    egui::Color32::from_rgba_unmultiplied(54, 20, 24, 232),
                    egui::Color32::from_rgb(210, 82, 82),
                )
            } else if tool_active {
                (
                    egui::Color32::WHITE,
                    egui::Color32::from_rgba_unmultiplied(36, 92, 162, 242),
                    egui::Color32::from_rgb(108, 192, 255),
                )
            } else {
                (
                    egui::Color32::from_rgb(212, 228, 245),
                    egui::Color32::from_rgba_unmultiplied(20, 26, 36, 232),
                    egui::Color32::from_rgb(84, 102, 128),
                )
            };
            widgets.push(SurfaceWidget3d {
                id: 10 + (i as u64),
                rect_pt: btn_rect,
                kind: SurfaceWidgetKind::Button {
                    icon: Some(ic),
                    text: String::new(),
                    selected: tool_active,
                    text_color: fg,
                    fill_color: bg,
                    border_color: bd,
                    action: act,
                },
            });
        }

        egui::vec2(total_w, total_h)
    } else {
        let size = egui::vec2(label_btn_w, label_btn_h);
        let rect = egui::Rect::from_center_size(egui::Pos2::ZERO, size);
        widgets.push(SurfaceWidget3d {
            id: 1,
            rect_pt: rect,
            kind: SurfaceWidgetKind::Button {
                icon: Some(UiIcon::ObjectLabels),
                text: label_text,
                selected: false,
                text_color,
                fill_color,
                border_color,
                action: SurfaceWidgetAction::SelectObject(object_id),
            },
        });
        size
    };

    let panel_rect_pt = egui::Rect::from_center_size(egui::Pos2::ZERO, total_size_pt);
    let label_height_mm = total_size_pt.y * mm_per_point;
    let front_margin_mm = 2.4_f32;
    let anchor_xy = Vec2::new(
        0.5 * (min_x + max_x),
        min_y - front_margin_mm - 0.5 * label_height_mm,
    );

    let panel_id = 0x1000_0000_0000_0000_u64 ^ object_id;

    SurfacePanel3d {
        id: panel_id,
        anchor: SurfaceAnchor3d::BuildPlate(anchor_xy),
        mm_per_point,
        bg_color: egui::Color32::TRANSPARENT,
        border_color: egui::Color32::TRANSPARENT,
        rounding_pt: 7.0,
        padding_pt: pad_pt,
        panel_rect_pt,
        widgets,
    }
}

/// Builds an interactive 3D surface-mounted widget card anchored directly onto a selected
/// surface orientation hint patch (`SurfaceAnchor3d::ObjectSurface`), allowing the user to
/// toggle `⊥ Orthogonal` / `∥ Tangent`, drag the hint `Weight` slider, or delete the hint
/// directly on the 3D surface.
fn build_surface_hint_widget_panel(
    ctx: &egui::Context,
    object_id: u64,
    patch_id: u32,
    local_centroid: Vec3,
    local_normal: Vec3,
    patch_area_mm2: f32,
    mode: SurfaceHintMode,
    weight: f32,
) -> SurfacePanel3d {
    let font_pt = 18.0_f32;
    let font_id = egui::FontId::proportional(font_pt);
    let (mode_icon, mode_text) = match mode {
        SurfaceHintMode::NormalOrthogonalToSurface => (
            UiIcon::SurfaceHintOrthogonal,
            format!("Patch #{patch_id}: ⊥ Orth"),
        ),
        SurfaceHintMode::NormalTangentToSurface => (
            UiIcon::SurfaceHintTangent,
            format!("Patch #{patch_id}: ∥ Tang"),
        ),
        SurfaceHintMode::ConformalSurfaceSkin => (
            UiIcon::SurfaceHintConformalSkin,
            format!("Patch #{patch_id}: Skin"),
        ),
    };
    let mode_text_size = ctx.fonts(|f| {
        f.layout_no_wrap(mode_text.clone(), font_id.clone(), egui::Color32::WHITE)
            .size()
    });

    let btn_w = (mode_text_size.x + 40.0).max(155.0);
    let close_w = 32.0_f32;
    let gap = 6.0_f32;
    let row_h = 28.0_f32;
    let slider_h = 24.0_f32;
    let pad = egui::vec2(10.0, 8.0);

    let inner_w = btn_w + gap + close_w;
    let inner_h = row_h + gap + slider_h;
    let total_size = egui::vec2(inner_w + 2.0 * pad.x, inner_h + 2.0 * pad.y);
    let panel_rect_pt = egui::Rect::from_center_size(egui::Pos2::ZERO, total_size);

    let top_y = panel_rect_pt.top() + pad.y;
    let left_x = panel_rect_pt.left() + pad.x;

    let mode_btn_rect =
        egui::Rect::from_min_size(egui::pos2(left_x, top_y), egui::vec2(btn_w, row_h));
    let close_btn_rect = egui::Rect::from_min_size(
        egui::pos2(left_x + btn_w + gap, top_y),
        egui::vec2(close_w, row_h),
    );
    let slider_rect = egui::Rect::from_min_size(
        egui::pos2(left_x, top_y + row_h + gap),
        egui::vec2(inner_w, slider_h),
    );

    let (mode_fill, mode_border) = match mode {
        SurfaceHintMode::NormalOrthogonalToSurface => (
            egui::Color32::from_rgba_unmultiplied(18, 62, 82, 235),
            egui::Color32::from_rgb(56, 218, 255),
        ),
        SurfaceHintMode::NormalTangentToSurface => (
            egui::Color32::from_rgba_unmultiplied(78, 48, 16, 235),
            egui::Color32::from_rgb(255, 178, 46),
        ),
        SurfaceHintMode::ConformalSurfaceSkin => (
            egui::Color32::from_rgba_unmultiplied(16, 68, 42, 235),
            egui::Color32::from_rgb(38, 247, 136),
        ),
    };

    let mut local_frame =
        SurfaceFrame3d::from_point_and_normal(local_centroid, local_normal, Vec3::X);
    local_frame.elevation_mm = 0.25;
    let characteristic_len = patch_area_mm2.sqrt().max(12.0);
    let mm_per_point = (characteristic_len * 0.0042).clamp(0.10, 0.20);
    let panel_id = 0x2000_0000_0000_0000_u64 ^ (object_id << 16) ^ (patch_id as u64);

    SurfacePanel3d {
        id: panel_id,
        anchor: SurfaceAnchor3d::ObjectSurface {
            object_id,
            local_frame,
        },
        mm_per_point,
        bg_color: egui::Color32::from_rgba_unmultiplied(14, 18, 25, 228),
        border_color: mode_border,
        rounding_pt: 7.0,
        padding_pt: pad,
        panel_rect_pt,
        widgets: vec![
            SurfaceWidget3d {
                id: 1,
                rect_pt: mode_btn_rect,
                kind: SurfaceWidgetKind::Button {
                    icon: Some(mode_icon),
                    text: mode_text,
                    selected: true,
                    text_color: egui::Color32::from_rgb(240, 248, 255),
                    fill_color: mode_fill,
                    border_color: mode_border,
                    action: SurfaceWidgetAction::CycleSurfaceHintMode {
                        object_id,
                        patch_id,
                    },
                },
            },
            SurfaceWidget3d {
                id: 2,
                rect_pt: close_btn_rect,
                kind: SurfaceWidgetKind::Button {
                    icon: Some(UiIcon::DeleteObject),
                    text: String::new(),
                    selected: false,
                    text_color: egui::Color32::from_rgb(255, 165, 165),
                    fill_color: egui::Color32::from_rgba_unmultiplied(64, 24, 28, 230),
                    border_color: egui::Color32::from_rgb(215, 85, 85),
                    action: SurfaceWidgetAction::RemoveSurfaceHint {
                        object_id,
                        patch_id,
                    },
                },
            },
            SurfaceWidget3d {
                id: 3,
                rect_pt: slider_rect,
                kind: SurfaceWidgetKind::Slider {
                    label: "Weight".to_string(),
                    value: weight,
                    min: 0.05,
                    max: 1.0,
                    suffix: String::new(),
                    action: SurfaceWidgetAction::SetSurfaceHintWeight {
                        object_id,
                        patch_id,
                    },
                },
            },
        ],
    }
}

/// Tessellates a [`SurfacePanel3d`] into a 3D Bevy [`Mesh`] positioned on `world_frame`.
fn tessellate_surface_panel_to_bevy_mesh(
    ctx: &egui::Context,
    panel: &SurfacePanel3d,
    world_frame: &SurfaceFrame3d,
    hovered_widget_id: Option<u64>,
) -> Mesh {
    let mut clipped_shapes: Vec<egui::epaint::ClippedShape> = Vec::new();
    let clip_rect = panel.panel_rect_pt.expand(16.0);

    // 1. Panel background & outer border (if non-transparent)
    if panel.bg_color.a() > 0 || panel.border_color.a() > 0 {
        clipped_shapes.push(egui::epaint::ClippedShape {
            clip_rect,
            shape: egui::Shape::rect_filled(panel.panel_rect_pt, panel.rounding_pt, panel.bg_color),
        });
        if panel.border_color.a() > 0 {
            clipped_shapes.push(egui::epaint::ClippedShape {
                clip_rect,
                shape: egui::Shape::rect_stroke(
                    panel.panel_rect_pt,
                    panel.rounding_pt,
                    egui::Stroke::new(1.5_f32, panel.border_color),
                ),
            });
        }
    }

    // 2. Child widgets
    for widget in &panel.widgets {
        let is_hovered = hovered_widget_id == Some(widget.id);
        match &widget.kind {
            SurfaceWidgetKind::Label {
                text,
                color,
                monospace,
            } => {
                let font_id = if *monospace {
                    egui::FontId::monospace(18.0)
                } else {
                    egui::FontId::proportional(18.0)
                };
                let galley = ctx.fonts(|f| f.layout_no_wrap(text.clone(), font_id, *color));
                let text_pos = widget.rect_pt.center() - 0.5 * galley.size();
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::galley(text_pos, galley, *color),
                });
            }
            SurfaceWidgetKind::Button {
                icon,
                text,
                selected,
                text_color,
                fill_color,
                border_color,
                ..
            } => {
                let fill = if is_hovered {
                    egui::Color32::from_rgba_unmultiplied(
                        fill_color.r().saturating_add(24),
                        fill_color.g().saturating_add(28),
                        fill_color.b().saturating_add(34),
                        fill_color.a(),
                    )
                } else {
                    *fill_color
                };
                let stroke_w = if is_hovered || *selected {
                    2.0_f32
                } else {
                    1.4_f32
                };
                let border = if is_hovered {
                    egui::Color32::from_rgb(255, 238, 145)
                } else {
                    *border_color
                };
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_filled(widget.rect_pt, 6.0, fill),
                });
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_stroke(
                        widget.rect_pt,
                        6.0,
                        egui::Stroke::new(stroke_w, border),
                    ),
                });

                let icon_side = (widget.rect_pt.height() * 0.62).clamp(12.0, 20.0);
                if text.is_empty() {
                    if let Some(ic) = icon {
                        let icon_rect = egui::Rect::from_center_size(
                            widget.rect_pt.center(),
                            egui::vec2(icon_side, icon_side),
                        );
                        for shape in build_icon_shapes(*ic, icon_rect, *text_color) {
                            clipped_shapes.push(egui::epaint::ClippedShape { clip_rect, shape });
                        }
                    }
                } else {
                    let font_id = egui::FontId::monospace(19.5);
                    let galley =
                        ctx.fonts(|f| f.layout_no_wrap(text.clone(), font_id, *text_color));
                    if let Some(ic) = icon {
                        let gap = 6.0_f32;
                        let total_w = icon_side + gap + galley.size().x;
                        let left_x = widget.rect_pt.center().x - 0.5 * total_w;
                        let icon_rect = egui::Rect::from_center_size(
                            egui::pos2(left_x + 0.5 * icon_side, widget.rect_pt.center().y),
                            egui::vec2(icon_side, icon_side),
                        );
                        for shape in build_icon_shapes(*ic, icon_rect, *text_color) {
                            clipped_shapes.push(egui::epaint::ClippedShape { clip_rect, shape });
                        }
                        let text_pos = egui::pos2(
                            left_x + icon_side + gap,
                            widget.rect_pt.center().y - 0.5 * galley.size().y,
                        );
                        clipped_shapes.push(egui::epaint::ClippedShape {
                            clip_rect,
                            shape: egui::Shape::galley(text_pos, galley, *text_color),
                        });
                    } else {
                        let text_pos = widget.rect_pt.center() - 0.5 * galley.size();
                        clipped_shapes.push(egui::epaint::ClippedShape {
                            clip_rect,
                            shape: egui::Shape::galley(text_pos, galley, *text_color),
                        });
                    }
                }
            }
            SurfaceWidgetKind::Toggle { label, value, .. } => {
                let fill = if *value {
                    egui::Color32::from_rgba_unmultiplied(24, 72, 92, 230)
                } else {
                    egui::Color32::from_rgba_unmultiplied(24, 28, 36, 220)
                };
                let border = if is_hovered {
                    egui::Color32::from_rgb(255, 232, 130)
                } else if *value {
                    egui::Color32::from_rgb(72, 214, 250)
                } else {
                    egui::Color32::from_gray(110)
                };
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_filled(widget.rect_pt, 5.0, fill),
                });
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_stroke(
                        widget.rect_pt,
                        5.0,
                        egui::Stroke::new(1.4_f32, border),
                    ),
                });
                let txt = format!(
                    "{} {}",
                    if *value { "[✓]" } else { "[ ]" },
                    label
                );
                let galley = ctx.fonts(|f| {
                    f.layout_no_wrap(
                        txt,
                        egui::FontId::proportional(17.0),
                        egui::Color32::from_gray(240),
                    )
                });
                let text_pos = widget.rect_pt.center() - 0.5 * galley.size();
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::galley(text_pos, galley, egui::Color32::from_gray(240)),
                });
            }
            SurfaceWidgetKind::Slider {
                label,
                value,
                min,
                max,
                suffix,
                ..
            } => {
                let track_bg = egui::Color32::from_rgba_unmultiplied(22, 27, 36, 235);
                let border = if is_hovered {
                    egui::Color32::from_rgb(255, 232, 130)
                } else {
                    egui::Color32::from_gray(95)
                };
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_filled(widget.rect_pt, 5.0, track_bg),
                });
                let t = ((*value - *min) / (*max - *min).max(1.0e-5)).clamp(0.0, 1.0);
                let fill_w = (widget.rect_pt.width() * t).max(2.0);
                let fill_rect = egui::Rect::from_min_size(
                    widget.rect_pt.min,
                    egui::vec2(fill_w, widget.rect_pt.height()),
                );
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_filled(
                        fill_rect,
                        5.0,
                        egui::Color32::from_rgba_unmultiplied(36, 128, 175, 225),
                    ),
                });
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::rect_stroke(
                        widget.rect_pt,
                        5.0,
                        egui::Stroke::new(1.3_f32, border),
                    ),
                });
                let txt = format!("{label}: {value:.2}{suffix}");
                let galley = ctx.fonts(|f| {
                    f.layout_no_wrap(
                        txt,
                        egui::FontId::monospace(15.5),
                        egui::Color32::from_rgb(245, 248, 255),
                    )
                });
                let text_pos = widget.rect_pt.center() - 0.5 * galley.size();
                clipped_shapes.push(egui::epaint::ClippedShape {
                    clip_rect,
                    shape: egui::Shape::galley(
                        text_pos,
                        galley,
                        egui::Color32::from_rgb(245, 248, 255),
                    ),
                });
            }
        }
    }

    let primitives = ctx.tessellate(clipped_shapes, 2.0);
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let n_arr = world_frame.normal.to_array();

    for (prim_idx, clipped_prim) in primitives.into_iter().enumerate() {
        let egui::epaint::Primitive::Mesh(emesh) = clipped_prim.primitive else {
            continue;
        };
        // Step each primitive layer by 1.5 microns along the surface normal so Painter's-order
        // strokes and font glyphs never z-fight with the panel's background rectangle.
        let layer_offset_mm = (prim_idx as f32) * 0.0015;
        let base_v = positions.len() as u32;

        for v in &emesh.vertices {
            let uv_mm = panel.panel_pt_to_uv_mm(v.pos);
            let world_p = world_frame.local_to_world(uv_mm.x, uv_mm.y, layer_offset_mm);
            positions.push(world_p.to_array());
            normals.push(n_arr);
            uvs.push([v.uv.x, v.uv.y]);
            let [r, g, b, a] = v.color.to_srgba_unmultiplied();
            colors.push(srgba_to_linear_rgba([
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0,
            ]));
        }

        for idx in emesh.indices {
            indices.push(base_v + idx);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

/// Ensures the shared `egui` font atlas `Image` and `StandardMaterial` are up to date in Bevy.
fn sync_egui_font_atlas_material(
    ctx: &egui::Context,
    surface_ui: &mut SurfaceUiState,
    images: &mut Assets<Image>,
    materials: &mut Assets<StandardMaterial>,
) -> Handle<StandardMaterial> {
    let (w, h, checksum) = ctx.fonts(|f| {
        let img = f.image();
        let w = img.width();
        let h = img.height();
        let stride = (img.pixels.len() / 64).max(1);
        let mut hash = (img.pixels.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for &px in img.pixels.iter().step_by(stride) {
            hash = hash.rotate_left(5) ^ (px.to_bits() as u64);
        }
        (w, h, hash)
    });

    let needs_update = surface_ui.font_atlas_material.is_none()
        || surface_ui.font_atlas_signature != (w, h, checksum);

    if needs_update && w > 0 && h > 0 {
        let rgba_bytes: Vec<u8> = ctx.fonts(|f| {
            let img = f.image();
            let mut bytes = Vec::with_capacity(w * h * 4);
            for c in img.srgba_pixels(None) {
                bytes.extend_from_slice(&c.to_array());
            }
            bytes
        });

        let bevy_img = Image::new(
            Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba_bytes,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let new_img_h = images.add(bevy_img);
        if let Some(old_img_h) = surface_ui.font_atlas_image.replace(new_img_h.clone()) {
            images.remove(&old_img_h);
        }

        if let Some(ref mat_h) = surface_ui.font_atlas_material {
            if let Some(mat) = materials.get_mut(mat_h) {
                mat.base_color_texture = Some(new_img_h);
            }
        } else {
            let mat_h = materials.add(StandardMaterial {
                base_color: Color::WHITE,
                base_color_texture: Some(new_img_h),
                unlit: true,
                alpha_mode: AlphaMode::Blend,
                depth_bias: 12.0,
                double_sided: true,
                cull_mode: None,
                ..default()
            });
            surface_ui.font_atlas_material = Some(mat_h);
        }
        surface_ui.font_atlas_signature = (w, h, checksum);
    }

    surface_ui
        .font_atlas_material
        .clone()
        .unwrap_or_default()
}

/// Rebuilds the active 3D surface panels (build-plate object labels + surface-mounted widgets),
/// performs 3D raycast interaction against them (occluded by closer 3D scene geometry), and
/// synchronizes their 3D Bevy `PbrBundle` meshes.
#[allow(clippy::too_many_arguments)]
pub fn sync_and_interact_surface_ui_system(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut contexts: EguiContexts,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    mut object_transforms: Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut surface_ui: ResMut<SurfaceUiState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let ctx = contexts.ctx_mut();

    // 1. Assemble all active 3D surface panels for this frame
    let mut panels: Vec<SurfacePanel3d> = Vec::new();

    if plate.show_object_labels {
        let active_plate_idx = plate.active_plate_index;
        let selected_id = plate.selected_object_id;
        let active_tool = plate.tool_mode;
        for obj in plate
            .objects
            .iter()
            .filter(|o| o.visible && o.plate_index == active_plate_idx)
        {
            let Ok((obj_tf, _)) = object_transforms.get(obj.entity) else {
                continue;
            };
            let (poly_xy, center_xy) =
                obj.exclude_object_polygon_and_center_scene_xy(obj_tf, &config, &vis_state);
            let is_sel = selected_id == Some(obj.object_id);
            panels.push(build_object_plate_label_panel(
                ctx,
                obj.object_id,
                obj.exclude_object_name(),
                &poly_xy,
                center_xy,
                is_sel,
                active_tool,
            ));
        }
    }

    // Mount an interactive 3D surface widget panel on the active object's selected surface hint patch
    if let (Some(sel_id), Some(patch_id)) =
        (plate.selected_object_id, plate.selected_surface_hint_patch)
    {
        if let Some(hint) = config
            .eikonal
            .surface_hints
            .iter()
            .find(|h| h.object_id == sel_id && h.patch_id == patch_id)
        {
            if let Some(obj) = plate
                .objects
                .iter()
                .find(|o| o.object_id == sel_id && o.visible && o.plate_index == plate.active_plate_index)
            {
                if let Some(patch) = obj.surface_patches.get(patch_id as usize) {
                    panels.push(build_surface_hint_widget_panel(
                        ctx,
                        sel_id,
                        patch_id,
                        patch.centroid,
                        patch.normal,
                        patch.area,
                        hint.mode,
                        hint.weight,
                    ));
                }
            }
        }
    }

    for custom in &surface_ui.custom_panels {
        panels.push(custom.clone());
    }

    // 2. Raycast cursor in 3D space against all active 3D surface panels
    let egui_wants_pointer = ctx.wants_pointer_input() || ctx.is_pointer_over_area();
    let cursor_ray: Option<Ray3d> = if !egui_wants_pointer {
        if let (Ok(window), Ok((camera, cam_global))) =
            (window_query.get_single(), camera_query.get_single())
        {
            window
                .cursor_position()
                .and_then(|pos| camera.viewport_to_world(cam_global, pos))
        } else {
            None
        }
    } else {
        None
    };

    if mouse_buttons.just_released(MouseButton::Left) {
        surface_ui.active_slider_drag = None;
    }

    // Handle active 3D surface slider dragging
    if let (Some((drag_panel_id, drag_widget_id)), Some(ray)) =
        (surface_ui.active_slider_drag, cursor_ray)
    {
        if mouse_buttons.pressed(MouseButton::Left) {
            if let Some(panel) = panels.iter().find(|p| p.id == drag_panel_id) {
                if let Some(world_frame) = panel.anchor.resolve_world_frame(&plate, &object_transforms) {
                    if let Some((_, uv_mm)) = world_frame.ray_intersect_local_uv(ray) {
                        let pt = panel.uv_mm_to_panel_pt(uv_mm);
                        if let Some(w) = panel.widgets.iter().find(|w| w.id == drag_widget_id) {
                            if let SurfaceWidgetKind::Slider {
                                min, max, action, ..
                            } = w.kind
                            {
                                let t = ((pt.x - w.rect_pt.left()) / w.rect_pt.width().max(1.0))
                                    .clamp(0.0, 1.0);
                                let new_val = min + t * (max - min);
                                apply_surface_slider_action(action, new_val, &mut config);
                                mouse_motion.clear();
                            }
                        }
                    }
                }
            }
        }
    }

    // Find closest unoccluded 3D surface widget under the cursor ray
    surface_ui.hovered_widget = None;
    let mut hovered_action: Option<(u64, u64, SurfaceWidgetAction, Option<(f32, f32, egui::Rect, egui::Pos2)>)> = None;

    if let Some(ray) = cursor_ray {
        // Compute closest 3D object mesh intersection distance so occluded surface widgets behind
        // an object do not capture hover/clicks
        let mut closest_obj_dist = f32::INFINITY;
        for obj in plate
            .objects
            .iter()
            .filter(|o| o.visible && o.plate_index == plate.active_plate_index)
        {
            if let Ok((obj_tf, _)) = object_transforms.get(obj.entity) {
                if let Some(d) = obj.raycast_mesh(obj_tf, ray) {
                    closest_obj_dist = closest_obj_dist.min(d);
                }
            }
        }

        let mut best_panel_t = f32::INFINITY;
        for panel in &panels {
            let Some(world_frame) = panel.anchor.resolve_world_frame(&plate, &object_transforms) else {
                continue;
            };
            let Some((ray_t, uv_mm)) = world_frame.ray_intersect_local_uv(ray) else {
                continue;
            };
            // Allow a small tolerance (1.5 mm) so widgets mounted directly on an object's surface patch
            // are not self-occluded by the host triangle they sit on.
            if ray_t > closest_obj_dist + 1.5 || ray_t >= best_panel_t {
                continue;
            }
            if let Some(widget) = panel.hit_test_widget(uv_mm) {
                best_panel_t = ray_t;
                surface_ui.hovered_widget = Some((panel.id, widget.id));
                let pt = panel.uv_mm_to_panel_pt(uv_mm);
                match widget.kind {
                    SurfaceWidgetKind::Button { action, .. }
                    | SurfaceWidgetKind::Toggle { action, .. } => {
                        hovered_action = Some((panel.id, widget.id, action, None));
                    }
                    SurfaceWidgetKind::Slider {
                        min, max, action, ..
                    } => {
                        hovered_action =
                            Some((panel.id, widget.id, action, Some((min, max, widget.rect_pt, pt))));
                    }
                    SurfaceWidgetKind::Label { .. } => {}
                }
            }
        }
    }

    if mouse_buttons.just_pressed(MouseButton::Left) {
        if let Some((panel_id, widget_id, action, slider_info)) = hovered_action {
            if let Some((min, max, rect_pt, hit_pt)) = slider_info {
                surface_ui.active_slider_drag = Some((panel_id, widget_id));
                let t = ((hit_pt.x - rect_pt.left()) / rect_pt.width().max(1.0)).clamp(0.0, 1.0);
                let new_val = min + t * (max - min);
                apply_surface_slider_action(action, new_val, &mut config);
            } else {
                apply_surface_click_action(
                    action,
                    &mut commands,
                    render_device.as_deref(),
                    &mut meshes,
                    &mut materials,
                    &mut plate,
                    &mut config,
                    &mut vis_state,
                    &mut object_transforms,
                );
            }
            mouse_motion.clear();
        }
    }

    // 3. Synchronize 3D Bevy meshes for all active panels and despawn stale ones
    let mat_handle =
        sync_egui_font_atlas_material(ctx, &mut surface_ui, &mut images, &mut materials);
    let hovered_pair = surface_ui.hovered_widget;

    let mut active_ids = std::collections::HashSet::with_capacity(panels.len());
    for panel in &panels {
        let Some(world_frame) = panel.anchor.resolve_world_frame(&plate, &object_transforms) else {
            continue;
        };
        active_ids.insert(panel.id);
        let hovered_w_id = hovered_pair.and_then(|(pid, wid)| (pid == panel.id).then_some(wid));
        let bevy_mesh =
            tessellate_surface_panel_to_bevy_mesh(ctx, panel, &world_frame, hovered_w_id);

        if let Some(cached) = surface_ui.panel_gpu_cache.get(&panel.id) {
            meshes.insert(&cached.mesh_handle, bevy_mesh);
        } else {
            let mesh_h = meshes.add(bevy_mesh);
            let entity = commands
                .spawn((
                    PbrBundle {
                        mesh: mesh_h.clone(),
                        material: mat_handle.clone(),
                        ..default()
                    },
                    SurfacePanelMeshEntity { panel_id: panel.id },
                ))
                .id();
            surface_ui.panel_gpu_cache.insert(
                panel.id,
                CachedSurfacePanelGpu {
                    entity,
                    mesh_handle: mesh_h,
                },
            );
        }
    }

    let stale_ids: Vec<u64> = surface_ui
        .panel_gpu_cache
        .keys()
        .copied()
        .filter(|id| !active_ids.contains(id))
        .collect();
    for stale_id in stale_ids {
        if let Some(cached) = surface_ui.panel_gpu_cache.remove(&stale_id) {
            commands.entity(cached.entity).despawn();
            meshes.remove(&cached.mesh_handle);
        }
    }

    surface_ui.panels = panels;
}

#[allow(clippy::too_many_arguments)]
fn apply_surface_click_action(
    action: SurfaceWidgetAction,
    commands: &mut Commands,
    render_device: Option<&RenderDevice>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    plate: &mut BuildPlateState,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    object_transforms: &mut Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
) {
    match action {
        SurfaceWidgetAction::SelectObject(obj_id) => {
            plate.selected_object_id = Some(obj_id);
            if let Some(obj) = plate.objects.iter().find(|o| o.object_id == obj_id) {
                plate.status_message = format!(
                    "Selected '{}' via 3D build-plate label.",
                    obj.name
                );
            }
        }
        SurfaceWidgetAction::ActivateToolForObject { object_id, mode } => {
            plate.selected_object_id = Some(object_id);
            plate.tool_mode = if plate.tool_mode == mode {
                ToolMode::Select
            } else {
                mode
            };
            if let Some(obj) = plate.objects.iter().find(|o| o.object_id == object_id) {
                plate.status_message = format!(
                    "Set tool to '{}' for '{}'.",
                    plate.tool_mode.label(),
                    obj.name
                );
            }
        }
        SurfaceWidgetAction::CenterObjectOnBed(object_id) => {
            if let Some(obj_idx) = plate.objects.iter().position(|o| o.object_id == object_id) {
                let ent = plate.objects[obj_idx].entity;
                let obj_name = plate.objects[obj_idx].name.clone();
                if let Ok((mut tf, _)) = object_transforms.get_mut(ent) {
                    let (w_min, w_max) = plate.objects[obj_idx].compute_world_aabb(&tf);
                    let w_center_xy = 0.5 * Vec2::new(w_min.x + w_max.x, w_min.y + w_max.y);
                    tf.translation.x -= w_center_xy.x;
                    tf.translation.y -= w_center_xy.y;
                    snap_object_to_bed(&plate.objects[obj_idx], &mut tf);
                    vis_state.dirty_planned_gcode = true;
                    plate.status_message = format!("Centered '{obj_name}' at (0, 0) on the build plate.");
                }
            }
        }
        SurfaceWidgetAction::RotateObjectZ90(object_id) => {
            if let Some(obj_idx) = plate.objects.iter().position(|o| o.object_id == object_id) {
                let ent = plate.objects[obj_idx].entity;
                let obj_name = plate.objects[obj_idx].name.clone();
                if let Ok((mut tf, _)) = object_transforms.get_mut(ent) {
                    tf.rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2) * tf.rotation;
                    snap_object_to_bed(&plate.objects[obj_idx], &mut tf);
                    vis_state.dirty_planned_gcode = true;
                    plate.status_message = format!("Rotated '{obj_name}' +90° around Z.");
                }
            }
        }
        SurfaceWidgetAction::CloneObject(object_id) => {
            clone_object_on_plate(
                commands,
                render_device,
                meshes,
                materials,
                plate,
                config,
                object_transforms,
                object_id,
                1,
            );
        }
        SurfaceWidgetAction::DeleteObject(object_id) => {
            remove_object_from_plate(commands, plate, object_id);
        }
        SurfaceWidgetAction::CycleSurfaceHintMode {
            object_id,
            patch_id,
        } => {
            if let Some(h) = config
                .eikonal
                .surface_hints
                .iter_mut()
                .find(|h| h.object_id == object_id && h.patch_id == patch_id)
            {
                h.mode = match h.mode {
                    SurfaceHintMode::NormalOrthogonalToSurface => {
                        SurfaceHintMode::NormalTangentToSurface
                    }
                    SurfaceHintMode::NormalTangentToSurface => {
                        SurfaceHintMode::ConformalSurfaceSkin
                    }
                    SurfaceHintMode::ConformalSurfaceSkin => {
                        SurfaceHintMode::NormalOrthogonalToSurface
                    }
                };
                vis_state.dirty_selected_layers = true;
                vis_state.dirty_toolpaths = true;
                plate.status_message = format!(
                    "Switched patch #{patch_id} hint to '{}' via 3D surface widget.",
                    h.mode.label()
                );
            }
        }
        SurfaceWidgetAction::RemoveSurfaceHint {
            object_id,
            patch_id,
        } => {
            config
                .eikonal
                .surface_hints
                .retain(|h| !(h.object_id == object_id && h.patch_id == patch_id));
            if plate.selected_surface_hint_patch == Some(patch_id) {
                plate.selected_surface_hint_patch = None;
            }
            vis_state.dirty_selected_layers = true;
            vis_state.dirty_toolpaths = true;
            plate.status_message =
                format!("Removed surface orientation hint on patch #{patch_id}.");
        }
        SurfaceWidgetAction::SetSurfaceHintWeight { .. } | SurfaceWidgetAction::Custom(_) => {}
    }
}

fn apply_surface_slider_action(
    action: SurfaceWidgetAction,
    new_val: f32,
    config: &mut SlicerConfig,
) {
    if let SurfaceWidgetAction::SetSurfaceHintWeight {
        object_id,
        patch_id,
    } = action
    {
        if let Some(h) = config
            .eikonal
            .surface_hints
            .iter_mut()
            .find(|h| h.object_id == object_id && h.patch_id == patch_id)
        {
            h.weight = new_val.clamp(0.05, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_frame_3d_raycast_and_transform_roundtrip() {
        let plate_frame = SurfaceFrame3d::build_plate_at(Vec2::new(15.0, -40.0));
        assert_eq!(plate_frame.u_axis, Vec3::X);
        assert_eq!(plate_frame.v_axis, Vec3::Y);
        assert_eq!(plate_frame.normal, Vec3::Z);

        // World point at local (u = +8 mm, v = -3 mm) on the elevated plate frame
        let wp = plate_frame.local_to_world(8.0, -3.0, 0.0);
        assert!((wp.x - 23.0).abs() < 1.0e-5);
        assert!((wp.y - (-43.0)).abs() < 1.0e-5);
        assert!((wp.z - plate_frame.elevation_mm).abs() < 1.0e-5);

        // Cast an oblique ray from above through `wp`
        let ray_origin = Vec3::new(0.0, -80.0, 60.0);
        let ray = Ray3d::new(ray_origin, (wp - ray_origin).normalize());
        let (t, uv) = plate_frame.ray_intersect_local_uv(ray).unwrap();
        assert!(t > 0.0);
        assert!((uv.x - 8.0).abs() < 1.0e-4);
        assert!((uv.y - (-3.0)).abs() < 1.0e-4);

        // Verify arbitrary 3D surface frame construction and object Transform propagation
        let local_frame = SurfaceFrame3d::from_point_and_normal(
            Vec3::new(10.0, 0.0, 5.0),
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::X,
        );
        assert!((local_frame.u_axis.dot(local_frame.normal)).abs() < 1.0e-5);
        assert!((local_frame.v_axis.dot(local_frame.normal)).abs() < 1.0e-5);
        assert!((local_frame.u_axis.dot(local_frame.v_axis)).abs() < 1.0e-5);

        let tf = Transform::from_xyz(12.0, -8.0, 0.0)
            .with_rotation(Quat::from_rotation_z(0.65));
        let world_f = local_frame.transformed(&tf);
        assert!((world_f.u_axis.length() - 1.0).abs() < 1.0e-5);
        assert!((world_f.v_axis.length() - 1.0).abs() < 1.0e-5);
        assert!((world_f.normal.length() - 1.0).abs() < 1.0e-5);
        assert!((world_f.u_axis.cross(world_f.v_axis) - world_f.normal).length() < 1.0e-4);
    }

    #[test]
    fn test_surface_panel_3d_coordinate_mapping_and_hit_test() {
        let rect_pt = egui::Rect::from_center_size(egui::Pos2::ZERO, egui::vec2(120.0, 32.0));
        let panel = SurfacePanel3d {
            id: 42,
            anchor: SurfaceAnchor3d::BuildPlate(Vec2::new(0.0, -30.0)),
            mm_per_point: 0.20,
            bg_color: egui::Color32::BLACK,
            border_color: egui::Color32::WHITE,
            rounding_pt: 6.0,
            padding_pt: egui::vec2(8.0, 4.0),
            panel_rect_pt: rect_pt,
            widgets: vec![SurfaceWidget3d {
                id: 7,
                rect_pt,
                kind: SurfaceWidgetKind::Button {
                    icon: Some(UiIcon::ObjectLabels),
                    text: "PART_1".to_string(),
                    selected: false,
                    text_color: egui::Color32::WHITE,
                    fill_color: egui::Color32::BLACK,
                    border_color: egui::Color32::WHITE,
                    action: SurfaceWidgetAction::SelectObject(1),
                },
            }],
        };

        assert!((panel.size_mm().x - 24.0).abs() < 1.0e-4);
        assert!((panel.size_mm().y - 6.4).abs() < 1.0e-4);
        assert!(panel.hit_test_widget(Vec2::new(5.0, -1.5)).is_some());
        assert!(panel.hit_test_widget(Vec2::new(20.0, 0.0)).is_none());
    }

    #[test]
    fn test_ui_icons_generate_non_empty_vector_shapes_within_bounds() {
        let target_rect = egui::Rect::from_min_max(egui::pos2(10.0, 20.0), egui::pos2(34.0, 44.0));
        let expanded = target_rect.expand(2.5);
        for icon in UiIcon::ALL {
            let shapes = build_icon_shapes(icon, target_rect, egui::Color32::WHITE);
            assert!(
                !shapes.is_empty(),
                "Icon {icon:?} produced zero vector shapes"
            );
            for shape in &shapes {
                let b = shape.visual_bounding_rect();
                assert!(
                    expanded.contains_rect(b),
                    "Icon {icon:?} shape bounding rect {b:?} exceeded {expanded:?}"
                );
            }
        }
    }
}
