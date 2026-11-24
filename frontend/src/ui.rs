use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, IoTaskPool};
use bevy_egui::{egui, EguiContexts};

use crate::bridge::ffi;
use crate::calibration::{
    generate_em_calibration_gcode, generate_pa_calibration_gcode,
    generate_transient_pressure_calibration_gcode, CalibrationUiState, CalibrationUtilityKind,
    EmCalibrationPass, PaCalibrationMode, TransientCalibrationPass,
};
use crate::config::{
    lookup_parameter_doc, BedShape, BrickLayerMode, ClearancePointXZ, ClearanceProfileMode,
    FiveAxisKinematicsMode, GcodeDialect, InfillFilamentMode, ObjectFilamentMode, OrderFieldMode,
    PointNormalHint, PrinterNetworkProtocol, PurgeMode, SlicerConfig, SolidInfillPattern,
    SparseInfillPattern, SupportStyle, SurfaceHintMode, ToolConfig, WallPrintOrder,
};
use crate::field_vis::{
    high_contrast_parameter_ramp, nozzle_sdf_reference_layer_height_mm, nozzle_sdf_signed_t,
    raw_toolpaths_sdf_observed_range, sample_nozzle_sdf_rgba_for_signed_t,
    sample_slice_colormap_rgba, FieldColormap, FieldVisualizationState, IsoIntersectionMode,
    ObjectSurfaceColorMode, SelectedLayersColorMode, SliceAxis, ToolpathColorMode,
    ToolpathQualityMode, BRANCH_PALETTE,
};
use crate::hull::rotation_to_place_face_on_bed;
use crate::io::threemf::{save_project_to_3mf, ExportableObject};
use crate::io::load_files_into_backend;
use crate::moonraker::{
    open_printer_web_ui_in_browser, query_moonraker_print_status, resolve_printer_web_ui_url,
    test_moonraker_connection, upload_gcode_to_moonraker_with_progress, CertPickKind,
    MoonrakerUiState,
};
use crate::presets::{LeftPanelTab, PresetCategory, PresetLibrary};
use crate::scene::{
    arrange_selected_subset_to_new_plate, clone_object_on_plate, compute_orbit_eye,
    remove_object_from_plate, run_auto_arrange_on_plate, snap_object_to_bed,
    spawn_loaded_objects, split_assembly_to_plates_by_material, BuildPlateInfo, BuildPlateState,
    CadObjectEntity, OrbitCamera, ToolMode,
};
use crate::surface_ui::{render_delineated_icon_pill_tab, render_icon_button, UiIcon};

fn rgba_to_color32(rgba: [f32; 4]) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        (rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (rgba[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

/// Renders a high-contrast horizontal color-bar legend with 5 specific numerical tick marks
/// (`0%`, `25%`, `50%`, `75%`, `100%`) and individual color swatches for each tick value.
fn draw_continuous_gradient_legend(
    ui: &mut egui::Ui,
    title: &str,
    min_val: f32,
    max_val: f32,
    unit: &str,
    decimals: usize,
    color_fn: impl Fn(f32) -> [f32; 4],
) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            let header = if unit.is_empty() {
                title.to_string()
            } else {
                format!("{title} ({unit})")
            };
            ui.small(
                egui::RichText::new(header)
                    .strong()
                    .color(egui::Color32::from_rgb(230, 238, 250)),
            );

            // 1. Multi-stop continuous gradient bar with 5 tick marks
            let bar_width = ui.available_width().clamp(160.0, 340.0);
            let bar_height = 14.0;
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(bar_width, bar_height + 5.0), egui::Sense::hover());
            let bar_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, bar_height));

            let steps = 48usize;
            let painter = ui.painter();
            for i in 0..steps {
                let t0 = (i as f32) / (steps as f32);
                let t1 = ((i + 1) as f32) / (steps as f32);
                let tm = 0.5 * (t0 + t1);
                let x0 = bar_rect.left() + bar_width * t0;
                let x1 = bar_rect.left() + bar_width * t1;
                let seg_rect = egui::Rect::from_min_max(
                    egui::pos2(x0, bar_rect.top()),
                    egui::pos2(x1 + 0.5, bar_rect.bottom()),
                );
                painter.rect_filled(seg_rect, 0.0, rgba_to_color32(color_fn(tm)));
            }
            painter.rect_stroke(
                bar_rect,
                2.0,
                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(95)),
            );

            // Draw 5 crisp tick lines at 0%, 25%, 50%, 75%, 100%
            for tick_idx in 0..=4 {
                let frac = (tick_idx as f32) * 0.25;
                let x = (bar_rect.left() + bar_width * frac)
                    .clamp(bar_rect.left() + 0.5, bar_rect.right() - 0.5);
                painter.line_segment(
                    [
                        egui::pos2(x, bar_rect.bottom() - 4.0),
                        egui::pos2(x, rect.bottom()),
                    ],
                    egui::Stroke::new(1.25_f32, egui::Color32::from_gray(235)),
                );
            }

            // 2. Specific numerical tick values with color swatches at 0%, 25%, 50%, 75%, 100%
            ui.columns(5, |cols| {
                for (i, col) in cols.iter_mut().enumerate() {
                    let frac = (i as f32) * 0.25;
                    let val = min_val + (max_val - min_val) * frac;
                    let c32 = rgba_to_color32(color_fn(frac));
                    col.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        let (dot_rect, _) =
                            ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
                        ui.painter().rect_filled(dot_rect, 2.0, c32);
                        let txt = format!("{val:.decimals$}");
                        ui.label(
                            egui::RichText::new(txt)
                                .size(10.5)
                                .monospace()
                                .color(egui::Color32::from_gray(235)),
                        );
                    });
                }
            });
        });
}

/// Renders a centered non-linear diverging colorbar legend for `ToolpathColorMode::ByNozzleSdf`,
/// keeping `0.0 mm` at center (`50%`) and marking `[-h, -h/4, 0, +h/4, +h]` along the scale.
fn draw_nozzle_sdf_legend(
    ui: &mut egui::Ui,
    h_ref_mm: f32,
    observed_range: Option<(f32, f32)>,
) {
    let decimals = if h_ref_mm < 0.16 { 3 } else { 2 };
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            ui.small(
                egui::RichText::new(format!(
                    "Nozzle SDF φ_sdf (mm, h = {h_ref_mm:.decimals$} mm)"
                ))
                .strong()
                .color(egui::Color32::from_rgb(230, 238, 250)),
            );

            let bar_width = ui.available_width().clamp(160.0, 340.0);
            let bar_height = 14.0;
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(bar_width, bar_height + 5.0),
                egui::Sense::hover(),
            );
            let bar_rect =
                egui::Rect::from_min_size(rect.min, egui::vec2(bar_width, bar_height));

            let steps = 64usize;
            let painter = ui.painter();
            for i in 0..steps {
                let t0 = (i as f32) / (steps as f32);
                let t1 = ((i + 1) as f32) / (steps as f32);
                let signed_t = 2.0 * (0.5 * (t0 + t1)) - 1.0;
                let x0 = bar_rect.left() + bar_width * t0;
                let x1 = bar_rect.left() + bar_width * t1;
                let seg_rect = egui::Rect::from_min_max(
                    egui::pos2(x0, bar_rect.top()),
                    egui::pos2(x1 + 0.5, bar_rect.bottom()),
                );
                painter.rect_filled(
                    seg_rect,
                    0.0,
                    rgba_to_color32(sample_nozzle_sdf_rgba_for_signed_t(signed_t)),
                );
            }
            painter.rect_stroke(
                bar_rect,
                2.0,
                egui::Stroke::new(1.0_f32, egui::Color32::from_gray(95)),
            );

            let landmarks = [-h_ref_mm, -0.25 * h_ref_mm, 0.0, 0.25 * h_ref_mm, h_ref_mm];
            for (idx, &val) in landmarks.iter().enumerate() {
                let signed_t = nozzle_sdf_signed_t(val, h_ref_mm);
                let frac = 0.5 * (1.0 + signed_t);
                let x = (bar_rect.left() + bar_width * frac)
                    .clamp(bar_rect.left() + 0.5, bar_rect.right() - 0.5);
                let top_y = if idx == 2 {
                    bar_rect.top() + 2.0
                } else {
                    bar_rect.bottom() - 4.0
                };
                painter.line_segment(
                    [egui::pos2(x, top_y), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(
                        if idx == 2 { 1.5_f32 } else { 1.25_f32 },
                        if idx == 2 {
                            egui::Color32::from_gray(45)
                        } else {
                            egui::Color32::from_gray(235)
                        },
                    ),
                );
            }

            ui.columns(5, |cols| {
                for (i, (col, &val)) in cols.iter_mut().zip(landmarks.iter()).enumerate() {
                    let signed_t = nozzle_sdf_signed_t(val, h_ref_mm);
                    let c32 = rgba_to_color32(sample_nozzle_sdf_rgba_for_signed_t(signed_t));
                    col.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 3.0;
                        let (dot_rect, _) =
                            ui.allocate_exact_size(egui::vec2(7.0, 7.0), egui::Sense::hover());
                        ui.painter().rect_filled(dot_rect, 2.0, c32);
                        let txt = if i == 2 {
                            format!("{:.decimals$}", 0.0_f32)
                        } else {
                            format!("{val:+.decimals$}")
                        };
                        ui.label(
                            egui::RichText::new(txt)
                                .size(10.5)
                                .monospace()
                                .color(egui::Color32::from_gray(235)),
                        );
                    });
                }
            });

            let quarter_h = 0.25 * h_ref_mm;
            let mut caption = format!(
                "Inside (<0) | ±¼h = ±{quarter_h:.decimals$} mm | Outside (>0)"
            );
            if let Some((lo, hi)) = observed_range {
                caption.push_str(&format!(" [slice: {lo:+.2}..{hi:+.2} mm]"));
            }
            ui.small(
                egui::RichText::new(caption).color(egui::Color32::from_rgb(180, 194, 214)),
            );
        });
}

/// Renders a categorical color swatch legend for discrete modes (Feature Types, Branches, Special Layers).
fn draw_categorical_swatch_legend(
    ui: &mut egui::Ui,
    title: &str,
    items: &[(&str, [f32; 4])],
) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            ui.small(
                egui::RichText::new(title)
                    .strong()
                    .color(egui::Color32::from_rgb(230, 238, 250)),
            );
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);
                for &(label, rgba) in items {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        let (swatch_rect, _) =
                            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter()
                            .rect_filled(swatch_rect, 2.0, rgba_to_color32(rgba));
                        ui.painter().rect_stroke(
                            swatch_rect,
                            2.0,
                            egui::Stroke::new(0.75_f32, egui::Color32::from_gray(80)),
                        );
                        ui.small(egui::RichText::new(label).color(egui::Color32::from_gray(230)));
                    });
                }
            });
        });
}

/// Renders a single interactive `[checkbox] [color swatch] Label (count)` cell in the combined
/// Line Types & Legend table. Clicking either the checkbox or the swatch/label toggles `flag`.
fn draw_line_type_toggle_cell(
    ui: &mut egui::Ui,
    flag: &mut bool,
    rgba: [f32; 4],
    label: &str,
    count: Option<u32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        if ui.checkbox(flag, "").changed() {
            changed = true;
        }
        let (swatch_rect, swatch_resp) =
            ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::click());
        let display_rgba = if *flag {
            rgba
        } else {
            [rgba[0] * 0.35, rgba[1] * 0.35, rgba[2] * 0.35, 0.55]
        };
        ui.painter()
            .rect_filled(swatch_rect, 2.0, rgba_to_color32(display_rgba));
        ui.painter().rect_stroke(
            swatch_rect,
            2.0,
            egui::Stroke::new(0.75_f32, egui::Color32::from_gray(85)),
        );

        let text = match count {
            Some(c) if c > 0 => format!("{label} ({c})"),
            _ => label.to_string(),
        };
        let text_color = if *flag {
            egui::Color32::from_gray(235)
        } else {
            egui::Color32::from_gray(135)
        };
        let lbl_resp = ui.add(
            egui::Label::new(egui::RichText::new(text).size(11.5).color(text_color))
                .sense(egui::Sense::click()),
        );
        if swatch_resp.clicked() || lbl_resp.clicked() {
            *flag = !*flag;
            changed = true;
        }
    });
    changed
}

/// Renders the combined interactive **Line Types & Legend** 2-column grid (combining visibility toggles,
/// color swatches, and path counts into a single compact card that fits cleanly in the right panel).
fn render_interactive_toolpath_line_types_legend(
    ui: &mut egui::Ui,
    vis_state: &mut FieldVisualizationState,
) {
    let counts = vis_state.cached_toolpaths.as_ref().map(|tp| {
        (
            tp.total_path_count,
            tp.outer_wall_count,
            tp.inner_wall_count,
            tp.bridge_count,
            tp.wave_overhang_count,
            tp.solid_infill_count,
            tp.sparse_infill_count,
            tp.support_count,
            tp.support_interface_count,
        )
    });

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(6.0, 5.0))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let header = if let Some((total, ..)) = counts {
                    format!("Line Types & Legend ({total} paths)")
                } else {
                    "Line Types & Legend".to_string()
                };
                ui.small(
                    egui::RichText::new(header)
                        .strong()
                        .color(egui::Color32::from_rgb(230, 238, 250)),
                );
                if ui.small_button("All").on_hover_text("Show all line types").clicked() {
                    vis_state.show_outer_walls = true;
                    vis_state.show_inner_walls = true;
                    vis_state.show_bridges = true;
                    vis_state.show_wave_overhangs = true;
                    vis_state.show_solid_infill = true;
                    vis_state.show_sparse_infill = true;
                    vis_state.show_gap_fill = true;
                    vis_state.show_supports = true;
                    vis_state.show_support_interface = true;
                    vis_state.show_purge_tower = true;
                    vis_state.show_scarf_seams = true;
                    vis_state.show_wipes = true;
                    vis_state.show_travel_moves = true;
                    vis_state.show_z_hops = true;
                    vis_state.dirty_toolpaths_mesh = true;
                }
                if ui
                    .small_button("Extrude")
                    .on_hover_text("Show extruding line types only (hide Travel & Z-Hop)")
                    .clicked()
                {
                    vis_state.show_outer_walls = true;
                    vis_state.show_inner_walls = true;
                    vis_state.show_bridges = true;
                    vis_state.show_wave_overhangs = true;
                    vis_state.show_solid_infill = true;
                    vis_state.show_sparse_infill = true;
                    vis_state.show_gap_fill = true;
                    vis_state.show_supports = true;
                    vis_state.show_support_interface = true;
                    vis_state.show_purge_tower = true;
                    vis_state.show_scarf_seams = true;
                    vis_state.show_wipes = true;
                    vis_state.show_travel_moves = false;
                    vis_state.show_z_hops = false;
                    vis_state.dirty_toolpaths_mesh = true;
                }
                if ui.small_button("None").on_hover_text("Hide all line types").clicked() {
                    vis_state.show_outer_walls = false;
                    vis_state.show_inner_walls = false;
                    vis_state.show_bridges = false;
                    vis_state.show_wave_overhangs = false;
                    vis_state.show_solid_infill = false;
                    vis_state.show_sparse_infill = false;
                    vis_state.show_gap_fill = false;
                    vis_state.show_supports = false;
                    vis_state.show_support_interface = false;
                    vis_state.show_purge_tower = false;
                    vis_state.show_scarf_seams = false;
                    vis_state.show_wipes = false;
                    vis_state.show_travel_moves = false;
                    vis_state.show_z_hops = false;
                    vis_state.dirty_toolpaths_mesh = true;
                }
            });

            ui.add_space(2.0);
            let mut mesh_dirty = false;
            let c_outer = counts.map(|c| c.1);
            let c_inner = counts.map(|c| c.2);
            let c_bridge = counts.map(|c| c.3);
            let c_wave = counts.map(|c| c.4);
            let c_solid = counts.map(|c| c.5);
            let c_sparse = counts.map(|c| c.6);
            let c_sup = counts.map(|c| c.7);
            let c_sup_intf = counts.map(|c| c.8);

            let col_w = ((ui.available_width() - 12.0) * 0.5).max(82.0);
            egui::Grid::new("combined_line_type_legend_grid")
                .num_columns(2)
                .min_col_width(col_w)
                .max_col_width(col_w)
                .spacing([8.0, 3.0])
                .striped(true)
                .show(ui, |ui| {
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_outer_walls,
                        [0.98, 0.44, 0.16, 1.0],
                        "Outer Wall",
                        c_outer,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_inner_walls,
                        [0.24, 0.90, 0.42, 1.0],
                        "Inner Wall",
                        c_inner,
                    );
                    ui.end_row();

                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_bridges,
                        [0.18, 0.52, 0.98, 1.0],
                        "Bridge",
                        c_bridge,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_wave_overhangs,
                        [0.98, 0.26, 0.68, 1.0],
                        "Wave Overhang",
                        c_wave,
                    );
                    ui.end_row();

                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_solid_infill,
                        [0.22, 0.76, 0.98, 1.0],
                        "Solid Infill",
                        c_solid,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_sparse_infill,
                        [0.98, 0.78, 0.18, 1.0],
                        "Sparse Infill",
                        c_sparse,
                    );
                    ui.end_row();

                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_supports,
                        [0.22, 0.88, 0.62, 1.0],
                        "Support Base",
                        c_sup,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_support_interface,
                        [0.12, 0.96, 0.88, 1.0],
                        "Support Intf",
                        c_sup_intf,
                    );
                    ui.end_row();

                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_gap_fill,
                        [0.98, 0.32, 0.88, 1.0],
                        "Gap Fill",
                        None,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_scarf_seams,
                        [0.76, 0.44, 0.98, 1.0],
                        "Scarf Seam",
                        None,
                    );
                    ui.end_row();

                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_wipes,
                        [0.68, 0.76, 0.86, 1.0],
                        "Wipe",
                        None,
                    );
                    mesh_dirty |= draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_purge_tower,
                        [0.96, 0.44, 0.64, 1.0],
                        "Purge Tower",
                        None,
                    );
                    ui.end_row();

                    let _ = draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_travel_moves,
                        [0.28, 0.86, 0.98, 1.0],
                        "Travel",
                        None,
                    );
                    let _ = draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_z_hops,
                        [0.98, 0.92, 0.22, 1.0],
                        "Z-Hop",
                        None,
                    );
                    ui.end_row();

                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.checkbox(&mut vis_state.show_toolpath_centerlines, "");
                        let lbl_resp = ui.add(
                            egui::Label::new(
                                egui::RichText::new("Centerlines")
                                    .size(11.5)
                                    .color(if vis_state.show_toolpath_centerlines {
                                        egui::Color32::from_gray(235)
                                    } else {
                                        egui::Color32::from_gray(135)
                                    }),
                            )
                            .sense(egui::Sense::click()),
                        );
                        if lbl_resp.clicked() {
                            vis_state.show_toolpath_centerlines =
                                !vis_state.show_toolpath_centerlines;
                        }
                    });
                    ui.end_row();
                });

            if mesh_dirty {
                vis_state.dirty_toolpaths_mesh = true;
            }
        });
}

/// Renders separate scrub controls for slicing layers (isosurfaces) and within-layer moves,
/// plus the toggle & live readout for tracking the 3D Nozzle Frame, Tool Clearance Profile,
/// and Bed-Coordinate Gravity Vector along the scrubbed trajectory.
fn render_layer_and_move_scrub_controls(
    ui: &mut egui::Ui,
    vis_state: &mut FieldVisualizationState,
    config: &mut SlicerConfig,
    obj_world_tf: Transform,
    show_within_layer_scrub: bool,
    id_prefix: &str,
) {
    let total_layers = vis_state.total_layer_count();
    if total_layers == 0 {
        return;
    }
    let max_idx = total_layers - 1;
    if vis_state.selected_layer_min_idx > max_idx {
        vis_state.selected_layer_min_idx = max_idx;
    }
    if vis_state.selected_layer_max_idx > max_idx {
        vis_state.selected_layer_max_idx = max_idx;
    }
    if vis_state.single_layer_scrub_mode {
        vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
    } else if vis_state.selected_layer_min_idx > vis_state.selected_layer_max_idx {
        vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
    }

    ui.push_id(id_prefix, |ui| {
        let mut layer_changed = false;

        // 1. Layer (Isosurface) Scrub Header & Step Buttons
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Layer Scrub ({}/{}):",
                    vis_state.selected_layer_max_idx + 1,
                    total_layers
                ))
                .strong(),
            );
            if ui
                .small_button("|<")
                .on_hover_text("Jump to first layer (isosurface #0)")
                .clicked()
                && vis_state.selected_layer_max_idx != 0
            {
                vis_state.selected_layer_max_idx = 0;
                vis_state.selected_layer_min_idx = 0;
                layer_changed = true;
            }
            if ui
                .small_button("<")
                .on_hover_text("Previous layer (isosurface)")
                .clicked()
                && vis_state.selected_layer_max_idx > 0
            {
                vis_state.selected_layer_max_idx -= 1;
                if vis_state.single_layer_scrub_mode
                    || vis_state.selected_layer_min_idx > vis_state.selected_layer_max_idx
                {
                    vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                }
                layer_changed = true;
            }
            if ui
                .small_button(">")
                .on_hover_text("Next layer (isosurface)")
                .clicked()
                && vis_state.selected_layer_max_idx < max_idx
            {
                vis_state.selected_layer_max_idx += 1;
                if vis_state.single_layer_scrub_mode {
                    vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                }
                layer_changed = true;
            }
            if ui
                .small_button(">|")
                .on_hover_text("Show all layers up to the top layer")
                .clicked()
                && (vis_state.selected_layer_max_idx != max_idx
                    || (!vis_state.single_layer_scrub_mode && vis_state.selected_layer_min_idx != 0))
            {
                vis_state.selected_layer_max_idx = max_idx;
                if vis_state.single_layer_scrub_mode {
                    vis_state.selected_layer_min_idx = max_idx;
                } else {
                    vis_state.selected_layer_min_idx = 0;
                }
                layer_changed = true;
            }
            if ui
                .checkbox(&mut vis_state.single_layer_scrub_mode, "Single Layer")
                .on_hover_text(
                    "Lock Start Layer = End Layer to scrub a single non-planar isosurface / toolpath layer at a time",
                )
                .changed()
            {
                if vis_state.single_layer_scrub_mode {
                    vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                } else {
                    vis_state.selected_layer_min_idx = 0;
                }
                layer_changed = true;
            }
        });

        if vis_state.single_layer_scrub_mode {
            if ui
                .add(
                    egui::Slider::new(&mut vis_state.selected_layer_max_idx, 0..=max_idx)
                        .text("Active Layer"),
                )
                .changed()
            {
                vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                layer_changed = true;
            }
        } else {
            let max_changed = ui
                .add(
                    egui::Slider::new(&mut vis_state.selected_layer_max_idx, 0..=max_idx)
                        .text("End Layer (Top)"),
                )
                .changed();
            let min_changed = ui
                .add(
                    egui::Slider::new(&mut vis_state.selected_layer_min_idx, 0..=max_idx)
                        .text("Start Layer"),
                )
                .changed();
            if min_changed && vis_state.selected_layer_min_idx > vis_state.selected_layer_max_idx {
                vis_state.selected_layer_max_idx = vis_state.selected_layer_min_idx;
            } else if max_changed
                && vis_state.selected_layer_max_idx < vis_state.selected_layer_min_idx
            {
                vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
            }
            layer_changed |= min_changed || max_changed;

            if ui
                .add(
                    egui::Slider::new(&mut vis_state.selected_layer_stride, 1..=10)
                        .text("Display Stride"),
                )
                .changed()
            {
                vis_state.dirty_selected_layers_mesh = true;
                vis_state.dirty_toolpaths_mesh = true;
                vis_state.dirty_slice = true;
            }
        }

        if layer_changed {
            vis_state.layer_move_scrub_limit = None;
            vis_state.dirty_selected_layers_mesh = true;
            vis_state.dirty_toolpaths_mesh = true;
            vis_state.dirty_slice = true;
        }

        // 2. Within-Layer Move Scrubbing (along the active top layer `selected_layer_max_idx`)
        if show_within_layer_scrub {
            let total_moves = vis_state.active_layer_total_moves();
            if total_moves > 0 {
                let mut curr_move = vis_state
                    .layer_move_scrub_limit
                    .unwrap_or(total_moves)
                    .clamp(1, total_moves);
                let pct = (curr_move as f32 / total_moves as f32) * 100.0;
                let mut move_changed = false;

                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "Within-Layer Move ({curr_move}/{total_moves}, {pct:.0}%):"
                        ))
                        .strong(),
                    );
                    if ui
                        .small_button("|<")
                        .on_hover_text("Jump to first move on the active layer")
                        .clicked()
                    {
                        curr_move = 1;
                        move_changed = true;
                    }
                    if ui
                        .small_button("<")
                        .on_hover_text("Step back 1 move (crosses to previous layer at move 1)")
                        .clicked()
                    {
                        if curr_move > 1 {
                            curr_move -= 1;
                            move_changed = true;
                        } else if vis_state.selected_layer_max_idx > 0 {
                            vis_state.selected_layer_max_idx -= 1;
                            if vis_state.single_layer_scrub_mode
                                || vis_state.selected_layer_min_idx
                                    > vis_state.selected_layer_max_idx
                            {
                                vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                            }
                            vis_state.layer_move_scrub_limit = None;
                            vis_state.dirty_selected_layers_mesh = true;
                            vis_state.dirty_toolpaths_mesh = true;
                            vis_state.dirty_slice = true;
                        }
                    }
                    if ui
                        .small_button(">")
                        .on_hover_text("Step forward 1 move (crosses to next layer at layer end)")
                        .clicked()
                    {
                        if curr_move < total_moves {
                            curr_move += 1;
                            move_changed = true;
                        } else if vis_state.selected_layer_max_idx < max_idx {
                            vis_state.selected_layer_max_idx += 1;
                            if vis_state.single_layer_scrub_mode {
                                vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                            }
                            vis_state.layer_move_scrub_limit = Some(1);
                            vis_state.dirty_selected_layers_mesh = true;
                            vis_state.dirty_toolpaths_mesh = true;
                            vis_state.dirty_slice = true;
                        }
                    }
                    if ui
                        .small_button(">|")
                        .on_hover_text("Show 100% of moves on the active layer")
                        .clicked()
                    {
                        curr_move = total_moves;
                        move_changed = true;
                    }
                });

                if ui
                    .add(
                        egui::Slider::new(&mut curr_move, 1..=total_moves)
                            .text("Move Index"),
                    )
                    .changed()
                {
                    move_changed = true;
                }

                if move_changed {
                    if curr_move >= total_moves {
                        vis_state.layer_move_scrub_limit = None;
                    } else {
                        vis_state.layer_move_scrub_limit = Some(curr_move);
                    }
                    vis_state.dirty_toolpaths_mesh = true;
                }
            }
        }

        // 3. Tool Clearance, Nozzle Frame & Gravity Tracking Toggle + Live Pose Readout
        ui.checkbox(
            &mut config.machine.show_nozzle_clearance_gizmo,
            "Show Tool Clearance, Nozzle Frame & Gravity at Scrub Point",
        )
        .on_hover_text(
            "Draw the active tool's 3D clearance profile, orthonormal Nozzle Frame (X_travel, Y_lateral, Z_nozzle), and Bed-Coordinate Gravity Vector tracking the active layer & within-layer scrub position",
        );

        if config.machine.show_nozzle_clearance_gizmo || vis_state.layer_move_scrub_limit.is_some()
        {
            if let Some(pose) = vis_state.compute_active_scrub_pose(config, obj_world_tf) {
                let speed_info = if pose.speed_mms > 0.0 {
                    format!(
                        " | v={:.0} mm/s | W={:.2} H={:.2} mm",
                        pose.speed_mms, pose.width_mm, pose.height_mm
                    )
                } else {
                    String::new()
                };
                let norm_source = if pose.uses_surface_normal {
                    "SurfNorm"
                } else {
                    "IsoNorm"
                };
                let (rot_l1, rot_l2) = config.machine.kinematics_mode.rotary_axis_letters();
                ui.small(
                    egui::RichText::new(format!(
                        "L#{} M{}/{} [{}] T{} | Pos=({:+.1},{:+.1},{:+.1}) | {}={:+.1}° {}={:+.1}° | θ_iso={:.1}° ({}) | φ_sdf={:+.3} mm | g_bed=({:+.2},{:+.2},{:+.2}){}",
                        pose.layer_index,
                        pose.move_index,
                        pose.total_layer_moves,
                        pose.move_label,
                        pose.tool_id,
                        pose.tip_bed.x,
                        pose.tip_bed.y,
                        pose.tip_bed.z,
                        rot_l1,
                        pose.b_deg,
                        rot_l2,
                        pose.c_deg,
                        pose.tool_iso_angle_deg,
                        norm_source,
                        pose.sdf_mm,
                        pose.gravity_bed.x,
                        pose.gravity_bed.y,
                        pose.gravity_bed.z,
                        speed_info,
                    ))
                    .monospace()
                    .color(egui::Color32::from_rgb(195, 220, 245)),
                );
            }
        }
    });
}

/// Renders the extrusion geometry quality (`Fast` vs `HighQualityPbr`), physical filament TD
/// translucency, lithophane backlight, and live alpha X-ray opacity controls.
fn render_toolpath_quality_and_transparency_controls(
    ui: &mut egui::Ui,
    vis_state: &mut FieldVisualizationState,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Extrusion Render:");
        for q_mode in [
            ToolpathQualityMode::Fast,
            ToolpathQualityMode::HighQualityPbr,
        ] {
            let selected = vis_state.toolpath_quality_mode == q_mode;
            if render_delineated_pill_tab(ui, selected, q_mode.short_label())
                .on_hover_text(match q_mode {
                    ToolpathQualityMode::Fast => {
                        "Fast 4-sided bead prisms merged into a single mesh for maximum interactive FPS."
                    }
                    ToolpathQualityMode::HighQualityPbr => {
                        "High-quality 12-vertex pressed-stadium (capsule) beads with continuous mitered joints, bullnose endcaps, per-filament Bevy PBR polymer sheen, TD-driven translucency, and soft cascaded shadows."
                    }
                })
                .clicked()
                && !selected
            {
                vis_state.toolpath_quality_mode = q_mode;
                vis_state.dirty_toolpaths_mesh = true;
            }
        }
    });

    if vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr {
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(
                &mut vis_state.enable_pbr_translucency,
                "TD-Based Filament Opacity",
            )
            .on_hover_text(
                "Set each filament's bead opacity directly from its Kubelka–Munk Transmission Distance (TD) and physical bead thickness (high-TD filaments are translucent; low-TD filaments are dense).",
            );
            ui.checkbox(
                &mut vis_state.lithophane_backlight_preview,
                "Lithophane Backlight",
            )
            .on_hover_text(
                "Boost rear backlight illumination behind the print to preview backlit lithophane glow through translucent extrusions.",
            );
            ui.checkbox(&mut vis_state.enable_hq_shadows, "Soft Shadows")
                .on_hover_text(
                    "Enable millimeter-calibrated cascaded directional shadows on the 3D studio key light.",
                );
        });
    }

    ui.horizontal(|ui| {
        let slider_lbl = if vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr
            && vis_state.enable_pbr_translucency
        {
            "Opacity Scale (× TD)"
        } else {
            "Extrusion Opacity"
        };
        let max_op = if vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr
            && vis_state.enable_pbr_translucency
        {
            1.50
        } else {
            1.00
        };
        if vis_state.toolpath_opacity > max_op {
            vis_state.toolpath_opacity = max_op;
        }
        ui.add(egui::Slider::new(&mut vis_state.toolpath_opacity, 0.10..=max_op).text(slider_lbl))
            .on_hover_text(
                "At 1.00 in High-Quality PBR mode, bead opacity matches each tool's physical Transmission Distance (TD). Lower below 1.00 to X-ray inside the core, or raise toward 1.50 to increase density.",
            );
        if (vis_state.toolpath_opacity - 1.0).abs() > 1.0e-3 && ui.small_button("1.0×").clicked() {
            vis_state.toolpath_opacity = 1.0;
        }
    });
}

/// Renders the active toolpath legend (either continuous 5-tick gradient bar or categorical swatches).
fn render_toolpath_color_legend(
    ui: &mut egui::Ui,
    vis_state: &FieldVisualizationState,
    config: &SlicerConfig,
) {
    if vis_state.toolpath_color_mode == ToolpathColorMode::ByFeatureType {
        draw_categorical_swatch_legend(
            ui,
            "Toolpath Feature Legend",
            &[
                ("Outer Wall", [0.98, 0.44, 0.16, 1.0]),
                ("Inner Wall", [0.24, 0.90, 0.42, 1.0]),
                ("Bridge", [0.18, 0.52, 0.98, 1.0]),
                ("Wave Overhang", [0.98, 0.26, 0.68, 1.0]),
                ("Solid Infill", [0.22, 0.76, 0.98, 1.0]),
                ("Sparse Infill", [0.98, 0.78, 0.18, 1.0]),
                ("Support Base", [0.22, 0.88, 0.62, 1.0]),
                ("Support Interface", [0.12, 0.96, 0.88, 1.0]),
                ("Gap Fill", [0.98, 0.32, 0.88, 1.0]),
                ("Scarf Seam", [0.76, 0.44, 0.98, 1.0]),
                ("Wipe", [0.68, 0.76, 0.86, 1.0]),
                ("Travel", [0.28, 0.86, 0.98, 1.0]),
                ("Z-Hop", [0.98, 0.92, 0.22, 1.0]),
            ],
        );
        return;
    }

    if vis_state.toolpath_color_mode == ToolpathColorMode::ByToolOrFilament {
        let mut owned_items: Vec<(String, [f32; 4])> = Vec::new();
        if config.slicing.color_mixing.enabled && config.tools.len() >= 2 {
            for tool in &config.tools {
                let td_mm = tool.effective_transmission_distance_mm();
                let label = format!(
                    "T{} ({}, {}, TD {:.1} mm)",
                    tool.id, tool.name, tool.filament_color_hex, td_mm
                );
                owned_items.push((label, tool.filament_color_srgba));
            }
        } else if !config.filament_slots.is_empty() {
            for slot in &config.filament_slots {
                let tool_id = config.tool_id_for_filament_slot(slot.slot_index as u8);
                let label = format!(
                    "Slot {} ({}, {}) → T{}",
                    slot.slot_index as usize + 1,
                    slot.name,
                    slot.color_hex,
                    tool_id
                );
                owned_items.push((label, slot.color_srgba));
            }
        } else {
            for (idx, tool) in config.tools.iter().enumerate() {
                owned_items.push((
                    format!(
                        "Slot {} → T{} ({}, {})",
                        idx + 1,
                        tool.id,
                        tool.name,
                        tool.filament_color_hex
                    ),
                    tool.filament_color_srgba,
                ));
            }
        }
        owned_items.push(("Support".to_string(), [0.22, 0.88, 0.62, 1.0]));
        owned_items.push(("Travel".to_string(), [0.28, 0.86, 0.98, 1.0]));
        owned_items.push(("Z-Hop".to_string(), [0.98, 0.92, 0.22, 1.0]));
        let borrowed: Vec<(&str, [f32; 4])> = owned_items
            .iter()
            .map(|(s, c)| (s.as_str(), *c))
            .collect();
        draw_categorical_swatch_legend(ui, "Tool / Filament Colour Legend", &borrowed);
        return;
    }

    if vis_state.toolpath_color_mode == ToolpathColorMode::ByOpticalComposite {
        let mut owned_items: Vec<(String, [f32; 4])> = Vec::new();
        for tool in &config.tools {
            let td_mm = tool.effective_transmission_distance_mm();
            owned_items.push((
                format!(
                    "T{} ({}, {}, TD {:.1} mm)",
                    tool.id, tool.name, tool.filament_color_hex, td_mm
                ),
                tool.filament_color_srgba,
            ));
        }
        owned_items.push(("Support".to_string(), [0.22, 0.88, 0.62, 1.0]));
        let borrowed: Vec<(&str, [f32; 4])> = owned_items
            .iter()
            .map(|(s, c)| (s.as_str(), *c))
            .collect();
        let title = format!(
            "Simulated Optical Stack ({})",
            config.slicing.color_mixing.lighting_mode.label()
        );
        draw_categorical_swatch_legend(ui, &title, &borrowed);
        return;
    }

    if vis_state.toolpath_color_mode == ToolpathColorMode::BySurfaceNormalUsage {
        draw_categorical_swatch_legend(
            ui,
            "Tool Orientation Source",
            &[
                ("Object Surface Normal", [0.14, 0.92, 0.46, 1.0]),
                ("Isosurface Normal (Outer Fallback)", [0.98, 0.62, 0.16, 1.0]),
                ("Isosurface Normal (Interior / Other)", [0.24, 0.52, 0.92, 1.0]),
                ("Travel", [0.28, 0.86, 0.98, 1.0]),
                ("Z-Hop", [0.98, 0.92, 0.22, 1.0]),
            ],
        );
        return;
    }

    if vis_state.toolpath_color_mode == ToolpathColorMode::ByNozzleSdf {
        let h_ref_mm = nozzle_sdf_reference_layer_height_mm(config);
        let observed_range = if let Some(ref planned) = vis_state.cached_planned_gcode {
            if planned.min_sdf_mm.is_finite()
                && planned.max_sdf_mm.is_finite()
                && planned.min_sdf_mm <= planned.max_sdf_mm
            {
                Some((planned.min_sdf_mm, planned.max_sdf_mm))
            } else {
                None
            }
        } else if let Some(ref tp_res) = vis_state.cached_toolpaths {
            raw_toolpaths_sdf_observed_range(tp_res)
        } else {
            None
        };
        draw_nozzle_sdf_legend(ui, h_ref_mm, observed_range);
        return;
    }

    if let Some(ref planned) = vis_state.cached_planned_gcode {
        let (title, min_v, max_v, unit, decimals) = match vis_state.toolpath_color_mode {
            ToolpathColorMode::ByLineWidth => (
                "Line Width",
                planned.min_width_mm,
                planned.max_width_mm,
                "mm",
                3,
            ),
            ToolpathColorMode::ByLayerHeight => (
                "Bead Height",
                planned.min_height_mm,
                planned.max_height_mm,
                "mm",
                3,
            ),
            ToolpathColorMode::BySpeed => (
                "Feedrate Speed",
                planned.min_speed_mms,
                planned.max_speed_mms,
                "mm/s",
                1,
            ),
            ToolpathColorMode::ByAcceleration => (
                "Segment Acceleration",
                planned.min_accel_mms2,
                planned.max_accel_mms2,
                "mm/s²",
                0,
            ),
            ToolpathColorMode::ByVolumetricFlow => (
                "Volumetric Flow Rate",
                planned.min_flow_mm3s,
                planned.max_flow_mm3s,
                "mm³/s",
                2,
            ),
            ToolpathColorMode::ByExtrusionMultiplier => (
                "Effective Flow Multiplier",
                planned.min_flow_mult,
                planned.max_flow_mult,
                "×",
                3,
            ),
            ToolpathColorMode::ByPressureAdvance => (
                "Dynamic Pressure Advance",
                planned.min_pa_s,
                planned.max_pa_s,
                "s",
                4,
            ),
            ToolpathColorMode::ByMeltZonePressure => (
                "Melt-Zone Pressure",
                planned.min_pressure_mm3s,
                planned.max_pressure_mm3s,
                "mm³/s eq",
                2,
            ),
            ToolpathColorMode::ByToolIsoNormalAngle => (
                "Tool vs. Isosurface Normal Angle",
                planned.min_tool_iso_angle_deg,
                planned
                    .max_tool_iso_angle_deg
                    .max(planned.min_tool_iso_angle_deg + 5.0),
                "°",
                1,
            ),
            ToolpathColorMode::ByFeatureType
            | ToolpathColorMode::ByToolOrFilament
            | ToolpathColorMode::ByOpticalComposite
            | ToolpathColorMode::BySurfaceNormalUsage
            | ToolpathColorMode::ByNozzleSdf => {
                unreachable!()
            }
        };
        draw_continuous_gradient_legend(
            ui,
            title,
            min_v,
            max_v,
            unit,
            decimals,
            high_contrast_parameter_ramp,
        );
    } else if vis_state.toolpath_color_mode == ToolpathColorMode::ByToolIsoNormalAngle {
        let max_angle_deg = config
            .active_tool()
            .map(|t| t.max_angle_from_bed_normal_deg)
            .unwrap_or(45.0)
            .max(15.0);
        draw_continuous_gradient_legend(
            ui,
            "Tool vs. Isosurface Normal Angle",
            0.0,
            max_angle_deg,
            "°",
            1,
            high_contrast_parameter_ramp,
        );
    } else {
        // Fallback when viewing un-planned raw shell toolpaths by line width
        let spec = config.slicing.outer_wall_width;
        draw_continuous_gradient_legend(
            ui,
            "Line Width",
            spec.min_mm,
            spec.max_mm,
            "mm",
            3,
            high_contrast_parameter_ramp,
        );
    }
}

/// Renders the active selected-layer isosurface legend.
fn render_selected_layers_color_legend(
    ui: &mut egui::Ui,
    vis_state: &FieldVisualizationState,
    config: &SlicerConfig,
) {
    match vis_state.selected_layers_color_mode {
        SelectedLayersColorMode::ByLayerThickness => {
            let lh = config.slicing.layer_height;
            draw_continuous_gradient_legend(
                ui,
                "Layer Thickness Δh",
                lh.min_mm,
                lh.max_mm,
                "mm",
                3,
                high_contrast_parameter_ramp,
            );
            draw_categorical_swatch_legend(
                ui,
                "Boundary Layer Overrides",
                &[
                    ("First Layer (Bed)", [0.18, 0.92, 0.96, 1.0]),
                    ("Top Layer (Branch Cap)", [0.96, 0.42, 0.92, 1.0]),
                ],
            );
        }
        SelectedLayersColorMode::ByBranch => {
            let branch_count = vis_state
                .cached_selected_layers
                .as_ref()
                .map(|r| (r.branch_count as usize).clamp(1, 8))
                .unwrap_or(4);
            let labels = [
                "Branch 0", "Branch 1", "Branch 2", "Branch 3", "Branch 4", "Branch 5", "Branch 6",
                "Branch 7",
            ];
            let mut items = Vec::with_capacity(branch_count);
            for i in 0..branch_count {
                let c = BRANCH_PALETTE[i % BRANCH_PALETTE.len()];
                items.push((labels[i], [c[0], c[1], c[2], 1.0]));
            }
            draw_categorical_swatch_legend(ui, "Topological Branch ID", &items);
        }
        SelectedLayersColorMode::AlternatingBands => {
            draw_categorical_swatch_legend(
                ui,
                "Alternating Layer Bands",
                &[
                    ("Even Layer (#2k)", [0.20, 0.84, 0.76, 1.0]),
                    ("Odd Layer (#2k+1)", [0.96, 0.66, 0.22, 1.0]),
                ],
            );
        }
    }
}

/// Renders a compact `?` help button for a configuration parameter backed by [`crate::config::PARAMETER_DOCS`].
/// Hovering or clicking the `?` button opens a rich structured popup with the parameter's
/// title, category, default value, units, summary, and detailed physical/geometric explanation.
fn param_help_button(ui: &mut egui::Ui, key: &str) {
    let Some(doc) = lookup_parameter_doc(key) else {
        return;
    };
    let popup_id = ui.make_persistent_id(("param_help_popup", key));
    let btn = ui.add(
        egui::Button::new(
            egui::RichText::new("?")
                .small()
                .strong()
                .color(egui::Color32::from_rgb(170, 215, 255)),
        )
        .small()
        .frame(true),
    );
    if btn.clicked() {
        ui.memory_mut(|mem| mem.toggle_popup(popup_id));
    }

    let render_card = |ui: &mut egui::Ui| {
        ui.set_max_width(340.0);
        ui.label(
            egui::RichText::new(doc.title)
                .strong()
                .color(egui::Color32::from_rgb(245, 248, 255)),
        );
        let meta = if doc.units.is_empty() {
            format!("{}  •  Default: {}", doc.category, doc.default_value)
        } else {
            format!(
                "{}  •  Default: {} ({})",
                doc.category, doc.default_value, doc.units
            )
        };
        ui.small(
            egui::RichText::new(meta)
                .italics()
                .color(egui::Color32::from_rgb(150, 195, 235)),
        );
        ui.separator();
        ui.label(egui::RichText::new(doc.summary).strong());
        ui.add_space(3.0);
        ui.label(doc.details);
        ui.add_space(2.0);
        ui.small(
            egui::RichText::new(format!("Config key: {}", doc.key))
                .monospace()
                .color(egui::Color32::GRAY),
        );
    };

    egui::popup::popup_below_widget(
        ui,
        popup_id,
        &btn,
        egui::PopupCloseBehavior::CloseOnClickOutside,
        render_card,
    );
    btn.on_hover_ui(render_card);
}

/// Renders a delineated horizontal folder-style tab bar spanning the full available width.
///
/// Each entry in `tabs` is `(label, has_modified_indicator)`.
/// - Inactive tabs are drawn with a recessed dark background, a 3-sided border, and a continuous
///   bottom baseline stroke so every tab is clearly separated.
/// - The active tab is drawn with a raised surface background, a 3px `active_accent` top highlight
///   bar, bold white text, and an open bottom edge that visually connects to the panel below.
///
/// Returns `Some(clicked_index)` if the user clicked a tab.
fn render_folder_tab_bar(
    ui: &mut egui::Ui,
    id_salt: &str,
    tabs: &[(&str, bool)],
    active_index: usize,
    height: f32,
    active_accent: egui::Color32,
) -> Option<usize> {
    render_folder_tab_bar_with_collapse(
        ui,
        id_salt,
        tabs,
        active_index,
        height,
        active_accent,
        false,
    )
    .0
}

fn render_folder_tab_bar_with_collapse(
    ui: &mut egui::Ui,
    id_salt: &str,
    tabs: &[(&str, bool)],
    active_index: usize,
    height: f32,
    active_accent: egui::Color32,
    show_collapse_btn: bool,
) -> (Option<usize>, bool) {
    if tabs.is_empty() {
        return (None, false);
    }

    let count = tabs.len();
    let gap = 3.0_f32;
    let total_width = ui.available_width().max(120.0);
    let collapse_reserve = if show_collapse_btn { 26.0_f32 } else { 0.0_f32 };
    let tabs_area_width = (total_width - collapse_reserve).max(80.0);
    let total_gaps = gap * (count.saturating_sub(1) as f32);
    let tab_width = ((tabs_area_width - total_gaps) / (count as f32)).max(28.0);

    let (bar_rect, _) =
        ui.allocate_exact_size(egui::vec2(total_width, height), egui::Sense::hover());

    let mut clicked_tab = None;
    let mut collapse_clicked = false;
    let baseline_y = bar_rect.max.y - 1.0;
    let baseline_stroke = egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(86, 96, 116));
    let tabs_max_x = bar_rect.min.x + tabs_area_width;

    // Pre-collect interaction responses before borrowing the painter
    let mut tab_rects_and_states = Vec::with_capacity(count);
    for idx in 0..count {
        let min_x = bar_rect.min.x + (idx as f32) * (tab_width + gap);
        let max_x = if idx + 1 == count {
            tabs_max_x
        } else {
            (min_x + tab_width).min(tabs_max_x)
        };
        let tab_rect = egui::Rect::from_min_max(
            egui::pos2(min_x, bar_rect.min.y),
            egui::pos2(max_x, baseline_y),
        );
        let tab_id = ui.id().with(id_salt).with(idx);
        let resp = ui.interact(tab_rect, tab_id, egui::Sense::click());
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if resp.clicked() {
            clicked_tab = Some(idx);
        }
        tab_rects_and_states.push((tab_rect, resp.hovered()));
    }

    let collapse_btn_state = if show_collapse_btn {
        let btn_rect = egui::Rect::from_min_max(
            egui::pos2(tabs_max_x + 3.0, bar_rect.min.y + 2.0),
            egui::pos2(bar_rect.max.x, baseline_y - 2.0),
        );
        let btn_id = ui.id().with(id_salt).with("collapse_btn");
        let resp = ui
            .interact(btn_rect, btn_id, egui::Sense::click())
            .on_hover_text("Collapse left Setup & Configuration panel");
        if resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if resp.clicked() {
            collapse_clicked = true;
        }
        Some((btn_rect, resp.hovered()))
    } else {
        None
    };

    let painter = ui.painter();

    // Draw continuous bottom baseline across the entire tab strip first
    painter.line_segment(
        [
            egui::pos2(bar_rect.min.x, baseline_y),
            egui::pos2(bar_rect.max.x, baseline_y),
        ],
        baseline_stroke,
    );

    if let Some((btn_rect, btn_hovered)) = collapse_btn_state {
        let btn_fill = if btn_hovered {
            egui::Color32::from_rgb(48, 58, 76)
        } else {
            egui::Color32::from_rgb(28, 33, 42)
        };
        painter.rect_filled(btn_rect, egui::Rounding::same(4.0), btn_fill);
        painter.rect_stroke(
            btn_rect,
            egui::Rounding::same(4.0),
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(76, 86, 106)),
        );
        let arrow_col = if btn_hovered {
            egui::Color32::WHITE
        } else {
            egui::Color32::from_rgb(190, 200, 218)
        };
        let galley = painter.layout_no_wrap(
            "◀".to_string(),
            egui::FontId::proportional(11.5),
            arrow_col,
        );
        let pos = egui::pos2(
            btn_rect.center().x - galley.size().x * 0.5,
            btn_rect.center().y - galley.size().y * 0.5,
        );
        painter.galley(pos, galley, arrow_col);
    }

    for (idx, (&(label, is_modified), &(tab_rect, hovered))) in
        tabs.iter().zip(tab_rects_and_states.iter()).enumerate()
    {
        let is_active = idx == active_index;

        let bg_fill = if is_active {
            egui::Color32::from_rgb(48, 56, 72)
        } else if hovered {
            egui::Color32::from_rgb(36, 42, 54)
        } else {
            egui::Color32::from_rgb(23, 26, 33)
        };

        // Tab body fill (active tab extends 1px over the baseline to leave its bottom open)
        let fill_rect = if is_active {
            egui::Rect::from_min_max(tab_rect.min, egui::pos2(tab_rect.max.x, baseline_y + 1.0))
        } else {
            egui::Rect::from_min_max(tab_rect.min, egui::pos2(tab_rect.max.x, baseline_y - 0.5))
        };
        painter.rect_filled(
            fill_rect,
            egui::Rounding {
                nw: 6.0,
                ne: 6.0,
                sw: 0.0,
                se: 0.0,
            },
            bg_fill,
        );

        // Top accent bar on the active tab
        if is_active {
            let accent_rect = egui::Rect::from_min_max(
                egui::pos2(tab_rect.min.x + 1.0, tab_rect.min.y),
                egui::pos2(tab_rect.max.x - 1.0, tab_rect.min.y + 3.0),
            );
            painter.rect_filled(
                accent_rect,
                egui::Rounding {
                    nw: 5.0,
                    ne: 5.0,
                    sw: 0.0,
                    se: 0.0,
                },
                active_accent,
            );
        }

        // 3-sided tab outline (left, chamfered top corners, top, right)
        let side_stroke = if is_active {
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(104, 118, 144))
        } else if hovered {
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(92, 104, 126))
        } else {
            egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(64, 72, 88))
        };
        let top_stroke = if is_active {
            egui::Stroke::new(1.5_f32, active_accent)
        } else {
            side_stroke
        };
        let corner = 4.0_f32;
        let p_bl = egui::pos2(tab_rect.min.x, baseline_y);
        let p_tl_low = egui::pos2(tab_rect.min.x, tab_rect.min.y + corner);
        let p_tl_high = egui::pos2(tab_rect.min.x + corner, tab_rect.min.y);
        let p_tr_high = egui::pos2(tab_rect.max.x - corner, tab_rect.min.y);
        let p_tr_low = egui::pos2(tab_rect.max.x, tab_rect.min.y + corner);
        let p_br = egui::pos2(tab_rect.max.x, baseline_y);

        painter.line_segment([p_bl, p_tl_low], side_stroke);
        painter.line_segment([p_tl_low, p_tl_high], top_stroke);
        painter.line_segment([p_tl_high, p_tr_high], top_stroke);
        painter.line_segment([p_tr_high, p_tr_low], top_stroke);
        painter.line_segment([p_tr_low, p_br], side_stroke);

        // Label + optional modified amber dot (clipped to tab bounds for narrow panels)
        let compact_tab = tab_width < 135.0;
        let font_id = if is_active {
            egui::FontId::proportional(if compact_tab { 12.2 } else { 13.5 })
        } else {
            egui::FontId::proportional(if compact_tab { 11.6 } else { 12.8 })
        };
        let text_color = if is_active {
            egui::Color32::WHITE
        } else if hovered {
            egui::Color32::from_rgb(232, 238, 248)
        } else {
            egui::Color32::from_rgb(172, 180, 198)
        };
        let tab_painter = painter.with_clip_rect(tab_rect.shrink2(egui::vec2(2.0, 0.0)));
        let galley = tab_painter.layout_no_wrap(label.to_string(), font_id, text_color);
        let dot_space = if is_modified { 10.0 } else { 0.0 };
        let content_w = galley.size().x + dot_space;
        let text_pos = egui::pos2(
            (tab_rect.center().x - content_w * 0.5).max(tab_rect.min.x + 4.0),
            tab_rect.center().y - galley.size().y * 0.5 + 1.0,
        );
        tab_painter.galley(text_pos, galley.clone(), text_color);

        if is_modified {
            let dot_center = egui::pos2(
                (text_pos.x + galley.size().x + 6.0).min(tab_rect.max.x - 5.0),
                tab_rect.center().y + 1.0,
            );
            tab_painter.circle_filled(dot_center, 3.2, egui::Color32::from_rgb(255, 192, 56));
        }
    }

    (clicked_tab, collapse_clicked)
}

/// Renders a clearly delineated bordered pill/tab button for inline mode & selector strips
/// (e.g., `ToolMode`, `Build Plate` switchers, `Extrusion Tool` tabs, and `Scalar Field` tabs).
fn render_delineated_pill_tab(
    ui: &mut egui::Ui,
    is_active: bool,
    label: impl Into<String>,
) -> egui::Response {
    let label_str = label.into();
    let text = if is_active {
        egui::RichText::new(label_str)
            .strong()
            .color(egui::Color32::WHITE)
    } else {
        egui::RichText::new(label_str).color(egui::Color32::from_rgb(195, 203, 218))
    };
    let fill = if is_active {
        egui::Color32::from_rgb(44, 96, 166)
    } else {
        egui::Color32::from_rgb(28, 32, 40)
    };
    let stroke = if is_active {
        egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(104, 178, 255))
    } else {
        egui::Stroke::new(1.0_f32, egui::Color32::from_rgb(72, 80, 96))
    };
    let btn = egui::Button::new(text)
        .fill(fill)
        .stroke(stroke)
        .rounding(egui::Rounding::same(5.0));
    ui.add(btn)
}

/// Renders the reusable Preset Selector Bar at the top of a configuration sub-tab (`Printer`, `Tool`, `Filament`, `Process`).
/// Returns `true` if a preset was selected or reverted so the caller can synchronize with the backend and mark caches dirty.
fn render_preset_selector_bar(
    ui: &mut egui::Ui,
    category: PresetCategory,
    preset_lib: &mut PresetLibrary,
    config: &mut SlicerConfig,
    status_message: &mut String,
) -> bool {
    let mut applied_or_reverted = false;
    let is_modified = preset_lib.is_category_modified(category, config);
    let is_builtin = preset_lib.is_selected_builtin(category);
    let selected_name = preset_lib.selected_name(category).to_string();
    let names = preset_lib.preset_names(category);

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new(format!("{} Preset:", category.label())).strong());

                let combo_label = if is_modified {
                    format!("{selected_name} (• modified)")
                } else {
                    selected_name.clone()
                };
                let combo_w = (ui.available_width() - 95.0).clamp(105.0, 210.0);
                let mut pick_name: Option<String> = None;
                egui::ComboBox::from_id_source(("preset_selector_combo", category.label()))
                    .selected_text(combo_label)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for (name, builtin) in &names {
                            let tag = if *builtin { "[Factory]" } else { "[User]" };
                            let is_sel = name == &selected_name;
                            if ui
                                .selectable_label(is_sel, format!("{name}  {tag}"))
                                .clicked()
                                && !is_sel
                            {
                                pick_name = Some(name.clone());
                            }
                        }
                    });

                if let Some(new_name) = pick_name {
                    if preset_lib.select_and_apply(category, &new_name, config) {
                        *status_message =
                            format!("Applied {} preset '{}'.", category.label(), new_name);
                        applied_or_reverted = true;
                    }
                }

                if is_modified {
                    ui.label(
                        egui::RichText::new("• modified")
                            .small()
                            .strong()
                            .color(egui::Color32::from_rgb(255, 195, 70)),
                    );
                } else {
                    ui.small(
                        egui::RichText::new(if is_builtin { "Factory" } else { "Saved" })
                            .color(egui::Color32::GRAY),
                    );
                }
            });

            ui.horizontal_wrapped(|ui| {
                // Save (overwrites current user preset; disabled for factory presets)
                let can_overwrite_save = !is_builtin && is_modified;
                if ui
                    .add_enabled(can_overwrite_save, egui::Button::new("Save").small())
                    .on_hover_text(if is_builtin {
                        "Factory presets are read-only; use 'Save As...' to create a user preset"
                    } else {
                        "Save changes to the current user preset on disk"
                    })
                    .clicked()
                {
                    match preset_lib.save_user_preset(category, &selected_name, config) {
                        Ok(msg) => *status_message = msg,
                        Err(err) => *status_message = format!("Preset save error: {err}"),
                    }
                }

                // Save As... (opens inline name input)
                let save_as_open = preset_lib.save_as_open_for == Some(category);
                if ui
                    .selectable_label(save_as_open, "Save As...")
                    .on_hover_text("Save current settings as a new named user preset")
                    .clicked()
                {
                    if save_as_open {
                        preset_lib.save_as_open_for = None;
                    } else {
                        preset_lib.save_as_open_for = Some(category);
                        preset_lib.save_as_name_buf = format!("{selected_name} (Custom)");
                    }
                }

                // Revert (restores selected preset values)
                if ui
                    .add_enabled(is_modified, egui::Button::new("Revert").small())
                    .on_hover_text("Discard unsaved edits and revert to the selected preset")
                    .clicked()
                {
                    if preset_lib.revert_selected(category, config) {
                        *status_message = format!(
                            "Reverted {} settings to '{}'.",
                            category.label(),
                            selected_name
                        );
                        applied_or_reverted = true;
                    }
                }

                // Delete (only enabled for user presets)
                if ui
                    .add_enabled(!is_builtin, egui::Button::new("Delete").small())
                    .on_hover_text(if is_builtin {
                        "Factory presets cannot be deleted"
                    } else {
                        "Delete this user preset from disk and revert to factory default"
                    })
                    .clicked()
                {
                    match preset_lib.delete_selected_user_preset(category, config) {
                        Ok(msg) => {
                            *status_message = format!("Deleted user preset '{msg}'.");
                            applied_or_reverted = true;
                        }
                        Err(err) => *status_message = format!("Preset delete error: {err}"),
                    }
                }
            });

            if preset_lib.save_as_open_for == Some(category) {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.small("New Name:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut preset_lib.save_as_name_buf)
                            .desired_width(175.0),
                    );
                    let enter_pressed =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.small_button("Confirm Save").clicked() || enter_pressed {
                        let new_name = preset_lib.save_as_name_buf.trim().to_string();
                        match preset_lib.save_user_preset(category, &new_name, config) {
                            Ok(saved_name) => {
                                *status_message = format!(
                                    "Saved {} preset '{saved_name}'.",
                                    category.label()
                                );
                                preset_lib.save_as_open_for = None;
                            }
                            Err(err) => *status_message = format!("Save As error: {err}"),
                        }
                    }
                    if ui.small_button("Cancel").clicked() {
                        preset_lib.save_as_open_for = None;
                    }
                });
            }
        });

    applied_or_reverted
}

/// Renders the Orca-style **STEP Assembly View & Component/Material Selector** card in `Project Setup`.
///
/// Allows users to:
/// - Toggle the 3D viewport between **Build Plate View** ($z = 0$ layout) and **3D Assembly View**
///   (the original assembled STEP CAD coordinates `(X, Y, Z)` with an interactive 3D Explode slider
///   and ghost rendering for unselected reference parts).
/// - Browse and check/uncheck entire sub-assemblies, material/appearance groups, or individual bodies.
/// - Run one-click subset arrangement actions: **"Arrange Selected on Plate"**,
///   **"Arrange Selected to New Plate"**, and **"Split by Material to Plates"**.
fn render_step_assembly_view_and_tree_section(
    ui: &mut egui::Ui,
    plate: &mut BuildPlateState,
    config: &SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    object_transforms: &mut Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
) {
    if plate.objects.is_empty() || (!plate.has_step_assembly() && plate.objects.len() <= 1) {
        return;
    }

    let total_parts = plate.objects.len();
    let printable_parts = plate.objects.iter().filter(|o| o.printable).count();

    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "STEP Assembly & Subset Selector ({printable_parts}/{total_parts} selected)"
                    ))
                    .strong()
                    .color(egui::Color32::from_rgb(225, 238, 255)),
                );
                let asm_active = plate.assembly_view_active;
                let toggle_label = if asm_active {
                    "3D Assembly View: ON"
                } else {
                    "3D Assembly View"
                };
                if render_delineated_pill_tab(ui, asm_active, toggle_label)
                    .on_hover_text(
                        "Toggle Orca-style 3D CAD Assembly View: shows all STEP components in their original 3D assembled coordinates so you can inspect and pick subsets of components or sub-assemblies to print.",
                    )
                    .clicked()
                {
                    let target = !asm_active;
                    plate.set_assembly_view_active(target, object_transforms);
                }
            });

            if plate.assembly_view_active {
                ui.horizontal_wrapped(|ui| {
                    ui.small("3D Explode:");
                    let mut explode = plate.assembly_explode_mm;
                    if ui
                        .add(egui::Slider::new(&mut explode, 0.0..=150.0).suffix(" mm"))
                        .on_hover_text(
                            "Radially separate components around the assembly centroid to inspect and click internal parts",
                        )
                        .changed()
                    {
                        plate.assembly_explode_mm = explode;
                        plate.apply_assembly_view_transforms(object_transforms);
                    }
                    if plate.assembly_explode_mm > 0.0 && ui.small_button("Reset").clicked() {
                        plate.assembly_explode_mm = 0.0;
                        plate.apply_assembly_view_transforms(object_transforms);
                    }
                    ui.checkbox(&mut plate.assembly_ghost_unselected, "Ghost Unselected")
                        .on_hover_text(
                            "Render unchecked (non-printable) components as translucent X-ray ghosts in 3D Assembly View",
                        );
                });
            }

            // Quick Subset Selection & Arrangement Action Bar
            let mut printable_changed = false;
            ui.horizontal_wrapped(|ui| {
                ui.small("Select:");
                if ui
                    .small_button("All")
                    .on_hover_text("Include all components in the assembly for printing")
                    .clicked()
                {
                    for obj in &mut plate.objects {
                        obj.printable = true;
                        obj.visible = true;
                    }
                    printable_changed = true;
                }
                if ui
                    .small_button("None")
                    .on_hover_text("Uncheck all components so you can pick only the subset you want to print")
                    .clicked()
                {
                    for obj in &mut plate.objects {
                        obj.printable = false;
                    }
                    printable_changed = true;
                }
                if ui
                    .small_button("Invert")
                    .on_hover_text("Invert the selected/printable subset of components")
                    .clicked()
                {
                    for obj in &mut plate.objects {
                        obj.printable = !obj.printable;
                    }
                    printable_changed = true;
                }
                if let Some(sel_id) = plate.selected_object_id {
                    if ui
                        .small_button("Only Active")
                        .on_hover_text("Check only the currently highlighted component for printing")
                        .clicked()
                    {
                        for obj in &mut plate.objects {
                            obj.printable = obj.object_id == sel_id;
                        }
                        printable_changed = true;
                    }
                }
                if ui
                    .checkbox(
                        &mut plate.hide_non_printable_on_plate,
                        "Hide Unselected on Plate",
                    )
                    .on_hover_text(
                        "When viewing or auto-arranging the build plate, hide components that are not checked for printing",
                    )
                    .changed()
                {
                    plate.sync_plate_visibility(object_transforms);
                }
            });

            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(
                        printable_parts > 0,
                        egui::Button::new(
                            egui::RichText::new(format!(
                                "Arrange Selected on Plate ({printable_parts})"
                            ))
                            .strong(),
                        )
                        .small(),
                    )
                    .on_hover_text(
                        "Exit Assembly View and auto-arrange only the checked printable subset on the active build plate",
                    )
                    .clicked()
                {
                    let active_p = plate.active_plate_index;
                    for obj in &mut plate.objects {
                        if obj.printable {
                            obj.plate_index = active_p;
                            obj.visible = true;
                        }
                    }
                    run_auto_arrange_on_plate(plate, config, object_transforms);
                    vis_state.dirty_planned_gcode = true;
                    vis_state.dirty_toolpaths_mesh = true;
                }

                if ui
                    .add_enabled(
                        printable_parts > 0,
                        egui::Button::new("Arrange Selected to New Plate").small(),
                    )
                    .on_hover_text(
                        "Create a new build plate, assign the checked subset of components to it, and auto-arrange them",
                    )
                    .clicked()
                {
                    arrange_selected_subset_to_new_plate(plate, config, object_transforms);
                    vis_state.dirty_planned_gcode = true;
                    vis_state.dirty_toolpaths_mesh = true;
                }

                let has_multiple_materials = {
                    let mut mats: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for o in &plate.objects {
                        if !o.step_material_name.is_empty() {
                            mats.insert(o.step_material_name.clone());
                        } else if let Some(&s) = o.triangle_filament_slots.first() {
                            mats.insert(format!("slot_{s}"));
                        }
                    }
                    mats.len() > 1
                };
                if has_multiple_materials
                    && ui
                        .add_enabled(
                            printable_parts > 0,
                            egui::Button::new("Split by Material to Plates").small(),
                        )
                        .on_hover_text(
                            "Split all checked printable components across separate build plates grouped by STEP material / colour",
                        )
                        .clicked()
                {
                    split_assembly_to_plates_by_material(plate, config, object_transforms);
                    vis_state.dirty_planned_gcode = true;
                    vis_state.dirty_toolpaths_mesh = true;
                }
            });

            ui.separator();

            // Grouping Mode Switcher: By CAD Assembly Hierarchy vs By Material / Appearance
            ui.horizontal_wrapped(|ui| {
                ui.small("Group By:");
                if render_delineated_pill_tab(
                    ui,
                    !plate.assembly_group_by_material,
                    "Assembly Hierarchy",
                )
                .clicked()
                {
                    plate.assembly_group_by_material = false;
                }
                if render_delineated_pill_tab(
                    ui,
                    plate.assembly_group_by_material,
                    "Material / Colour",
                )
                .clicked()
                {
                    plate.assembly_group_by_material = true;
                }
            });

            // Build grouped index lists in order of appearance
            let mut groups: Vec<(String, Option<[f32; 4]>, Vec<usize>)> = Vec::new();
            for (idx, obj) in plate.objects.iter().enumerate() {
                let (group_title, swatch_opt) = if plate.assembly_group_by_material {
                    let mat_name = if !obj.step_material_name.is_empty() {
                        if obj.step_material_density_g_cm3 > 0.0 {
                            format!(
                                "{} ({:.2} g/cm³)",
                                obj.step_material_name, obj.step_material_density_g_cm3
                            )
                        } else {
                            obj.step_material_name.clone()
                        }
                    } else if let Some(&slot) = obj.triangle_filament_slots.first() {
                        config
                            .filament_slots
                            .get(slot as usize)
                            .map(|s| s.name.clone())
                            .unwrap_or_else(|| format!("Filament Slot {}", slot + 1))
                    } else {
                        "Unassigned / Default Material".to_string()
                    };
                    let swatch = obj.custom_color_srgba.or_else(|| {
                        obj.triangle_filament_slots
                            .first()
                            .and_then(|&s| config.filament_slots.get(s as usize))
                            .map(|slot| slot.color_srgba)
                    });
                    (mat_name, swatch)
                } else {
                    let folder = if obj.assembly_path.len() > 1 {
                        obj.assembly_path[..obj.assembly_path.len() - 1].join(" / ")
                    } else {
                        std::path::Path::new(&obj.source_file)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Assembly Root")
                            .to_string()
                    };
                    (folder, None)
                };

                if let Some(existing) = groups.iter_mut().find(|(g, _, _)| g == &group_title) {
                    if existing.1.is_none() && swatch_opt.is_some() {
                        existing.1 = swatch_opt;
                    }
                    existing.2.push(idx);
                } else {
                    groups.push((group_title, swatch_opt, vec![idx]));
                }
            }

            let current_selected_id = plate.selected_object_id;
            let mut new_selected_id: Option<u64> = None;

            egui::ScrollArea::vertical()
                .id_source("step_assembly_tree_scroll")
                .max_height(210.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for (g_idx, (group_title, group_swatch, member_indices)) in
                        groups.iter().enumerate()
                    {
                        let group_total = member_indices.len();
                        let group_checked = member_indices
                            .iter()
                            .filter(|&&i| plate.objects[i].printable)
                            .count();
                        let mut all_checked = group_checked == group_total && group_total > 0;

                        ui.push_id(("step_asm_group", g_idx), |ui| {
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .checkbox(&mut all_checked, "")
                                    .on_hover_text(
                                        "Check/uncheck all components in this sub-assembly or material group",
                                    )
                                    .changed()
                                {
                                    for &i in member_indices {
                                        plate.objects[i].printable = all_checked;
                                        if all_checked {
                                            plate.objects[i].visible = true;
                                        }
                                    }
                                    printable_changed = true;
                                }
                                if let Some(rgba) = group_swatch {
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(rect, 2.0, rgba_to_color32(*rgba));
                                }
                                ui.label(
                                    egui::RichText::new(format!(
                                        "{group_title} ({group_checked}/{group_total})"
                                    ))
                                    .strong()
                                    .color(egui::Color32::from_rgb(205, 220, 242)),
                                );
                            });

                            ui.indent(("step_asm_group_indent", g_idx), |ui| {
                                for &obj_idx in member_indices {
                                    let obj = &mut plate.objects[obj_idx];
                                    let is_selected = current_selected_id == Some(obj.object_id);
                                    ui.horizontal_wrapped(|ui| {
                                        if ui
                                            .checkbox(&mut obj.printable, "")
                                            .on_hover_text("Include this component for printing")
                                            .changed()
                                        {
                                            if obj.printable {
                                                obj.visible = true;
                                            }
                                            printable_changed = true;
                                        }
                                        let part_swatch = obj.custom_color_srgba.or_else(|| {
                                            obj.triangle_filament_slots
                                                .first()
                                                .and_then(|&s| {
                                                    config.filament_slots.get(s as usize)
                                                })
                                                .map(|slot| slot.color_srgba)
                                        });
                                        if let Some(rgba) = part_swatch {
                                            let (rect, _) = ui.allocate_exact_size(
                                                egui::vec2(9.0, 9.0),
                                                egui::Sense::hover(),
                                            );
                                            ui.painter()
                                                .rect_filled(rect, 2.0, rgba_to_color32(rgba));
                                        }
                                        let mat_tag = if !obj.step_material_name.is_empty()
                                            && !plate.assembly_group_by_material
                                        {
                                            format!(" [{}]", obj.step_material_name)
                                        } else {
                                            String::new()
                                        };
                                        let leaf_label = format!(
                                            "{}{} ({} tris)",
                                            obj.name,
                                            mat_tag,
                                            obj.indices.len() / 3
                                        );
                                        if ui.selectable_label(is_selected, leaf_label).clicked() {
                                            new_selected_id = Some(obj.object_id);
                                        }
                                    });
                                }
                            });
                        });
                    }
                });

            if let Some(sel_id) = new_selected_id {
                plate.selected_object_id = Some(sel_id);
            }
            if printable_changed {
                plate.sync_plate_visibility(object_transforms);
                vis_state.dirty_planned_gcode = true;
                vis_state.dirty_toolpaths_mesh = true;
            }
        });
}

/// Renders the top workflow/editing toolbar, the left Project Setup & Configuration/Presets panel,
/// and the right View & Inspection panel.
#[allow(clippy::too_many_arguments)]
pub fn ui_workspace_system(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut contexts: EguiContexts,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
    mut preset_lib: ResMut<PresetLibrary>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut calib_state: ResMut<CalibrationUiState>,
    mut moonraker_state: ResMut<MoonrakerUiState>,
    mut object_transforms: Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    mut camera_query: Query<
        (&Camera, &GlobalTransform, &mut OrbitCamera, &mut Transform),
        Without<CadObjectEntity>,
    >,
) {
    plate.ui_hovered_surface_hint_patch = None;
    plate.ui_hovered_point_hint_id = None;
    vis_state.highlighted_selected_layer = None;
    config.sync_auto_voxel_size();
    let Some(ctx) = contexts.try_ctx_mut() else {
        return;
    };
    vis_state.ui_pointer_down = ctx.is_using_pointer();

    let is_io_busy = plate.pending_load_task.is_some() || plate.pending_save_task.is_some();
    let is_computing = vis_state.is_busy();
    let has_eikonal = vis_state.available_fields.iter().any(|f| f == "eikonal");

    for obj in &mut plate.objects {
        if let Ok((t, _)) = object_transforms.get(obj.entity) {
            obj.cached_rotation = t.rotation;
        }
    }

    // Pre-fetch selected object info for the workflow bar and panels
    let selected_obj_info: Option<(u64, usize, String, Quat)> =
        plate.selected_object_id.and_then(|sel_id| {
            let idx = plate.objects.iter().position(|o| o.object_id == sel_id)?;
            let name = plate.objects[idx].name.clone();
            let rot = plate.objects[idx].cached_rotation;
            Some((sel_id, idx, name, rot))
        });

    // ========================================================================
    // 1. Top Toolbar: Project I/O, Viewport Editing Tools & Workflow Pipeline
    // ========================================================================
    let screen_width = ctx.screen_rect().width().max(640.0);
    let top_bar_resp = egui::TopBottomPanel::top("top_editing_toolbar").show(ctx, |ui| {
        // Row 1: Left Panel Toggle, File I/O, Viewport Editing Mode, Scene Actions, and Right Panel Toggle
        ui.horizontal_wrapped(|ui| {
            ui.toggle_value(
                &mut preset_lib.show_left_panel,
                egui::RichText::new("◧ Setup & Config").strong(),
            )
            .on_hover_text(
                "Show/hide the left Project Setup & Configuration side panel",
            );
            ui.separator();

            if render_icon_button(
                ui,
                UiIcon::ImportFile,
                "Import CAD / Mesh / 3MF...",
                false,
                !is_io_busy,
            )
            .clicked()
            {
                plate.status_message =
                    "Select one or more STEP, STL, OBJ, or 3MF files...".to_string();
                let task = IoTaskPool::get().spawn(async move {
                    let handles = rfd::AsyncFileDialog::new()
                        .add_filter(
                            "Supported 3D Files (.step, .stp, .stl, .obj, .3mf)",
                            &[
                                "step", "stp", "STEP", "STP", "stl", "STL", "obj", "OBJ", "3mf",
                                "3MF",
                            ],
                        )
                        .add_filter("STEP CAD Assemblies", &["step", "stp", "STEP", "STP"])
                        .add_filter("STL Meshes", &["stl", "STL"])
                        .add_filter("OBJ Meshes", &["obj", "OBJ"])
                        .add_filter("3MF Projects", &["3mf", "3MF"])
                        .pick_files()
                        .await?;

                    let paths: Vec<_> =
                        handles.into_iter().map(|h| h.path().to_path_buf()).collect();
                    Some(load_files_into_backend(&paths))
                });
                plate.pending_load_task = Some(task);
            }

            if render_icon_button(
                ui,
                UiIcon::SaveProject,
                "Save 3MF...",
                false,
                !is_io_busy && !plate.objects.is_empty(),
            )
            .clicked()
            {
                let mut export_list = Vec::with_capacity(plate.objects.len());
                for obj in &plate.objects {
                    let tf = object_transforms
                        .get(obj.entity)
                        .map(|(t, _)| *t)
                        .unwrap_or(Transform::IDENTITY);
                    export_list.push(ExportableObject {
                        object_id: obj.object_id,
                        name: obj.name.clone(),
                        has_brep: obj.has_brep,
                        printable: obj.printable,
                        plate_index: obj.plate_index,
                        transform: tf,
                        vertices: obj.vertices.clone(),
                        indices: obj.indices.clone(),
                        filament_mode: obj.filament_mode,
                        custom_color_srgba: obj.custom_color_srgba,
                    });
                }
                let plate_names_snapshot = plate.plate_names();
                let cfg_snapshot = (*config).clone();
                let default_3mf_filename = plate
                    .project_3mf_filename
                    .clone()
                    .unwrap_or_else(|| "project.3mf".to_string());
                plate.status_message = "Choosing destination for 3MF export...".to_string();

                let save_task = IoTaskPool::get().spawn(async move {
                    let file_handle = rfd::AsyncFileDialog::new()
                        .add_filter("3MF Project Archive", &["3mf"])
                        .set_file_name(&default_3mf_filename)
                        .save_file()
                        .await?;
                    let out_path = file_handle.path().to_path_buf();
                    let res = save_project_to_3mf(
                        &out_path,
                        &plate_names_snapshot,
                        &export_list,
                        &cfg_snapshot,
                    )
                    .map(|()| out_path.to_string_lossy().into_owned());
                    Some(res)
                });
                plate.pending_save_task = Some(save_task);
            }

            let has_planned = vis_state.cached_planned_gcode.is_some();
            if render_icon_button(
                ui,
                UiIcon::ExportGcode,
                "Export G-code...",
                false,
                has_planned && !is_computing && !is_io_busy,
            )
            .on_hover_text("Export planned non-planar or calibration G-code (.gcode)")
            .clicked()
            {
                if let Some(ref planned) = vis_state.cached_planned_gcode {
                    let stem = if let Some((_, _, ref obj_name, _)) = selected_obj_info {
                        std::path::Path::new(obj_name)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("manifold2_print")
                            .to_string()
                    } else if !calib_state.last_generated_filename_stem.is_empty() {
                        calib_state.last_generated_filename_stem.clone()
                    } else {
                        "manifold2_calibration".to_string()
                    };
                    let default_filename = if config.gcode.gcode_dialect == GcodeDialect::BambuLab {
                        format!("{stem}.gcode.3mf")
                    } else {
                        format!("{stem}.gcode")
                    };
                    let gcode_bytes = planned.gcode_text.clone();
                    let cfg_snapshot = (*config).clone();
                    let total_moves =
                        planned.extruding_move_count + planned.travel_move_count;
                    plate.status_message =
                        "Choosing destination for G-code export...".to_string();

                    let save_task = IoTaskPool::get().spawn(async move {
                        let file_handle = rfd::AsyncFileDialog::new()
                            .add_filter("G-code / Bambu 3MF (.gcode, .gcode.3mf, .3mf, .nc)", &["gcode", "3mf", "nc"])
                            .set_file_name(&default_filename)
                            .save_file()
                            .await?;
                        let out_path = file_handle.path().to_path_buf();
                        let out_str = out_path.to_string_lossy().to_ascii_lowercase();
                        let payload_res = if out_str.ends_with(".3mf") {
                            crate::io::threemf::build_bambu_gcode_3mf_bytes(
                                &gcode_bytes,
                                Some(&cfg_snapshot),
                            )
                        } else {
                            Ok(gcode_bytes.into_bytes())
                        };
                        let res = payload_res.and_then(|bytes| {
                            let kb = bytes.len() as f64 / 1024.0;
                            std::fs::write(&out_path, bytes)
                                .map(|()| {
                                    format!(
                                        "{} ({:.1} KB, {} moves)",
                                        out_path.display(),
                                        kb,
                                        total_moves
                                    )
                                })
                                .map_err(|e| e.to_string())
                        });
                        Some(res)
                    });
                    plate.pending_save_task = Some(save_task);
                }
            }

            if config.network.is_configured() {
                let proto_label = config.network.protocol.short_label();
                let is_uploading = moonraker_state.pending_upload_task.is_some();
                let can_upload =
                    has_planned && !is_computing && !is_io_busy && !is_uploading;
                let mut trigger_upload: Option<bool> = None;

                let upload_btn_label = if is_uploading && !moonraker_state.upload_start_print {
                    "Uploading..."
                } else {
                    "Upload G-code"
                };
                if ui
                    .add_enabled(can_upload, egui::Button::new(upload_btn_label))
                    .on_hover_text(format!(
                        "Upload planned G-code to {proto_label} ({}) without starting a print",
                        config.network.printer_uri.trim()
                    ))
                    .clicked()
                {
                    trigger_upload = Some(false);
                }

                let upload_print_label = if is_uploading && moonraker_state.upload_start_print {
                    "Uploading & Printing..."
                } else {
                    "Upload & Print"
                };
                if ui
                    .add_enabled(
                        can_upload,
                        egui::Button::new(
                            egui::RichText::new(upload_print_label)
                                .strong()
                                .color(egui::Color32::from_rgb(130, 230, 165)),
                        ),
                    )
                    .on_hover_text(format!(
                        "Upload planned G-code to {proto_label} ({}) and immediately start printing",
                        config.network.printer_uri.trim()
                    ))
                    .clicked()
                {
                    trigger_upload = Some(true);
                }

                if let Some(start_print) = trigger_upload {
                    if let Some(ref planned) = vis_state.cached_planned_gcode {
                        let stem = if let Some((_, _, ref obj_name, _)) = selected_obj_info {
                            std::path::Path::new(obj_name)
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("manifold2_print")
                                .to_string()
                        } else if !calib_state.last_generated_filename_stem.is_empty() {
                            calib_state.last_generated_filename_stem.clone()
                        } else {
                            "manifold2_calibration".to_string()
                        };
                        spawn_network_gcode_upload(
                            &mut moonraker_state,
                            config.network.clone(),
                            format!("{stem}.gcode"),
                            planned.gcode_text.clone(),
                            start_print,
                            &mut plate.status_message,
                        );
                    }
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("Tool:").strong());
            for mode in [
                ToolMode::Select,
                ToolMode::MoveOnBed,
                ToolMode::Rotate,
                ToolMode::PlaceOnFace,
                ToolMode::SurfaceHintOrthogonal,
                ToolMode::SurfaceHintTangent,
                ToolMode::PointHint,
            ] {
                let selected = plate.tool_mode == mode;
                if render_delineated_icon_pill_tab(
                    ui,
                    selected,
                    Some(UiIcon::for_tool_mode(mode)),
                    mode.label(),
                )
                .clicked()
                {
                    plate.tool_mode = mode;
                }
            }

            ui.separator();
            ui.label(egui::RichText::new("Plate:").strong());
            let mut switch_to_plate: Option<usize> = None;
            for p_idx in 0..plate.plates.len() {
                let p_count = plate.plate_object_count(p_idx);
                let p_label = format!("{} ({p_count})", plate.plates[p_idx].name);
                let is_active = plate.active_plate_index == p_idx;
                if render_delineated_icon_pill_tab(
                    ui,
                    is_active,
                    Some(UiIcon::BuildPlate),
                    p_label,
                )
                .clicked()
                    && !is_active
                {
                    switch_to_plate = Some(p_idx);
                }
            }
            if render_icon_button(ui, UiIcon::AddPlate, "Plate", true, true)
                .on_hover_text("Add a new empty build plate")
                .clicked()
            {
                plate.add_plate(&mut object_transforms);
            }
            if let Some(new_p_idx) = switch_to_plate {
                plate.switch_active_plate(new_p_idx, &mut object_transforms);
                vis_state.dirty_planned_gcode = true;
                vis_state.dirty_toolpaths_mesh = true;
                let active_name = plate.active_plate_name().to_string();
                let active_cnt = plate.active_plate_object_count();
                plate.status_message =
                    format!("Switched to {active_name} ({active_cnt} object(s)).");
            }

            ui.separator();
            if render_icon_button(ui, UiIcon::AutoArrange, "Auto-Arrange", false, true).clicked() {
                run_auto_arrange_on_plate(&mut plate, &config, &mut object_transforms);
                vis_state.dirty_planned_gcode = true;
                vis_state.dirty_toolpaths_mesh = true;
            }
            if plate.has_step_assembly() || plate.objects.len() > 1 {
                let asm_active = plate.assembly_view_active;
                if render_delineated_pill_tab(
                    ui,
                    asm_active,
                    if asm_active {
                        "Assembly View: ON"
                    } else {
                        "Assembly View"
                    },
                )
                .on_hover_text(
                    "Toggle Orca-style 3D STEP Assembly View (original 3D CAD assembled positions vs arranged build plate)",
                )
                .clicked()
                {
                    let target = !asm_active;
                    plate.set_assembly_view_active(target, &mut object_transforms);
                }
            }
            if render_icon_button(ui, UiIcon::FrameCamera, "Frame All", false, true).clicked() {
                frame_camera_to_plate(&plate, &config, &mut camera_query);
            }
            if render_delineated_icon_pill_tab(
                ui,
                plate.show_object_labels,
                Some(UiIcon::ObjectLabels),
                "Object Labels",
            )
            .on_hover_text(
                "Show/hide each object's Klipper exclude_object name label on the build plate",
            )
            .clicked()
            {
                plate.show_object_labels = !plate.show_object_labels;
            }

            ui.separator();
            ui.toggle_value(
                &mut calib_state.show_window,
                egui::RichText::new("Calibration Utilities").strong(),
            )
            .on_hover_text(
                "Open Extrusion Multiplier (Orca-style + Die Swell), Pressure Advance (Ellis-style), and Transient Melt-Zone Pressure calibration generators",
            );
            ui.toggle_value(
                &mut preset_lib.show_view_panel,
                egui::RichText::new("◨ View & Inspect").strong(),
            )
            .on_hover_text(
                "Show/hide the right View & Inspection side panel",
            );
        });

        ui.separator();

        // Row 2: Slicing & G-code Workflow Pipeline Bar
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Workflow:").strong());
            if let Some((sel_id, _, ref obj_name, obj_rotation)) = selected_obj_info {
                let step1_label = if has_eikonal {
                    "1. Recompute Fields (✓)"
                } else {
                    "1. Compute Fields"
                };
                if ui
                    .add_enabled(!is_computing, egui::Button::new(step1_label))
                    .on_hover_text("Compute OpenVDB SDF, constrained Eikonal order field, overhang, and support_needed grids")
                    .clicked()
                {
                    spawn_field_compute_task(
                        &mut vis_state,
                        &mut plate,
                        &config,
                        sel_id,
                        obj_name,
                        obj_rotation,
                    );
                }

                let step2_label = if let Some(ref res) = vis_state.cached_selected_layers {
                    format!("2. Select Isosurfaces ({} layers)", res.layers.len())
                } else {
                    "2. Select Isosurfaces".to_string()
                };
                if ui
                    .add_enabled(!is_computing, egui::Button::new(step2_label))
                    .on_hover_text("Select non-planar slicing isosurfaces across topological branches")
                    .clicked()
                {
                    vis_state.show_selected_layers = true;
                    if vis_state.solid_opacity > 0.45 {
                        vis_state.solid_opacity = 0.12;
                    }
                    let printable_ids = plate.active_plate_printable_object_ids();
                    let all_have_eikonal = has_eikonal
                        && printable_ids
                            .iter()
                            .all(|&id| ffi::list_object_fields(id).iter().any(|f| f == "eikonal"));
                    if all_have_eikonal {
                        let need_toolpaths = vis_state.show_toolpaths;
                        vis_state.spawn_slicing_task_for_objects(
                            sel_id,
                            printable_ids,
                            &config,
                            true,
                            need_toolpaths,
                        );
                        plate.status_message = format!(
                            "Selecting non-planar isosurfaces for '{obj_name}' in background..."
                        );
                    } else {
                        spawn_field_compute_task(
                            &mut vis_state,
                            &mut plate,
                            &config,
                            sel_id,
                            obj_name,
                            obj_rotation,
                        );
                    }
                }

                let step3_label = if let Some(ref tp) = vis_state.cached_toolpaths {
                    let active_printable_ids = plate.active_plate_printable_object_ids();
                    let plate_paths: u32 = active_printable_ids
                        .iter()
                        .filter_map(|id| vis_state.plate_unmixed_toolpaths.get(id))
                        .map(|t| t.total_path_count)
                        .sum();
                    let count = if plate_paths > 0 {
                        plate_paths
                    } else {
                        tp.total_path_count
                    };
                    format!("3. Generate Toolpaths & G-code ({count} paths)")
                } else {
                    "3. Generate Toolpaths & G-code".to_string()
                };
                if ui
                    .add_enabled(!is_computing, egui::Button::new(step3_label))
                    .on_hover_text("Generate non-planar shell/infill/support toolpaths and plan Klipper G-code")
                    .clicked()
                {
                    vis_state.show_toolpaths = true;
                    if vis_state.solid_opacity > 0.35 {
                        vis_state.solid_opacity = 0.10;
                    }
                    let printable_ids = plate.active_plate_printable_object_ids();
                    let all_have_eikonal = has_eikonal
                        && printable_ids
                            .iter()
                            .all(|&id| ffi::list_object_fields(id).iter().any(|f| f == "eikonal"));
                    if all_have_eikonal {
                        let printable_cnt = printable_ids.len().max(1);
                        let need_layers = vis_state.cached_selected_layers.is_none()
                            || vis_state.dirty_selected_layers;
                        vis_state.spawn_slicing_task_for_objects(
                            sel_id,
                            printable_ids,
                            &config,
                            need_layers,
                            true,
                        );
                        plate.status_message = format!(
                            "Generating non-planar toolpaths & G-code for {printable_cnt} printable object(s) in background..."
                        );
                    } else {
                        spawn_field_compute_task(
                            &mut vis_state,
                            &mut plate,
                            &config,
                            sel_id,
                            obj_name,
                            obj_rotation,
                        );
                    }
                }

                if ui
                    .add_enabled(
                        !is_computing,
                        egui::Button::new(egui::RichText::new("Slice & Plan All").strong()),
                    )
                    .on_hover_text("Run the full pipeline: compute OpenVDB fields, select layers, generate toolpaths, and plan G-code")
                    .clicked()
                {
                    vis_state.show_toolpaths = true;
                    if vis_state.solid_opacity > 0.35 {
                        vis_state.solid_opacity = 0.12;
                    }
                    spawn_field_compute_task(
                        &mut vis_state,
                        &mut plate,
                        &config,
                        sel_id,
                        obj_name,
                        obj_rotation,
                    );
                }

                if let Some((frac, stage)) = vis_state.active_progress() {
                    ui.ctx().request_repaint();
                    ui.add(
                        egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                            .desired_width(210.0)
                            .show_percentage()
                            .animate(true)
                            .text(stage),
                    );
                } else if let Some(ref planned) = vis_state.cached_planned_gcode {
                    let total_sec = planned.total_time_s.round() as u64;
                    let hrs = total_sec / 3600;
                    let mins = (total_sec % 3600) / 60;
                    let secs = total_sec % 60;
                    ui.small(
                        egui::RichText::new(format!(
                            "Est: {:02}h {:02}m {:02}s | {:.2} m ({:.1} g)",
                            hrs,
                            mins,
                            secs,
                            planned.total_filament_mm / 1000.0,
                            planned.total_filament_grams
                        ))
                        .color(egui::Color32::from_rgb(175, 225, 190)),
                    );
                }
            } else {
                ui.small("Import and select a 3D object to compute fields, slice non-planar layers, and generate G-code.");
            }

            if config.network.is_configured() {
                ui.separator();
                render_moonraker_toolbar_status_strip(
                    ui,
                    &config,
                    &mut moonraker_state,
                    &mut plate.status_message,
                );
            }
        });
    });

    // ========================================================================
    // 2. Left Panel: Project Setup vs. Configuration & Saved Presets
    // ========================================================================
    let left_default_w = (screen_width * 0.22).clamp(245.0, 300.0);
    let left_max_w = (screen_width * 0.46).max(340.0);
    if preset_lib.show_left_panel {
        egui::SidePanel::left("slicer_left_panel")
            .default_width(left_default_w)
            .width_range(200.0..=left_max_w)
            .resizable(true)
            .show(ctx, |outer_ui| {
                // Lock the SidePanel's outer `min_rect` width to its exact allocated `max_rect` width
                // so wide child widgets can never push the panel edge past the resize handle or trigger
                // auto-expansion when loading a .3mf file.
                let panel_rect = outer_ui.max_rect();
                let mut child_ui = outer_ui.child_ui(panel_rect, *outer_ui.layout(), None);
                child_ui.set_clip_rect(panel_rect);
                let ui = &mut child_ui;
                let panel_w = panel_rect.width();
                ui.spacing_mut().slider_width = (panel_w * 0.34).clamp(62.0, 135.0);
                ui.spacing_mut().combo_width = (panel_w * 0.42).clamp(85.0, 175.0);

                // Top-level Left Panel Mode Switcher (Folder-style tab bar + integrated collapse button)
                let any_preset_dirty = PresetCategory::ALL
                    .iter()
                    .any(|&cat| preset_lib.is_category_modified(cat, &config));
                let is_narrow_left = ui.available_width() < 310.0;
                let setup_label = if is_narrow_left {
                    format!("Setup ({})", plate.objects.len())
                } else {
                    format!("Project Setup ({})", plate.objects.len())
                };
                let cfg_label = if is_narrow_left {
                    "Config & Presets"
                } else {
                    "Configuration & Presets"
                };
                let top_tabs = [
                    (setup_label.as_str(), false),
                    (cfg_label, any_preset_dirty),
                ];
                let active_top_idx = match preset_lib.left_tab {
                    LeftPanelTab::ProjectSetup => 0,
                    LeftPanelTab::Configuration => 1,
                };
                let (clicked_top, collapse_left) = render_folder_tab_bar_with_collapse(
                    ui,
                    "left_panel_primary_tabs",
                    &top_tabs,
                    active_top_idx,
                    30.0,
                    egui::Color32::from_rgb(76, 166, 255),
                    true,
                );
                if let Some(clicked_idx) = clicked_top {
                    preset_lib.left_tab = if clicked_idx == 0 {
                        LeftPanelTab::ProjectSetup
                    } else {
                        LeftPanelTab::Configuration
                    };
                }
                if collapse_left {
                    preset_lib.show_left_panel = false;
                }
                ui.add_space(4.0);

                // When in Configuration & Presets mode, keep the 4 category sub-tabs and preset
                // selector bar pinned above the scroll area so they stay visible while scrolling.
                if preset_lib.left_tab == LeftPanelTab::Configuration {
                    let sub_tabs = [
                        (
                            PresetCategory::Printer.label(),
                            preset_lib.is_category_modified(PresetCategory::Printer, &config),
                        ),
                        (
                            PresetCategory::Tool.label(),
                            preset_lib.is_category_modified(PresetCategory::Tool, &config),
                        ),
                        (
                            PresetCategory::Filament.label(),
                            preset_lib.is_category_modified(PresetCategory::Filament, &config),
                        ),
                        (
                            PresetCategory::Process.label(),
                            preset_lib.is_category_modified(PresetCategory::Process, &config),
                        ),
                    ];
                    let active_sub_idx = PresetCategory::ALL
                        .iter()
                        .position(|&c| c == preset_lib.config_subtab)
                        .unwrap_or(0);
                    if let Some(clicked_sub) = render_folder_tab_bar(
                        ui,
                        "left_panel_config_subtabs",
                        &sub_tabs,
                        active_sub_idx,
                        28.0,
                        egui::Color32::from_rgb(240, 168, 56),
                    ) {
                        preset_lib.config_subtab = PresetCategory::ALL[clicked_sub];
                    }
                    ui.add_space(4.0);

                    let active_category = preset_lib.config_subtab;
                    if render_preset_selector_bar(
                        ui,
                        active_category,
                        &mut preset_lib,
                        &mut config,
                        &mut plate.status_message,
                    ) {
                        if active_category == PresetCategory::Printer {
                            moonraker_state.last_test_result = None;
                            moonraker_state.last_status = None;
                            moonraker_state.last_upload_result = None;
                            moonraker_state.last_poll_instant = None;
                        }
                        if !is_computing {
                            sync_config_to_backend(&config);
                        }
                        vis_state.dirty_selected_layers = true;
                        vis_state.dirty_toolpaths = true;
                        vis_state.dirty_planned_gcode = true;
                    }
                    ui.add_space(6.0);
                }

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width());
                        if preset_lib.left_tab == LeftPanelTab::ProjectSetup {
                    // ================================================================
                    // TAB 1: PROJECT SETUP (Build Plates, Objects, Placement, Hints)
                    // ================================================================
                    egui::Frame::group(ui.style())
                        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Build Plates ({})",
                                        plate.plates.len()
                                    ))
                                    .strong(),
                                );
                                if render_icon_button(ui, UiIcon::AddPlate, "Add Plate", true, true)
                                    .on_hover_text("Add a new empty build plate to the project")
                                    .clicked()
                                {
                                    plate.add_plate(&mut object_transforms);
                                }
                                if plate.plates.len() > 1
                                    && render_icon_button(
                                        ui,
                                        UiIcon::DeleteObject,
                                        "Remove Plate",
                                        true,
                                        true,
                                    )
                                    .on_hover_text(
                                        "Remove the currently active build plate and any objects on it",
                                    )
                                    .clicked()
                                {
                                    let active_idx = plate.active_plate_index;
                                    plate.remove_plate(
                                        &mut commands,
                                        active_idx,
                                        &mut object_transforms,
                                    );
                                }
                            });

                            let mut switch_to_plate: Option<usize> = None;
                            ui.horizontal_wrapped(|ui| {
                                for p_idx in 0..plate.plates.len() {
                                    let p_count = plate.plate_object_count(p_idx);
                                    let p_label =
                                        format!("{} ({p_count})", plate.plates[p_idx].name);
                                    let is_active = plate.active_plate_index == p_idx;
                                    if render_delineated_icon_pill_tab(
                                        ui,
                                        is_active,
                                        Some(UiIcon::BuildPlate),
                                        p_label,
                                    )
                                    .clicked()
                                        && !is_active
                                    {
                                        switch_to_plate = Some(p_idx);
                                    }
                                }
                            });
                            if let Some(new_p_idx) = switch_to_plate {
                                plate.switch_active_plate(new_p_idx, &mut object_transforms);
                                let active_name = plate.active_plate_name().to_string();
                                let active_cnt = plate.active_plate_object_count();
                                plate.status_message = format!(
                                    "Switched to {active_name} ({active_cnt} object(s))."
                                );
                            }

                            let active_idx = plate.active_plate_index;
                            if let Some(active_plate_info) = plate.plates.get_mut(active_idx) {
                                ui.horizontal_wrapped(|ui| {
                                    ui.small("Plate Name:");
                                    let edit_w = (ui.available_width() - 12.0).clamp(90.0, 180.0);
                                    ui.add(
                                        egui::TextEdit::singleline(&mut active_plate_info.name)
                                            .desired_width(edit_w),
                                    );
                                });
                            }
                        });

                    ui.add_space(4.0);
                    render_step_assembly_view_and_tree_section(
                        ui,
                        &mut plate,
                        &config,
                        &mut vis_state,
                        &mut object_transforms,
                    );
                    ui.add_space(4.0);
                    let active_plate_idx = plate.active_plate_index;
                    let active_plate_name = plate.active_plate_name().to_string();
                    let active_obj_count = plate.active_plate_object_count();

                    ui.horizontal_wrapped(|ui| {
                        ui.heading(format!(
                            "Objects on {active_plate_name} ({active_obj_count})"
                        ));
                        if active_obj_count > 0
                            && render_icon_button(ui, UiIcon::DeleteObject, "Clear Plate", true, true)
                                .clicked()
                        {
                            let plate_ids: Vec<u64> = plate
                                .objects
                                .iter()
                                .filter(|o| o.plate_index == active_plate_idx)
                                .map(|o| o.object_id)
                                .collect();
                            for id in plate_ids {
                                vis_state.plate_unmixed_toolpaths.remove(&id);
                                vis_state.plate_mixed_toolpaths.remove(&id);
                                remove_object_from_plate(&mut commands, &mut plate, id);
                            }
                            vis_state.dirty_planned_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        if plate.plates.len() > 1
                            && !plate.objects.is_empty()
                            && render_icon_button(
                                ui,
                                UiIcon::DeleteObject,
                                "Clear All Plates",
                                true,
                                true,
                            )
                            .clicked()
                        {
                            let all_ids: Vec<u64> =
                                plate.objects.iter().map(|o| o.object_id).collect();
                            for id in all_ids {
                                vis_state.plate_unmixed_toolpaths.remove(&id);
                                vis_state.plate_mixed_toolpaths.remove(&id);
                                remove_object_from_plate(&mut commands, &mut plate, id);
                            }
                            vis_state.dirty_planned_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                    });

                    if active_obj_count == 0 {
                        ui.small(format!(
                            "No objects on {active_plate_name}. Click 'Import CAD / Mesh / 3MF...' in the top toolbar or switch plates above."
                        ));
                    } else {
                        let mut remove_id: Option<u64> = None;
                        let mut clone_id: Option<u64> = None;
                        let mut select_id: Option<u64> = None;
                        let current_selected_id = plate.selected_object_id;

                        for obj in &mut plate.objects {
                            if obj.plate_index != active_plate_idx {
                                continue;
                            }
                            let is_selected = current_selected_id == Some(obj.object_id);
                            ui.horizontal_wrapped(|ui| {
                                if ui
                                    .checkbox(&mut obj.visible, "")
                                    .on_hover_text("Show/hide object in the 3D viewport")
                                    .changed()
                                {
                                    if let Ok((_, mut vis)) = object_transforms.get_mut(obj.entity)
                                    {
                                        *vis = if obj.visible {
                                            Visibility::Inherited
                                        } else {
                                            Visibility::Hidden
                                        };
                                    }
                                }

                                if ui
                                    .checkbox(&mut obj.printable, "Print")
                                    .on_hover_text(
                                        "Include this object when slicing and generating G-code (.3mf printable attribute)",
                                    )
                                    .changed()
                                {
                                    vis_state.dirty_planned_gcode = true;
                                    vis_state.dirty_toolpaths_mesh = true;
                                }

                                let tag = if obj.has_brep { "[STEP]" } else { "[Mesh]" };
                                let label = format!(
                                    "{} {} ({} tris)",
                                    tag,
                                    obj.name,
                                    obj.indices.len() / 3
                                );
                                if ui.selectable_label(is_selected, label).clicked() {
                                    select_id = Some(obj.object_id);
                                }

                                if render_object_filament_mode_combo(
                                    ui,
                                    obj.object_id,
                                    &mut obj.filament_mode,
                                    obj.custom_color_srgba,
                                    &config,
                                    true,
                                ) {
                                    vis_state.dirty_color_mixing = true;
                                    vis_state.dirty_surface_colors = true;
                                    vis_state.dirty_planned_gcode = true;
                                    vis_state.dirty_toolpaths_mesh = true;
                                }

                                if render_icon_button(ui, UiIcon::CloneObject, "Clone", true, true)
                                    .on_hover_text(
                                        "Add a duplicate of this object onto this build plate",
                                    )
                                    .clicked()
                                {
                                    clone_id = Some(obj.object_id);
                                }

                                if render_icon_button(ui, UiIcon::DeleteObject, "Remove", true, true)
                                    .on_hover_text("Remove this object from the build plate")
                                    .clicked()
                                {
                                    remove_id = Some(obj.object_id);
                                }
                            });
                        }

                        if let Some(id) = select_id {
                            plate.selected_object_id = Some(id);
                        }
                        if let Some(id) = clone_id {
                            let prev_len = plate.objects.len();
                            clone_object_on_plate(
                                &mut commands,
                                render_device.as_deref(),
                                &mut meshes,
                                &mut materials,
                                &mut plate,
                                &mut config,
                                &object_transforms,
                                id,
                                1,
                            );
                            let new_ids: Vec<u64> = plate.objects[prev_len..]
                                .iter()
                                .map(|o| o.object_id)
                                .collect();
                            if let Some(unmixed) = vis_state.plate_unmixed_toolpaths.get(&id).cloned()
                            {
                                for nid in &new_ids {
                                    vis_state
                                        .plate_unmixed_toolpaths
                                        .insert(*nid, unmixed.clone());
                                }
                            }
                            if let Some(mixed) = vis_state.plate_mixed_toolpaths.get(&id).cloned() {
                                for nid in &new_ids {
                                    vis_state.plate_mixed_toolpaths.insert(*nid, mixed.clone());
                                }
                            }
                            vis_state.dirty_planned_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        if let Some(id) = remove_id {
                            vis_state.plate_unmixed_toolpaths.remove(&id);
                            vis_state.plate_mixed_toolpaths.remove(&id);
                            remove_object_from_plate(&mut commands, &mut plate, id);
                            vis_state.dirty_planned_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                    }

                    let plate_has_multicolor_obj = plate
                        .objects
                        .iter()
                        .any(|o| !o.triangle_filament_slots.is_empty());
                    if !config.filament_slots.is_empty()
                        || plate_has_multicolor_obj
                        || config.tools.len() >= 2
                    {
                        ui.add_space(6.0);
                        render_3mf_filament_slot_mapping_section(
                            ui,
                            &mut config,
                            &mut vis_state,
                            &mut plate,
                            &mut meshes,
                            is_computing,
                            "project_setup_filament_slot_mapping",
                        );
                    }

                    // --- Selected Object Transform & Convex Hull Face Placement ---
                    if let Some(sel_id) = plate.selected_object_id {
                        if let Some(obj_idx) =
                            plate.objects.iter().position(|o| o.object_id == sel_id)
                        {
                            ui.add_space(8.0);
                            ui.separator();
                            let obj_name = plate.objects[obj_idx].name.clone();
                            let obj_entity = plate.objects[obj_idx].entity;
                            let obj_plate_idx = plate.objects[obj_idx].plate_index;
                            let hull_face_count = plate.objects[obj_idx].hull.faces.len();

                            ui.heading(format!("Selected: {obj_name}"));
                            if ui
                                .checkbox(
                                    &mut plate.objects[obj_idx].printable,
                                    "Printable (Include in Slice & G-code)",
                                )
                                .on_hover_text(
                                    "When unchecked, this object is excluded from slicing, toolpath preview, and G-code output (saved to .3mf as printable=0).",
                                )
                                .changed()
                            {
                                vis_state.dirty_planned_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }

                            // Per-Object Filament / Colour Mode & Optical CMYKW Selector
                            render_selected_object_filament_and_color_controls(
                                ui,
                                &mut plate.objects[obj_idx],
                                &mut config,
                                &mut vis_state,
                                &mut meshes,
                            );

                            // Move object between build plates
                            let mut move_to_plate: Option<usize> = None;
                            let mut move_to_new_plate = false;
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Assigned Plate:");
                                let cur_p_name = plate
                                    .plates
                                    .get(obj_plate_idx)
                                    .map(|p| p.name.clone())
                                    .unwrap_or_else(|| format!("Plate {}", obj_plate_idx + 1));
                                egui::ComboBox::from_id_source("selected_object_plate_combo")
                                    .selected_text(cur_p_name)
                                    .width(120.0)
                                    .show_ui(ui, |ui| {
                                        for (p_idx, p_info) in plate.plates.iter().enumerate() {
                                            let is_cur = p_idx == obj_plate_idx;
                                            if ui
                                                .selectable_label(is_cur, &p_info.name)
                                                .clicked()
                                                && !is_cur
                                            {
                                                move_to_plate = Some(p_idx);
                                            }
                                        }
                                    });
                                if render_icon_button(
                                    ui,
                                    UiIcon::BuildPlate,
                                    "Move to New Plate",
                                    true,
                                    true,
                                )
                                .on_hover_text(
                                    "Create a new build plate and move this object onto it",
                                )
                                .clicked()
                                {
                                    move_to_new_plate = true;
                                }
                            });
                            if move_to_new_plate {
                                let new_p_idx = plate.plates.len();
                                let new_p_name = format!("Plate {}", new_p_idx + 1);
                                plate.plates.push(BuildPlateInfo {
                                    name: new_p_name.clone(),
                                });
                                move_to_plate = Some(new_p_idx);
                            }
                            if let Some(target_p_idx) = move_to_plate {
                                plate.objects[obj_idx].plate_index = target_p_idx;
                                plate.active_plate_index = target_p_idx;
                                plate.selected_object_id = Some(sel_id);
                                plate.sync_plate_visibility(&mut object_transforms);
                                vis_state.dirty_planned_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                                let target_name = plate.active_plate_name().to_string();
                                plate.status_message =
                                    format!("Moved '{obj_name}' to {target_name}.");
                            }

                            if plate.clone_copies_count == 0 {
                                plate.clone_copies_count = 1;
                            }
                            let mut pending_clone_copies: Option<usize> = None;
                            ui.horizontal_wrapped(|ui| {
                                ui.label("Clone Object:");
                                if render_icon_button(ui, UiIcon::CloneObject, "+1 Clone", false, true)
                                    .on_hover_text(
                                        "Add 1 clone of the selected object to the build plate",
                                    )
                                    .clicked()
                                {
                                    pending_clone_copies = Some(1);
                                }
                                ui.add(
                                    egui::DragValue::new(&mut plate.clone_copies_count)
                                        .range(1..=64)
                                        .prefix("×"),
                                );
                                if render_icon_button(
                                    ui,
                                    UiIcon::CloneObject,
                                    "Add Clones",
                                    false,
                                    true,
                                )
                                .on_hover_text(
                                    "Add the specified number of clones of the selected object",
                                )
                                .clicked()
                                {
                                    pending_clone_copies = Some(plate.clone_copies_count as usize);
                                }
                            });
                            if let Some(copies) = pending_clone_copies {
                                let prev_len = plate.objects.len();
                                clone_object_on_plate(
                                    &mut commands,
                                    render_device.as_deref(),
                                    &mut meshes,
                                    &mut materials,
                                    &mut plate,
                                    &mut config,
                                    &object_transforms,
                                    sel_id,
                                    copies,
                                );
                                let new_ids: Vec<u64> = plate.objects[prev_len..]
                                    .iter()
                                    .map(|o| o.object_id)
                                    .collect();
                                if let Some(unmixed) =
                                    vis_state.plate_unmixed_toolpaths.get(&sel_id).cloned()
                                {
                                    for nid in &new_ids {
                                        vis_state
                                            .plate_unmixed_toolpaths
                                            .insert(*nid, unmixed.clone());
                                    }
                                }
                                if let Some(mixed) =
                                    vis_state.plate_mixed_toolpaths.get(&sel_id).cloned()
                                {
                                    for nid in &new_ids {
                                        vis_state
                                            .plate_mixed_toolpaths
                                            .insert(*nid, mixed.clone());
                                    }
                                }
                                vis_state.dirty_planned_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }

                            if let Ok((mut tf, _)) = object_transforms.get_mut(obj_entity) {
                                let (w_min, w_max) =
                                    plate.objects[obj_idx].compute_world_aabb(&tf);
                                let extents = (w_max - w_min).max(Vec3::splat(1.0e-4));
                                let (b_min, b_max) =
                                    plate.objects[obj_idx].compute_base_world_aabb(tf.rotation);
                                let base_extents = (b_max - b_min).max(Vec3::splat(1.0e-4));

                                let gcode_offset_xy = config.scene_to_gcode_offset().xy();
                                let mut bed_x = tf.translation.x + gcode_offset_xy.x;
                                let mut bed_y = tf.translation.y + gcode_offset_xy.y;
                                let mut changed = false;

                                // --- Scale by Proportion (%) or Bounding Box Dimension (mm) ---
                                let cur_scale = plate.objects[obj_idx].scale_factors;
                                let cur_scale_rot = plate.objects[obj_idx].scale_rotation;
                                let is_uniform_scale = (cur_scale.x - cur_scale.y).abs() < 1.0e-5
                                    && (cur_scale.y - cur_scale.z).abs() < 1.0e-5;
                                let rot_matches =
                                    is_uniform_scale || cur_scale_rot.angle_between(tf.rotation) < 1.0e-4;
                                let effective_scale = if rot_matches {
                                    cur_scale
                                } else {
                                    extents / base_extents
                                };

                                let mut scale_pct_x = effective_scale.x * 100.0;
                                let mut scale_pct_y = effective_scale.y * 100.0;
                                let mut scale_pct_z = effective_scale.z * 100.0;
                                let mut dim_mm_x = extents.x;
                                let mut dim_mm_y = extents.y;
                                let mut dim_mm_z = extents.z;
                                let mut pending_scale_change: Option<(Vec3, Quat)> = None;

                                ui.horizontal_wrapped(|ui| {
                                    ui.checkbox(
                                        &mut plate.objects[obj_idx].lock_scale_axes,
                                        "Lock Axes (Uniform Scale)",
                                    )
                                    .on_hover_text(
                                        "When checked, editing any X/Y/Z scale percentage or bounding-box dimension scales all three axes proportionally",
                                    );
                                    if ui
                                        .small_button("50%")
                                        .on_hover_text("Halve current object dimensions (×0.5)")
                                        .clicked()
                                    {
                                        pending_scale_change =
                                            Some((cur_scale * 0.5, cur_scale_rot));
                                    }
                                    if ui
                                        .small_button("200%")
                                        .on_hover_text("Double current object dimensions (×2.0)")
                                        .clicked()
                                    {
                                        pending_scale_change =
                                            Some((cur_scale * 2.0, cur_scale_rot));
                                    }
                                    if ui
                                        .small_button("in→mm")
                                        .on_hover_text(
                                            "Convert inches to millimeters (scale by 2540% / ×25.4)",
                                        )
                                        .clicked()
                                    {
                                        pending_scale_change =
                                            Some((cur_scale * 25.4, cur_scale_rot));
                                    }
                                    if render_icon_button(
                                        ui,
                                        UiIcon::ResetTransform,
                                        "100%",
                                        false,
                                        (cur_scale - Vec3::ONE).abs().max_element() > 1.0e-4,
                                    )
                                    .on_hover_text("Reset scale to original 100% imported dimensions")
                                    .clicked()
                                    {
                                        pending_scale_change =
                                            Some((Vec3::ONE, Quat::IDENTITY));
                                    }
                                });

                                let lock_axes = plate.objects[obj_idx].lock_scale_axes;
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Scale (%):");
                                    ui.label("X");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut scale_pct_x)
                                                .speed(0.5)
                                                .range(0.1..=100_000.0)
                                                .max_decimals(2)
                                                .suffix("%"),
                                        )
                                        .changed()
                                    {
                                        let new_fx = (scale_pct_x / 100.0).clamp(0.001, 1000.0);
                                        if lock_axes {
                                            let ratio = new_fx / effective_scale.x.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.x = new_fx;
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                    ui.label("Y");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut scale_pct_y)
                                                .speed(0.5)
                                                .range(0.1..=100_000.0)
                                                .max_decimals(2)
                                                .suffix("%"),
                                        )
                                        .changed()
                                    {
                                        let new_fy = (scale_pct_y / 100.0).clamp(0.001, 1000.0);
                                        if lock_axes {
                                            let ratio = new_fy / effective_scale.y.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.y = new_fy;
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                    ui.label("Z");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut scale_pct_z)
                                                .speed(0.5)
                                                .range(0.1..=100_000.0)
                                                .max_decimals(2)
                                                .suffix("%"),
                                        )
                                        .changed()
                                    {
                                        let new_fz = (scale_pct_z / 100.0).clamp(0.001, 1000.0);
                                        if lock_axes {
                                            let ratio = new_fz / effective_scale.z.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.z = new_fz;
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                });

                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Size (mm):");
                                    ui.label("X");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut dim_mm_x)
                                                .speed(0.25)
                                                .range(0.05..=100_000.0)
                                                .max_decimals(2)
                                                .suffix(" mm"),
                                        )
                                        .changed()
                                    {
                                        let target_x = dim_mm_x.clamp(0.05, 100_000.0);
                                        if lock_axes {
                                            let ratio = target_x / extents.x.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.x = (target_x / base_extents.x).clamp(0.001, 1000.0);
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                    ui.label("Y");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut dim_mm_y)
                                                .speed(0.25)
                                                .range(0.05..=100_000.0)
                                                .max_decimals(2)
                                                .suffix(" mm"),
                                        )
                                        .changed()
                                    {
                                        let target_y = dim_mm_y.clamp(0.05, 100_000.0);
                                        if lock_axes {
                                            let ratio = target_y / extents.y.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.y = (target_y / base_extents.y).clamp(0.001, 1000.0);
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                    ui.label("Z");
                                    if ui
                                        .add(
                                            egui::DragValue::new(&mut dim_mm_z)
                                                .speed(0.25)
                                                .range(0.05..=100_000.0)
                                                .max_decimals(2)
                                                .suffix(" mm"),
                                        )
                                        .changed()
                                    {
                                        let target_z = dim_mm_z.clamp(0.05, 100_000.0);
                                        if lock_axes {
                                            let ratio = target_z / extents.z.max(1.0e-5);
                                            pending_scale_change =
                                                Some((cur_scale * ratio, cur_scale_rot));
                                        } else {
                                            let mut s = effective_scale;
                                            s.z = (target_z / base_extents.z).clamp(0.001, 1000.0);
                                            pending_scale_change = Some((s, tf.rotation));
                                        }
                                    }
                                });

                                if let Some((new_scale, new_rot)) = pending_scale_change {
                                    if plate.objects[obj_idx].apply_scale_in_frame(
                                        &mut commands,
                                        render_device.as_deref(),
                                        &mut meshes,
                                        &mut config,
                                        &mut vis_state,
                                        new_scale,
                                        new_rot,
                                    ) {
                                        changed = true;
                                        let (n_min, n_max) =
                                            plate.objects[obj_idx].compute_world_aabb(&tf);
                                        let n_ext = n_max - n_min;
                                        let s = plate.objects[obj_idx].scale_factors;
                                        plate.status_message = format!(
                                            "Scaled '{}' to {:.1}% × {:.1}% × {:.1}% ({:.2} × {:.2} × {:.2} mm).",
                                            obj_name,
                                            s.x * 100.0,
                                            s.y * 100.0,
                                            s.z * 100.0,
                                            n_ext.x,
                                            n_ext.y,
                                            n_ext.z
                                        );
                                    }
                                }
                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Bed X (mm):");
                                    if ui
                                        .add(egui::DragValue::new(&mut bed_x).speed(0.5))
                                        .changed()
                                    {
                                        tf.translation.x = bed_x - gcode_offset_xy.x;
                                        changed = true;
                                    }
                                    ui.label("Y (mm):");
                                    if ui
                                        .add(egui::DragValue::new(&mut bed_y).speed(0.5))
                                        .changed()
                                    {
                                        tf.translation.y = bed_y - gcode_offset_xy.y;
                                        changed = true;
                                    }
                                    if render_icon_button(
                                        ui,
                                        UiIcon::CenterOnBed,
                                        "Center",
                                        true,
                                        true,
                                    )
                                    .on_hover_text("Move object to the center of the build plate")
                                    .clicked()
                                    {
                                        tf.translation.x = 0.0;
                                        tf.translation.y = 0.0;
                                        changed = true;
                                    }
                                });

                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Quick Rotate:");
                                    if render_icon_button(
                                        ui,
                                        UiIcon::RotateObject,
                                        "X +90°",
                                        false,
                                        true,
                                    )
                                    .clicked()
                                    {
                                        tf.rotation = (Quat::from_rotation_x(
                                            std::f32::consts::FRAC_PI_2,
                                        ) * tf.rotation)
                                            .normalize();
                                        changed = true;
                                    }
                                    if render_icon_button(
                                        ui,
                                        UiIcon::RotateObject,
                                        "Y +90°",
                                        false,
                                        true,
                                    )
                                    .clicked()
                                    {
                                        tf.rotation = (Quat::from_rotation_y(
                                            std::f32::consts::FRAC_PI_2,
                                        ) * tf.rotation)
                                            .normalize();
                                        changed = true;
                                    }
                                    if render_icon_button(
                                        ui,
                                        UiIcon::RotateObject,
                                        "Z +45°",
                                        false,
                                        true,
                                    )
                                    .clicked()
                                    {
                                        tf.rotation = (Quat::from_rotation_z(
                                            std::f32::consts::FRAC_PI_4,
                                        ) * tf.rotation)
                                            .normalize();
                                        changed = true;
                                    }
                                    if render_icon_button(
                                        ui,
                                        UiIcon::ResetTransform,
                                        "Reset",
                                        false,
                                        true,
                                    )
                                    .clicked()
                                    {
                                        tf.rotation = Quat::IDENTITY;
                                        changed = true;
                                    }
                                });

                                // Simplified Convex Hull (<= 27 faces) quick selector
                                ui.collapsing(
                                    format!(
                                        "Place on Convex Hull Face ({hull_face_count}/27 faces)"
                                    ),
                                    |ui| {
                                        ui.small(
                                            "Tip: Select 'Place on Face (Hull)' in the top toolbar to click a hull face directly in the 3D viewport, or pick a support face below:",
                                        );
                                        let mut chosen_normal: Option<(usize, Vec3)> = None;
                                        for face in &plate.objects[obj_idx].hull.faces {
                                            let label = format!(
                                                "Face #{:02} — Area: {:.1} mm²  n=({:.2}, {:.2}, {:.2})",
                                                face.face_index + 1,
                                                face.area,
                                                face.normal.x,
                                                face.normal.y,
                                                face.normal.z
                                            );
                                            if render_icon_button(
                                                ui,
                                                UiIcon::LayOnFace,
                                                label,
                                                true,
                                                true,
                                            )
                                            .clicked()
                                            {
                                                chosen_normal = Some((face.face_index, face.normal));
                                            }
                                        }
                                        if let Some((f_idx, normal)) = chosen_normal {
                                            tf.rotation =
                                                rotation_to_place_face_on_bed(tf.rotation, normal);
                                            changed = true;
                                            plate.status_message = format!(
                                                "Oriented '{}' onto hull face #{}.",
                                                obj_name,
                                                f_idx + 1
                                            );
                                        }
                                    },
                                );

                                if changed {
                                    snap_object_to_bed(&plate.objects[obj_idx], &mut tf);
                                    vis_state.dirty_planned_gcode = true;
                                }
                            }

                            // --- Per-Object Tangency & Orthogonality Hints ---
                            ui.add_space(8.0);
                            ui.separator();
                            let patch_count = plate.objects[obj_idx].surface_patches.len();
                            let obj_has_brep = plate.objects[obj_idx].has_brep;
                            let obj_local_center = 0.5
                                * (plate.objects[obj_idx].local_min
                                    + plate.objects[obj_idx].local_max);
                            let surface_hint_count = config
                                .eikonal
                                .surface_hints
                                .iter()
                                .filter(|h| h.object_id == sel_id)
                                .count();
                            let point_hint_count = config
                                .eikonal
                                .point_hints
                                .iter()
                                .filter(|h| h.object_id == sel_id)
                                .count();

                            egui::CollapsingHeader::new(
                                egui::RichText::new(format!(
                                    "Orientation Hints ({} surface, {} point)",
                                    surface_hint_count, point_hint_count
                                ))
                                .strong(),
                            )
                            .id_source("tangency_orthogonality_hints_section")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.small(if obj_has_brep {
                                    format!(
                                        "Soft constraints for Eikonal isosurface normal (∇τ). Selectable STEP B-Rep faces: {patch_count}. Never overrides tool clearance or angle limits."
                                    )
                                } else {
                                    format!(
                                        "Soft constraints for Eikonal isosurface normal (∇τ). Contiguous near-planar mesh patches: {patch_count}. Never overrides tool clearance or angle limits."
                                    )
                                });

                                ui.horizontal_wrapped(|ui| {
                                    ui.label("Placement Tool:");
                                    for mode in [
                                        ToolMode::SurfaceHintOrthogonal,
                                        ToolMode::SurfaceHintTangent,
                                        ToolMode::SurfaceHintConformalSkin,
                                        ToolMode::PointHint,
                                    ] {
                                        let active = plate.tool_mode == mode;
                                        if render_delineated_icon_pill_tab(
                                            ui,
                                            active,
                                            Some(UiIcon::for_tool_mode(mode)),
                                            mode.label(),
                                        )
                                        .clicked()
                                        {
                                            plate.tool_mode =
                                                if active { ToolMode::Select } else { mode };
                                        }
                                    }
                                });

                                // 1. Surface Face Hints List
                                ui.separator();
                                egui::CollapsingHeader::new(
                                    egui::RichText::new(format!(
                                        "Surface Face Hints ({surface_hint_count})"
                                    ))
                                    .strong(),
                                )
                                .id_source("surface_face_hints_list")
                                .default_open(true)
                                .show(ui, |ui| {
                                    let mut surf_hints_dirty = false;
                                    if surface_hint_count > 0
                                        && render_icon_button(
                                            ui,
                                            UiIcon::DeleteObject,
                                            "Clear Surface Hints",
                                            true,
                                            true,
                                        )
                                        .clicked()
                                    {
                                        config
                                            .eikonal
                                            .surface_hints
                                            .retain(|h| h.object_id != sel_id);
                                        plate.selected_surface_hint_patch = None;
                                        surf_hints_dirty = true;
                                    }

                                    if surface_hint_count == 0 {
                                        ui.small(
                                            "Click faces in the viewport using 'Hint: Face ⊥ Normal' (cyan), 'Hint: Face ∥ Normal' (gold), or 'Hint: Conformal Skin (SDF Layers)' (emerald).",
                                        );
                                    } else {
                                        let mut remove_surf_idx: Option<usize> = None;
                                        for (idx, sh) in
                                            config.eikonal.surface_hints.iter_mut().enumerate()
                                        {
                                            if sh.object_id != sel_id {
                                                continue;
                                            }
                                            let is_sel = plate.selected_surface_hint_patch
                                                == Some(sh.patch_id);
                                            let row_resp = ui.horizontal_wrapped(|ui| {
                                                let kind_str = if obj_has_brep {
                                                    "STEP Face"
                                                } else {
                                                    "Patch"
                                                };
                                                if ui
                                                    .selectable_label(
                                                        is_sel,
                                                        format!("{kind_str} #{}", sh.patch_id + 1),
                                                    )
                                                    .on_hover_text(
                                                        "Click to select/deselect this surface hint and toggle its 3D arrows",
                                                    )
                                                    .clicked()
                                                {
                                                    plate.selected_surface_hint_patch = if is_sel {
                                                        None
                                                    } else {
                                                        Some(sh.patch_id)
                                                    };
                                                }
                                                let hint_icon = match sh.mode {
                                                    SurfaceHintMode::NormalOrthogonalToSurface => {
                                                        UiIcon::SurfaceHintOrthogonal
                                                    }
                                                    SurfaceHintMode::NormalTangentToSurface => {
                                                        UiIcon::SurfaceHintTangent
                                                    }
                                                    SurfaceHintMode::ConformalSurfaceSkin => {
                                                        UiIcon::SurfaceHintConformalSkin
                                                    }
                                                };
                                                if render_icon_button(
                                                    ui,
                                                    hint_icon,
                                                    sh.mode.short_label(),
                                                    true,
                                                    true,
                                                )
                                                .on_hover_text("Click to cycle between Normal ⊥ Surface, Normal ∥ Surface, and Conformal Skin (SDF Layers)")
                                                .clicked()
                                                {
                                                    sh.mode = match sh.mode {
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
                                                    surf_hints_dirty = true;
                                                }
                                                if sh.mode == SurfaceHintMode::ConformalSurfaceSkin {
                                                    if ui
                                                        .add(
                                                            egui::DragValue::new(
                                                                &mut sh.skin_layer_count,
                                                            )
                                                            .speed(0.2)
                                                            .range(0..=12)
                                                            .prefix("Layers: "),
                                                        )
                                                        .on_hover_text("Number of conformal SDF skin layers to erode and slice along this surface (set to 0 to exclude this patch from automatic conformal skin)")
                                                        .changed()
                                                    {
                                                        surf_hints_dirty = true;
                                                    }
                                                } else {
                                                    if ui
                                                        .add(
                                                            egui::DragValue::new(&mut sh.influence_depth_mm)
                                                                .speed(0.2)
                                                                .range(0.5..=50.0)
                                                                .prefix("d="),
                                                        )
                                                        .changed()
                                                    {
                                                        surf_hints_dirty = true;
                                                    }
                                                    if ui
                                                        .add(
                                                            egui::DragValue::new(&mut sh.weight)
                                                                .speed(0.02)
                                                                .range(0.05..=1.0)
                                                                .prefix("w="),
                                                        )
                                                        .changed()
                                                    {
                                                        surf_hints_dirty = true;
                                                    }
                                                }
                                                if render_icon_button(
                                                    ui,
                                                    UiIcon::DeleteObject,
                                                    "",
                                                    true,
                                                    true,
                                                )
                                                .on_hover_text("Remove this surface hint")
                                                .clicked()
                                                {
                                                    remove_surf_idx = Some(idx);
                                                }
                                            });
                                            if row_resp.response.hovered()
                                                || ui.rect_contains_pointer(row_resp.response.rect)
                                            {
                                                plate.ui_hovered_surface_hint_patch =
                                                    Some(sh.patch_id);
                                            }
                                        }
                                        if let Some(idx) = remove_surf_idx {
                                            let removed_patch =
                                                config.eikonal.surface_hints[idx].patch_id;
                                            config.eikonal.surface_hints.remove(idx);
                                            if plate.selected_surface_hint_patch
                                                == Some(removed_patch)
                                            {
                                                plate.selected_surface_hint_patch = None;
                                            }
                                            surf_hints_dirty = true;
                                        }
                                    }
                                    if surf_hints_dirty {
                                        sync_object_hints_to_backend(&config, &plate, sel_id);
                                        vis_state.dirty_selected_layers = true;
                                        vis_state.dirty_toolpaths = true;
                                    }
                                });

                                // 2. 3D Point Hints in Space
                                ui.separator();
                                egui::CollapsingHeader::new(
                                    egui::RichText::new(format!(
                                        "3D Point Hints in Space ({point_hint_count})"
                                    ))
                                    .strong(),
                                )
                                .id_source("point_hints_in_space_list")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        if render_icon_button(
                                            ui,
                                            UiIcon::AddPlate,
                                            "Add at Center",
                                            true,
                                            true,
                                        )
                                        .clicked()
                                        {
                                            let next_id = config
                                                .eikonal
                                                .point_hints
                                                .iter()
                                                .map(|h| h.id)
                                                .max()
                                                .unwrap_or(0)
                                                + 1;
                                            let def_radius =
                                                config.eikonal.default_point_hint_radius_mm;
                                            let def_weight = config.eikonal.default_hint_weight;
                                            config.eikonal.point_hints.push(PointNormalHint {
                                                id: next_id,
                                                object_id: sel_id,
                                                local_position: [
                                                    obj_local_center.x,
                                                    obj_local_center.y,
                                                    obj_local_center.z,
                                                ],
                                                local_direction: [0.0, 0.0, 1.0],
                                                influence_radius_mm: def_radius,
                                                weight: def_weight,
                                            });
                                            plate.selected_point_hint_id = Some(next_id);
                                        }
                                        if point_hint_count > 0
                                            && render_icon_button(
                                                ui,
                                                UiIcon::DeleteObject,
                                                "Clear Point Hints",
                                                true,
                                                true,
                                            )
                                            .clicked()
                                        {
                                            config
                                                .eikonal
                                                .point_hints
                                                .retain(|h| h.object_id != sel_id);
                                            plate.selected_point_hint_id = None;
                                        }
                                    });

                                    if point_hint_count == 0 {
                                        ui.small(
                                            "Select '3D Point Hint' and click on the 2D slice plane (or object surface) in the viewport, then drag the arrow tip to aim.",
                                        );
                                    } else {
                                        let mut remove_pt_idx: Option<usize> = None;
                                        for (idx, ph) in
                                            config.eikonal.point_hints.iter_mut().enumerate()
                                        {
                                            if ph.object_id != sel_id {
                                                continue;
                                            }
                                            let is_sel = plate.selected_point_hint_id == Some(ph.id);
                                            let group_resp = ui.group(|ui| {
                                                ui.horizontal_wrapped(|ui| {
                                                    if ui
                                                        .selectable_label(
                                                            is_sel,
                                                            format!("Point Hint #{}", ph.id),
                                                        )
                                                        .on_hover_text(
                                                            "Click to select/deselect this point hint and toggle its 3D arrow",
                                                        )
                                                        .clicked()
                                                    {
                                                        plate.selected_point_hint_id = if is_sel {
                                                            None
                                                        } else {
                                                            Some(ph.id)
                                                        };
                                                    }
                                                    ui.add(
                                                        egui::DragValue::new(
                                                            &mut ph.influence_radius_mm,
                                                        )
                                                        .speed(0.25)
                                                        .range(1.0..=60.0)
                                                        .prefix("r=")
                                                        .suffix("mm"),
                                                    );
                                                    ui.add(
                                                        egui::DragValue::new(&mut ph.weight)
                                                            .speed(0.02)
                                                            .range(0.05..=1.0)
                                                            .prefix("w="),
                                                    );
                                                    if render_icon_button(
                                                        ui,
                                                        UiIcon::DeleteObject,
                                                        "",
                                                        true,
                                                        true,
                                                    )
                                                    .on_hover_text("Remove this 3D point hint")
                                                    .clicked()
                                                    {
                                                        remove_pt_idx = Some(idx);
                                                    }
                                                });

                                                ui.horizontal_wrapped(|ui| {
                                                    ui.small("Local Pos:");
                                                    ui.add(
                                                        egui::DragValue::new(
                                                            &mut ph.local_position[0],
                                                        )
                                                        .speed(0.5)
                                                        .prefix("X:"),
                                                    );
                                                    ui.add(
                                                        egui::DragValue::new(
                                                            &mut ph.local_position[1],
                                                        )
                                                        .speed(0.5)
                                                        .prefix("Y:"),
                                                    );
                                                    ui.add(
                                                        egui::DragValue::new(
                                                            &mut ph.local_position[2],
                                                        )
                                                        .speed(0.5)
                                                        .prefix("Z:"),
                                                    );
                                                });

                                                let dir = Vec3::from_array(ph.local_direction)
                                                    .try_normalize()
                                                    .unwrap_or(Vec3::Z);
                                                let mut elev_deg =
                                                    dir.z.clamp(-1.0, 1.0).asin().to_degrees();
                                                let mut azim_deg =
                                                    dir.y.atan2(dir.x).to_degrees();
                                                let mut dir_changed = false;
                                                ui.horizontal_wrapped(|ui| {
                                                    ui.small("Normal Dir:");
                                                    dir_changed |= ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut elev_deg,
                                                                -90.0..=90.0,
                                                            )
                                                            .text("Elev°"),
                                                        )
                                                        .changed();
                                                    dir_changed |= ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut azim_deg,
                                                                -180.0..=180.0,
                                                            )
                                                            .text("Azim°"),
                                                        )
                                                        .changed();
                                                });
                                                if dir_changed {
                                                    let el = elev_deg.to_radians();
                                                    let az = azim_deg.to_radians();
                                                    let new_dir = Vec3::new(
                                                        el.cos() * az.cos(),
                                                        el.cos() * az.sin(),
                                                        el.sin(),
                                                    )
                                                    .normalize_or_zero();
                                                    if new_dir.length_squared() > 0.5 {
                                                        ph.local_direction = new_dir.to_array();
                                                    }
                                                }
                                            });
                                            if group_resp.response.hovered()
                                                || ui.rect_contains_pointer(group_resp.response.rect)
                                            {
                                                plate.ui_hovered_point_hint_id = Some(ph.id);
                                            }
                                        }
                                        if let Some(idx) = remove_pt_idx {
                                            let removed_id = config.eikonal.point_hints[idx].id;
                                            config.eikonal.point_hints.remove(idx);
                                            if plate.selected_point_hint_id == Some(removed_id) {
                                                plate.selected_point_hint_id = None;
                                            }
                                        }
                                    }
                                });
                            });
                        }
                    }

                    // --- Auto-Arrangement Settings ---
                    ui.add_space(8.0);
                    ui.separator();
                    egui::CollapsingHeader::new(
                        egui::RichText::new("Auto-Arrangement Settings").strong(),
                    )
                    .id_source("project_auto_arrangement_settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::Slider::new(
                                    &mut config.arrangement.part_clearance_mm,
                                    0.5..=25.0,
                                )
                                .text("Part Clearance (mm)"),
                            );
                            param_help_button(ui, "arrangement.margins_mm");
                        });
                        ui.add(
                            egui::Slider::new(
                                &mut config.arrangement.bed_edge_margin_mm,
                                0.0..=25.0,
                            )
                            .text("Bed Margin (mm)"),
                        );
                        if ui.button("Auto-Arrange Plate Now (Spiral)").clicked() {
                            run_auto_arrange_on_plate(&mut plate, &config, &mut object_transforms);
                        }
                    });

                    // --- Telemetry & 3MF Import Notices ---
                    ui.add_space(8.0);
                    ui.separator();
                    ui.label(egui::RichText::new("Telemetry").strong());
                    ui.small(format!(
                        "Total Scene: {} vertices | {} triangles",
                        plate.total_vertices(),
                        plate.total_triangles()
                    ));
                    ui.small(&plate.status_message);

                    if !plate.import_notices.is_empty() {
                        ui.collapsing(
                            format!("Import Notices ({})", plate.import_notices.len()),
                            |ui| {
                                for notice in &plate.import_notices {
                                    ui.small(format!("• {notice}"));
                                }
                            },
                        );
                    }

                    if !plate.import_errors.is_empty() {
                        ui.collapsing(
                            format!("Import Errors ({})", plate.import_errors.len()),
                            |ui| {
                                for err in &plate.import_errors {
                                    ui.colored_label(egui::Color32::LIGHT_RED, err);
                                }
                            },
                        );
                    }
                } else {
                    // ================================================================
                    // TAB 2: CONFIGURATION & SAVED PRESETS (Printer | Tool | Filament | Process)
                    // (Sub-tabs and Preset Selector Bar are pinned above the ScrollArea)
                    // ================================================================
                    match preset_lib.config_subtab {
                        PresetCategory::Printer => {
                            render_printer_config_tab(
                                ui,
                                &mut config,
                                &mut vis_state,
                                &mut moonraker_state,
                                &mut plate.status_message,
                                is_computing,
                            );
                        }
                        PresetCategory::Tool => {
                            render_tool_config_tab(
                                ui,
                                &mut config,
                                &mut vis_state,
                                &mut preset_lib,
                                is_computing,
                            );
                        }
                        PresetCategory::Filament => {
                            render_filament_config_tab(
                                ui,
                                &mut config,
                                &mut vis_state,
                                &mut calib_state,
                                &mut preset_lib,
                                is_computing,
                            );
                        }
                        PresetCategory::Process => {
                            render_process_config_tab(
                                ui,
                                &mut config,
                                &mut vis_state,
                                &mut plate,
                                selected_obj_info.as_ref(),
                                is_computing,
                                has_eikonal,
                            );
                        }
                    }
                    }
                });
                outer_ui.allocate_rect(panel_rect, egui::Sense::hover());
            });
    }

    // ========================================================================
    // 3. Right Panel: View & Inspection (Never Marks Presets Dirty)
    // ========================================================================
    let right_default_w = (screen_width * 0.20).clamp(230.0, 280.0);
    let right_max_w = (screen_width * 0.42).max(320.0);
    if preset_lib.show_view_panel {
        egui::SidePanel::right("slicer_right_panel")
            .default_width(right_default_w)
            .width_range(200.0..=right_max_w)
            .resizable(true)
            .show(ctx, |outer_ui| {
                let panel_rect = outer_ui.max_rect();
                let mut child_ui = outer_ui.child_ui(panel_rect, *outer_ui.layout(), None);
                child_ui.set_clip_rect(panel_rect);
                let ui = &mut child_ui;
                let panel_w = panel_rect.width();
                ui.spacing_mut().slider_width = (panel_w * 0.34).clamp(62.0, 135.0);
                ui.spacing_mut().combo_width = (panel_w * 0.42).clamp(85.0, 175.0);

                ui.horizontal(|ui| {
                    ui.heading("View & Inspection");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("▶")
                            .on_hover_text("Collapse right View & Inspection panel")
                            .clicked()
                        {
                            preset_lib.show_view_panel = false;
                        }
                    });
                });
                ui.small(
                    egui::RichText::new(
                        "Visualization & inspection controls (never modify saved presets)",
                    )
                    .color(egui::Color32::GRAY),
                );
                ui.separator();

                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(ui.available_width());
                        render_view_and_inspection_panel(
                            ui,
                            &mut config,
                            &mut vis_state,
                            &mut plate,
                            selected_obj_info.as_ref(),
                            is_computing,
                            has_eikonal,
                        );
                    });
                outer_ui.allocate_rect(panel_rect, egui::Sense::hover());
            });
    }

    // Floating viewport-edge expand tabs when Left or Right side panels are collapsed
    let top_bar_bottom_y = top_bar_resp.response.rect.max.y + 8.0;
    if !preset_lib.show_left_panel {
        egui::Area::new(egui::Id::new("collapsed_left_panel_expand_tab"))
            .fixed_pos(egui::pos2(8.0, top_bar_bottom_y))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::symmetric(6.0, 4.0))
                    .show(ui, |ui| {
                        let lbl = format!("▶ Setup ({}) / Config", plate.objects.len());
                        if ui
                            .button(egui::RichText::new(lbl).strong())
                            .on_hover_text("Expand left Project Setup & Configuration side panel")
                            .clicked()
                        {
                            preset_lib.show_left_panel = true;
                        }
                    });
            });
    }
    if !preset_lib.show_view_panel {
        egui::Area::new(egui::Id::new("collapsed_right_panel_expand_tab"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, top_bar_bottom_y))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::symmetric(6.0, 4.0))
                    .show(ui, |ui| {
                        if ui
                            .button(egui::RichText::new("◀ View & Inspect").strong())
                            .on_hover_text("Expand right View & Inspection side panel")
                            .clicked()
                        {
                            preset_lib.show_view_panel = true;
                        }
                    });
            });
    }

    // Top-right viewport HUD legend overlay when the View & Inspection panel is closed
    let show_hud_toolpaths = vis_state.show_toolpaths
        && (vis_state.cached_toolpaths.is_some() || vis_state.cached_planned_gcode.is_some());
    let show_hud_layers =
        vis_state.show_selected_layers && vis_state.cached_selected_layers.is_some();
    if !preset_lib.show_view_panel
        && (plate.selected_object().is_some() || vis_state.cached_planned_gcode.is_some())
        && (show_hud_toolpaths || show_hud_layers)
    {
        egui::Window::new("Viewport Color Legend")
            .id(egui::Id::new("viewport_parameter_color_legend_hud"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-14.0, 44.0))
            .resizable(false)
            .collapsible(true)
            .default_width(310.0)
            .show(ctx, |ui| {
                ui.set_min_width(290.0);
                ui.set_max_width(330.0);
                if show_hud_toolpaths {
                    if vis_state.toolpath_color_mode != ToolpathColorMode::ByFeatureType {
                        render_toolpath_color_legend(ui, &vis_state, &config);
                        ui.add_space(4.0);
                    }
                    render_interactive_toolpath_line_types_legend(ui, &mut vis_state);
                }
                if show_hud_toolpaths && show_hud_layers {
                    ui.add_space(4.0);
                    ui.separator();
                }
                if show_hud_layers {
                    render_selected_layers_color_legend(ui, &vis_state, &config);
                }
                ui.add_space(4.0);
                ui.separator();
                let hud_obj_tf = vis_state
                    .last_surface_color_transform
                    .unwrap_or(Transform::IDENTITY);
                render_layer_and_move_scrub_controls(
                    ui,
                    &mut vis_state,
                    &mut config,
                    hud_obj_tf,
                    true,
                    "hud_viewport_scrub",
                );
            });
    }

    // 4. Floating Calibration Utilities Window (EM + Die Swell, Ellis PA, Transient Melt-Zone Pressure)
    render_calibration_utilities_window(
        ctx,
        &mut calib_state,
        &mut config,
        &mut vis_state,
        &mut plate,
        &mut moonraker_state,
        is_io_busy,
        is_computing,
    );
}

/// Initializes streaming upload tracking state and spawns a non-blocking network upload task.
fn spawn_network_gcode_upload(
    moonraker_state: &mut MoonrakerUiState,
    net_cfg: crate::config::MoonrakerNetworkConfig,
    upload_filename: String,
    gcode_bytes: String,
    start_print: bool,
    status_message: &mut String,
) {
    *status_message = if start_print {
        format!(
            "Uploading '{upload_filename}' to {} and starting print...",
            net_cfg.printer_uri.trim()
        )
    } else {
        format!(
            "Uploading '{upload_filename}' to {}...",
            net_cfg.printer_uri.trim()
        )
    };
    moonraker_state.upload_start_print = start_print;
    moonraker_state.last_upload_result = None;
    moonraker_state
        .upload_progress
        .reset(gcode_bytes.len() as u64, status_message.as_str());
    let prog = moonraker_state.upload_progress.clone();
    let task = IoTaskPool::get().spawn(async move {
        upload_gcode_to_moonraker_with_progress(
            &net_cfg,
            &upload_filename,
            gcode_bytes,
            start_print,
            Some(&prog),
        )
    });
    moonraker_state.pending_upload_task = Some(task);
}

/// Renders a compact network printer status monitor (state badge, active filename, progress bar,
/// elapsed/ETA, refresh button, and "Web UI ↗" browser launcher) in the top workflow bar.
fn render_moonraker_toolbar_status_strip(
    ui: &mut egui::Ui,
    config: &SlicerConfig,
    moonraker_state: &mut MoonrakerUiState,
    status_message: &mut String,
) {
    let proto_label = config.network.protocol.short_label();
    ui.label(egui::RichText::new("Printer:").strong());
    if let Some(snap) = moonraker_state.active_upload_progress() {
        ui.ctx().request_repaint();
        ui.label(
            egui::RichText::new("[Uploading]")
                .strong()
                .color(egui::Color32::from_rgb(120, 205, 250)),
        )
        .on_hover_text(&snap.stage);
        ui.spinner();
        ui.add(
            egui::ProgressBar::new(snap.fraction.clamp(0.0, 1.0))
                .desired_width(165.0)
                .animate(true)
                .text(snap.bar_text()),
        )
        .on_hover_text(&snap.stage);
    } else {
        match moonraker_state.last_status {
            Some(Ok(ref st)) => {
                let state_lower = st.state.to_ascii_lowercase();
                let (badge_text, badge_color) = match state_lower.as_str() {
                    "printing" => ("Printing", egui::Color32::from_rgb(95, 225, 145)),
                    "paused" => ("Paused", egui::Color32::from_rgb(245, 195, 85)),
                    "complete" => ("Complete", egui::Color32::from_rgb(110, 215, 235)),
                    "standby" | "ready" => ("Standby", egui::Color32::from_rgb(175, 195, 215)),
                    "cancelled" => ("Cancelled", egui::Color32::from_rgb(230, 155, 110)),
                    "error" => ("Error", egui::Color32::from_rgb(245, 105, 105)),
                    _ => (st.state.as_str(), egui::Color32::from_rgb(185, 195, 210)),
                };
                let hover_msg = if st.message.is_empty() {
                    format!("{proto_label} printer state: {}", st.state)
                } else {
                    format!("{proto_label} printer state: {} — {}", st.state, st.message)
                };
                ui.label(
                    egui::RichText::new(format!("[{badge_text}]"))
                        .strong()
                        .color(badge_color),
                )
                .on_hover_text(hover_msg);

                if !st.filename.is_empty() {
                    ui.small(
                        egui::RichText::new(&st.filename)
                            .monospace()
                            .color(egui::Color32::from_gray(225)),
                    )
                    .on_hover_text(format!("Current G-code file active on {proto_label}"));
                }

                if st.is_printing_or_paused()
                    || state_lower == "complete"
                    || st.progress > 0.001
                {
                    let pct_text = if let Some(eta_str) = st.formatted_eta() {
                        format!("{:.0}% (ETA {eta_str})", (st.progress * 100.0).clamp(0.0, 100.0))
                    } else {
                        format!("{:.0}%", (st.progress * 100.0).clamp(0.0, 100.0))
                    };
                    ui.add(
                        egui::ProgressBar::new(st.progress.clamp(0.0, 1.0))
                            .desired_width(145.0)
                            .text(pct_text),
                    )
                    .on_hover_text(format!(
                        "Elapsed: {} | Progress: {:.1}%",
                        st.formatted_elapsed(),
                        st.progress * 100.0
                    ));
                }
            }
            Some(Err(ref err)) => {
                ui.label(
                    egui::RichText::new("[Offline]")
                        .strong()
                        .color(egui::Color32::from_rgb(235, 115, 115)),
                )
                .on_hover_text(err);
            }
            None => {
                if moonraker_state.pending_status_task.is_some() {
                    ui.ctx().request_repaint();
                    ui.spinner();
                    ui.small(egui::RichText::new("Polling...").color(egui::Color32::GRAY));
                } else {
                    ui.small(egui::RichText::new("Idle").color(egui::Color32::GRAY));
                }
            }
        }
    }

    if ui
        .add_enabled(
            moonraker_state.pending_status_task.is_none(),
            egui::Button::new("↻").small(),
        )
        .on_hover_text(format!("Refresh {proto_label} printer status now"))
        .clicked()
    {
        let net_cfg = config.network.clone();
        let cached_meta = moonraker_state.cached_metadata_estimate.clone();
        moonraker_state.last_poll_instant = Some(std::time::Instant::now());
        moonraker_state.pending_status_task = Some(IoTaskPool::get().spawn(async move {
            query_moonraker_print_status(&net_cfg, cached_meta)
        }));
    }

    let resolved_web_url = resolve_printer_web_ui_url(&config.network)
        .unwrap_or_else(|_| config.network.printer_uri.clone());
    if ui
        .small_button("Web UI ↗")
        .on_hover_text(format!(
            "Open printer Web UI ({resolved_web_url}) in your system browser"
        ))
        .clicked()
    {
        match open_printer_web_ui_in_browser(&config.network) {
            Ok(url) => {
                *status_message = format!("Opened printer Web UI ({url}) in system browser.");
            }
            Err(err) => {
                *status_message = err;
            }
        }
    }
}

/// Renders the `Printer` configuration sub-tab:
/// Print Bed Geometry & Origin, 5/6-Axis Machine Kinematics & Bed Tilt (`NozzleFrame` & `Gravity Vector`),
/// Motion Limits & Stepper Dynamics, Klipper/Moonraker Output, Start/End G-code Templates,
/// and Network Printing (Moonraker / Klipper).
fn render_printer_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    moonraker_state: &mut MoonrakerUiState,
    status_message: &mut String,
    is_computing: bool,
) {
    let mut dirty_backend = false;
    let mut dirty_gcode = false;

    egui::CollapsingHeader::new(egui::RichText::new("Print Bed Geometry & Origin").strong())
        .id_source("printer_bed_geometry_section")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                dirty_gcode |= ui
                    .selectable_value(
                        &mut config.bed.shape,
                        BedShape::Rectangular,
                        "Rectangular",
                    )
                    .changed();
                dirty_gcode |= ui
                    .selectable_value(
                        &mut config.bed.shape,
                        BedShape::Circular,
                        "Circular (Delta)",
                    )
                    .changed();
                param_help_button(ui, "bed.geometry");
            });

            ui.horizontal_wrapped(|ui| {
                ui.label("Quick Sizes:");
                if ui.small_button("256³ (Bambu/Voron)").clicked() {
                    config.bed.width_x = 256.0;
                    config.bed.depth_y = 256.0;
                    config.bed.height_z = 256.0;
                    dirty_gcode = true;
                }
                if ui.small_button("250×210 (Prusa)").clicked() {
                    config.bed.width_x = 250.0;
                    config.bed.depth_y = 210.0;
                    config.bed.height_z = 220.0;
                    dirty_gcode = true;
                }
                if ui.small_button("350³ (Large)").clicked() {
                    config.bed.width_x = 350.0;
                    config.bed.depth_y = 350.0;
                    config.bed.height_z = 350.0;
                    dirty_gcode = true;
                }
            });

            dirty_gcode |= ui
                .add(
                    egui::Slider::new(&mut config.bed.width_x, 50.0..=600.0).text("Width X (mm)"),
                )
                .changed();
            if config.bed.shape == BedShape::Rectangular {
                dirty_gcode |= ui
                    .add(
                        egui::Slider::new(&mut config.bed.depth_y, 50.0..=600.0)
                            .text("Depth Y (mm)"),
                    )
                    .changed();
            }
            dirty_gcode |= ui
                .add(egui::Slider::new(&mut config.bed.height_z, 50.0..=600.0).text("Max Z (mm)"))
                .changed();
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .checkbox(
                        &mut config.gcode.gcode_origin_at_bed_center,
                        "G-code Origin at Bed Center (0, 0) instead of Front-Left",
                    )
                    .changed();
                param_help_button(ui, "gcode.gcode_origin_at_bed_center");
            });
        });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Multi-Axis Kinematics & Gravity (Bed Coordinates)").strong(),
    )
    .id_source("printer_kinematics_gravity_section")
    .default_open(true)
    .show(ui, |ui| {
        let queried_frame = config.query_nozzle_frame(Vec3::ZERO, Vec3::X);
        let queried_gravity = config.query_gravity_bed(Vec3::ZERO);

        ui.small(format!(
            "Nozzle +Z (opp. nozzle): [{:.2}, {:.2}, {:.2}]",
            queried_frame.nozzle_axis.x, queried_frame.nozzle_axis.y, queried_frame.nozzle_axis.z
        ));
        ui.small(format!(
            "Travel +X: [{:.2}, {:.2}, {:.2}] | Lateral +Y: [{:.2}, {:.2}, {:.2}]",
            queried_frame.travel_axis.x,
            queried_frame.travel_axis.y,
            queried_frame.travel_axis.z,
            queried_frame.lateral_axis.x,
            queried_frame.lateral_axis.y,
            queried_frame.lateral_axis.z
        ));
        ui.small(format!(
            "Gravity Vector (Bed Coords): [{:.2}, {:.2}, {:.2}]",
            queried_gravity.x, queried_gravity.y, queried_gravity.z
        ));

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("Kinematics:");
            for mode in FiveAxisKinematicsMode::ALL {
                let clicked = ui
                    .selectable_value(
                        &mut config.machine.kinematics_mode,
                        mode,
                        mode.short_label(),
                    )
                    .on_hover_text(mode.label())
                    .changed();
                if clicked {
                    if mode == FiveAxisKinematicsMode::FiveAxisTiltingBedAB
                        && config.machine.b_min_deg >= 0.0
                    {
                        config.machine.b_min_deg = -45.0;
                        config.machine.b_max_deg = 45.0;
                    }
                    dirty_gcode = true;
                    dirty_backend = true;
                }
            }
            param_help_button(ui, "machine.kinematics_mode");
        });

        if config.machine.kinematics_mode == FiveAxisKinematicsMode::FiveAxisTiltingBedAB {
            ui.horizontal_wrapped(|ui| {
                ui.label("A Tilt Range (Rx °):");
                let a_min_changed = ui
                    .add(
                        egui::DragValue::new(&mut config.machine.a_min_deg)
                            .speed(0.5)
                            .range(-90.0..=0.0)
                            .prefix("Min: ")
                            .suffix("°"),
                    )
                    .changed();
                let a_max_changed = ui
                    .add(
                        egui::DragValue::new(&mut config.machine.a_max_deg)
                            .speed(0.5)
                            .range(0.0..=90.0)
                            .prefix("Max: ")
                            .suffix("°"),
                    )
                    .changed();
                dirty_gcode |= a_min_changed || a_max_changed;
                dirty_backend |= a_min_changed || a_max_changed;
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("B Tilt Range (Ry °):");
                let b_min_changed = ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_min_deg)
                            .speed(0.5)
                            .range(-90.0..=0.0)
                            .prefix("Min: ")
                            .suffix("°"),
                    )
                    .changed();
                let b_max_changed = ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_max_deg)
                            .speed(0.5)
                            .range(0.0..=90.0)
                            .prefix("Max: ")
                            .suffix("°"),
                    )
                    .changed();
                dirty_gcode |= b_min_changed || b_max_changed;
                dirty_backend |= b_min_changed || b_max_changed;
                param_help_button(ui, "machine.five_axis_bc_limits");
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .checkbox(&mut config.machine.invert_a, "Invert A")
                    .changed();
                dirty_gcode |= ui
                    .checkbox(&mut config.machine.invert_b, "Invert B")
                    .changed();
                ui.small("• Gravity dynamically tracks A & B bed tilt");
            });
            ui.horizontal(|ui| {
                ui.label("Max A / B Speed (°/s):");
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.max_a_speed_deg_s)
                            .speed(1.0)
                            .range(1.0..=720.0)
                            .prefix("A: "),
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.max_b_speed_deg_s)
                            .speed(1.0)
                            .range(1.0..=720.0)
                            .prefix("B: "),
                    )
                    .changed();
                param_help_button(ui, "machine.five_axis_bc_dynamics");
            });
            ui.horizontal(|ui| {
                ui.label("A / B Accel (°/s²):");
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.a_accel_deg_s2)
                            .speed(10.0)
                            .range(10.0..=10000.0)
                            .prefix("A: "),
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_accel_deg_s2)
                            .speed(10.0)
                            .range(10.0..=10000.0)
                            .prefix("B: "),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .add(
                        egui::Slider::new(
                            &mut config.machine.orientation_smoothing_window_mm,
                            0.0..=15.0,
                        )
                        .text("Orientation Smoothing Window (mm)"),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .checkbox(
                        &mut config.machine.use_inverse_time_feedrate_g93,
                        "Use G93 Inverse-Time Feedrate (1/min)",
                    )
                    .changed();
            });
            ui.separator();
        } else if config.machine.kinematics_mode == FiveAxisKinematicsMode::FiveAxisBC {
            ui.horizontal_wrapped(|ui| {
                ui.label("B Tilt Range (°):");
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_min_deg)
                            .speed(0.5)
                            .range(-90.0..=0.0)
                            .prefix("Min: ")
                            .suffix("°"),
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_max_deg)
                            .speed(0.5)
                            .range(0.0..=90.0)
                            .prefix("Max: ")
                            .suffix("°"),
                    )
                    .changed();
                param_help_button(ui, "machine.five_axis_bc_limits");
            });
            ui.horizontal_wrapped(|ui| {
                dirty_gcode |= ui
                    .checkbox(
                        &mut config.machine.c_continuous,
                        "Continuous C Rotation (360°+)",
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.c_zero_offset_deg)
                            .speed(1.0)
                            .range(-180.0..=180.0)
                            .prefix("C Offset: ")
                            .suffix("°"),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .checkbox(&mut config.machine.invert_b, "Invert B")
                    .changed();
                dirty_gcode |= ui
                    .checkbox(&mut config.machine.invert_c, "Invert C")
                    .changed();
                dirty_gcode |= ui
                    .checkbox(
                        &mut config.machine.bed_tilts_with_b,
                        "Bed Tilts With B (Gravity Coupling)",
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Max B / C Speed (°/s):");
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.max_b_speed_deg_s)
                            .speed(1.0)
                            .range(1.0..=720.0)
                            .prefix("B: "),
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.max_c_speed_deg_s)
                            .speed(1.0)
                            .range(1.0..=1080.0)
                            .prefix("C: "),
                    )
                    .changed();
                param_help_button(ui, "machine.five_axis_bc_dynamics");
            });
            ui.horizontal(|ui| {
                ui.label("B / C Accel (°/s²):");
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.b_accel_deg_s2)
                            .speed(10.0)
                            .range(10.0..=10000.0)
                            .prefix("B: "),
                    )
                    .changed();
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.machine.c_accel_deg_s2)
                            .speed(10.0)
                            .range(10.0..=10000.0)
                            .prefix("C: "),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .add(
                        egui::Slider::new(
                            &mut config.machine.orientation_smoothing_window_mm,
                            0.0..=15.0,
                        )
                        .text("Orientation Smoothing Window (mm)"),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .checkbox(
                        &mut config.machine.use_inverse_time_feedrate_g93,
                        "Use G93 Inverse-Time Feedrate (1/min)",
                    )
                    .changed();
            });
            ui.separator();
        }

        let mut tilt_changed = false;
        ui.horizontal(|ui| {
            tilt_changed |= ui
                .add(
                    egui::Slider::new(&mut config.machine.bed_tilt_rx_deg, -60.0..=60.0)
                        .text("5-Axis Bed Tilt Rx (°)"),
                )
                .changed();
            param_help_button(ui, "machine.bed_tilt_deg");
        });
        ui.horizontal(|ui| {
            tilt_changed |= ui
                .add(
                    egui::Slider::new(&mut config.machine.bed_tilt_ry_deg, -60.0..=60.0)
                        .text("5-Axis Bed Tilt Ry (°)"),
                )
                .changed();
            param_help_button(ui, "machine.bed_tilt_deg");
        });
        if ui.small_button("Reset Level Bed (0°, 0°)").clicked() {
            config.machine.bed_tilt_rx_deg = 0.0;
            config.machine.bed_tilt_ry_deg = 0.0;
            tilt_changed = true;
        }
        if tilt_changed {
            config.machine.update_gravity_from_bed_tilt();
            dirty_backend = true;
        }
    });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Motion Limits, Accelerations & Stepper Dynamics").strong(),
    )
    .id_source("printer_motion_limits_section")
    .default_open(true)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal(|ui| {
            ui.label("Wall Accel Out / In (mm/s²):");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.outer_wall_accel_mms2)
                        .speed(50.0)
                        .range(100.0..=30000.0)
                        .prefix("Out: "),
                )
                .changed();
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.inner_wall_accel_mms2)
                        .speed(50.0)
                        .range(100.0..=30000.0)
                        .prefix("In: "),
                )
                .changed();
            param_help_button(ui, "gcode.accelerations_mms2");
        });
        ui.horizontal(|ui| {
            ui.label("Infill / Travel Accel (mm/s²):");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.infill_accel_mms2)
                        .speed(50.0)
                        .range(100.0..=30000.0)
                        .prefix("Inf: "),
                )
                .changed();
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.travel_accel_mms2)
                        .speed(50.0)
                        .range(100.0..=50000.0)
                        .prefix("Trv: "),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Travel / Z-Hop Speed (mm/s):");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.travel_speed_mms)
                        .speed(2.0)
                        .range(20.0..=1000.0)
                        .prefix("Trv: "),
                )
                .changed();
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.z_hop_speed_mms)
                        .speed(0.5)
                        .range(2.0..=150.0)
                        .prefix("Hop: "),
                )
                .changed();
            param_help_button(ui, "gcode.z_hop_speed_mms");
        });
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::Slider::new(&mut gc.square_corner_velocity_mms, 0.5..=25.0)
                        .text("SCV (mm/s)"),
                )
                .changed();
            param_help_button(ui, "gcode.square_corner_velocity_mms");
        });
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::Slider::new(&mut gc.minimum_cruise_ratio, 0.0..=0.90)
                        .text("Min Cruise Ratio"),
                )
                .changed();
            param_help_button(ui, "gcode.minimum_cruise_ratio");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.enable_corner_flow_comp,
                    "Klipper SCV Corner Flow Compensation",
                )
                .changed();
            param_help_button(ui, "gcode.corner_flow_comp");
        });
        if gc.enable_corner_flow_comp {
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .add(
                        egui::Slider::new(&mut gc.corner_flow_comp_ratio, 0.0..=1.0)
                            .text("Corner Deduction Ratio"),
                    )
                    .changed();
                param_help_button(ui, "gcode.corner_flow_comp");
            });
        }
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.use_stepper_dynamics,
                    "Stepper Back-EMF Torque Roll-off",
                )
                .changed();
            param_help_button(ui, "gcode.use_stepper_dynamics");
        });
        if gc.use_stepper_dynamics {
            ui.horizontal(|ui| {
                dirty_gcode |= ui
                    .add(
                        egui::Slider::new(&mut gc.max_stepper_speed_mms, 100.0..=1200.0)
                            .text("Stepper Pull-Out Speed (mm/s)"),
                    )
                    .changed();
                param_help_button(ui, "gcode.max_stepper_speed_mms");
            });
        }
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::Slider::new(&mut gc.z_travel_penalty, 1.0..=15.0)
                        .suffix("×")
                        .text("Z-Travel Penalty"),
                )
                .changed();
            param_help_button(ui, "gcode.z_travel_penalty");
        });
    });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("G-code Dialect, Firmware & Templates").strong(),
    )
    .id_source("printer_gcode_templates_section")
    .default_open(false)
    .show(ui, |ui| {
        let mut dialect_switched = false;
        {
            let gc = &mut config.gcode;
            ui.horizontal_wrapped(|ui| {
                ui.label("Firmware Dialect:");
                for dialect in GcodeDialect::ALL {
                    if ui
                        .selectable_value(&mut gc.gcode_dialect, dialect, dialect.short_label())
                        .on_hover_text(dialect.label())
                        .changed()
                    {
                        dirty_gcode = true;
                        dialect_switched = true;
                    }
                }
                param_help_button(ui, "gcode.gcode_dialect");
            });
        }
        if dialect_switched {
            match config.gcode.gcode_dialect {
                GcodeDialect::Prusa => {
                    config.network.protocol = PrinterNetworkProtocol::PrusaLink;
                    if config.network.username.trim().is_empty()
                        || config.network.username.eq_ignore_ascii_case("bblp")
                    {
                        config.network.username = "maker".to_string();
                    }
                    moonraker_state.last_test_result = None;
                    moonraker_state.last_status = None;
                }
                GcodeDialect::BambuLab => {
                    config.network.protocol = PrinterNetworkProtocol::BambuLan;
                    if config.network.username.trim().is_empty()
                        || config.network.username.eq_ignore_ascii_case("maker")
                    {
                        config.network.username = "bblp".to_string();
                    }
                    moonraker_state.last_test_result = None;
                    moonraker_state.last_status = None;
                }
                GcodeDialect::Klipper => {
                    config.network.protocol = PrinterNetworkProtocol::Moonraker;
                    moonraker_state.last_test_result = None;
                    moonraker_state.last_status = None;
                }
                GcodeDialect::RepRapFirmware => {}
            }
        }
        let gc = &mut config.gcode;
        ui.horizontal_wrapped(|ui| {
            let dyn_pa_label = match gc.gcode_dialect {
                GcodeDialect::Klipper => "Emit Dynamic SET_PRESSURE_ADVANCE Commands",
                GcodeDialect::Prusa | GcodeDialect::RepRapFirmware => {
                    "Emit Dynamic M572 Pressure Advance Commands"
                }
                GcodeDialect::BambuLab => "Emit Dynamic M900 K Pressure Advance Commands",
            };
            dirty_gcode |= ui
                .checkbox(&mut gc.enable_dynamic_pa, dyn_pa_label)
                .changed();
            param_help_button(ui, "gcode.enable_dynamic_pa");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.enable_slicer_pa_subdivision,
                    "Slicer-Side Trapezoidal PA Subdivision",
                )
                .changed();
            param_help_button(ui, "gcode.enable_slicer_pa_subdivision");
        });
        ui.horizontal_wrapped(|ui| {
            let exclude_label = match gc.gcode_dialect {
                GcodeDialect::Klipper => "Emit Klipper EXCLUDE_OBJECT commands",
                GcodeDialect::Prusa => "Emit Prusa M486 Cancel-Object Labels",
                GcodeDialect::RepRapFirmware => "Emit RepRapFirmware M486 Object Cancellation",
                GcodeDialect::BambuLab => "Emit Bambu Lab M624/M625 Skip-Object Labels",
            };
            dirty_gcode |= ui
                .checkbox(&mut gc.emit_exclude_object, exclude_label)
                .changed();
            param_help_button(ui, "gcode.emit_exclude_object");
        });
        ui.horizontal_wrapped(|ui| {
            let thumb_label = match gc.gcode_dialect {
                GcodeDialect::Prusa => "Embed Prusa / Buddy PNG Thumbnails (32x32 & 300x300)",
                GcodeDialect::BambuLab => "Embed Bambu Studio PNG Thumbnails (32x32 & 300x300)",
                _ => "Embed Moonraker PNG Thumbnails (32x32 & 300x300)",
            };
            dirty_gcode |= ui
                .checkbox(&mut gc.emit_thumbnails, thumb_label)
                .changed();
            param_help_button(ui, "gcode.emit_thumbnails");
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.small(
                "Placeholders: {tool_id}, {nozzle_temp}, {bed_temp}, {nozzle_diameter}, {bbox_min_x..z}, {bbox_max_x..z}, {bbox_width_x}, {bbox_depth_y}, {bbox_center_x..y}, {layer_count}",
            );
            param_help_button(ui, "gcode.templates");
        });
        ui.label("Start G-code Template:");
        dirty_gcode |= ui
            .add(
                egui::TextEdit::multiline(&mut gc.start_gcode_template)
                    .code_editor()
                    .desired_rows(5),
            )
            .changed();
        ui.label("End G-code Template:");
        dirty_gcode |= ui
            .add(
                egui::TextEdit::multiline(&mut gc.end_gcode_template)
                    .code_editor()
                    .desired_rows(4),
            )
            .changed();
    });

    ui.add_space(4.0);
    let net_section_title = match config.network.protocol {
        PrinterNetworkProtocol::Moonraker => "Network Printing (Moonraker / Klipper)",
        PrinterNetworkProtocol::PrusaLink => "Network Printing (PrusaLink / Prusa Buddy)",
        PrinterNetworkProtocol::BambuLan => "Network Printing (Bambu Lab LAN Mode)",
    };
    egui::CollapsingHeader::new(egui::RichText::new(net_section_title).strong())
        .id_source("printer_network_moonraker_section")
        .default_open(true)
        .show(ui, |ui| {
            let mut net_changed = false;
            ui.horizontal_wrapped(|ui| {
                ui.label("Protocol:");
                for proto in PrinterNetworkProtocol::ALL {
                    if ui
                        .selectable_value(&mut config.network.protocol, proto, proto.choice_label())
                        .on_hover_text(proto.label())
                        .changed()
                    {
                        net_changed = true;
                        moonraker_state.last_test_result = None;
                        moonraker_state.last_status = None;
                        if proto == PrinterNetworkProtocol::PrusaLink
                            && (config.network.username.trim().is_empty()
                                || config.network.username.eq_ignore_ascii_case("bblp"))
                        {
                            config.network.username = "maker".to_string();
                        } else if proto == PrinterNetworkProtocol::BambuLan
                            && (config.network.username.trim().is_empty()
                                || config.network.username.eq_ignore_ascii_case("maker"))
                        {
                            config.network.username = "bblp".to_string();
                        }
                    }
                }
                param_help_button(ui, "network.protocol");
            });

            let (uri_label, uri_hint) = match config.network.protocol {
                PrinterNetworkProtocol::Moonraker => ("Printer URI:", "http://printer.local:7125"),
                PrinterNetworkProtocol::PrusaLink => ("Printer URI:", "http://prusa-core-one.local"),
                PrinterNetworkProtocol::BambuLan => ("Printer IP / Host:", "192.168.1.50"),
            };
            ui.horizontal(|ui| {
                ui.label(uri_label);
                net_changed |= ui
                    .add(
                        egui::TextEdit::singleline(&mut config.network.printer_uri)
                            .hint_text(uri_hint)
                            .desired_width(210.0),
                    )
                    .changed();
                param_help_button(ui, "network.printer_uri");
            });

            if config.network.protocol != PrinterNetworkProtocol::BambuLan {
                let web_ui_hint = match config.network.protocol {
                    PrinterNetworkProtocol::Moonraker => "Auto (Printer URI without :7125)",
                    PrinterNetworkProtocol::PrusaLink | PrinterNetworkProtocol::BambuLan => {
                        "Auto (Same as Printer URI)"
                    }
                };
                ui.horizontal(|ui| {
                    ui.label("Web UI URI:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.web_ui_uri)
                                .hint_text(web_ui_hint)
                                .desired_width(175.0),
                        )
                        .changed();
                    let can_open_ui = config.network.is_configured();
                    let web_hover = resolve_printer_web_ui_url(&config.network)
                        .unwrap_or_else(|_| "Enter a Printer URI above first".to_string());
                    if ui
                        .add_enabled(can_open_ui, egui::Button::new("Open Web UI ↗"))
                        .on_hover_text(format!("Launch {web_hover} in your system browser"))
                        .clicked()
                    {
                        match open_printer_web_ui_in_browser(&config.network) {
                            Ok(url) => {
                                *status_message =
                                    format!("Opened printer Web UI ({url}) in system browser.");
                            }
                            Err(err) => {
                                *status_message = err;
                            }
                        }
                    }
                });
            }

            if config.network.protocol == PrinterNetworkProtocol::BambuLan {
                ui.horizontal(|ui| {
                    ui.label("Serial Number:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.serial_number)
                                .hint_text("e.g. 01P00A... or 00M00A...")
                                .desired_width(175.0),
                        )
                        .on_hover_text(
                            "15-character Bambu Lab printer Serial Number (used for MQTT device/<SERIAL>/request and report topics)",
                        )
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("Access Code:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.password)
                                .password(!moonraker_state.show_password)
                                .hint_text("8-character LAN Access Code")
                                .desired_width(175.0),
                        )
                        .on_hover_text(
                            "8-character LAN Access Code from Printer Settings -> LAN Only Mode (authenticates as user 'bblp' on FTPS :990 and MQTT :8883)",
                        )
                        .changed();
                    ui.checkbox(&mut moonraker_state.show_password, "Show");
                });

                ui.horizontal_wrapped(|ui| {
                    net_changed |= ui
                        .checkbox(&mut config.network.bambu_use_ams, "Use AMS")
                        .changed();
                    net_changed |= ui
                        .checkbox(&mut config.network.bambu_bed_leveling, "Auto Bed Leveling")
                        .changed();
                    ui.small("• Enable LAN Only Mode (+ Developer Mode on 2025+ FW)");
                });
            } else if config.network.protocol == PrinterNetworkProtocol::PrusaLink {
                ui.horizontal(|ui| {
                    ui.label("Username:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.username)
                                .hint_text("maker (default)")
                                .desired_width(175.0),
                        )
                        .changed();
                    param_help_button(ui, "network.credentials");
                });

                ui.horizontal(|ui| {
                    ui.label("Password:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.password)
                                .password(!moonraker_state.show_password)
                                .hint_text("PrusaLink password")
                                .desired_width(175.0),
                        )
                        .changed();
                    ui.checkbox(&mut moonraker_state.show_password, "Show");
                });
            }

            if config.network.protocol != PrinterNetworkProtocol::BambuLan {
                ui.horizontal(|ui| {
                    ui.label("API Key:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.api_key)
                                .password(!moonraker_state.show_api_key)
                                .hint_text("Optional X-Api-Key")
                                .desired_width(175.0),
                        )
                        .changed();
                    ui.checkbox(&mut moonraker_state.show_api_key, "Show");
                    param_help_button(ui, "network.api_key");
                });
            }

        egui::CollapsingHeader::new("TLS / mTLS Certificates (Optional)")
            .id_source("printer_network_tls_certs_subsection")
            .default_open(false)
            .show(ui, |ui| {
                let pick_busy = moonraker_state.pending_cert_pick_task.is_some();
                ui.horizontal(|ui| {
                    ui.small("CA Cert (PEM):");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.ca_cert_pem_path)
                                .hint_text("/path/to/ca.pem")
                                .desired_width(145.0),
                        )
                        .changed();
                    if ui
                        .add_enabled(!pick_busy, egui::Button::new("Browse...").small())
                        .clicked()
                    {
                        let task = IoTaskPool::get().spawn(async move {
                            let handle = rfd::AsyncFileDialog::new()
                                .add_filter("PEM Certificate", &["pem", "crt", "cer"])
                                .pick_file()
                                .await?;
                            Some(handle.path().to_string_lossy().into_owned())
                        });
                        moonraker_state.pending_cert_pick_task =
                            Some((CertPickKind::CaCert, task));
                    }
                    if !config.network.ca_cert_pem_path.is_empty()
                        && ui.small_button("✕").on_hover_text("Clear path").clicked()
                    {
                        config.network.ca_cert_pem_path.clear();
                        net_changed = true;
                    }
                    param_help_button(ui, "network.tls_certs");
                });

                ui.horizontal(|ui| {
                    ui.small("Client Cert:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.client_cert_pem_path)
                                .hint_text("/path/to/client.pem")
                                .desired_width(155.0),
                        )
                        .changed();
                    if ui
                        .add_enabled(!pick_busy, egui::Button::new("Browse...").small())
                        .clicked()
                    {
                        let task = IoTaskPool::get().spawn(async move {
                            let handle = rfd::AsyncFileDialog::new()
                                .add_filter("PEM Certificate", &["pem", "crt", "cer"])
                                .pick_file()
                                .await?;
                            Some(handle.path().to_string_lossy().into_owned())
                        });
                        moonraker_state.pending_cert_pick_task =
                            Some((CertPickKind::ClientCert, task));
                    }
                    if !config.network.client_cert_pem_path.is_empty()
                        && ui.small_button("✕").on_hover_text("Clear path").clicked()
                    {
                        config.network.client_cert_pem_path.clear();
                        net_changed = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.small("Client Key:");
                    net_changed |= ui
                        .add(
                            egui::TextEdit::singleline(&mut config.network.client_key_pem_path)
                                .hint_text("/path/to/client.key")
                                .desired_width(155.0),
                        )
                        .changed();
                    if ui
                        .add_enabled(!pick_busy, egui::Button::new("Browse...").small())
                        .clicked()
                    {
                        let task = IoTaskPool::get().spawn(async move {
                            let handle = rfd::AsyncFileDialog::new()
                                .add_filter("PEM Private Key", &["pem", "key"])
                                .pick_file()
                                .await?;
                            Some(handle.path().to_string_lossy().into_owned())
                        });
                        moonraker_state.pending_cert_pick_task =
                            Some((CertPickKind::ClientKey, task));
                    }
                    if !config.network.client_key_pem_path.is_empty()
                        && ui.small_button("✕").on_hover_text("Clear path").clicked()
                    {
                        config.network.client_key_pem_path.clear();
                        net_changed = true;
                    }
                });
            });

        ui.horizontal(|ui| {
            net_changed |= ui
                .checkbox(
                    &mut config.network.enable_status_polling,
                    "Monitor Printer Status",
                )
                .changed();
            net_changed |= ui
                .add_enabled(
                    config.network.enable_status_polling,
                    egui::DragValue::new(&mut config.network.poll_interval_s)
                        .speed(0.5)
                        .range(1.0..=30.0)
                        .prefix("Every ")
                        .suffix(" s"),
                )
                .changed();
        });

        if net_changed {
            config.network.sanitize();
            moonraker_state.last_poll_instant = Some(std::time::Instant::now());
        }

        ui.horizontal_wrapped(|ui| {
            let proto_label = config.network.protocol.short_label();
            let test_hover = match config.network.protocol {
                PrinterNetworkProtocol::Moonraker => {
                    "Verify connectivity, API key, and TLS certificates against Moonraker (/server/info & /printer/info) without printing anything"
                }
                PrinterNetworkProtocol::PrusaLink => {
                    "Verify connectivity, HTTP Digest username/password (or API key), and TLS certificates against PrusaLink (/api/v1/info & /api/v1/status) without printing anything"
                }
                PrinterNetworkProtocol::BambuLan => {
                    "Verify LAN connectivity, Access Code ('bblp'), and MQTT-TLS (:8883) status against the Bambu Lab printer without printing anything"
                }
            };
            let is_testing = moonraker_state.pending_test_task.is_some();
            let can_test = config.network.is_configured() && !is_testing;
            if ui
                .add_enabled(
                    can_test,
                    egui::Button::new(if is_testing {
                        "Testing Connection..."
                    } else {
                        "Test Connection"
                    }),
                )
                .on_hover_text(test_hover)
                .clicked()
            {
                let net_cfg = config.network.clone();
                *status_message = format!(
                    "Testing connection to {proto_label} at {}...",
                    net_cfg.printer_uri.trim()
                );
                let test_cfg = net_cfg.clone();
                moonraker_state.pending_test_task = Some(IoTaskPool::get().spawn(async move {
                    test_moonraker_connection(&test_cfg)
                }));
                if moonraker_state.pending_status_task.is_none() {
                    let cached_meta = moonraker_state.cached_metadata_estimate.clone();
                    moonraker_state.last_poll_instant = Some(std::time::Instant::now());
                    moonraker_state.pending_status_task =
                        Some(IoTaskPool::get().spawn(async move {
                            query_moonraker_print_status(&net_cfg, cached_meta)
                        }));
                }
            }
            if is_testing {
                ui.ctx().request_repaint();
                ui.spinner();
            }
        });

        if let Some(ref test_res) = moonraker_state.last_test_result {
            match test_res {
                Ok(report) => {
                    ui.small(
                        egui::RichText::new(format!("✓ {}", report.summary_line()))
                            .color(egui::Color32::from_rgb(130, 230, 165)),
                    );
                }
                Err(err) => {
                    ui.small(
                        egui::RichText::new(format!("✗ {err}"))
                            .color(egui::Color32::from_rgb(245, 115, 115)),
                    );
                }
            }
        } else if let Some(Err(ref poll_err)) = moonraker_state.last_status {
            ui.small(
                egui::RichText::new(format!("✗ {poll_err}"))
                    .color(egui::Color32::from_rgb(245, 115, 115)),
            );
        }

        if let Some(snap) = moonraker_state.active_upload_progress() {
            ui.ctx().request_repaint();
            ui.horizontal(|ui| {
                ui.spinner();
                ui.small(
                    egui::RichText::new(&snap.stage)
                        .color(egui::Color32::from_rgb(145, 215, 245)),
                );
            });
            ui.add(
                egui::ProgressBar::new(snap.fraction.clamp(0.0, 1.0))
                    .desired_width(ui.available_width().clamp(140.0, 320.0))
                    .animate(true)
                    .text(snap.bar_text()),
            );
        } else if let Some(ref up_res) = moonraker_state.last_upload_result {
            match up_res {
                Ok(report) => {
                    ui.small(
                        egui::RichText::new(format!("↑ {}", report.summary_line()))
                            .color(egui::Color32::from_rgb(145, 215, 245)),
                    );
                }
                Err(err) => {
                    ui.small(
                        egui::RichText::new(format!("↑ Upload error: {err}"))
                            .color(egui::Color32::from_rgb(245, 115, 115)),
                    );
                }
            }
        }
    });

    if dirty_backend && !is_computing {
        sync_config_to_backend(config);
    }
    if dirty_gcode {
        config.machine.sanitize();
        config.gcode.sanitize();
        vis_state.dirty_planned_gcode = true;
    }
}

/// Renders the `Tool` configuration sub-tab:
/// Multi-Tool Registry (`T0`, `T1`, ...), Nozzle Diameter, 5/6-Axis Rotational Limits,
/// Hotend Max Volumetric Flow, Retraction/Wipe/Z-Hop, and Local-Frame XZ Clearance Profile.
fn render_tool_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    preset_lib: &mut PresetLibrary,
    is_computing: bool,
) {
    let mut dirty_backend = false;
    let mut dirty_gcode = false;

    if config.tools.is_empty() {
        config.tools.push(ToolConfig::default());
        config.active_tool_index = 0;
        dirty_backend = true;
    }
    if config.active_tool_index >= config.tools.len() {
        config.active_tool_index = config.tools.len() - 1;
    }

    egui::CollapsingHeader::new(
        egui::RichText::new("Extrusion Tool Registry & Nozzle Geometry").strong(),
    )
    .id_source("tool_registry_nozzle_section")
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            let current_active = config.active_tool_index;
            let mut clicked_idx = None;
            for (idx, tool) in config.tools.iter().enumerate() {
                let selected = current_active == idx;
                if render_delineated_pill_tab(
                    ui,
                    selected,
                    format!(
                        "T{} ({:.2}mm, {})",
                        tool.id, tool.nozzle_diameter_mm, tool.filament_type
                    ),
                )
                .clicked()
                {
                    clicked_idx = Some(idx);
                }
            }
            if let Some(idx) = clicked_idx {
                config.active_tool_index = idx;
                config.sync_gcode_from_active_tool();
                if preset_lib
                    .filaments
                    .iter()
                    .any(|p| p.name == config.filament_preset_name)
                {
                    preset_lib.selected_filament = config.filament_preset_name.clone();
                }
                dirty_backend = true;
                dirty_gcode = true;
            }
            if ui.small_button("+ Add Tool").clicked() {
                let next_id = config.tools.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                config.tools.push(ToolConfig::standard_tool(
                    next_id,
                    format!("T{next_id} - 0.4mm"),
                    0.40,
                    215.0,
                ));
                config.active_tool_index = config.tools.len() - 1;
                config.sync_gcode_from_active_tool();
                dirty_backend = true;
                dirty_gcode = true;
            }
            if config.tools.len() > 1 && ui.small_button("Remove Tool").clicked() {
                let rm_idx = config.active_tool_index;
                config.tools.remove(rm_idx);
                if config.active_tool_index >= config.tools.len() {
                    config.active_tool_index = config.tools.len() - 1;
                }
                config.sync_gcode_from_active_tool();
                dirty_backend = true;
                dirty_gcode = true;
            }
        });

        let active_idx = config.active_tool_index;
        let filament_preset_list: Vec<(String, bool)> = preset_lib
            .filaments
            .iter()
            .map(|p| (p.name.clone(), p.is_builtin))
            .collect();
        let mut chosen_filament_preset: Option<String> = None;
        let mut sync_active_to_gcode = false;

        if let Some(tool) = config.tools.get_mut(active_idx) {
            ui.horizontal(|ui| {
                ui.label("Tool Name:");
                if ui.text_edit_singleline(&mut tool.name).changed() {
                    dirty_backend = true;
                }
            });
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut tool.nozzle_diameter_mm, 0.15..=2.0)
                            .text("Nozzle Diameter (mm)"),
                    )
                    .changed()
                {
                    dirty_backend = true;
                    dirty_gcode = true;
                }
                param_help_button(ui, "tools.nozzle_diameter_mm");
            });

            // Per-tool Filament Preset & Material Type assignment
            ui.horizontal_wrapped(|ui| {
                ui.label("Tool Filament Preset:");
                let cur_preset_lbl = if tool.filament_preset_name.trim().is_empty() {
                    format!("Custom ({})", tool.filament_type)
                } else {
                    tool.filament_preset_name.clone()
                };
                let combo_w = (ui.available_width() - 12.0).clamp(110.0, 205.0);
                egui::ComboBox::from_id_source(("tool_filament_preset_combo", tool.id))
                    .selected_text(cur_preset_lbl)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for (p_name, builtin) in &filament_preset_list {
                            let tag = if *builtin { "[Factory]" } else { "[User]" };
                            let is_sel = &tool.filament_preset_name == p_name;
                            if ui
                                .selectable_label(is_sel, format!("{p_name}  {tag}"))
                                .clicked()
                            {
                                chosen_filament_preset = Some(p_name.clone());
                            }
                        }
                    });
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Filament Type:");
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut tool.filament_type)
                            .desired_width(60.0)
                            .hint_text("PLA"),
                    )
                    .on_hover_text("Material type assigned to this tool (PLA, PETG, ABS, ASA, TPU, PVA, PA-CF, etc.)")
                    .changed()
                {
                    sync_active_to_gcode = true;
                    dirty_gcode = true;
                }
                for mat_name in ["PLA", "PETG", "ABS", "ASA", "TPU", "PVA", "PA-CF"] {
                    let is_mat = tool.filament_type.eq_ignore_ascii_case(mat_name);
                    if ui
                        .selectable_label(is_mat, mat_name)
                        .on_hover_text(format!(
                            "Apply {mat_name} temperature, cooling fan, density & PA defaults to T{}",
                            tool.id
                        ))
                        .clicked()
                    {
                        tool.apply_filament_type_defaults(mat_name);
                        sync_active_to_gcode = true;
                        dirty_backend = true;
                        dirty_gcode = true;
                    }
                }
                param_help_button(ui, "gcode.filament_type");
            });
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut tool.temperature_celsius, 170.0..=350.0)
                            .text("Nozzle Temp (°C)"),
                    )
                    .changed()
                {
                    sync_active_to_gcode = true;
                    dirty_backend = true;
                    dirty_gcode = true;
                }
                if ui
                    .add(
                        egui::DragValue::new(&mut tool.fan_speed_percent)
                            .speed(1.0)
                            .range(0.0..=100.0)
                            .prefix("Fan: ")
                            .suffix("%"),
                    )
                    .on_hover_text("Part cooling fan speed for this tool's filament")
                    .changed()
                {
                    sync_active_to_gcode = true;
                    dirty_gcode = true;
                }
                param_help_button(ui, "tools.temperature_celsius");
            });

            if render_tool_optical_properties_controls(ui, tool) {
                vis_state.dirty_color_mixing = true;
                vis_state.dirty_toolpaths_mesh = true;
                vis_state.dirty_surface_colors = true;
                dirty_gcode = true;
            }

            ui.horizontal(|ui| {
                if ui
                    .checkbox(
                        &mut tool.has_rotational_axes,
                        "Tool Has Rotational Axes (5/6-Axis)",
                    )
                    .changed()
                {
                    dirty_backend = true;
                }
                param_help_button(ui, "tools.rotational_axes");
            });
            if tool.has_rotational_axes {
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Slider::new(&mut tool.max_angle_from_bed_normal_deg, 0.0..=85.0)
                                .text("Max Angle from Bed Normal (°)"),
                        )
                        .changed()
                    {
                        dirty_backend = true;
                    }
                    param_help_button(ui, "tools.rotational_axes");
                });
            }
        }

        if let Some(p_name) = chosen_filament_preset {
            if let Some(fp) = preset_lib
                .filaments
                .iter()
                .find(|p| p.name == p_name)
                .cloned()
            {
                if let Some(tool) = config.tools.get_mut(active_idx) {
                    fp.apply_to_tool(tool);
                }
                fp.apply_to_config(config);
                preset_lib.selected_filament = p_name;
                dirty_backend = true;
                dirty_gcode = true;
                vis_state.dirty_color_mixing = true;
                vis_state.dirty_toolpaths_mesh = true;
            }
        } else if sync_active_to_gcode {
            config.sync_gcode_from_active_tool();
            if preset_lib
                .filaments
                .iter()
                .any(|p| p.name == config.filament_preset_name)
            {
                preset_lib.selected_filament = config.filament_preset_name.clone();
            }
        }

        if config.active_tool().is_some_and(|t| t.has_rotational_axes) {
            ui.horizontal(|ui| {
                if ui
                    .checkbox(
                        &mut config.slicing.orient_outer_surfaces_to_normal,
                        "Orient Outer Surfaces Almost Parallel to Surface Normal",
                    )
                    .on_hover_text(
                        "On outer surfaces, if the tool can be oriented within the 'Almost Parallel' angle of the outward surface normal (subject to tool tilt & bed clearance limits), orient to that; else use the isosurface normal.",
                    )
                    .changed()
                {
                    dirty_backend = true;
                }
                param_help_button(ui, "slicing.orient_outer_surfaces_to_normal");
            });
            if config.slicing.orient_outer_surfaces_to_normal {
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Slider::new(
                                &mut config.slicing.outer_surface_parallel_angle_deg,
                                0.0..=85.0,
                            )
                            .suffix("°")
                            .text("Almost Parallel Angle Threshold"),
                        )
                        .on_hover_text(
                            "Maximum residual angle between the reachable clamped tool axis and the outward surface normal before falling back to the isosurface normal.",
                        )
                        .changed()
                    {
                        dirty_backend = true;
                    }
                    param_help_button(ui, "slicing.outer_surface_parallel_angle_deg");
                });
            }
        }

        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut config.gcode.enable_flat_nozzle_comp,
                    "Flat-Nozzle Annular Slope Clearance",
                )
                .changed();
            param_help_button(ui, "gcode.enable_flat_nozzle_comp");
        });
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::Slider::new(&mut config.gcode.max_volumetric_flow_mm3s, 1.0..=50.0)
                        .text("Max Hotend Volumetric Flow (mm³/s)"),
                )
                .changed();
            param_help_button(ui, "gcode.max_volumetric_flow_mm3s");
        });
    });

    ui.add_space(4.0);
    render_3mf_filament_slot_mapping_section_config_only(
        ui,
        config,
        vis_state,
        &mut dirty_backend,
        &mut dirty_gcode,
        "tool_tab_filament_slot_mapping",
    );

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Extruder Retraction, Wipe & Z-Hop").strong(),
    )
    .id_source("tool_retraction_wipe_zhop_section")
    .default_open(true)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.retract_length_mm)
                        .speed(0.05)
                        .range(0.0..=8.0)
                        .prefix("Retract: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.retract_length_mm");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.retract_speed_mms)
                        .speed(1.0)
                        .range(5.0..=150.0)
                        .prefix("Speed: ")
                        .suffix(" mm/s"),
                )
                .changed();
            param_help_button(ui, "gcode.retract_speed_mms");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.min_travel_for_retract_mm)
                        .speed(0.1)
                        .range(0.1..=20.0)
                        .prefix("Min Travel: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.min_travel_for_retract_mm");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.z_hop_mm)
                        .speed(0.05)
                        .range(0.0..=3.0)
                        .prefix("Z-Hop: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.z_hop_mm");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.wipe_length_mm)
                        .speed(0.05)
                        .range(0.0..=6.0)
                        .prefix("Wipe Length: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.wipe_length_mm");
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.pre_retract_taper_mm)
                        .speed(0.05)
                        .range(0.0..=8.0)
                        .prefix("Pre-Retract Taper: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.pre_retract_taper_mm");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .add(
                    egui::DragValue::new(&mut gc.tool_change_time_s)
                        .speed(0.5)
                        .range(0.0..=600.0)
                        .prefix("Toolchange Time: ")
                        .suffix(" s"),
                )
                .on_hover_text(
                    "Estimated duration of each physical toolchange or MMU filament swap (T{id}) added to print time estimates",
                )
                .changed();
            param_help_button(ui, "gcode.tool_change_time_s");
        });
        ui.horizontal_wrapped(|ui| {
            dirty_gcode |= ui
                .checkbox(&mut gc.tool_change_hop_before, "Hop Before Toolchange")
                .on_hover_text(
                    "Hop above the highest so-far printed material before emitting T{id}",
                )
                .changed();
            param_help_button(ui, "gcode.tool_change_hop_before");
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.tool_change_approach_after,
                    "Approach After Toolchange",
                )
                .on_hover_text(
                    "After T{id}, move to an approach position above the next chain start (same height as a hop) before descending",
                )
                .changed();
            param_help_button(ui, "gcode.tool_change_approach_after");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Purge Mode:");
            let combo_w = (ui.available_width() - 32.0).clamp(115.0, 220.0);
            egui::ComboBox::from_id_source("printer_tab_purge_mode_combo")
                .selected_text(gc.purge_mode.short_label())
                .width(combo_w)
                .show_ui(ui, |ui| {
                    for mode in PurgeMode::ALL {
                        dirty_gcode |= ui
                            .selectable_value(&mut gc.purge_mode, mode, mode.label())
                            .changed();
                    }
                });
            param_help_button(ui, "gcode.purge_mode");
        });
        if gc.purge_mode != PurgeMode::Disabled {
            ui.horizontal_wrapped(|ui| {
                dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut gc.min_purge_volume_mm3)
                            .speed(1.0)
                            .range(1.0..=500.0)
                            .prefix("Min Purge Vol: ")
                            .suffix(" mm³"),
                    )
                    .on_hover_text(
                        "Minimum polymer volume (mm³) to purge after each toolchange before printing exterior walls",
                    )
                    .changed();
                param_help_button(ui, "gcode.min_purge_volume_mm3");
                if gc.purge_mode.uses_purge_tower() {
                    dirty_gcode |= ui
                        .checkbox(&mut gc.purge_tower_auto_dimensions, "Auto Tower Size")
                        .on_hover_text(
                            "Automatically choose rectangular prism width & depth from Min Purge Volume and layer height",
                        )
                        .changed();
                    dirty_gcode |= ui
                        .checkbox(&mut gc.purge_tower_brim, "Tower Brim")
                        .changed();
                }
            });
            if gc.purge_mode.uses_purge_tower() && !gc.purge_tower_auto_dimensions {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Tower Prism (W × D):");
                    dirty_gcode |= ui
                        .add(
                            egui::DragValue::new(&mut gc.purge_tower_width_mm)
                                .speed(0.5)
                                .range(8.0..=160.0)
                                .prefix("W: ")
                                .suffix(" mm"),
                        )
                        .changed();
                    dirty_gcode |= ui
                        .add(
                            egui::DragValue::new(&mut gc.purge_tower_depth_mm)
                                .speed(0.5)
                                .range(6.0..=120.0)
                                .prefix("D: ")
                                .suffix(" mm"),
                        )
                        .changed();
                    param_help_button(ui, "gcode.purge_tower_auto_dimensions");
                });
            }
            if gc.purge_mode.uses_purge_tower() {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Tower Pos:");
                    dirty_gcode |= ui
                        .checkbox(&mut gc.purge_tower_auto_position, "Auto")
                        .on_hover_text(
                            "Automatically place the purge tower outside the objects' bounding box on the build plate",
                        )
                        .changed();
                    if ui
                        .add(
                            egui::DragValue::new(&mut gc.purge_tower_position_xy[0])
                                .speed(0.5)
                                .range(-250.0..=250.0)
                                .prefix("X: ")
                                .suffix(" mm"),
                        )
                        .on_hover_text(
                            "Purge tower center X on build plate (0 = bed center). Dragging switches to manual placement.",
                        )
                        .changed()
                    {
                        gc.purge_tower_auto_position = false;
                        dirty_gcode = true;
                    }
                    if ui
                        .add(
                            egui::DragValue::new(&mut gc.purge_tower_position_xy[1])
                                .speed(0.5)
                                .range(-250.0..=250.0)
                                .prefix("Y: ")
                                .suffix(" mm"),
                        )
                        .on_hover_text(
                            "Purge tower center Y on build plate (0 = bed center). Dragging switches to manual placement.",
                        )
                        .changed()
                    {
                        gc.purge_tower_auto_position = false;
                        dirty_gcode = true;
                    }
                });
            }
        }
    });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Clearance Profile (Local Nozzle Frame)").strong(),
    )
    .id_source("tool_clearance_profile_xz_section")
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Profile Mode:");
            param_help_button(ui, "tools.clearance_profile_xz");
        });

        let active_idx = config.active_tool_index;
        if let Some(tool) = config.tools.get_mut(active_idx) {
            egui::Frame::none()
                .fill(egui::Color32::from_rgb(18, 22, 29))
                .rounding(egui::Rounding::same(6.0))
                .inner_margin(egui::Margin::symmetric(5.0, 4.0))
                .stroke(egui::Stroke::new(
                    1.0_f32,
                    egui::Color32::from_rgb(48, 56, 70),
                ))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        for mode in ClearanceProfileMode::ALL {
                            if render_delineated_pill_tab(
                                ui,
                                tool.clearance_mode == mode,
                                mode.label(),
                            )
                            .clicked()
                            {
                                tool.clearance_mode = mode;
                                dirty_backend = true;
                            }
                        }
                    });
                });

            let (alpha_x_deg, alpha_y_deg) = tool.max_layer_slope_xy_deg();
            ui.small(format!(
                "Max collision-free layer slope: α_max,X = {alpha_x_deg:.1}°, α_max,Y = {alpha_y_deg:.1}°"
            ));
            ui.add_space(2.0);

            match tool.clearance_mode {
                ClearanceProfileMode::PiecewiseXZ => {
                    ui.small("Radius X (mm) from nozzle axis vs. height Z (mm) along nozzle axis:");
                    let mut remove_pt_idx: Option<usize> = None;
                    let profile_len = tool.clearance_profile_xz.len();
                    for (p_idx, pt) in tool.clearance_profile_xz.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(format!("#{p_idx}"));
                            if ui
                                .add(
                                    egui::DragValue::new(&mut pt.x_radius_mm)
                                        .speed(0.1)
                                        .range(0.0..=200.0)
                                        .prefix("X (r): ")
                                        .suffix(" mm"),
                                )
                                .changed()
                            {
                                dirty_backend = true;
                            }
                            if ui
                                .add(
                                    egui::DragValue::new(&mut pt.z_height_mm)
                                        .speed(0.1)
                                        .range(0.0..=300.0)
                                        .prefix("Z: ")
                                        .suffix(" mm"),
                                )
                                .changed()
                            {
                                dirty_backend = true;
                            }
                            if profile_len > 2 && ui.small_button("✕").clicked() {
                                remove_pt_idx = Some(p_idx);
                            }
                        });
                    }
                    if let Some(rm_idx) = remove_pt_idx {
                        tool.clearance_profile_xz.remove(rm_idx);
                        dirty_backend = true;
                    }
                    ui.horizontal(|ui| {
                        if ui.small_button("+ Add XZ Point").clicked() {
                            let (next_x, next_z) = tool
                                .clearance_profile_xz
                                .last()
                                .map(|p| (p.x_radius_mm + 4.0, p.z_height_mm + 10.0))
                                .unwrap_or((1.0, 0.0));
                            tool.clearance_profile_xz
                                .push(ClearancePointXZ::new(next_x, next_z));
                            dirty_backend = true;
                        }
                        if ui.small_button("Sort by Z").clicked() {
                            tool.clearance_profile_xz.sort_by(|a, b| {
                                a.z_height_mm
                                    .partial_cmp(&b.z_height_mm)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            dirty_backend = true;
                        }
                    });
                }
                ClearanceProfileMode::ParametricCone => {
                    ui.small("Straight-sided circular cone revolved around local nozzle_axis (+Z):");
                    ui.horizontal(|ui| {
                        ui.label("Tip Radius (z=0):");
                        if ui
                            .add(
                                egui::DragValue::new(&mut tool.cone_tip_radius_mm)
                                    .speed(0.05)
                                    .range(0.1..=50.0)
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                        ui.label("Height Z:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut tool.cone_height_mm)
                                    .speed(0.5)
                                    .range(2.0..=300.0)
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Cone Half-Angle θ:");
                        if ui
                            .add(
                                egui::Slider::new(&mut tool.cone_half_angle_deg, 5.0..=80.0)
                                    .suffix("°"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    if let Some((r_top, _)) =
                        tool.clearance_half_widths_xy_at_z(tool.cone_height_mm)
                    {
                        ui.small(format!(
                            "Top radius at Z = {:.1} mm: {:.1} mm (diameter {:.1} mm)",
                            tool.cone_height_mm,
                            r_top,
                            2.0 * r_top
                        ));
                    }
                }
                ClearanceProfileMode::RectilinearCone => {
                    ui.small(
                        "4-sided rectangular pyramid cone in local nozzle frame (X = travel_axis, Y = lateral_axis):",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Tip Half-Width (z=0):");
                        if ui
                            .add(
                                egui::DragValue::new(&mut tool.rect_tip_half_width_x_mm)
                                    .speed(0.05)
                                    .range(0.1..=50.0)
                                    .prefix("±X: ")
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                        if ui
                            .add(
                                egui::DragValue::new(&mut tool.rect_tip_half_width_y_mm)
                                    .speed(0.05)
                                    .range(0.1..=50.0)
                                    .prefix("±Y: ")
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Half-Angle X (XZ):");
                        if ui
                            .add(
                                egui::Slider::new(&mut tool.rect_half_angle_x_deg, 5.0..=80.0)
                                    .suffix("°"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Half-Angle Y (YZ):");
                        if ui
                            .add(
                                egui::Slider::new(&mut tool.rect_half_angle_y_deg, 5.0..=80.0)
                                    .suffix("°"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Cone Height Z:");
                        if ui
                            .add(
                                egui::DragValue::new(&mut tool.rect_height_mm)
                                    .speed(0.5)
                                    .range(2.0..=300.0)
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            dirty_backend = true;
                        }
                    });
                    if let Some((wx_top, wy_top)) =
                        tool.clearance_half_widths_xy_at_z(tool.rect_height_mm)
                    {
                        ui.small(format!(
                            "Top footprint at Z = {:.1} mm: {:.1} × {:.1} mm (±{:.1} X, ±{:.1} Y)",
                            tool.rect_height_mm,
                            2.0 * wx_top,
                            2.0 * wy_top,
                            wx_top,
                            wy_top
                        ));
                    }
                }
            }
        }
    });

    if dirty_backend && !is_computing {
        sync_config_to_backend(config);
    }
    if dirty_gcode {
        config.machine.sanitize();
        config.gcode.sanitize();
        vis_state.dirty_planned_gcode = true;
    }
}

/// Renders per-tool filament colour (sRGB swatch + standard `#RRGGBB` / `rgb(...)` / CSS encoding)
/// and Kubelka-Munk Transmission Distance (`TD` in mm, with Auto default estimation).
/// Returns `true` if any optical property was modified.
fn render_tool_optical_properties_controls(ui: &mut egui::Ui, tool: &mut ToolConfig) -> bool {
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.label("Colour & TD:");
        let mut c32 = egui::Color32::from_rgba_unmultiplied(
            (tool.filament_color_srgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (tool.filament_color_srgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (tool.filament_color_srgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            255,
        );
        if ui
            .color_edit_button_srgba(&mut c32)
            .on_hover_text(format!(
                "Actual opaque filament colour R_∞ for T{} ({})",
                tool.id, tool.filament_color_hex
            ))
            .changed()
        {
            tool.set_color_from_srgba([
                c32.r() as f32 / 255.0,
                c32.g() as f32 / 255.0,
                c32.b() as f32 / 255.0,
                1.0,
            ]);
            changed = true;
        }

        let mut color_text = tool.filament_color_hex.clone();
        let resp = ui
            .add(
                egui::TextEdit::singleline(&mut color_text)
                    .desired_width(74.0)
                    .hint_text("#RRGGBB"),
            )
            .on_hover_text(
                "Standard colour encoding (#RGB, #RRGGBB, rgb(r,g,b), or CSS colour name like 'cyan', 'magenta', 'yellow', 'white', 'black')",
            );
        if (resp.lost_focus() || resp.changed()) && color_text != tool.filament_color_hex {
            if tool.set_color_from_encoding(&color_text) {
                changed = true;
            }
        }

        let auto_est_td = crate::color_optics::estimate_default_td_mm(
            &tool.filament_type,
            [
                tool.filament_color_srgba[0],
                tool.filament_color_srgba[1],
                tool.filament_color_srgba[2],
            ],
        );
        let mut is_auto_td = tool.transmission_distance_mm.is_none();
        let auto_lbl = if is_auto_td {
            format!("Auto TD ({auto_est_td:.1} mm)")
        } else {
            "Auto TD".to_string()
        };
        if ui
            .checkbox(&mut is_auto_td, auto_lbl)
            .on_hover_text(
                "Estimate default Kubelka-Munk Transmission Distance (TD) from filament material & pigment lightness (~2.5 mm standard opaque baseline)",
            )
            .changed()
        {
            if is_auto_td {
                tool.transmission_distance_mm = None;
            } else {
                tool.transmission_distance_mm = Some(auto_est_td);
            }
            changed = true;
        }
        if let Some(ref mut td_mm) = tool.transmission_distance_mm {
            if ui
                .add(
                    egui::DragValue::new(td_mm)
                        .speed(0.1)
                        .range(0.1..=30.0)
                        .prefix("TD: ")
                        .suffix(" mm"),
                )
                .on_hover_text(
                    "HueForge / Kubelka-Munk Transmission Distance (mm to ~99% optical opacity)",
                )
                .changed()
            {
                changed = true;
            }
        }
        param_help_button(ui, "tool.transmission_distance_mm");
    });
    changed
}

/// Applies a multi-filament optical palette (e.g. 5-tool CMYKW) to `config`, enables optical
/// colour mixing, ensures shell wall count supports `optical_shell_layers`, and refreshes preview flags.
fn apply_quick_cmykw_palette_setup(
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    palette: &[crate::color_optics::PaletteFilamentEntry],
) -> bool {
    config.apply_optical_palette_to_tools(palette);
    config.slicing.color_mixing.enabled = true;
    let needed_inner = config
        .slicing
        .color_mixing
        .optical_shell_layers
        .saturating_sub(1);
    let mut need_reslice = false;
    if config.slicing.inner_wall_count < needed_inner {
        config.slicing.inner_wall_count = needed_inner;
        need_reslice = true;
    }
    vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
    vis_state.toolpath_quality_mode = ToolpathQualityMode::HighQualityPbr;
    vis_state.dirty_color_mixing = true;
    vis_state.dirty_planned_gcode = true;
    vis_state.dirty_toolpaths_mesh = true;
    if need_reslice {
        vis_state.dirty_toolpaths = true;
    }
    need_reslice
}

/// Renders the 3MF Colour / Filament Slot -> Machine Tool mapping card in the Project Setup tab,
/// updating both `SlicerConfig` and any unselected multi-colour `SceneObject` meshes when colours change.
fn render_3mf_filament_slot_mapping_section(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    _plate: &mut BuildPlateState,
    _meshes: &mut Assets<Mesh>,
    is_computing: bool,
    id_salt: &str,
) {
    let mut dirty_backend = false;
    let mut dirty_gcode = false;

    render_3mf_filament_slot_mapping_section_config_only(
        ui,
        config,
        vis_state,
        &mut dirty_backend,
        &mut dirty_gcode,
        id_salt,
    );

    if dirty_backend && !is_computing {
        sync_config_to_backend(config);
    }
    if dirty_gcode {
        vis_state.dirty_planned_gcode = true;
        vis_state.dirty_toolpaths_mesh = true;
    }
}

/// Renders the collapsible **"3MF Colour / Filament → Tool Mapping"** UI section allowing the user
/// to map each 3MF colour/filament slot (`ThreeMfFilamentSlot`) to any tool in `config.tools`,
/// edit its display colour swatch, or auto-create/assign 1:1 tools.
fn render_3mf_filament_slot_mapping_section_config_only(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    dirty_backend: &mut bool,
    dirty_gcode: &mut bool,
    id_salt: &str,
) {
    let slot_count = config.filament_slots.len();
    let virtual_count = config
        .filament_slots
        .iter()
        .filter(|s| s.is_virtual_mix())
        .count();
    let header_title = if slot_count > 0 {
        if virtual_count > 0 {
            format!(
                "3MF Colour / Virtual Filament Mapping ({slot_count} Slots, {virtual_count} Virtual)"
            )
        } else {
            format!("3MF Colour / Filament → Tool Mapping ({slot_count} Slots)")
        }
    } else {
        "3MF Colour / Virtual Filament Mapping".to_string()
    };

    egui::CollapsingHeader::new(egui::RichText::new(header_title).strong())
        .id_source(id_salt)
        .default_open(slot_count > 0)
        .show(ui, |ui| {
            ui.small(
                "Map each colour/filament slot to a single machine tool (T0, T1, ...) or define a Virtual Filament composed of 2+ real tools dithered together in an integer ratio.",
            );

            ui.horizontal_wrapped(|ui| {
                if ui
                    .small_button("+ Add Filament Slot")
                    .on_hover_text("Add a filament / colour slot manually")
                    .clicked()
                {
                    let next_idx = config.filament_slots.len().min(254) as u32;
                    let (hex, srgba) = crate::config::DEFAULT_FILAMENT_SLOT_PALETTE
                        [(next_idx as usize) % crate::config::DEFAULT_FILAMENT_SLOT_PALETTE.len()];
                    let tool_idx = (next_idx as usize).min(config.tools.len().saturating_sub(1));
                    config
                        .filament_slots
                        .push(crate::config::ThreeMfFilamentSlot {
                            slot_index: next_idx,
                            name: format!("Filament {} (PLA)", next_idx as usize + 1),
                            filament_type: "PLA".to_string(),
                            color_hex: hex.to_string(),
                            color_srgba: srgba,
                            tool_index: tool_idx,
                            ..Default::default()
                        });
                    vis_state.dirty_surface_colors = true;
                    vis_state.dirty_color_mixing = true;
                    vis_state.dirty_toolpaths_mesh = true;
                    *dirty_gcode = true;
                }

                if ui
                    .small_button("+ Add Virtual Filament (Ratio Mix)")
                    .on_hover_text(
                        "Add a Virtual Filament composed of two or more real tools dithered together in a custom integer ratio (e.g. 2:1 T0:T1)",
                    )
                    .clicked()
                {
                    if config.tools.is_empty() {
                        config.tools.push(ToolConfig::default());
                    }
                    if config.tools.len() < 2 {
                        let base_nozzle = config.tools[0].nozzle_diameter_mm;
                        let base_temp = config.tools[0].temperature_celsius;
                        let new_id = config.tools[0].id + 1;
                        let mut t1 = ToolConfig::standard_tool(
                            new_id,
                            format!("T{new_id} - {base_nozzle:.1}mm Extruder"),
                            base_nozzle,
                            base_temp,
                        );
                        t1.set_color_from_srgba(crate::config::DEFAULT_FILAMENT_SLOT_PALETTE[1].1);
                        config.tools.push(t1);
                        *dirty_backend = true;
                    }
                    let next_idx = config.filament_slots.len().min(254) as u32;
                    let comps = vec![
                        crate::config::VirtualFilamentComponent {
                            tool_index: 0,
                            ratio: 1,
                        },
                        crate::config::VirtualFilamentComponent {
                            tool_index: 1.min(config.tools.len() - 1),
                            ratio: 1,
                        },
                    ];
                    let mut vslot = crate::config::ThreeMfFilamentSlot {
                        slot_index: next_idx,
                        name: format!("Virtual Mix #{}", next_idx as usize + 1),
                        filament_type: config
                            .tools
                            .first()
                            .map(|t| t.filament_type.clone())
                            .unwrap_or_else(|| "PLA".to_string()),
                        color_hex: "#888888".to_string(),
                        color_srgba: [0.53, 0.53, 0.53, 1.0],
                        tool_index: 0,
                        dither_components: comps,
                        dither_mode: crate::config::VirtualFilamentDitherMode::ByWallAndLayer,
                    };
                    vslot.sync_color_from_dither_components(&config.tools);
                    config.filament_slots.push(vslot);
                    vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                    vis_state.dirty_surface_colors = true;
                    vis_state.dirty_color_mixing = true;
                    vis_state.dirty_toolpaths_mesh = true;
                    *dirty_gcode = true;
                }

                if !config.filament_slots.is_empty() {
                    if ui
                        .small_button("Auto-Create 1:1 Tools (T0..Tn)")
                        .on_hover_text(
                            "Ensure the machine tool registry has at least one tool per 3MF filament slot and map Slot k → Tool Tk",
                        )
                        .clicked()
                    {
                        let needed = config.filament_slots.len();
                        let base_nozzle = config
                            .tools
                            .first()
                            .map(|t| t.nozzle_diameter_mm)
                            .unwrap_or(0.40);
                        let base_temp = config
                            .tools
                            .first()
                            .map(|t| t.temperature_celsius)
                            .unwrap_or(215.0);
                        while config.tools.len() < needed {
                            let new_id = config.tools.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                            let slot_num = config.tools.len() + 1;
                            config.tools.push(ToolConfig::standard_tool(
                                new_id,
                                format!("T{new_id} - Slot {slot_num} ({base_nozzle:.1}mm)"),
                                base_nozzle,
                                base_temp,
                            ));
                        }
                        for (i, slot) in config.filament_slots.iter_mut().enumerate() {
                            slot.slot_index = i as u32;
                            let t_idx = i.min(config.tools.len().saturating_sub(1));
                            slot.tool_index = t_idx;
                            if let Some(tool) = config.tools.get_mut(t_idx) {
                                tool.apply_filament_type_defaults(&slot.filament_type);
                                tool.set_color_from_srgba(slot.color_srgba);
                            }
                        }
                        config.sync_gcode_from_active_tool();
                        *dirty_backend = true;
                        *dirty_gcode = true;
                        vis_state.dirty_toolpaths_mesh = true;
                    }

                    if ui
                        .small_button("Sync Slot Types → Tools")
                        .on_hover_text(
                            "Apply each 3MF filament slot's material type (temperature, fan, density & PA defaults) to its mapped tool",
                        )
                        .clicked()
                    {
                        for slot in &config.filament_slots {
                            if let Some(tool) = config.tools.get_mut(slot.tool_index) {
                                tool.apply_filament_type_defaults(&slot.filament_type);
                            }
                        }
                        config.sync_gcode_from_active_tool();
                        *dirty_backend = true;
                        *dirty_gcode = true;
                    }

                    if ui
                        .small_button("Map All → Active Tool")
                        .on_hover_text("Map all 3MF filament slots to the currently active tool")
                        .clicked()
                    {
                        let active = config
                            .active_tool_index
                            .min(config.tools.len().saturating_sub(1));
                        for slot in &mut config.filament_slots {
                            slot.tool_index = active;
                        }
                        *dirty_gcode = true;
                        vis_state.dirty_toolpaths_mesh = true;
                    }
                }
                *dirty_gcode |= ui
                    .add(
                        egui::DragValue::new(&mut config.gcode.tool_change_time_s)
                            .speed(0.5)
                            .range(0.0..=600.0)
                            .prefix("Toolchange: ")
                            .suffix(" s"),
                    )
                    .on_hover_text(
                        "Estimated duration of each T{id} toolchange / MMU filament swap added to print time estimation",
                    )
                    .changed();
                param_help_button(ui, "gcode.tool_change_time_s");
            });
            ui.horizontal_wrapped(|ui| {
                *dirty_gcode |= ui
                    .checkbox(
                        &mut config.gcode.tool_change_hop_before,
                        "Hop Before Toolchange",
                    )
                    .on_hover_text(
                        "Hop above the highest so-far printed material before emitting T{id}",
                    )
                    .changed();
                param_help_button(ui, "gcode.tool_change_hop_before");
                *dirty_gcode |= ui
                    .checkbox(
                        &mut config.gcode.tool_change_approach_after,
                        "Approach After Toolchange",
                    )
                    .on_hover_text(
                        "After T{id}, move to an approach position above the next chain start (same height as a hop) before descending",
                    )
                    .changed();
                param_help_button(ui, "gcode.tool_change_approach_after");
            });

            // --- Infill / Support Filament Selection & Toolchange Purge Controls ---
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Infill Filament:").strong());
                let cur_infill_lbl = config
                    .slicing
                    .infill_filament_mode
                    .short_label(&config.tools, &config.filament_slots);
                let combo_w = (ui.available_width() - 32.0).clamp(110.0, 215.0);
                let mut infill_changed = false;
                egui::ComboBox::from_id_source((id_salt, "infill_filament_mode_combo"))
                    .selected_text(cur_infill_lbl)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        infill_changed |= ui
                            .selectable_value(
                                &mut config.slicing.infill_filament_mode,
                                InfillFilamentMode::FollowObject,
                                "Follow Object / Region Filament",
                            )
                            .changed();
                        infill_changed |= ui
                            .selectable_value(
                                &mut config.slicing.infill_filament_mode,
                                InfillFilamentMode::AutoPurgeIntoInfill,
                                "Auto (Current Tool / Purge into Infill)",
                            )
                            .changed();
                        ui.separator();
                        for (t_idx, t) in config.tools.iter().enumerate() {
                            infill_changed |= ui
                                .selectable_value(
                                    &mut config.slicing.infill_filament_mode,
                                    InfillFilamentMode::SingleTool(t_idx),
                                    format!("Tool T{} — {} ({})", t.id, t.name, t.filament_color_hex),
                                )
                                .changed();
                        }
                        if !config.filament_slots.is_empty() {
                            ui.separator();
                            for (s_idx, _) in config.filament_slots.iter().enumerate() {
                                let s_u8 = s_idx.min(126) as u8;
                                let s_lbl = InfillFilamentMode::FilamentSlot(s_u8)
                                    .label(&config.tools, &config.filament_slots);
                                infill_changed |= ui
                                    .selectable_value(
                                        &mut config.slicing.infill_filament_mode,
                                        InfillFilamentMode::FilamentSlot(s_u8),
                                        s_lbl,
                                    )
                                    .changed();
                            }
                        }
                    });
                if infill_changed {
                    vis_state.dirty_color_mixing = true;
                    vis_state.dirty_toolpaths_mesh = true;
                    *dirty_gcode = true;
                }
                param_help_button(ui, "slicing.infill_filament_mode");
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Support Base / Intf:").strong());
                let sup_base_lbl = match config.slicing.support_tool_index {
                    None => "Active Tool".to_string(),
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| format!("T{} ({})", t.id, t.filament_type))
                        .unwrap_or_else(|| format!("T{idx}")),
                };
                let sup_intf_lbl = match config.slicing.support_interface_tool_index {
                    None => "Same as Base".to_string(),
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| format!("T{} ({})", t.id, t.filament_type))
                        .unwrap_or_else(|| format!("T{idx}")),
                };
                egui::ComboBox::from_id_source((id_salt, "sup_base_tool_combo"))
                    .selected_text(format!("Base: {sup_base_lbl}"))
                    .width(96.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut config.slicing.support_tool_index,
                                None,
                                "Active Object Tool",
                            )
                            .changed()
                        {
                            *dirty_backend = true;
                            *dirty_gcode = true;
                            vis_state.dirty_color_mixing = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        for (idx, t) in config.tools.iter().enumerate() {
                            if ui
                                .selectable_value(
                                    &mut config.slicing.support_tool_index,
                                    Some(idx),
                                    format!("T{} — {} ({})", t.id, t.name, t.filament_color_hex),
                                )
                                .changed()
                            {
                                *dirty_backend = true;
                                *dirty_gcode = true;
                                vis_state.dirty_color_mixing = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }
                        }
                    });
                egui::ComboBox::from_id_source((id_salt, "sup_intf_tool_combo"))
                    .selected_text(format!("Intf: {sup_intf_lbl}"))
                    .width(96.0)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut config.slicing.support_interface_tool_index,
                                None,
                                "Follow Support Base Tool",
                            )
                            .changed()
                        {
                            *dirty_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        for (idx, t) in config.tools.iter().enumerate() {
                            if ui
                                .selectable_value(
                                    &mut config.slicing.support_interface_tool_index,
                                    Some(idx),
                                    format!("T{} — {} ({})", t.id, t.name, t.filament_color_hex),
                                )
                                .changed()
                            {
                                *dirty_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }
                        }
                    });
                param_help_button(ui, "slicing.support_interface_tool_index");
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Purge Mode:").strong());
                let combo_w = (ui.available_width() - 32.0).clamp(110.0, 215.0);
                let mut purge_mode_changed = false;
                egui::ComboBox::from_id_source((id_salt, "purge_mode_combo"))
                    .selected_text(config.gcode.purge_mode.short_label())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for mode in PurgeMode::ALL {
                            purge_mode_changed |= ui
                                .selectable_value(&mut config.gcode.purge_mode, mode, mode.label())
                                .changed();
                        }
                    });
                if purge_mode_changed {
                    if config.gcode.purge_mode.uses_infill_purge()
                        && config.slicing.infill_filament_mode == InfillFilamentMode::FollowObject
                    {
                        config.slicing.infill_filament_mode =
                            InfillFilamentMode::AutoPurgeIntoInfill;
                    }
                    *dirty_gcode = true;
                    vis_state.dirty_toolpaths_mesh = true;
                }
                param_help_button(ui, "gcode.purge_mode");
            });

            if config.gcode.purge_mode != PurgeMode::Disabled {
                let est_strips =
                    (config.tools.len().saturating_sub(1).max(1) * 3).clamp(3, 12);
                let ([resolved_cx, resolved_cy], [eff_w, eff_d]) =
                    config.resolve_purge_tower_footprint_scene(None, est_strips);
                ui.horizontal_wrapped(|ui| {
                    *dirty_gcode |= ui
                        .add(
                            egui::DragValue::new(&mut config.gcode.min_purge_volume_mm3)
                                .speed(1.0)
                                .range(1.0..=500.0)
                                .prefix("Min Purge: ")
                                .suffix(" mm³"),
                        )
                        .on_hover_text(
                            "Minimum polymer volume (mm³) to purge after each toolchange",
                        )
                        .changed();
                    param_help_button(ui, "gcode.min_purge_volume_mm3");
                    if config.gcode.purge_mode.uses_purge_tower() {
                        *dirty_gcode |= ui
                            .checkbox(&mut config.gcode.purge_tower_auto_dimensions, "Auto Size")
                            .on_hover_text(format!(
                                "Automatically size horizontal rectangular footprint ({eff_w:.1} × {eff_d:.1} mm) so purges pack side-by-side horizontally"
                            ))
                            .changed();
                        *dirty_gcode |= ui
                            .checkbox(&mut config.gcode.purge_tower_brim, "Brim")
                            .changed();
                    }
                });
                if config.gcode.purge_mode.uses_purge_tower() {
                    if config.gcode.purge_tower_auto_dimensions {
                        ui.small(
                            egui::RichText::new(format!(
                                "Horizontal Tower Footprint: {eff_w:.1} × {eff_d:.1} mm (packs side-by-side horizontally)"
                            ))
                            .color(egui::Color32::from_rgb(150, 210, 245)),
                        );
                    } else {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Tower Prism (W × D):");
                            *dirty_gcode |= ui
                                .add(
                                    egui::DragValue::new(&mut config.gcode.purge_tower_width_mm)
                                        .speed(0.5)
                                        .range(8.0..=160.0)
                                        .prefix("W: ")
                                        .suffix(" mm"),
                                )
                                .changed();
                            *dirty_gcode |= ui
                                .add(
                                    egui::DragValue::new(&mut config.gcode.purge_tower_depth_mm)
                                        .speed(0.5)
                                        .range(6.0..=120.0)
                                        .prefix("D: ")
                                        .suffix(" mm"),
                                )
                                .changed();
                            param_help_button(ui, "gcode.purge_tower_auto_dimensions");
                        });
                    }

                    // Pre-slicing & live build-plate placement controls
                    let mut pos_x = if config.gcode.purge_tower_auto_position {
                        resolved_cx
                    } else {
                        config.gcode.purge_tower_position_xy[0]
                    };
                    let mut pos_y = if config.gcode.purge_tower_auto_position {
                        resolved_cy
                    } else {
                        config.gcode.purge_tower_position_xy[1]
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Tower Pos:");
                        if ui
                            .checkbox(&mut config.gcode.purge_tower_auto_position, "Auto")
                            .on_hover_text(
                                "Automatically position the purge tower outside the objects' bounding box on the build plate",
                            )
                            .changed()
                        {
                            *dirty_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        if ui
                            .add(
                                egui::DragValue::new(&mut pos_x)
                                    .speed(0.5)
                                    .range(-250.0..=250.0)
                                    .prefix("X: ")
                                    .suffix(" mm"),
                            )
                            .on_hover_text(
                                "Purge tower X coordinate on the build plate (0 = bed center). Dragging switches to manual placement.",
                            )
                            .changed()
                        {
                            config.gcode.purge_tower_auto_position = false;
                            config.gcode.purge_tower_position_xy[0] = pos_x;
                            config.gcode.purge_tower_position_xy[1] = pos_y;
                            *dirty_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        if ui
                            .add(
                                egui::DragValue::new(&mut pos_y)
                                    .speed(0.5)
                                    .range(-250.0..=250.0)
                                    .prefix("Y: ")
                                    .suffix(" mm"),
                            )
                            .on_hover_text(
                                "Purge tower Y coordinate on the build plate (0 = bed center). Dragging switches to manual placement.",
                            )
                            .changed()
                        {
                            config.gcode.purge_tower_auto_position = false;
                            config.gcode.purge_tower_position_xy[0] = pos_x;
                            config.gcode.purge_tower_position_xy[1] = pos_y;
                            *dirty_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.small("Place:");
                        let hx = (0.5 * config.bed.width_x - 0.5 * eff_w - 6.0).max(10.0);
                        let hy = (0.5 * config.bed.depth_y - 0.5 * eff_d - 6.0).max(10.0);
                        for (lbl, px, py) in [
                            ("Rear", 0.0, (0.65 * hy).min(hy)),
                            ("Front", 0.0, (-0.65 * hy).max(-hy)),
                            ("Right", (0.65 * hx).min(hx), 0.0),
                            ("Left", (-0.65 * hx).max(-hx), 0.0),
                        ] {
                            if ui
                                .small_button(lbl)
                                .on_hover_text(format!(
                                    "Place purge tower at {lbl} of build plate ({px:.1}, {py:.1} mm)"
                                ))
                                .clicked()
                            {
                                config.gcode.purge_tower_auto_position = false;
                                config.gcode.purge_tower_position_xy = [px, py];
                                *dirty_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }
                        }
                    });
                }
            }
            ui.horizontal_wrapped(|ui| {
                if ui
                    .checkbox(
                        &mut config.slicing.color_mixing.enabled,
                        "Optical Colour Mixing (CMYKW / N-Tool)",
                    )
                    .on_hover_text(
                        "Approximate input surface colours via Kubelka-Munk optical filament stacking & short extrusions across loaded tools (T0..Tn)",
                    )
                    .changed()
                {
                    if config.slicing.color_mixing.enabled {
                        vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                    }
                    vis_state.dirty_color_mixing = true;
                    vis_state.dirty_toolpaths_mesh = true;
                    *dirty_gcode = true;
                }
                if ui
                    .small_button("Setup PLA CMYKW (5 Tools)")
                    .on_hover_text(
                        "Configure T0..T4 as Cyan, Magenta, Yellow, Black & White PLA with calibrated Kubelka-Munk Transmission Distances and enable optical colour mixing",
                    )
                    .clicked()
                {
                    apply_quick_cmykw_palette_setup(
                        config,
                        vis_state,
                        &crate::color_optics::bambu_pla_cmykw_palette(),
                    );
                    *dirty_backend = true;
                    *dirty_gcode = true;
                }
                if ui
                    .small_button("Setup Translucent CMYKW")
                    .on_hover_text(
                        "Configure T0..T4 as high-TD translucent Cyan, Magenta, Yellow, Black & opaque White backing",
                    )
                    .clicked()
                {
                    apply_quick_cmykw_palette_setup(
                        config,
                        vis_state,
                        &crate::color_optics::translucent_cmykw_palette(),
                    );
                    *dirty_backend = true;
                    *dirty_gcode = true;
                }
                param_help_button(ui, "slicing.color_mixing.enabled");
            });
            if config.slicing.color_mixing.enabled {
                ui.small(
                    egui::RichText::new(format!(
                        "Optical Mixing Active: 3MF colours/textures act as target surface colours approximated across {} loaded tools (configure full Kubelka-Munk & lithophane settings in Process tab).",
                        config.tools.len()
                    ))
                    .color(egui::Color32::from_rgb(120, 215, 255)),
                );
            }

            if config.filament_slots.is_empty() {
                ui.small(
                    egui::RichText::new(
                        "No multi-colour 3MF filament slots active. Import a multi-colour .3mf or click '+ Add Filament Slot'.",
                    )
                    .color(egui::Color32::GRAY),
                );
                return;
            }

            ui.add_space(3.0);
            let tool_labels: Vec<(usize, String)> = config
                .tools
                .iter()
                .enumerate()
                .map(|(idx, t)| {
                    (
                        idx,
                        format!(
                            "T{} — {} ({:.2}mm, {} {:.0}°C)",
                            t.id, t.name, t.nozzle_diameter_mm, t.filament_type, t.temperature_celsius
                        ),
                    )
                })
                .collect();

            let tools_snapshot = config.tools.clone();
            let mut remove_slot_idx: Option<usize> = None;
            let mut apply_slot_type_to_tool: Option<(usize, String)> = None;
            let total_slots = config.filament_slots.len();
            for (s_idx, slot) in config.filament_slots.iter_mut().enumerate() {
                slot.slot_index = s_idx as u32;
                if slot.tool_index >= tool_labels.len() && !tool_labels.is_empty() {
                    slot.tool_index = tool_labels.len() - 1;
                }
                for comp in &mut slot.dither_components {
                    if comp.tool_index >= tool_labels.len() && !tool_labels.is_empty() {
                        comp.tool_index = tool_labels.len() - 1;
                    }
                    comp.ratio = comp.ratio.clamp(1, 64);
                }

                let is_virtual = slot.is_virtual_mix();
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(format!("#{}", s_idx + 1))
                            .strong()
                            .monospace(),
                    );

                    // Interactive colour swatch picker
                    let mut c32 = egui::Color32::from_rgba_unmultiplied(
                        (slot.color_srgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (slot.color_srgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (slot.color_srgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                        255,
                    );
                    if ui
                        .color_edit_button_srgba(&mut c32)
                        .on_hover_text(format!(
                            "Slot #{} target colour ({}) — click to customize",
                            s_idx + 1,
                            slot.color_hex
                        ))
                        .changed()
                    {
                        slot.color_srgba = [
                            (c32.r() as f32) / 255.0,
                            (c32.g() as f32) / 255.0,
                            (c32.b() as f32) / 255.0,
                            1.0,
                        ];
                        slot.color_hex =
                            format!("#{:02X}{:02X}{:02X}", c32.r(), c32.g(), c32.b());
                        vis_state.dirty_triangle_slot_remap = true;
                        vis_state.dirty_surface_colors = true;
                        vis_state.dirty_color_mixing = true;
                        vis_state.dirty_toolpaths_mesh = true;
                        *dirty_gcode = true;
                    }

                    ui.add(
                        egui::TextEdit::singleline(&mut slot.name)
                            .desired_width(72.0)
                            .hint_text("Slot Name"),
                    )
                    .on_hover_text(format!(
                        "3MF Filament Type: {} | Colour: {}",
                        slot.filament_type, slot.color_hex
                    ));

                    // Editable filament material type for this 3MF slot (also applies defaults to mapped tool)
                    egui::ComboBox::from_id_source((id_salt, "slot_mat_type", s_idx))
                        .selected_text(if slot.filament_type.trim().is_empty() {
                            "PLA"
                        } else {
                            slot.filament_type.as_str()
                        })
                        .width(54.0)
                        .show_ui(ui, |ui| {
                            for mat in ["PLA", "PETG", "ABS", "ASA", "TPU", "PVA", "PA-CF"] {
                                let is_cur = slot.filament_type.eq_ignore_ascii_case(mat);
                                if ui.selectable_label(is_cur, mat).clicked() {
                                    slot.filament_type = mat.to_string();
                                    apply_slot_type_to_tool =
                                        Some((slot.tool_index, mat.to_string()));
                                }
                            }
                            ui.separator();
                            ui.horizontal(|ui| {
                                ui.small("Custom:");
                                if ui
                                    .add(
                                        egui::TextEdit::singleline(&mut slot.filament_type)
                                            .desired_width(60.0),
                                    )
                                    .changed()
                                {
                                    apply_slot_type_to_tool =
                                        Some((slot.tool_index, slot.filament_type.clone()));
                                }
                            });
                        });

                    ui.label("→");

                    if !is_virtual {
                        let current_tool_text = tool_labels
                            .get(slot.tool_index)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_else(|| format!("T{}", slot.tool_index));
                        egui::ComboBox::from_id_source((id_salt, "slot_tool_map", s_idx))
                            .selected_text(current_tool_text)
                            .width(96.0)
                            .show_ui(ui, |ui| {
                                for (t_idx, t_lbl) in &tool_labels {
                                    if ui
                                        .selectable_value(&mut slot.tool_index, *t_idx, t_lbl)
                                        .changed()
                                    {
                                        *dirty_gcode = true;
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                }
                            });

                        if tools_snapshot.len() >= 2
                            && ui
                                .small_button("⇄ Mix Ratio")
                                .on_hover_text(
                                    "Convert this slot into a Virtual Filament composed of 2+ real tools dithered together in an integer ratio",
                                )
                                .clicked()
                        {
                            slot.dither_components =
                                crate::color_optics::find_best_integer_ratio_for_target_srgba(
                                    slot.color_srgba,
                                    &tools_snapshot,
                                    8,
                                );
                            slot.sync_color_from_dither_components(&tools_snapshot);
                            vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                            vis_state.dirty_surface_colors = true;
                            vis_state.dirty_color_mixing = true;
                            vis_state.dirty_toolpaths_mesh = true;
                            *dirty_gcode = true;
                        }
                    } else {
                        let blended = slot.compute_blended_srgba(&tools_snapshot);
                        let b_c32 = egui::Color32::from_rgba_unmultiplied(
                            (blended[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (blended[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                            (blended[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                            255,
                        );
                        let summary = slot.format_ratio_summary(&tools_snapshot);
                        ui.label(
                            egui::RichText::new(format!("Mix {summary}"))
                                .strong()
                                .color(b_c32),
                        )
                        .on_hover_text(
                            "Virtual Filament integer ratio across real tools (colour text shows optically blended ratio result)",
                        );

                        egui::ComboBox::from_id_source((id_salt, "slot_dither_mode", s_idx))
                            .selected_text(slot.dither_mode.short_label())
                            .width(92.0)
                            .show_ui(ui, |ui| {
                                for dm in crate::config::VirtualFilamentDitherMode::ALL {
                                    if ui
                                        .selectable_value(&mut slot.dither_mode, dm, dm.label())
                                        .changed()
                                    {
                                        vis_state.dirty_color_mixing = true;
                                        vis_state.dirty_toolpaths_mesh = true;
                                        *dirty_gcode = true;
                                    }
                                }
                            });

                        if ui
                            .small_button("1:1")
                            .on_hover_text("Revert this Virtual Filament back to a single 1:1 tool mapping")
                            .clicked()
                        {
                            slot.tool_index = slot.primary_tool_index(tools_snapshot.len());
                            slot.dither_components.clear();
                            vis_state.dirty_surface_colors = true;
                            vis_state.dirty_color_mixing = true;
                            vis_state.dirty_toolpaths_mesh = true;
                            *dirty_gcode = true;
                        }
                    }

                    if total_slots > 1
                        && ui
                            .small_button("✕")
                            .on_hover_text("Remove this filament slot")
                            .clicked()
                    {
                        remove_slot_idx = Some(s_idx);
                    }
                });

                // Render indented integer-ratio component controls when `slot` is a Virtual Filament
                if slot.is_virtual_mix() {
                    let mut ratio_changed = false;
                    let mut remove_comp_idx: Option<usize> = None;
                    let comp_count = slot.dither_components.len();
                    let total_parts = slot.total_ratio_parts().max(1) as f32;

                    ui.indent((id_salt, "virtual_mix_indent", s_idx), |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for (c_idx, comp) in slot.dither_components.iter_mut().enumerate() {
                                if c_idx > 0 {
                                    ui.label(egui::RichText::new(":").strong());
                                }
                                let pct = ((comp.ratio.max(1) as f32) / total_parts * 100.0).round();
                                ratio_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut comp.ratio)
                                            .speed(0.15)
                                            .range(1..=64)
                                            .suffix("×"),
                                    )
                                    .on_hover_text(format!(
                                        "Integer ratio weight for component #{} ({:.0}% of virtual filament)",
                                        c_idx + 1,
                                        pct
                                    ))
                                    .changed();

                                let short_tool = tools_snapshot
                                    .get(comp.tool_index)
                                    .map(|t| format!("T{} ({})", t.id, t.filament_color_hex))
                                    .unwrap_or_else(|| format!("T{}", comp.tool_index));
                                egui::ComboBox::from_id_source((
                                    id_salt,
                                    "virtual_comp_tool",
                                    s_idx,
                                    c_idx,
                                ))
                                .selected_text(short_tool)
                                .width(102.0)
                                .show_ui(ui, |ui| {
                                    for (t_idx, t_lbl) in &tool_labels {
                                        if ui
                                            .selectable_value(&mut comp.tool_index, *t_idx, t_lbl)
                                            .changed()
                                        {
                                            ratio_changed = true;
                                        }
                                    }
                                });

                                if comp_count > 2
                                    && ui
                                        .small_button("−")
                                        .on_hover_text("Remove this component from the Virtual Filament")
                                        .clicked()
                                {
                                    remove_comp_idx = Some(c_idx);
                                }
                            }

                            if slot.dither_components.len() < tools_snapshot.len().max(2)
                                && ui
                                    .small_button("+ Tool")
                                    .on_hover_text("Add another real filament tool to this Virtual Filament mix")
                                    .clicked()
                            {
                                let next_t = (0..tools_snapshot.len())
                                    .find(|&ti| {
                                        !slot
                                            .dither_components
                                            .iter()
                                            .any(|c| c.tool_index == ti)
                                    })
                                    .unwrap_or(0);
                                slot.dither_components.push(
                                    crate::config::VirtualFilamentComponent {
                                        tool_index: next_t,
                                        ratio: 1,
                                    },
                                );
                                ratio_changed = true;
                            }

                            if ui
                                .small_button("Auto-Fit")
                                .on_hover_text(
                                    "Automatically compute the optimal integer ratio of loaded real tools to match this slot's target colour",
                                )
                                .clicked()
                            {
                                slot.dither_components =
                                    crate::color_optics::find_best_integer_ratio_for_target_srgba(
                                        slot.color_srgba,
                                        &tools_snapshot,
                                        8,
                                    );
                                ratio_changed = true;
                            }
                        });
                    });

                    if let Some(rc_idx) = remove_comp_idx {
                        if slot.dither_components.len() > 2 {
                            slot.dither_components.remove(rc_idx);
                            ratio_changed = true;
                        }
                    }
                    if ratio_changed {
                        slot.sync_color_from_dither_components(&tools_snapshot);
                        vis_state.dirty_surface_colors = true;
                        vis_state.dirty_color_mixing = true;
                        vis_state.dirty_toolpaths_mesh = true;
                        *dirty_gcode = true;
                    }
                }
            }

            if let Some((t_idx, mat_type)) = apply_slot_type_to_tool {
                if let Some(tool) = config.tools.get_mut(t_idx) {
                    tool.apply_filament_type_defaults(&mat_type);
                }
                config.sync_gcode_from_active_tool();
                *dirty_backend = true;
                *dirty_gcode = true;
            }

            if let Some(rm_idx) = remove_slot_idx {
                config.filament_slots.remove(rm_idx);
                for (i, s) in config.filament_slots.iter_mut().enumerate() {
                    s.slot_index = i as u32;
                }
                vis_state.dirty_surface_colors = true;
                vis_state.dirty_toolpaths_mesh = true;
                *dirty_gcode = true;
            }
        });
}

/// Renders the `Filament` configuration sub-tab:
/// Material & Temperatures, Flow Multipliers, Part Cooling Fan Schedule,
/// and Non-Newtonian Rheology (Dynamic PA, Die Swell, Corner Flow Comp, Transient Pressure, Ooze Relaxation).
fn render_filament_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    calib_state: &mut CalibrationUiState,
    preset_lib: &mut PresetLibrary,
    is_computing: bool,
) {
    let mut dirty_backend = false;
    let mut gc_changed = false;

    if config.tools.len() > 1 {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::symmetric(8.0, 5.0))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Editing Tool Filament:").strong());
                    let current_active = config.active_tool_index;
                    let mut clicked_idx = None;
                    for (idx, tool) in config.tools.iter().enumerate() {
                        let selected = current_active == idx;
                        if render_delineated_pill_tab(
                            ui,
                            selected,
                            format!(
                                "T{} ({}, {:.0}°C)",
                                tool.id, tool.filament_type, tool.temperature_celsius
                            ),
                        )
                        .on_hover_text(format!(
                            "Edit filament settings & preset for Tool T{} ({})",
                            tool.id, tool.name
                        ))
                        .clicked()
                        {
                            clicked_idx = Some(idx);
                        }
                    }
                    if let Some(idx) = clicked_idx {
                        config.active_tool_index = idx;
                        config.sync_gcode_from_active_tool();
                        if preset_lib
                            .filaments
                            .iter()
                            .any(|p| p.name == config.filament_preset_name)
                        {
                            preset_lib.selected_filament = config.filament_preset_name.clone();
                        }
                        dirty_backend = true;
                        vis_state.dirty_planned_gcode = true;
                    }
                });
            });
        ui.add_space(4.0);
    }

    egui::CollapsingHeader::new(
        egui::RichText::new("Material, Diameter & Temperatures").strong(),
    )
    .id_source("filament_material_temp_section")
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Material:");
            gc_changed |= ui
                .add(
                    egui::TextEdit::singleline(&mut config.gcode.filament_type)
                        .desired_width(64.0),
                )
                .changed();
            let mut apply_quick_mat: Option<&'static str> = None;
            for (preset_name, density, fan_pct) in [
                ("PLA", 1.24f32, 100.0f32),
                ("PETG", 1.27, 40.0),
                ("ABS", 1.04, 15.0),
                ("ASA", 1.07, 15.0),
                ("TPU", 1.21, 50.0),
                ("PVA", 1.23, 70.0),
                ("PA-CF", 1.15, 20.0),
            ] {
                if ui
                    .selectable_label(
                        config
                            .gcode
                            .filament_type
                            .eq_ignore_ascii_case(preset_name),
                        preset_name,
                    )
                    .on_hover_text(format!(
                        "Set active tool material to {preset_name} (density {density:.2} g/cm³, fan {fan_pct:.0}%)"
                    ))
                    .clicked()
                {
                    apply_quick_mat = Some(preset_name);
                }
            }
            if let Some(mat_name) = apply_quick_mat {
                let active_idx = config.active_tool_index;
                if let Some(tool) = config.tools.get_mut(active_idx) {
                    tool.apply_filament_type_defaults(mat_name);
                }
                config.sync_gcode_from_active_tool();
                if preset_lib
                    .filaments
                    .iter()
                    .any(|p| p.name == config.filament_preset_name)
                {
                    preset_lib.selected_filament = config.filament_preset_name.clone();
                }
                dirty_backend = true;
                gc_changed = true;
            }
            param_help_button(ui, "gcode.filament_type");
        });

        ui.horizontal_wrapped(|ui| {
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.gcode.filament_diameter_mm)
                        .speed(0.01)
                        .range(1.0..=3.5)
                        .prefix("Diam: ")
                        .suffix(" mm"),
                )
                .changed();
            if ui.small_button("1.75").clicked() {
                config.gcode.filament_diameter_mm = 1.75;
                gc_changed = true;
            }
            if ui.small_button("2.85").clicked() {
                config.gcode.filament_diameter_mm = 2.85;
                gc_changed = true;
            }
            param_help_button(ui, "gcode.filament_diameter_mm");

            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.gcode.filament_density_g_cm3)
                        .speed(0.01)
                        .range(0.5..=4.0)
                        .prefix("Density: ")
                        .suffix(" g/cm³"),
                )
                .changed();
            param_help_button(ui, "gcode.filament_density_g_cm3");
        });

        let active_idx = config.active_tool_index;
        if let Some(tool) = config.tools.get_mut(active_idx) {
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Slider::new(&mut tool.temperature_celsius, 170.0..=350.0)
                            .text("Nozzle Temperature (°C)"),
                    )
                    .changed()
                {
                    dirty_backend = true;
                    gc_changed = true;
                }
                param_help_button(ui, "tools.temperature_celsius");
            });
            if render_tool_optical_properties_controls(ui, tool) {
                vis_state.dirty_color_mixing = true;
                vis_state.dirty_toolpaths_mesh = true;
                vis_state.dirty_surface_colors = true;
                gc_changed = true;
            }
        }
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut config.gcode.bed_temperature_celsius, 0.0..=130.0)
                        .text("Bed Temperature (°C)"),
                )
                .changed();
            param_help_button(ui, "gcode.bed_temperature_celsius");
        });
    });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Flow Multipliers & Part Cooling Fan").strong(),
    )
    .id_source("filament_flow_cooling_section")
    .default_open(true)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.extrusion_multiplier, 0.50..=1.50)
                        .suffix("×")
                        .text("Extrusion Mult"),
                )
                .changed();
            if ui
                .small_button("Calibrate EM...")
                .on_hover_text(
                    "Open Orca-style Extrusion Multiplier & High-Flow Die Swell calibration generator",
                )
                .clicked()
            {
                calib_state.show_window = true;
                calib_state.active_kind = CalibrationUtilityKind::ExtrusionMultiplier;
            }
            param_help_button(ui, "gcode.extrusion_multiplier");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.first_layer_flow_multiplier, 0.70..=1.50)
                        .suffix("×")
                        .text("1st Layer Mult"),
                )
                .changed();
            param_help_button(ui, "gcode.first_layer_flow_multiplier");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.bridge_flow_multiplier, 0.50..=1.50)
                        .suffix("×")
                        .text("Bridge Flow Mult"),
                )
                .changed();
            param_help_button(ui, "gcode.bridge_flow_multiplier");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.wave_overhang_flow_multiplier, 0.50..=1.50)
                        .suffix("×")
                        .text("Wave Overhang Mult"),
                )
                .changed();
            param_help_button(ui, "gcode.wave_overhang_flow_multiplier");
        });

        ui.separator();
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.fan_speed_percent, 0.0..=100.0)
                        .suffix("%")
                        .text("Part Fan Speed"),
                )
                .changed();
            param_help_button(ui, "gcode.fan_speed_percent");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.bridge_fan_speed_percent, 0.0..=100.0)
                        .suffix("%")
                        .text("Bridge Fan Speed"),
                )
                .changed();
            param_help_button(ui, "gcode.bridge_fan_speed_percent");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.wave_overhang_fan_speed_percent, 0.0..=100.0)
                        .suffix("%")
                        .text("Wave Fan Speed"),
                )
                .changed();
            param_help_button(ui, "gcode.wave_overhang_fan_speed_percent");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.first_layer_fan_speed_percent, 0.0..=100.0)
                        .suffix("%")
                        .text("Initial Fan Speed"),
                )
                .changed();
            param_help_button(ui, "gcode.first_layer_fan_speed_percent");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.disable_fan_first_layers)
                        .speed(0.1)
                        .range(0..=20)
                        .prefix("Use Initial Fan for First: ")
                        .suffix(" layer(s)"),
                )
                .changed();
            param_help_button(ui, "gcode.disable_fan_first_layers");
        });
        ui.separator();
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_dynamic_fan_speed,
                    "Dynamic Volumetric & Airflow Cooling Fan",
                )
                .changed();
            param_help_button(ui, "gcode.enable_dynamic_fan_speed");
        });
        if gc.enable_dynamic_fan_speed {
            ui.horizontal(|ui| {
                ui.label("Dynamic Fan Range:");
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.min_fan_speed_percent)
                            .speed(1.0)
                            .range(0.0..=100.0)
                            .prefix("Min: ")
                            .suffix("%"),
                    )
                    .changed();
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.max_fan_speed_percent)
                            .speed(1.0)
                            .range(0.0..=100.0)
                            .prefix("Max: ")
                            .suffix("%"),
                    )
                    .changed();
            });
            ui.horizontal(|ui| {
                gc_changed |= ui
                    .add(
                        egui::Slider::new(&mut gc.target_layer_cooling_time_s, 1.0..=60.0)
                            .suffix(" s")
                            .text("Target Layer Cooling Time"),
                    )
                    .changed();
            });
        }
    });

    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Rheology, Dynamic PA, Die Swell & Ooze").strong(),
    )
    .id_source("filament_rheology_pa_section")
    .default_open(true)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.pa_low_flow_s, 0.0..=0.25)
                        .text("PA Low Flow k_low (s)"),
                )
                .changed();
            if ui
                .small_button("Calibrate PA...")
                .on_hover_text(
                    "Open Ellis-style 90° sharp-corner chevron Pressure Advance calibration generator",
                )
                .clicked()
            {
                calib_state.show_window = true;
                calib_state.active_kind = CalibrationUtilityKind::PressureAdvance;
            }
            param_help_button(ui, "gcode.pa_low_high_s");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::Slider::new(&mut gc.pa_high_flow_s, 0.0..=0.25)
                        .text("PA High Flow k_high (s)"),
                )
                .changed();
            param_help_button(ui, "gcode.pa_low_high_s");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("PA Ref Flows (mm³/s):");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.pa_ref_low_flow_mm3s)
                        .speed(0.2)
                        .range(0.2..=25.0)
                        .prefix("Q_low: "),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.pa_ref_high_flow_mm3s)
                        .speed(0.5)
                        .range(1.0..=80.0)
                        .prefix("Q_high: "),
                )
                .changed();
            param_help_button(ui, "gcode.pa_ref_flows_mm3s");
        });

        let mut low_flow_swell_mult = (1.0 / gc.swell_low.max(1.0)).clamp(0.50, 1.00);
        let mut high_flow_swell_mult = (1.0 / gc.swell_high.max(1.0)).clamp(0.50, 1.00);
        let mut swell_mult_changed = false;
        ui.horizontal(|ui| {
            swell_mult_changed |= ui
                .add(
                    egui::Slider::new(&mut low_flow_swell_mult, 0.60..=1.00)
                        .suffix("×")
                        .text("Low-Flow Swell Mult (1/swell)"),
                )
                .changed();
            param_help_button(ui, "gcode.swell_multipliers");
        });
        ui.horizontal(|ui| {
            swell_mult_changed |= ui
                .add(
                    egui::Slider::new(&mut high_flow_swell_mult, 0.60..=1.00)
                        .suffix("×")
                        .text("High-Flow Swell Mult (1/swell)"),
                )
                .changed();
            param_help_button(ui, "gcode.swell_multipliers");
        });
        if swell_mult_changed {
            gc.swell_low = (1.0 / low_flow_swell_mult.clamp(0.50, 1.00)).clamp(1.0, 2.0);
            gc.swell_high =
                (1.0 / high_flow_swell_mult.clamp(0.50, 1.00)).clamp(gc.swell_low, 2.0);
            gc_changed = true;
        }

        ui.horizontal_wrapped(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_transient_pressure_comp,
                    "Melt-Zone Transient Pressure Compensation",
                )
                .changed();
            if ui
                .small_button("Calibrate...")
                .on_hover_text(
                    "Open Constant-Speed Flow-Step & Staccato Striation calibration generator for γ and τ_melt",
                )
                .clicked()
            {
                calib_state.show_window = true;
                calib_state.active_kind = CalibrationUtilityKind::TransientPressure;
            }
            param_help_button(ui, "gcode.transient_pressure_comp");
        });
        if gc.enable_transient_pressure_comp {
            ui.horizontal_wrapped(|ui| {
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.transient_pressure_exponent)
                            .speed(0.01)
                            .range(0.05..=1.0)
                            .prefix("Exp γ: "),
                    )
                    .changed();
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.transient_pressure_tau_s)
                            .speed(0.01)
                            .range(0.01..=5.0)
                            .prefix("τ_melt: ")
                            .suffix(" s"),
                    )
                    .changed();
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.transient_min_flow_multiplier)
                            .speed(0.01)
                            .range(0.40..=1.0)
                            .prefix("Min Mult: ")
                            .suffix("×"),
                    )
                    .changed();
                param_help_button(ui, "gcode.transient_pressure_comp");
            });
        }
        ui.horizontal_wrapped(|ui| {
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.ooze_relaxation_extra_prime_mm)
                        .speed(0.005)
                        .range(-0.30..=0.30)
                        .prefix("Ooze Prime: ")
                        .suffix(" mm"),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.ooze_relaxation_tau_s)
                        .speed(0.05)
                        .range(0.10..=15.0)
                        .prefix("τ: ")
                        .suffix(" s"),
                )
                .changed();
            param_help_button(ui, "gcode.ooze_relaxation");
        });
    });

    if dirty_backend && !is_computing {
        sync_config_to_backend(config);
    }
    if gc_changed {
        config.gcode.sanitize();
        config.sync_active_tool_from_gcode();
        vis_state.dirty_planned_gcode = true;
    }
}

/// Renders the `Process` configuration sub-tab:
/// Variable Layer Height, Volumetric Eikonal Solver & Overhang Bias, Shells & Line Widths,
/// Infill, Overhangs & Support Generation, 3D Scarf Seams, and Feature Print Speeds.
fn render_process_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    plate: &mut BuildPlateState,
    selected_obj_info: Option<&(u64, usize, String, Quat)>,
    is_computing: bool,
    has_eikonal: bool,
) {
    let active_nozzle_mm = config
        .active_tool()
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(0.40);

    // 1. Variable Slice Layer Height & Non-Planar Slope
    egui::CollapsingHeader::new(
        egui::RichText::new("Variable Slice Layer Height & Slope").strong(),
    )
    .id_source("process_layer_height_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut lh_changed = false;
        ui.horizontal(|ui| {
            lh_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.layer_height.target_mm, 0.04..=1.20)
                        .text("Target Thickness (mm)"),
                )
                .changed();
            param_help_button(ui, "slicing.layer_height.target_mm");
        });
        ui.horizontal_wrapped(|ui| {
            lh_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.layer_height.min_mm)
                        .speed(0.01)
                        .range(0.02..=1.00)
                        .prefix("Min: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "slicing.layer_height.min_mm");
            lh_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.layer_height.max_mm)
                        .speed(0.01)
                        .range(0.05..=1.50)
                        .prefix("Max: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "slicing.layer_height.max_mm");
        });
        ui.horizontal_wrapped(|ui| {
            lh_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.layer_height.first_layer_mm)
                        .speed(0.01)
                        .range(0.02..=1.50)
                        .prefix("First Layer: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "slicing.layer_height.first_layer_mm");
            lh_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.layer_height.top_layer_mm)
                        .speed(0.01)
                        .range(0.02..=1.50)
                        .prefix("Top Layer: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "slicing.layer_height.top_layer_mm");
        });
        ui.horizontal(|ui| {
            lh_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.max_nonplanar_slope_deg, 0.0..=65.0)
                        .text("Max Non-Planar Slope (°)"),
                )
                .changed();
            param_help_button(ui, "slicing.max_nonplanar_slope_deg");
        });

        ui.separator();
        let mut skin_field_changed = false;
        ui.horizontal_wrapped(|ui| {
            let auto_changed = ui
                .checkbox(
                    &mut config.slicing.enable_auto_conformal_skin,
                    "Auto Conformal SDF Top Skin",
                )
                .on_hover_text(
                    "Automatically erode N layers of top/exit surfaces where the order field arrives at an oblique angle and slice them conformally along SDF level sets (standing off around rising protrusions).",
                )
                .changed();
            lh_changed |= auto_changed;
            skin_field_changed |= auto_changed;
            param_help_button(ui, "slicing.enable_auto_conformal_skin");
            lh_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.conformal_skin_layers)
                        .speed(0.2)
                        .range(1..=12)
                        .prefix("Skin Layers: "),
                )
                .on_hover_text("Default number of conformal SDF skin layers (N) for auto-detected or newly hinted conformal skin surfaces.")
                .changed();
            param_help_button(ui, "slicing.conformal_skin_layers");
            lh_changed |= ui
                .checkbox(
                    &mut config.slicing.one_perimeter_on_conformal_patches,
                    "One Perimeter on Conformal Patches",
                )
                .on_hover_text(
                    "Print only the outer wall on the outermost layer (m = N) of each conformal patch, replacing inner walls with solid infill.",
                )
                .changed();
            param_help_button(ui, "slicing.one_perimeter_on_conformal_patches");
            if config.slicing.enable_auto_conformal_skin {
                lh_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.conformal_skin_min_oblique_deg)
                            .speed(0.5)
                            .range(2.0..=75.0)
                            .prefix("Min Oblique: ")
                            .suffix("°"),
                    )
                    .on_hover_text("Minimum angle between the order field gradient (∇τ) and the outward surface normal (∇φ) required to trigger automatic conformal skinning.")
                    .changed();
                param_help_button(ui, "slicing.conformal_skin_min_oblique_deg");
                let slope_changed = ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.conformal_skin_max_slope_deg)
                            .speed(0.5)
                            .range(5.0..=85.0)
                            .prefix("Max Slope: ")
                            .suffix("°"),
                    )
                    .on_hover_text("Maximum surface slope relative to the local nozzle axis (ẑ_nozzle) eligible for automatic conformal skinning.")
                    .changed();
                lh_changed |= slope_changed;
                skin_field_changed |= slope_changed;
                param_help_button(ui, "slicing.conformal_skin_max_slope_deg");
            }
        });

        if lh_changed {
            config.slicing.sanitize();
            if config.eikonal.auto_voxel_size {
                config.sync_auto_voxel_size();
            }
            if !is_computing {
                sync_config_to_backend(config);
            }
            if skin_field_changed
                && has_eikonal
                && !is_computing
                && config.eikonal.order_field_mode == OrderFieldMode::ConstrainedNonPlanar
            {
                if let Some(&(sel_id, _, ref obj_name, obj_rotation)) = selected_obj_info {
                    spawn_field_compute_task(
                        vis_state,
                        plate,
                        config,
                        sel_id,
                        obj_name,
                        obj_rotation,
                    );
                }
            }
            vis_state.dirty_selected_layers = true;
            vis_state.dirty_toolpaths = true;
        }
    });

    // 2. Volumetric Order Field & Eikonal Solver
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Volumetric Order Field & Eikonal Solver").strong(),
    )
    .id_source("process_eikonal_solver_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut eik_changed = false;
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Order Field:").strong());
            for mode in OrderFieldMode::ALL {
                let selected = config.eikonal.order_field_mode == mode;
                if render_delineated_pill_tab(ui, selected, mode.label())
                    .on_hover_text(match mode {
                        OrderFieldMode::ConstrainedNonPlanar => {
                            "Constrained non-planar Eikonal solver with nozzle-clearance cone projection and overhang surface bias."
                        }
                        OrderFieldMode::HeightOnlyPlanar => {
                            "Height-only planar order field τ(x) = max(0, x · ẑ_nozzle − z_min,bed) evaluated along the queried nozzle axis with 0° tool tilt."
                        }
                    })
                    .clicked()
                    && !selected
                {
                    config.eikonal.order_field_mode = mode;
                    eik_changed = true;
                }
            }
            param_help_button(ui, "eikonal.order_field_mode");
        });
        if config.eikonal.order_field_mode == OrderFieldMode::HeightOnlyPlanar {
            ui.small(
                egui::RichText::new(
                    "Planar mode: τ(x) = max(0, x · ẑ_nozzle(x) − z_min,bed) along the queried nozzle frame (0° tilt). Gravity-aware overhang and support detection remain active.",
                )
                .color(egui::Color32::from_rgb(135, 205, 245)),
            );
        }
        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            if ui
                .checkbox(
                    &mut config.eikonal.auto_voxel_size,
                    "Auto Fine Voxel (Nozzle + Layer)",
                )
                .changed()
            {
                if config.eikonal.auto_voxel_size {
                    config.sync_auto_voxel_size();
                }
                eik_changed = true;
            }
            param_help_button(ui, "eikonal.auto_voxel_size");

            eik_changed |= ui
                .checkbox(
                    &mut config.eikonal.enable_multi_resolution_grid,
                    "Dual-Grid Multi-Res",
                )
                .changed();
            param_help_button(ui, "eikonal.enable_multi_resolution_grid");

            if config.eikonal.enable_multi_resolution_grid {
                eik_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.eikonal.coarse_grid_ratio)
                            .speed(0.1)
                            .range(2..=8)
                            .prefix("Coarse: ")
                            .suffix("×"),
                    )
                    .changed();
                param_help_button(ui, "eikonal.coarse_grid_ratio");
            }
        });

        let non_planar_active =
            config.eikonal.order_field_mode == OrderFieldMode::ConstrainedNonPlanar;
        ui.horizontal_wrapped(|ui| {
            if ui
                .add(
                    egui::Slider::new(&mut config.eikonal.voxel_size_mm, 0.08..=2.0)
                        .text("Fine Voxel (mm)"),
                )
                .changed()
            {
                config.eikonal.auto_voxel_size = false;
                eik_changed = true;
            }
            param_help_button(ui, "eikonal.voxel_size_mm");
            ui.add_enabled_ui(non_planar_active, |ui| {
                eik_changed |= ui
                    .add(
                        egui::Slider::new(&mut config.eikonal.solver_iterations, 1..=12)
                            .text("Iters"),
                    )
                    .changed();
                param_help_button(ui, "eikonal.solver_iterations");
            });
        });

        ui.horizontal_wrapped(|ui| {
            eik_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.eikonal.narrow_band_width_voxels)
                        .speed(0.5)
                        .range(2..=24)
                        .prefix("Band: ")
                        .suffix(" vox"),
                )
                .changed();
            param_help_button(ui, "eikonal.narrow_band_width_voxels");
            ui.add_enabled_ui(non_planar_active, |ui| {
                eik_changed |= ui
                    .add(
                        egui::Slider::new(&mut config.eikonal.curvature_regularization, 0.0..=1.0)
                            .text("Curvature Reg."),
                    )
                    .changed();
                param_help_button(ui, "eikonal.curvature_regularization");
            });
        });

        ui.separator();
        ui.add_enabled_ui(non_planar_active, |ui| {
            ui.horizontal_wrapped(|ui| {
                eik_changed |= ui
                    .checkbox(
                        &mut config.eikonal.enable_overhang_orientation_bias,
                        "Overhang Conformal / Orthogonal Surface Bias",
                    )
                    .changed();
                param_help_button(ui, "eikonal.enable_overhang_orientation_bias");
            });
            if config.eikonal.enable_overhang_orientation_bias {
                ui.horizontal_wrapped(|ui| {
                    eik_changed |= ui
                        .add(
                            egui::Slider::new(&mut config.eikonal.overhang_bias_weight, 0.0..=1.0)
                                .text("Bias Weight"),
                        )
                        .changed();
                    param_help_button(ui, "eikonal.overhang_bias_weight");
                    eik_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.eikonal.overhang_bias_transition_deg)
                                .speed(0.5)
                                .range(1.0..=45.0)
                                .prefix("Transition: ")
                                .suffix("°"),
                        )
                        .changed();
                    param_help_button(ui, "eikonal.overhang_bias_transition_deg");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.small("Orthogonal Band:");
                    eik_changed |= ui
                        .add(
                            egui::DragValue::new(
                                &mut config.eikonal.orthogonal_supported_angle_deg,
                            )
                            .speed(0.5)
                            .range(0.0..=89.0)
                            .prefix("Supported ≤ ")
                            .suffix("°"),
                        )
                        .changed();
                    param_help_button(ui, "eikonal.orthogonal_supported_angle_deg");
                    eik_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.eikonal.orthogonal_overhang_angle_deg)
                                .speed(0.5)
                                .range(0.0..=89.0)
                                .prefix("Overhang ≤ ")
                                .suffix("°"),
                        )
                        .changed();
                    param_help_button(ui, "eikonal.orthogonal_overhang_angle_deg");
                });
            }
        });

        ui.separator();
        ui.small("Default Soft Hint Placement Parameters:");
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::DragValue::new(&mut config.eikonal.default_surface_hint_depth_mm)
                    .speed(0.5)
                    .range(1.0..=30.0)
                    .prefix("Face Depth: ")
                    .suffix(" mm"),
            );
            param_help_button(ui, "eikonal.default_surface_hint_depth_mm");
            ui.add(
                egui::DragValue::new(&mut config.eikonal.default_point_hint_radius_mm)
                    .speed(0.5)
                    .range(1.5..=40.0)
                    .prefix("Pt Radius: ")
                    .suffix(" mm"),
            );
            param_help_button(ui, "eikonal.default_point_hint_radius_mm");
            ui.add(
                egui::DragValue::new(&mut config.eikonal.default_hint_weight)
                    .speed(0.02)
                    .range(0.05..=1.0)
                    .prefix("Weight: "),
            );
            param_help_button(ui, "eikonal.default_hint_weight");
        });

        if eik_changed {
            config.eikonal.sanitize();
            if !is_computing {
                sync_config_to_backend(config);
            }
        }
    });

    // 3. Shells, Perimeters, Line Widths & Infill
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Shells, Line Widths & Infill").strong(),
    )
    .id_source("process_shells_infill_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut tp_cfg_changed = false;
        let mut gc_changed = false;

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.shell_thickness_mm, 0.20..=8.00)
                        .text("Shell Thickness (-φ_sdf <= T_shell mm)"),
                )
                .changed();
            param_help_button(ui, "slicing.shell_thickness_mm");
        });

        ui.horizontal_wrapped(|ui| {
            tp_cfg_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.inner_wall_count)
                        .speed(0.1)
                        .range(0..=12)
                        .prefix("Inner Walls: "),
                )
                .changed();
            param_help_button(ui, "slicing.inner_wall_count");
            tp_cfg_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.min_extrusion_length_mm)
                        .speed(0.02)
                        .range(0.0..=10.0)
                        .prefix("Min Len: ")
                        .suffix(" mm"),
                )
                .changed();
            if ui.small_button("2× Nozzle").clicked() {
                config.slicing.min_extrusion_length_mm = 2.0 * active_nozzle_mm;
                tp_cfg_changed = true;
            }
            param_help_button(ui, "slicing.min_extrusion_length_mm");
        });

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .checkbox(
                    &mut config.slicing.one_perimeter_on_conformal_patches,
                    "One Perimeter on Conformal Patches",
                )
                .on_hover_text(
                    "Print only the outer wall on the outermost layer (m = N) of each conformal patch, replacing inner walls with solid infill.",
                )
                .changed();
            param_help_button(ui, "slicing.one_perimeter_on_conformal_patches");
        });

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .checkbox(
                    &mut config.slicing.enable_top_surface_lofting,
                    "Top-Surface Lofting",
                )
                .on_hover_text(
                    "Lift exposed top-surface inner walls and solid infill along the local nozzle axis toward the CAD surface (up to max layer height) and scale extrusion volume to smooth shallow non-conformal slopes.",
                )
                .changed();
            param_help_button(ui, "slicing.enable_top_surface_lofting");
        });

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .checkbox(
                    &mut config.slicing.orient_outer_surfaces_to_normal,
                    "Orient Tool to Outer Surface Normal (else Isosurface Normal)",
                )
                .on_hover_text(
                    "On multi-axis tools, if the tool can be oriented within the 'Almost Parallel' angle of the outward surface normal on outer surfaces, do that; else use the isosurface normal.",
                )
                .changed();
            param_help_button(ui, "slicing.orient_outer_surfaces_to_normal");
        });
        if config.slicing.orient_outer_surfaces_to_normal {
            ui.horizontal(|ui| {
                tp_cfg_changed |= ui
                    .add(
                        egui::Slider::new(
                            &mut config.slicing.outer_surface_parallel_angle_deg,
                            0.0..=85.0,
                        )
                        .suffix("°")
                        .text("Almost Parallel Angle Threshold"),
                    )
                    .on_hover_text(
                        "Maximum angle between the reachable clamped tool axis and the outward surface normal considered 'almost parallel'; surfaces steeper than this fall back to the isosurface normal.",
                    )
                    .changed();
                param_help_button(ui, "slicing.outer_surface_parallel_angle_deg");
            });
        }

        ui.horizontal(|ui| {
            ui.label("Wall Order:");
            egui::ComboBox::from_id_source("process_wall_order_combo")
                .selected_text(config.gcode.wall_print_order.label())
                .show_ui(ui, |ui| {
                    for order in [WallPrintOrder::InnerOuterInner, WallPrintOrder::OutsideIn] {
                        gc_changed |= ui
                            .selectable_value(
                                &mut config.gcode.wall_print_order,
                                order,
                                order.label(),
                            )
                            .changed();
                    }
                });
            param_help_button(ui, "gcode.wall_print_order");
        });

        ui.horizontal(|ui| {
            ui.label("Brick Layers:");
            egui::ComboBox::from_id_source("process_brick_layer_mode_combo")
                .selected_text(config.slicing.brick_layer_mode.label())
                .show_ui(ui, |ui| {
                    for mode in BrickLayerMode::ALL {
                        tp_cfg_changed |= ui
                            .selectable_value(
                                &mut config.slicing.brick_layer_mode,
                                mode,
                                mode.label(),
                            )
                            .changed();
                    }
                });
            param_help_button(ui, "slicing.brick_layer_mode");
        });

        ui.separator();
        ui.horizontal(|ui| {
            ui.small(egui::RichText::new("Line Widths (Target / Min / Max mm):").strong());
            param_help_button(ui, "slicing.line_widths");
        });
        for (label, lw) in [
            ("Outer Wall:", &mut config.slicing.outer_wall_width),
            ("Inner Wall:", &mut config.slicing.inner_wall_width),
            ("Solid Infill:", &mut config.slicing.solid_infill_width),
            ("Sparse Infill:", &mut config.slicing.sparse_infill_width),
        ] {
            ui.horizontal_wrapped(|ui| {
                ui.label(label);
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut lw.target_mm)
                            .speed(0.01)
                            .range(0.10..=2.50)
                            .prefix("T: "),
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut lw.min_mm)
                            .speed(0.01)
                            .range(0.08..=2.00)
                            .prefix("Min: "),
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut lw.max_mm)
                            .speed(0.01)
                            .range(0.12..=3.00)
                            .prefix("Max: "),
                    )
                    .changed();
            });
        }

        ui.separator();
        ui.horizontal(|ui| {
            egui::ComboBox::from_label("Solid Infill Pattern")
                .selected_text(config.slicing.solid_infill_pattern.label())
                .show_ui(ui, |ui| {
                    for pat in [
                        SolidInfillPattern::Monotonic,
                        SolidInfillPattern::Concentric,
                    ] {
                        if ui
                            .selectable_value(
                                &mut config.slicing.solid_infill_pattern,
                                pat,
                                pat.label(),
                            )
                            .changed()
                        {
                            tp_cfg_changed = true;
                        }
                    }
                });
            param_help_button(ui, "slicing.solid_infill_pattern");
        });

        ui.horizontal(|ui| {
            egui::ComboBox::from_id_source("sparse_infill_pattern_combo")
                .selected_text(format!(
                    "Sparse: {}",
                    config.slicing.sparse_infill_pattern.label()
                ))
                .show_ui(ui, |ui| {
                    for pat in SparseInfillPattern::ALL {
                        if ui
                            .selectable_value(
                                &mut config.slicing.sparse_infill_pattern,
                                pat,
                                pat.label(),
                            )
                            .changed()
                        {
                            tp_cfg_changed = true;
                        }
                    }
                });
            param_help_button(ui, "slicing.sparse_infill_pattern");
        });

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.infill_density_percent, 0.0..=100.0)
                        .suffix("%")
                        .text("Sparse Density"),
                )
                .changed();
            param_help_button(ui, "slicing.infill_density_percent");
        });

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.infill_angle_deg, -180.0..=180.0)
                        .suffix("°")
                        .text("Infill Angle"),
                )
                .changed();
            param_help_button(ui, "slicing.infill_angle_deg");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Sparse Infill Filament:");
            let cur_infill_lbl = config
                .slicing
                .infill_filament_mode
                .short_label(&config.tools, &config.filament_slots);
            let combo_w = (ui.available_width() - 32.0).clamp(105.0, 205.0);
            let mut infill_mode_changed = false;
            egui::ComboBox::from_id_source("process_sparse_infill_filament_combo")
                .selected_text(cur_infill_lbl)
                .width(combo_w)
                .show_ui(ui, |ui| {
                    infill_mode_changed |= ui
                        .selectable_value(
                            &mut config.slicing.infill_filament_mode,
                            InfillFilamentMode::FollowObject,
                            "Follow Object / Region Filament",
                        )
                        .changed();
                    infill_mode_changed |= ui
                        .selectable_value(
                            &mut config.slicing.infill_filament_mode,
                            InfillFilamentMode::AutoPurgeIntoInfill,
                            "Auto (Current Tool / Purge into Infill)",
                        )
                        .changed();
                    ui.separator();
                    for (t_idx, t) in config.tools.iter().enumerate() {
                        infill_mode_changed |= ui
                            .selectable_value(
                                &mut config.slicing.infill_filament_mode,
                                InfillFilamentMode::SingleTool(t_idx),
                                format!("Tool T{} — {} ({})", t.id, t.name, t.filament_color_hex),
                            )
                            .changed();
                    }
                    if !config.filament_slots.is_empty() {
                        ui.separator();
                        for (s_idx, _) in config.filament_slots.iter().enumerate() {
                            let s_u8 = s_idx.min(126) as u8;
                            let s_lbl = InfillFilamentMode::FilamentSlot(s_u8)
                                .label(&config.tools, &config.filament_slots);
                            infill_mode_changed |= ui
                                .selectable_value(
                                    &mut config.slicing.infill_filament_mode,
                                    InfillFilamentMode::FilamentSlot(s_u8),
                                    s_lbl,
                                )
                                .changed();
                        }
                    }
                });
            if infill_mode_changed {
                vis_state.dirty_color_mixing = true;
                vis_state.dirty_toolpaths_mesh = true;
                gc_changed = true;
            }
            param_help_button(ui, "slicing.infill_filament_mode");
        });

        if tp_cfg_changed {
            config.slicing.sanitize();
            if !is_computing {
                sync_config_to_backend(config);
            }
            vis_state.dirty_toolpaths = true;
            vis_state.dirty_surface_colors = true;
        }
        if gc_changed {
            config.gcode.sanitize();
            vis_state.dirty_planned_gcode = true;
        }
    });

    // 4. Overhangs, Bridging & Unreachable Support Generation
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Overhangs, Bridging & Support Generation").strong(),
    )
    .id_source("process_overhangs_supports_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut tp_cfg_changed = false;
        let mut support_field_cfg_changed = false;

        ui.horizontal(|ui| {
            tp_cfg_changed |= ui
                .add(
                    egui::Slider::new(
                        &mut config.slicing.overhang_angle_threshold_deg,
                        10.0..=85.0,
                    )
                    .suffix("°")
                    .text("Overhang Threshold"),
                )
                .changed();
            param_help_button(ui, "slicing.overhang_angle_threshold_deg");
        });
        ui.horizontal(|ui| {
            let bridge_changed = ui
                .add(
                    egui::Slider::new(&mut config.slicing.max_bridge_length_mm, 0.0..=200.0)
                        .suffix(" mm")
                        .text("Max Bridge Length"),
                )
                .changed();
            tp_cfg_changed |= bridge_changed;
            support_field_cfg_changed |= bridge_changed;
            param_help_button(ui, "slicing.max_bridge_length_mm");
        });
        ui.horizontal(|ui| {
            let mut overlap_pct = config.slicing.wave_overhang_overlap_fraction * 100.0;
            if ui
                .add(
                    egui::Slider::new(&mut overlap_pct, 1.0..=45.0)
                        .suffix("%")
                        .text("Wave Overlap"),
                )
                .changed()
            {
                config.slicing.wave_overhang_overlap_fraction = overlap_pct / 100.0;
                tp_cfg_changed = true;
            }
            param_help_button(ui, "slicing.wave_overhang_overlap_fraction");
        });

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            tp_cfg_changed |= ui
                .checkbox(
                    &mut config.slicing.enable_support_generation,
                    "Enable Support Generation",
                )
                .changed();
            param_help_button(ui, "slicing.enable_support_generation");
            let seed_changed = ui
                .checkbox(
                    &mut config.slicing.seed_order_field_from_supports,
                    "Seed Order Field (τ)",
                )
                .changed();
            tp_cfg_changed |= seed_changed;
            support_field_cfg_changed |= seed_changed;
            param_help_button(ui, "slicing.seed_order_field_from_supports");
        });
        ui.horizontal_wrapped(|ui| {
            let only_unreach_changed = ui
                .checkbox(
                    &mut config.slicing.support_only_unreachable_and_bridges,
                    "Only Unreachable & Bridges >",
                )
                .on_hover_text(
                    "When enabled, only generate support for causally unreachable regions (floating islands / downward local minima) and two-pier bridges exceeding Max Bridge Length.",
                )
                .changed();
            tp_cfg_changed |= only_unreach_changed;
            support_field_cfg_changed |= only_unreach_changed;
            let bridge_len_changed = ui
                .add(
                    egui::DragValue::new(&mut config.slicing.max_bridge_length_mm)
                        .speed(0.5)
                        .range(0.0..=200.0)
                        .suffix(" mm"),
                )
                .changed();
            tp_cfg_changed |= bridge_len_changed;
            support_field_cfg_changed |= bridge_len_changed;
            param_help_button(ui, "slicing.support_only_unreachable_and_bridges");
        });

        if config.slicing.enable_support_generation {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Support Tool:").strong());
                let selected_support_label = match config.slicing.support_tool_index {
                    None => {
                        let obj_tool_lbl = config
                            .active_tool()
                            .map(|t| format!("Active Tool ({})", t.name))
                            .unwrap_or_else(|| "Active Object Tool".to_string());
                        obj_tool_lbl
                    }
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| format!("{} ({:.2}mm, {:.0}°C)", t.name, t.nozzle_diameter_mm, t.temperature_celsius))
                        .unwrap_or_else(|| format!("Tool #{idx}")),
                };
                let combo_w = (ui.available_width() - 55.0).clamp(105.0, 210.0);
                egui::ComboBox::from_id_source("support_tool_selector_combo")
                    .selected_text(selected_support_label)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut config.slicing.support_tool_index,
                                None,
                                "Active Object Tool (Same Material)",
                            )
                            .changed()
                        {
                            tp_cfg_changed = true;
                        }
                        for (idx, tool) in config.tools.iter().enumerate() {
                            let lbl = format!(
                                "{} ({:.2}mm, {:.0}°C)",
                                tool.name, tool.nozzle_diameter_mm, tool.temperature_celsius
                            );
                            if ui
                                .selectable_value(
                                    &mut config.slicing.support_tool_index,
                                    Some(idx),
                                    lbl,
                                )
                                .changed()
                            {
                                config.slicing.support_line_width_mm = tool.nozzle_diameter_mm;
                                tp_cfg_changed = true;
                            }
                        }
                    });
                if ui
                    .small_button("+ Tool")
                    .on_hover_text("Add a dedicated support tool/extruder to the Multi-Tool Registry")
                    .clicked()
                {
                    let next_id = config.tools.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                    let new_name = format!("T{next_id} - Support Material");
                    config.tools.push(ToolConfig::standard_tool(
                        next_id,
                        new_name,
                        active_nozzle_mm,
                        215.0,
                    ));
                    config.slicing.support_tool_index = Some(config.tools.len() - 1);
                    tp_cfg_changed = true;
                }
                param_help_button(ui, "slicing.support_tool_index");
            });

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Support Interface Tool:").strong());
                let selected_intf_label = match config.slicing.support_interface_tool_index {
                    None => "Follow Support Base Tool".to_string(),
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| {
                            format!(
                                "{} ({:.2}mm, {:.0}°C)",
                                t.name, t.nozzle_diameter_mm, t.temperature_celsius
                            )
                        })
                        .unwrap_or_else(|| format!("Tool #{idx}")),
                };
                let combo_w = (ui.available_width() - 32.0).clamp(105.0, 210.0);
                egui::ComboBox::from_id_source("support_interface_tool_selector_combo")
                    .selected_text(selected_intf_label)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_value(
                                &mut config.slicing.support_interface_tool_index,
                                None,
                                "Follow Support Base Tool",
                            )
                            .changed()
                        {
                            vis_state.dirty_planned_gcode = true;
                            vis_state.dirty_toolpaths_mesh = true;
                        }
                        for (idx, tool) in config.tools.iter().enumerate() {
                            let lbl = format!(
                                "{} ({:.2}mm, {:.0}°C)",
                                tool.name, tool.nozzle_diameter_mm, tool.temperature_celsius
                            );
                            if ui
                                .selectable_value(
                                    &mut config.slicing.support_interface_tool_index,
                                    Some(idx),
                                    lbl,
                                )
                                .changed()
                            {
                                vis_state.dirty_planned_gcode = true;
                                vis_state.dirty_toolpaths_mesh = true;
                            }
                        }
                    });
                param_help_button(ui, "slicing.support_interface_tool_index");
            });

            if let Some(sup_idx) = config.slicing.support_tool_index {
                if let Some(sup_tool) = config.tools.get_mut(sup_idx) {
                    ui.horizontal_wrapped(|ui| {
                        ui.small(egui::RichText::new(format!("T{} Support Material:", sup_tool.id)).strong());
                        if ui
                            .add(
                                egui::DragValue::new(&mut sup_tool.nozzle_diameter_mm)
                                    .speed(0.02)
                                    .range(0.15..=2.0)
                                    .prefix("Nozzle: ")
                                    .suffix(" mm"),
                            )
                            .changed()
                        {
                            tp_cfg_changed = true;
                        }
                        if ui
                            .add(
                                egui::DragValue::new(&mut sup_tool.temperature_celsius)
                                    .speed(1.0)
                                    .range(150.0..=350.0)
                                    .prefix("Temp: ")
                                    .suffix(" °C"),
                            )
                            .changed()
                        {
                            vis_state.dirty_planned_gcode = true;
                        }
                    });
                }
            }

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Support Style:").strong());
                for style in SupportStyle::ALL {
                    let selected = config.slicing.support_style == style;
                    if render_delineated_pill_tab(ui, selected, style.label())
                        .on_hover_text(match style {
                            SupportStyle::ColumnarGrid => {
                                "Classic gravity-aligned rectilinear scanline columns."
                            }
                            SupportStyle::NonPlanarTree => {
                                "Organic coalescing tree branches that steer around the part's 3D SDF φ_sdf(x) along gravity, merge into hollow/ribbed trunks, and orient nozzle beads along the 3D branch tangent."
                            }
                        })
                        .clicked()
                        && !selected
                    {
                        config.slicing.support_style = style;
                        tp_cfg_changed = true;
                    }
                }
                param_help_button(ui, "slicing.support_style");
            });

            ui.horizontal_wrapped(|ui| {
                tp_cfg_changed |= ui
                    .checkbox(
                        &mut config.slicing.support_on_bed_only,
                        "On Build Plate Only (Avoid Upper Part Surfaces)",
                    )
                    .on_hover_text(
                        "Prevents supports from resting on top of upper model surfaces. In Non-Planar Tree mode, branches above the model steer laterally around the part's 3D SDF φ_sdf(x) to anchor exclusively on the build plate.",
                    )
                    .changed();
                param_help_button(ui, "slicing.support_on_bed_only");
            });

            if config.slicing.support_style == SupportStyle::NonPlanarTree {
                ui.horizontal_wrapped(|ui| {
                    tp_cfg_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.slicing.tree_branch_angle_deg)
                                .speed(0.5)
                                .range(15.0..=65.0)
                                .prefix("Branch Angle: ")
                                .suffix("°"),
                        )
                        .on_hover_text(
                            "Maximum lateral lean angle of tree branches relative to anti-gravity (-g_bed) when routing around the model and coalescing into trunks.",
                        )
                        .changed();
                    tp_cfg_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.slicing.tree_branch_wall_count)
                                .speed(0.1)
                                .range(1..=4)
                                .prefix("Walls: "),
                        )
                        .on_hover_text(
                            "Number of concentric perimeter walls per hollow tree branch/trunk.",
                        )
                        .changed();
                    param_help_button(ui, "slicing.tree_support_params");
                });
                ui.horizontal_wrapped(|ui| {
                    tp_cfg_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.slicing.tree_tip_diameter_mm)
                                .speed(0.05)
                                .range(0.8..=8.0)
                                .prefix("Tip Ø: ")
                                .suffix(" mm"),
                        )
                        .changed();
                    tp_cfg_changed |= ui
                        .add(
                            egui::DragValue::new(&mut config.slicing.tree_trunk_max_diameter_mm)
                                .speed(0.1)
                                .range(2.0..=30.0)
                                .prefix("Max Trunk Ø: ")
                                .suffix(" mm"),
                        )
                        .changed();
                });
            }

            ui.horizontal_wrapped(|ui| {
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_z_gap_mm)
                            .speed(0.01)
                            .range(0.0..=2.0)
                            .prefix("Z Gap: ")
                            .suffix(" mm"),
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_xy_gap_mm)
                            .speed(0.02)
                            .range(0.05..=5.0)
                            .prefix("XY Gap: ")
                            .suffix(" mm"),
                    )
                    .changed();
                param_help_button(ui, "slicing.support_gaps_mm");
            });
            ui.horizontal_wrapped(|ui| {
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_interface_layers)
                            .speed(0.1)
                            .range(0..=10)
                            .prefix("Roof Intf Layers: "),
                    )
                    .on_hover_text(
                        "Number of dense support interface layers printed immediately beneath overhanging part roofs.",
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_bed_interface_layers)
                            .speed(0.1)
                            .range(0..=8)
                            .prefix("Bed Floor Layers: "),
                    )
                    .on_hover_text(
                        "Number of dense bottom support floor layers printed directly on the build plate for bed adhesion (0 = disabled).",
                    )
                    .changed();
                param_help_button(ui, "slicing.support_interface_layers");
            });
            ui.horizontal_wrapped(|ui| {
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_base_spacing_mm)
                            .speed(0.05)
                            .range(0.6..=10.0)
                            .prefix("Base Spacing: ")
                            .suffix(" mm"),
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_interface_spacing_mm)
                            .speed(0.01)
                            .range(0.0..=4.0)
                            .prefix("Intf Gap: ")
                            .suffix(" mm"),
                    )
                    .on_hover_text(
                        "Edge-to-edge gap between adjacent support interface extrusions (0.0 mm = solid 100% touching roof/floor interface).",
                    )
                    .changed();
                tp_cfg_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.slicing.support_line_width_mm)
                            .speed(0.01)
                            .range(0.15..=2.0)
                            .prefix("Width: ")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Extrusion line width for support structures.")
                    .changed();
                param_help_button(ui, "slicing.support_interface_spacing_mm");
            });
        }

        if support_field_cfg_changed && has_eikonal && !is_computing {
            config.slicing.sanitize();
            if let Some(&(sel_id, _, ref obj_name, obj_rotation)) = selected_obj_info {
                spawn_field_compute_task(
                    vis_state,
                    plate,
                    config,
                    sel_id,
                    obj_name,
                    obj_rotation,
                );
            }
        } else if tp_cfg_changed {
            config.slicing.sanitize();
            if !is_computing {
                sync_config_to_backend(config);
                if let Some(&(_, _, _, obj_rotation)) = selected_obj_info {
                    let inv_rot = obj_rotation.inverse();
                    let local_nozzle = (inv_rot * Vec3::from_array(config.machine.nozzle_axis_bed))
                        .normalize_or_zero();
                    if local_nozzle.length_squared() > 1.0e-6 {
                        ffi::set_nozzle_axis_bed(local_nozzle.to_array());
                    }
                    let local_gravity =
                        (inv_rot * Vec3::from_array(config.machine.gravity_direction_bed))
                            .normalize_or_zero();
                    if local_gravity.length_squared() > 1.0e-6 {
                        ffi::set_gravity_direction_bed(local_gravity.to_array());
                    }
                }
            }
            vis_state.dirty_toolpaths = true;
        }
    });

    // 4b. Optical Colour Approximation (CMYKW & Arbitrary Filaments / Lithophane Optics)
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Optical Colour Approximation (CMYKW & Arbitrary Filaments)").strong(),
    )
    .id_source("process_optical_color_mixing_section")
    .default_open(config.slicing.color_mixing.enabled || config.tools.len() >= 2)
    .show(ui, |ui| {
        let mut mix_changed = false;
        let mut need_backend_sync = false;

        ui.horizontal_wrapped(|ui| {
            if ui
                .checkbox(
                    &mut config.slicing.color_mixing.enabled,
                    "Enable Kubelka-Munk Optical Colour Mixing",
                )
                .on_hover_text(
                    "Approximate 3MF surface colours & PNG textures by stacking translucent filaments across outer shell layers/walls and subdividing perimeters into short extrusions.",
                )
                .changed()
            {
                if config.slicing.color_mixing.enabled {
                    vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                }
                mix_changed = true;
            }
            param_help_button(ui, "slicing.color_mixing.enabled");

            if ui
                .small_button("Setup PLA CMYKW (5 Tools)")
                .on_hover_text(
                    "Configure T0..T4 as Cyan, Magenta, Yellow, Black & White PLA with calibrated Transmission Distances",
                )
                .clicked()
            {
                apply_quick_cmykw_palette_setup(
                    config,
                    vis_state,
                    &crate::color_optics::bambu_pla_cmykw_palette(),
                );
                need_backend_sync = true;
                mix_changed = true;
            }
            if ui
                .small_button("Setup Translucent CMYKW")
                .on_hover_text(
                    "Configure T0..T4 as high-TD translucent Cyan, Magenta, Yellow, Black & opaque White backing",
                )
                .clicked()
            {
                apply_quick_cmykw_palette_setup(
                    config,
                    vis_state,
                    &crate::color_optics::translucent_cmykw_palette(),
                );
                need_backend_sync = true;
                mix_changed = true;
            }
        });

        if config.slicing.color_mixing.enabled && config.tools.len() < 2 {
            ui.small(
                egui::RichText::new(
                    "Add at least 2 extrusion tools in the Tool tab (or click 'Setup PLA CMYKW (5 Tools)') to mix colours across multiple filaments.",
                )
                .color(egui::Color32::from_rgb(255, 195, 90)),
            );
        }

        ui.horizontal_wrapped(|ui| {
            ui.label("Illumination:");
            let combo_w = (ui.available_width() - 28.0).clamp(105.0, 215.0);
            egui::ComboBox::from_id_source("color_mixing_lighting_mode_combo")
                .selected_text(config.slicing.color_mixing.lighting_mode.label())
                .width(combo_w)
                .show_ui(ui, |ui| {
                    for mode in [
                        crate::color_optics::OpticalLightingMode::FrontLitReflectance,
                        crate::color_optics::OpticalLightingMode::BackLitTransmittance,
                        crate::color_optics::OpticalLightingMode::MixedFrontAndBack,
                    ] {
                        mix_changed |= ui
                            .selectable_value(
                                &mut config.slicing.color_mixing.lighting_mode,
                                mode,
                                mode.label(),
                            )
                            .changed();
                    }
                });
            param_help_button(ui, "slicing.color_mixing.lighting_mode");
        });

        ui.horizontal_wrapped(|ui| {
            mix_changed |= ui
                .checkbox(
                    &mut config.slicing.color_mixing.use_adjacent_layers,
                    "Adjacent-Layer Blending",
                )
                .on_hover_text(
                    "Blend out-of-gamut colours across consecutive layers in addition to through-depth Kubelka-Munk stacking.",
                )
                .changed();
            mix_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.color_mixing.optical_shell_layers)
                        .speed(0.1)
                        .range(1..=6)
                        .prefix("Shell Slabs: "),
                )
                .on_hover_text(
                    "Number of outer shell slabs (walls / top-bottom skin layers) stacked optically over the internal backing core.",
                )
                .changed();
            let total_walls = config.slicing.inner_wall_count + 1;
            if config.slicing.color_mixing.enabled
                && total_walls < config.slicing.color_mixing.optical_shell_layers
                && ui
                    .small_button(format!(
                        "Set {} Walls",
                        config.slicing.color_mixing.optical_shell_layers
                    ))
                    .on_hover_text(
                        "Increase perimeter wall count so side walls have enough concentric shells for full through-depth optical stacking.",
                    )
                    .clicked()
            {
                config.slicing.inner_wall_count =
                    config.slicing.color_mixing.optical_shell_layers.saturating_sub(1);
                need_backend_sync = true;
                vis_state.dirty_toolpaths = true;
            }
            param_help_button(ui, "slicing.color_mixing.optical_shell_layers");
        });

        ui.horizontal_wrapped(|ui| {
            mix_changed |= ui
                .checkbox(
                    &mut config.slicing.color_mixing.use_short_extrusions,
                    "Short Extrusions Along Path",
                )
                .on_hover_text(
                    "Subdivide perimeter & skin extrusions into short segments with Oklab error diffusion along the path.",
                )
                .changed();
            if config.slicing.color_mixing.use_short_extrusions {
                mix_changed |= ui
                    .add(
                        egui::DragValue::new(
                            &mut config.slicing.color_mixing.min_short_extrusion_mm,
                        )
                        .speed(0.1)
                        .range(0.8..=30.0)
                        .prefix("Min Len: ")
                        .suffix(" mm"),
                    )
                    .on_hover_text(
                        "Minimum physical extrusion segment length (mm) between colour swaps along a perimeter/skin path.",
                    )
                    .changed();
            }
            param_help_button(ui, "slicing.color_mixing.min_short_extrusion_mm");
        });

        if config.slicing.color_mixing.lighting_mode.uses_opaque_core_backing() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Core Backing:");
                let backing_lbl = match config.slicing.color_mixing.backing_tool_index {
                    None => "Auto (Brightest Neutral / White)".to_string(),
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| format!("T{} — {} ({})", t.id, t.name, t.filament_color_hex))
                        .unwrap_or_else(|| format!("Tool #{idx}")),
                };
                let combo_w = (ui.available_width() - 8.0).clamp(105.0, 215.0);
                egui::ComboBox::from_id_source("color_mixing_backing_tool_combo")
                    .selected_text(backing_lbl)
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        mix_changed |= ui
                            .selectable_value(
                                &mut config.slicing.color_mixing.backing_tool_index,
                                None,
                                "Auto (Brightest Neutral / White)",
                            )
                            .changed();
                        for (idx, t) in config.tools.iter().enumerate() {
                            let lbl =
                                format!("T{} — {} ({})", t.id, t.name, t.filament_color_hex);
                            mix_changed |= ui
                                .selectable_value(
                                    &mut config.slicing.color_mixing.backing_tool_index,
                                    Some(idx),
                                    lbl,
                                )
                                .changed();
                        }
                    });
            });
        }

        ui.horizontal_wrapped(|ui| {
            mix_changed |= ui
                .add(
                    egui::Slider::new(&mut config.slicing.color_mixing.dither_strength, 0.0..=1.0)
                        .text("Oklab Dither"),
                )
                .on_hover_text(
                    "Strength of Oklab perceptual error diffusion & adjacent-layer spatial dithering (0 = nearest optical stack, 1 = full dithering).",
                )
                .changed();
            mix_changed |= ui
                .add(
                    egui::DragValue::new(&mut config.slicing.color_mixing.tool_switch_penalty)
                        .speed(0.002)
                        .range(0.0..=0.10)
                        .prefix("Hysteresis: "),
                )
                .on_hover_text(
                    "Oklab ΔE hysteresis penalty discouraging marginal filament switches along continuous paths.",
                )
                .changed();
        });

        if !config.tools.is_empty() {
            ui.separator();
            ui.small(egui::RichText::new("Loaded Tool Optical Palette (Colour R_∞ & TD):").strong());
            for tool in &mut config.tools {
                if render_tool_optical_properties_controls(ui, tool) {
                    mix_changed = true;
                    vis_state.dirty_surface_colors = true;
                }
            }
        }

        if need_backend_sync && !is_computing {
            sync_config_to_backend(config);
        }
        if mix_changed {
            config.slicing.color_mixing.sanitize();
            vis_state.dirty_color_mixing = true;
            vis_state.dirty_planned_gcode = true;
            vis_state.dirty_toolpaths_mesh = true;
        }
    });

    // 5. 3D Conformal Scarf Joint & Feature Print Speeds
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("3D Scarf Joint & Feature Print Speeds").strong(),
    )
    .id_source("process_scarf_and_speeds_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut gc_changed = false;
        let gc = &mut config.gcode;

        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_scarf_joint,
                    "3D Conformal Scarf Joint (Closed Loops)",
                )
                .changed();
            param_help_button(ui, "gcode.enable_scarf_joint");
        });
        if gc.enable_scarf_joint {
            ui.horizontal(|ui| {
                gc_changed |= ui
                    .add(
                        egui::Slider::new(&mut gc.scarf_length_mm, 1.0..=25.0)
                            .text("Scarf Length (mm)"),
                    )
                    .changed();
                param_help_button(ui, "gcode.scarf_length_mm");
            });
            ui.horizontal(|ui| {
                gc_changed |= ui
                    .add(
                        egui::Slider::new(&mut gc.scarf_start_height_ratio, 0.02..=0.50)
                            .text("Scarf Start Height Ratio"),
                    )
                    .changed();
                param_help_button(ui, "gcode.scarf_start_height_ratio");
            });
        }
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.seam_gap_mm)
                        .speed(0.01)
                        .range(0.0..=1.0)
                        .prefix("Seam Gap: ")
                        .suffix(" mm"),
                )
                .changed();
            param_help_button(ui, "gcode.seam_gap_mm");
        });

        ui.separator();
        ui.small(egui::RichText::new("First-Layer Brim & Lever-Arm Governor:").strong());
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_auto_brim,
                    "Auto Brim for Tall / Narrow Parts",
                )
                .changed();
            param_help_button(ui, "gcode.brim_width_mm");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.brim_width_mm)
                        .speed(0.2)
                        .range(0.0..=30.0)
                        .prefix("Brim Width: ")
                        .suffix(" mm (0 = auto)"),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.brim_gap_mm)
                        .speed(0.01)
                        .range(0.0..=1.0)
                        .prefix("Gap: ")
                        .suffix(" mm"),
                )
                .changed();
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_lever_arm_slowdown,
                    "Lever-Arm & Curled-Corner Slowdown",
                )
                .changed();
            param_help_button(ui, "gcode.enable_lever_arm_slowdown");
        });
        if gc.enable_lever_arm_slowdown {
            ui.horizontal(|ui| {
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.lever_arm_reference_ratio)
                            .speed(0.1)
                            .range(0.5..=15.0)
                            .prefix("Ref Ratio H/W: "),
                    )
                    .changed();
                gc_changed |= ui
                    .add(
                        egui::DragValue::new(&mut gc.lever_arm_min_speed_fraction)
                            .speed(0.02)
                            .range(0.10..=1.0)
                            .prefix("Min Speed Frac: "),
                    )
                    .changed();
            });
        }

        ui.separator();
        ui.small(egui::RichText::new("Feature Print Speeds (mm/s):").strong());
        ui.horizontal_wrapped(|ui| {
            ui.label("Outer / Inner Wall:");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.outer_wall_speed_mms)
                        .speed(1.0)
                        .range(5.0..=500.0)
                        .prefix("Out: "),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.inner_wall_speed_mms)
                        .speed(1.0)
                        .range(5.0..=500.0)
                        .prefix("In: "),
                )
                .changed();
            param_help_button(ui, "gcode.wall_speeds_mms");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Solid / Sparse Infill:");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.solid_infill_speed_mms)
                        .speed(1.0)
                        .range(5.0..=600.0)
                        .prefix("Solid: "),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.sparse_infill_speed_mms)
                        .speed(1.0)
                        .range(5.0..=600.0)
                        .prefix("Sparse: "),
                )
                .changed();
            param_help_button(ui, "gcode.infill_speeds_mms");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Bridge / Wave:");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.bridge_speed_mms)
                        .speed(1.0)
                        .range(5.0..=400.0)
                        .prefix("Brg: "),
                )
                .changed();
            param_help_button(ui, "gcode.bridge_speed_mms");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.wave_overhang_speed_mms)
                        .speed(1.0)
                        .range(5.0..=400.0)
                        .prefix("Wave: "),
                )
                .changed();
            param_help_button(ui, "gcode.wave_overhang_speed_mms");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("1st Layer / Gap Fill:");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.first_layer_speed_mms)
                        .speed(1.0)
                        .range(5.0..=200.0)
                        .prefix("1st: "),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.gap_fill_speed_mms)
                        .speed(1.0)
                        .range(5.0..=400.0)
                        .prefix("Gap: "),
                )
                .changed();
            param_help_button(ui, "gcode.gap_fill_speed_mms");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Support Base / Intf:");
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.support_speed_mms)
                        .speed(1.0)
                        .range(5.0..=400.0)
                        .prefix("Base: "),
                )
                .changed();
            gc_changed |= ui
                .add(
                    egui::DragValue::new(&mut gc.support_interface_speed_mms)
                        .speed(1.0)
                        .range(5.0..=300.0)
                        .prefix("Intf: "),
                )
                .changed();
        });

        if gc_changed {
            config.gcode.sanitize();
            vis_state.dirty_planned_gcode = true;
        }
    });
}

/// Renders the dedicated Right-Hand `View & Inspection` Panel:
/// 1. Scene & CAD Display (`CAD Mesh Opacity`, `Object Surface Color`, `Build Volume Box`, `Nozzle & Gravity Gizmo`)
/// 2. Volumetric Field Visualization (`sdf`, `eikonal`, `overhang`, `tool_tilt_deg`, `support_needed`, 3D Isosurface, 2D Slice Plane)
/// 3. Selected Slicing Isosurfaces (`τ_k` layers, branch/thickness colors, layer range & interactive list)
/// 4. 3D Toolpaths & G-code Preview (Feature visibility toggles, `ToolpathColorMode`, legends & telemetry)
fn render_view_and_inspection_panel(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    plate: &mut BuildPlateState,
    selected_obj_info: Option<&(u64, usize, String, Quat)>,
    is_computing: bool,
    has_eikonal: bool,
) {
    // 1. CAD & Scene Display
    egui::CollapsingHeader::new(egui::RichText::new("CAD & Scene Display").strong())
        .id_source("view_cad_scene_display_section")
        .default_open(true)
        .show(ui, |ui| {
            ui.add(
                egui::Slider::new(&mut vis_state.solid_opacity, 0.0..=1.0)
                    .text("CAD Mesh Opacity"),
            );

            ui.horizontal(|ui| {
                ui.label("Surface Color:");
                let combo_w = (ui.available_width() - 8.0).clamp(130.0, 220.0);
                egui::ComboBox::from_id_source("object_surface_color_mode_combo")
                    .selected_text(vis_state.surface_color_mode.short_label())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for mode in ObjectSurfaceColorMode::ALL {
                            if ui
                                .selectable_value(
                                    &mut vis_state.surface_color_mode,
                                    mode,
                                    mode.label(),
                                )
                                .changed()
                            {
                                vis_state.dirty_surface_colors = true;
                            }
                        }
                    });
            });

            match vis_state.surface_color_mode {
                ObjectSurfaceColorMode::DefaultCad => {}
                ObjectSurfaceColorMode::Overhang => {
                    draw_continuous_gradient_legend(
                        ui,
                        "Surface Overhang vs. Gravity (0°=Up / 90°=Vert / 135°=45° Overhang / 180°=Down)",
                        0.0,
                        180.0,
                        "°",
                        0,
                        high_contrast_parameter_ramp,
                    );
                    ui.small(format!(
                        "Overhang Range: [{:.1}°, {:.1}°] (Mean {:.1}°) | Gravity Bed = ({:.2}, {:.2}, {:.2})",
                        vis_state.surface_color_min_deg,
                        vis_state.surface_color_max_deg,
                        vis_state.surface_color_mean_deg,
                        config.machine.gravity_direction_bed[0],
                        config.machine.gravity_direction_bed[1],
                        config.machine.gravity_direction_bed[2],
                    ));
                }
                ObjectSurfaceColorMode::ToolAngleFromNormal => {
                    draw_continuous_gradient_legend(
                        ui,
                        "Tool Angle from Surface Normal (0°=⊥ Conformal / 90°=∥ Tangent / 180°=Opposite)",
                        0.0,
                        180.0,
                        "°",
                        0,
                        high_contrast_parameter_ramp,
                    );
                    let source_label = if vis_state.surface_color_uses_solved_tool {
                        "Solved 3D Tool Axis Field"
                    } else {
                        "Nominal Machine Nozzle Frame"
                    };
                    ui.small(format!(
                        "Tool-Normal Angle: [{:.1}°, {:.1}°] (Mean {:.1}°) | {source_label}",
                        vis_state.surface_color_min_deg,
                        vis_state.surface_color_max_deg,
                        vis_state.surface_color_mean_deg,
                    ));
                }
                ObjectSurfaceColorMode::SupportNeeded => {
                    draw_continuous_gradient_legend(
                        ui,
                        "Unreachable Order Field / Support Needed (0%=Supported / 100%=Needs Support)",
                        0.0,
                        100.0,
                        "%",
                        0,
                        high_contrast_parameter_ramp,
                    );
                    let source_label = if vis_state.surface_color_uses_solved_tool {
                        "Solved Constrained Order Field ('support_needed')"
                    } else {
                        "Not Computed Yet (Click '1. Compute Fields')"
                    };
                    ui.small(format!(
                        "Support Needed: [{:.0}%, {:.0}%] (Mean {:.1}%) | {source_label}",
                        vis_state.surface_color_min_deg,
                        vis_state.surface_color_max_deg,
                        vis_state.surface_color_mean_deg,
                    ));
                }
            }

            let mut hq_pbr_scene =
                vis_state.toolpath_quality_mode == ToolpathQualityMode::HighQualityPbr;
            if ui
                .checkbox(
                    &mut hq_pbr_scene,
                    "High-Quality PBR Shading, Clearcoat & Soft Shadows",
                )
                .on_hover_text(
                    "Enable Bevy PBR clearcoat polymer sheen, cascaded soft directional shadows, and 12-vertex pressed-capsule toolpath rendering.",
                )
                .changed()
            {
                vis_state.toolpath_quality_mode = if hq_pbr_scene {
                    ToolpathQualityMode::HighQualityPbr
                } else {
                    ToolpathQualityMode::Fast
                };
                vis_state.dirty_toolpaths_mesh = true;
            }
            ui.checkbox(
                &mut plate.show_object_labels,
                "Show Build-Plate Object Labels (EXCLUDE_OBJECT)",
            );
            ui.checkbox(&mut config.bed.show_volume_box, "Show Build Volume Box");
            ui.checkbox(
                &mut config.machine.show_nozzle_clearance_gizmo,
                "Show 3D Nozzle Frame, Clearance & Gravity Gizmo",
            );
        });

    let Some(&(sel_id, _, ref obj_name, obj_rotation)) = selected_obj_info else {
        ui.add_space(8.0);
        if vis_state.cached_planned_gcode.is_some() {
            ui.separator();
            egui::CollapsingHeader::new(
                egui::RichText::new("3D Toolpaths & Calibration Preview").strong(),
            )
            .id_source("view_standalone_calibration_toolpaths_section")
            .default_open(true)
            .show(ui, |ui| {
                if ui
                    .checkbox(&mut vis_state.show_toolpaths, "Show 3D Toolpaths")
                    .changed()
                {
                    vis_state.dirty_toolpaths_mesh = true;
                }
                render_layer_and_move_scrub_controls(
                    ui,
                    vis_state,
                    config,
                    Transform::IDENTITY,
                    true,
                    "standalone_calibration_scrub",
                );
                if vis_state.show_toolpaths {
                    render_toolpath_quality_and_transparency_controls(ui, vis_state);
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        ui.label("Color By:");
                        let combo_w = (ui.available_width() - 8.0).clamp(130.0, 220.0);
                        egui::ComboBox::from_id_source("standalone_toolpath_color_mode_combo")
                            .selected_text(vis_state.toolpath_color_mode.short_label())
                            .width(combo_w)
                            .show_ui(ui, |ui| {
                                for mode in ToolpathColorMode::ALL {
                                    if ui
                                        .selectable_value(
                                            &mut vis_state.toolpath_color_mode,
                                            mode,
                                            mode.label(),
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                }
                            });
                    });
                    if vis_state.toolpath_color_mode != ToolpathColorMode::ByFeatureType {
                        render_toolpath_color_legend(ui, vis_state, config);
                        ui.add_space(3.0);
                    }
                    render_interactive_toolpath_line_types_legend(ui, vis_state);
                }
            });
        } else {
            ui.small("Select an object on the build plate (or generate a Calibration pattern) to inspect 3D toolpaths.");
        }
        return;
    };

    // 2. Volumetric Field Visualization (OpenVDB)
    ui.add_space(6.0);
    ui.separator();
    egui::CollapsingHeader::new(
        egui::RichText::new("Volumetric Field Inspector (OpenVDB)").strong(),
    )
    .id_source("view_openvdb_field_inspector_section")
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label("Scalar Field:");
            let avail = vis_state.available_fields.clone();
            if avail.is_empty() {
                ui.small("(not computed yet)");
            } else {
                for fname in &avail {
                    let selected = &vis_state.active_field_name == fname;
                    if render_delineated_pill_tab(ui, selected, fname.as_str()).clicked()
                        && !selected
                    {
                        vis_state.active_field_name = fname.clone();
                        vis_state.mark_all_dirty();
                        if !is_computing {
                            vis_state.refresh_stats_for(sel_id);
                        }
                    }
                }
            }
        });

        let (min_val, max_val) = if let Some(ref stats) = vis_state.cached_stats {
            ui.small(format!(
                "Active Voxels: {} | Voxel: {:.2} mm | Range: [{:.2}, {:.2}]",
                stats.active_voxel_count, stats.voxel_size, stats.min_value, stats.max_value
            ));
            (stats.min_value, stats.max_value)
        } else {
            (-10.0, 10.0)
        };

        // 3D Isosurface controls (volumeToMesh)
        if ui
            .checkbox(
                &mut vis_state.show_isosurface,
                "Show 3D Isosurface (volumeToMesh)",
            )
            .changed()
        {
            vis_state.dirty_isosurface = true;
            if vis_state.show_isosurface && vis_state.solid_opacity > 0.5 {
                vis_state.solid_opacity = 0.25;
            }
            if vis_state.show_isosurface
                && vis_state.available_fields.is_empty()
                && vis_state.pending_field_task.is_none()
            {
                spawn_field_compute_task(
                    vis_state,
                    plate,
                    config,
                    sel_id,
                    obj_name,
                    obj_rotation,
                );
            }
        }

        if vis_state.show_isosurface {
            let lo = min_val.min(max_val - 0.01);
            let hi = max_val.max(lo + 0.01);
            if ui
                .add(egui::Slider::new(&mut vis_state.isovalue, lo..=hi).text("Isovalue"))
                .changed()
            {
                vis_state.dirty_isosurface = true;
                if vis_state.active_field_name != "sdf" {
                    vis_state.dirty_slice = true;
                }
            }
            if ui
                .add(egui::Slider::new(&mut vis_state.adaptivity, 0.0..=0.5).text("Adaptivity"))
                .changed()
            {
                vis_state.dirty_isosurface = true;
            }

            if vis_state.active_field_name != "sdf" {
                egui::ComboBox::from_label("Object Intersection")
                    .selected_text(vis_state.intersection_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            IsoIntersectionMode::ClipToObjectInterior,
                            IsoIntersectionMode::HighlightInsideDimOutside,
                            IsoIntersectionMode::FullUnclipped,
                        ] {
                            if ui
                                .selectable_value(
                                    &mut vis_state.intersection_mode,
                                    mode,
                                    mode.label(),
                                )
                                .changed()
                            {
                                vis_state.dirty_isosurface = true;
                            }
                        }
                    });

                ui.checkbox(
                    &mut vis_state.show_intersection_contour,
                    "Show 3D Object Intersection Contour (φ_sdf = 0)",
                );

                ui.small(format!(
                    "Inside (φ <= 0): {} tris | Exterior (φ > 0): {} tris | Contour: {} segs",
                    vis_state.interior_triangles,
                    vis_state.exterior_triangles,
                    vis_state.contour_segments_local.len()
                ));
            } else {
                ui.small(format!(
                    "Isosurface Mesh: {} triangles",
                    vis_state.isosurface_triangles
                ));
            }
        }

        ui.separator();

        // 2D Cross-Section Slice Plane controls (GridSampler)
        if ui
            .checkbox(
                &mut vis_state.show_slice_plane,
                "Show 2D Cross-Section Slice Plane",
            )
            .changed()
        {
            vis_state.dirty_slice = true;
            if vis_state.show_slice_plane && vis_state.solid_opacity > 0.5 {
                vis_state.solid_opacity = 0.25;
            }
            if vis_state.show_slice_plane
                && vis_state.available_fields.is_empty()
                && vis_state.pending_field_task.is_none()
            {
                spawn_field_compute_task(
                    vis_state,
                    plate,
                    config,
                    sel_id,
                    obj_name,
                    obj_rotation,
                );
            }
        }

        if vis_state.show_slice_plane {
            ui.horizontal_wrapped(|ui| {
                ui.label("Plane:");
                for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
                    if ui
                        .selectable_label(vis_state.slice_axis == axis, axis.label())
                        .clicked()
                    {
                        vis_state.slice_axis = axis;
                        vis_state.dirty_slice = true;
                    }
                }
            });

            let plane_coord = vis_state.slice_plane_coord;
            if ui
                .add(
                    egui::Slider::new(&mut vis_state.slice_normalized_pos, 0.0..=1.0)
                        .text(format!("Slice Offset ({plane_coord:.1} mm)")),
                )
                .changed()
            {
                vis_state.dirty_slice = true;
            }

            egui::ComboBox::from_label("Colormap")
                .selected_text(vis_state.colormap.label())
                .show_ui(ui, |ui| {
                    for cmap in [
                        FieldColormap::CoolWarmSigned,
                        FieldColormap::ViridisSequential,
                        FieldColormap::ZebraIsoBands,
                    ] {
                        if ui
                            .selectable_label(vis_state.colormap == cmap, cmap.label())
                            .clicked()
                        {
                            vis_state.colormap = cmap;
                            vis_state.dirty_slice = true;
                        }
                    }
                });

            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut vis_state.show_iso_contours, "Iso-Contour Lines")
                    .changed()
                {
                    vis_state.dirty_slice = true;
                }
                if ui
                    .checkbox(&mut vis_state.fade_exterior_background, "Fade Background")
                    .changed()
                {
                    vis_state.dirty_slice = true;
                }
            });

            let cmap = vis_state.colormap;
            let field_unit = match vis_state.active_field_name.as_str() {
                "sdf" | "eikonal" => "mm",
                "overhang" => "cos θ",
                _ => "",
            };
            draw_continuous_gradient_legend(
                ui,
                &format!(
                    "Slice Colormap ({}) — {}",
                    cmap.label(),
                    vis_state.active_field_name
                ),
                min_val,
                max_val,
                field_unit,
                2,
                |t| sample_slice_colormap_rgba(cmap, t, min_val, max_val),
            );
        }
    });

    // 3. Selected Slicing Isosurfaces Inspector
    ui.add_space(6.0);
    ui.separator();
    egui::CollapsingHeader::new(
        egui::RichText::new("Non-Planar Layers (τ_k) Inspector").strong(),
    )
    .id_source("view_selected_slicing_isosurfaces_section")
    .default_open(true)
    .show(ui, |ui| {
        if ui
            .checkbox(
                &mut vis_state.show_selected_layers,
                "Show Selected Isosurfaces",
            )
            .changed()
        {
            vis_state.dirty_selected_layers_mesh = true;
            vis_state.dirty_slice = true;
            if vis_state.show_selected_layers && vis_state.solid_opacity > 0.45 {
                vis_state.solid_opacity = 0.12;
            }
            if vis_state.show_selected_layers && !has_eikonal && !is_computing {
                spawn_field_compute_task(
                    vis_state,
                    plate,
                    config,
                    sel_id,
                    obj_name,
                    obj_rotation,
                );
            }
        }

        if vis_state.show_selected_layers {
            ui.horizontal(|ui| {
                if ui
                    .checkbox(&mut vis_state.show_selected_layer_surfaces, "3D Surfaces")
                    .changed()
                {
                    vis_state.dirty_selected_layers_mesh = true;
                }
                ui.checkbox(
                    &mut vis_state.show_selected_layer_contours,
                    "3D Perimeter Contours",
                );
            });

            ui.horizontal(|ui| {
                ui.label("Color By:");
                let combo_w = (ui.available_width() - 8.0).clamp(130.0, 220.0);
                egui::ComboBox::from_id_source("selected_layers_color_mode")
                    .selected_text(vis_state.selected_layers_color_mode.short_label())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for mode in [
                            SelectedLayersColorMode::ByLayerThickness,
                            SelectedLayersColorMode::ByBranch,
                            SelectedLayersColorMode::AlternatingBands,
                        ] {
                            if ui
                                .selectable_value(
                                    &mut vis_state.selected_layers_color_mode,
                                    mode,
                                    mode.label(),
                                )
                                .changed()
                            {
                                vis_state.dirty_selected_layers_mesh = true;
                            }
                        }
                    });
            });

            render_selected_layers_color_legend(ui, vis_state, config);
        }

        let obj_world_tf = vis_state
            .last_surface_color_transform
            .unwrap_or_else(|| Transform::from_rotation(obj_rotation));

        if let Some(ref res) = vis_state.cached_selected_layers.clone() {
            let total_layers = res.layers.len();
            if total_layers > 0 {
                render_layer_and_move_scrub_controls(
                    ui,
                    vis_state,
                    config,
                    obj_world_tf,
                    !vis_state.show_toolpaths,
                    "selected_isosurfaces_scrub",
                );

                let mut global_min_h = f32::MAX;
                let mut global_max_h = 0.0f32;
                let mut sum_h = 0.0f32;
                for l in &res.layers {
                    global_min_h = global_min_h.min(l.mean_thickness_mm);
                    global_max_h = global_max_h.max(l.mean_thickness_mm);
                    sum_h += l.mean_thickness_mm;
                }
                let avg_h = sum_h / (total_layers as f32);

                ui.small(format!(
                    "Selected: {} layers across {} branch(es) | Avg Δh: {:.3} mm ([{:.3}, {:.3}] mm)",
                    total_layers, res.branch_count, avg_h, global_min_h, global_max_h
                ));

                egui::CollapsingHeader::new(format!("Selected Isosurface List ({total_layers})"))
                    .id_source("selected_isosurface_layer_list")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.small(
                            "Hover a layer to highlight it in 3D; click to inspect in the single-isosurface viewer.",
                        );
                        egui::ScrollArea::vertical()
                            .max_height(180.0)
                            .show(ui, |ui| {
                                for (idx, layer) in res.layers.iter().enumerate() {
                                    let tag = match (layer.is_first_layer, layer.is_top_layer) {
                                        (true, true) => " [FIRST+TOP]",
                                        (true, false) => " [FIRST]",
                                        (false, true) => " [TOP]",
                                        (false, false) => "",
                                    };
                                    let label = format!(
                                        "#{} (B{} {}/{})  τ={:.2} mm | Δh={:.3} mm [{:.2}..{:.2}]{}",
                                        layer.layer_index,
                                        layer.branch_id,
                                        layer.branch_layer_index + 1,
                                        layer.branch_layer_count,
                                        layer.isovalue,
                                        layer.mean_thickness_mm,
                                        layer.min_thickness_mm,
                                        layer.max_thickness_mm,
                                        tag
                                    );
                                    let resp = ui.selectable_label(
                                        vis_state.highlighted_selected_layer == Some(idx),
                                        label,
                                    );
                                    if resp.hovered() {
                                        vis_state.highlighted_selected_layer = Some(idx);
                                    }
                                    if resp.clicked() {
                                        vis_state.active_field_name = "eikonal".to_string();
                                        vis_state.isovalue = layer.isovalue;
                                        vis_state.show_isosurface = true;
                                        if vis_state.solid_opacity > 0.5 {
                                            vis_state.solid_opacity = 0.25;
                                        }
                                        vis_state.dirty_isosurface = true;
                                        vis_state.dirty_slice = true;
                                    }
                                }
                            });
                    });
            }
        }
    });

    // 4. 3D Toolpaths & G-code Preview
    ui.add_space(6.0);
    ui.separator();
    egui::CollapsingHeader::new(
        egui::RichText::new("3D Toolpaths & G-code Preview").strong(),
    )
    .id_source("view_3d_toolpaths_gcode_preview_section")
    .default_open(true)
    .show(ui, |ui| {
        if ui
            .checkbox(&mut vis_state.show_toolpaths, "Show 3D Shell Toolpaths")
            .changed()
        {
            vis_state.dirty_toolpaths_mesh = true;
            if vis_state.show_toolpaths && vis_state.solid_opacity > 0.35 {
                vis_state.solid_opacity = 0.10;
            }
            if vis_state.show_toolpaths && !has_eikonal && !is_computing {
                spawn_field_compute_task(
                    vis_state,
                    plate,
                    config,
                    sel_id,
                    obj_name,
                    obj_rotation,
                );
            }
        }

        if vis_state.show_toolpaths {
            let obj_world_tf = vis_state
                .last_surface_color_transform
                .unwrap_or_else(|| Transform::from_rotation(obj_rotation));
            render_layer_and_move_scrub_controls(
                ui,
                vis_state,
                config,
                obj_world_tf,
                true,
                "toolpaths_gcode_preview_scrub",
            );
            ui.add_space(3.0);

            render_toolpath_quality_and_transparency_controls(ui, vis_state);
            ui.add_space(2.0);

            ui.horizontal(|ui| {
                ui.label("Color By:");
                let combo_w = (ui.available_width() - 8.0).clamp(130.0, 220.0);
                egui::ComboBox::from_id_source("toolpath_color_mode_combo")
                    .selected_text(vis_state.toolpath_color_mode.short_label())
                    .width(combo_w)
                    .show_ui(ui, |ui| {
                        for mode in ToolpathColorMode::ALL {
                            if ui
                                .selectable_value(
                                    &mut vis_state.toolpath_color_mode,
                                    mode,
                                    mode.label(),
                                )
                                .changed()
                            {
                                vis_state.dirty_toolpaths_mesh = true;
                            }
                        }
                    });
            });

            if vis_state.toolpath_color_mode != ToolpathColorMode::ByFeatureType {
                render_toolpath_color_legend(ui, vis_state, config);
                ui.add_space(3.0);
            }

            render_interactive_toolpath_line_types_legend(ui, vis_state);
        }

        if let Some(ref planned) = vis_state.cached_planned_gcode {
            let total_sec = planned.total_time_s.round() as u64;
            let hrs = total_sec / 3600;
            let mins = (total_sec % 3600) / 60;
            let secs = total_sec % 60;
            ui.small(format!(
                "Est. Time: {:02}h {:02}m {:02}s | Filament: {:.2} m ({:.1} g, {:.0} mm³)",
                hrs,
                mins,
                secs,
                planned.total_filament_mm / 1000.0,
                planned.total_filament_grams,
                planned.total_filament_volume_mm3
            ));
            if planned.toolchange_count > 0 {
                let tc_total_s =
                    (planned.toolchange_count as f32) * config.gcode.tool_change_time_s.max(0.0);
                ui.small(format!(
                    "Moves: {} extruding, {} travels, {} retracts, {} toolchanges ({:.0}s)",
                    planned.extruding_move_count,
                    planned.travel_move_count,
                    planned.retract_count,
                    planned.toolchange_count,
                    tc_total_s,
                ));
            } else {
                ui.small(format!(
                    "Moves: {} extruding, {} travels, {} retracts",
                    planned.extruding_move_count, planned.travel_move_count, planned.retract_count,
                ));
            }
        }
    });
}

/// Synchronizes the Rust `SlicerConfig` (nozzle axis, bed-coordinate gravity vector,
/// multi-tool specifications & XZ clearance profiles, variable layer height, and
/// Eikonal solver iterations) into the C++ compute backend.
pub fn sync_config_to_backend(config: &SlicerConfig) {
    ffi::set_nozzle_axis_bed(config.machine.nozzle_axis_bed);
    ffi::set_gravity_direction_bed(config.machine.gravity_direction_bed);
    ffi::set_bed_tilts_with_rotary_axes(config.machine.gravity_couples_to_bed_tilt());

    let is_five_axis = config.machine.kinematics_mode.is_five_axis();
    let machine_max_tilt_deg = match config.machine.kinematics_mode {
        FiveAxisKinematicsMode::FiveAxisTiltingBedAB => config
            .machine
            .a_min_deg
            .abs()
            .max(config.machine.a_max_deg.abs())
            .max(config.machine.b_min_deg.abs())
            .max(config.machine.b_max_deg.abs()),
        FiveAxisKinematicsMode::FiveAxisBC => config
            .machine
            .b_min_deg
            .abs()
            .max(config.machine.b_max_deg.abs()),
        FiveAxisKinematicsMode::ThreeAxis => 0.0,
    };

    let ffi_tools: Vec<ffi::ToolSpec> = config
        .tools
        .iter()
        .map(|tool| {
            let mut spec = tool.to_ffi_tool_spec();
            // On a 5-axis machine (e.g. a 5-axis tilting-bed printer where the bed tilts by A and B
            // while the gantry remains stationary), the rotary axes tilt every tool relative to bed
            // coordinates even if the toolhead preset itself is a 3-axis gantry toolhead.
            if is_five_axis && machine_max_tilt_deg > 1.0e-3 {
                spec.has_rotational_axes = true;
                if !tool.has_rotational_axes || spec.max_angle_from_bed_normal_deg <= 1.0e-3 {
                    spec.max_angle_from_bed_normal_deg = machine_max_tilt_deg;
                } else {
                    spec.max_angle_from_bed_normal_deg =
                        spec.max_angle_from_bed_normal_deg.min(machine_max_tilt_deg);
                }
            }
            spec
        })
        .collect();
    ffi::set_machine_tools(&ffi_tools, config.active_tool_index as u32);

    ffi::set_slice_layer_height(ffi::LayerHeightSpec {
        target_mm: config.slicing.layer_height.target_mm,
        min_mm: config.slicing.layer_height.min_mm,
        max_mm: config.slicing.layer_height.max_mm,
        first_layer_mm: config.slicing.layer_height.first_layer_mm,
        top_layer_mm: config.slicing.layer_height.top_layer_mm,
    });

    let nozzle_diameter_mm = config
        .tools
        .get(config.active_tool_index)
        .or_else(|| config.tools.first())
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(0.4);

    ffi::set_shell_path_spec(ffi::ShellPathSpec {
        shell_thickness_mm: config.slicing.shell_thickness_mm,
        inner_wall_count: config.slicing.inner_wall_count,
        outer_wall_width: ffi::LineWidthSpec {
            target_mm: config.slicing.outer_wall_width.target_mm,
            min_mm: config.slicing.outer_wall_width.min_mm,
            max_mm: config.slicing.outer_wall_width.max_mm,
        },
        inner_wall_width: ffi::LineWidthSpec {
            target_mm: config.slicing.inner_wall_width.target_mm,
            min_mm: config.slicing.inner_wall_width.min_mm,
            max_mm: config.slicing.inner_wall_width.max_mm,
        },
        solid_infill_width: ffi::LineWidthSpec {
            target_mm: config.slicing.solid_infill_width.target_mm,
            min_mm: config.slicing.solid_infill_width.min_mm,
            max_mm: config.slicing.solid_infill_width.max_mm,
        },
        solid_infill_pattern: config.slicing.solid_infill_pattern.as_u8(),
        min_extrusion_length_mm: config.slicing.min_extrusion_length_mm,
        sparse_infill_density: (config.slicing.infill_density_percent / 100.0).clamp(0.0, 1.0),
        sparse_infill_width: ffi::LineWidthSpec {
            target_mm: config.slicing.sparse_infill_width.target_mm,
            min_mm: config.slicing.sparse_infill_width.min_mm,
            max_mm: config.slicing.sparse_infill_width.max_mm,
        },
        sparse_infill_pattern: config.slicing.sparse_infill_pattern.as_u8(),
        infill_angle_deg: config.slicing.infill_angle_deg,
        nozzle_diameter_mm,
        overhang_angle_threshold_deg: config.slicing.overhang_angle_threshold_deg,
        max_bridge_length_mm: config.slicing.max_bridge_length_mm,
        wave_overhang_overlap_fraction: config.slicing.wave_overhang_overlap_fraction,
        one_perimeter_on_conformal_patches: config.slicing.one_perimeter_on_conformal_patches,
        orient_outer_surfaces_to_normal: config.slicing.orient_outer_surfaces_to_normal,
        outer_surface_parallel_angle_deg: config.slicing.outer_surface_parallel_angle_deg,
        brick_layer_mode: config.slicing.brick_layer_mode.as_u8(),
        enable_top_surface_lofting: config.slicing.enable_top_surface_lofting,
    });

    ffi::set_order_field_mode(config.eikonal.order_field_mode.as_u8());
    ffi::set_eikonal_solver_iterations(config.eikonal.solver_iterations);
    ffi::set_multi_resolution_grid_config(
        config.eikonal.enable_multi_resolution_grid,
        config.eikonal.coarse_grid_ratio,
    );
    ffi::set_overhang_orientation_bias_spec(ffi::OverhangBiasSpec {
        enabled: config.eikonal.enable_overhang_orientation_bias,
        weight: config.eikonal.overhang_bias_weight,
        orthogonal_supported_angle_deg: config.eikonal.orthogonal_supported_angle_deg,
        orthogonal_overhang_angle_deg: config.eikonal.orthogonal_overhang_angle_deg,
        transition_width_deg: config.eikonal.overhang_bias_transition_deg,
    });
    ffi::set_support_spec(ffi::SupportSpec {
        enabled: config.slicing.enable_support_generation,
        support_tool_index: config
            .slicing
            .support_tool_index
            .map(|idx| idx as i32)
            .unwrap_or(-1),
        support_style: config.slicing.support_style.as_u8(),
        on_bed_only: config.slicing.support_on_bed_only,
        seed_order_field: config.slicing.seed_order_field_from_supports,
        only_unreachable_and_long_bridges: config.slicing.support_only_unreachable_and_bridges,
        max_bridge_length_mm: config.slicing.max_bridge_length_mm,
        z_gap_mm: config.slicing.support_z_gap_mm,
        xy_gap_mm: config.slicing.support_xy_gap_mm,
        interface_layers: config.slicing.support_interface_layers,
        bed_interface_layers: config.slicing.support_bed_interface_layers,
        base_spacing_mm: config.slicing.support_base_spacing_mm,
        interface_spacing_mm: config.slicing.support_interface_spacing_mm,
        line_width_mm: config.slicing.support_line_width_mm,
        tree_branch_angle_deg: config.slicing.tree_branch_angle_deg,
        tree_tip_diameter_mm: config.slicing.tree_tip_diameter_mm,
        tree_trunk_max_diameter_mm: config.slicing.tree_trunk_max_diameter_mm,
        tree_branch_wall_count: config.slicing.tree_branch_wall_count,
    });
    ffi::set_conformal_skin_spec(ffi::ConformalSkinSpec {
        auto_enabled: config.slicing.enable_auto_conformal_skin,
        default_skin_layers: config.slicing.conformal_skin_layers,
        min_oblique_angle_deg: config.slicing.conformal_skin_min_oblique_deg,
        max_slope_angle_deg: config.slicing.conformal_skin_max_slope_deg,
    });
}

/// Compiles all active surface-face hints and 3D point hints for `object_id` into
/// backend `ffi::SpatialOrientationHint` records (in object-local OpenVDB coordinates).
pub fn collect_object_hints_for_backend(
    config: &SlicerConfig,
    plate: &BuildPlateState,
    object_id: u64,
) -> Vec<ffi::SpatialOrientationHint> {
    let mut ffi_hints = Vec::new();

    if let Some(obj) = plate.objects.iter().find(|o| o.object_id == object_id) {
        for sh in config
            .eikonal
            .surface_hints
            .iter()
            .filter(|h| h.object_id == object_id)
        {
            if let Some(patch) = obj.surface_patches.get(sh.patch_id as usize) {
                let kind = match sh.mode {
                    SurfaceHintMode::NormalOrthogonalToSurface => 1u8, // HintKind::Orthogonality
                    SurfaceHintMode::NormalTangentToSurface => 0u8,    // HintKind::Tangency
                    SurfaceHintMode::ConformalSurfaceSkin => 2u8,      // Conformal SDF Surface Skin
                };
                let sample_count = patch.sample_points.len().max(1) as f32;
                let sample_spacing = (patch.area / sample_count).sqrt();
                let effective_radius = sh.influence_depth_mm.max(sample_spacing * 1.35);

                for &(pos, normal) in &patch.sample_points {
                    ffi_hints.push(ffi::SpatialOrientationHint {
                        kind,
                        position_bed: [pos.x, pos.y, pos.z],
                        direction_bed: [normal.x, normal.y, normal.z],
                        influence_radius_mm: effective_radius,
                        weight: sh.weight,
                        apply_on_surface_only: true,
                        skin_layer_count: sh.skin_layer_count,
                    });
                }
            }
        }
    }

    for ph in config
        .eikonal
        .point_hints
        .iter()
        .filter(|h| h.object_id == object_id)
    {
        ffi_hints.push(ffi::SpatialOrientationHint {
            kind: 1u8, // HintKind::Orthogonality (aligns \nabla \tau with ph.local_direction)
            position_bed: ph.local_position,
            direction_bed: ph.local_direction,
            influence_radius_mm: ph.influence_radius_mm,
            weight: ph.weight,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        });
    }

    ffi_hints
}

/// Compiles all active surface-face hints and 3D point hints for `object_id` into
/// backend `ffi::SpatialOrientationHint` records (in object-local OpenVDB coordinates)
/// and pushes them to the C++ Eikonal solver.
pub fn sync_object_hints_to_backend(
    config: &SlicerConfig,
    plate: &BuildPlateState,
    object_id: u64,
) {
    let ffi_hints = collect_object_hints_for_backend(config, plate, object_id);
    ffi::set_spatial_orientation_hints(&ffi_hints);
}

/// Spawns an asynchronous background task to compute OpenVDB SDF, constrained Eikonal,
/// overhang, and 5/6-axis tool orientation fields for all printable objects on the active
/// build plate (plus `object_id` if explicitly selected) without blocking the UI thread.
fn spawn_field_compute_task(
    vis_state: &mut FieldVisualizationState,
    plate: &mut BuildPlateState,
    config: &SlicerConfig,
    object_id: u64,
    object_name: &str,
    obj_rotation: Quat,
) {
    sync_config_to_backend(config);
    let nozzle_bed = Vec3::from_array(config.machine.nozzle_axis_bed);
    let gravity_bed = Vec3::from_array(config.machine.gravity_direction_bed);
    let active_plate_idx = plate.active_plate_index;

    let mut batch: Vec<(u64, Quat, Vec<ffi::SpatialOrientationHint>)> = Vec::new();
    for obj in &plate.objects {
        if obj.plate_index == active_plate_idx && obj.printable && obj.object_id != object_id {
            batch.push((
                obj.object_id,
                obj.cached_rotation,
                collect_object_hints_for_backend(config, plate, obj.object_id),
            ));
        }
    }
    // Always place `object_id` last so the backend's global nozzle/gravity/hints state matches
    // the selected object when the background task finishes.
    batch.push((
        object_id,
        obj_rotation,
        collect_object_hints_for_backend(config, plate, object_id),
    ));

    let batch_count = batch.len();
    let voxel_size = config.eikonal.voxel_size_mm;
    let narrow_band = config.eikonal.narrow_band_width_voxels as f32;
    let task = IoTaskPool::get().spawn(async move {
        let mut selected_ok = false;
        for (oid, rot, hints) in batch {
            let inv_rot = rot.inverse();
            let local_nozzle = (inv_rot * nozzle_bed).normalize_or_zero();
            if local_nozzle.length_squared() > 1.0e-6 {
                ffi::set_nozzle_axis_bed(local_nozzle.to_array());
            }
            let local_gravity = (inv_rot * gravity_bed).normalize_or_zero();
            if local_gravity.length_squared() > 1.0e-6 {
                ffi::set_gravity_direction_bed(local_gravity.to_array());
            }
            ffi::set_spatial_orientation_hints(&hints);
            let ok = ffi::recompute_object_fields(oid, voxel_size, narrow_band);
            if oid == object_id {
                selected_ok = ok;
            }
        }
        selected_ok
    });
    vis_state.last_solved_field_rotation = Some(obj_rotation);
    let status_target = if batch_count > 1 {
        format!("{batch_count} objects on {}", plate.active_plate_name())
    } else {
        format!("'{object_name}'")
    };
    vis_state.pending_field_task = Some((object_id, status_target.clone(), task));
    plate.status_message = format!(
        "Computing OpenVDB scalar fields for {status_target} in background (voxel {voxel_size:.2} mm)..."
    );
}

/// Polls background import/export/field-computation tasks on each frame without blocking the macOS main thread.
#[allow(clippy::too_many_arguments)]
pub fn poll_io_tasks_system(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
    mut preset_lib: ResMut<PresetLibrary>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut moonraker_state: ResMut<MoonrakerUiState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut object_transforms: Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    mut camera_query: Query<
        (&Camera, &GlobalTransform, &mut OrbitCamera, &mut Transform),
        Without<CadObjectEntity>,
    >,
) {
    // 1. Poll pending multi-file import task
    if let Some(ref mut task) = plate.pending_load_task {
        if let Some(result) = block_on(future::poll_once(task)) {
            plate.pending_load_task = None;

            let Some(mut report) = result else {
                plate.status_message = "File import cancelled.".to_string();
                return;
            };

            if let Some(restored_cfg) = report.restored_config {
                *config = restored_cfg;
                preset_lib.selected_printer = config.printer_preset_name.clone();
                preset_lib.selected_tool = config.tool_preset_name.clone();
                preset_lib.selected_filament = config.filament_preset_name.clone();
                preset_lib.selected_process = config.process_preset_name.clone();
                sync_config_to_backend(&config);
            } else if !report.filament_slots.is_empty() {
                config.apply_imported_filament_slots(report.filament_slots.clone());
                sync_config_to_backend(&config);
                vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
            }
            if let Some(proj_filename) = report.manifold_project_filename {
                plate.project_3mf_filename = Some(proj_filename);
            }

            plate.import_notices = report.notices;
            plate.import_errors = report.errors;

            let loaded_count = report.objects.len();
            if loaded_count > 0 {
                if !report.plates.is_empty() {
                    if plate.objects.is_empty() {
                        plate.plates = report
                            .plates
                            .into_iter()
                            .map(|p| BuildPlateInfo { name: p.name })
                            .collect();
                        plate.active_plate_index = 0;
                    } else {
                        for (idx, p) in report.plates.into_iter().enumerate() {
                            if idx < plate.plates.len() {
                                plate.plates[idx].name = p.name;
                            } else {
                                plate.plates.push(BuildPlateInfo { name: p.name });
                            }
                        }
                    }
                } else {
                    // Single-plate / CAD / STL import: place newly imported objects onto the currently active plate
                    let active_idx = plate.active_plate_index;
                    for obj in &mut report.objects {
                        obj.plate_index = active_idx;
                    }
                }

                spawn_loaded_objects(
                    &mut commands,
                    render_device.as_deref(),
                    &mut meshes,
                    &mut materials,
                    &mut plate,
                    report.objects,
                );
                // Populate initial painted filament / true-colour vertex colors on multi-colour objects
                let any_true_color = plate.objects.iter().any(|o| !o.vertex_colors_rgba.is_empty());
                if any_true_color && config.slicing.color_mixing.enabled {
                    vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                }
                for obj in &plate.objects {
                    if !obj.vertex_filament_slots.is_empty() || !obj.vertex_colors_rgba.is_empty() {
                        if let Some(mesh) = meshes.get_mut(&obj.mesh_handle) {
                            mesh.insert_attribute(
                                Mesh::ATTRIBUTE_COLOR,
                                crate::field_vis::default_cad_vertex_colors(
                                    obj.vertices.len(),
                                    &obj.vertex_filament_slots,
                                    &obj.vertex_colors_rgba,
                                    &config,
                                ),
                            );
                        }
                    }
                }
                vis_state.dirty_surface_colors = true;
                plate.sync_plate_visibility(&mut object_transforms);
                frame_camera_to_plate(&plate, &config, &mut camera_query);
                if plate.plates.len() > 1 {
                    plate.status_message = format!(
                        "Loaded {loaded_count} object(s) across {} build plates (viewing {}).",
                        plate.plates.len(),
                        plate.active_plate_name()
                    );
                } else {
                    plate.status_message = format!(
                        "Loaded {loaded_count} object(s); {} total on {}.",
                        plate.active_plate_object_count(),
                        plate.active_plate_name()
                    );
                }
            } else if !plate.import_errors.is_empty() {
                plate.status_message = plate.import_errors[0].clone();
            } else {
                plate.status_message = "No supported 3D objects found in selected file(s).".to_string();
            }
        }
    }

    // 2. Poll pending 3MF project or G-code file save task
    if let Some(ref mut task) = plate.pending_save_task {
        if let Some(result) = block_on(future::poll_once(task)) {
            plate.pending_save_task = None;
            match result {
                Some(Ok(saved_info)) => {
                    plate.status_message = format!("Exported to {saved_info}.");
                }
                Some(Err(err)) => {
                    plate.status_message = format!("Export error: {err}");
                }
                None => {
                    plate.status_message = "Export cancelled.".to_string();
                }
            }
        }
    }

    // 3. Poll pending background OpenVDB scalar field computation task
    if let Some((obj_id, ref obj_name, ref mut task)) = vis_state.pending_field_task {
        if let Some(succeeded) = block_on(future::poll_once(task)) {
            let finished_name = obj_name.clone();
            vis_state.pending_field_task = None;
            if succeeded {
                if plate.selected_object_id == Some(obj_id) {
                    vis_state.available_fields = ffi::list_object_fields(obj_id);
                    if !vis_state.available_fields.is_empty()
                        && !vis_state
                            .available_fields
                            .contains(&vis_state.active_field_name)
                    {
                        vis_state.active_field_name = vis_state.available_fields[0].clone();
                    }
                    vis_state.mark_all_dirty();
                    vis_state.refresh_stats_for(obj_id);
                    if vis_state.show_selected_layers || vis_state.show_toolpaths {
                        let need_toolpaths = vis_state.show_toolpaths;
                        let printable_ids = plate.active_plate_printable_object_ids();
                        vis_state.spawn_slicing_task_for_objects(
                            obj_id,
                            printable_ids,
                            &config,
                            true,
                            need_toolpaths,
                        );
                    }
                }
                plate.status_message = format!(
                    "Computed OpenVDB SDF & constrained Eikonal fields for '{finished_name}' (voxel {:.2} mm, {} iters).",
                    config.eikonal.voxel_size_mm,
                    config.eikonal.solver_iterations
                );
            } else {
                plate.status_message = format!(
                    "Failed to compute OpenVDB scalar fields for '{finished_name}'."
                );
            }
        }
    }

    // 4. Poll pending background layer selection & shell toolpath generation task
    if let Some((obj_id, ref mut task)) = vis_state.pending_slicing_task {
        if let Some((layers_opt, plate_toolpaths)) = block_on(future::poll_once(task)) {
            vis_state.pending_slicing_task = None;
            if plate.selected_object_id == Some(obj_id) {
                let num_layers = layers_opt
                    .as_ref()
                    .map(|l| l.layers.len())
                    .or_else(|| vis_state.cached_selected_layers.as_ref().map(|l| l.layers.len()))
                    .unwrap_or(0);
                let num_objs = plate_toolpaths.len();
                let num_paths: Option<u32> = if plate_toolpaths.is_empty() {
                    None
                } else {
                    Some(
                        plate_toolpaths
                            .iter()
                            .map(|(_, t)| t.total_path_count)
                            .sum(),
                    )
                };
                vis_state.apply_slicing_task_result(layers_opt, plate_toolpaths);
                if let Some(paths) = num_paths {
                    plate.status_message = format!(
                        "Generated {paths} non-planar shell toolpaths across {num_objs} printable object(s) ({num_layers} layers)."
                    );
                } else {
                    plate.status_message = format!(
                        "Selected {num_layers} non-planar slicing isosurfaces."
                    );
                }
            }
        }
    }

    // 5. Poll pending Moonraker certificate file picker task
    if let Some((kind, ref mut task)) = moonraker_state.pending_cert_pick_task {
        if let Some(picked_opt) = block_on(future::poll_once(task)) {
            moonraker_state.pending_cert_pick_task = None;
            if let Some(path_str) = picked_opt {
                match kind {
                    CertPickKind::CaCert => config.network.ca_cert_pem_path = path_str,
                    CertPickKind::ClientCert => config.network.client_cert_pem_path = path_str,
                    CertPickKind::ClientKey => config.network.client_key_pem_path = path_str,
                }
                config.network.sanitize();
            }
        }
    }

    // 6. Poll pending printer connection test task
    let proto_label = config.network.protocol.short_label();
    if let Some(ref mut task) = moonraker_state.pending_test_task {
        if let Some(res) = block_on(future::poll_once(task)) {
            moonraker_state.pending_test_task = None;
            match &res {
                Ok(report) => {
                    plate.status_message = report.summary_line();
                }
                Err(err) => {
                    plate.status_message = format!("{proto_label} connection test failed: {err}");
                }
            }
            moonraker_state.last_test_result = Some(res);
        }
    }

    // 7. Poll pending printer G-code upload / upload-and-print task
    if let Some(ref mut task) = moonraker_state.pending_upload_task {
        if let Some(res) = block_on(future::poll_once(task)) {
            moonraker_state.pending_upload_task = None;
            match &res {
                Ok(report) => {
                    plate.status_message = report.summary_line();
                    // Force an immediate status refresh after upload / print start
                    moonraker_state.last_poll_instant = None;
                }
                Err(err) => {
                    plate.status_message = format!("{proto_label} upload failed: {err}");
                }
            }
            moonraker_state.last_upload_result = Some(res);
        } else {
            let snap = moonraker_state.upload_progress.snapshot();
            if !snap.stage.is_empty() {
                plate.status_message = snap.stage;
            }
        }
    }

    // 8. Poll pending Moonraker live status task & schedule periodic status polling
    if let Some(ref mut task) = moonraker_state.pending_status_task {
        if let Some((status_res, updated_cache)) = block_on(future::poll_once(task)) {
            moonraker_state.pending_status_task = None;
            moonraker_state.cached_metadata_estimate = updated_cache;
            moonraker_state.last_status = Some(status_res);
        }
    } else if config.network.is_configured() && config.network.enable_status_polling {
        let interval = std::time::Duration::from_secs_f32(config.network.poll_interval_s.max(1.0));
        let due = moonraker_state
            .last_poll_instant
            .is_none_or(|t| t.elapsed() >= interval);
        if due {
            let net_cfg = config.network.clone();
            let cached_meta = moonraker_state.cached_metadata_estimate.clone();
            moonraker_state.last_poll_instant = Some(std::time::Instant::now());
            moonraker_state.pending_status_task = Some(IoTaskPool::get().spawn(async move {
                query_moonraker_print_status(&net_cfg, cached_meta)
            }));
        }
    }
}

fn frame_camera_to_plate(
    plate: &BuildPlateState,
    config: &SlicerConfig,
    camera_query: &mut Query<
        (&Camera, &GlobalTransform, &mut OrbitCamera, &mut Transform),
        Without<CadObjectEntity>,
    >,
) {
    let bed_span = config.bed.width_x.max(config.bed.depth_y).max(100.0);
    for (_, _, mut orbit, mut transform) in camera_query.iter_mut() {
        orbit.focus = Vec3::new(0.0, 0.0, 20.0);
        orbit.radius = if plate.objects.is_empty() {
            bed_span * 1.35
        } else {
            bed_span * 1.15
        };
        let eye = compute_orbit_eye(&orbit);
        *transform = Transform::from_translation(eye).looking_at(orbit.focus, Vec3::Z);
    }
}

/// Renders the floating Calibration Utilities window for:
/// 1. Orca-style Extrusion Multiplier (Pass 1 Coarse, Pass 2 Fine) & Pass 3 High-Flow Die Swell (`swell_high`),
/// 2. Ellis-style 90° sharp-corner chevron Pressure Advance (Dual-Band Shear-Thinning `k_low` + `k_high` or Single-Band),
/// 3. Constant-Speed Flow-Step & Staccato Striation Transient Melt-Zone Pressure (`γ` and `τ_melt`).
fn render_calibration_utilities_window(
    ctx: &egui::Context,
    calib_state: &mut CalibrationUiState,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    plate: &mut BuildPlateState,
    moonraker_state: &mut MoonrakerUiState,
    is_io_busy: bool,
    is_computing: bool,
) {
    if !calib_state.show_window {
        return;
    }

    let mut open = calib_state.show_window;
    egui::Window::new("Calibration Utilities — Rheology, Flow & Pressure")
        .id(egui::Id::new("calibration_utilities_floating_window"))
        .open(&mut open)
        .default_width(510.0)
        .min_width(440.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Top utility kind tab strip
            ui.horizontal_wrapped(|ui| {
                for kind in CalibrationUtilityKind::ALL {
                    let is_active = calib_state.active_kind == kind;
                    if render_delineated_pill_tab(ui, is_active, kind.label()).clicked() {
                        calib_state.active_kind = kind;
                    }
                }
            });
            ui.small(
                egui::RichText::new(
                    "All patterns respect the active machine's 5/6-axis NozzleFrame and Gravity Vector in bed coordinates.",
                )
                .color(egui::Color32::GRAY),
            );
            ui.separator();

            let mut request_preview = false;
            let mut request_export = false;
            let mut request_upload: Option<bool> = None;
            let mut config_modified = false;

            // Calibration Tool / Extruder Selector Bar
            if config.tools.is_empty() {
                config.tools.push(ToolConfig::default());
                config.active_tool_index = 0;
            }
            if let Some(sel_idx) = calib_state.selected_tool_index {
                if sel_idx >= config.tools.len() {
                    calib_state.selected_tool_index = Some(config.tools.len() - 1);
                }
            }
            let resolved_tool_idx = calib_state
                .selected_tool_index
                .unwrap_or(config.active_tool_index)
                .min(config.tools.len() - 1);

            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Calibration Tool:").strong());
                let selected_tool_label = match calib_state.selected_tool_index {
                    None => config
                        .tools
                        .get(resolved_tool_idx)
                        .map(|t| {
                            format!(
                                "Active Tool: {} ({:.2}mm, {:.0}°C)",
                                t.name, t.nozzle_diameter_mm, t.temperature_celsius
                            )
                        })
                        .unwrap_or_else(|| "Active Tool".to_string()),
                    Some(idx) => config
                        .tools
                        .get(idx)
                        .map(|t| {
                            format!(
                                "{} (T{}, {:.2}mm, {:.0}°C)",
                                t.name, t.id, t.nozzle_diameter_mm, t.temperature_celsius
                            )
                        })
                        .unwrap_or_else(|| format!("Tool #{idx}")),
                };
                egui::ComboBox::from_id_source("calibration_tool_selector_combo")
                    .selected_text(selected_tool_label)
                    .width(235.0)
                    .show_ui(ui, |ui| {
                        let active_name = config
                            .active_tool()
                            .map(|t| {
                                format!(
                                    "Active Tool ({} — {:.2}mm, {:.0}°C)",
                                    t.name, t.nozzle_diameter_mm, t.temperature_celsius
                                )
                            })
                            .unwrap_or_else(|| "Active Tool".to_string());
                        ui.selectable_value(
                            &mut calib_state.selected_tool_index,
                            None,
                            active_name,
                        );
                        for (idx, tool) in config.tools.iter().enumerate() {
                            let lbl = format!(
                                "{} (T{}, {:.2}mm, {:.0}°C)",
                                tool.name, tool.id, tool.nozzle_diameter_mm, tool.temperature_celsius
                            );
                            ui.selectable_value(
                                &mut calib_state.selected_tool_index,
                                Some(idx),
                                lbl,
                            );
                        }
                    });

                if ui
                    .small_button("+ Tool")
                    .on_hover_text("Add a new tool/extruder to the Multi-Tool Registry and select it for calibration")
                    .clicked()
                {
                    let next_id = config.tools.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                    let base_noz = config
                        .tools
                        .get(resolved_tool_idx)
                        .map(|t| t.nozzle_diameter_mm)
                        .unwrap_or(0.40);
                    let base_temp = config
                        .tools
                        .get(resolved_tool_idx)
                        .map(|t| t.temperature_celsius)
                        .unwrap_or(215.0);
                    config.tools.push(ToolConfig::standard_tool(
                        next_id,
                        format!("T{next_id} - {base_noz:.1}mm Extruder"),
                        base_noz,
                        base_temp,
                    ));
                    calib_state.selected_tool_index = Some(config.tools.len() - 1);
                    config_modified = true;
                }
            });

            if let Some(cal_tool) = config.tools.get_mut(resolved_tool_idx) {
                ui.horizontal_wrapped(|ui| {
                    ui.small(
                        egui::RichText::new(format!("T{} Settings:", cal_tool.id)).strong(),
                    );
                    if ui
                        .add(
                            egui::DragValue::new(&mut cal_tool.nozzle_diameter_mm)
                                .speed(0.02)
                                .range(0.15..=2.0)
                                .prefix("Nozzle: ")
                                .suffix(" mm"),
                        )
                        .changed()
                    {
                        config_modified = true;
                    }
                    if ui
                        .add(
                            egui::DragValue::new(&mut cal_tool.temperature_celsius)
                                .speed(1.0)
                                .range(150.0..=380.0)
                                .prefix("Nozzle Temp: ")
                                .suffix(" °C"),
                        )
                        .changed()
                    {
                        config_modified = true;
                    }
                    if ui
                        .add(
                            egui::DragValue::new(&mut config.gcode.bed_temperature_celsius)
                                .speed(1.0)
                                .range(0.0..=150.0)
                                .prefix("Bed Temp: ")
                                .suffix(" °C"),
                        )
                        .changed()
                    {
                        config_modified = true;
                    }
                });
            }
            ui.separator();

            match calib_state.active_kind {
                CalibrationUtilityKind::ExtrusionMultiplier => {
                    ui.label(
                        egui::RichText::new(
                            "Orca-Style Surface Quality Coupon Array & High-Flow Die Swell",
                        )
                        .strong(),
                    );
                    ui.small(
                        "Prints numbered coupons with a solid monotonic top surface and an inset inspection frame. Inspect the center region where 45° lines meet without gaps or ridges.",
                    );
                    ui.add_space(4.0);

                    let current_em = config.gcode.extrusion_multiplier;
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Calibration Pass:");
                        for pass in EmCalibrationPass::ALL {
                            let is_sel = calib_state.em.pass == pass;
                            if ui
                                .selectable_label(is_sel, pass.short_label())
                                .on_hover_text(pass.label())
                                .clicked()
                                && !is_sel
                            {
                                calib_state.em.apply_pass_defaults(pass, current_em);
                            }
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut calib_state.em.coupon_width_mm)
                                .speed(0.5)
                                .range(14.0..=45.0)
                                .prefix("Coupon W: ")
                                .suffix(" mm"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.em.coupon_depth_mm)
                                .speed(0.5)
                                .range(14.0..=45.0)
                                .prefix("Coupon D: ")
                                .suffix(" mm"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.em.top_test_layers)
                                .speed(0.2)
                                .range(3..=8)
                                .prefix("Top Layers: "),
                        );
                    });

                    if calib_state.em.pass == EmCalibrationPass::Pass3HighFlowDieSwell {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.start_swell_high)
                                    .speed(0.01)
                                    .range(1.00..=1.50)
                                    .prefix("Start swell_high: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.end_swell_high)
                                    .speed(0.01)
                                    .range(1.02..=1.80)
                                    .prefix("End swell_high: "),
                            );
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.step_swell_high)
                                    .speed(0.005)
                                    .range(0.01..=0.20)
                                    .prefix("Step: "),
                            );
                        });
                        ui.small(format!(
                            "Pass 3 sweeps swell_high from {:.2} to {:.2} (step {:.2}) at high-flow Q_high = {:.1} mm³/s.",
                            calib_state.em.start_swell_high,
                            calib_state.em.end_swell_high,
                            calib_state.em.step_swell_high,
                            config.gcode.pa_ref_high_flow_mm3s,
                        ));
                    } else {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.start_delta_pct)
                                    .speed(0.5)
                                    .range(-25.0..=0.0)
                                    .prefix("Start Δ: ")
                                    .suffix("%"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.end_delta_pct)
                                    .speed(0.5)
                                    .range(0.0..=25.0)
                                    .prefix("End Δ: ")
                                    .suffix("%"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut calib_state.em.step_delta_pct)
                                    .speed(0.25)
                                    .range(0.25..=10.0)
                                    .prefix("Step: ")
                                    .suffix("%"),
                            );
                        });
                        ui.checkbox(
                            &mut calib_state.em.isolate_die_swell_in_em_passes,
                            "Neutralize Die-Swell Scaling in Pass 1 & 2 (Isolate Base EM)",
                        );
                    }

                    ui.add_space(6.0);
                    egui::Frame::group(ui.style())
                        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("Result Calculator & One-Click Apply").strong(),
                            );
                            if calib_state.em.pass == EmCalibrationPass::Pass3HighFlowDieSwell {
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(
                                            &mut calib_state.em.selected_winner_value,
                                        )
                                        .speed(0.01)
                                        .range(1.00..=1.80)
                                        .prefix("Best Coupon swell_high: "),
                                    );
                                    let new_swell_high = calib_state
                                        .em
                                        .selected_winner_value
                                        .clamp(config.gcode.swell_low, 2.0);
                                    ui.label(format!(
                                        "→ 1/swell_high = {:.3}×",
                                        1.0 / new_swell_high
                                    ));
                                    if ui.button("Apply swell_high").clicked() {
                                        config.gcode.swell_high = new_swell_high;
                                        config_modified = true;
                                        plate.status_message = format!(
                                            "Applied calibrated swell_high = {new_swell_high:.4} (1/swell = {:.3}×).",
                                            1.0 / new_swell_high
                                        );
                                    }
                                });
                            } else {
                                calib_state.em.base_extrusion_multiplier =
                                    config.gcode.extrusion_multiplier;
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(
                                            &mut calib_state.em.selected_winner_value,
                                        )
                                        .speed(0.25)
                                        .range(-20.0..=20.0)
                                        .prefix("Best Coupon Label Δ%: ")
                                        .suffix("%"),
                                    );
                                    let new_em = calib_state.em.compute_updated_em();
                                    ui.label(format!(
                                        "New EM = {:.4} × (1 {:+.2}%) = {:.4}×",
                                        config.gcode.extrusion_multiplier,
                                        calib_state.em.selected_winner_value,
                                        new_em
                                    ));
                                    if ui.button("Apply New EM").clicked() {
                                        config.gcode.extrusion_multiplier = new_em;
                                        config_modified = true;
                                        plate.status_message = format!(
                                            "Applied calibrated Extrusion Multiplier = {new_em:.4}×."
                                        );
                                    }
                                });
                            }
                        });
                }

                CalibrationUtilityKind::PressureAdvance => {
                    ui.label(
                        egui::RichText::new(
                            "Ellis-Style 90° Sharp-Corner Chevron Pattern (Klipper / RRF PA)",
                        )
                        .strong(),
                    );
                    ui.small(
                        "Prints slow→fast→slow 90° chevrons stepping Pressure Advance line-by-line, with vector-stroke PA value labels on the frame. Choose the line with the sharpest, most uniform corner.",
                    );
                    ui.add_space(4.0);

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Mode:");
                        for mode in PaCalibrationMode::ALL {
                            ui.selectable_value(
                                &mut calib_state.pa.mode,
                                mode,
                                mode.label(),
                            );
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.pa_start_s)
                                .speed(0.002)
                                .range(0.0..=0.50)
                                .prefix("Start PA: ")
                                .suffix(" s"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.pa_end_s)
                                .speed(0.002)
                                .range(0.005..=0.80)
                                .prefix("End PA: ")
                                .suffix(" s"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.pa_step_s)
                                .speed(0.0005)
                                .range(0.0005..=0.10)
                                .prefix("Step: ")
                                .suffix(" s"),
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.slow_speed_mms)
                                .speed(1.0)
                                .range(5.0..=80.0)
                                .prefix("Slow Speed: ")
                                .suffix(" mm/s"),
                        );
                        if calib_state.pa.mode == PaCalibrationMode::SingleBand {
                            ui.add(
                                egui::DragValue::new(&mut calib_state.pa.fast_speed_mms)
                                    .speed(2.0)
                                    .range(40.0..=500.0)
                                    .prefix("Fast Speed: ")
                                    .suffix(" mm/s"),
                            );
                        }
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.accel_mms2)
                                .speed(100.0)
                                .range(500.0..=30000.0)
                                .prefix("Accel: ")
                                .suffix(" mm/s²"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.pa.test_layer_count)
                                .speed(0.2)
                                .range(2..=6)
                                .prefix("Layers: "),
                        );
                    });
                    ui.checkbox(
                        &mut calib_state.pa.include_adhesion_ribbons,
                        "Print 1-Layer Adhesion Ribbons Under Chevron Vertices & Apexes",
                    );

                    if calib_state.pa.mode == PaCalibrationMode::DualBandShearThinning {
                        let cross_section = (config.slicing.outer_wall_width.target_mm
                            * config.slicing.layer_height.target_mm)
                            .max(0.04);
                        let v_low =
                            (config.gcode.pa_ref_low_flow_mm3s / cross_section).clamp(20.0, 300.0);
                        let v_high =
                            (config.gcode.pa_ref_high_flow_mm3s / cross_section).clamp(50.0, 350.0);
                        ui.small(format!(
                            "Dual-Band prints Left Bank at Q_low = {:.1} mm³/s ({v_low:.0} mm/s) for k_low and Right Bank at Q_high = {:.1} mm³/s ({v_high:.0} mm/s) for k_high.",
                            config.gcode.pa_ref_low_flow_mm3s,
                            config.gcode.pa_ref_high_flow_mm3s,
                        ));
                    }

                    ui.add_space(6.0);
                    egui::Frame::group(ui.style())
                        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("Apply Calibrated Pressure Advance").strong(),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.pa.winner_pa_low_s)
                                        .speed(0.001)
                                        .range(0.0..=0.50)
                                        .prefix("Best k_low: ")
                                        .suffix(" s"),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.pa.winner_pa_high_s)
                                        .speed(0.001)
                                        .range(0.0..=0.50)
                                        .prefix("Best k_high: ")
                                        .suffix(" s"),
                                );
                                if ui.button("Apply Dual-Band PA").clicked() {
                                    config.gcode.pa_low_flow_s = calib_state.pa.winner_pa_low_s;
                                    config.gcode.pa_high_flow_s = calib_state.pa.winner_pa_high_s;
                                    config.gcode.enable_dynamic_pa = true;
                                    config_modified = true;
                                    plate.status_message = format!(
                                        "Applied calibrated PA: k_low = {:.4} s, k_high = {:.4} s.",
                                        config.gcode.pa_low_flow_s, config.gcode.pa_high_flow_s
                                    );
                                }
                                if ui
                                    .button("Apply as Single PA")
                                    .on_hover_text("Set both k_low and k_high to Best k_low")
                                    .clicked()
                                {
                                    config.gcode.pa_low_flow_s = calib_state.pa.winner_pa_low_s;
                                    config.gcode.pa_high_flow_s = calib_state.pa.winner_pa_low_s;
                                    config_modified = true;
                                    plate.status_message = format!(
                                        "Applied single-value PA = {:.4} s to both k_low and k_high.",
                                        config.gcode.pa_low_flow_s
                                    );
                                }
                            });
                        });
                }

                CalibrationUtilityKind::TransientPressure => {
                    ui.label(
                        egui::RichText::new(
                            "Constant-Speed Flow-Step & Staccato Striation Pattern (γ & τ_melt)",
                        )
                        .strong(),
                    );
                    ui.small(
                        "Maintains strictly constant toolhead speed (Δv = 0) across an abrupt 3.5:1 cross-section step-down at the X = 0 center tick mark. Firmware PA is 100% inactive, isolating viscoelastic melt-chamber relaxation.",
                    );
                    ui.add_space(4.0);

                    ui.horizontal_wrapped(|ui| {
                        ui.label("Calibration Pass:");
                        for pass in TransientCalibrationPass::ALL {
                            ui.selectable_value(
                                &mut calib_state.transient.pass,
                                pass,
                                pass.label(),
                            );
                        }
                    });

                    match calib_state.transient.pass {
                        TransientCalibrationPass::Pass1ExponentGamma => {
                            calib_state.transient.base_tau_s =
                                config.gcode.transient_pressure_tau_s;
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.start_gamma)
                                        .speed(0.01)
                                        .range(0.0..=0.80)
                                        .prefix("Start γ: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.end_gamma)
                                        .speed(0.01)
                                        .range(0.05..=1.00)
                                        .prefix("End γ: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.step_gamma)
                                        .speed(0.005)
                                        .range(0.01..=0.25)
                                        .prefix("Step γ: "),
                                );
                            });
                            ui.small(format!(
                                "Pass 1 sweeps exponent γ at fixed τ_melt = {:.2} s. Pick the rib where the post-step bulge right after the X = 0 center tick disappears without a necking gap.",
                                config.gcode.transient_pressure_tau_s
                            ));
                        }
                        TransientCalibrationPass::Pass2RelaxationTau => {
                            calib_state.transient.base_exponent_gamma =
                                config.gcode.transient_pressure_exponent;
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.start_tau_s)
                                        .speed(0.01)
                                        .range(0.02..=2.0)
                                        .prefix("Start τ_melt: ")
                                        .suffix(" s"),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.end_tau_s)
                                        .speed(0.01)
                                        .range(0.05..=5.0)
                                        .prefix("End τ_melt: ")
                                        .suffix(" s"),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.step_tau_s)
                                        .speed(0.01)
                                        .range(0.01..=1.0)
                                        .prefix("Step τ: ")
                                        .suffix(" s"),
                                );
                            });
                            ui.small(format!(
                                "Pass 2 sweeps relaxation time τ_melt at fixed γ = {:.2}. Pick the rib where width stays uniform across the entire decay tail (5–25 mm past X = 0) and across the staccato pulses.",
                                config.gcode.transient_pressure_exponent
                            ));
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::DragValue::new(
                                &mut calib_state.transient.constant_test_speed_mms,
                            )
                            .speed(1.0)
                            .range(20.0..=200.0)
                            .prefix("Const Speed (Δv=0): ")
                            .suffix(" mm/s"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.transient.min_flow_multiplier)
                                .speed(0.01)
                                .range(0.40..=1.0)
                                .prefix("Min Mult: ")
                                .suffix("×"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut calib_state.transient.rib_layer_count)
                                .speed(0.2)
                                .range(3..=8)
                                .prefix("Rib Layers: "),
                        );
                    });

                    ui.add_space(6.0);
                    egui::Frame::group(ui.style())
                        .inner_margin(egui::Margin::symmetric(8.0, 6.0))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Apply Calibrated Transient Melt-Zone Parameters",
                                )
                                .strong(),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::DragValue::new(
                                        &mut calib_state.transient.base_exponent_gamma,
                                    )
                                    .speed(0.01)
                                    .range(0.05..=1.0)
                                    .prefix("Best γ: "),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut calib_state.transient.base_tau_s)
                                        .speed(0.01)
                                        .range(0.01..=5.0)
                                        .prefix("Best τ_melt: ")
                                        .suffix(" s"),
                                );
                                if ui.button("Apply γ & τ_melt").clicked() {
                                    config.gcode.enable_transient_pressure_comp = true;
                                    config.gcode.transient_pressure_exponent =
                                        calib_state.transient.base_exponent_gamma.clamp(0.05, 1.0);
                                    config.gcode.transient_pressure_tau_s =
                                        calib_state.transient.base_tau_s.clamp(0.01, 5.0);
                                    config_modified = true;
                                    plate.status_message = format!(
                                        "Applied calibrated Transient Pressure Comp: γ = {:.3}, τ_melt = {:.3} s.",
                                        config.gcode.transient_pressure_exponent,
                                        config.gcode.transient_pressure_tau_s,
                                    );
                                }
                            });
                        });
                }
            }

            if config_modified {
                config.gcode.sanitize();
                vis_state.dirty_planned_gcode = true;
            }

            ui.add_space(6.0);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(
                        !is_computing,
                        egui::Button::new(
                            egui::RichText::new("Preview in 3D Viewport").strong(),
                        ),
                    )
                    .on_hover_text(
                        "Generate this calibration pattern and inspect its 3D toolpaths colored by Flow Multiplier, PA, or Melt-Zone Pressure",
                    )
                    .clicked()
                {
                    request_preview = true;
                }

                if ui
                    .add_enabled(
                        !is_computing && !is_io_busy,
                        egui::Button::new("Export Calibration G-code..."),
                    )
                    .on_hover_text("Generate and save standalone Klipper/Moonraker G-code for this calibration utility")
                    .clicked()
                {
                    request_export = true;
                }

                if config.network.is_configured() {
                    let proto_label = config.network.protocol.short_label();
                    let is_uploading = moonraker_state.pending_upload_task.is_some();
                    let can_upload = !is_computing && !is_io_busy && !is_uploading;
                    let upload_btn_label = if is_uploading && !moonraker_state.upload_start_print {
                        "Uploading..."
                    } else {
                        "Upload to Printer"
                    };
                    if ui
                        .add_enabled(can_upload, egui::Button::new(upload_btn_label))
                        .on_hover_text(format!(
                            "Generate and upload calibration G-code to {proto_label} ({}) without starting a print",
                            config.network.printer_uri.trim()
                        ))
                        .clicked()
                    {
                        request_upload = Some(false);
                    }
                    let upload_print_label = if is_uploading && moonraker_state.upload_start_print {
                        "Uploading & Printing..."
                    } else {
                        "Upload & Print"
                    };
                    if ui
                        .add_enabled(
                            can_upload,
                            egui::Button::new(
                                egui::RichText::new(upload_print_label)
                                    .strong()
                                    .color(egui::Color32::from_rgb(130, 230, 165)),
                            ),
                        )
                        .on_hover_text(format!(
                            "Generate, upload calibration G-code to {proto_label} ({}), and immediately start printing",
                            config.network.printer_uri.trim()
                        ))
                        .clicked()
                    {
                        request_upload = Some(true);
                    }
                }
            });

            if !calib_state.last_summary_message.is_empty() {
                ui.add_space(4.0);
                ui.small(
                    egui::RichText::new(&calib_state.last_summary_message)
                        .color(egui::Color32::from_rgb(175, 225, 190)),
                );
            }

            if request_preview || request_export || request_upload.is_some() {
                calib_state.em.base_extrusion_multiplier = config.gcode.extrusion_multiplier;
                calib_state.em.tool_index = calib_state.selected_tool_index;
                calib_state.pa.tool_index = calib_state.selected_tool_index;
                calib_state.transient.tool_index = calib_state.selected_tool_index;
                let tool_id_tag = config
                    .tools
                    .get(resolved_tool_idx)
                    .map(|t| t.id)
                    .unwrap_or(0);
                let (res, stem, preferred_color_mode) = match calib_state.active_kind {
                    CalibrationUtilityKind::ExtrusionMultiplier => {
                        let pass_tag = match calib_state.em.pass {
                            EmCalibrationPass::Pass1Coarse => "em_pass1_coarse",
                            EmCalibrationPass::Pass2Fine => "em_pass2_fine",
                            EmCalibrationPass::Pass3HighFlowDieSwell => "em_pass3_dieswell",
                        };
                        (
                            generate_em_calibration_gcode(&calib_state.em, config),
                            format!("calib_t{tool_id_tag}_{pass_tag}"),
                            ToolpathColorMode::ByExtrusionMultiplier,
                        )
                    }
                    CalibrationUtilityKind::PressureAdvance => {
                        let mode_tag = match calib_state.pa.mode {
                            PaCalibrationMode::DualBandShearThinning => "pa_dualband",
                            PaCalibrationMode::SingleBand => "pa_singleband",
                        };
                        (
                            generate_pa_calibration_gcode(&calib_state.pa, config),
                            format!("calib_t{tool_id_tag}_{mode_tag}"),
                            ToolpathColorMode::ByPressureAdvance,
                        )
                    }
                    CalibrationUtilityKind::TransientPressure => {
                        let pass_tag = match calib_state.transient.pass {
                            TransientCalibrationPass::Pass1ExponentGamma => "transient_pass1_gamma",
                            TransientCalibrationPass::Pass2RelaxationTau => "transient_pass2_tau",
                        };
                        (
                            generate_transient_pressure_calibration_gcode(
                                &calib_state.transient,
                                config,
                            ),
                            format!("calib_t{tool_id_tag}_{pass_tag}"),
                            ToolpathColorMode::ByExtrusionMultiplier,
                        )
                    }
                };

                let total_sec = res.total_time_s.round() as u64;
                let mins = total_sec / 60;
                let secs = total_sec % 60;
                calib_state.last_generated_filename_stem = stem.clone();
                calib_state.last_summary_message = format!(
                    "Generated '{}' — {} layers, {} extruding moves | Est. {:02}m {:02}s | {:.2} m ({:.1} g)",
                    stem,
                    res.layers.len(),
                    res.extruding_move_count,
                    mins,
                    secs,
                    res.total_filament_mm / 1000.0,
                    res.total_filament_grams,
                );

                // Always update the 3D viewport preview so the user can inspect the generated calibration pattern
                let max_layer_idx = res.layers.len().saturating_sub(1);
                vis_state.selected_layer_min_idx = 0;
                vis_state.selected_layer_max_idx = max_layer_idx;
                vis_state.selected_layer_stride = 1;
                vis_state.show_toolpaths = true;
                vis_state.toolpath_color_mode = preferred_color_mode;
                vis_state.cached_planned_gcode = Some(res.clone());
                vis_state.dirty_planned_gcode = false;
                vis_state.dirty_toolpaths_mesh = true;
                plate.status_message = calib_state.last_summary_message.clone();

                if request_export {
                    let default_filename = format!("{stem}.gcode");
                    let gcode_bytes = res.gcode_text;
                    let total_moves = res.extruding_move_count + res.travel_move_count;
                    plate.status_message =
                        "Choosing destination for calibration G-code export...".to_string();

                    let save_task = IoTaskPool::get().spawn(async move {
                        let file_handle = rfd::AsyncFileDialog::new()
                            .add_filter("G-code (.gcode, .nc)", &["gcode", "nc"])
                            .set_file_name(&default_filename)
                            .save_file()
                            .await?;
                        let out_path = file_handle.path().to_path_buf();
                        let kb = gcode_bytes.len() as f64 / 1024.0;
                        let write_res = std::fs::write(&out_path, gcode_bytes)
                            .map(|()| {
                                format!(
                                    "{} ({:.1} KB, {} moves)",
                                    out_path.display(),
                                    kb,
                                    total_moves
                                )
                            })
                            .map_err(|e| e.to_string());
                        Some(write_res)
                    });
                    plate.pending_save_task = Some(save_task);
                } else if let Some(start_print) = request_upload {
                    spawn_network_gcode_upload(
                        moonraker_state,
                        config.network.clone(),
                        format!("{stem}.gcode"),
                        res.gcode_text,
                        start_print,
                        &mut plate.status_message,
                    );
                }
            }
        });
    calib_state.show_window = open;
}

/// Renders a compact or full-width per-object `ObjectFilamentMode` `ComboBox` with a live colour swatch.
/// Returns `true` if `filament_mode` was modified.
fn render_object_filament_mode_combo(
    ui: &mut egui::Ui,
    object_id: u64,
    filament_mode: &mut ObjectFilamentMode,
    custom_color_srgba: Option<[f32; 4]>,
    config: &SlicerConfig,
    compact: bool,
) -> bool {
    let mut changed = false;

    let swatch_rgba = match *filament_mode {
        ObjectFilamentMode::SingleTool(t) => config
            .tools
            .get(t)
            .or_else(|| config.tools.first())
            .map(|tool| tool.filament_color_srgba)
            .unwrap_or([0.24, 0.62, 0.92, 1.0]),
        ObjectFilamentMode::FilamentSlot(s) => config.color_srgba_for_filament_slot(s),
        ObjectFilamentMode::OpticalFrontLit | ObjectFilamentMode::OpticalBackLitLithophane => {
            custom_color_srgba.unwrap_or([0.18, 0.82, 0.96, 1.0])
        }
        _ => custom_color_srgba.unwrap_or_else(|| {
            config
                .active_tool()
                .map(|t| t.filament_color_srgba)
                .unwrap_or([0.24, 0.62, 0.92, 1.0])
        }),
    };
    let swatch_col32 = egui::Color32::from_rgb(
        (swatch_rgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (swatch_rgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (swatch_rgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    );
    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 3.0, swatch_col32);
    ui.painter().rect_stroke(
        rect,
        3.0,
        egui::Stroke::new(1.0_f32, egui::Color32::from_gray(90)),
    );

    let selected_text = if compact {
        filament_mode.short_label(config)
    } else {
        filament_mode.full_label(config)
    };
    let combo_id = if compact {
        format!("obj_filament_mode_compact_{object_id}")
    } else {
        format!("obj_filament_mode_full_{object_id}")
    };

    let combo_w = if compact {
        92.0
    } else {
        (ui.available_width() - 95.0).clamp(115.0, 240.0)
    };
    egui::ComboBox::from_id_source(combo_id)
        .selected_text(selected_text)
        .width(combo_w)
        .show_ui(ui, |ui| {
            let def_mode = ObjectFilamentMode::ProjectDefault;
            if ui
                .selectable_label(*filament_mode == def_mode, def_mode.full_label(config))
                .on_hover_text(
                    "Follow the project's default active tool, imported 3MF/OBJ slots, and global colour mixing setting.",
                )
                .clicked()
            {
                *filament_mode = def_mode;
                changed = true;
            }

            ui.separator();
            ui.small(egui::RichText::new("Real Machine Tools (T0..Tn):").strong());
            for (t_idx, tool) in config.tools.iter().enumerate() {
                let mode = ObjectFilamentMode::SingleTool(t_idx);
                let is_sel = *filament_mode == mode;
                let tc = egui::Color32::from_rgb(
                    (tool.filament_color_srgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (tool.filament_color_srgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                    (tool.filament_color_srgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                );
                let lbl = format!(
                    "■ T{} — {} ({}, {})",
                    tool.id, tool.name, tool.filament_type, tool.filament_color_hex
                );
                if ui
                    .selectable_label(is_sel, egui::RichText::new(lbl).color(tc))
                    .on_hover_text(
                        "Print this entire object with this specific machine tool, bypassing optical colour mixing.",
                    )
                    .clicked()
                {
                    *filament_mode = mode;
                    changed = true;
                }
            }

            if !config.filament_slots.is_empty() {
                ui.separator();
                ui.small(
                    egui::RichText::new("Filament Slots & Virtual Ratio Mixes:").strong(),
                );
                for (s_idx, slot) in config.filament_slots.iter().enumerate() {
                    if s_idx > 239 {
                        break;
                    }
                    let mode = ObjectFilamentMode::FilamentSlot(s_idx as u8);
                    let is_sel = *filament_mode == mode;
                    let sc = egui::Color32::from_rgb(
                        (slot.color_srgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (slot.color_srgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
                        (slot.color_srgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
                    );
                    let lbl = format!("■ {}", mode.full_label(config));
                    if ui
                        .selectable_label(is_sel, egui::RichText::new(lbl).color(sc))
                        .on_hover_text(
                            "Print this object using this filament slot (including any Virtual Filament integer-ratio dither mix).",
                        )
                        .clicked()
                    {
                        *filament_mode = mode;
                        changed = true;
                    }
                }
            }

            ui.separator();
            ui.small(
                egui::RichText::new("Per-Object Optical Colour Mixing (CMYKW / N-Tool):").strong(),
            );
            for opt_mode in [
                ObjectFilamentMode::OpticalFrontLit,
                ObjectFilamentMode::OpticalBackLitLithophane,
                ObjectFilamentMode::DirectSlotsNoOptical,
            ] {
                let is_sel = *filament_mode == opt_mode;
                if ui
                    .selectable_label(is_sel, opt_mode.full_label(config))
                    .clicked()
                {
                    *filament_mode = opt_mode;
                    changed = true;
                }
            }
        });

    changed
}

/// Renders the per-object `Filament / Colour Mode` selector, optional target colour picker,
/// and quick CMYKW / Virtual Ratio actions inside the `Selected: {obj_name}` inspector.
fn render_selected_object_filament_and_color_controls(
    ui: &mut egui::Ui,
    obj: &mut crate::scene::SceneObject,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    _meshes: &mut Assets<Mesh>,
) {
    let mut mode_or_color_changed = false;

    ui.horizontal_wrapped(|ui| {
        ui.label("Filament / Mode:");
        if render_object_filament_mode_combo(
            ui,
            obj.object_id,
            &mut obj.filament_mode,
            obj.custom_color_srgba,
            config,
            false,
        ) {
            mode_or_color_changed = true;
        }
    });

    let is_optical_override = matches!(
        obj.filament_mode,
        ObjectFilamentMode::OpticalFrontLit | ObjectFilamentMode::OpticalBackLitLithophane
    );
    if is_optical_override && config.tools.len() < 2 {
        ui.horizontal_wrapped(|ui| {
            ui.small(
                egui::RichText::new("Optical CMYKW mixing requires >= 2 tools.")
                    .color(egui::Color32::from_rgb(245, 190, 90)),
            );
            if ui
                .small_button("Load 5-Tool PLA CMYKW Palette")
                .on_hover_text(
                    "Auto-configure T0..T4 with Cyan, Magenta, Yellow, Black, and White PLA filaments.",
                )
                .clicked()
            {
                config.apply_optical_palette_to_tools(
                    &crate::color_optics::bambu_pla_cmykw_palette(),
                );
                sync_config_to_backend(config);
                mode_or_color_changed = true;
            }
        });
    }

    let is_unpainted =
        obj.triangle_filament_slots.is_empty() && obj.triangle_colors_rgba.is_empty();
    let show_target_color_row = is_unpainted
        || is_optical_override
        || obj.custom_color_srgba.is_some();

    if show_target_color_row {
        ui.horizontal_wrapped(|ui| {
            let mut has_custom = obj.custom_color_srgba.is_some();
            if ui
                .checkbox(&mut has_custom, "Object Target Colour:")
                .on_hover_text(
                    "Assign a target sRGB colour to this object for per-object CMYKW optical mixing, Virtual Ratio auto-fitting, or 3D viewport tinting.",
                )
                .changed()
            {
                if has_custom {
                    let default_col = config
                        .active_tool()
                        .map(|t| t.filament_color_srgba)
                        .unwrap_or([0.92, 0.36, 0.18, 1.0]);
                    obj.custom_color_srgba = Some(default_col);
                } else {
                    obj.custom_color_srgba = None;
                }
                mode_or_color_changed = true;
            }

            if let Some(ref mut col_srgba) = obj.custom_color_srgba {
                let mut rgb = [col_srgba[0], col_srgba[1], col_srgba[2]];
                if ui
                    .color_edit_button_rgb(&mut rgb)
                    .on_hover_text("Pick target sRGB colour for this object")
                    .changed()
                {
                    col_srgba[0] = rgb[0];
                    col_srgba[1] = rgb[1];
                    col_srgba[2] = rgb[2];
                    col_srgba[3] = 1.0;
                    mode_or_color_changed = true;
                }
                let hex = crate::config::ThreeMfFilamentSlot::format_hex_color(*col_srgba);
                ui.monospace(&hex);

                if config.tools.len() >= 2
                    && ui
                        .small_button("⇄ Auto-Fit Virtual Ratio")
                        .on_hover_text(
                            "Create a new Virtual Filament slot dithered in the best integer ratio of 2–3 real tools to match this colour, and assign this object to it.",
                        )
                        .clicked()
                {
                    let target_col = *col_srgba;
                    let comps = crate::color_optics::find_best_integer_ratio_for_target_srgba(
                        target_col,
                        &config.tools,
                        10,
                    );
                    let next_idx = config.filament_slots.len().min(239);
                    let mut new_slot = crate::config::ThreeMfFilamentSlot::default_for_slot(
                        next_idx,
                        0,
                    );
                    new_slot.dither_components = comps;
                    new_slot.sync_color_from_dither_components(&config.tools);
                    let ratio_str = new_slot.format_ratio_summary(&config.tools);
                    new_slot.name = format!(
                        "Virtual #{} ({} -> {})",
                        next_idx + 1,
                        ratio_str,
                        new_slot.color_hex
                    );
                    config.filament_slots.push(new_slot);
                    obj.filament_mode = ObjectFilamentMode::FilamentSlot(next_idx as u8);
                    vis_state.toolpath_color_mode = ToolpathColorMode::ByToolOrFilament;
                    mode_or_color_changed = true;
                }
            }
        });
    }

    if mode_or_color_changed {
        vis_state.dirty_color_mixing = true;
        vis_state.dirty_surface_colors = true;
        vis_state.dirty_planned_gcode = true;
        vis_state.dirty_toolpaths_mesh = true;
    }
}
