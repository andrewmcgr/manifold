use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, IoTaskPool};
use bevy_egui::{egui, EguiContexts};

use crate::bridge::ffi;
use crate::config::{
    lookup_parameter_doc, BedShape, ClearancePointXZ, ClearanceProfileMode, PointNormalHint,
    SlicerConfig, SolidInfillPattern, SparseInfillPattern, SurfaceHintMode, ToolConfig,
    WallPrintOrder,
};
use crate::field_vis::{
    high_contrast_parameter_ramp, sample_slice_colormap_rgba, FieldColormap,
    FieldVisualizationState, IsoIntersectionMode, ObjectSurfaceColorMode,
    SelectedLayersColorMode, SliceAxis, ToolpathColorMode, BRANCH_PALETTE,
};
use crate::hull::rotation_to_place_face_on_bed;
use crate::io::threemf::{save_project_to_3mf, ExportableObject};
use crate::io::load_files_into_backend;
use crate::presets::{LeftPanelTab, PresetCategory, PresetLibrary};
use crate::scene::{
    clone_object_on_plate, compute_orbit_eye, remove_object_from_plate, run_auto_arrange_on_plate,
    snap_object_to_bed, spawn_loaded_objects, BuildPlateInfo, BuildPlateState, CadObjectEntity,
    OrbitCamera, ToolMode,
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

            let col_w = ((ui.available_width() - 8.0) * 0.5).max(115.0);
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
                    let _ = draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_travel_moves,
                        [0.28, 0.86, 0.98, 1.0],
                        "Travel",
                        None,
                    );
                    ui.end_row();

                    let _ = draw_line_type_toggle_cell(
                        ui,
                        &mut vis_state.show_z_hops,
                        [0.98, 0.92, 0.22, 1.0],
                        "Z-Hop",
                        None,
                    );
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
            ToolpathColorMode::ByFeatureType => unreachable!(),
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
    if tabs.is_empty() {
        return None;
    }

    let count = tabs.len();
    let gap = 3.0_f32;
    let total_width = ui.available_width().max(120.0);
    let total_gaps = gap * (count.saturating_sub(1) as f32);
    let tab_width = ((total_width - total_gaps) / (count as f32)).max(28.0);

    let (bar_rect, _) =
        ui.allocate_exact_size(egui::vec2(total_width, height), egui::Sense::hover());

    let mut clicked_tab = None;
    let baseline_y = bar_rect.max.y - 1.0;
    let baseline_stroke = egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(86, 96, 116));

    // Pre-collect interaction responses before borrowing the painter
    let mut tab_rects_and_states = Vec::with_capacity(count);
    for idx in 0..count {
        let min_x = bar_rect.min.x + (idx as f32) * (tab_width + gap);
        let max_x = if idx + 1 == count {
            bar_rect.max.x
        } else {
            (min_x + tab_width).min(bar_rect.max.x)
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

    let painter = ui.painter();

    // Draw continuous bottom baseline across the entire tab strip first
    painter.line_segment(
        [
            egui::pos2(bar_rect.min.x, baseline_y),
            egui::pos2(bar_rect.max.x, baseline_y),
        ],
        baseline_stroke,
    );

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

        // Label + optional modified amber dot
        let font_id = if is_active {
            egui::FontId::proportional(13.5)
        } else {
            egui::FontId::proportional(12.8)
        };
        let text_color = if is_active {
            egui::Color32::WHITE
        } else if hovered {
            egui::Color32::from_rgb(232, 238, 248)
        } else {
            egui::Color32::from_rgb(172, 180, 198)
        };
        let galley = painter.layout_no_wrap(label.to_string(), font_id, text_color);
        let dot_space = if is_modified { 10.0 } else { 0.0 };
        let content_w = galley.size().x + dot_space;
        let text_pos = egui::pos2(
            (tab_rect.center().x - content_w * 0.5).max(tab_rect.min.x + 5.0),
            tab_rect.center().y - galley.size().y * 0.5 + 1.0,
        );
        painter.galley(text_pos, galley.clone(), text_color);

        if is_modified {
            let dot_center = egui::pos2(
                (text_pos.x + galley.size().x + 6.0).min(tab_rect.max.x - 5.0),
                tab_rect.center().y + 1.0,
            );
            painter.circle_filled(dot_center, 3.2, egui::Color32::from_rgb(255, 192, 56));
        }
    }

    clicked_tab
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
                let mut pick_name: Option<String> = None;
                egui::ComboBox::from_id_source(("preset_selector_combo", category.label()))
                    .selected_text(combo_label)
                    .width(210.0)
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

    // Pre-fetch selected object info for the workflow bar and panels
    let selected_obj_info: Option<(u64, usize, String, Quat)> =
        plate.selected_object_id.and_then(|sel_id| {
            let idx = plate.objects.iter().position(|o| o.object_id == sel_id)?;
            let name = plate.objects[idx].name.clone();
            let ent = plate.objects[idx].entity;
            let rot = object_transforms
                .get(ent)
                .map(|(t, _)| t.rotation)
                .unwrap_or(Quat::IDENTITY);
            Some((sel_id, idx, name, rot))
        });

    // ========================================================================
    // 1. Top Toolbar: Project I/O, Viewport Editing Tools & Workflow Pipeline
    // ========================================================================
    egui::TopBottomPanel::top("top_editing_toolbar").show(ctx, |ui| {
        // Row 1: File I/O, Viewport Editing Mode, Scene Actions, and Right Panel Toggle
        ui.horizontal_wrapped(|ui| {
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
                    "Select one or more STEP, STL, or 3MF files...".to_string();
                let task = IoTaskPool::get().spawn(async move {
                    let handles = rfd::AsyncFileDialog::new()
                        .add_filter(
                            "Supported 3D Files (.step, .stp, .stl, .3mf)",
                            &["step", "stp", "STEP", "STP", "stl", "STL", "3mf", "3MF"],
                        )
                        .add_filter("STEP CAD Assemblies", &["step", "stp", "STEP", "STP"])
                        .add_filter("STL Meshes", &["stl", "STL"])
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
                        plate_index: obj.plate_index,
                        transform: tf,
                        vertices: obj.vertices.clone(),
                        indices: obj.indices.clone(),
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
            .on_hover_text("Export planned non-planar G-code (.gcode) for the selected object")
            .clicked()
            {
                if let (Some((_, _, ref obj_name, _)), Some(ref planned)) =
                    (&selected_obj_info, &vis_state.cached_planned_gcode)
                {
                    let stem = std::path::Path::new(obj_name)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("manifold2_print");
                    let default_filename = format!("{stem}.gcode");
                    let gcode_bytes = planned.gcode_text.clone();
                    let total_moves =
                        planned.extruding_move_count + planned.travel_move_count;
                    plate.status_message =
                        "Choosing destination for G-code export...".to_string();

                    let save_task = IoTaskPool::get().spawn(async move {
                        let file_handle = rfd::AsyncFileDialog::new()
                            .add_filter("G-code (.gcode, .nc)", &["gcode", "nc"])
                            .set_file_name(&default_filename)
                            .save_file()
                            .await?;
                        let out_path = file_handle.path().to_path_buf();
                        let kb = gcode_bytes.len() as f64 / 1024.0;
                        let res = std::fs::write(&out_path, gcode_bytes)
                            .map(|()| {
                                format!(
                                    "{} ({:.1} KB, {} moves)",
                                    out_path.display(),
                                    kb,
                                    total_moves
                                )
                            })
                            .map_err(|e| e.to_string());
                        Some(res)
                    });
                    plate.pending_save_task = Some(save_task);
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
                let active_name = plate.active_plate_name().to_string();
                let active_cnt = plate.active_plate_object_count();
                plate.status_message =
                    format!("Switched to {active_name} ({active_cnt} object(s)).");
            }

            ui.separator();
            if render_icon_button(ui, UiIcon::AutoArrange, "Auto-Arrange", false, true).clicked() {
                run_auto_arrange_on_plate(&mut plate, &config, &mut object_transforms);
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
                &mut preset_lib.show_view_panel,
                egui::RichText::new("View & Inspection Panel").strong(),
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
                    if has_eikonal {
                        let need_toolpaths = vis_state.show_toolpaths;
                        vis_state.spawn_slicing_task(sel_id, &config, true, need_toolpaths);
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
                    format!("3. Generate Toolpaths & G-code ({} paths)", tp.total_path_count)
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
                    if has_eikonal {
                        let need_layers = vis_state.cached_selected_layers.is_none()
                            || vis_state.dirty_selected_layers;
                        vis_state.spawn_slicing_task(sel_id, &config, need_layers, true);
                        plate.status_message = format!(
                            "Generating non-planar toolpaths & G-code for '{obj_name}' in background..."
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

                if is_computing {
                    let prog = ffi::get_field_compute_progress();
                    ui.ctx().request_repaint();
                    ui.add(
                        egui::ProgressBar::new(prog.fraction.clamp(0.0, 1.0))
                            .desired_width(180.0)
                            .show_percentage()
                            .animate(true)
                            .text(prog.stage),
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
        });
    });

    // ========================================================================
    // 2. Left Panel: Project Setup vs. Configuration & Saved Presets
    // ========================================================================
    egui::SidePanel::left("slicer_control_panel")
        .default_width(390.0)
        .resizable(true)
        .show(ctx, |ui| {
            // Top-level Left Panel Mode Switcher (Folder-style tab bar)
            let any_preset_dirty = PresetCategory::ALL
                .iter()
                .any(|&cat| preset_lib.is_category_modified(cat, &config));
            let setup_label = format!("Project Setup ({})", plate.objects.len());
            let top_tabs = [
                (setup_label.as_str(), false),
                ("Configuration & Presets", any_preset_dirty),
            ];
            let active_top_idx = match preset_lib.left_tab {
                LeftPanelTab::ProjectSetup => 0,
                LeftPanelTab::Configuration => 1,
            };
            if let Some(clicked_idx) = render_folder_tab_bar(
                ui,
                "left_panel_primary_tabs",
                &top_tabs,
                active_top_idx,
                30.0,
                egui::Color32::from_rgb(76, 166, 255),
            ) {
                preset_lib.left_tab = if clicked_idx == 0 {
                    LeftPanelTab::ProjectSetup
                } else {
                    LeftPanelTab::Configuration
                };
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
                    if !is_computing {
                        sync_config_to_backend(&config);
                    }
                    vis_state.dirty_selected_layers = true;
                    vis_state.dirty_toolpaths = true;
                    vis_state.dirty_planned_gcode = true;
                }
                ui.add_space(6.0);
            }

            egui::ScrollArea::vertical().show(ui, |ui| {
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
                                ui.horizontal(|ui| {
                                    ui.small("Plate Name:");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut active_plate_info.name)
                                            .desired_width(180.0),
                                    );
                                });
                            }
                        });

                    ui.add_space(4.0);
                    let active_plate_idx = plate.active_plate_index;
                    let active_plate_name = plate.active_plate_name().to_string();
                    let active_obj_count = plate.active_plate_object_count();

                    ui.horizontal(|ui| {
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
                                remove_object_from_plate(&mut commands, &mut plate, id);
                            }
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
                                remove_object_from_plate(&mut commands, &mut plate, id);
                            }
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
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut obj.visible, "").changed() {
                                    if let Ok((_, mut vis)) = object_transforms.get_mut(obj.entity)
                                    {
                                        *vis = if obj.visible {
                                            Visibility::Inherited
                                        } else {
                                            Visibility::Hidden
                                        };
                                    }
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
                        }
                        if let Some(id) = remove_id {
                            remove_object_from_plate(&mut commands, &mut plate, id);
                        }
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

                            // Move object between build plates
                            let mut move_to_plate: Option<usize> = None;
                            let mut move_to_new_plate = false;
                            ui.horizontal(|ui| {
                                ui.label("Assigned Plate:");
                                let cur_p_name = plate
                                    .plates
                                    .get(obj_plate_idx)
                                    .map(|p| p.name.clone())
                                    .unwrap_or_else(|| format!("Plate {}", obj_plate_idx + 1));
                                egui::ComboBox::from_id_source("selected_object_plate_combo")
                                    .selected_text(cur_p_name)
                                    .width(140.0)
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
                                let target_name = plate.active_plate_name().to_string();
                                plate.status_message =
                                    format!("Moved '{obj_name}' to {target_name}.");
                            }

                            if plate.clone_copies_count == 0 {
                                plate.clone_copies_count = 1;
                            }
                            let mut pending_clone_copies: Option<usize> = None;
                            ui.horizontal(|ui| {
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
                            }

                            if let Ok((mut tf, _)) = object_transforms.get_mut(obj_entity) {
                                let (w_min, w_max) =
                                    plate.objects[obj_idx].compute_world_aabb(&tf);
                                let extents = w_max - w_min;
                                ui.small(format!(
                                    "World Size: {:.2} × {:.2} × {:.2} mm",
                                    extents.x, extents.y, extents.z
                                ));

                                let gcode_offset_xy = config.scene_to_gcode_offset().xy();
                                let mut bed_x = tf.translation.x + gcode_offset_xy.x;
                                let mut bed_y = tf.translation.y + gcode_offset_xy.y;
                                let mut changed = false;
                                ui.horizontal(|ui| {
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

                                ui.horizontal(|ui| {
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
                                    }

                                    if surface_hint_count == 0 {
                                        ui.small(
                                            "Click faces in the viewport using 'Surface Hint: Normal ⊥ Surface' (gold) or 'Normal ∥ Surface' (cyan).",
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
                                            let row_resp = ui.horizontal(|ui| {
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
                                                };
                                                if render_icon_button(
                                                    ui,
                                                    hint_icon,
                                                    sh.mode.short_label(),
                                                    true,
                                                    true,
                                                )
                                                .on_hover_text("Click to toggle between Normal ⊥ Surface and Normal ∥ Surface")
                                                .clicked()
                                                {
                                                    sh.mode = match sh.mode {
                                                        SurfaceHintMode::NormalOrthogonalToSurface => {
                                                            SurfaceHintMode::NormalTangentToSurface
                                                        }
                                                        SurfaceHintMode::NormalTangentToSurface => {
                                                            SurfaceHintMode::NormalOrthogonalToSurface
                                                        }
                                                    };
                                                }
                                                ui.add(
                                                    egui::DragValue::new(&mut sh.influence_depth_mm)
                                                        .speed(0.2)
                                                        .range(0.5..=50.0)
                                                        .prefix("d="),
                                                );
                                                ui.add(
                                                    egui::DragValue::new(&mut sh.weight)
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
                                        }
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
                                                ui.horizontal(|ui| {
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

                                                ui.horizontal(|ui| {
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
                                                ui.horizontal(|ui| {
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
                            render_printer_config_tab(ui, &mut config, &mut vis_state, is_computing);
                        }
                        PresetCategory::Tool => {
                            render_tool_config_tab(ui, &mut config, &mut vis_state, is_computing);
                        }
                        PresetCategory::Filament => {
                            render_filament_config_tab(ui, &mut config, &mut vis_state, is_computing);
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
        });

    // ========================================================================
    // 3. Right Panel: View & Inspection (Never Marks Presets Dirty)
    // ========================================================================
    if preset_lib.show_view_panel {
        egui::SidePanel::right("view_and_inspection_panel")
            .default_width(340.0)
            .min_width(285.0)
            .max_width(400.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("View & Inspection");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("✕")
                            .on_hover_text("Hide View & Inspection panel")
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

                egui::ScrollArea::vertical().show(ui, |ui| {
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
            });
    }

    // Top-right viewport HUD legend overlay when the View & Inspection panel is closed
    let show_hud_toolpaths = vis_state.show_toolpaths
        && (vis_state.cached_toolpaths.is_some() || vis_state.cached_planned_gcode.is_some());
    let show_hud_layers =
        vis_state.show_selected_layers && vis_state.cached_selected_layers.is_some();
    if !preset_lib.show_view_panel
        && plate.selected_object().is_some()
        && (show_hud_toolpaths || show_hud_layers)
    {
        egui::Window::new("Viewport Color Legend")
            .id(egui::Id::new("viewport_parameter_color_legend_hud"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-14.0, 14.0))
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
            });
    }
}

/// Renders the `Printer` configuration sub-tab:
/// Print Bed Geometry & Origin, 5/6-Axis Machine Kinematics & Bed Tilt (`NozzleFrame` & `Gravity Vector`),
/// Motion Limits & Stepper Dynamics, Klipper/Moonraker Output, and Start/End G-code Templates.
fn render_printer_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    is_computing: bool,
) {
    let mut dirty_backend = false;
    let mut dirty_gcode = false;

    egui::CollapsingHeader::new(egui::RichText::new("Print Bed Geometry & Origin").strong())
        .id_source("printer_bed_geometry_section")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
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

            ui.horizontal(|ui| {
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
        egui::RichText::new("Klipper, Moonraker & G-code Templates").strong(),
    )
    .id_source("printer_gcode_templates_section")
    .default_open(false)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.emit_exclude_object,
                    "Emit Klipper EXCLUDE_OBJECT commands",
                )
                .changed();
            param_help_button(ui, "gcode.emit_exclude_object");
        });
        ui.horizontal(|ui| {
            dirty_gcode |= ui
                .checkbox(
                    &mut gc.emit_thumbnails,
                    "Embed Moonraker PNG Thumbnails (32x32 & 300x300)",
                )
                .changed();
            param_help_button(ui, "gcode.emit_thumbnails");
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.small(
                "Placeholders: {tool_id}, {nozzle_temp}, {bed_temp}, {nozzle_diameter}, {bbox_min_x..z}, {bbox_max_x..z}, {bbox_center_x..y}, {layer_count}",
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

    if dirty_backend && !is_computing {
        sync_config_to_backend(config);
    }
    if dirty_gcode {
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
                    format!("T{} ({:.2}mm)", tool.id, tool.nozzle_diameter_mm),
                )
                .clicked()
                {
                    clicked_idx = Some(idx);
                }
            }
            if let Some(idx) = clicked_idx {
                config.active_tool_index = idx;
                dirty_backend = true;
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
                dirty_backend = true;
            }
            if config.tools.len() > 1 && ui.small_button("Remove Tool").clicked() {
                let rm_idx = config.active_tool_index;
                config.tools.remove(rm_idx);
                if config.active_tool_index >= config.tools.len() {
                    config.active_tool_index = config.tools.len() - 1;
                }
                dirty_backend = true;
            }
        });

        let active_idx = config.active_tool_index;
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
    egui::CollapsingHeader::new(
        egui::RichText::new("Extruder Retraction, Wipe & Z-Hop").strong(),
    )
    .id_source("tool_retraction_wipe_zhop_section")
    .default_open(true)
    .show(ui, |ui| {
        let gc = &mut config.gcode;
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        config.gcode.sanitize();
        vis_state.dirty_planned_gcode = true;
    }
}

/// Renders the `Filament` configuration sub-tab:
/// Material & Temperatures, Flow Multipliers, Part Cooling Fan Schedule,
/// and Non-Newtonian Rheology (Dynamic PA, Die Swell, Corner Flow Comp, Transient Pressure, Ooze Relaxation).
fn render_filament_config_tab(
    ui: &mut egui::Ui,
    config: &mut SlicerConfig,
    vis_state: &mut FieldVisualizationState,
    is_computing: bool,
) {
    let mut dirty_backend = false;
    let mut gc_changed = false;

    egui::CollapsingHeader::new(
        egui::RichText::new("Material, Diameter & Temperatures").strong(),
    )
    .id_source("filament_material_temp_section")
    .default_open(true)
    .show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label("Material:");
            gc_changed |= ui
                .add(
                    egui::TextEdit::singleline(&mut config.gcode.filament_type)
                        .desired_width(64.0),
                )
                .changed();
            for (preset_name, density, fan_pct) in [
                ("PLA", 1.24f32, 100.0f32),
                ("PETG", 1.27, 40.0),
                ("ABS", 1.04, 15.0),
                ("ASA", 1.07, 15.0),
                ("TPU", 1.21, 50.0),
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
                        "Set material to {preset_name} (density {density:.2} g/cm³, fan {fan_pct:.0}%)"
                    ))
                    .clicked()
                {
                    config.gcode.filament_type = preset_name.to_string();
                    config.gcode.filament_density_g_cm3 = density;
                    config.gcode.fan_speed_percent = fan_pct;
                    gc_changed = true;
                }
            }
            param_help_button(ui, "gcode.filament_type");
        });

        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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

        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_dynamic_pa,
                    "Emit Dynamic M572 / SET_PRESSURE_ADVANCE",
                )
                .changed();
            param_help_button(ui, "gcode.enable_dynamic_pa");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_slicer_pa_subdivision,
                    "Slicer-Side Trapezoidal PA Subdivision",
                )
                .changed();
            param_help_button(ui, "gcode.enable_slicer_pa_subdivision");
        });
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_corner_flow_comp,
                    "Klipper SCV Corner Flow Compensation",
                )
                .changed();
            param_help_button(ui, "gcode.corner_flow_comp");
        });
        if gc.enable_corner_flow_comp {
            ui.horizontal(|ui| {
                gc_changed |= ui
                    .add(
                        egui::Slider::new(&mut gc.corner_flow_comp_ratio, 0.0..=1.0)
                            .text("Corner Deduction Ratio"),
                    )
                    .changed();
                param_help_button(ui, "gcode.corner_flow_comp");
            });
        }
        ui.horizontal(|ui| {
            gc_changed |= ui
                .checkbox(
                    &mut gc.enable_transient_pressure_comp,
                    "Melt-Zone Transient Pressure Compensation",
                )
                .changed();
            param_help_button(ui, "gcode.transient_pressure_comp");
        });
        if gc.enable_transient_pressure_comp {
            ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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

        if lh_changed {
            config.slicing.layer_height.sanitize();
            if config.eikonal.auto_voxel_size {
                config.sync_auto_voxel_size();
            }
            if !is_computing {
                sync_config_to_backend(config);
            }
            vis_state.dirty_selected_layers = true;
            vis_state.dirty_toolpaths = true;
        }
    });

    // 2. Volumetric Eikonal Field Solver & Overhang Surface Bias
    ui.add_space(4.0);
    egui::CollapsingHeader::new(
        egui::RichText::new("Volumetric Eikonal Solver & Overhang Bias").strong(),
    )
    .id_source("process_eikonal_solver_section")
    .default_open(true)
    .show(ui, |ui| {
        let mut eik_changed = false;
        ui.horizontal(|ui| {
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

        ui.horizontal(|ui| {
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
            eik_changed |= ui
                .add(
                    egui::Slider::new(&mut config.eikonal.solver_iterations, 1..=12).text("Iters"),
                )
                .changed();
            param_help_button(ui, "eikonal.solver_iterations");
        });

        ui.horizontal(|ui| {
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
            eik_changed |= ui
                .add(
                    egui::Slider::new(&mut config.eikonal.curvature_regularization, 0.0..=1.0)
                        .text("Curvature Reg."),
                )
                .changed();
            param_help_button(ui, "eikonal.curvature_regularization");
        });

        ui.separator();
        ui.horizontal(|ui| {
            eik_changed |= ui
                .checkbox(
                    &mut config.eikonal.enable_overhang_orientation_bias,
                    "Overhang Conformal / Orthogonal Surface Bias",
                )
                .changed();
            param_help_button(ui, "eikonal.enable_overhang_orientation_bias");
        });
        if config.eikonal.enable_overhang_orientation_bias {
            ui.horizontal(|ui| {
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
            ui.horizontal(|ui| {
                ui.small("Orthogonal Band:");
                eik_changed |= ui
                    .add(
                        egui::DragValue::new(&mut config.eikonal.orthogonal_supported_angle_deg)
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

        ui.separator();
        ui.small("Default Soft Hint Placement Parameters:");
        ui.horizontal(|ui| {
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

        ui.horizontal(|ui| {
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
            ui.horizontal(|ui| {
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

        if tp_cfg_changed {
            config.slicing.sanitize();
            if !is_computing {
                sync_config_to_backend(config);
            }
            vis_state.dirty_toolpaths = true;
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
            ui.horizontal(|ui| {
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
            ui.horizontal(|ui| {
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
                            .range(0..=10)
                            .prefix("Bed Intf Layers: "),
                    )
                    .on_hover_text(
                        "Number of dense support interface layers printed on the build plate at the base of support columns for strong bed adhesion.",
                    )
                    .changed();
                param_help_button(ui, "slicing.support_bed_interface_layers");
            });
            ui.horizontal(|ui| {
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
                            .speed(0.02)
                            .range(0.3..=4.0)
                            .prefix("Intf Spacing: ")
                            .suffix(" mm"),
                    )
                    .changed();
                param_help_button(ui, "slicing.support_density");
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
        ui.small(egui::RichText::new("Feature Print Speeds (mm/s):").strong());
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.horizontal(|ui| {
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
        ui.small("Select an object on the build plate to inspect its OpenVDB fields, non-planar layers, and 3D toolpaths.");
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
            ui.horizontal(|ui| {
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

        if let Some(ref res) = vis_state.cached_selected_layers.clone() {
            let total_layers = res.layers.len();
            if total_layers > 0 {
                let max_idx = total_layers - 1;
                let mut range_changed = false;
                let min_changed = ui
                    .add(
                        egui::Slider::new(&mut vis_state.selected_layer_min_idx, 0..=max_idx)
                            .text("Start Layer"),
                    )
                    .changed();
                let max_changed = ui
                    .add(
                        egui::Slider::new(&mut vis_state.selected_layer_max_idx, 0..=max_idx)
                            .text("End Layer"),
                    )
                    .changed();
                if min_changed
                    && vis_state.selected_layer_min_idx > vis_state.selected_layer_max_idx
                {
                    vis_state.selected_layer_max_idx = vis_state.selected_layer_min_idx;
                } else if max_changed
                    && vis_state.selected_layer_max_idx < vis_state.selected_layer_min_idx
                {
                    vis_state.selected_layer_min_idx = vis_state.selected_layer_max_idx;
                }
                range_changed |= min_changed || max_changed;
                range_changed |= ui
                    .add(
                        egui::Slider::new(&mut vis_state.selected_layer_stride, 1..=10)
                            .text("Display Stride"),
                    )
                    .changed();
                if range_changed {
                    vis_state.dirty_selected_layers_mesh = true;
                    vis_state.dirty_toolpaths_mesh = true;
                    vis_state.dirty_slice = true;
                }

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
            ui.small(format!(
                "Moves: {} extruding, {} travels, {} retracts",
                planned.extruding_move_count, planned.travel_move_count, planned.retract_count,
            ));
        }
    });
}

/// Synchronizes the Rust `SlicerConfig` (nozzle axis, bed-coordinate gravity vector,
/// multi-tool specifications & XZ clearance profiles, variable layer height, and
/// Eikonal solver iterations) into the C++ compute backend.
pub fn sync_config_to_backend(config: &SlicerConfig) {
    ffi::set_nozzle_axis_bed(config.machine.nozzle_axis_bed);
    ffi::set_gravity_direction_bed(config.machine.gravity_direction_bed);

    let ffi_tools: Vec<ffi::ToolSpec> = config
        .tools
        .iter()
        .map(ToolConfig::to_ffi_tool_spec)
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
    });

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
    });
}

/// Compiles all active surface-face hints and 3D point hints for `object_id` into
/// backend `ffi::SpatialOrientationHint` records (in object-local OpenVDB coordinates)
/// and pushes them to the C++ Eikonal solver.
pub fn sync_object_hints_to_backend(
    config: &SlicerConfig,
    plate: &BuildPlateState,
    object_id: u64,
) {
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
        });
    }

    ffi::set_spatial_orientation_hints(&ffi_hints);
}

/// Spawns an asynchronous background task to compute OpenVDB SDF, constrained Eikonal,
/// overhang, and 5/6-axis tool orientation fields without blocking the UI thread.
fn spawn_field_compute_task(
    vis_state: &mut FieldVisualizationState,
    plate: &mut BuildPlateState,
    config: &SlicerConfig,
    object_id: u64,
    object_name: &str,
    obj_rotation: Quat,
) {
    sync_config_to_backend(config);
    let inv_rot = obj_rotation.inverse();
    let local_nozzle = (inv_rot * Vec3::from_array(config.machine.nozzle_axis_bed))
        .normalize_or_zero();
    if local_nozzle.length_squared() > 1.0e-6 {
        ffi::set_nozzle_axis_bed(local_nozzle.to_array());
    }
    let local_gravity = (inv_rot * Vec3::from_array(config.machine.gravity_direction_bed))
        .normalize_or_zero();
    if local_gravity.length_squared() > 1.0e-6 {
        ffi::set_gravity_direction_bed(local_gravity.to_array());
    }
    sync_object_hints_to_backend(config, plate, object_id);
    let voxel_size = config.eikonal.voxel_size_mm;
    let narrow_band = config.eikonal.narrow_band_width_voxels as f32;
    let task = IoTaskPool::get().spawn(async move {
        ffi::recompute_object_fields(object_id, voxel_size, narrow_band)
    });
    vis_state.last_solved_field_rotation = Some(obj_rotation);
    vis_state.pending_field_task = Some((object_id, object_name.to_string(), task));
    plate.status_message = format!(
        "Computing OpenVDB scalar fields for '{object_name}' in background (voxel {voxel_size:.2} mm)..."
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
                        vis_state.spawn_slicing_task(
                            obj_id,
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
        if let Some((layers_opt, toolpaths_opt)) = block_on(future::poll_once(task)) {
            vis_state.pending_slicing_task = None;
            if plate.selected_object_id == Some(obj_id) {
                let num_layers = layers_opt
                    .as_ref()
                    .map(|l| l.layers.len())
                    .or_else(|| vis_state.cached_selected_layers.as_ref().map(|l| l.layers.len()))
                    .unwrap_or(0);
                let num_paths = toolpaths_opt.as_ref().map(|t| t.total_path_count);
                vis_state.apply_slicing_task_result(layers_opt, toolpaths_opt);
                if let Some(paths) = num_paths {
                    plate.status_message = format!(
                        "Generated {paths} non-planar shell toolpaths across {num_layers} layers."
                    );
                } else {
                    plate.status_message = format!(
                        "Selected {num_layers} non-planar slicing isosurfaces."
                    );
                }
            }
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
