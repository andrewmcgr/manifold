use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, IoTaskPool};
use bevy_egui::{egui, EguiContexts};

use crate::bridge::ffi;
use crate::config::{
    BedShape, ClearancePointXZ, PointNormalHint, SlicerConfig, SolidInfillPattern,
    SparseInfillPattern, SurfaceHintMode, ToolConfig,
};
use crate::field_vis::{
    FieldColormap, FieldVisualizationState, IsoIntersectionMode, SelectedLayersColorMode,
    SliceAxis, ToolpathColorMode,
};
use crate::hull::rotation_to_place_face_on_bed;
use crate::io::threemf::{save_project_to_3mf, ExportableObject};
use crate::io::load_files_into_backend;
use crate::scene::{
    compute_orbit_eye, remove_object_from_plate, run_auto_arrange_on_plate, snap_object_to_bed,
    spawn_loaded_objects, BuildPlateState, CadObjectEntity, OrbitCamera, ToolMode,
};

/// Renders the top editing mode toolbar and the left control/object/configuration side panel.
#[allow(clippy::too_many_arguments)]
pub fn ui_workspace_system(
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
    mut vis_state: ResMut<FieldVisualizationState>,
    mut object_transforms: Query<(&mut Transform, &mut Visibility), With<CadObjectEntity>>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), Without<CadObjectEntity>>,
) {
    plate.ui_hovered_surface_hint_patch = None;
    plate.ui_hovered_point_hint_id = None;
    vis_state.highlighted_selected_layer = None;
    let ctx = contexts.ctx_mut();
    vis_state.ui_pointer_down = ctx.is_using_pointer();

    // ========================================================================
    // 1. Top Editing Mode & Action Toolbar
    // ========================================================================
    egui::TopBottomPanel::top("top_editing_toolbar").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Editing Mode:").strong());
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
                if ui.selectable_label(selected, mode.label()).clicked() {
                    plate.tool_mode = mode;
                }
            }

            ui.separator();

            if ui.button("Auto-Arrange (Spiral)").clicked() {
                run_auto_arrange_on_plate(&mut plate, &config, &mut object_transforms);
            }

            if ui.button("Frame All").clicked() {
                frame_camera_to_plate(&plate, &config, &mut camera_query);
            }
        });
    });

    // ========================================================================
    // 2. Left Inspector, File I/O, Object Tree & Bed Configuration Panel
    // ========================================================================
    egui::SidePanel::left("slicer_control_panel")
        .default_width(360.0)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Manifold2 Non-Planar Slicer");
                ui.separator();

                // --- File Import & 3MF Project Export ---
                ui.label(egui::RichText::new("Project & CAD I/O").strong());
                let is_busy =
                    plate.pending_load_task.is_some() || plate.pending_save_task.is_some();

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!is_busy, egui::Button::new("Import CAD / Mesh / 3MF..."))
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

                    if ui
                        .add_enabled(
                            !is_busy && !plate.objects.is_empty(),
                            egui::Button::new("Save 3MF..."),
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
                                transform: tf,
                                vertices: obj.vertices.clone(),
                                indices: obj.indices.clone(),
                            });
                        }
                        let cfg_snapshot = (*config).clone();
                        plate.status_message = "Choosing destination for 3MF export...".to_string();

                        let save_task = IoTaskPool::get().spawn(async move {
                            let file_handle = rfd::AsyncFileDialog::new()
                                .add_filter("3MF Project Archive", &["3mf"])
                                .set_file_name("project.3mf")
                                .save_file()
                                .await?;
                            let out_path = file_handle.path().to_path_buf();
                            let res = save_project_to_3mf(&out_path, &export_list, &cfg_snapshot)
                                .map(|()| out_path.to_string_lossy().into_owned());
                            Some(res)
                        });
                        plate.pending_save_task = Some(save_task);
                    }
                });

                ui.add_space(8.0);
                ui.separator();

                // --- Build Plate Objects & Decomposed Assemblies ---
                ui.horizontal(|ui| {
                    ui.heading(format!("Objects ({})", plate.objects.len()));
                    if !plate.objects.is_empty() && ui.small_button("Clear All").clicked() {
                        let all_ids: Vec<u64> =
                            plate.objects.iter().map(|o| o.object_id).collect();
                        for id in all_ids {
                            remove_object_from_plate(&mut commands, &mut plate, id);
                        }
                    }
                });

                if plate.objects.is_empty() {
                    ui.small("No objects loaded. Import a STEP assembly, STL, or 3MF file.");
                } else {
                    let mut remove_id: Option<u64> = None;
                    let mut select_id: Option<u64> = None;
                    let current_selected_id = plate.selected_object_id;

                    for obj in &mut plate.objects {
                        let is_selected = current_selected_id == Some(obj.object_id);
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut obj.visible, "").changed() {
                                if let Ok((_, mut vis)) = object_transforms.get_mut(obj.entity) {
                                    *vis = if obj.visible {
                                        Visibility::Inherited
                                    } else {
                                        Visibility::Hidden
                                    };
                                }
                            }

                            let tag = if obj.has_brep { "[STEP]" } else { "[Mesh]" };
                            let label = format!("{} {} ({} tris)", tag, obj.name, obj.indices.len() / 3);
                            if ui.selectable_label(is_selected, label).clicked() {
                                select_id = Some(obj.object_id);
                            }

                            if ui.small_button("Remove").clicked() {
                                remove_id = Some(obj.object_id);
                            }
                        });
                    }

                    if let Some(id) = select_id {
                        plate.selected_object_id = Some(id);
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
                        let hull_face_count = plate.objects[obj_idx].hull.faces.len();

                        ui.heading(format!("Selected: {obj_name}"));

                        let mut obj_rotation = Quat::IDENTITY;
                        if let Ok((mut tf, _)) = object_transforms.get_mut(obj_entity) {
                            let (w_min, w_max) =
                                plate.objects[obj_idx].compute_world_aabb(&tf);
                            let extents = w_max - w_min;
                            ui.small(format!(
                                "World Size: {:.2} × {:.2} × {:.2} mm",
                                extents.x, extents.y, extents.z
                            ));

                            let mut changed = false;
                            ui.horizontal(|ui| {
                                ui.label("Bed X (mm):");
                                changed |= ui
                                    .add(egui::DragValue::new(&mut tf.translation.x).speed(0.5))
                                    .changed();
                                ui.label("Y (mm):");
                                changed |= ui
                                    .add(egui::DragValue::new(&mut tf.translation.y).speed(0.5))
                                    .changed();
                            });

                            ui.horizontal(|ui| {
                                ui.label("Quick Rotate:");
                                if ui.button("X +90°").clicked() {
                                    tf.rotation =
                                        (Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)
                                            * tf.rotation)
                                            .normalize();
                                    changed = true;
                                }
                                if ui.button("Y +90°").clicked() {
                                    tf.rotation =
                                        (Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
                                            * tf.rotation)
                                            .normalize();
                                    changed = true;
                                }
                                if ui.button("Z +45°").clicked() {
                                    tf.rotation =
                                        (Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)
                                            * tf.rotation)
                                            .normalize();
                                    changed = true;
                                }
                                if ui.button("Reset").clicked() {
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
                                        if ui.small_button(label).clicked() {
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
                            }
                            obj_rotation = tf.rotation;
                        }

                        // --- Tangency & Orthogonality Hints (Soft Constraints) ---
                        ui.add_space(8.0);
                        ui.separator();
                        let patch_count = plate.objects[obj_idx].surface_patches.len();
                        let obj_has_brep = plate.objects[obj_idx].has_brep;
                        let obj_local_center =
                            0.5 * (plate.objects[obj_idx].local_min + plate.objects[obj_idx].local_max);
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
                                "Tangency & Orthogonality Hints ({} surface, {} point)",
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
                                    if ui.selectable_label(active, mode.label()).clicked() {
                                        plate.tool_mode =
                                            if active { ToolMode::Select } else { mode };
                                    }
                                }
                            });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(
                                        &mut config.eikonal.default_surface_hint_depth_mm,
                                        1.0..=30.0,
                                    )
                                    .text("Face Depth (mm)"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut config.eikonal.default_point_hint_radius_mm,
                                        1.5..=40.0,
                                    )
                                    .text("Point Radius (mm)"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(
                                    &mut config.eikonal.default_hint_weight,
                                    0.05..=1.0,
                                )
                                .text("Default Soft Weight"),
                            );

                            // 1. Surface Face Hints List (Collapsible)
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
                                    && ui.small_button("Clear Surface Hints").clicked()
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
                                        let is_sel =
                                            plate.selected_surface_hint_patch == Some(sh.patch_id);
                                        let row_resp = ui.horizontal(|ui| {
                                            let kind_str =
                                                if obj_has_brep { "STEP Face" } else { "Patch" };
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
                                            if ui
                                                .small_button(sh.mode.short_label())
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
                                            if ui.small_button("✕").clicked() {
                                                remove_surf_idx = Some(idx);
                                            }
                                        });
                                        if row_resp.response.hovered()
                                            || ui.rect_contains_pointer(row_resp.response.rect)
                                        {
                                            plate.ui_hovered_surface_hint_patch = Some(sh.patch_id);
                                        }
                                    }
                                    if let Some(idx) = remove_surf_idx {
                                        let removed_patch =
                                            config.eikonal.surface_hints[idx].patch_id;
                                        config.eikonal.surface_hints.remove(idx);
                                        if plate.selected_surface_hint_patch == Some(removed_patch)
                                        {
                                            plate.selected_surface_hint_patch = None;
                                        }
                                    }
                                }
                            });

                            // 2. 3D Point Hints in Space (Collapsible)
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
                                    if ui.small_button("+ Add at Center").clicked() {
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
                                        && ui.small_button("Clear Point Hints").clicked()
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
                                                if ui.small_button("✕").clicked() {
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

                                            // Spherical Azimuth / Elevation controls for normal direction
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

                        // --- Generic Volumetric Field Visualization (OpenVDB) ---
                        ui.add_space(8.0);
                        ui.separator();
                        egui::CollapsingHeader::new(
                            egui::RichText::new("Volumetric Field Visualization (OpenVDB)").strong(),
                        )
                        .default_open(true)
                        .show(ui, |ui| {
                            let is_computing = vis_state.is_busy();

                            // Field Selector
                            ui.horizontal(|ui| {
                                ui.label("Scalar Field:");
                                let avail = vis_state.available_fields.clone();
                                if avail.is_empty() {
                                    ui.small("(not computed yet)");
                                } else {
                                    for fname in &avail {
                                        let selected = &vis_state.active_field_name == fname;
                                        if ui.selectable_label(selected, fname).clicked() && !selected {
                                            vis_state.active_field_name = fname.clone();
                                            vis_state.mark_all_dirty();
                                            if !is_computing {
                                                vis_state.refresh_stats_for(sel_id);
                                            }
                                        }
                                    }
                                }
                            });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(&mut config.eikonal.voxel_size_mm, 0.15..=3.0)
                                        .text("Voxel (mm)"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut config.eikonal.solver_iterations,
                                        1..=12,
                                    )
                                    .text("Iters"),
                                );
                                let btn_label = if vis_state.available_fields.is_empty() {
                                    "Compute Fields"
                                } else {
                                    "Recompute Fields"
                                };
                                if ui
                                    .add_enabled(!is_computing, egui::Button::new(btn_label).small())
                                    .clicked()
                                {
                                    spawn_field_compute_task(
                                        &mut vis_state,
                                        &mut plate,
                                        &config,
                                        sel_id,
                                        &obj_name,
                                        obj_rotation,
                                    );
                                }
                            });

                            if is_computing {
                                let prog = ffi::get_field_compute_progress();
                                ui.ctx().request_repaint();
                                ui.add(
                                    egui::ProgressBar::new(prog.fraction.clamp(0.0, 1.0))
                                        .show_percentage()
                                        .animate(true)
                                        .text(prog.stage),
                                );
                            }

                            let (min_val, max_val) = if let Some(ref stats) = vis_state.cached_stats
                            {
                                ui.small(format!(
                                    "Active Voxels: {} | Voxel Size: {:.2} mm | Range: [{:.2}, {:.2}]",
                                    stats.active_voxel_count,
                                    stats.voxel_size,
                                    stats.min_value,
                                    stats.max_value
                                ));
                                (stats.min_value, stats.max_value)
                            } else {
                                (-10.0, 10.0)
                            };

                            ui.add(
                                egui::Slider::new(&mut vis_state.solid_opacity, 0.0..=1.0)
                                    .text("CAD Mesh Opacity"),
                            );

                            ui.separator();

                            // 1. 3D Isosurface controls (volumeToMesh)
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
                                        &mut vis_state,
                                        &mut plate,
                                        &config,
                                        sel_id,
                                        &obj_name,
                                        obj_rotation,
                                    );
                                }
                            }

                            if vis_state.show_isosurface {
                                let lo = min_val.min(max_val - 0.01);
                                let hi = max_val.max(lo + 0.01);
                                if ui
                                    .add(
                                        egui::Slider::new(&mut vis_state.isovalue, lo..=hi)
                                            .text("Isovalue"),
                                    )
                                    .changed()
                                {
                                    vis_state.dirty_isosurface = true;
                                    if vis_state.active_field_name != "sdf" {
                                        vis_state.dirty_slice = true;
                                    }
                                }
                                if ui
                                    .add(
                                        egui::Slider::new(&mut vis_state.adaptivity, 0.0..=0.5)
                                            .text("Adaptivity"),
                                    )
                                    .changed()
                                {
                                    vis_state.dirty_isosurface = true;
                                }

                                if vis_state.active_field_name != "sdf" {
                                    egui::ComboBox::from_label("Object Intersection")
                                        .selected_text(vis_state.intersection_mode.label())
                                        .show_ui(ui, |ui| {
                                            for mode in [
                                                IsoIntersectionMode::HighlightInsideDimOutside,
                                                IsoIntersectionMode::ClipToObjectInterior,
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
                                        "Inside Object (φ <= 0): {} tris | Exterior (φ > 0): {} tris | Contour: {} segs",
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

                            // 2. 2D Cross-Section Slice Plane controls (GridSampler)
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
                                        &mut vis_state,
                                        &mut plate,
                                        &config,
                                        sel_id,
                                        &obj_name,
                                        obj_rotation,
                                    );
                                }
                            }

                            if vis_state.show_slice_plane {
                                ui.horizontal(|ui| {
                                    ui.label("Plane:");
                                    for axis in [SliceAxis::X, SliceAxis::Y, SliceAxis::Z] {
                                        if ui
                                            .selectable_label(
                                                vis_state.slice_axis == axis,
                                                axis.label(),
                                            )
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
                                        egui::Slider::new(
                                            &mut vis_state.slice_normalized_pos,
                                            0.0..=1.0,
                                        )
                                        .text(format!(
                                            "Slice Offset ({plane_coord:.1} mm)"
                                        )),
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
                                                .selectable_label(
                                                    vis_state.colormap == cmap,
                                                    cmap.label(),
                                                )
                                                .clicked()
                                            {
                                                vis_state.colormap = cmap;
                                                vis_state.dirty_slice = true;
                                            }
                                        }
                                    });

                                ui.horizontal(|ui| {
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_iso_contours,
                                            "Iso-Contour Lines",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_slice = true;
                                    }
                                    if ui
                                        .checkbox(
                                            &mut vis_state.fade_exterior_background,
                                            "Fade Background",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_slice = true;
                                    }
                                });
                            }
                        });

                        // --- Selected Slicing Isosurfaces (Bidirectional Branch-Aware) ---
                        ui.add_space(8.0);
                        ui.separator();
                        egui::CollapsingHeader::new(
                            egui::RichText::new("Selected Slicing Isosurfaces").strong(),
                        )
                        .id_source("selected_slicing_isosurfaces_section")
                        .default_open(true)
                        .show(ui, |ui| {
                            let is_computing = vis_state.is_busy();
                            let has_eikonal = vis_state
                                .available_fields
                                .iter()
                                .any(|f| f == "eikonal");

                            let mut lh_changed = false;
                            lh_changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut config.slicing.layer_height.target_mm,
                                        0.04..=1.20,
                                    )
                                    .text("Target Thickness (mm)"),
                                )
                                .changed();
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
                                lh_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut config.slicing.layer_height.max_mm)
                                            .speed(0.01)
                                            .range(0.05..=1.50)
                                            .prefix("Max: ")
                                            .suffix(" mm"),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                lh_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.layer_height.first_layer_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.02..=1.50)
                                        .prefix("First Layer: ")
                                        .suffix(" mm"),
                                    )
                                    .changed();
                                lh_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.layer_height.top_layer_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.02..=1.50)
                                        .prefix("Top Layer: ")
                                        .suffix(" mm"),
                                    )
                                    .changed();
                            });

                            if lh_changed {
                                config.slicing.layer_height.sanitize();
                                if !is_computing {
                                    sync_config_to_backend(&config);
                                }
                                vis_state.dirty_selected_layers = true;
                                vis_state.dirty_toolpaths = true;
                            }

                            ui.horizontal(|ui| {
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
                                        vis_state.solid_opacity = 0.22;
                                    }
                                    if vis_state.show_selected_layers
                                        && !has_eikonal
                                        && !is_computing
                                    {
                                        spawn_field_compute_task(
                                            &mut vis_state,
                                            &mut plate,
                                            &config,
                                            sel_id,
                                            &obj_name,
                                            obj_rotation,
                                        );
                                    }
                                }

                                if ui
                                    .add_enabled(
                                        !is_computing,
                                        egui::Button::new("Select Isosurfaces").small(),
                                    )
                                    .clicked()
                                {
                                    vis_state.show_selected_layers = true;
                                    if vis_state.solid_opacity > 0.45 {
                                        vis_state.solid_opacity = 0.22;
                                    }
                                    if has_eikonal {
                                        let need_toolpaths = vis_state.show_toolpaths;
                                        vis_state.spawn_slicing_task(
                                            sel_id,
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
                                            &obj_name,
                                            obj_rotation,
                                        );
                                    }
                                }
                            });

                            if vis_state.show_selected_layers {
                                ui.horizontal(|ui| {
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_selected_layer_surfaces,
                                            "3D Surfaces",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_selected_layers_mesh = true;
                                    }
                                    ui.checkbox(
                                        &mut vis_state.show_selected_layer_contours,
                                        "3D Perimeter Contours",
                                    );
                                });

                                egui::ComboBox::from_id_source("selected_layers_color_mode")
                                    .selected_text(vis_state.selected_layers_color_mode.label())
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
                            }

                            if let Some(ref res) = vis_state.cached_selected_layers.clone() {
                                let total_layers = res.layers.len();
                                if total_layers > 0 {
                                    let max_idx = total_layers - 1;
                                    let mut range_changed = false;
                                    ui.horizontal(|ui| {
                                        range_changed |= ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut vis_state.selected_layer_min_idx,
                                                    0..=max_idx,
                                                )
                                                .text("Start Layer"),
                                            )
                                            .changed();
                                        range_changed |= ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut vis_state.selected_layer_max_idx,
                                                    0..=max_idx,
                                                )
                                                .text("End Layer"),
                                            )
                                            .changed();
                                    });
                                    if vis_state.selected_layer_min_idx
                                        > vis_state.selected_layer_max_idx
                                    {
                                        vis_state.selected_layer_max_idx =
                                            vis_state.selected_layer_min_idx;
                                    }
                                    range_changed |= ui
                                        .add(
                                            egui::Slider::new(
                                                &mut vis_state.selected_layer_stride,
                                                1..=10,
                                            )
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
                                        "Selected: {} layers across {} branch(es) | Avg Δh: {:.3} mm (range [{:.3}, {:.3}] mm)",
                                        total_layers,
                                        res.branch_count,
                                        avg_h,
                                        global_min_h,
                                        global_max_h
                                    ));

                                    egui::CollapsingHeader::new(format!(
                                        "Selected Isosurface List ({total_layers})"
                                    ))
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
                                                    let tag = match (
                                                        layer.is_first_layer,
                                                        layer.is_top_layer,
                                                    ) {
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
                                                        vis_state.highlighted_selected_layer
                                                            == Some(idx),
                                                        label,
                                                    );
                                                    if resp.hovered() {
                                                        vis_state.highlighted_selected_layer =
                                                            Some(idx);
                                                    }
                                                    if resp.clicked() {
                                                        vis_state.active_field_name =
                                                            "eikonal".to_string();
                                                        vis_state.isovalue = layer.isovalue;
                                                        vis_state.show_isosurface = true;
                                                        vis_state.dirty_isosurface = true;
                                                        vis_state.dirty_slice = true;
                                                    }
                                                }
                                            });
                                    });
                                }
                            } else if !has_eikonal {
                                ui.small(
                                    "Click 'Select Isosurfaces' to compute the Eikonal field and select non-planar layers.",
                                );
                            }
                        });

                        // --- Non-Planar Shell & Toolpath Generation ---
                        ui.add_space(8.0);
                        ui.separator();
                        egui::CollapsingHeader::new(
                            egui::RichText::new("Shell & Toolpath Generation").strong(),
                        )
                        .id_source("shell_and_toolpath_generation_section")
                        .default_open(true)
                        .show(ui, |ui| {
                            let is_computing = vis_state.is_busy();
                            let has_eikonal = vis_state
                                .available_fields
                                .iter()
                                .any(|f| f == "eikonal");
                            let active_nozzle_mm = config
                                .active_tool()
                                .map(|t| t.nozzle_diameter_mm)
                                .unwrap_or(0.40);

                            let mut tp_cfg_changed = false;
                            tp_cfg_changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut config.slicing.shell_thickness_mm,
                                        0.20..=8.00,
                                    )
                                    .text("Shell Thickness (-φ_sdf <= T_shell mm)"),
                                )
                                .changed();

                            ui.horizontal(|ui| {
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(&mut config.slicing.inner_wall_count)
                                            .speed(0.1)
                                            .range(0..=12)
                                            .prefix("Inner Walls: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.min_extrusion_length_mm,
                                        )
                                        .speed(0.02)
                                        .range(0.0..=10.0)
                                        .prefix("Min Len: ")
                                        .suffix(" mm"),
                                    )
                                    .changed();
                                if ui.small_button("2× Nozzle").clicked() {
                                    config.slicing.min_extrusion_length_mm =
                                        2.0 * active_nozzle_mm;
                                    tp_cfg_changed = true;
                                }
                            });

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

                            tp_cfg_changed |= ui
                                .add(
                                    egui::Slider::new(
                                        &mut config.slicing.infill_density_percent,
                                        0.0..=100.0,
                                    )
                                    .suffix("%")
                                    .text("Sparse Density"),
                                )
                                .changed();

                            tp_cfg_changed |= ui
                                .add(
                                    egui::Slider::new(&mut config.slicing.infill_angle_deg, -180.0..=180.0)
                                        .suffix("°")
                                        .text("Infill Angle"),
                                )
                                .changed();

                            // Line Widths: Outer Wall, Inner Wall, Solid Infill, Sparse Infill (Target, Min, Max)
                            ui.small(egui::RichText::new("Line Widths (Target / Min / Max mm):").strong());
                            ui.horizontal(|ui| {
                                ui.label("Outer Wall:");
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.outer_wall_width.target_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.10..=2.50)
                                        .prefix("T: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.outer_wall_width.min_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.08..=2.00)
                                        .prefix("Min: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.outer_wall_width.max_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.12..=3.00)
                                        .prefix("Max: "),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Inner Wall:");
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.inner_wall_width.target_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.10..=2.50)
                                        .prefix("T: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.inner_wall_width.min_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.08..=2.00)
                                        .prefix("Min: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.inner_wall_width.max_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.12..=3.00)
                                        .prefix("Max: "),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Solid Infill:");
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.solid_infill_width.target_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.10..=2.50)
                                        .prefix("T: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.solid_infill_width.min_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.08..=2.00)
                                        .prefix("Min: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.solid_infill_width.max_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.12..=3.00)
                                        .prefix("Max: "),
                                    )
                                    .changed();
                            });
                            ui.horizontal(|ui| {
                                ui.label("Sparse Infill:");
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.sparse_infill_width.target_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.10..=2.50)
                                        .prefix("T: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.sparse_infill_width.min_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.08..=2.00)
                                        .prefix("Min: "),
                                    )
                                    .changed();
                                tp_cfg_changed |= ui
                                    .add(
                                        egui::DragValue::new(
                                            &mut config.slicing.sparse_infill_width.max_mm,
                                        )
                                        .speed(0.01)
                                        .range(0.12..=3.00)
                                        .prefix("Max: "),
                                    )
                                    .changed();
                            });

                            if tp_cfg_changed {
                                config.slicing.sanitize();
                                if !is_computing {
                                    sync_config_to_backend(&config);
                                }
                                vis_state.dirty_toolpaths = true;
                            }

                            ui.horizontal(|ui| {
                                if ui
                                    .checkbox(
                                        &mut vis_state.show_toolpaths,
                                        "Show 3D Shell Toolpaths",
                                    )
                                    .changed()
                                {
                                    vis_state.dirty_toolpaths_mesh = true;
                                    if vis_state.show_toolpaths && vis_state.solid_opacity > 0.35 {
                                        vis_state.solid_opacity = 0.18;
                                    }
                                    if vis_state.show_toolpaths && !has_eikonal && !is_computing {
                                        spawn_field_compute_task(
                                            &mut vis_state,
                                            &mut plate,
                                            &config,
                                            sel_id,
                                            &obj_name,
                                            obj_rotation,
                                        );
                                    }
                                }

                                if ui
                                    .add_enabled(
                                        !is_computing,
                                        egui::Button::new("Generate Shell Toolpaths").small(),
                                    )
                                    .clicked()
                                {
                                    vis_state.show_toolpaths = true;
                                    if vis_state.solid_opacity > 0.35 {
                                        vis_state.solid_opacity = 0.18;
                                    }
                                    if has_eikonal {
                                        let need_layers = vis_state.cached_selected_layers.is_none()
                                            || vis_state.dirty_selected_layers;
                                        vis_state.spawn_slicing_task(
                                            sel_id,
                                            &config,
                                            need_layers,
                                            true,
                                        );
                                        plate.status_message = format!(
                                            "Generating non-planar shell toolpaths for '{obj_name}' in background..."
                                        );
                                    } else {
                                        spawn_field_compute_task(
                                            &mut vis_state,
                                            &mut plate,
                                            &config,
                                            sel_id,
                                            &obj_name,
                                            obj_rotation,
                                        );
                                    }
                                }
                            });

                            if vis_state.show_toolpaths {
                                ui.horizontal_wrapped(|ui| {
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_outer_walls,
                                            "Outer Walls",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_inner_walls,
                                            "Inner Walls",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_solid_infill,
                                            "Solid Infill & Gap Fill",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                    if ui
                                        .checkbox(
                                            &mut vis_state.show_sparse_infill,
                                            "Sparse Infill",
                                        )
                                        .changed()
                                    {
                                        vis_state.dirty_toolpaths_mesh = true;
                                    }
                                    ui.checkbox(
                                        &mut vis_state.show_toolpath_centerlines,
                                        "Centerlines",
                                    );
                                });

                                egui::ComboBox::from_id_source("toolpath_color_mode_combo")
                                    .selected_text(vis_state.toolpath_color_mode.label())
                                    .show_ui(ui, |ui| {
                                        for mode in [
                                            ToolpathColorMode::ByFeatureType,
                                            ToolpathColorMode::ByLineWidth,
                                        ] {
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
                            }

                            if let Some(ref tp_res) = vis_state.cached_toolpaths {
                                ui.small(format!(
                                    "Paths: {} total ({} outer, {} inner, {} solid/gap, {} sparse)",
                                    tp_res.total_path_count,
                                    tp_res.outer_wall_count,
                                    tp_res.inner_wall_count,
                                    tp_res.solid_infill_count,
                                    tp_res.sparse_infill_count
                                ));
                            }
                        });
                    }
                }

                // --- Machine Tools, Clearance Profiles & Multi-Axis Kinematics ---
                ui.add_space(8.0);
                ui.separator();
                ui.collapsing("Machine Tools & Multi-Axis Kinematics", |ui| {
                    let mut dirty_backend_sync = false;

                    // 1. Multi-Tool Registry (Nozzle Diameter, Temperature, XZ Clearance Profile)
                    ui.label(egui::RichText::new("Extrusion Tools").strong());
                    if config.tools.is_empty() {
                        config.tools.push(ToolConfig::default());
                        config.active_tool_index = 0;
                        dirty_backend_sync = true;
                    }
                    if config.active_tool_index >= config.tools.len() {
                        config.active_tool_index = config.tools.len() - 1;
                    }

                    ui.horizontal_wrapped(|ui| {
                        let current_active = config.active_tool_index;
                        let mut clicked_idx = None;
                        for (idx, tool) in config.tools.iter().enumerate() {
                            let selected = current_active == idx;
                            if ui
                                .selectable_label(selected, format!("T{} ({:.2}mm)", tool.id, tool.nozzle_diameter_mm))
                                .clicked()
                            {
                                clicked_idx = Some(idx);
                            }
                        }
                        if let Some(idx) = clicked_idx {
                            config.active_tool_index = idx;
                            dirty_backend_sync = true;
                        }
                        if ui.small_button("+ Add Tool").clicked() {
                            let next_id = config
                                .tools
                                .iter()
                                .map(|t| t.id)
                                .max()
                                .unwrap_or(0)
                                + 1;
                            config.tools.push(ToolConfig::standard_tool(
                                next_id,
                                format!("T{next_id} - 0.4mm"),
                                0.40,
                                215.0,
                            ));
                            config.active_tool_index = config.tools.len() - 1;
                            dirty_backend_sync = true;
                        }
                        if config.tools.len() > 1 && ui.small_button("Remove Tool").clicked() {
                            let rm_idx = config.active_tool_index;
                            config.tools.remove(rm_idx);
                            if config.active_tool_index >= config.tools.len() {
                                config.active_tool_index = config.tools.len() - 1;
                            }
                            dirty_backend_sync = true;
                        }
                    });

                    let active_idx = config.active_tool_index;
                    if let Some(tool) = config.tools.get_mut(active_idx) {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            if ui.text_edit_singleline(&mut tool.name).changed() {
                                dirty_backend_sync = true;
                            }
                        });
                        if ui
                            .add(
                                egui::Slider::new(&mut tool.nozzle_diameter_mm, 0.15..=2.0)
                                    .text("Nozzle Diameter (mm)"),
                            )
                            .changed()
                        {
                            dirty_backend_sync = true;
                        }
                        if ui
                            .add(
                                egui::Slider::new(&mut tool.temperature_celsius, 170.0..=350.0)
                                    .text("Temperature (°C)"),
                            )
                            .changed()
                        {
                            dirty_backend_sync = true;
                        }
                        if ui
                            .checkbox(
                                &mut tool.has_rotational_axes,
                                "Tool Has Rotational Axes (5/6-Axis)",
                            )
                            .changed()
                        {
                            dirty_backend_sync = true;
                        }
                        if tool.has_rotational_axes
                            && ui
                                .add(
                                    egui::Slider::new(
                                        &mut tool.max_angle_from_bed_normal_deg,
                                        0.0..=85.0,
                                    )
                                    .text("Max Angle from Bed Normal (°)"),
                                )
                                .changed()
                        {
                            dirty_backend_sync = true;
                        }

                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new("Clearance Profile (XZ in Nozzle Frame)").strong(),
                        );
                        ui.small("Radius X (mm) from nozzle center vs. height Z (mm) above tip:");

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
                                    dirty_backend_sync = true;
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
                                    dirty_backend_sync = true;
                                }
                                if profile_len > 2 && ui.small_button("✕").clicked() {
                                    remove_pt_idx = Some(p_idx);
                                }
                            });
                        }
                        if let Some(rm_idx) = remove_pt_idx {
                            tool.clearance_profile_xz.remove(rm_idx);
                            dirty_backend_sync = true;
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
                                dirty_backend_sync = true;
                            }
                            if ui.small_button("Sort by Z").clicked() {
                                tool.clearance_profile_xz.sort_by(|a, b| {
                                    a.z_height_mm
                                        .partial_cmp(&b.z_height_mm)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                dirty_backend_sync = true;
                            }
                        });
                    }

                    ui.separator();

                    // 2. Nozzle Local Frame & Bed-Coordinate Gravity Vector
                    ui.label(
                        egui::RichText::new("Nozzle Frame & Gravity (Bed Coordinates)").strong(),
                    );
                    let queried_frame = config.query_nozzle_frame(Vec3::ZERO, Vec3::X);
                    let queried_gravity = config.query_gravity_bed(Vec3::ZERO);

                    ui.small(format!(
                        "Nozzle +Z (opp. nozzle): [{:.2}, {:.2}, {:.2}]",
                        queried_frame.nozzle_axis.x,
                        queried_frame.nozzle_axis.y,
                        queried_frame.nozzle_axis.z
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
                    tilt_changed |= ui
                        .add(
                            egui::Slider::new(&mut config.machine.bed_tilt_rx_deg, -60.0..=60.0)
                                .text("5-Axis Bed Tilt Rx (°)"),
                        )
                        .changed();
                    tilt_changed |= ui
                        .add(
                            egui::Slider::new(&mut config.machine.bed_tilt_ry_deg, -60.0..=60.0)
                                .text("5-Axis Bed Tilt Ry (°)"),
                        )
                        .changed();
                    if ui.small_button("Reset Level Bed (0°, 0°)").clicked() {
                        config.machine.bed_tilt_rx_deg = 0.0;
                        config.machine.bed_tilt_ry_deg = 0.0;
                        tilt_changed = true;
                    }
                    if tilt_changed {
                        config.machine.update_gravity_from_bed_tilt();
                        dirty_backend_sync = true;
                    }

                    ui.checkbox(
                        &mut config.machine.show_nozzle_clearance_gizmo,
                        "Show 3D Nozzle Frame, Clearance & Gravity Gizmo",
                    );

                    if dirty_backend_sync && !vis_state.is_busy() {
                        sync_config_to_backend(&config);
                    }
                });

                // --- Print Bed & Slice Configuration ---
                ui.add_space(8.0);
                ui.separator();
                ui.collapsing("Print Bed & Slice Configuration", |ui| {
                    ui.label(egui::RichText::new("Variable Slice Layer Height (mm)").strong());
                    let mut lh_changed = false;
                    lh_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut config.slicing.layer_height.target_mm,
                                0.04..=1.20,
                            )
                            .text("Target Layer Height (mm)"),
                        )
                        .changed();
                    lh_changed |= ui
                        .add(
                            egui::Slider::new(&mut config.slicing.layer_height.min_mm, 0.02..=0.80)
                                .text("Min Layer Height (mm)"),
                        )
                        .changed();
                    lh_changed |= ui
                        .add(
                            egui::Slider::new(&mut config.slicing.layer_height.max_mm, 0.05..=1.50)
                                .text("Max Layer Height (mm)"),
                        )
                        .changed();
                    lh_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut config.slicing.layer_height.first_layer_mm,
                                0.02..=1.20,
                            )
                            .text("First Layer Height (mm)"),
                        )
                        .changed();
                    lh_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut config.slicing.layer_height.top_layer_mm,
                                0.02..=1.20,
                            )
                            .text("Top Layer Height (mm)"),
                        )
                        .changed();
                    if lh_changed {
                        config.slicing.layer_height.sanitize();
                        if !vis_state.is_busy() {
                            sync_config_to_backend(&config);
                        }
                        vis_state.dirty_selected_layers = true;
                        vis_state.dirty_toolpaths = true;
                    }

                    ui.add(
                        egui::Slider::new(&mut config.slicing.max_nonplanar_slope_deg, 0.0..=65.0)
                            .text("Max Slope (°)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut config.eikonal.voxel_size_mm, 0.10..=2.0)
                            .text("OpenVDB Voxel (mm)"),
                    );

                    ui.separator();
                    ui.label(egui::RichText::new("Print Bed Geometry").strong());
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut config.bed.shape,
                            BedShape::Rectangular,
                            "Rectangular",
                        );
                        ui.selectable_value(
                            &mut config.bed.shape,
                            BedShape::Circular,
                            "Circular (Delta)",
                        );
                    });

                    ui.horizontal(|ui| {
                        ui.label("Presets:");
                        if ui.small_button("256³ (Bambu/Voron)").clicked() {
                            config.bed.width_x = 256.0;
                            config.bed.depth_y = 256.0;
                            config.bed.height_z = 256.0;
                        }
                        if ui.small_button("250×210 (Prusa)").clicked() {
                            config.bed.width_x = 250.0;
                            config.bed.depth_y = 210.0;
                            config.bed.height_z = 220.0;
                        }
                        if ui.small_button("350³ (Large)").clicked() {
                            config.bed.width_x = 350.0;
                            config.bed.depth_y = 350.0;
                            config.bed.height_z = 350.0;
                        }
                    });

                    ui.add(
                        egui::Slider::new(&mut config.bed.width_x, 50.0..=600.0)
                            .text("Width X (mm)"),
                    );
                    if config.bed.shape == BedShape::Rectangular {
                        ui.add(
                            egui::Slider::new(&mut config.bed.depth_y, 50.0..=600.0)
                                .text("Depth Y (mm)"),
                        );
                    }
                    ui.add(
                        egui::Slider::new(&mut config.bed.height_z, 50.0..=600.0)
                            .text("Max Z (mm)"),
                    );
                    ui.checkbox(&mut config.bed.show_volume_box, "Show Build Volume Box");

                    ui.separator();
                    ui.label(egui::RichText::new("Auto-Arrangement Settings").strong());
                    ui.add(
                        egui::Slider::new(&mut config.arrangement.part_clearance_mm, 0.5..=25.0)
                            .text("Part Clearance (mm)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut config.arrangement.bed_edge_margin_mm, 0.0..=25.0)
                            .text("Bed Margin (mm)"),
                    );
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
            });
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
        .map(|t| ffi::ToolSpec {
            id: t.id,
            name: t.name.clone(),
            nozzle_diameter_mm: t.nozzle_diameter_mm,
            temperature_celsius: t.temperature_celsius,
            has_rotational_axes: t.has_rotational_axes,
            max_angle_from_bed_normal_deg: t.max_angle_from_bed_normal_deg,
            clearance_profile_xz: t
                .clearance_profile_xz
                .iter()
                .map(|p| ffi::ClearanceXZ {
                    x_radius_mm: p.x_radius_mm,
                    z_height_mm: p.z_height_mm,
                })
                .collect(),
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
    });

    ffi::set_eikonal_solver_iterations(config.eikonal.solver_iterations);
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
    mut vis_state: ResMut<FieldVisualizationState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut camera_query: Query<(&mut OrbitCamera, &mut Transform), Without<CadObjectEntity>>,
) {
    // 1. Poll pending multi-file import task
    if let Some(ref mut task) = plate.pending_load_task {
        if let Some(result) = block_on(future::poll_once(task)) {
            plate.pending_load_task = None;

            let Some(report) = result else {
                plate.status_message = "File import cancelled.".to_string();
                return;
            };

            if let Some(restored_cfg) = report.restored_config {
                *config = restored_cfg;
                sync_config_to_backend(&config);
            }

            plate.import_notices = report.notices;
            plate.import_errors = report.errors;

            let loaded_count = report.objects.len();
            if loaded_count > 0 {
                spawn_loaded_objects(
                    &mut commands,
                    render_device.as_deref(),
                    &mut meshes,
                    &mut materials,
                    &mut plate,
                    report.objects,
                );
                frame_camera_to_plate(&plate, &config, &mut camera_query);
                plate.status_message = format!(
                    "Loaded {loaded_count} object(s); {} total on build plate.",
                    plate.objects.len()
                );
            } else if !plate.import_errors.is_empty() {
                plate.status_message = plate.import_errors[0].clone();
            } else {
                plate.status_message = "No supported 3D objects found in selected file(s).".to_string();
            }
        }
    }

    // 2. Poll pending 3MF project save task
    if let Some(ref mut task) = plate.pending_save_task {
        if let Some(result) = block_on(future::poll_once(task)) {
            plate.pending_save_task = None;
            match result {
                Some(Ok(saved_path)) => {
                    plate.status_message = format!("Saved 3MF project & config to '{saved_path}'.");
                }
                Some(Err(err)) => {
                    plate.status_message = format!("3MF save error: {err}");
                }
                None => {
                    plate.status_message = "3MF save cancelled.".to_string();
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
    camera_query: &mut Query<(&mut OrbitCamera, &mut Transform), Without<CadObjectEntity>>,
) {
    let bed_span = config.bed.width_x.max(config.bed.depth_y).max(100.0);
    for (mut orbit, mut transform) in camera_query.iter_mut() {
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
