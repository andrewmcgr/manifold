use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{block_on, IoTaskPool};
use bevy_egui::{egui, EguiContexts};

use crate::bridge::ffi;
use crate::config::{BedShape, SlicerConfig};
use crate::field_vis::{FieldColormap, FieldVisualizationState, SliceAxis};
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
    let ctx = contexts.ctx_mut();

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
                        }

                        // --- Generic Volumetric Field Visualization (OpenVDB) ---
                        ui.add_space(8.0);
                        ui.separator();
                        egui::CollapsingHeader::new(
                            egui::RichText::new("Volumetric Field Visualization (OpenVDB)").strong(),
                        )
                        .default_open(true)
                        .show(ui, |ui| {
                            // Field Selector
                            ui.horizontal(|ui| {
                                ui.label("Scalar Field:");
                                let avail = vis_state.available_fields.clone();
                                for fname in &avail {
                                    let selected = &vis_state.active_field_name == fname;
                                    if ui.selectable_label(selected, fname).clicked() && !selected {
                                        vis_state.active_field_name = fname.clone();
                                        vis_state.mark_all_dirty();
                                        vis_state.refresh_stats_for(sel_id);
                                    }
                                }
                            });

                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::Slider::new(&mut config.eikonal.voxel_size_mm, 0.15..=3.0)
                                        .text("Voxel (mm)"),
                                );
                                if ui.small_button("Recompute Fields").clicked() {
                                    if ffi::recompute_object_fields(
                                        sel_id,
                                        config.eikonal.voxel_size_mm,
                                        config.eikonal.narrow_band_width_voxels as f32,
                                    ) {
                                        vis_state.mark_all_dirty();
                                        vis_state.refresh_stats_for(sel_id);
                                        plate.status_message = format!(
                                            "Recomputed OpenVDB SDF & scalar fields for '{}' (voxel {:.2} mm).",
                                            obj_name, config.eikonal.voxel_size_mm
                                        );
                                    }
                                }
                            });

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
                                ui.small(format!(
                                    "Isosurface Mesh: {} triangles",
                                    vis_state.isosurface_triangles
                                ));
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
                    }
                }

                // --- Print Bed & Extensible Slicer Configuration ---
                ui.add_space(8.0);
                ui.separator();
                ui.collapsing("Print Bed & Slicer Configuration", |ui| {
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

                    ui.separator();
                    ui.label(egui::RichText::new("Non-Planar & Eikonal Slicing (Preview)").strong());
                    ui.add(
                        egui::Slider::new(&mut config.slicing.base_layer_height_mm, 0.05..=0.40)
                            .text("Base Layer (mm)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut config.slicing.max_nonplanar_slope_deg, 0.0..=65.0)
                            .text("Max Slope (°)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut config.eikonal.voxel_size_mm, 0.10..=2.0)
                            .text("OpenVDB Voxel (mm)"),
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

/// Polls background import/export tasks on each frame without blocking the macOS main thread.
#[allow(clippy::too_many_arguments)]
pub fn poll_io_tasks_system(
    mut commands: Commands,
    render_device: Option<Res<RenderDevice>>,
    mut plate: ResMut<BuildPlateState>,
    mut config: ResMut<SlicerConfig>,
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
