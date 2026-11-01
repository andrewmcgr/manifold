pub mod stl;
pub mod threemf;

use std::path::{Path, PathBuf};
use bevy::prelude::*;
use crate::bridge::ffi::{self, DecomposedPartInfo, RenderVertex};
use crate::config::SlicerConfig;
use crate::hull::{
    build_surface_patches, SimplifiedConvexHull, SurfacePatch, MAX_SIMPLIFIED_HULL_FACES,
};

/// Complete CPU-side payload for a single object loaded and processed by the C++ backend.
#[derive(Clone, Debug)]
pub struct LoadedObjectData {
    pub object_id: u64,
    pub name: String,
    pub source_file: String,
    pub has_brep: bool,
    pub initial_translation: Vec3,
    pub initial_rotation: Quat,
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
    pub local_min: Vec3,
    pub local_max: Vec3,
    pub hull: SimplifiedConvexHull,
    pub surface_patches: Vec<SurfacePatch>,
    pub triangle_to_patch: Vec<u32>,
}

/// Summary of an asynchronous multi-file import operation.
#[derive(Clone, Debug, Default)]
pub struct LoadedBatchReport {
    pub objects: Vec<LoadedObjectData>,
    pub restored_config: Option<SlicerConfig>,
    pub notices: Vec<String>,
    pub errors: Vec<String>,
}

/// Loads one or more CAD/mesh/project files (`.step`, `.stp`, `.stl`, `.3mf`)
/// through the C++ backend and builds their local meshes and $\le 27$-face convex hulls.
pub fn load_files_into_backend(paths: &[PathBuf]) -> LoadedBatchReport {
    let mut report = LoadedBatchReport::default();

    for path in paths {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        match ext.as_str() {
            "step" | "stp" => {
                load_step_file_parts(path, & path.to_string_lossy(), &mut report);
            }
            "stl" => match stl::load_stl_file(path) {
                Ok(obj) => report.objects.push(obj),
                Err(err) => report.errors.push(format!("{}: {err}", path.display())),
            },
            "3mf" => {
                threemf::load_3mf_archive(path, &mut report);
            }
            other => {
                report.errors.push(format!(
                    "Unsupported file extension '.{other}' for {}",
                    path.display()
                ));
            }
        }
    }

    report
}

/// Decomposes a STEP file via the C++ OCCT backend (`ffi::load_step_assembly`)
/// and constructs `LoadedObjectData` for every solid/shell part in the assembly.
pub fn load_step_file_parts(
    step_path: &Path,
    display_source: &str,
    report: &mut LoadedBatchReport,
) -> Vec<LoadedObjectData> {
    let path_str = step_path.to_string_lossy();
    let decomposed = ffi::load_step_assembly(&path_str);

    if decomposed.is_empty() {
        report
            .errors
            .push(format!("OCCT failed to read or tessellate STEP file: {display_source}"));
        return Vec::new();
    }

    if decomposed.len() > 1 {
        report.notices.push(format!(
            "Decomposed STEP assembly '{}' into {} independent parts.",
            step_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(display_source),
            decomposed.len()
        ));
    }

    let mut loaded_parts = Vec::with_capacity(decomposed.len());
    for part in decomposed {
        if let Some(obj) = build_loaded_object_from_part(part, display_source) {
            loaded_parts.push(obj);
        }
    }

    // Center the STEP part (or multi-part STEP assembly as a group) at the bed center `(0, 0)`
    // in scene coordinates while preserving relative XY offsets between sibling parts.
    if !loaded_parts.is_empty() {
        let mut min_xy = Vec2::splat(f32::INFINITY);
        let mut max_xy = Vec2::splat(f32::NEG_INFINITY);
        for obj in &loaded_parts {
            let t_xy = obj.initial_translation.xy();
            min_xy = min_xy.min(t_xy + obj.local_min.xy());
            max_xy = max_xy.max(t_xy + obj.local_max.xy());
        }
        if min_xy.x.is_finite() && max_xy.x.is_finite() {
            let assembly_center_xy = 0.5 * (min_xy + max_xy);
            for obj in &mut loaded_parts {
                obj.initial_translation.x -= assembly_center_xy.x;
                obj.initial_translation.y -= assembly_center_xy.y;
            }
        }
    }

    for obj in &loaded_parts {
        report.objects.push(obj.clone());
    }
    loaded_parts
}

/// Fetches the centered vertices and indices from the C++ backend for `part.object_id`,
/// computes local AABB bounds, builds its $\le 27$-face `SimplifiedConvexHull`, and segments
/// its surface into selectable `SurfacePatch` regions (exact STEP `TopoDS_Face`s or contiguous
/// near-planar mesh regions).
pub fn build_loaded_object_from_part(
    part: DecomposedPartInfo,
    source_file: &str,
) -> Option<LoadedObjectData> {
    if part.object_id == 0 {
        return None;
    }

    let vertices = ffi::get_object_vertices(part.object_id);
    let indices = ffi::get_object_indices(part.object_id);
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let brep_face_ids = if part.has_brep {
        ffi::get_object_triangle_face_ids(part.object_id)
    } else {
        Vec::new()
    };

    let (local_min, local_max) = compute_vertex_bounds(&vertices);
    let hull = SimplifiedConvexHull::from_vertices(&vertices, MAX_SIMPLIFIED_HULL_FACES);
    let (surface_patches, triangle_to_patch) =
        build_surface_patches(&vertices, &indices, &brep_face_ids);

    Some(LoadedObjectData {
        object_id: part.object_id,
        name: part.name,
        source_file: source_file.to_string(),
        has_brep: part.has_brep,
        // Place at the assembly's XY coordinates with z=0 sitting on the print bed
        initial_translation: Vec3::new(part.origin_x, part.origin_y, 0.0),
        initial_rotation: Quat::IDENTITY,
        vertices,
        indices,
        local_min,
        local_max,
        hull,
        surface_patches,
        triangle_to_patch,
    })
}

pub fn compute_vertex_bounds(vertices: &[RenderVertex]) -> (Vec3, Vec3) {
    let mut min_pt = Vec3::splat(f32::INFINITY);
    let mut max_pt = Vec3::splat(f32::NEG_INFINITY);
    for v in vertices {
        let p = Vec3::new(v.x, v.y, v.z);
        min_pt = min_pt.min(p);
        max_pt = max_pt.max(p);
    }
    if min_pt.x.is_infinite() {
        (Vec3::ZERO, Vec3::ZERO)
    } else {
        (min_pt, max_pt)
    }
}
