use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use bevy::prelude::*;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::bridge::ffi::{self, RenderVertex};
use crate::config::SlicerConfig;
use crate::io::{
    build_loaded_object_from_part, load_step_file_parts, stl, LoadedBatchReport, LoadedObjectData,
};

const MANIFOLD_CONFIG_ENTRY: &str = "Metadata/manifold2_config.toml";
const MANIFOLD_SCENE_ENTRY: &str = "Metadata/manifold2_scene.toml";

/// Saved metadata for objects exported by Manifold2 so that embedded STEP B-Rep
/// shapes inside `/Metadata/` reconnect with their exact world transforms upon reload.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SavedSceneManifest {
    objects: Vec<SavedObjectEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SavedObjectEntry {
    name: String,
    embedded_step_entry: Option<String>,
    translation: [f32; 3],
    rotation_xyzw: [f32; 4],
}

/// Snapshot of an active build-plate object passed to the 3MF exporter.
pub struct ExportableObject {
    pub object_id: u64,
    pub name: String,
    pub has_brep: bool,
    pub transform: Transform,
    pub vertices: Vec<RenderVertex>,
    pub indices: Vec<u32>,
}

/// Raw `<mesh>` parsed from a `.model` XML inside a `.3mf` archive.
#[derive(Clone, Debug, Default)]
struct ParsedXmlMesh {
    positions: Vec<f32>,
    indices: Vec<u32>,
}

/// `<component>` reference inside an `<object><components>` block.
#[derive(Clone, Debug)]
struct ParsedComponentRef {
    object_id: String,
    model_path: Option<String>,
    transform: Mat4,
}

/// `<object>` resource inside a `.model` XML file.
#[derive(Clone, Debug, Default)]
struct ParsedXmlObject {
    id: String,
    name: Option<String>,
    mesh: Option<ParsedXmlMesh>,
    components: Vec<ParsedComponentRef>,
}

/// `<item>` entry inside `<build>`.
#[derive(Clone, Debug)]
struct ParsedBuildItem {
    object_id: String,
    model_path: Option<String>,
    transform: Mat4,
}

/// All `<object>` resources and `<build>` items parsed from a single `.model` XML stream.
#[derive(Clone, Debug, Default)]
struct ParsedModelFile {
    objects: HashMap<String, ParsedXmlObject>,
    build_items: Vec<ParsedBuildItem>,
}

/// Loads a `.3mf` OPC/ZIP archive, supporting:
/// - Standard 3MF `/3D/3dmodel.model` and Production Extension sub-models (`/3D/Objects/*.model`)
/// - Embedded `.step` / `.stp` CAD files
/// - Embedded `.stl` mesh files
/// - Saved Manifold2 configuration (`/Metadata/manifold2_config.toml`)
/// - Graceful ignoring + reporting of unsupported files from third-party slicers
pub fn load_3mf_archive(path: &Path, report: &mut LoadedBatchReport) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            report
                .errors
                .push(format!("Failed to open 3MF archive {}: {e}", path.display()));
            return;
        }
    };

    let mut archive = match ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) => {
            report
                .errors
                .push(format!("Invalid 3MF ZIP archive {}: {e}", path.display()));
            return;
        }
    };

    let archive_label = path.to_string_lossy().into_owned();
    let mut model_files: HashMap<String, ParsedModelFile> = HashMap::new();
    let mut embedded_step_entries: HashMap<String, Vec<u8>> = HashMap::new();
    let mut embedded_stl_entries: Vec<(String, Vec<u8>)> = Vec::new();
    let mut scene_manifest: Option<SavedSceneManifest> = None;
    let mut is_manifold_project = false;
    let mut skipped_third_party_entries: Vec<String> = Vec::new();

    for i in 0..archive.len() {
        let mut entry = match archive.by_index(i) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if entry.is_dir() {
            continue;
        }

        let raw_name = entry.name().trim_start_matches('/').to_string();
        let lower_name = raw_name.to_ascii_lowercase();

        if raw_name == MANIFOLD_CONFIG_ENTRY {
            let mut text = String::new();
            if entry.read_to_string(&mut text).is_ok() {
                match SlicerConfig::from_toml_str(&text) {
                    Ok(cfg) => {
                        is_manifold_project = true;
                        report.restored_config = Some(cfg);
                        report.notices.push(format!(
                            "Restored Manifold2 slicer & bed configuration from '{}'.",
                            path.file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or(&archive_label)
                        ));
                    }
                    Err(err) => {
                        report
                            .notices
                            .push(format!("Ignored malformed {MANIFOLD_CONFIG_ENTRY}: {err}"));
                    }
                }
            }
            continue;
        }

        if raw_name == MANIFOLD_SCENE_ENTRY {
            let mut text = String::new();
            if entry.read_to_string(&mut text).is_ok() {
                if let Ok(manifest) = toml::from_str::<SavedSceneManifest>(&text) {
                    is_manifold_project = true;
                    scene_manifest = Some(manifest);
                }
            }
            continue;
        }

        // Standard OPC relationship / content-type plumbing
        if lower_name == "[content_types].xml" || lower_name.ends_with(".rels") {
            continue;
        }

        if lower_name.ends_with(".model") {
            let mut xml_bytes = Vec::new();
            if entry.read_to_end(&mut xml_bytes).is_ok() {
                match parse_3mf_model_xml(&xml_bytes) {
                    Ok(parsed) => {
                        model_files.insert(raw_name, parsed);
                    }
                    Err(err) => {
                        report
                            .errors
                            .push(format!("Failed to parse 3MF XML '{raw_name}': {err}"));
                    }
                }
            }
            continue;
        }

        if lower_name.ends_with(".step") || lower_name.ends_with(".stp") {
            let mut bytes = Vec::new();
            if entry.read_to_end(&mut bytes).is_ok() {
                embedded_step_entries.insert(raw_name, bytes);
            }
            continue;
        }

        if lower_name.ends_with(".stl") {
            let mut bytes = Vec::new();
            if entry.read_to_end(&mut bytes).is_ok() {
                embedded_stl_entries.push((raw_name, bytes));
            }
            continue;
        }

        // Any other file inside the 3MF (e.g., PrusaSlicer/OrcaSlicer/BambuStudio
        // .config, .gcode, thumbnails, custom profile JSONs) is safely ignored.
        skipped_third_party_entries.push(raw_name);
    }

    if is_manifold_project {
        if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
            report.manifold_project_filename = Some(fname.to_string());
        }
    }

    if !skipped_third_party_entries.is_empty() {
        let preview: Vec<&str> = skipped_third_party_entries
            .iter()
            .take(4)
            .map(String::as_str)
            .collect();
        let suffix = if skipped_third_party_entries.len() > 4 {
            format!(" (+{} more)", skipped_third_party_entries.len() - 4)
        } else {
            String::new()
        };
        report.notices.push(format!(
            "Ignored {} unsupported/third-party 3MF entry(s): {}{}",
            skipped_third_party_entries.len(),
            preview.join(", "),
            suffix
        ));
    }

    // 1. If this 3MF has embedded STEP files, load them through OCCT!
    //    Track which embedded STEP files were matched to a SavedSceneManifest entry
    //    so we don't duplicate mesh fallbacks that were written for compatibility.
    let mut manifest_step_loaded = false;
    if let Some(ref manifest) = scene_manifest {
        for saved_obj in &manifest.objects {
            if let Some(ref step_entry) = saved_obj.embedded_step_entry {
                let normalized = step_entry.trim_start_matches('/');
                if let Some(step_bytes) = embedded_step_entries.remove(normalized) {
                    if let Some(mut loaded_objs) = load_embedded_step_bytes(
                        normalized,
                        &step_bytes,
                        &archive_label,
                        report,
                    ) {
                        for obj in &mut loaded_objs {
                            obj.name = saved_obj.name.clone();
                            obj.initial_translation = Vec3::from_array(saved_obj.translation);
                            obj.initial_rotation =
                                Quat::from_array(saved_obj.rotation_xyzw).normalize();
                        }
                        if let Some(last) = report.objects.last_mut() {
                            if loaded_objs.len() == 1 {
                                last.name = saved_obj.name.clone();
                                last.initial_translation = Vec3::from_array(saved_obj.translation);
                                last.initial_rotation =
                                    Quat::from_array(saved_obj.rotation_xyzw).normalize();
                            }
                        }
                        manifest_step_loaded = true;
                    }
                }
            }
        }
    }

    // Load any remaining standalone embedded STEP files inside the 3MF
    for (entry_name, step_bytes) in &embedded_step_entries {
        let _ = load_embedded_step_bytes(entry_name, step_bytes, &archive_label, report);
    }

    // 2. Load any embedded STL files inside the 3MF
    for (entry_name, stl_bytes) in embedded_stl_entries {
        let mut cursor = Cursor::new(stl_bytes);
        let stem = Path::new(&entry_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("embedded_stl");
        match stl::load_stl_from_reader(&mut cursor, stem, &format!("{archive_label}::{entry_name}"))
        {
            Ok(obj) => report.objects.push(obj),
            Err(err) => report.errors.push(err),
        }
    }

    // 3. Instantiate `<build><item>` objects from `/3D/3dmodel.model` (or fallback root model)
    //    If all objects in a Manifold2 3MF were already restored from lossless STEP B-Reps,
    //    only load the non-STEP objects from `<build>`.
    if manifest_step_loaded
        && scene_manifest
            .as_ref()
            .map_or(false, |m| m.objects.iter().all(|o| o.embedded_step_entry.is_some()))
    {
        return;
    }

    let root_model_key = model_files
        .keys()
        .find(|k| k.eq_ignore_ascii_case("3d/3dmodel.model"))
        .cloned()
        .or_else(|| model_files.keys().next().cloned());

    let Some(root_key) = root_model_key else {
        return;
    };

    let Some(root_model) = model_files.get(&root_key).cloned() else {
        return;
    };

    let file_stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("3mf_model");

    if root_model.build_items.is_empty() {
        // Fallback if `<build>` was omitted: instantiate every mesh object directly
        for obj in root_model.objects.values() {
            if let Some(ref mesh) = obj.mesh {
                let name = obj
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{file_stem}_obj_{}", obj.id));
                instantiate_3mf_mesh(&name, mesh, Mat4::IDENTITY, &archive_label, report);
            }
        }
    } else {
        let mut visited = HashSet::new();
        for (idx, item) in root_model.build_items.iter().enumerate() {
            let target_model = item
                .model_path
                .as_deref()
                .map(|p| p.trim_start_matches('/').to_string())
                .unwrap_or_else(|| root_key.clone());

            visited.clear();
            instantiate_3mf_object_recursive(
                &model_files,
                &target_model,
                &item.object_id,
                item.transform,
                &format!("{file_stem}_{}", idx + 1),
                &archive_label,
                &mut visited,
                report,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn instantiate_3mf_object_recursive(
    model_files: &HashMap<String, ParsedModelFile>,
    model_key: &str,
    object_id: &str,
    accumulated_transform: Mat4,
    fallback_name: &str,
    archive_label: &str,
    visited: &mut HashSet<(String, String)>,
    report: &mut LoadedBatchReport,
) {
    let visit_key = (model_key.to_string(), object_id.to_string());
    if !visited.insert(visit_key) {
        return;
    }

    // Case-insensitive lookup for model file path inside ZIP
    let resolved_model = model_files.get(model_key).or_else(|| {
        model_files
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(model_key))
            .map(|(_, v)| v)
    });

    let Some(model_file) = resolved_model else {
        report.notices.push(format!(
            "Skipped missing 3MF sub-model reference '{model_key}' (object {object_id})"
        ));
        return;
    };

    let Some(xml_obj) = model_file.objects.get(object_id) else {
        return;
    };

    let obj_name = xml_obj
        .name
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or(fallback_name);

    if let Some(ref mesh) = xml_obj.mesh {
        instantiate_3mf_mesh(obj_name, mesh, accumulated_transform, archive_label, report);
    }

    for (comp_idx, comp) in xml_obj.components.iter().enumerate() {
        let next_model = comp
            .model_path
            .as_deref()
            .map(|p| p.trim_start_matches('/').to_string())
            .unwrap_or_else(|| model_key.to_string());
        let next_transform = accumulated_transform * comp.transform;
        let child_fallback = format!("{obj_name}_part_{}", comp_idx + 1);
        instantiate_3mf_object_recursive(
            model_files,
            &next_model,
            &comp.object_id,
            next_transform,
            &child_fallback,
            archive_label,
            visited,
            report,
        );
    }
}

fn instantiate_3mf_mesh(
    name: &str,
    mesh: &ParsedXmlMesh,
    world_transform: Mat4,
    archive_label: &str,
    report: &mut LoadedBatchReport,
) {
    if mesh.positions.len() < 9 || mesh.indices.len() < 3 {
        return;
    }

    // Apply the 3MF `<item>` / `<component>` transform to the vertices before registering
    // in the C++ backend so the backend computes accurate world-scale normals and bounds
    // while recentering the local mesh at (origin_x, origin_y, z_min).
    let num_verts = mesh.positions.len() / 3;
    let mut transformed_positions = Vec::with_capacity(mesh.positions.len());
    for i in 0..num_verts {
        let local_p = Vec3::new(
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        );
        let world_p = world_transform.transform_point3(local_p);
        transformed_positions.push(world_p.x);
        transformed_positions.push(world_p.y);
        transformed_positions.push(world_p.z);
    }

    let part_info = ffi::register_triangle_mesh(name, &transformed_positions, &mesh.indices);
    if let Some(obj) = build_loaded_object_from_part(part_info, archive_label) {
        report.objects.push(obj);
    }
}

fn load_embedded_step_bytes(
    entry_name: &str,
    step_bytes: &[u8],
    archive_label: &str,
    report: &mut LoadedBatchReport,
) -> Option<Vec<LoadedObjectData>> {
    let safe_filename = entry_name.replace(['/', '\\', ':'], "_");
    let temp_path = std::env::temp_dir().join(format!(
        "manifold2_3mf_{}_{}",
        std::process::id(),
        safe_filename
    ));
    if std::fs::write(&temp_path, step_bytes).is_err() {
        return None;
    }

    let loaded = load_step_file_parts(
        &temp_path,
        &format!("{archive_label}::{entry_name}"),
        report,
    );
    let _ = std::fs::remove_file(&temp_path);
    Some(loaded)
}

/// Saves the current build plate arrangement, meshes, embedded STEP B-Rep geometry
/// (when available), and `SlicerConfig` into a standards-compliant `.3mf` archive.
pub fn save_project_to_3mf(
    out_path: &Path,
    objects: &[ExportableObject],
    config: &SlicerConfig,
) -> Result<(), String> {
    let file =
        File::create(out_path).map_err(|e| format!("Failed to create {}: {e}", out_path.display()))?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

    // 1. Write `[Content_Types].xml`
    let content_types_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
  <Default Extension="toml" ContentType="application/toml"/>
  <Default Extension="step" ContentType="application/step"/>
</Types>"#;
    zip.start_file("[Content_Types].xml", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(content_types_xml.as_bytes())
        .map_err(|e| e.to_string())?;

    // 2. Write `_rels/.rels`
    let rels_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"#;
    zip.start_file("_rels/.rels", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(rels_xml.as_bytes())
        .map_err(|e| e.to_string())?;

    // 3. Write `3D/3dmodel.model` containing all object meshes and their 3x4 `<item transform="..."/>`
    let mut model_xml = String::with_capacity(65536);
    model_xml.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Application">Manifold2 Non-Planar Slicer</metadata>
  <resources>
"#,
    );

    for (idx, obj) in objects.iter().enumerate() {
        let xml_id = idx + 1;
        let escaped_name = escape_xml_attr(&obj.name);
        model_xml.push_str(&format!(
            "    <object id=\"{xml_id}\" name=\"{escaped_name}\" type=\"model\">\n      <mesh>\n        <vertices>\n"
        ));
        for v in &obj.vertices {
            model_xml.push_str(&format!(
                "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
                v.x, v.y, v.z
            ));
        }
        model_xml.push_str("        </vertices>\n        <triangles>\n");
        for tri in obj.indices.chunks_exact(3) {
            model_xml.push_str(&format!(
                "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
                tri[0], tri[1], tri[2]
            ));
        }
        model_xml.push_str("        </triangles>\n      </mesh>\n    </object>\n");
    }

    model_xml.push_str("  </resources>\n  <build>\n");
    for (idx, obj) in objects.iter().enumerate() {
        let xml_id = idx + 1;
        let transform_str = format_3mf_transform(&obj.transform);
        model_xml.push_str(&format!(
            "    <item objectid=\"{xml_id}\" transform=\"{transform_str}\"/>\n"
        ));
    }
    model_xml.push_str("  </build>\n</model>\n");

    zip.start_file("3D/3dmodel.model", options)
        .map_err(|e| e.to_string())?;
    zip.write_all(model_xml.as_bytes())
        .map_err(|e| e.to_string())?;

    // 4. Embed `/Metadata/manifold2_config.toml`
    let config_toml = config
        .to_toml_string()
        .map_err(|e| format!("Failed to serialize SlicerConfig: {e}"))?;
    zip.start_file(MANIFOLD_CONFIG_ENTRY, options)
        .map_err(|e| e.to_string())?;
    zip.write_all(config_toml.as_bytes())
        .map_err(|e| e.to_string())?;

    // 5. Export any B-Rep STEP shapes into `/Metadata/cad_<id>.step` and write `manifold2_scene.toml`
    let mut scene_manifest = SavedSceneManifest::default();
    for obj in objects {
        let mut embedded_step_entry = None;
        if obj.has_brep {
            let temp_step = std::env::temp_dir().join(format!(
                "manifold2_export_{}_{}.step",
                std::process::id(),
                obj.object_id
            ));
            let temp_str = temp_step.to_string_lossy();
            if ffi::export_object_step_to_path(obj.object_id, &temp_str) {
                if let Ok(step_bytes) = std::fs::read(&temp_step) {
                    let entry_path = format!("Metadata/cad_{}.step", obj.object_id);
                    if zip.start_file(&entry_path, options).is_ok()
                        && zip.write_all(&step_bytes).is_ok()
                    {
                        embedded_step_entry = Some(entry_path);
                    }
                }
            }
            let _ = std::fs::remove_file(&temp_step);
        }

        scene_manifest.objects.push(SavedObjectEntry {
            name: obj.name.clone(),
            embedded_step_entry,
            translation: obj.transform.translation.to_array(),
            rotation_xyzw: obj.transform.rotation.to_array(),
        });
    }

    if let Ok(scene_toml) = toml::to_string_pretty(&scene_manifest) {
        zip.start_file(MANIFOLD_SCENE_ENTRY, options)
            .map_err(|e| e.to_string())?;
        zip.write_all(scene_toml.as_bytes())
            .map_err(|e| e.to_string())?;
    }

    zip.finish().map_err(|e| e.to_string())?;
    Ok(())
}

/// Parses a 3MF `.model` XML buffer using `quick-xml`, extracting `<object>` meshes,
/// `<components>`, and `<build><item>` entries while ignoring unknown slicer tags.
fn parse_3mf_model_xml(xml_bytes: &[u8]) -> Result<ParsedModelFile, String> {
    let mut reader = Reader::from_reader(xml_bytes);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut model_file = ParsedModelFile::default();
    let mut current_object: Option<ParsedXmlObject> = None;
    let mut current_mesh: Option<ParsedXmlMesh> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                let qname = e.name();
                let local = local_tag_name(qname.as_ref());
                match local {
                    b"object" => {
                        let mut id = String::new();
                        let mut name = None;
                        for attr in e.attributes().flatten() {
                            match local_tag_name(attr.key.as_ref()) {
                                b"id" => {
                                    id = String::from_utf8_lossy(&attr.value).into_owned();
                                }
                                b"name" => {
                                    name = Some(String::from_utf8_lossy(&attr.value).into_owned());
                                }
                                _ => {}
                            }
                        }
                        current_object = Some(ParsedXmlObject {
                            id,
                            name,
                            mesh: None,
                            components: Vec::new(),
                        });
                    }
                    b"mesh" => {
                        current_mesh = Some(ParsedXmlMesh::default());
                    }
                    b"vertex" => {
                        if let Some(ref mut mesh) = current_mesh {
                            parse_vertex_element(e, mesh);
                        }
                    }
                    b"triangle" => {
                        if let Some(ref mut mesh) = current_mesh {
                            parse_triangle_element(e, mesh);
                        }
                    }
                    b"component" => {
                        if let Some(ref mut obj) = current_object {
                            if let Some(comp) = parse_component_element(e) {
                                obj.components.push(comp);
                            }
                        }
                    }
                    b"item" => {
                        if let Some(item) = parse_build_item_element(e) {
                            model_file.build_items.push(item);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(ref e)) => {
                let qname = e.name();
                let local = local_tag_name(qname.as_ref());
                match local {
                    b"vertex" => {
                        if let Some(ref mut mesh) = current_mesh {
                            parse_vertex_element(e, mesh);
                        }
                    }
                    b"triangle" => {
                        if let Some(ref mut mesh) = current_mesh {
                            parse_triangle_element(e, mesh);
                        }
                    }
                    b"component" => {
                        if let Some(ref mut obj) = current_object {
                            if let Some(comp) = parse_component_element(e) {
                                obj.components.push(comp);
                            }
                        }
                    }
                    b"item" => {
                        if let Some(item) = parse_build_item_element(e) {
                            model_file.build_items.push(item);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let qname = e.name();
                let local = local_tag_name(qname.as_ref());
                match local {
                    b"mesh" => {
                        if let (Some(ref mut obj), Some(mesh)) =
                            (&mut current_object, current_mesh.take())
                        {
                            obj.mesh = Some(mesh);
                        }
                    }
                    b"object" => {
                        if let Some(obj) = current_object.take() {
                            if !obj.id.is_empty() {
                                model_file.objects.insert(obj.id.clone(), obj);
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML error at position {}: {e}", reader.error_position())),
            _ => {}
        }
        buf.clear();
    }

    Ok(model_file)
}

fn local_tag_name(raw: &[u8]) -> &[u8] {
    if let Some(pos) = raw.iter().rposition(|&b| b == b':') {
        &raw[pos + 1..]
    } else {
        raw
    }
}

fn parse_vertex_element(e: &quick_xml::events::BytesStart<'_>, mesh: &mut ParsedXmlMesh) {
    let mut x = 0.0_f32;
    let mut y = 0.0_f32;
    let mut z = 0.0_f32;
    for attr in e.attributes().flatten() {
        let val_str = std::str::from_utf8(&attr.value).unwrap_or("0");
        match local_tag_name(attr.key.as_ref()) {
            b"x" => x = val_str.parse().unwrap_or(0.0),
            b"y" => y = val_str.parse().unwrap_or(0.0),
            b"z" => z = val_str.parse().unwrap_or(0.0),
            _ => {}
        }
    }
    mesh.positions.push(x);
    mesh.positions.push(y);
    mesh.positions.push(z);
}

fn parse_triangle_element(e: &quick_xml::events::BytesStart<'_>, mesh: &mut ParsedXmlMesh) {
    let mut v1 = 0_u32;
    let mut v2 = 0_u32;
    let mut v3 = 0_u32;
    for attr in e.attributes().flatten() {
        let val_str = std::str::from_utf8(&attr.value).unwrap_or("0");
        match local_tag_name(attr.key.as_ref()) {
            b"v1" => v1 = val_str.parse().unwrap_or(0),
            b"v2" => v2 = val_str.parse().unwrap_or(0),
            b"v3" => v3 = val_str.parse().unwrap_or(0),
            _ => {}
        }
    }
    mesh.indices.push(v1);
    mesh.indices.push(v2);
    mesh.indices.push(v3);
}

fn parse_component_element(e: &quick_xml::events::BytesStart<'_>) -> Option<ParsedComponentRef> {
    let mut object_id = String::new();
    let mut model_path = None;
    let mut transform = Mat4::IDENTITY;

    for attr in e.attributes().flatten() {
        let val_str = String::from_utf8_lossy(&attr.value);
        match local_tag_name(attr.key.as_ref()) {
            b"objectid" => object_id = val_str.into_owned(),
            b"path" => model_path = Some(val_str.into_owned()),
            b"transform" => transform = parse_3mf_transform(&val_str),
            _ => {}
        }
    }

    if object_id.is_empty() {
        None
    } else {
        Some(ParsedComponentRef {
            object_id,
            model_path,
            transform,
        })
    }
}

fn parse_build_item_element(e: &quick_xml::events::BytesStart<'_>) -> Option<ParsedBuildItem> {
    let mut object_id = String::new();
    let mut model_path = None;
    let mut transform = Mat4::IDENTITY;

    for attr in e.attributes().flatten() {
        let val_str = String::from_utf8_lossy(&attr.value);
        match local_tag_name(attr.key.as_ref()) {
            b"objectid" => object_id = val_str.into_owned(),
            b"path" => model_path = Some(val_str.into_owned()),
            b"transform" => transform = parse_3mf_transform(&val_str),
            _ => {}
        }
    }

    if object_id.is_empty() {
        None
    } else {
        Some(ParsedBuildItem {
            object_id,
            model_path,
            transform,
        })
    }
}

/// Parses a 3MF 12-float affine transformation string (`m00 m01 m02 m10 m11 m12 m20 m21 m22 tx ty tz`)
/// into a column-major `Mat4`.
pub fn parse_3mf_transform(s: &str) -> Mat4 {
    let vals: Vec<f32> = s
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f32>().ok())
        .collect();
    if vals.len() != 12 {
        return Mat4::IDENTITY;
    }

    // In 3MF Core Specification, a row vector [x y z 1] is multiplied on the right
    // by the 4x3 matrix:
    //   [ m00 m01 m02 0 ]
    //   [ m10 m11 m12 0 ]
    //   [ m20 m21 m22 0 ]
    //   [ tx  ty  tz  1 ]
    // Which corresponds to column vectors in glam/Bevy `Mat4::from_cols`:
    Mat4::from_cols(
        Vec4::new(vals[0], vals[1], vals[2], 0.0),
        Vec4::new(vals[3], vals[4], vals[5], 0.0),
        Vec4::new(vals[6], vals[7], vals[8], 0.0),
        Vec4::new(vals[9], vals[10], vals[11], 1.0),
    )
}

/// Formats a Bevy `Transform` into the 3MF 12-float affine matrix string.
pub fn format_3mf_transform(transform: &Transform) -> String {
    let mat = transform.compute_matrix();
    let c0 = mat.col(0);
    let c1 = mat.col(1);
    let c2 = mat.col(2);
    let c3 = mat.col(3);
    format!(
        "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
        c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z, c3.x, c3.y, c3.z
    )
}

fn escape_xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_3mf_transform_roundtrip() {
        let t = Transform {
            translation: Vec3::new(42.5, -18.0, 0.0),
            rotation: Quat::from_rotation_z(0.75) * Quat::from_rotation_x(0.3),
            scale: Vec3::ONE,
        };
        let encoded = format_3mf_transform(&t);
        let decoded = parse_3mf_transform(&encoded);
        let expected = t.compute_matrix();

        for col in 0..4 {
            let diff = (decoded.col(col) - expected.col(col)).length();
            assert!(diff < 1e-4, "Column {col} mismatch: {diff}");
        }
    }

    #[test]
    fn test_parse_3mf_model_with_unknown_third_party_tags() {
        let sample_xml = br#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">
  <metadata name="slic3rpe:Version3mf">1</metadata>
  <resources>
    <object id="1" name="TestTetra" type="model">
      <slic3rpe:custom_tag foo="bar"/>
      <mesh>
        <vertices>
          <vertex x="0" y="0" z="0"/>
          <vertex x="10" y="0" z="0"/>
          <vertex x="0" y="10" z="0"/>
          <vertex x="0" y="0" z="10"/>
        </vertices>
        <triangles>
          <triangle v1="0" v2="2" v3="1"/>
          <triangle v1="0" v2="1" v3="3"/>
          <triangle v1="0" v2="3" v3="2"/>
          <triangle v1="1" v2="2" v3="3"/>
        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" transform="1 0 0 0 1 0 0 0 1 15 -20 0"/>
  </build>
</model>"#;

        let parsed = parse_3mf_model_xml(sample_xml).expect("Should parse 3MF XML cleanly");
        assert_eq!(parsed.objects.len(), 1);
        assert_eq!(parsed.build_items.len(), 1);
        let obj = parsed.objects.get("1").unwrap();
        assert_eq!(obj.name.as_deref(), Some("TestTetra"));
        let mesh = obj.mesh.as_ref().unwrap();
        assert_eq!(mesh.positions.len(), 12);
        assert_eq!(mesh.indices.len(), 12);
    }

    #[test]
    fn test_manifold_3mf_project_filename_recalled_on_load() {
        let temp_3mf = std::env::temp_dir().join(format!(
            "custom_manifold_project_{}.3mf",
            std::process::id()
        ));
        let expected_filename = temp_3mf
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap()
            .to_string();

        let export_objs = vec![ExportableObject {
            object_id: 1,
            name: "Tetra".to_string(),
            has_brep: false,
            transform: Transform::IDENTITY,
            vertices: vec![
                RenderVertex { x: 0.0, y: 0.0, z: 0.0, nx: 0.0, ny: 0.0, nz: -1.0 },
                RenderVertex { x: 10.0, y: 0.0, z: 0.0, nx: 0.0, ny: 0.0, nz: -1.0 },
                RenderVertex { x: 0.0, y: 10.0, z: 0.0, nx: 0.0, ny: 0.0, nz: -1.0 },
                RenderVertex { x: 0.0, y: 0.0, z: 10.0, nx: 0.0, ny: 0.0, nz: 1.0 },
            ],
            indices: vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
        }];
        let cfg = SlicerConfig::default();
        save_project_to_3mf(&temp_3mf, &export_objs, &cfg).expect("Should write Manifold 3MF");

        let mut report = LoadedBatchReport::default();
        load_3mf_archive(&temp_3mf, &mut report);
        let _ = std::fs::remove_file(&temp_3mf);

        assert_eq!(
            report.manifold_project_filename.as_deref(),
            Some(expected_filename.as_str())
        );
        assert!(report.restored_config.is_some());
        for obj in &report.objects {
            let _ = ffi::remove_cad_object(obj.object_id);
        }
    }
}
