//! Scratch diagnostic for a regression introduced by the
//! `has_solid_support_one_layer_back` veto in
//! `slicing::stitch_wall_gaps`: a genuine overhang near the max-X end of
//! `pug_v4_m_sop_85mm.stl` (around order ~5.5, using the user's real
//! `profile.json`) stopped being generated once `shell_thickness` is 2
//! wall lines (0.8mm) instead of 1 (0.4mm). Not part of the crate's
//! public surface or test suite -- run manually:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example diagnose_missing_overhang -- \
//!     pug_v4_m_sop_85mm.stl /Users/amcgregor/3D/profile.json
//! ```
//!
//! Loads the real machine+config profile (mirroring `manifold-gui`'s
//! `Profile` shape) via `serde_json`, slices with `shell_thickness` as
//! given and again forced to one wall line, and reports `Overhang`
//! segment counts + locations for both, focused on the max-X region.

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

fn slice_and_report(
    label: &str,
    mesh: &manifold_core::mesh::Mesh,
    machine: &Machine,
    config: &SlicerConfig,
) -> anyhow::Result<()> {
    let object = Object::new(ObjectId(0), mesh.clone(), ToolId(0));
    let objects = vec![object];
    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], config)?;
    let paths = toolpath::plan(&layers, &objects, &machine.tools, config)?;

    let (min, max) = mesh.bounding_box().expect("non-empty mesh");
    let max_x_band = max.x - 0.15 * (max.x - min.x);

    let mut total_overhang = 0usize;
    let mut near_max_x_overhang = 0usize;
    let mut near_order_5_5 = 0usize;
    let mut sample_points: Vec<DVec3> = Vec::new();

    for path in &paths {
        let n = path.points.len();
        for (i, seg) in path.segments.iter().enumerate() {
            if seg.kind != MoveKind::Overhang {
                continue;
            }
            let start = path.points[i];
            let end = path.points[(i + 1) % n];
            total_overhang += 1;
            if start.x >= max_x_band || end.x >= max_x_band {
                near_max_x_overhang += 1;
                if (seg.order - 5.5).abs() < 0.6 {
                    near_order_5_5 += 1;
                    if sample_points.len() < 5 {
                        sample_points.push(start);
                    }
                }
            }
        }
    }

    println!("--- {label} ---");
    println!(
        "shell_thickness={:.2} wall_count={}",
        config.shell_thickness,
        config.wall_count()
    );
    println!("mesh bbox: min={min:.3?} max={max:.3?} (max_x_band threshold={max_x_band:.3})");
    println!("total Overhang segments: {total_overhang}");
    println!("Overhang segments near max-X band: {near_max_x_overhang}");
    println!("...of those, near order~5.5 (+/-0.6): {near_order_5_5}");
    for p in &sample_points {
        println!("  sample overhang point: {p:.3?}");
    }
    println!();
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let stl_path = args.next().expect("usage: <stl path> <profile.json path>");
    let profile_path = args.next().expect("usage: <stl path> <profile.json path>");

    let file = std::fs::File::open(&stl_path)?;
    let mesh = stl::load_stl(file)?;

    let profile_json: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(&profile_path)?)?;
    let mut machine: Machine = serde_json::from_value(profile_json["machine"].clone())?;
    let mut config: SlicerConfig = serde_json::from_value(profile_json["config"].clone())?;

    // Effectively unbounded build volume so this scratch check isolates
    // overhang-generation behavior from the unrelated build-volume
    // validation step (see verify_wall_gap_stitching.rs for the same
    // workaround).
    machine.build_volume = BoundingVolume::Aabb {
        min: DVec3::splat(-1_000.0),
        max: DVec3::splat(1_000.0),
    };

    println!("slicing {stl_path} with real profile {profile_path}...");
    println!();

    // As-configured (shell_thickness=0.8mm, i.e. 2 wall lines per the
    // user's profile.json).
    slice_and_report("as-configured (2 walls)", &mesh, &machine, &config)?;

    // Forced to a single wall line (shell_thickness == wall_line_width),
    // reproducing the user's "it works with 0.4mm shell thickness"
    // comparison case.
    config.shell_thickness = config.wall_line_width;
    slice_and_report("forced single wall (1 wall)", &mesh, &machine, &config)?;

    Ok(())
}
