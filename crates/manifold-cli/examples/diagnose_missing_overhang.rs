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

    let mut total_overhang = 0usize;
    let mut overhang_by_order: std::collections::BTreeMap<i32, usize> =
        std::collections::BTreeMap::new();
    for path in &paths {
        for seg in &path.segments {
            if seg.kind == MoveKind::Overhang {
                total_overhang += 1;
                let order_key = (seg.order * 100.0).round() as i32;
                *overhang_by_order.entry(order_key).or_insert(0) += 1;
            }
        }
    }
    println!("--- {label} ---");
    println!("Total Overhang segments across whole print: {total_overhang}");
    if !overhang_by_order.is_empty() {
        println!("Overhang segments by order:");
        for (order_key, count) in overhang_by_order {
            println!(
                "  Order {:.2}: {} segments",
                order_key as f64 / 100.0,
                count
            );
        }
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
