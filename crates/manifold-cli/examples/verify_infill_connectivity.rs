//! Scratch verification for the infill connected-component grouping fix in
//! `manifold_core::infill` (monotonic fill now groups scan-line spans into
//! connected regions via row-by-row union-find with a tolerant overlap
//! margin, instead of a single strictly-ascending path across the whole
//! layer). Not part of the crate's public surface or test suite -- run
//! manually against a real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_infill_connectivity -- pug_v4_l_sop_85mm.stl
//! ```
//!
//! Reports, per layer, the number of infill `Path`s and the average
//! points-per-path. A regression to the old exact-overlap behavior
//! fragmented infill into thousands of tiny (few-point) paths per layer;
//! this check fails loudly if that happens again.

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::tool::Tool;
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    // Effectively unbounded build volume, same rationale as the other
    // verify_* scratch examples: isolates this check from the unrelated
    // build-volume-placement issue affecting the real pug mesh.
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let config = SlicerConfig::default();

    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;

    // Group paths by (approximate) layer via their Z -- paths don't carry
    // an explicit layer index, so bucket by rounded Z instead.
    use std::collections::BTreeMap;
    let mut by_layer: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for path in &paths {
        let is_infill = path.segments.iter().any(|s| s.kind == MoveKind::Infill);
        if !is_infill {
            continue;
        }
        let Some(&first) = path.points.first() else {
            continue;
        };
        let z_key = (first.z * 1000.0).round() as i64;
        by_layer.entry(z_key).or_default().push(path.points.len());
    }

    if by_layer.is_empty() {
        println!("RESULT: FAIL -- no infill paths found at all.");
        std::process::exit(1);
    }

    let mut worst_path_count = 0usize;
    let mut worst_layer_z = 0i64;
    let mut total_paths = 0usize;
    let mut total_points = 0usize;
    let mut fragmented_layers = 0usize;

    for (&z_key, point_counts) in &by_layer {
        let path_count = point_counts.len();
        let points_sum: usize = point_counts.iter().sum();
        let avg_points = points_sum as f64 / path_count as f64;

        total_paths += path_count;
        total_points += points_sum;

        // The old bug fragmented one part's infill from ~400 paths/layer
        // into ~7,600, each just a few points long -- flag any layer with
        // a large path count AND a tiny average points-per-path as
        // suspected fragmentation.
        if path_count > 1000 && avg_points < 10.0 {
            fragmented_layers += 1;
        }

        if path_count > worst_path_count {
            worst_path_count = path_count;
            worst_layer_z = z_key;
        }
    }

    let overall_avg_points = total_points as f64 / total_paths as f64;
    println!(
        "layers_with_infill={} total_infill_paths={total_paths} \
         overall_avg_points_per_path={overall_avg_points:.1} \
         worst_layer_z={:.3}mm worst_layer_path_count={worst_path_count}",
        by_layer.len(),
        worst_layer_z as f64 / 1000.0,
    );

    if fragmented_layers > 0 {
        println!(
            "RESULT: FAIL -- {fragmented_layers} layer(s) show suspected infill \
             fragmentation (>1000 paths and <10 avg points/path)."
        );
        std::process::exit(1);
    }

    println!(
        "RESULT: PASS -- infill paths are grouped into reasonably sized \
         connected-component runs, no fragmentation detected."
    );
    Ok(())
}
