//! Scratch verification for travel-move ordering optimization
//! (`toolpath::optimize_travel_order`). Not part of the crate's public
//! surface or test suite -- run manually against a real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_travel_optimization -- pug_v4_l_sop_85mm.stl
//! ```
//!
//! Slices the same mesh twice, once with
//! `travel_order_optimization_enabled: false` and once with the default
//! (`true`), and reports total travel (non-extruding G0) distance for
//! both -- confirming the optimization pass materially reduces travel
//! without changing which paths/points are printed (same extruding
//! distance/point count either way, just reordered/reversed).

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::tool::Tool;
use manifold_core::toolpath::{MoveKind, Path};
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

fn travel_and_extrude_distance(paths: &[Path]) -> (f64, f64, usize) {
    let mut travel = 0.0;
    let mut extrude = 0.0;
    let mut point_count = 0usize;
    for path in paths {
        point_count += path.points.len();
        for (i, segment) in path.segments.iter().enumerate() {
            let Some(&a) = path.points.get(i) else {
                continue;
            };
            let Some(&b) = path.points.get(i + 1) else {
                continue;
            };
            let d = a.distance(b);
            if segment.kind == MoveKind::Travel {
                travel += d;
            } else {
                extrude += d;
            }
        }
        // The very first move of each path is always a plain positioning
        // move (see `gcode::emit`'s doc comment) -- account for it here
        // too, using the path's own first segment's *arrival* classification
        // is not applicable (it has no incoming segment), so treat the
        // very first hop from wherever the previous path left off
        // separately, in `main`, where consecutive paths are known.
    }
    (travel, extrude, point_count)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    // Effectively unbounded build volume so this scratch check isolates
    // travel-distance behavior from the unrelated build-volume-placement
    // issue (see verify_flat_nozzle.rs's identical rationale).
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let mut results = Vec::new();
    for (label, enabled) in [("disabled (baseline)", false), ("enabled", true)] {
        let config = SlicerConfig {
            travel_order_optimization_enabled: enabled,
            ..SlicerConfig::default()
        };

        let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
        let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;

        // Inter-path travel: the plain positioning move from one path's
        // last point to the next path's first point (this is where
        // "long travel move to make a tiny extrusion then travel back"
        // shows up -- it's invisible to `travel_and_extrude_distance`,
        // which only sees *within*-path segments).
        let mut inter_path_travel = 0.0;
        for window in paths.windows(2) {
            if let (Some(&last), Some(&first)) = (window[0].points.last(), window[1].points.first())
            {
                inter_path_travel += last.distance(first);
            }
        }

        let (intra_travel, extrude, point_count) = travel_and_extrude_distance(&paths);
        let total_travel = intra_travel + inter_path_travel;

        println!(
            "{label}: paths={} points={point_count} extrude_distance={extrude:.2}mm \
             intra_path_travel={intra_travel:.2}mm inter_path_travel={inter_path_travel:.2}mm \
             total_travel={total_travel:.2}mm",
            paths.len()
        );

        results.push((label, total_travel, extrude, point_count));
    }

    let (_, baseline_travel, baseline_extrude, baseline_points) = results[0];
    let (_, optimized_travel, optimized_extrude, optimized_points) = results[1];

    println!();
    if baseline_points != optimized_points {
        println!(
            "RESULT: FAIL -- point count changed ({baseline_points} -> {optimized_points}); \
             optimization must not add/drop geometry."
        );
        std::process::exit(1);
    }
    if (baseline_extrude - optimized_extrude).abs() > 1e-6 * baseline_extrude.max(1.0) {
        println!(
            "RESULT: FAIL -- extrude distance changed ({baseline_extrude:.3}mm -> \
             {optimized_extrude:.3}mm); optimization must only reorder/reverse paths, \
             not change printed geometry."
        );
        std::process::exit(1);
    }
    if optimized_travel >= baseline_travel {
        println!(
            "RESULT: FAIL -- optimized travel ({optimized_travel:.2}mm) did not improve on \
             baseline ({baseline_travel:.2}mm)."
        );
        std::process::exit(1);
    }

    let reduction_pct = 100.0 * (baseline_travel - optimized_travel) / baseline_travel;
    println!(
        "RESULT: PASS -- travel reduced from {baseline_travel:.2}mm to {optimized_travel:.2}mm \
         ({reduction_pct:.1}% reduction), extrude distance and point count unchanged."
    );
    Ok(())
}
