//! Scratch probe: slice an STL with the Eikonal order field and dump each
//! layer's loop statistics, flagging layers whose wall-0 geometry is
//! missing or suspiciously thin relative to neighbors (topology-change
//! dropouts). Run manually:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_layer_dropouts -- Voron_Design_Cube_v7.stl 15
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::toolpath::{self, MoveKind};
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;
use manifold_fidget::ScalarField;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Voron_Design_Cube_v7.stl".to_string());
    let degrees: f64 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("degrees must be a number"))
        .unwrap_or(15.0);

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];
    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        ..SlicerConfig::default()
    };
    let slope_profile = SlopeProfile::new(vec![(0.0, degrees)]);

    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!(
        "sliced {} layer(s), layer_height {}",
        layers.len(),
        config.layer_height
    );

    let mut prev_order: Option<f64> = None;
    for layer in &layers {
        // Per-wall loop count and point count.
        let mut wall_stats: Vec<(usize, usize, usize)> = Vec::new(); // (wall_index, loops, pts)
        for l in &layer.loops {
            if let Some(s) = wall_stats.iter_mut().find(|s| s.0 == l.wall_index) {
                s.1 += 1;
                s.2 += l.points.len();
            } else {
                wall_stats.push((l.wall_index, 1, l.points.len()));
            }
        }
        wall_stats.sort_by_key(|s| s.0);

        // Loop bbox to see where geometry lives.
        let (mut lmin, mut lmax) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
        for l in &layer.loops {
            for &p in &l.points {
                lmin = lmin.min(p);
                lmax = lmax.max(p);
            }
        }

        let gap = prev_order
            .map(|p| layer.order - p)
            .unwrap_or(config.layer_height);
        let gap_flag = if (gap - config.layer_height).abs() > 1e-6 {
            " <-- ORDER GAP"
        } else {
            ""
        };
        let empty_flag = if layer.loops.is_empty() {
            " <-- EMPTY"
        } else {
            ""
        };

        let stats: Vec<String> = wall_stats
            .iter()
            .map(|(w, n, p)| format!("w{w}:{n}loops/{p}pts"))
            .collect();
        println!(
            "layer {:3} order {:7.3} | {} | z[{:.1},{:.1}]{}{}",
            layer.index,
            layer.order,
            if stats.is_empty() {
                "-".to_string()
            } else {
                stats.join(" ")
            },
            lmin.z,
            lmax.z,
            gap_flag,
            empty_flag,
        );
        prev_order = Some(layer.order);
    }

    // Now run toolpath planning and compare per-layer wall-0 loop counts
    // before vs after: retain_contained_paths drops whole loops when any
    // point pokes outside the mesh SDF, which would be invisible in the
    // slicing-layer stats above. Dropped loops also emit tracing warnings.
    println!("\nplanning toolpaths...");
    let tools = vec![manifold_core::tool::Tool::new(ToolId(0), 0.4)];
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        &config,
        &slope_profile,
        &mut |_| {},
    )?;

    // Wall-0 paths per layer order (match on the segments' stamped order).
    for layer in &layers {
        let sliced_wall0 = layer.loops.iter().filter(|l| l.wall_index == 0).count();
        let planned_wall0 = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .any(|s| s.kind == MoveKind::WallOuter && (s.order - layer.order).abs() < 1e-9)
            })
            .count();
        if planned_wall0 < sliced_wall0 {
            // Which sliced loops were dropped? Match sliced loops to planned
            // paths by first point; report point counts so we can tell real
            // walls (thousands of points) from spurious fragments.
            let mut dropped_sizes: Vec<usize> = Vec::new();
            for l in layer.loops.iter().filter(|l| l.wall_index == 0) {
                let survived = paths.iter().any(|p| {
                    p.segments.iter().any(|s| {
                        s.kind == MoveKind::WallOuter && (s.order - layer.order).abs() < 1e-9
                    }) && p
                        .points
                        .first()
                        .is_some_and(|&q| (q - l.points[0]).length() < 0.3)
                });
                if !survived {
                    dropped_sizes.push(l.points.len());
                }
            }
            println!(
                "layer {:3} order {:7.3}: {} wall-0 loops sliced but only {} planned <-- DROPPED sizes {:?}",
                layer.index, layer.order, sliced_wall0, planned_wall0, dropped_sizes
            );
        }
    }
    // Measure how far outside the mesh SDF the worst point of each sliced
    // wall loop sits: retain_contained_paths uses a 1e-6 epsilon, so any
    // loop whose max signed distance exceeds that gets dropped wholesale.
    println!("\nwall-loop max signed distance distribution (mm):");
    let mut worst_overall: f64 = f64::NEG_INFINITY;
    let mut histogram = [0usize; 7]; // <=0, <=1e-6, <=0.01, <=0.05, <=0.1, <=0.2, >0.2
    for layer in &layers {
        let Some(sdf) = layer.mesh_sdf.as_ref() else {
            continue;
        };
        for l in &layer.loops {
            let max_d = l
                .points
                .iter()
                .map(|&p| sdf.sample(p).value)
                .fold(f64::NEG_INFINITY, f64::max);
            worst_overall = worst_overall.max(max_d);
            let bucket = if max_d <= 0.0 {
                0
            } else if max_d <= 1e-6 {
                1
            } else if max_d <= 0.01 {
                2
            } else if max_d <= 0.05 {
                3
            } else if max_d <= 0.1 {
                4
            } else if max_d <= 0.2 {
                5
            } else {
                6
            };
            histogram[bucket] += 1;
            if max_d > 0.2 {
                let worst_pt = l
                    .points
                    .iter()
                    .max_by(|a, b| sdf.sample(**a).value.total_cmp(&sdf.sample(**b).value))
                    .unwrap();
                println!(
                    "  layer {:3} order {:7.3} wall{} loop ({} pts): max {:+.3} mm at ({:.2}, {:.2}, {:.2})",
                    layer.index, layer.order, l.wall_index, l.points.len(),
                    max_d, worst_pt.x, worst_pt.y, worst_pt.z
                );
            }
        }
    }
    println!(
        "buckets: <=0: {}, <=1e-6: {}, <=0.01: {}, <=0.05: {}, <=0.1: {}, <=0.2: {}, >0.2: {}; worst {:+.3} mm",
        histogram[0], histogram[1], histogram[2], histogram[3], histogram[4], histogram[5], histogram[6],
        worst_overall
    );

    println!("done");
    Ok(())
}
