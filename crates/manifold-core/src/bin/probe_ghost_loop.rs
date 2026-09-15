//! Scratch diagnostic: dumps every planned `Path` touching a world-space box,
//! to check for the "ghost sheet" duplicate-isosurface-loop bug (a second,
//! near-flat WallInner/Infill loop at the same order value/wall_index as a
//! real non-planar wall), and reports each path's own deviation from its
//! target order value on the real order field.
//! Usage: cargo run --release --bin probe_ghost_loop -- <mesh.stl> <profile.json>

use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::{ordering, plan_toolpaths, slicing, SlicerConfig, Workspace};
use std::io::BufReader;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mesh_path = &args[1];
    let profile_path = &args[2];

    let json = std::fs::read_to_string(profile_path).expect("read profile");
    let profile: Profile = serde_json::from_str(&json).expect("parse profile");

    let file = std::fs::File::open(mesh_path).expect("open mesh");
    let mesh = manifold_core::stl::load_stl(BufReader::new(file)).expect("parse stl");
    let mut objects = vec![Object::new(
        manifold_core::ids::ObjectId(0),
        mesh,
        manifold_core::ids::ToolId(0),
    )];
    let mut machine = profile.machine;
    center_on_bed(&mut objects, &machine.build_volume);
    let _ = &mut machine;

    let workspace = Workspace::new(objects, machine, profile.config);
    let paths = plan_toolpaths(&workspace).expect("plan toolpaths");

    let strategy = ordering::strategy_for(workspace.config.object_ordering);
    let order = strategy.order(&workspace.objects).expect("order");
    let layers =
        slicing::slice_workspace(&workspace.objects, &order, &workspace.config).expect("slice");

    let (xmin, xmax) = (145.0, 170.0);
    let (ymin, ymax) = (175.0, 185.0);
    let (zmin, zmax) = (0.70, 0.82);

    for (i, path) in paths.iter().enumerate() {
        let in_box = path.points.iter().any(|p| {
            p.x >= xmin && p.x <= xmax && p.y >= ymin && p.y <= ymax && p.z >= zmin && p.z <= zmax
        });
        if !in_box {
            continue;
        }
        let zs: Vec<f64> = path.points.iter().map(|p| p.z).collect();
        let zmin_p = zs.iter().cloned().fold(f64::INFINITY, f64::min);
        let zmax_p = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let kind = path.segments.first().map(|s| s.kind);
        let target_order = path.segments.first().map(|s| s.order);

        let dev_report = target_order.map(|target| {
            let layer = layers
                .iter()
                .min_by(|a, b| {
                    (a.order - target)
                        .abs()
                        .total_cmp(&(b.order - target).abs())
                })
                .expect("some layer");
            let devs: Vec<f64> = path
                .points
                .iter()
                .map(|p| (layer.order_field.order(*p) - target).abs())
                .collect();
            let max_dev = devs.iter().cloned().fold(0.0, f64::max);
            let mean_dev = devs.iter().sum::<f64>() / devs.len().max(1) as f64;
            (mean_dev, max_dev)
        });

        eprintln!(
            "path[{i}]: kind={:?} target_order={:?} segs={} pts={} z=[{:.6},{:.6}] dev(mean,max)={:?}",
            kind,
            target_order,
            path.segments.len(),
            path.points.len(),
            zmin_p,
            zmax_p,
            dev_report,
        );
    }

    // Verify seed_proximity() sanity: near-0 at bed contact and inside a
    // known patch-seeded top-surface cluster, larger away from both.
    if let Some(layer) = layers.first() {
        let field = layer.order_field.as_ref();
        let bed_point = glam::DVec3::new(155.0, 181.0, 0.01);
        eprintln!(
            "\nseed_proximity(bed contact z=0.01) = {:?}",
            field.seed_proximity(bed_point)
        );
        let mid_air = glam::DVec3::new(155.0, 181.0, 0.8);
        eprintln!(
            "seed_proximity(mid-height z=4.0, far from any seed) = {:?}",
            field.seed_proximity(mid_air)
        );
    }
    // Known patch-seeded top-surface cluster from probe_top_seeds output:
    // z=6.8 cx=33.331 cy=-0.000, area=29.35 (largest cluster).
    let patch_layer = layers
        .iter()
        .min_by(|a, b| (a.order - 6.8).abs().total_cmp(&(b.order - 6.8).abs()));
    if let Some(layer) = patch_layer {
        let patch_point = glam::DVec3::new(155.0, 181.0, 8.5);
        eprintln!(
            "seed_proximity(patch top-surface cluster z=6.8) = {:?} (layer order={:.4})",
            layer.order_field.seed_proximity(patch_point),
            layer.order
        );
    }
}
