//! Scratch diagnostic: for one layer (by target order value), measure SDF
//! distances at every stage of the sparse-infill pipeline to locate where
//! points escape the mesh solid:
//!   1. wall loops (layer.loops)
//!   2. slicing's infill_boundary / solid_fill_boundary
//!   3. sparse region loops BEFORE reconstruction (2D difference at ref height)
//!   4. sparse region loops AFTER reconstruct_on_order_field_near
//!   5. final Infill/TopSurface toolpath paths at that order
//!     plus vertical SDF/order profiles at the worst final points.
//!
//! Usage: cargo run --release --bin probe_layer_diag -- <mesh.stl> <profile.json> <target_order>

use manifold_core::machine::Machine;
use manifold_core::mesh::Mesh;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::MoveKind;
use manifold_core::{order_field, plan_toolpaths, polygon2d, slicing, SlicerConfig, Workspace};
use manifold_fidget::mesh_sdf::MeshSdf;
use manifold_fidget::{contour::plane_basis, ScalarField};
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
    let target_order: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(11.847);

    let json = std::fs::read_to_string(profile_path).expect("read profile");
    let profile: Profile = serde_json::from_str(&json).expect("parse profile");

    let file = std::fs::File::open(mesh_path).expect("open mesh");
    let mesh = manifold_core::stl::load_stl(BufReader::new(file)).expect("parse stl");
    let mut objects = vec![Object::new(
        manifold_core::ids::ObjectId(0),
        mesh,
        manifold_core::ids::ToolId(0),
    )];
    let machine = profile.machine;
    center_on_bed(&mut objects, &machine.build_volume);

    let workspace = Workspace::new(objects.clone(), machine, profile.config.clone());
    let paths = plan_toolpaths(&workspace).expect("plan toolpaths");

    let strategy = manifold_core::ordering::strategy_for(workspace.config.object_ordering);
    let order = strategy.order(&workspace.objects).expect("order");
    let layers = slicing::slice_workspace_with_progress(
        &workspace.objects,
        &order,
        &workspace.config,
        &workspace.machine.slope_profile(),
        &mut |_| {},
    )
    .expect("slice");

    // World-space SDF.
    let obj = &workspace.objects[0];
    let world_mesh = Mesh::new(
        obj.mesh
            .vertices
            .iter()
            .map(|&v| obj.transform.transform_point(v))
            .collect(),
        obj.mesh.indices.clone(),
    );
    let faces: Vec<[usize; 3]> = world_mesh
        .indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();
    let sdf = MeshSdf::new(world_mesh.vertices.clone(), faces);

    // Find the target layer (closest order value).
    if args.get(4).is_some_and(|a| a == "list") {
        for l in layers.iter() {
            println!("layer {} order={:.6}", l.index, l.order);
        }
        return;
    }
    if args.get(4).is_some_and(|a| a == "paths") {
        let mut po: Vec<(f64, usize)> = Vec::new();
        for p in paths.iter() {
            let o = p.segments.first().map(|s| s.order).unwrap_or(f64::NAN);
            if !o.is_finite() {
                continue;
            }
            match po.iter_mut().find(|(k, _)| (k - o).abs() < 1e-9) {
                Some((_, c)) => *c += 1,
                None => po.push((o, 1)),
            }
        }
        po.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        for (o, c) in po.iter() {
            println!("pathorder {:.6} paths={}", o, c);
        }
        return;
    }
    let layer = layers
        .iter()
        .min_by(|a, b| {
            (a.order - target_order)
                .abs()
                .total_cmp(&(b.order - target_order).abs())
        })
        .expect("layers non-empty");
    println!(
        "target layer: index={} order={:.4} ({} requested)",
        layer.index, layer.order, target_order
    );

    fn stats(label: &str, pts: &[glam::DVec3], sdf: &MeshSdf) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut out = 0;
        for p in pts {
            let v = sdf.sample(*p).value;
            min = min.min(v);
            max = max.max(v);
            sum += v;
            if v > 0.05 {
                out += 1;
            }
        }
        if pts.is_empty() {
            println!("{label}: <empty>");
            return;
        }
        println!(
            "{label}: pts={} sdf min={:.3} mean={:.3} max={:.3} outside(>0.05)={}",
            pts.len(),
            min,
            sum / pts.len() as f64,
            max,
            out
        );
    }

    // 1. Wall loops
    for (i, loop_) in layer.loops.iter().enumerate() {
        stats(
            &format!("wall-{} island={} open={}", i, loop_.island, loop_.is_open),
            &loop_.points,
            &sdf,
        );
    }

    // 2. Slicing boundaries
    let infill_pts: Vec<glam::DVec3> = layer.infill_boundary.iter().flatten().copied().collect();
    stats("infill_boundary", &infill_pts, &sdf);
    let solid_pts: Vec<glam::DVec3> = layer
        .solid_fill_boundary
        .iter()
        .flatten()
        .copied()
        .collect();
    stats("solid_fill_boundary", &solid_pts, &sdf);

    // 3. Sparse region 2D difference, BEFORE reconstruction (at ref height).
    let (axis, apex, _slope) =
        order_field::resolve_axis_apex_slope(profile.config.order_field, &profile.config);
    let (basis1, basis2) = plane_basis(axis);
    let infill_2d = polygon2d::canonicalize(&polygon2d::to_2d(
        &layer.infill_boundary,
        basis1,
        basis2,
        apex,
    ));
    let solid_2d = if !layer.solid_fill_boundary.is_empty() {
        polygon2d::canonicalize(&polygon2d::to_2d(
            &layer.solid_fill_boundary,
            basis1,
            basis2,
            apex,
        ))
    } else {
        Vec::new()
    };
    let min_area = 0.25 * profile.config.nozzle_diameter * profile.config.nozzle_diameter;
    let sparse_2d = if layer.solid_fill_boundary.is_empty() {
        infill_2d.clone()
    } else {
        polygon2d::filter_min_area(&polygon2d::difference(&infill_2d, &solid_2d), min_area)
    };
    // Reference height per point: nearest infill_boundary point's z (same as
    // `reconstruct_on_order_field_near`'s seed selection; axis == +Z so
    // `along` == world z here).
    let infill_refs: Vec<(f64, f64, f64)> = layer
        .infill_boundary
        .iter()
        .flatten()
        .map(|&p| {
            let rel = p - apex;
            (rel.dot(basis1), rel.dot(basis2), rel.dot(axis))
        })
        .collect();
    let mut pre_pts: Vec<glam::DVec3> = Vec::new();
    let mut seed_orders: Vec<f64> = Vec::new();
    for loop2d in sparse_2d.iter() {
        for &[u, v] in loop2d.iter() {
            let along = infill_refs
                .iter()
                .min_by(|a, b| {
                    let da = (a.0 - u).powi(2) + (a.1 - v).powi(2);
                    let db = (b.0 - u).powi(2) + (b.1 - v).powi(2);
                    da.total_cmp(&db)
                })
                .map(|&(_, _, a3)| a3)
                .unwrap_or(0.0);
            let p = apex + basis1 * u + basis2 * v + axis * along;
            pre_pts.push(p);
            seed_orders.push(layer.order_field.order(p));
        }
    }
    stats(
        "sparse-region PRE-reconstruction (at ref height)",
        &pre_pts,
        &sdf,
    );
    {
        let n_in_infill = pre_pts
            .iter()
            .filter(|&p| {
                let rel = *p - apex;
                polygon2d::contains_point(&infill_2d, [rel.dot(basis1), rel.dot(basis2)])
            })
            .count();
        let n_in_solid = pre_pts
            .iter()
            .filter(|&p| {
                let rel = *p - apex;
                polygon2d::contains_point(&solid_2d, [rel.dot(basis1), rel.dot(basis2)])
            })
            .count();
        println!(
            "  pre points inside 2D infill_boundary: {}/{} ; inside 2D solid_fill: {}",
            n_in_infill,
            pre_pts.len(),
            n_in_solid
        );
    }
    if !seed_orders.is_empty() {
        let res: Vec<f64> = seed_orders
            .iter()
            .map(|o| (o - layer.order).abs())
            .collect();
        let mut rs = res.clone();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  seed |order-layer.order|: p50={:.3} p99={:.3} max={:.3}",
            rs[res.len() / 2],
            rs[(res.len() as f64 * 0.99) as usize],
            res.iter().cloned().fold(0.0, f64::max)
        );
    }

    // 4. Sparse region AFTER reconstruction.
    let references: Vec<Vec<glam::DVec3>> = layer
        .infill_boundary
        .iter()
        .chain(layer.solid_fill_boundary.iter())
        .cloned()
        .collect();
    let max_along = (profile.config.layer_height * 20.0).max(5.0);
    let post_pts: Vec<Vec<glam::DVec3>> = order_field::reconstruct_on_order_field_near(
        sparse_2d.clone(),
        &references,
        basis1,
        basis2,
        axis,
        apex,
        layer.order,
        max_along,
        layer.order_field.as_ref(),
    );
    let post_flat: Vec<glam::DVec3> = post_pts.iter().flatten().copied().collect();
    stats("sparse-region POST-reconstruction", &post_flat, &sdf);
    let post_res: Vec<f64> = post_flat
        .iter()
        .map(|p| (layer.order_field.order(*p) - layer.order).abs())
        .collect();
    if !post_res.is_empty() {
        let mut rs = post_res.clone();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  post |order-layer.order|: p50={:.3} p99={:.3} max={:.3}",
            rs[rs.len() / 2],
            rs[(rs.len() as f64 * 0.99) as usize],
            rs.iter().cloned().fold(0.0, f64::max)
        );
    }

    // 5. Final Infill/TopSurface paths at this order.
    let mut worst: Vec<(f64, glam::DVec3)> = Vec::new();
    for path in paths.iter() {
        let kind = path
            .segments
            .first()
            .map(|s| s.kind)
            .unwrap_or(MoveKind::Travel);
        if !matches!(kind, MoveKind::Infill | MoveKind::TopSurface) {
            continue;
        }
        let order_val = path.segments.first().map(|s| s.order).unwrap_or(f64::NAN);
        if (order_val - layer.order).abs() > 1e-6 {
            continue;
        }
        let closed = path.segments.len() == path.points.len();
        let n = path.points.len();
        for i in 0..path.segments.len() {
            let dest = if closed {
                path.points[(i + 1) % n]
            } else {
                path.points[i + 1]
            };
            let v = sdf.sample(dest).value;
            if v > 0.05 {
                worst.push((v, dest));
            }
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "\nfinal Infill/TopSurface paths at this layer: {} outside points; top 12:",
        worst.len()
    );
    for (v, p) in worst.iter().take(12) {
        println!("    SDF={:7.3}  ({:8.3}, {:8.3}, {:8.3})", v, p.x, p.y, p.z);
    }

    // 6. Column analysis at the worst points: SDF + order along Z, and the
    //    nearest infill/solid-fill reference in XY (the seed that reconstruction
    //    would use for that (u,v)).
    let refs: Vec<glam::DVec3> = layer
        .infill_boundary
        .iter()
        .chain(layer.solid_fill_boundary.iter())
        .flatten()
        .copied()
        .collect();
    for (_, p) in worst.iter().take(5) {
        println!("\ncolumn analysis at ({:.3}, {:.3}):", p.x, p.y);
        let mut best_ref: Option<(f64, glam::DVec3, f64, f64)> = None;
        for &r in refs.iter() {
            let d2 = (r.x - p.x).powi(2) + (r.y - p.y).powi(2);
            let ord = layer.order_field.order(r);
            let s = sdf.sample(r).value;
            if best_ref.as_ref().is_none_or(|(bd, _, _, _)| d2 < *bd) {
                best_ref = Some((d2, r, ord, s));
            }
        }
        let mut near: Vec<(f64, glam::DVec3, f64, f64)> = refs
            .iter()
            .map(|&r| {
                let d2 = (r.x - p.x).powi(2) + (r.y - p.y).powi(2);
                (
                    d2.sqrt(),
                    r,
                    layer.order_field.order(r),
                    sdf.sample(r).value,
                )
            })
            .filter(|(d, _, _, _)| *d < 3.0)
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        println!("  refs within 3mm XY: {} ; nearest 10:", near.len());
        for (d, r, ord, s) in near.iter().take(10) {
            println!(
                "    ({:8.3}, {:8.3}, z={:7.3}) d_xy={:5.3} order={:8.3} sdf={:7.3}",
                r.x, r.y, r.z, d, ord, s
            );
        }
        println!("  z        sdf-mm  order");
        let mut i = 0;
        let mut z = 3.0;
        while z <= 14.0 && i < 48 {
            let pt = glam::DVec3::new(p.x, p.y, z);
            let v = sdf.sample(pt).value;
            let o = layer.order_field.order(pt);
            if v.is_finite() || o.is_finite() {
                println!(
                    "  {:7.3}  {:8.3}  {:9.3}",
                    z,
                    if v.is_finite() { v } else { f64::NAN },
                    o
                );
            }
            z += 0.25;
            i += 1;
        }
    }
}
