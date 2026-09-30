//! Scratch diagnostic: report every planned extruding path that has points
//! outside the mesh solid (SDF > 0.05), with per-kind order-band statistics,
//! to quantify "sparse infill outside the object" defects (e.g. TestObj1.stl
//! midline at order ~11.1).
//!
//! Usage: cargo run --release --bin probe_infill_outside -- <mesh.stl> <profile.json>

use manifold_core::machine::Machine;
use manifold_core::mesh::Mesh;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::{MoveKind, Path};
use manifold_core::{plan_toolpaths, SlicerConfig, Workspace};
use manifold_fidget::{mesh_sdf::MeshSdf, ScalarField};
use std::io::BufReader;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn kind_name(kind: MoveKind) -> &'static str {
    match kind {
        MoveKind::WallOuter => "WallOuter",
        MoveKind::WallInner => "WallInner",
        MoveKind::Infill => "Infill",
        MoveKind::Bridge => "Bridge",
        MoveKind::Overhang => "Overhang",
        MoveKind::TopSurface => "TopSurface",
        MoveKind::Travel => "Travel",
        MoveKind::Wipe => "Wipe",
        MoveKind::DebugExcluded => "DebugExcluded",
    }
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
    let machine = profile.machine;
    center_on_bed(&mut objects, &machine.build_volume);

    let workspace = Workspace::new(objects, machine, profile.config.clone());
    let paths: Vec<Path> = plan_toolpaths(&workspace).expect("plan toolpaths");

    // World-space SDF over the placed object (same construction as
    // `slicing::slice_object_with_progress`).
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

    let (bbox_min, bbox_max) = world_mesh.bounding_box().expect("non-empty mesh");
    println!(
        "object bbox: min=({:.2},{:.2},{:.2}) max=({:.2},{:.2},{:.2}) nozzle={:.3} layer_h={:.3}",
        bbox_min.x,
        bbox_min.y,
        bbox_min.z,
        bbox_max.x,
        bbox_max.y,
        bbox_max.z,
        profile.config.nozzle_diameter,
        profile.config.layer_height
    );

    struct Report {
        kind: MoveKind,
        order: f64,
        n_pts: usize,
        n_outside: usize,
        n_gross: usize,
        max_sdf: f64,
        samples: Vec<(f64, glam::DVec3, MoveKind)>,
    }
    let mut reports: Vec<Report> = Vec::new();

    for path in paths.iter() {
        let fill_kind = path
            .segments
            .iter()
            .find(|s| {
                matches!(
                    s.kind,
                    MoveKind::Infill
                        | MoveKind::TopSurface
                        | MoveKind::WallOuter
                        | MoveKind::WallInner
                        | MoveKind::Bridge
                        | MoveKind::Overhang
                )
            })
            .map(|s| s.kind);
        let Some(kind) = fill_kind else {
            continue;
        };
        let order = path.segments.first().map(|s| s.order).unwrap_or(f64::NAN);
        let n = path.points.len();
        if n == 0 {
            continue;
        }
        let closed = path.segments.len() == n;
        let (mut n_outside, mut n_gross, mut max_sdf, mut samples) =
            (0, 0, f64::NEG_INFINITY, Vec::new());
        for i in 0..path.segments.len() {
            if !matches!(
                path.segments[i].kind,
                MoveKind::Infill
                    | MoveKind::TopSurface
                    | MoveKind::WallOuter
                    | MoveKind::WallInner
                    | MoveKind::Bridge
                    | MoveKind::Overhang
            ) {
                continue;
            }
            let dest = if closed {
                path.points[(i + 1) % n]
            } else {
                path.points[i + 1]
            };
            let v = sdf.sample(dest).value;
            max_sdf = max_sdf.max(v);
            if v > 0.05 {
                n_outside += 1;
                if v > profile.config.nozzle_diameter {
                    n_gross += 1;
                }
                samples.push((v, dest, path.segments[i].kind));
            }
        }
        if n_outside > 0 {
            reports.push(Report {
                kind,
                order,
                n_pts: path.points.len(),
                n_outside,
                n_gross,
                max_sdf,
                samples,
            });
        }
    }

    reports.sort_by(|a, b| {
        a.order
            .partial_cmp(&b.order)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "\n{} paths with points outside the solid (SDF > 0.05):",
        reports.len()
    );
    let mut by_kind: std::collections::BTreeMap<&'static str, (usize, usize, f64)> =
        std::collections::BTreeMap::new();
    for r in reports.iter() {
        let entry = by_kind
            .entry(kind_name(r.kind))
            .or_insert((0, 0, f64::NEG_INFINITY));
        entry.0 += 1;
        entry.1 += r.n_outside;
        entry.2 = entry.2.max(r.max_sdf);
    }
    println!("kind         paths  outside-pts  max-sdf-mm");
    for (name, (paths, pts, max_sdf)) in by_kind.iter() {
        println!("{:<12} {:>5}  {:>11}  {max_sdf:>8.3}", name, paths, pts);
    }

    println!("\nper-path detail (order sorted):");
    for r in reports.iter() {
        println!(
            "order={:7.3} {:<10} pts={:<5} outside={:<4} gross={:<3} max_sdf={:7.3}  sample:",
            r.order,
            kind_name(r.kind),
            r.n_pts,
            r.n_outside,
            r.n_gross,
            r.max_sdf
        );
        let mut worst = r.samples.clone();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (v, p, k) in worst.iter().take(6) {
            println!(
                "    ({:8.3}, {:8.3}, {:8.3})  sdf={:+.3} kind={}",
                p.x,
                p.y,
                p.z,
                v,
                kind_name(*k)
            );
        }
    }
}
