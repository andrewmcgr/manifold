//! Scratch probe: investigate a reported bug where the volume audit shows
//! roughly half the first layer's shell/infill missing on TestObj1.stl with
//! a real saved profile.json. Focuses on layers near z=0 and the affected
//! world-x region (roughly x > 178mm, per the volume-audit report), diffing
//! sliced loop geometry against planned toolpaths the same way
//! `probe_layer_dropouts.rs` does, but scoped to the real profile and the
//! specific affected region instead of a synthetic Eikonal config.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_first_layer_dropout -- \
//!     /path/to/mesh.stl /path/to/profile.json [x_threshold] [max_layer_index]
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{center_on_bed, Object};
use manifold_core::toolpath::{self, MoveKind};
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::ScalarField;

#[derive(serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "TestObj1.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/Users/amcgregor/3D/profile.json".to_string());
    let x_threshold: f64 = std::env::args()
        .nth(3)
        .map(|s| s.parse().expect("x_threshold must be a number"))
        .unwrap_or(178.0);
    let max_layer_index: usize = std::env::args()
        .nth(4)
        .map(|s| s.parse().expect("max_layer_index must be a number"))
        .unwrap_or(3);

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);

    // Mesh bounding box in world space, for orientation.
    let (mut mmin, mut mmax) = (
        glam::DVec3::splat(f64::INFINITY),
        glam::DVec3::splat(f64::NEG_INFINITY),
    );
    for &v in &objects[0].mesh.vertices {
        let w = objects[0].transform.transform_point(v);
        mmin = mmin.min(w);
        mmax = mmax.max(w);
    }
    println!(
        "World mesh bbox: x[{:.1},{:.1}] y[{:.1},{:.1}] z[{:.1},{:.1}]",
        mmin.x, mmax.x, mmin.y, mmax.y, mmin.z, mmax.z
    );
    println!("Investigating region x > {x_threshold}, layers 0..={max_layer_index}\n");

    let layers = slicing::slice_object(&objects[0], &profile.config)?;
    println!("Sliced into {} layers", layers.len());

    for layer in layers.iter().take(max_layer_index + 1) {
        let mut wall_stats: Vec<(usize, usize, usize)> = Vec::new();
        let mut wall_stats_region: Vec<(usize, usize, usize)> = Vec::new();
        for l in &layer.loops {
            let touches_region = l.points.iter().any(|p| p.x > x_threshold);
            let entry = wall_stats
                .iter_mut()
                .find(|s: &&mut (usize, usize, usize)| s.0 == l.wall_index);
            match entry {
                Some(s) => {
                    s.1 += 1;
                    s.2 += l.points.len();
                }
                None => wall_stats.push((l.wall_index, 1, l.points.len())),
            }
            if touches_region {
                let entry_r = wall_stats_region
                    .iter_mut()
                    .find(|s: &&mut (usize, usize, usize)| s.0 == l.wall_index);
                match entry_r {
                    Some(s) => {
                        s.1 += 1;
                        s.2 += l.points.len();
                    }
                    None => wall_stats_region.push((l.wall_index, 1, l.points.len())),
                }
            }
        }
        wall_stats.sort_by_key(|s| s.0);
        wall_stats_region.sort_by_key(|s| s.0);

        println!(
            "layer {:3} order {:7.3} z~{:.2} | total walls: {} | region (x>{x_threshold}) walls: {}",
            layer.index,
            layer.order,
            layer
                .loops
                .first()
                .and_then(|l| l.points.first())
                .map(|p| p.z)
                .unwrap_or(f64::NAN),
            wall_stats
                .iter()
                .map(|(w, n, p)| format!("w{w}:{n}loops/{p}pts"))
                .collect::<Vec<_>>()
                .join(" "),
            if wall_stats_region.is_empty() {
                "NONE <-- SUSPECT".to_string()
            } else {
                wall_stats_region
                    .iter()
                    .map(|(w, n, p)| format!("w{w}:{n}loops/{p}pts"))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        );

        // Solid-fill / infill boundary presence in the affected region.
        let solid_in_region = layer
            .solid_fill_boundary
            .iter()
            .any(|poly| poly.iter().any(|p| p.x > x_threshold));
        let infill_in_region = layer
            .infill_boundary
            .iter()
            .any(|poly| poly.iter().any(|p| p.x > x_threshold));
        println!(
            "           solid_fill_boundary in region: {solid_in_region} ({} polys total) | infill_boundary in region: {infill_in_region} ({} polys total)",
            layer.solid_fill_boundary.len(),
            layer.infill_boundary.len()
        );

        // Mesh SDF sanity check: sample a handful of points at this layer's
        // z in the affected region, directly against the raw mesh SDF (same
        // ground truth the volume audit itself uses), to see whether the
        // MESH says material should exist there at all (it must, since the
        // object's own geometry is intact) -- isolating whether this is a
        // slicing decision bug or something upstream.
        if let Some(sdf) = &layer.mesh_sdf {
            let z = layer.order;
            let mut inside_count = 0;
            let mut total = 0;
            for i in 0..20 {
                let x = x_threshold + 0.5 + i as f64 * 0.2;
                let y = (mmin.y + mmax.y) / 2.0;
                let sample = sdf.sample(glam::DVec3::new(x, y, z));
                total += 1;
                if sample.value <= 0.0 {
                    inside_count += 1;
                }
            }
            println!(
                "           mesh SDF sanity (20 samples along x from {:.1}, at y={:.1}, z={z:.2}): {inside_count}/{total} inside the mesh",
                x_threshold + 0.5,
                (mmin.y + mmax.y) / 2.0
            );
        }
        println!();
    }

    println!("planning toolpaths...");
    let paths = toolpath::plan(&layers, &objects, &profile.machine.tools, &profile.config)?;
    println!("Planned into {} paths\n", paths.len());

    for layer in layers.iter().take(max_layer_index + 1) {
        let sliced_wall0_region = layer
            .loops
            .iter()
            .filter(|l| l.wall_index == 0 && l.points.iter().any(|p| p.x > x_threshold))
            .count();
        let planned_wall0_region = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .any(|s| s.kind == MoveKind::WallOuter && (s.order - layer.order).abs() < 1e-9)
                    && p.points.iter().any(|q| q.x > x_threshold)
            })
            .count();
        println!(
            "layer {:3} order {:7.3}: sliced wall-0 loops in region: {sliced_wall0_region}, planned wall-0 paths in region: {planned_wall0_region}",
            layer.index, layer.order
        );

        let region_infill_paths = paths
            .iter()
            .filter(|p| {
                p.segments
                    .iter()
                    .any(|s| s.kind == MoveKind::Infill && (s.order - layer.order).abs() < 1e-9)
                    && p.points.iter().any(|q| q.x > x_threshold)
            })
            .count();
        println!("           planned infill paths in region: {region_infill_paths}");
    }

    Ok(())
}
