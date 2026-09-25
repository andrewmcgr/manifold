//! Per-layer planned-path dump: order, infill/solid boundary loop counts and
//! areas, and planned extruded segment/E-length totals attributed to each
//! layer by path z.
//!
//! Usage: probe_layer_dump <mesh.stl> [profile.json]
//!
//! Temporary diagnostic -- compares layer-by-layer planned output between
//! trees to attribute total-count deltas to specific layers.

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{center_on_bed, Object};

#[derive(serde::Deserialize)]
struct Profile {
    machine: manifold_core::machine::Machine,
    config: manifold_core::SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    let mesh_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Test5.stl".to_string());
    let profile_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "examples/profile.json".to_string());

    let mesh = manifold_core::stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&json)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let mut objects = vec![object];
    center_on_bed(&mut objects, &profile.machine.build_volume);
    let object = &objects[0];

    let empty_slope = std::env::args().any(|a| a == "--empty-slope");
    let slope_profile = if empty_slope {
        manifold_fidget::slope_profile::SlopeProfile::new(Vec::new())
    } else {
        profile.machine.slope_profile()
    };
    let layers = manifold_core::slicing::slice_object_with_progress(
        object,
        &profile.config,
        &slope_profile,
        &mut |_| {},
    )?;

    let mut on_progress = |_f: f64| {};
    let paths = manifold_core::toolpath::plan_with_progress(
        &layers,
        &objects,
        &profile.machine.tools,
        &profile.config,
        Some(&profile.machine),
        &slope_profile,
        &mut on_progress,
    )?;

    fn area(loops: &[Vec<glam::DVec3>]) -> f64 {
        loops
            .iter()
            .map(|lp| {
                let mut a = 0.0;
                for w in 0..lp.len() {
                    let p = lp[w];
                    let q = lp[(w + 1) % lp.len()];
                    a += p.x * q.y - q.x * p.y;
                }
                a.abs() / 2.0
            })
            .sum()
    }

    // Attribute each path to the layer whose order its z is closest to.
    let orders: Vec<f64> = layers.iter().map(|l| l.order).collect();
    let mut segs_by_layer = vec![0usize; layers.len()];
    let mut e_by_layer = vec![0.0f64; layers.len()];
    let mut paths_by_layer = vec![0usize; layers.len()];
    for path in &paths {
        let z = path.points.first().map(|p| p.z).unwrap_or(f64::NAN);
        let li = orders
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - z).abs().total_cmp(&(b.1 - z).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let n_ext = path
            .segments
            .iter()
            .filter(|s| s.extrusion_length > 0.0)
            .count();
        let e_len: f64 = path
            .segments
            .iter()
            .filter(|s| s.extrusion_length > 0.0)
            .map(|s| s.extrusion_length)
            .sum();
        segs_by_layer[li] += n_ext;
        e_by_layer[li] += e_len;
        paths_by_layer[li] += 1;
    }

    let mut tot_segs = 0usize;
    let mut tot_e = 0.0f64;
    for (i, layer) in layers.iter().enumerate() {
        tot_segs += segs_by_layer[i];
        tot_e += e_by_layer[i];
        println!(
            "L{i} order={:.4} ib={}({:.1}mm2) sf={}({:.1}mm2) paths={} segs={} e={:.3}",
            layer.order,
            layer.infill_boundary.len(),
            area(&layer.infill_boundary),
            layer.solid_fill_boundary.len(),
            area(&layer.solid_fill_boundary),
            paths_by_layer[i],
            segs_by_layer[i],
            e_by_layer[i],
        );
    }
    println!(
        "TOTAL layers={} paths={} segs={tot_segs} e={tot_e:.3}",
        layers.len(),
        paths.len()
    );
    Ok(())
}
