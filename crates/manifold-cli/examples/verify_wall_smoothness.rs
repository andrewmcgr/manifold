//! Scratch probe: quantify wall-loop noise for a given STL and order
//! field kind. For each wall loop, compares each point to a local
//! moving-average smoothed version of the loop and reports the RMS and
//! max residual plus a coarse dominant-wavelength estimate — if the
//! noise is Eikonal-grid quantization, residuals should sit near the
//! grid pitch h = min(layer_height, nozzle_diameter) / 2 and largely
//! vanish for the Height field (analytic, no grid).
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_wall_noise -- pug_v4_l_sop_85mm.stl eikonal
//! ```

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{self, Object};
use manifold_core::order_field::OrderFieldKind;
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pug_v4_l_sop_85mm.stl".to_string());
    let kind = match std::env::args().nth(2).as_deref() {
        Some("height") => OrderFieldKind::Height,
        Some("conical") => OrderFieldKind::Conical,
        _ => OrderFieldKind::Eikonal,
    };

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);
    let config = SlicerConfig {
        order_field: kind,
        ..SlicerConfig::default()
    };
    let grid_h = config.layer_height.min(config.nozzle_diameter) / 2.0;
    let slope_profile = SlopeProfile::new(vec![(0.0, 15.0)]);
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!(
        "{} layers ({kind:?}), expected grid pitch h = {grid_h:.3} mm",
        layers.len()
    );

    // Per-loop zigzag metric: distance from each point to the midpoint of
    // its immediate neighbors (discrete second difference). On a smooth,
    // densely sampled curve this is tiny (~curvature * spacing^2 / 2);
    // grid-quantization noise shows up as a large, sign-alternating
    // second difference at the sample pitch. Stitched/unsupported points
    // are skipped (deliberate off-surface geometry, not noise).
    let mut residuals: Vec<f64> = Vec::new();
    let mut spacing_sum = 0.0f64;
    let mut alternations = 0usize;
    let mut alternation_n = 0usize;
    let mut max_resid = 0.0f64;
    let mut max_at = DVec3::ZERO;
    let mut per_layer: Vec<(usize, f64, f64)> = Vec::new(); // (index, median, max)
    for layer in &layers {
        let mut layer_res: Vec<f64> = Vec::new();
        for wall in &layer.loops {
            let n = wall.points.len() as isize;
            if n < 8 {
                continue;
            }
            let mut prev_offset: Option<DVec3> = None;
            for i in 0..n {
                let skip = |j: isize| wall.unsupported.get(j.rem_euclid(n) as usize) == Some(&true);
                if skip(i - 1) || skip(i) || skip(i + 1) {
                    prev_offset = None;
                    continue;
                }
                let a = wall.points[(i - 1).rem_euclid(n) as usize];
                let p = wall.points[i as usize];
                let b = wall.points[(i + 1).rem_euclid(n) as usize];
                let mid = (a + b) * 0.5;
                let offset = p - mid;
                let r = offset.length();
                spacing_sum += a.distance(b) * 0.5;
                layer_res.push(r);
                if r > max_resid {
                    max_resid = r;
                    max_at = p;
                }
                // Sign-alternation: consecutive second-difference vectors
                // pointing opposite ways = zigzag; same way = smooth arc.
                if let Some(prev) = prev_offset {
                    alternation_n += 1;
                    if prev.dot(offset) < 0.0 {
                        alternations += 1;
                    }
                }
                prev_offset = Some(offset);
            }
        }
        if !layer_res.is_empty() {
            layer_res.sort_by(f64::total_cmp);
            let median = layer_res[layer_res.len() / 2];
            let max = *layer_res.last().unwrap();
            per_layer.push((layer.index, median, max));
            residuals.extend(layer_res);
        }
    }

    residuals.sort_by(f64::total_cmp);
    let pct = |q: f64| residuals[((residuals.len() - 1) as f64 * q) as usize];
    println!(
        "second-difference residuals over {} points (mean half-spacing {:.4} mm):",
        residuals.len(),
        spacing_sum / residuals.len().max(1) as f64
    );
    println!(
        "  p50 {:.4}  p90 {:.4}  p99 {:.4}  max {:.4} mm (at {:.3?})",
        pct(0.50),
        pct(0.90),
        pct(0.99),
        max_resid,
        max_at
    );
    println!(
        "  sign-alternation rate: {:.1}% (50% = random, >>50% = zigzag)",
        alternations as f64 / alternation_n.max(1) as f64 * 100.0
    );
    println!("worst 8 layers by median residual:");
    per_layer.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (index, median, max) in per_layer.iter().take(8) {
        println!("  layer {index:4}: median {median:.4} mm, max {max:.4} mm");
    }
    Ok(())
}
