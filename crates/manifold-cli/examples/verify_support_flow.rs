//! Manual verification for support-aware flow (`support_fractions_at`):
//! slice a real STL, plan toolpaths, and report per-layer
//! `Segment::support_fraction` statistics plus the flow (extrusion
//! length) delta versus the uniform stadium model.
//!
//! Expectations:
//! - First layer: overwhelmingly fully supported (bed contact).
//! - Interior layers of a solid: mostly fully supported (mesh SDF).
//! - `Overhang`-classified segments should skew toward low fractions.
//! - No NaN/out-of-range fractions anywhere.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_support_flow -- pug_v4_l_sop_85mm.stl eikonal
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::object::{self, Object};
use manifold_core::order_field::OrderFieldKind;
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath, SlicerConfig};
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
        min: glam::DVec3::ZERO,
        max: glam::DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);
    let config = SlicerConfig {
        order_field: kind,
        ..SlicerConfig::default()
    };
    let slope_profile = SlopeProfile::new(vec![(0.0, 15.0)]);
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("{} layers ({kind:?})", layers.len());

    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &[],
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    let mut bad = 0usize;
    let mut total_extrude = 0.0f64;
    let mut per_layer: std::collections::BTreeMap<u64, (usize, usize, f64)> = Default::default();
    let mut overhang = (0usize, 0.0f64); // (count, fraction sum)
    let mut supported = (0usize, 0.0f64);
    for path in &paths {
        for segment in &path.segments {
            if segment.kind == MoveKind::Travel {
                continue;
            }
            let f = segment.support_fraction;
            if !f.is_finite() || !(0.0..=1.0).contains(&f) {
                bad += 1;
            }
            total_extrude += segment.extrusion_length;
            let entry = per_layer
                .entry(segment.order.to_bits())
                .or_insert((0, 0, 0.0));
            entry.0 += 1;
            if f >= 0.999 {
                entry.1 += 1;
            }
            entry.2 += f;
            if segment.kind == MoveKind::Overhang {
                overhang.0 += 1;
                overhang.1 += f;
            } else {
                supported.0 += 1;
                supported.1 += f;
            }
        }
    }

    println!("total extrusion length: {total_extrude:.1} mm of filament");
    println!("out-of-range/NaN fractions: {bad}");
    let mean = |(n, s): (usize, f64)| if n > 0 { s / n as f64 } else { f64::NAN };
    println!(
        "mean support fraction: overhang segments {:.3} ({} segs), other {:.3} ({} segs)",
        mean(overhang),
        overhang.0,
        mean(supported),
        supported.0
    );
    println!("first/last 5 layers (by order): full-support share, mean fraction");
    let rows: Vec<_> = per_layer.iter().collect();
    for (i, (order_bits, (n, full, sum))) in rows.iter().enumerate() {
        if i < 5 || i + 5 >= rows.len() {
            println!(
                "  order {:8.3}: {:5} segs, {:5.1}% full, mean {:.3}",
                f64::from_bits(**order_bits),
                n,
                *full as f64 / (*n).max(1) as f64 * 100.0,
                sum / (*n).max(1) as f64
            );
        } else if i == 5 {
            println!("  ...");
        }
    }
    Ok(())
}
