//! Scratch verification for the flat-nozzle-tip contour compensation
//! (`toolpath::compensate_flat_nozzle`). Not part of the crate's public
//! surface or test suite -- run manually against the real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_flat_nozzle -- pug_v4_l_sop_85mm.stl
//! ```

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::tool::Tool;
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath, SlicerConfig};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    // Effectively unbounded build volume so this scratch check isolates
    // geometry sanity from the unrelated build-volume-placement issue.
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    for (label, order_field, flat_diameter) in [
        ("height, no compensation", OrderFieldKind::Height, Some(0.0)),
        ("height, default compensation", OrderFieldKind::Height, None),
        (
            "conical, no compensation",
            OrderFieldKind::Conical,
            Some(0.0),
        ),
        (
            "conical, default compensation",
            OrderFieldKind::Conical,
            None,
        ),
    ] {
        let config = SlicerConfig {
            order_field,
            nozzle_flat_diameter: flat_diameter,
            order_field_slope: if order_field == OrderFieldKind::Conical {
                0.3
            } else {
                0.0
            },
            ..SlicerConfig::default()
        };

        let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
        let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;

        let mut max_coord = f64::MIN;
        let mut min_coord = f64::MAX;
        let mut nan_count = 0usize;
        let mut wall_points = 0usize;
        for path in &paths {
            let is_wall = path
                .segments
                .first()
                .is_some_and(|s| matches!(s.kind, MoveKind::WallOuter | MoveKind::WallInner));
            if !is_wall {
                continue;
            }
            for p in &path.points {
                wall_points += 1;
                if !p.x.is_finite() || !p.y.is_finite() || !p.z.is_finite() {
                    nan_count += 1;
                    continue;
                }
                max_coord = max_coord.max(p.x).max(p.y).max(p.z);
                min_coord = min_coord.min(p.x).min(p.y).min(p.z);
            }
        }

        println!(
            "{label}: wall_points={wall_points} nan={nan_count} min={min_coord:.3} max={max_coord:.3}"
        );
    }

    Ok(())
}
