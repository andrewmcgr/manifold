//! Scratch verification for skipping Z-hops between infill-to-infill
//! travel jumps within the same patch (`toolpath::insert_z_hops_into_path`).
//! Not part of the crate's public surface or test suite -- run manually
//! against the real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_infill_zhop -- pug_v4_l_sop_85mm.stl
//! ```
use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
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
    // z-hop behavior from the unrelated build-volume-placement issue (see
    // verify_flat_nozzle.rs for the same workaround).
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let config = SlicerConfig {
        z_hop_enabled: true,
        z_hop_height: 0.4,
        ..SlicerConfig::default()
    };

    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;

    let mut infill_bounded_hops = 0usize;
    let mut other_hops = 0usize;
    let mut infill_travel_edges = 0usize;

    for path in &paths {
        let n = path.segments.len();
        for (i, segment) in path.segments.iter().enumerate() {
            if segment.kind != MoveKind::Travel {
                continue;
            }
            infill_travel_edges += usize::from(
                i > 0 && path.segments[i - 1].kind == MoveKind::Infill
                    || (i + 1 < n && path.segments[i + 1].kind == MoveKind::Infill),
            );
        }
        // A hopped run leaves a telltale Z pattern: a raised point (Z
        // higher than both neighbors) surrounded by Travel segments.
        for i in 1..path.points.len().saturating_sub(1) {
            let raised = path.points[i].z > path.points[i - 1].z + 1e-6
                && path.points[i].z > path.points[i + 1].z + 1e-6;
            if !raised {
                continue;
            }
            let incoming_infill = i >= 2 && path.segments[i - 2].kind == MoveKind::Infill;
            let outgoing_infill =
                i + 1 < path.segments.len() && path.segments[i + 1].kind == MoveKind::Infill;
            if incoming_infill && outgoing_infill {
                infill_bounded_hops += 1;
            } else {
                other_hops += 1;
            }
        }
    }

    println!("infill-to-infill travel edges: {infill_travel_edges}");
    println!("hops found bounded by infill on both sides (should be 0): {infill_bounded_hops}");
    println!("other hops (walls, patch-to-patch, etc.): {other_hops}");

    Ok(())
}
