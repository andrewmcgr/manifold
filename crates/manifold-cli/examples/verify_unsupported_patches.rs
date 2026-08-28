//! Manual verification for the order-aware unsupported-patch detector
//! (`manifold_core::verification::find_unsupported_patches`): slice a
//! real STL with the Eikonal order field (top+bottom conforming by
//! default) and report every unsupported patch found.
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_unsupported_patches -- \
//!     pug_v4_m_sop_85mm.stl
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::{self, Object};
use manifold_core::order_field::OrderFieldKind;
use manifold_core::verification::{find_unsupported_patches, UnsupportedPatchOptions};
use manifold_core::{slicing, stl, toolpath, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pug_v4_m_sop_85mm.stl".to_string());
    // Second arg: which conformal surfaces to enable — "both" (default),
    // "top", "bottom", or "none".
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "both".to_string());
    let (conform_top, conform_bottom) = match mode.as_str() {
        "top" => (true, false),
        "bottom" => (false, true),
        "none" => (false, false),
        _ => (true, true),
    };

    // Third arg: optional profile.json (GUI format: {"machine": ..., "config": ...}).
    // When given, its `config` and machine slope profile are used instead of
    // the defaults, reproducing GUI slices exactly.
    let profile_path = std::env::args().nth(3);

    let (mut config, slope_profile) = match &profile_path {
        Some(p) => {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
            let machine: Machine = serde_json::from_value(v["machine"].clone())?;
            let config: SlicerConfig = serde_json::from_value(v["config"].clone())?;
            let slope = machine.slope_profile();
            (config, slope)
        }
        None => (SlicerConfig::default(), SlopeProfile::from_angle(15.0)),
    };
    config.order_field = OrderFieldKind::Eikonal;
    config.eikonal_conform_top_surfaces = conform_top;
    config.eikonal_conform_bottom_surfaces = conform_bottom;

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: glam::DVec3::ZERO,
        max: glam::DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);

    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("{} layers", layers.len());

    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &[],
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    let mut options = UnsupportedPatchOptions::for_config(&config);
    // Fourth arg "all": include planner-acknowledged unsupported beads too.
    if std::env::args().nth(4).as_deref() == Some("all") {
        options.include_acknowledged = true;
    }
    // Island hunt: closed loops where (almost) every extruding segment
    // probes into air — i.e. contour islands materializing mid-air with no
    // connection to already-printed material, regardless of stamped kind
    // or support_fraction (the GUI shows these as floating red loops).
    {
        use manifold_core::verification::floating_loops;
        let islands = floating_loops(&paths, &layers, &config);
        println!("{} floating closed loop(s):", islands.len());
        for isl in islands {
            println!(
                "  order {:7.3}: {:8.1} mm loop, centroid ({:7.2},{:7.2},{:7.2}), {:5.1}% unsupported, kinds {:?}",
                isl.order,
                isl.total_length_mm,
                isl.centroid.x,
                isl.centroid.y,
                isl.centroid.z,
                100.0 * isl.unsupported_fraction,
                isl.length_by_kind,
            );
            if isl
                .length_by_kind
                .first()
                .is_some_and(|(k, _)| matches!(k, manifold_core::toolpath::MoveKind::WallOuter))
            {
                for (mid, probe, sdf, ord) in &isl.probe_samples {
                    println!(
                        "      bead ({:7.2},{:7.2},{:7.2}) -> probe ({:7.2},{:7.2},{:7.2}) sdf {:7.3} order {:7.3}",
                        mid.x, mid.y, mid.z, probe.x, probe.y, probe.z, sdf, ord
                    );
                }
            }
        }
    }

    let options_used = options;
    let patches = find_unsupported_patches(&paths, &layers, &config, &options_used);

    // Per-layer extruded length + z-extent: a conformally *flattened* patch
    // shows up as one layer with a huge length and/or z-span spike.
    let mut by_order: std::collections::BTreeMap<u64, (f64, f64, f64)> =
        std::collections::BTreeMap::new();
    for path in &paths {
        for (i, seg) in path.segments.iter().enumerate() {
            if seg.extrusion_length <= 0.0 {
                continue;
            }
            let a = path.points[i];
            let b = path.points[(i + 1) % path.points.len()];
            let e = by_order.entry(seg.order.to_bits()).or_insert((
                0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ));
            e.0 += (b - a).length();
            e.1 = e.1.min(a.z.min(b.z));
            e.2 = e.2.max(a.z.max(b.z));
        }
    }
    let mut rows: Vec<(f64, f64, f64, f64)> = by_order
        .iter()
        .map(|(bits, (len, zmin, zmax))| (f64::from_bits(*bits), *len, *zmin, *zmax))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top 12 layers by extruded length (order, mm, z-span):");
    for (order, len, zmin, zmax) in rows.iter().take(12) {
        println!(
            "  order {order:7.3}: {len:8.1} mm, z {zmin:6.2}..{zmax:6.2} (span {:5.2})",
            zmax - zmin
        );
    }
    println!("layers near order 4.8 (4.2..5.4):");
    let mut near: Vec<_> = rows.iter().filter(|r| r.0 > 4.2 && r.0 < 5.4).collect();
    near.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (order, len, zmin, zmax) in near {
        println!(
            "  order {order:7.3}: {len:8.1} mm, z {zmin:6.2}..{zmax:6.2} (span {:5.2})",
            zmax - zmin
        );
    }

    if patches.is_empty() {
        println!("no unsupported patches found");
        return Ok(());
    }
    println!(
        "{} unsupported patch(es) (min length {:.1} mm, link radius {:.2} mm):",
        patches.len(),
        options.min_patch_length_mm,
        options.link_radius_mm
    );
    for patch in &patches {
        println!(
            "  order {:7.3}: {:8.1} mm over {:5} segs (max seg {:6.2} mm), centroid ({:7.2}, {:7.2}, {:7.2}), \
             bbox ({:6.2},{:6.2},{:6.2})..({:6.2},{:6.2},{:6.2}) kinds {:?}",
            patch.order,
            patch.total_length_mm,
            patch.segment_count,
            patch.max_segment_length_mm,
            patch.centroid.x,
            patch.centroid.y,
            patch.centroid.z,
            patch.min.x,
            patch.min.y,
            patch.min.z,
            patch.max.x,
            patch.max.y,
            patch.max.z,
            patch.length_by_kind,
        );
    }
    Ok(())
}
