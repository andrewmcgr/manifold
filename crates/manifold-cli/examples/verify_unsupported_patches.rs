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

    let mesh = stl::load_stl(std::fs::File::open(&path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: glam::DVec3::ZERO,
        max: glam::DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);

    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        eikonal_conform_top_surfaces: conform_top,
        eikonal_conform_bottom_surfaces: conform_bottom,
        ..SlicerConfig::default()
    };
    let slope_profile = SlopeProfile::from_angle(15.0);
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

    let options = UnsupportedPatchOptions::for_config(&config);
    let patches = find_unsupported_patches(&paths, &layers, &config, &options);

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
