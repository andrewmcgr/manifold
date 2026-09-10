//! Scratch probe: slice an STL using a real saved `Profile` JSON (machine +
//! config, same schema as `manifold-gui`'s `Profile`) and dump per-layer
//! loop statistics, flagging empty/missing layers. Run manually:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example probe_profile_layers -- \
//!     /path/to/profile.json /path/to/model.stl
//! ```

use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::toolpath;
use manifold_core::{slicing, stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;

#[derive(Debug, serde::Deserialize)]
struct Profile {
    machine: Machine,
    config: SlicerConfig,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let profile_path = std::env::args()
        .nth(1)
        .expect("usage: probe_profile_layers <profile.json> <model.stl>");
    let mesh_path = std::env::args()
        .nth(2)
        .expect("usage: probe_profile_layers <profile.json> <model.stl>");

    let profile_json = std::fs::read_to_string(&profile_path)?;
    let profile: Profile = serde_json::from_str(&profile_json)?;
    let config = profile.config;
    let machine = profile.machine;

    println!(
        "order_field = {:?}, wave_overhangs_enabled = {}, layer_height = {}",
        config.order_field, config.wave_overhangs_enabled, config.layer_height
    );

    let mesh = stl::load_stl(std::fs::File::open(&mesh_path)?)?;
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);

    let slope_profile = SlopeProfile::new(machine.eikonal_slope_profile.clone());

    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!(
        "sliced {} layer(s), layer_height {}",
        layers.len(),
        config.layer_height
    );

    let mut prev_order: Option<f64> = None;
    for layer in &layers {
        let mut wall_stats: Vec<(usize, usize, usize)> = Vec::new();
        for l in &layer.loops {
            if let Some(s) = wall_stats.iter_mut().find(|s| s.0 == l.wall_index) {
                s.1 += 1;
                s.2 += l.points.len();
            } else {
                wall_stats.push((l.wall_index, 1, l.points.len()));
            }
        }
        wall_stats.sort_by_key(|s| s.0);

        let gap = prev_order
            .map(|p| layer.order - p)
            .unwrap_or(config.layer_height);
        let gap_flag = if (gap - config.layer_height).abs() > 1e-6 {
            " <-- ORDER GAP"
        } else {
            ""
        };
        let empty_flag = if layer.loops.is_empty() {
            " <-- EMPTY"
        } else {
            ""
        };

        let stats: Vec<String> = wall_stats
            .iter()
            .map(|(w, n, p)| format!("w{w}:{n}loops/{p}pts"))
            .collect();
        println!(
            "layer {:3} order {:7.3} | {} {}{}",
            layer.index,
            layer.order,
            if stats.is_empty() {
                "-".to_string()
            } else {
                stats.join(" ")
            },
            gap_flag,
            empty_flag,
        );
        prev_order = Some(layer.order);
    }

    println!("\nplanning toolpaths...");
    let tools = machine.tools.clone();
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &tools,
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )?;

    for layer in &layers[0..=4] {
        let sliced_wall0 = layer.loops.iter().filter(|l| l.wall_index == 0).count();
        let mut kind_counts: std::collections::BTreeMap<String, usize> = Default::default();
        let mut path_count = 0usize;
        for p in &paths {
            let matches_layer = p
                .segments
                .iter()
                .any(|s| (s.order - layer.order).abs() < 1e-6);
            if matches_layer {
                path_count += 1;
                for s in &p.segments {
                    if (s.order - layer.order).abs() < 1e-6 {
                        *kind_counts.entry(format!("{:?}", s.kind)).or_insert(0) += 1;
                    }
                }
            }
        }
        println!(
            "LAYER {:3} (order {:7.3}): paths={} sliced_wall0={} segment_kinds={:?}",
            layer.index, layer.order, path_count, sliced_wall0, kind_counts
        );
    }

    println!("done");
    Ok(())
}
