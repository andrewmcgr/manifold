//! `manifold` CLI: drives `manifold-core` headlessly to slice a mesh to Gcode.
//!
//! TODO(roadmap): Phase 8 (see ROADMAP.md) — accept multiple input files
//! (+ a per-file tool assignment flag) and build a multi-object
//! `Workspace` instead of the single-tool-for-all-objects placeholder
//! below.
//!
//! TODO(roadmap): Phase 1 (see ROADMAP.md) — add STL loading; only `.3mf`
//! input is wired up today.

use anyhow::{bail, Context, Result};
use clap::Parser;
use glam::DVec3;
use manifold_core::{
    bounds::BoundingVolume, ids::ToolId, machine::Machine, slice_to_gcode, threemf, SlicerConfig,
    Workspace,
};
use std::fs::File;
use std::path::PathBuf;

/// Non-planar slicer CLI.
#[derive(Debug, Parser)]
#[command(name = "manifold", version, about)]
struct Cli {
    /// Input mesh file (currently only 3MF is supported).
    input: PathBuf,

    /// Output Gcode file.
    #[arg(short, long, default_value = "out.gcode")]
    output: PathBuf,

    /// Layer height in millimeters.
    #[arg(long, default_value_t = 0.2)]
    layer_height: f64,

    /// Nozzle diameter in millimeters.
    #[arg(long, default_value_t = 0.4)]
    nozzle_diameter: f64,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    tracing::info!(input = %cli.input.display(), "loading mesh");
    let objects = load_objects(&cli.input)?;

    let config = SlicerConfig {
        layer_height: cli.layer_height,
        nozzle_diameter: cli.nozzle_diameter,
    };

    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        Vec::new(),
    );
    let workspace = Workspace::new(objects, machine, config);

    let gcode = slice_to_gcode(&workspace)?;
    std::fs::write(&cli.output, gcode)?;
    tracing::info!(output = %cli.output.display(), "wrote gcode");

    Ok(())
}

/// Load every object from `input`, dispatching on its file extension.
///
/// All loaded objects are assigned to [`ToolId(0)`], since multi-tool
/// input assignment (Phase 8, see ROADMAP.md) is not yet wired up.
fn load_objects(input: &PathBuf) -> Result<Vec<manifold_core::object::Object>> {
    let extension = input
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match extension.as_str() {
        "3mf" => {
            let file = File::open(input)
                .with_context(|| format!("failed to open {}", input.display()))?;
            let objects = threemf::load_3mf(file, ToolId(0))?;
            Ok(objects)
        }
        other => bail!(
            "unsupported input format {:?} for {}: only .3mf is supported today (STL loading is pending, see ROADMAP.md)",
            other,
            input.display()
        ),
    }
}
