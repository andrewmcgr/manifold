//! `manifold` CLI: drives `manifold-core` headlessly to slice a mesh to Gcode.
//!
//! TODO(roadmap): Phase 8 (see ROADMAP.md) — accept multiple input files
//! (+ a per-file tool assignment flag) and build a multi-object
//! `Workspace` instead of the single-object/single-tool placeholder
//! below.

use anyhow::Result;
use clap::Parser;
use glam::DVec3;
use manifold_core::{
    bounds::BoundingVolume,
    ids::{ObjectId, ToolId},
    machine::Machine,
    mesh::Mesh,
    object::Object,
    slice_to_gcode, SlicerConfig, Workspace,
};
use std::path::PathBuf;

/// Non-planar slicer CLI.
#[derive(Debug, Parser)]
#[command(name = "manifold", version, about)]
struct Cli {
    /// Input mesh file (e.g. STL).
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
    // TODO: real mesh loading (STL/3MF) belongs in manifold-core.
    let mesh = Mesh::default();

    let config = SlicerConfig {
        layer_height: cli.layer_height,
        nozzle_diameter: cli.nozzle_diameter,
    };

    // Single-object, single-tool workspace until Phase 8 wires up real
    // multi-file/multi-tool input (see ROADMAP.md).
    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        Vec::new(),
    );
    let workspace = Workspace::new(vec![object], machine, config);

    let gcode = slice_to_gcode(&workspace)?;
    std::fs::write(&cli.output, gcode)?;
    tracing::info!(output = %cli.output.display(), "wrote gcode");

    Ok(())
}
