//! `manifold` CLI: drives `manifold-core` headlessly to slice mesh(es) to
//! Gcode. Accepts multiple input files, each optionally suffixed with a
//! tool id (`path[:tool]`) for per-file tool assignment.

use anyhow::{bail, Context, Result};
use clap::Parser;
use glam::DVec3;
use manifold_core::{
    bounds::BoundingVolume, ids::ObjectId, ids::ToolId, infill::InfillPatternKind,
    machine::Machine, object::Object, order_field::OrderFieldKind, slice_to_gcode, stl, threemf,
    tool::Tool, SlicerConfig, Workspace,
};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Non-planar slicer CLI.
#[derive(Debug, Parser)]
#[command(name = "manifold", version, about)]
struct Cli {
    /// Input mesh file(s) (STL or 3MF). Each entry may optionally suffix a
    /// tool id to assign that file's objects to, e.g. `part.stl:1`
    /// (defaults to tool `0` if omitted).
    #[arg(required = true, num_args = 1..)]
    inputs: Vec<String>,

    /// Output Gcode file.
    #[arg(short, long, default_value = "out.gcode")]
    output: std::path::PathBuf,

    /// Layer height in millimeters.
    #[arg(long, default_value_t = 0.2)]
    layer_height: f64,

    /// Nozzle diameter in millimeters, applied to every tool referenced by
    /// `inputs`. Per-tool nozzle diameters are a future follow-up.
    #[arg(long, default_value_t = 0.4)]
    nozzle_diameter: f64,

    /// Order field used to walk the slicing isosurfaces.
    #[arg(long, value_enum, default_value_t = OrderFieldArg::Height)]
    order_field: OrderFieldArg,

    /// Infill pattern generated for sparse regions.
    #[arg(long, value_enum)]
    sparse_infill_pattern: Option<InfillPatternArg>,

    /// Infill pattern generated for solid top/bottom layers.
    #[arg(long, value_enum)]
    solid_infill_pattern: Option<InfillPatternArg>,

    /// Infill pattern generated inside each layer (legacy).
    #[arg(long, value_enum, default_value_t = InfillPatternArg::Cubic)]
    infill_pattern: InfillPatternArg,

    /// Slope-limit profile for the `eikonal` order field, as comma-separated
    /// `height_mm:max_degrees` breakpoints (height measured above the
    /// mesh's build-plate contact surface), e.g. `0:45,4:2` for a tight
    /// 45deg limit near the build plate loosening to 2deg above 4mm.
    /// Ignored unless `--order-field eikonal`. Defaults to no limit
    /// (unconstrained) if omitted.
    #[arg(long)]
    eikonal_slope_profile: Option<String>,

    /// Whether the Eikonal order field blends with the top surface.
    #[arg(long, default_value_t = false)]
    eikonal_conform_top_surfaces: bool,

    /// Whether wave overhang toolpath generation is enabled (Huygens wave propagation).
    #[arg(long, default_value_t = true)]
    wave_overhangs: bool,

    /// Overlap distance (mm) between adjacent wave overhang tracks.
    #[arg(long)]
    wave_overhang_overlap: Option<f64>,

    /// Speed (mm/s) for wave overhang printing moves.
    #[arg(long)]
    wave_overhang_speed: Option<f64>,

    /// Flow multiplier for wave overhang teardrop beads.
    #[arg(long)]
    wave_overhang_flow: Option<f64>,

    /// Part cooling fan speed percentage (0 to 100).
    #[arg(long)]
    fan_speed: Option<f64>,

    /// Overhang part cooling fan speed percentage (0 to 100).
    #[arg(long)]
    overhang_fan_speed: Option<f64>,

    /// Number of initial layers to keep part cooling fan disabled.
    #[arg(long)]
    fan_layer_delay: Option<u32>,

    /// Speed deadband percentage (e.g. 10.0%) for compacting G-code feedrate commands.
    #[arg(long)]
    speed_deadband: Option<f64>,

    /// Acceleration deadband percentage (e.g. 20.0%) for compacting acceleration commands.
    #[arg(long)]
    acceleration_deadband: Option<f64>,

    /// Klipper square corner velocity limit (mm/s).
    #[arg(long)]
    square_corner_velocity: Option<f64>,
}

/// Parses a `--eikonal-slope-profile` argument of comma-separated
/// `x:z` pairs into clearance points for `Machine::eikonal_slope_profile`.
fn parse_slope_profile(s: &str) -> Result<Vec<(f64, f64)>, String> {
    s.split(',')
        .map(|pair| {
            let (x_str, z_str) = pair
                .split_once(':')
                .ok_or_else(|| format!("expected `x:z`, got `{pair}`"))?;
            let x: f64 = x_str
                .trim()
                .parse()
                .map_err(|_| format!("invalid x radius `{x_str}` in `{pair}`"))?;
            let z: f64 = z_str
                .trim()
                .parse()
                .map_err(|_| format!("invalid z height `{z_str}` in `{pair}`"))?;
            Ok((x, z))
        })
        .collect()
}

/// CLI-mirror of `manifold_core::order_field::OrderFieldKind` so `clap` can
/// derive argument parsing. Converted into the library enum when building
/// `SlicerConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum OrderFieldArg {
    #[default]
    Height,
    Conical,
    Eikonal,
}

impl From<OrderFieldArg> for OrderFieldKind {
    fn from(arg: OrderFieldArg) -> Self {
        match arg {
            OrderFieldArg::Height => OrderFieldKind::Height,
            OrderFieldArg::Conical => OrderFieldKind::Conical,
            OrderFieldArg::Eikonal => OrderFieldKind::Eikonal,
        }
    }
}

/// CLI-mirror of `manifold_core::infill::InfillPatternKind` so `clap` can
/// derive argument parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum InfillPatternArg {
    Monotonic,
    Concentric,
    AllWalls,
    #[default]
    Cubic,
    None,
}

impl From<InfillPatternArg> for InfillPatternKind {
    fn from(arg: InfillPatternArg) -> Self {
        match arg {
            InfillPatternArg::Monotonic => InfillPatternKind::Monotonic,
            InfillPatternArg::Concentric => InfillPatternKind::Concentric,
            InfillPatternArg::AllWalls => InfillPatternKind::AllWalls,
            InfillPatternArg::Cubic => InfillPatternKind::Cubic,
            InfillPatternArg::None => InfillPatternKind::None,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    let mut objects = Vec::new();
    let mut next_object_id = 0u32;
    for entry in &cli.inputs {
        let (path, tool) = parse_input_entry(entry)?;
        tracing::info!(input = %path.display(), tool = tool.0, "loading mesh");
        objects.extend(load_objects(&path, tool, &mut next_object_id)?);
    }

    let eikonal_slope_profile = match &cli.eikonal_slope_profile {
        Some(s) => parse_slope_profile(s).map_err(|e| anyhow::anyhow!(e))?,
        None => Vec::new(),
    };

    let config = SlicerConfig {
        layer_height: cli.layer_height,
        nozzle_diameter: cli.nozzle_diameter,
        order_field: cli.order_field.into(),
        eikonal_conform_top_surfaces: cli.eikonal_conform_top_surfaces,
        wave_overhangs_enabled: cli.wave_overhangs,
        wave_overhang_overlap: cli.wave_overhang_overlap,
        wave_overhang_speed: cli.wave_overhang_speed.map(|s| s * 60.0),
        wave_overhang_flow: cli.wave_overhang_flow,
        fan_speed_percent: cli.fan_speed,
        overhang_fan_speed_percent: cli.overhang_fan_speed,
        fan_layer_delay: cli.fan_layer_delay,
        speed_deadband_percent: cli.speed_deadband,
        acceleration_deadband_percent: cli.acceleration_deadband,
        square_corner_velocity: cli.square_corner_velocity,
        sparse_infill_pattern: cli.sparse_infill_pattern.map(Into::into),
        solid_infill_pattern: cli.solid_infill_pattern.map(Into::into),
        infill_pattern: cli.infill_pattern.into(),
        ..SlicerConfig::default()
    };

    let mut machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        tools_for(&objects, cli.nozzle_diameter),
    );
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);
    machine.eikonal_slope_profile = eikonal_slope_profile;
    let workspace = Workspace::new(objects, machine, config);

    let gcode = slice_to_gcode(&workspace)?;
    std::fs::write(&cli.output, gcode)?;
    tracing::info!(output = %cli.output.display(), "wrote gcode");

    Ok(())
}

/// Parse one `inputs` entry: `path` or `path:tool`. The tool suffix must be
/// a valid `u32`; its absence defaults to tool `0`.
fn parse_input_entry(entry: &str) -> Result<(std::path::PathBuf, ToolId)> {
    match entry.rsplit_once(':') {
        Some((path, tool)) => {
            let tool: u32 = tool
                .parse()
                .with_context(|| format!("invalid tool id {tool:?} in input {entry:?}"))?;
            Ok((std::path::PathBuf::from(path), ToolId(tool)))
        }
        None => Ok((std::path::PathBuf::from(entry), ToolId(0))),
    }
}

/// One `Tool` per distinct tool id referenced by `objects`, sorted by id,
/// all sharing `nozzle_diameter` (per-tool nozzle diameters are a future
/// follow-up — see `Cli::nozzle_diameter`).
fn tools_for(objects: &[Object], nozzle_diameter: f64) -> Vec<Tool> {
    let mut tool_ids: Vec<ToolId> = objects.iter().map(|object| object.tool).collect();
    tool_ids.sort();
    tool_ids.dedup();
    tool_ids
        .into_iter()
        .map(|id| Tool::new(id, nozzle_diameter))
        .collect()
}

/// Load every object from `path`, dispatching on its file extension,
/// assigning them all to `tool` and allocating sequential `ObjectId`s
/// starting from `next_object_id`.
fn load_objects(path: &Path, tool: ToolId, next_object_id: &mut u32) -> Result<Vec<Object>> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match extension.as_str() {
        "3mf" => {
            let file =
                File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
            let mut objects = threemf::load_3mf(file, tool)?;
            for object in &mut objects {
                object.id = ObjectId(*next_object_id);
                *next_object_id += 1;
            }
            Ok(objects)
        }
        "stl" => {
            let file =
                File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
            let mesh = stl::load_stl(BufReader::new(file))?;
            let id = ObjectId(*next_object_id);
            *next_object_id += 1;
            Ok(vec![Object::new(id, mesh, tool)])
        }
        other => bail!(
            "unsupported input format {:?} for {}: only .3mf and .stl are supported today",
            other,
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_input_entry_defaults_to_tool_zero() {
        let (path, tool) = parse_input_entry("part.stl").unwrap();
        assert_eq!(path, std::path::PathBuf::from("part.stl"));
        assert_eq!(tool, ToolId(0));
    }

    #[test]
    fn parse_input_entry_reads_explicit_tool_suffix() {
        let (path, tool) = parse_input_entry("part.stl:2").unwrap();
        assert_eq!(path, std::path::PathBuf::from("part.stl"));
        assert_eq!(tool, ToolId(2));
    }

    #[test]
    fn parse_input_entry_rejects_non_numeric_tool_suffix() {
        assert!(parse_input_entry("part.stl:abc").is_err());
    }

    #[test]
    fn tools_for_deduplicates_and_sorts_referenced_tool_ids() {
        let objects = vec![
            Object::new(ObjectId(0), manifold_core::mesh::Mesh::default(), ToolId(2)),
            Object::new(ObjectId(1), manifold_core::mesh::Mesh::default(), ToolId(0)),
            Object::new(ObjectId(2), manifold_core::mesh::Mesh::default(), ToolId(2)),
        ];

        let tools = tools_for(&objects, 0.4);

        assert_eq!(
            tools.iter().map(|tool| tool.id).collect::<Vec<_>>(),
            vec![ToolId(0), ToolId(2)]
        );
        assert!(tools.iter().all(|tool| tool.nozzle_diameter == 0.4));
    }

    #[test]
    fn load_objects_allocates_sequential_ids_across_calls() {
        let ascii = b"solid triangle
            facet normal 0 0 1
                outer loop
                    vertex 0 0 0
                    vertex 10 0 0
                    vertex 5 10 0
                endloop
            endfacet
            endsolid triangle";
        let dir = std::env::temp_dir();
        let path = dir.join(format!("manifold_cli_test_{}.stl", std::process::id()));
        std::fs::write(&path, ascii).unwrap();

        let mut next_object_id = 5;
        let objects = load_objects(&path, ToolId(3), &mut next_object_id).unwrap();

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id, ObjectId(5));
        assert_eq!(objects[0].tool, ToolId(3));
        assert_eq!(next_object_id, 6);

        std::fs::remove_file(&path).unwrap();
    }
}
