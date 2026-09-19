//! `manifold` CLI: drives `manifold-core` headlessly to slice mesh(es) to
//! Gcode. Accepts multiple input files, each optionally suffixed with a
//! tool id (`path[:tool]`) for per-file tool assignment.

mod printer;

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
#[derive(Parser)]
#[command(name = "manifold", version = manifold_core::version::MANIFOLD_VERSION, about)]
struct Cli {
    /// Input mesh file(s) (STL or 3MF). Each entry may optionally suffix a
    /// tool id to assign that file's objects to, e.g. `part.stl:1`
    /// (defaults to tool `0` if omitted).
    #[arg(num_args = 0..)]
    inputs: Vec<String>,

    /// Output Gcode file.
    #[arg(short, long, default_value = "out.gcode")]
    output: std::path::PathBuf,

    /// Load machine + slicer settings from a saved profile JSON file (the
    /// same format `manifold-gui`'s Settings panel writes via its Save
    /// Profile action: a top-level `{"machine": ..., "config": ...}`
    /// object). When given, this fully replaces both the machine (build
    /// volume, tools, slope-profile clearance, etc.) and the slicer
    /// config that would otherwise be built from the individual flags
    /// below -- layer height, nozzle diameter, order field, infill
    /// settings, and every other slicing flag are all ignored in favor
    /// of the profile's own values. `--inputs`/`--output` and the
    /// printer upload flags still apply normally.
    #[arg(long)]
    profile: Option<std::path::PathBuf>,

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

    /// Wall/perimeter printing order within each island.
    #[arg(long, value_enum)]
    wall_order: Option<WallOrderArg>,

    /// Slope-limit profile for the `eikonal` order field, as comma-separated
    /// `height_mm:max_degrees` breakpoints (height measured above the
    /// mesh's build-plate contact surface), e.g. `0:45,4:2` for a tight
    /// 45deg limit near the build plate loosening to 2deg above 4mm.
    /// Ignored unless `--order-field eikonal`. Defaults to no limit
    /// (unconstrained) if omitted.
    #[arg(long)]
    eikonal_slope_profile: Option<String>,

    /// Whether the Eikonal order field blends with the top surface.
    /// Whether wave overhang toolpath generation is enabled (Huygens wave propagation).
    #[arg(long, default_value_t = true)]
    wave_overhangs: bool,

    /// Whether anisotropic FSM boundary metric tensor blending is enabled.
    #[arg(long, default_value_t = false)]
    fsm_boundary_metrics: bool,

    /// Aspect ratio for top surface tangency under the anisotropic FSM order field.
    #[arg(long)]
    fsm_top_tangency: Option<f64>,

    /// Aspect ratio for wall surface orthogonality under the anisotropic FSM order field.
    #[arg(long)]
    fsm_wall_ortho: Option<f64>,

    /// Subsurface skin depth (mm) for anisotropic FSM tensor blending.
    #[arg(long)]
    fsm_skin_depth: Option<f64>,

    /// Maximum sweep iterations for the anisotropic FSM solver.
    #[arg(long)]
    fsm_sweeps: Option<usize>,

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

    /// Target nozzle temperature in °C.
    #[arg(long)]
    nozzle_temp: Option<f64>,

    /// Target heated bed temperature in °C.
    #[arg(long)]
    bed_temp: Option<f64>,

    /// Target heated chamber temperature in °C.
    #[arg(long)]
    chamber_temp: Option<f64>,

    /// Enable unified thermodynamic and non-Newtonian fluid dynamics model.
    #[arg(long)]
    fluid_dynamics: bool,

    /// Enable time-based dynamic residual pressure flow compensation.
    #[arg(long)]
    transient_pressure_compensation: bool,

    /// Disable kinematic and geometric corner overlap flow compensation (enabled by default).
    #[arg(long)]
    no_corner_flow_compensation: bool,

    /// Compensation multiplier ratio for corner flow compensation (default 1.0).
    #[arg(long)]
    corner_flow_compensation_ratio: Option<f64>,

    /// Minimum compensation multiplier M_min for transient nozzle pressure flow compensation.
    #[arg(long)]
    transient_pressure_min_multiplier: Option<f64>,

    /// Sensitivity exponent beta for transient nozzle pressure flow compensation.
    #[arg(long)]
    transient_pressure_beta: Option<f64>,

    /// Static mechanical retraction distance (mm) when fluid dynamics model is enabled.
    #[arg(long)]
    static_retraction: Option<f64>,

    /// Emit periodic Moonraker-visible checkpoints (`// action:slicer_checkpoint
    /// {"num": N, "rem": R}`) reporting Manifold's own remaining-time estimate
    /// through the print, so it can be compared against the printer's actual
    /// progress via Moonraker's API.
    #[arg(long)]
    slicer_checkpoints: bool,

    /// Target spacing (seconds of Manifold's own estimated elapsed print time)
    /// between emitted checkpoints when --slicer-checkpoints is set (default 5.0).
    #[arg(long)]
    slicer_checkpoint_interval: Option<f64>,

    /// Moonraker printer URL (e.g. http://192.168.1.50:7125)
    #[arg(long)]
    printer_url: Option<String>,

    /// Moonraker API key (if authentication is enabled)
    #[arg(long)]
    printer_api_key: Option<String>,

    /// Upload the sliced Gcode to the printer via Moonraker
    #[arg(long)]
    upload: bool,

    /// Start printing immediately after upload
    #[arg(long)]
    print: bool,

    /// Tail print progress and temperatures in the terminal until completion
    #[arg(long)]
    monitor: bool,
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
    DualIso,
    AnisotropicFsm,
}

impl From<OrderFieldArg> for OrderFieldKind {
    fn from(arg: OrderFieldArg) -> Self {
        match arg {
            OrderFieldArg::Height => OrderFieldKind::Height,
            OrderFieldArg::Conical => OrderFieldKind::Conical,
            OrderFieldArg::Eikonal => OrderFieldKind::Eikonal,
            OrderFieldArg::DualIso => OrderFieldKind::DualIso,
            OrderFieldArg::AnisotropicFsm => OrderFieldKind::AnisotropicFsm,
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
    Gyroid,
    SchwarzD,
    SchwarzP,
    None,
}

impl From<InfillPatternArg> for InfillPatternKind {
    fn from(arg: InfillPatternArg) -> Self {
        match arg {
            InfillPatternArg::Monotonic => InfillPatternKind::Monotonic,
            InfillPatternArg::Concentric => InfillPatternKind::Concentric,
            InfillPatternArg::AllWalls => InfillPatternKind::AllWalls,
            InfillPatternArg::Cubic => InfillPatternKind::Cubic,
            InfillPatternArg::Gyroid => InfillPatternKind::Gyroid,
            InfillPatternArg::SchwarzD => InfillPatternKind::SchwarzD,
            InfillPatternArg::SchwarzP => InfillPatternKind::SchwarzP,
            InfillPatternArg::None => InfillPatternKind::None,
        }
    }
}

/// CLI-mirror of `manifold_core::WallOrder` so `clap` can derive argument parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum WallOrderArg {
    #[default]
    InnerOuterInner,
    OutsideIn,
}

impl From<WallOrderArg> for manifold_core::WallOrder {
    fn from(arg: WallOrderArg) -> Self {
        match arg {
            WallOrderArg::InnerOuterInner => manifold_core::WallOrder::InnerOuterInner,
            WallOrderArg::OutsideIn => manifold_core::WallOrder::OutsideIn,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let intent = printer::intent(&cli)?;
    if intent == printer::Intent::Monitor {
        let session = manifold_printer::PrinterSessionHandle::spawn(printer::config(&cli));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        return rt.block_on(printer::interruptible(
            printer::monitor(&session, None),
            &std::sync::atomic::AtomicBool::new(false),
        ));
    }

    let mut objects = Vec::new();
    let mut next_object_id = 0u32;
    for entry in &cli.inputs {
        let (path, tool) = parse_input_entry(entry)?;
        tracing::info!(input = %path.display(), tool = tool.0, "loading mesh");
        objects.extend(load_objects(&path, tool, &mut next_object_id)?);
    }

    let (config, mut machine) = resolve_config_and_machine(&cli, &objects)?;
    manifold_core::object::center_on_bed(&mut objects, &machine.build_volume);
    if let Some(s) = &cli.eikonal_slope_profile {
        machine.eikonal_slope_profile = parse_slope_profile(s).map_err(|e| anyhow::anyhow!(e))?;
    }
    let workspace = Workspace::new(objects, machine, config);

    let gcode = slice_to_gcode(&workspace)?;
    std::fs::write(&cli.output, &gcode)?;
    tracing::info!(output = %cli.output.display(), "wrote gcode");

    if let printer::Intent::Upload { start, monitor } = intent {
        let config = printer::config(&cli);
        let client = manifold_printer::MoonrakerClient::new(config.clone())?;
        let session = monitor.then(|| manifold_printer::PrinterSessionHandle::spawn(config));
        let filename = cli
            .output
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("out.gcode");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let mutation_in_flight = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress_state = mutation_in_flight.clone();
        rt.block_on(printer::interruptible(
            async {
                if let Some(session) = &session {
                    printer::wait_ready(session).await?;
                }
                let before_start = session.as_ref().map(|s| s.latest_telemetry());
                eprintln!(
                    "Uploading to {}. A transmitted request cannot be recalled by exiting.",
                    client.base_url()
                );
                let outcome = client
                    .upload_gcode_with_progress(
                        filename,
                        gcode.into_bytes(),
                        start,
                        Some(std::sync::Arc::new(move |_, _| {
                            progress_state.store(true, std::sync::atomic::Ordering::SeqCst);
                        })),
                    )
                    .await?;
                mutation_in_flight.store(false, std::sync::atomic::Ordering::SeqCst);
                eprintln!(
                    "{:?}: {}/{}",
                    outcome.disposition, outcome.root, outcome.path
                );
                if start {
                    printer::require_started(&outcome)?;
                }
                if let (Some(session), Some(before_start)) = (&session, before_start) {
                    printer::monitor(session, Some((outcome.path, before_start))).await?;
                }
                Ok::<(), anyhow::Error>(())
            },
            &mutation_in_flight,
        ))?;
    }

    Ok(())
}

/// Resolves the `SlicerConfig` + `Machine` to use for this run: loaded
/// wholesale from `--profile` when given (see `Cli::profile`'s doc
/// comment for why the individual slicing flags are ignored in that
/// case), or built from `cli`'s individual flags otherwise, exactly as
/// before `--profile` existed.
///
/// Extracted as its own function (rather than inlined in `main`) so it's
/// directly unit-testable without exercising `main`'s I/O side effects
/// (writing Gcode, uploading, etc.).
fn resolve_config_and_machine(cli: &Cli, objects: &[Object]) -> Result<(SlicerConfig, Machine)> {
    if let Some(profile_path) = &cli.profile {
        let json: serde_json::Value = serde_json::from_reader(BufReader::new(
            File::open(profile_path)
                .with_context(|| format!("opening profile {}", profile_path.display()))?,
        ))
        .with_context(|| format!("parsing profile {} as JSON", profile_path.display()))?;
        let machine: Machine =
            serde_json::from_value(json["machine"].clone()).with_context(|| {
                format!(
                    "reading \"machine\" from profile {}",
                    profile_path.display()
                )
            })?;
        let config: SlicerConfig =
            serde_json::from_value(json["config"].clone()).with_context(|| {
                format!("reading \"config\" from profile {}", profile_path.display())
            })?;
        return Ok((config, machine));
    }

    let config = SlicerConfig {
        layer_height: cli.layer_height,
        nozzle_diameter: cli.nozzle_diameter,
        order_field: cli.order_field.into(),
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
        default_nozzle_temperature: cli.nozzle_temp,
        bed_temperature: cli.bed_temp,
        chamber_temperature: cli.chamber_temp,
        fluid_dynamics: if cli.fluid_dynamics || cli.static_retraction.is_some() {
            let mut cfg = manifold_core::fluid_dynamics::FluidDynamicsConfig::default();
            if let Some(sr) = cli.static_retraction {
                cfg.static_retraction_mm = sr;
            }
            Some(cfg)
        } else {
            None
        },
        sparse_infill_pattern: cli.sparse_infill_pattern.map(Into::into),
        solid_infill_pattern: cli.solid_infill_pattern.map(Into::into),
        infill_pattern: cli.infill_pattern.into(),
        wall_order: cli.wall_order.map(Into::into),
        fsm_boundary_metrics_enabled: cli.fsm_boundary_metrics,
        fsm_top_tangency_aspect: cli.fsm_top_tangency,
        fsm_wall_ortho_aspect: cli.fsm_wall_ortho,
        fsm_skin_depth_mm: cli.fsm_skin_depth,
        fsm_max_sweeps: cli.fsm_sweeps,
        enable_corner_flow_compensation: !cli.no_corner_flow_compensation,
        corner_flow_compensation_ratio: cli.corner_flow_compensation_ratio,
        enable_transient_pressure_compensation: cli.transient_pressure_compensation,
        transient_pressure_min_multiplier: cli.transient_pressure_min_multiplier,
        transient_pressure_beta: cli.transient_pressure_beta,
        enable_slicer_checkpoints: cli.slicer_checkpoints,
        slicer_checkpoint_interval_seconds: cli.slicer_checkpoint_interval,
        ..SlicerConfig::default()
    };
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::ZERO,
            max: DVec3::new(200.0, 200.0, 200.0),
        },
        tools_for(objects, cli.nozzle_diameter, cli.nozzle_temp),
    );
    Ok((config, machine))
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
fn tools_for(objects: &[Object], nozzle_diameter: f64, nozzle_temp: Option<f64>) -> Vec<Tool> {
    let mut tool_ids: Vec<ToolId> = objects.iter().map(|object| object.tool).collect();
    tool_ids.sort();
    tool_ids.dedup();
    tool_ids
        .into_iter()
        .map(|id| {
            let mut tool = Tool::new(id, nozzle_diameter);
            tool.nozzle_temperature = nozzle_temp;
            tool
        })
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

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("object")
        .to_string();

    match extension.as_str() {
        "3mf" => {
            let file =
                File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
            let mut objects = threemf::load_3mf(file, tool)?;
            let multiple = objects.len() > 1;
            for (idx, object) in objects.iter_mut().enumerate() {
                object.id = ObjectId(*next_object_id);
                *next_object_id += 1;
                if object.name.is_none() {
                    object.name = Some(if multiple {
                        format!("{} #{}", stem, idx + 1)
                    } else {
                        stem.clone()
                    });
                }
            }
            Ok(objects)
        }
        "stl" => {
            let file =
                File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
            let mesh = stl::load_stl(BufReader::new(file))?;
            let id = ObjectId(*next_object_id);
            *next_object_id += 1;
            let mut obj = Object::new(id, mesh, tool);
            obj.name = Some(stem);
            Ok(vec![obj])
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
    fn monitor_only_accepts_no_mesh_input() {
        let cli = Cli::try_parse_from([
            "manifold",
            "--monitor",
            "--printer-url",
            "http://127.0.0.1:7125",
        ]);
        assert!(cli.is_ok(), "monitor-only must not require a mesh");
    }

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

        let tools = tools_for(&objects, 0.4, Some(230.0));

        assert_eq!(
            tools.iter().map(|tool| tool.id).collect::<Vec<_>>(),
            vec![ToolId(0), ToolId(2)]
        );
        assert!(tools.iter().all(|tool| tool.nozzle_diameter == 0.4));
        assert!(tools.iter().all(|tool| tool.nozzle_temperature() == 230.0));
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

    #[test]
    fn test_cli_printer_args() {
        use clap::Parser;
        let args = vec![
            "manifold",
            "model.stl",
            "--printer-url",
            "http://192.168.1.50:7125",
            "--upload",
            "--print",
            "--monitor",
        ];
        let cli = Cli::parse_from(args);
        assert_eq!(cli.printer_url.as_deref(), Some("http://192.168.1.50:7125"));
        assert!(cli.upload);
        assert!(cli.print);
        assert!(cli.monitor);
    }

    #[test]
    fn cli_accepts_profile_flag() {
        use clap::Parser;
        let args = vec!["manifold", "model.stl", "--profile", "profile.json"];
        let cli = Cli::parse_from(args);
        assert_eq!(
            cli.profile.as_deref(),
            Some(std::path::Path::new("profile.json"))
        );
    }

    /// A trimmed but structurally faithful copy of a real profile saved by
    /// `manifold-gui`'s Settings panel (the exact bug report this feature
    /// fixes: the CLI previously had no way to load this file at all).
    /// Keeps every field `Machine`/`Tool` require (no `#[serde(default)]`)
    /// plus a handful of `config` values distinctive enough to prove real
    /// fields actually round-trip, not just that parsing doesn't error.
    const SAMPLE_PROFILE_JSON: &str = r#"{
        "machine": {
            "substrate_transform": [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0, 0.0,0.0,0.0],
            "build_volume": {
                "kind": "Aabb",
                "min": [0.0, 0.0, 0.0],
                "max": [350.0, 350.0, 300.0]
            },
            "tools": [
                {
                    "id": 0,
                    "nozzle_diameter": 0.4,
                    "mount": [1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0, 0.0,0.0,0.0],
                    "collision_envelope": { "kind": "Sphere", "center": [0.0,0.0,0.0], "radius": 0.0 },
                    "extrusion_multiplier": 1.05
                }
            ],
            "axis_count": 3
        },
        "config": {
            "description": "Voron 2.4",
            "layer_height": 0.2,
            "first_layer_height": null,
            "first_layer_print_speed": 4800.0,
            "first_layer_extrusion_multiplier": 1.0,
            "first_layer_line_width": 0.4,
            "fan_speed_percent": 35.0,
            "overhang_fan_speed_percent": 66.0,
            "fan_layer_delay": null,
            "nozzle_diameter": 0.4,
            "object_ordering": "Sequential",
            "wall_line_width": 0.4,
            "shell_thickness": 1.2,
            "wall_offset": 0.2,
            "slope_compensation_mode": "VolumetricModulation",
            "wall_order": "OutsideIn",
            "min_bead_width_ratio": null,
            "max_bead_width_ratio": null,
            "bead_clearance_compensation_enabled": null,
            "sparse_infill_pattern": "SchwarzD",
            "solid_infill_pattern": "AllWalls",
            "infill_pattern": "Cubic",
            "infill_line_width": 0.4,
            "infill_angle_deg": 45.0,
            "infill_density": 0.2,
            "top_layers": 3,
            "bottom_layers": 3,
            "order_field": "AnisotropicFsm",
            "eikonal_surface_order_weight": 0.0,
            "eikonal_conform_top_surfaces": false,
            "eikonal_enforce_monotonic_growth": false,
            "eikonal_conform_bottom_surfaces": false,
            "eikonal_conformal_max_angle_deg": null,
            "eikonal_conformal_bottom_max_angle_deg": 10.0,
            "eikonal_conformal_skin_depth_mm": null,
            "order_field_apex": [0.0, 0.0, 0.0],
            "order_field_axis": [0.0, 0.0, 1.0],
            "order_field_slope": 0.0,
            "filament_density_g_cm3": null,
            "filament_diameter": 1.75,
            "start_gcode": "",
            "end_gcode": "",
            "travel_speed": 41400.0,
            "print_speed": 22200.0,
            "outer_wall_speed": 21600.0,
            "inner_wall_speed": 21600.0,
            "infill_speed": 21600.0,
            "solid_infill_speed": 21600.0,
            "bridge_speed": 7200.0,
            "default_acceleration": 7000.0,
            "outer_wall_acceleration": 5000.0,
            "inner_wall_acceleration": 5000.0,
            "infill_acceleration": 7000.0,
            "travel_acceleration": 7000.0,
            "first_layer_acceleration": 5000.0,
            "max_volumetric_speed": 35.0,
            "pressure_advance": 0.034,
            "pre_retract_taper_distance": 1.0,
            "min_travel_for_retract": 1.3,
            "retraction_length": 0.7,
            "retraction_speed": 2400.0,
            "unretract_speed": null,
            "unretract_extra_length": 0.0,
            "wipe_distance": 2.0,
            "wipe_enabled": true,
            "use_firmware_retraction": false,
            "scarf_joint_enabled": false,
            "scarf_joint_length": 4.5,
            "scarf_joint_steps": 18,
            "scarf_joint_start_height_fraction": 0.25,
            "scarf_joint_flow_ratio": 0.8,
            "seam_gap": 0.4,
            "z_hop_enabled": false,
            "z_hop_height": 0.02,
            "path_simplify_enabled": false,
            "path_simplify_tolerance": 0.04,
            "nozzle_flat_diameter": null,
            "travel_order_optimization_enabled": true,
            "travel_collision_avoidance_enabled": true,
            "z_travel_penalty": 8.0,
            "wave_overhangs_enabled": true,
            "wave_overhang_overlap": 0.1,
            "wave_overhang_speed": 3000.0,
            "wave_overhang_flow": 1.0,
            "speed_deadband_percent": 10.0,
            "acceleration_deadband_percent": 20.0,
            "square_corner_velocity": 6.0,
            "default_nozzle_temperature": 255.0,
            "bed_temperature": 105.0,
            "chamber_temperature": 55.0,
            "fluid_dynamics": {
                "pa_calibration_low": [0.049, 6.5],
                "pa_calibration_high": [0.031, 32.0],
                "heater_block_temp_c": 240.0,
                "reference_temp_c": 240.0,
                "max_fan_temp_drop_c": 8.0,
                "ooze_time_constant_ref_s": 0.1,
                "ooze_max_length_ref_mm": 0.0,
                "static_retraction_mm": 0.6,
                "max_retraction_mm": 1.5,
                "pa_deadband": 0.1,
                "swell_ratio_low": 1.0,
                "swell_ratio_high": 1.0
            },
            "minimum_cruise_ratio": null,
            "enable_corner_flow_compensation": true,
            "corner_flow_compensation_ratio": 1.0,
            "enable_transient_pressure_compensation": false,
            "transient_pressure_min_multiplier": 0.95,
            "transient_pressure_beta": 0.5,
            "enable_slicer_pressure_advance": true,
            "slicer_pa_tolerance_mm": 0.005,
            "slicer_pa_min_segment_length": 0.4,
            "slicer_pa_max_frequency_hz": 960.0,
            "fsm_boundary_metrics_enabled": true,
            "fsm_top_tangency_aspect": 0.4,
            "fsm_wall_ortho_aspect": 0.8,
            "fsm_skin_depth_mm": null,
            "fsm_max_sweeps": 13,
            "fsm_seed_surfaces_enabled": true,
            "fsm_seed_max_angle_deg": null,
            "end_of_print_wipe_enabled": true,
            "end_of_print_wipe_distance": null,
            "end_of_print_clearance_z_lift": null,
            "enable_slicer_checkpoints": false,
            "slicer_checkpoint_interval_seconds": null
        },
        "moonraker": {
            "url": "http://example.local:7125/",
            "api_key": null,
            "auto_connect": true
        }
    }"#;

    fn write_temp_profile(json: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "manifold_cli_profile_test_{}_{}.json",
            std::process::id(),
            seq
        ));
        std::fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn resolve_config_and_machine_loads_a_real_saved_profile() {
        let path = write_temp_profile(SAMPLE_PROFILE_JSON);
        let cli = Cli::parse_from(["manifold", "--profile", path.to_str().unwrap()]);

        let (config, machine) = resolve_config_and_machine(&cli, &[]).unwrap();

        assert_eq!(config.order_field, OrderFieldKind::AnisotropicFsm);
        assert_eq!(config.wall_offset, 0.2);
        assert_eq!(config.shell_thickness, 1.2);
        assert_eq!(config.bottom_layers, 3);
        assert_eq!(config.top_layers, 3);
        assert_eq!(config.infill_density, 0.2);
        assert_eq!(
            machine.build_volume,
            BoundingVolume::Aabb {
                min: DVec3::ZERO,
                max: DVec3::new(350.0, 350.0, 300.0),
            }
        );
        assert_eq!(machine.tools.len(), 1);
        assert_eq!(machine.tools[0].nozzle_diameter, 0.4);

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn resolve_config_and_machine_profile_overrides_individual_flags() {
        // The profile's own layer_height (0.2) must win over a conflicting
        // --layer-height flag passed alongside --profile -- see Cli::profile's
        // doc comment: when a profile is given, the individual slicing
        // flags are ignored entirely, not merged.
        let path = write_temp_profile(SAMPLE_PROFILE_JSON);
        let cli = Cli::parse_from([
            "manifold",
            "--profile",
            path.to_str().unwrap(),
            "--layer-height",
            "0.35",
        ]);

        let (config, _machine) = resolve_config_and_machine(&cli, &[]).unwrap();
        assert_eq!(
            config.layer_height, 0.2,
            "profile's layer_height must win over a conflicting --layer-height flag"
        );

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn resolve_config_and_machine_reports_a_clear_error_for_a_missing_profile() {
        let cli = Cli::parse_from(["manifold", "--profile", "/nonexistent/profile.json"]);
        let err = resolve_config_and_machine(&cli, &[]).unwrap_err();
        assert!(
            err.to_string().contains("profile"),
            "error should mention the profile path, got: {err}"
        );
    }
}
