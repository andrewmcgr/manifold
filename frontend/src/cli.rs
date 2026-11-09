use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::arrange::{auto_arrange_spiral, ArrangeItem};
use crate::bridge::ffi;
use crate::calibration::{
    generate_em_calibration_gcode, generate_pa_calibration_gcode,
    generate_transient_pressure_calibration_gcode, EmCalibrationPass, EmCalibrationSpec,
    PaCalibrationMode, PaCalibrationSpec, TransientCalibrationPass, TransientCalibrationSpec,
};
use crate::config::{SlicerConfig, PARAMETER_DOCS};
use crate::gcode::{
    expand_gcode_template, format_exclude_object_define, format_moonraker_estimated_time,
    plan_and_emit_gcode_for_object, render_toolpath_preview_thumbnails, PlannedGcodeResult,
    PlannedLayerGcode,
};
use crate::io::threemf::{save_project_to_3mf, ExportableObject};
use crate::io::{load_files_into_backend, LoadedObjectData};
use crate::presets::{
    FilamentPreset, PresetCategory, PresetLibrary, PrinterPreset, ProcessPreset, ToolPreset,
};
use crate::ui::sync_config_to_backend;

/// Output serialization format for profile inspection and generation commands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileFormat {
    Json,
    Toml,
}

impl ProfileFormat {
    pub fn from_path_or_default(path: Option<&Path>, explicit: Option<Self>) -> Self {
        if let Some(fmt) = explicit {
            return fmt;
        }
        if let Some(p) = path {
            if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                if ext.eq_ignore_ascii_case("toml") {
                    return Self::Toml;
                }
                if ext.eq_ignore_ascii_case("json") {
                    return Self::Json;
                }
            }
        }
        Self::Json
    }

    pub fn parse_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "toml" => Ok(Self::Toml),
            other => Err(format!(
                "Unsupported profile format '{other}' (expected 'json' or 'toml')"
            )),
        }
    }
}

/// Options when launching the interactive GUI from the command line.
#[derive(Clone, Debug, Default)]
pub struct GuiLaunchOptions {
    pub initial_files: Vec<PathBuf>,
    pub initial_config: Option<SlicerConfig>,
}

/// Result of parsing command-line arguments in `main()`.
pub enum CliDispatch {
    /// Launch the interactive 3D Bevy/egui GUI (optionally preloaded with files/config).
    LaunchGui(GuiLaunchOptions),
    /// Executed a headless CLI command; exit the process with this status code.
    Exit(i32),
}

/// Entry point called at the start of `main()`.
/// If no CLI arguments are passed (or `gui` is invoked), returns `CliDispatch::LaunchGui`.
/// Otherwise executes the requested headless CLI command (`slice`, `profile`, `help`, etc.)
/// and returns `CliDispatch::Exit(code)`.
pub fn handle_cli_args_from_env() -> CliDispatch {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run_cli_with_args(&args)
}

pub fn run_cli_with_args(args: &[String]) -> CliDispatch {
    if args.is_empty() {
        return CliDispatch::LaunchGui(GuiLaunchOptions::default());
    }

    let first = args[0].as_str();
    match first {
        "-h" | "--help" | "help" => {
            print_top_level_help();
            CliDispatch::Exit(0)
        }
        "-V" | "--version" | "version" => {
            println!("manifold-frontend {}", env!("CARGO_PKG_VERSION"));
            CliDispatch::Exit(0)
        }
        "gui" => match parse_gui_args(&args[1..]) {
            Ok(opts) => CliDispatch::LaunchGui(opts),
            Err(err) => {
                eprintln!("Error: {err}");
                CliDispatch::Exit(2)
            }
        },
        "slice" | "s" => match cmd_slice(&args[1..]) {
            Ok(()) => CliDispatch::Exit(0),
            Err(err) => {
                eprintln!("Error: {err}");
                CliDispatch::Exit(1)
            }
        },
        "profile" | "profiles" | "preset" | "presets" | "config" => {
            match cmd_profile(&args[1..]) {
                Ok(()) => CliDispatch::Exit(0),
                Err(err) => {
                    eprintln!("Error: {err}");
                    CliDispatch::Exit(1)
                }
            }
        }
        "calibrate" | "cal" => match cmd_calibrate(&args[1..]) {
            Ok(()) => CliDispatch::Exit(0),
            Err(err) => {
                eprintln!("Error: {err}");
                CliDispatch::Exit(1)
            }
        },
        // Convenience shorthand: if the user passes flags like `-p profile.json -o out.gcode part.stl`
        // directly without the `slice` verb, treat it as `slice` when `-o`/`--output`/`--slice` is present,
        // or launch GUI with the loaded file(s) if only CAD/mesh/3MF paths are given.
        _ => {
            let has_slice_flag = args.iter().any(|a| {
                matches!(
                    a.as_str(),
                    "-o" | "--output" | "--slice" | "--save-3mf" | "--plate" | "--arrange"
                ) || a.starts_with("--output=")
            });
            if has_slice_flag {
                let filtered: Vec<String> = args
                    .iter()
                    .filter(|a| a.as_str() != "--slice")
                    .cloned()
                    .collect();
                match cmd_slice(&filtered) {
                    Ok(()) => CliDispatch::Exit(0),
                    Err(err) => {
                        eprintln!("Error: {err}");
                        CliDispatch::Exit(1)
                    }
                }
            } else if args.iter().all(|a| is_supported_3d_file_arg(a)) {
                CliDispatch::LaunchGui(GuiLaunchOptions {
                    initial_files: args.iter().map(PathBuf::from).collect(),
                    initial_config: None,
                })
            } else {
                // Default to attempting `slice` if any 3D file is present alongside `--profile` / `--set`
                if args.iter().any(|a| is_supported_3d_file_arg(a)) {
                    match cmd_slice(args) {
                        Ok(()) => CliDispatch::Exit(0),
                        Err(err) => {
                            eprintln!("Error: {err}");
                            CliDispatch::Exit(1)
                        }
                    }
                } else {
                    eprintln!("Error: Unknown command or option '{first}'.\n");
                    print_top_level_help();
                    CliDispatch::Exit(2)
                }
            }
        }
    }
}

fn is_supported_3d_file_arg(arg: &str) -> bool {
    if arg.starts_with('-') {
        return false;
    }
    let ext = Path::new(arg)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(ext.as_str(), "step" | "stp" | "stl" | "3mf")
}

pub fn print_top_level_help() {
    println!(
        concat!(
            "Manifold Slicer — Multi-Axis Non-Planar Additive Manufacturing Slicer & CLI\n\n",
            "USAGE:\n",
            "  manifold-frontend                         Launch interactive 3D GUI\n",
            "  manifold-frontend gui [OPTIONS] [FILES]   Launch GUI with preloaded profile/files\n",
            "  manifold-frontend slice [OPTIONS] <FILES> Slice STEP/STL/3MF objects to G-code\n",
            "  manifold-frontend calibrate <KIND> [...]  Generate EM/Swell, PA, or Transient calibration G-code\n",
            "  manifold-frontend profile <VERB> [...]    Inspect, create, diff, and edit profiles/presets\n\n",
            "SLICE OPTIONS (`manifold-frontend slice --help` for full details):\n",
            "  -p, --profile <FILE>          Load full or partial profile (.json or .toml; repeatable)\n",
            "      --printer <NAME_OR_FILE>  Apply a Printer preset by name or file path\n",
            "      --tool <NAME_OR_FILE>     Apply a Tool preset by name or file path\n",
            "      --filament <NAME_OR_FILE> Apply a Filament preset by name or file path\n",
            "      --process <NAME_OR_FILE>  Apply a Process preset by name or file path\n",
            "  -s, --set <KEY=VALUE>         Override any profile parameter (repeatable, e.g.\n",
            "                                `-s slicing.layer_height.target_mm=0.16`)\n",
            "  -o, --output <PATH>           Output .gcode file (or directory for multi-plate, or `-`)\n",
            "      --save-3mf <PATH>         Save the configured & arranged project as a .3mf archive\n",
            "      --plate <INDEX_OR_NAME>   Slice only a specific 1-based plate index or plate name\n",
            "      --arrange / --no-arrange  Enable or disable spiral auto-arrangement on each plate\n",
            "      --place-on-face <INDEX>   Orient imported object(s) onto convex-hull face #INDEX\n",
            "      --rotate <RX,RY,RZ>       Apply Euler XYZ rotation in degrees before slicing\n",
            "  -q, --quiet                   Suppress progress output on stderr\n\n",
            "PROFILE & PRESET VERBS (`manifold-frontend profile --help` for full details):\n",
            "  profile init [OUT_FILE]       Create a new .json or .toml profile from presets/overrides\n",
            "  profile show [FILE_OR_PRESET] Display a profile or preset (or `--key <PATH>`)\n",
            "  profile get  <SOURCE> <KEY>   Print the value of a dotted parameter key\n",
            "  profile set  <TARGET> <K=V>.. Edit one or more parameters in a profile file or preset\n",
            "  profile diff <A> [B]          Show parameters that differ from defaults (or between A & B)\n",
            "  profile keys [--filter <STR>] List all editable profile keys, defaults, units & docs\n",
            "  profile list [--category <C>] List built-in and user-saved presets\n",
            "  profile save-preset <C> <N>   Save a user preset (printer|tool|filament|process)\n",
            "  profile delete-preset <C> <N> Delete a user-saved preset\n",
            "  profile validate <FILE>       Validate a .json or .toml profile file\n"
        )
    );
}

fn print_slice_help() {
    println!(
        concat!(
            "Manifold Slicer — Headless Slicing (`manifold-frontend slice`)\n\n",
            "USAGE:\n",
            "  manifold-frontend slice [OPTIONS] <INPUT_FILES>...\n\n",
            "INPUT FILES:\n",
            "  One or more `.step`, `.stp`, `.stl`, or `.3mf` files (including multi-plate `.3mf`\n",
            "  projects from Manifold, OrcaSlicer, Bambu Studio, or PrusaSlicer).\n\n",
            "OPTIONS:\n",
            "  -p, --profile <FILE>          Apply a full or partial JSON/TOML profile (repeatable).\n",
            "                                Later `--profile` and `--set` flags override earlier ones.\n",
            "      --printer <NAME_OR_FILE>  Select a Printer preset by name or load from `.json`/`.toml`\n",
            "      --tool <NAME_OR_FILE>     Select a Tool preset by name or load from `.json`/`.toml`\n",
            "      --filament <NAME_OR_FILE> Select a Filament preset by name or load from `.json`/`.toml`\n",
            "      --process <NAME_OR_FILE>  Select a Process preset by name or load from `.json`/`.toml`\n",
            "  -s, --set <KEY=VALUE>         Override any dotted (or unambiguous leaf) parameter key\n",
            "  -D <KEY=VALUE>                Alias for `--set <KEY=VALUE>`\n",
            "  -o, --output <PATH>           Destination `.gcode` file path, directory (when slicing\n",
            "                                multiple plates), or `-` for stdout.\n",
            "      --save-3mf <PATH>         Also save the arranged & configured project to a `.3mf` file\n",
            "      --plate <INDEX_OR_NAME>   Slice only the specified 1-based plate number (e.g. `1`)\n",
            "                                or plate name.\n",
            "      --arrange                 Force spiral auto-arrangement on each plate before slicing\n",
            "      --no-arrange              Disable automatic overlap arrangement\n",
            "      --place-on-face <INDEX>   Align hull face #INDEX onto the print bed (`z = 0`)\n",
            "      --rotate <RX,RY,RZ>       Rotate imported objects by Euler XYZ degrees (e.g. `0,0,45`)\n",
            "  -q, --quiet                   Suppress progress messages on stderr\n\n",
            "EXAMPLES:\n",
            "  manifold-frontend slice -p my_profile.json part.step -o part.gcode\n",
            "  manifold-frontend slice --filament \"Generic PETG\" -s infill_density_percent=30 part.stl\n",
            "  manifold-frontend slice \"Test Objects/Iron heart MOD (deepergrain).3mf\" --plate 1 -o plate1.gcode\n"
        )
    );
}

fn print_profile_help() {
    println!(
        concat!(
            "Manifold Slicer — Profile & Preset Management (`manifold-frontend profile`)\n\n",
            "USAGE:\n",
            "  manifold-frontend profile <VERB> [OPTIONS] [ARGS]\n\n",
            "VERBS:\n",
            "  init [OUTPUT_FILE]            Create a new `.json` (or `.toml`) profile file.\n",
            "                                Options: `--format json|toml`, `--from <FILE>`,\n",
            "                                `--printer <P>`, `--tool <T>`, `--filament <F>`,\n",
            "                                `--process <PR>`, `--category <CAT>`, `-s/--set <K=V>`.\n",
            "  show [FILE_OR_PRESET]         Print a profile file, named preset, or default config.\n",
            "                                Options: `--key <PATH>`, `--category <CAT>`, `--format json|toml`.\n",
            "  get <FILE_OR_PRESET> <KEY>    Read a single parameter (or sub-object) by dotted path\n",
            "                                or unambiguous leaf key name.\n",
            "  set <TARGET> <KEY=VALUE>...   Modify one or more parameters in a `.json`/`.toml` profile\n",
            "                                file (or user preset). Use `-o <OUT_FILE>` to write to a\n",
            "                                new file instead of updating `<TARGET>` in place.\n",
            "  diff <A> [B]                  Compare profile/preset `<A>` against defaults (or `<B>`).\n",
            "  keys [--filter <SUBSTR>]      List all parameter keys, default values, units, and docs.\n",
            "  list [--category <CAT>]       List all built-in factory and user-saved presets.\n",
            "  save-preset <CAT> <NAME>      Save a user preset (`printer`, `tool`, `filament`, `process`)\n",
            "                                with optional `--from <FILE>` and `-s/--set <K=V>`.\n",
            "  delete-preset <CAT> <NAME>    Delete a user-saved preset.\n",
            "  validate <FILE>               Check that a `.json` or `.toml` profile parses and sanitizes.\n\n",
            "EXAMPLES:\n",
            "  manifold-frontend profile init profile.json --filament \"Generic PETG\" -s layer_height.target_mm=0.16\n",
            "  manifold-frontend profile get profile.json slicing.layer_height.target_mm\n",
            "  manifold-frontend profile set profile.json gcode.bed_temperature_celsius=80 infill_density_percent=25\n",
            "  manifold-frontend profile diff profile.json\n",
            "  manifold-frontend profile keys --filter overhang\n"
        )
    );
}

fn print_calibrate_help() {
    println!(
        concat!(
            "Manifold Slicer — Rheology & Extrusion Calibration (`manifold-frontend calibrate`)\n\n",
            "USAGE:\n",
            "  manifold-frontend calibrate <em|pa|transient> [OPTIONS] [-o <OUT.gcode>]\n\n",
            "KINDS:\n",
            "  em | flow | swell             Orca-style surface quality coupon array (9 coupons:\n",
            "                                Pass 1 Coarse EM, Pass 2 Fine EM, Pass 3 High-Flow Die Swell).\n",
            "  pa | pressure-advance         Ellis-style 90° sharp-corner chevron pattern\n",
            "                                (Dual-Band Shear-Thinning k_low + k_high or Single-Band).\n",
            "  transient | melt-pressure     Constant-Speed Flow-Step & Staccato Striation pattern\n",
            "                                (Δv = 0 across X = 0 step-down; Pass 1 γ, Pass 2 τ_melt).\n\n",
            "OPTIONS:\n",
            "  -p, --profile <FILE>          Load full or partial profile (.json or .toml; repeatable)\n",
            "      --printer <NAME_OR_FILE>  Select a Printer preset\n",
            "      --tool <NAME_OR_FILE>     Select a Tool preset\n",
            "      --filament <NAME_OR_FILE> Select a Filament preset\n",
            "      --process <NAME_OR_FILE>  Select a Process preset\n",
            "  -s, --set <KEY=VALUE>         Override any profile parameter (repeatable)\n",
            "      --pass <1|2|3>            Pass number (`1`, `2`, or `3` for `em`; `1` or `2` for `transient`)\n",
            "      --mode <dual|single>      PA mode (`dual` for k_low + k_high or `single`)\n",
            "      --start <VAL>             Sweep start value (PA s, γ, or τ_melt s)\n",
            "      --end <VAL>               Sweep end value (PA s, γ, or τ_melt s)\n",
            "      --step <VAL>              Sweep step size (PA s, γ, or τ_melt s)\n",
            "      --speed <MM_S>            Fast print speed (`pa` single-band) or constant speed (`transient`)\n",
            "      --accel <MM_S2>           Corner acceleration for `pa` pattern (mm/s²)\n",
            "  -o, --output <PATH>           Destination `.gcode` file path (or `-` for stdout)\n",
            "  -q, --quiet                   Suppress summary output on stderr\n\n",
            "EXAMPLES:\n",
            "  manifold-frontend calibrate em --pass 1 --filament \"Generic PETG\" -o em_pass1.gcode\n",
            "  manifold-frontend calibrate em --pass 3 -o em_pass3_swell.gcode\n",
            "  manifold-frontend calibrate pa --mode dual --start 0.0 --end 0.06 --step 0.005 -o pa_dual.gcode\n",
            "  manifold-frontend calibrate transient --pass 1 -o transient_gamma.gcode\n",
            "  manifold-frontend calibrate transient --pass 2 -s transient_pressure_exponent=0.26 -o transient_tau.gcode\n"
        )
    );
}

// ============================================================================
// Profile & Preset JSON / TOML Loading, Merging & Dotted-Key Editing
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct ProfileBuildSpec {
    pub base_files: Vec<PathBuf>,
    pub printer: Option<String>,
    pub tool: Option<String>,
    pub filament: Option<String>,
    pub process: Option<String>,
    pub overrides: Vec<(String, String)>,
}

impl ProfileBuildSpec {
    pub fn has_any(&self) -> bool {
        !self.base_files.is_empty()
            || self.printer.is_some()
            || self.tool.is_some()
            || self.filament.is_some()
            || self.process.is_some()
            || !self.overrides.is_empty()
    }

    /// Applies this `ProfileBuildSpec` onto an existing `SlicerConfig` (e.g. `SlicerConfig::default()`
    /// or a `.3mf` project's embedded config).
    pub fn apply_onto(&self, cfg: &mut SlicerConfig, lib: &PresetLibrary) -> Result<(), String> {
        for path in &self.base_files {
            apply_profile_file_onto_config(cfg, path)?;
        }
        if let Some(ref p) = self.printer {
            apply_category_preset_or_file(cfg, lib, PresetCategory::Printer, p)?;
        }
        if let Some(ref t) = self.tool {
            apply_category_preset_or_file(cfg, lib, PresetCategory::Tool, t)?;
        }
        if let Some(ref f) = self.filament {
            apply_category_preset_or_file(cfg, lib, PresetCategory::Filament, f)?;
        }
        if let Some(ref pr) = self.process {
            apply_category_preset_or_file(cfg, lib, PresetCategory::Process, pr)?;
        }
        for (k, v) in &self.overrides {
            set_config_key_value(cfg, k, v)?;
        }
        sanitize_and_sync_config(cfg);
        Ok(())
    }
}

/// Sanitizes all sub-configs and synchronizes derived fields (`auto_voxel_size`, normalized
/// kinematic vectors in bed coordinates per `AGENTS.md` Rule 1 & Rule 2).
pub fn sanitize_and_sync_config(cfg: &mut SlicerConfig) {
    if cfg.tools.is_empty() {
        cfg.tools.push(crate::config::ToolConfig::default());
        cfg.active_tool_index = 0;
    } else if cfg.active_tool_index >= cfg.tools.len() {
        cfg.active_tool_index = cfg.tools.len() - 1;
    }
    let nz = Vec3::from_array(cfg.machine.nozzle_axis_bed)
        .try_normalize()
        .unwrap_or(Vec3::Z);
    cfg.machine.nozzle_axis_bed = nz.to_array();
    let grav = Vec3::from_array(cfg.machine.gravity_direction_bed)
        .try_normalize()
        .unwrap_or(Vec3::NEG_Z);
    cfg.machine.gravity_direction_bed = grav.to_array();

    cfg.eikonal.sanitize();
    cfg.slicing.sanitize();
    cfg.gcode.sanitize();
    cfg.sync_auto_voxel_size();
}

pub fn parse_preset_category(s: &str) -> Result<PresetCategory, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "printer" | "printers" | "machine" => Ok(PresetCategory::Printer),
        "tool" | "tools" | "nozzle" | "extruder" => Ok(PresetCategory::Tool),
        "filament" | "filaments" | "material" => Ok(PresetCategory::Filament),
        "process" | "processes" | "slicing" | "print" => Ok(PresetCategory::Process),
        other => Err(format!(
            "Unknown preset category '{other}' (expected 'printer', 'tool', 'filament', or 'process')"
        )),
    }
}

/// Applies either a named preset from `PresetLibrary` (case-insensitive fallback) or a `.json`/`.toml`
/// preset file on disk for the given `category`.
pub fn apply_category_preset_or_file(
    cfg: &mut SlicerConfig,
    lib: &PresetLibrary,
    category: PresetCategory,
    name_or_path: &str,
) -> Result<(), String> {
    let path = Path::new(name_or_path);
    if path.exists() && path.is_file() {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
        return apply_category_preset_str(cfg, category, &text, path);
    }

    match category {
        PresetCategory::Printer => {
            if let Some(p) = lib
                .printers
                .iter()
                .find(|p| p.name == name_or_path || p.name.eq_ignore_ascii_case(name_or_path))
            {
                p.apply_to_config(cfg);
                return Ok(());
            }
        }
        PresetCategory::Tool => {
            if let Some(p) = lib
                .tools
                .iter()
                .find(|p| p.name == name_or_path || p.name.eq_ignore_ascii_case(name_or_path))
            {
                p.apply_to_config(cfg);
                return Ok(());
            }
        }
        PresetCategory::Filament => {
            if let Some(p) = lib
                .filaments
                .iter()
                .find(|p| p.name == name_or_path || p.name.eq_ignore_ascii_case(name_or_path))
            {
                p.apply_to_config(cfg);
                return Ok(());
            }
        }
        PresetCategory::Process => {
            if let Some(p) = lib
                .processes
                .iter()
                .find(|p| p.name == name_or_path || p.name.eq_ignore_ascii_case(name_or_path))
            {
                p.apply_to_config(cfg);
                return Ok(());
            }
        }
    }

    let available: Vec<String> = lib
        .preset_names(category)
        .into_iter()
        .map(|(n, _)| format!("'{n}'"))
        .collect();
    Err(format!(
        "{} preset '{}' not found (neither a file nor a known preset; available: {})",
        category.label(),
        name_or_path,
        available.join(", ")
    ))
}

fn apply_category_preset_str(
    cfg: &mut SlicerConfig,
    category: PresetCategory,
    text: &str,
    source_path: &Path,
) -> Result<(), String> {
    let is_toml = source_path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
        || (!text.trim_start().starts_with('{'));

    match category {
        PresetCategory::Printer => {
            let p: PrinterPreset = if is_toml {
                toml::from_str(text).map_err(|e| format!("{}: {e}", source_path.display()))?
            } else {
                serde_json::from_str(text)
                    .map_err(|e| format!("{}: {e}", source_path.display()))?
            };
            p.apply_to_config(cfg);
        }
        PresetCategory::Tool => {
            let p: ToolPreset = if is_toml {
                toml::from_str(text).map_err(|e| format!("{}: {e}", source_path.display()))?
            } else {
                serde_json::from_str(text)
                    .map_err(|e| format!("{}: {e}", source_path.display()))?
            };
            p.apply_to_config(cfg);
        }
        PresetCategory::Filament => {
            let p: FilamentPreset = if is_toml {
                toml::from_str(text).map_err(|e| format!("{}: {e}", source_path.display()))?
            } else {
                serde_json::from_str(text)
                    .map_err(|e| format!("{}: {e}", source_path.display()))?
            };
            p.apply_to_config(cfg);
        }
        PresetCategory::Process => {
            let p: ProcessPreset = if is_toml {
                toml::from_str(text).map_err(|e| format!("{}: {e}", source_path.display()))?
            } else {
                serde_json::from_str(text)
                    .map_err(|e| format!("{}: {e}", source_path.display()))?
            };
            p.apply_to_config(cfg);
        }
    }
    Ok(())
}

/// Parses a `.json` or `.toml` profile file (supporting full `SlicerConfig`, partial nested JSON/TOML
/// trees, flat/dotted key maps, or single-category preset files) and merges it onto `cfg`.
pub fn apply_profile_file_onto_config(cfg: &mut SlicerConfig, path: &Path) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read profile '{}': {e}", path.display()))?;
    apply_profile_str_onto_config(cfg, &text, Some(path))
}

pub fn apply_profile_str_onto_config(
    cfg: &mut SlicerConfig,
    text: &str,
    source_path: Option<&Path>,
) -> Result<(), String> {
    let label = source_path
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<profile>".to_string());
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err(format!("Profile '{label}' is empty."));
    }

    let prefer_json = source_path
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        || trimmed.starts_with('{');

    let incoming_val: JsonValue = if prefer_json {
        serde_json::from_str(trimmed)
            .or_else(|_| {
                toml::from_str::<toml::Value>(trimmed)
                    .map_err(|e| e.to_string())
                    .and_then(|tv| serde_json::to_value(tv).map_err(|e| e.to_string()))
            })
            .map_err(|e| format!("Failed to parse profile '{label}': {e}"))?
    } else {
        toml::from_str::<toml::Value>(trimmed)
            .map_err(|e| e.to_string())
            .and_then(|tv| serde_json::to_value(tv).map_err(|e| e.to_string()))
            .or_else(|_| serde_json::from_str(trimmed).map_err(|e| e.to_string()))
            .map_err(|e| format!("Failed to parse profile '{label}': {e}"))?
    };

    let JsonValue::Object(map) = incoming_val else {
        return Err(format!(
            "Profile '{label}' root must be a JSON/TOML object/table."
        ));
    };

    apply_json_object_onto_config(cfg, &map, &label)
}

/// Merges a JSON object map onto `cfg`. Supports:
/// 1. Standard nested `SlicerConfig` sections (`bed`, `arrangement`, `slicing`, `gcode`, `eikonal`, `machine`, `tools`, ...).
/// 2. Flat or dotted parameter keys (e.g. `"slicing.layer_height.target_mm": 0.16` or `"infill_density_percent": 30.0`),
///    including category preset files (`PrinterPreset`, `ToolPreset`, `FilamentPreset`, `ProcessPreset`).
fn apply_json_object_onto_config(
    cfg: &mut SlicerConfig,
    map: &JsonMap<String, JsonValue>,
    label: &str,
) -> Result<(), String> {
    let mut cfg_val = serde_json::to_value(&*cfg).map_err(|e| e.to_string())?;
    let mut deferred_leaves: Vec<(String, JsonValue)> = Vec::new();

    if let JsonValue::Object(ref mut root_obj) = cfg_val {
        for (k, v) in map {
            if k == "name" || k == "is_builtin" {
                continue;
            }
            if let Some(existing) = root_obj.get_mut(k) {
                merge_json_values(existing, v.clone());
            } else {
                deferred_leaves.push((k.clone(), v.clone()));
            }
        }
    }

    *cfg = serde_json::from_value(cfg_val)
        .map_err(|e| format!("Invalid parameter type in '{label}': {e}"))?;

    // Apply any flat/shorthand or dotted keys (which also handles category preset files!)
    let mut updated_bed_tilt = false;
    let mut explicit_gravity = false;
    if map
        .get("machine")
        .and_then(|m| m.as_object())
        .is_some_and(|m| m.contains_key("bed_tilt_rx_deg") || m.contains_key("bed_tilt_ry_deg"))
    {
        updated_bed_tilt = true;
    }
    if map
        .get("machine")
        .and_then(|m| m.as_object())
        .is_some_and(|m| m.contains_key("gravity_direction_bed"))
    {
        explicit_gravity = true;
    }

    for (raw_key, val) in deferred_leaves {
        if raw_key.ends_with("bed_tilt_rx_deg") || raw_key.ends_with("bed_tilt_ry_deg") {
            updated_bed_tilt = true;
        }
        if raw_key.ends_with("gravity_direction_bed") {
            explicit_gravity = true;
        }
        set_config_json_value(cfg, &raw_key, val)
            .map_err(|e| format!("In '{label}': {e}"))?;
    }

    if updated_bed_tilt && !explicit_gravity {
        cfg.machine.update_gravity_from_bed_tilt();
    }
    sanitize_and_sync_config(cfg);
    Ok(())
}

fn merge_json_values(target: &mut JsonValue, patch: JsonValue) {
    match (target, patch) {
        (JsonValue::Object(t_map), JsonValue::Object(p_map)) => {
            for (k, v) in p_map {
                if let Some(t_child) = t_map.get_mut(&k) {
                    merge_json_values(t_child, v);
                } else {
                    t_map.insert(k, v);
                }
            }
        }
        (t, p) => {
            *t = p;
        }
    }
}

/// Resolves a user-supplied parameter key (exact dotted path like `slicing.layer_height.target_mm`,
/// preset-style alias like `nozzle_temperature_celsius`, or unambiguous suffix like `layer_height.target_mm`
/// or `infill_density_percent`) to its canonical dotted path in `SlicerConfig`.
pub fn resolve_canonical_config_key(cfg_val: &JsonValue, raw_key: &str) -> Result<String, String> {
    let key = raw_key.trim();
    if key.is_empty() {
        return Err("Parameter key cannot be empty.".to_string());
    }

    // Preset field aliases that map to active tool or bed fields
    match key {
        "nozzle_temperature_celsius" | "nozzle_temp" => {
            return Ok("tools.0.temperature_celsius".to_string());
        }
        "nozzle_diameter_mm" | "nozzle_diameter" => {
            return Ok("tools.0.nozzle_diameter_mm".to_string());
        }
        "has_rotational_axes" => {
            return Ok("tools.0.has_rotational_axes".to_string());
        }
        "max_angle_from_bed_normal_deg" => {
            return Ok("tools.0.max_angle_from_bed_normal_deg".to_string());
        }
        "clearance_profile_xz" => {
            return Ok("tools.0.clearance_profile_xz".to_string());
        }
        "clearance_mode" => {
            return Ok("tools.0.clearance_mode".to_string());
        }
        "cone_tip_radius_mm" => {
            return Ok("tools.0.cone_tip_radius_mm".to_string());
        }
        "cone_half_angle_deg" => {
            return Ok("tools.0.cone_half_angle_deg".to_string());
        }
        "cone_height_mm" => {
            return Ok("tools.0.cone_height_mm".to_string());
        }
        "rect_tip_half_width_x_mm" => {
            return Ok("tools.0.rect_tip_half_width_x_mm".to_string());
        }
        "rect_tip_half_width_y_mm" => {
            return Ok("tools.0.rect_tip_half_width_y_mm".to_string());
        }
        "rect_half_angle_x_deg" => {
            return Ok("tools.0.rect_half_angle_x_deg".to_string());
        }
        "rect_half_angle_y_deg" => {
            return Ok("tools.0.rect_half_angle_y_deg".to_string());
        }
        "rect_height_mm" => {
            return Ok("tools.0.rect_height_mm".to_string());
        }
        "bed_shape" => {
            return Ok("bed.shape".to_string());
        }
        "support_bed_interface" | "support_bottom_interface_layers" => {
            return Ok("slicing.support_bed_interface_layers".to_string());
        }
        _ => {}
    }

    // 1. Check if `key` is already a valid exact path from the root
    if get_json_by_dotted_path(cfg_val, key).is_some() {
        return Ok(key.to_string());
    }

    // 2. Collect all dotted paths in `cfg_val` and look for unambiguous suffix matches
    let mut all_paths = Vec::new();
    collect_all_dotted_paths(cfg_val, "", &mut all_paths);

    let suffix_with_dot = format!(".{key}");
    let matches: Vec<&String> = all_paths
        .iter()
        .filter(|p| p.as_str() == key || p.ends_with(&suffix_with_dot))
        .collect();

    if matches.len() == 1 {
        return Ok(matches[0].clone());
    } else if matches.len() > 1 {
        let choices: Vec<&str> = matches.iter().map(|s| s.as_str()).collect();
        return Err(format!(
            "Ambiguous parameter key '{key}'; matches multiple paths: {}",
            choices.join(", ")
        ));
    }

    // Case-insensitive fallback
    let lower_suffix = suffix_with_dot.to_ascii_lowercase();
    let lower_key = key.to_ascii_lowercase();
    let ci_matches: Vec<&String> = all_paths
        .iter()
        .filter(|p| {
            let lp = p.to_ascii_lowercase();
            lp == lower_key || lp.ends_with(&lower_suffix)
        })
        .collect();
    if ci_matches.len() == 1 {
        return Ok(ci_matches[0].clone());
    }

    Err(format!(
        "Unknown parameter key '{key}'. Run `manifold-frontend profile keys` to list all valid keys."
    ))
}

fn collect_all_dotted_paths(val: &JsonValue, prefix: &str, out: &mut Vec<String>) {
    if !prefix.is_empty() {
        out.push(prefix.to_string());
    }
    match val {
        JsonValue::Object(map) => {
            for (k, child) in map {
                let next = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_all_dotted_paths(child, &next, out);
            }
        }
        JsonValue::Array(arr) => {
            for (idx, child) in arr.iter().enumerate() {
                let next = if prefix.is_empty() {
                    idx.to_string()
                } else {
                    format!("{prefix}.{idx}")
                };
                collect_all_dotted_paths(child, &next, out);
            }
        }
        _ => {}
    }
}

pub fn collect_leaf_dotted_paths(val: &JsonValue, prefix: &str, out: &mut Vec<(String, JsonValue)>) {
    match val {
        JsonValue::Object(map) => {
            for (k, child) in map {
                let next = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                collect_leaf_dotted_paths(child, &next, out);
            }
        }
        _ => {
            out.push((prefix.to_string(), val.clone()));
        }
    }
}

pub fn get_json_by_dotted_path<'a>(root: &'a JsonValue, dotted_path: &str) -> Option<&'a JsonValue> {
    let mut curr = root;
    for part in dotted_path.split('.') {
        if part.is_empty() {
            continue;
        }
        match curr {
            JsonValue::Object(map) => {
                curr = map.get(part)?;
            }
            JsonValue::Array(arr) => {
                let idx: usize = part.parse().ok()?;
                curr = arr.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(curr)
}

fn get_json_mut_by_dotted_path<'a>(
    root: &'a mut JsonValue,
    dotted_path: &str,
) -> Option<&'a mut JsonValue> {
    let mut curr = root;
    for part in dotted_path.split('.') {
        if part.is_empty() {
            continue;
        }
        match curr {
            JsonValue::Object(map) => {
                curr = map.get_mut(part)?;
            }
            JsonValue::Array(arr) => {
                let idx: usize = part.parse().ok()?;
                curr = arr.get_mut(idx)?;
            }
            _ => return None,
        }
    }
    Some(curr)
}

/// Coerces a user-provided CLI string value `raw_val` to match the expected JSON type of `target`.
fn parse_cli_value_for_target(target: &JsonValue, raw_val: &str, key: &str) -> Result<JsonValue, String> {
    let trimmed = raw_val.trim();
    match target {
        JsonValue::Bool(_) => match trimmed.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(JsonValue::Bool(true)),
            "false" | "0" | "no" | "off" => Ok(JsonValue::Bool(false)),
            _ => Err(format!(
                "Expected boolean (`true` or `false`) for '{key}', got '{raw_val}'"
            )),
        },
        JsonValue::Number(num) => {
            if num.is_i64() || num.is_u64() {
                if let Ok(i) = trimmed.parse::<i64>() {
                    return Ok(JsonValue::Number(i.into()));
                }
                if let Ok(f) = trimmed.parse::<f64>() {
                    return Ok(JsonValue::Number((f.round() as i64).into()));
                }
                Err(format!(
                    "Expected integer number for '{key}', got '{raw_val}'"
                ))
            } else {
                let f = trimmed.parse::<f64>().map_err(|_| {
                    format!("Expected floating-point number for '{key}', got '{raw_val}'")
                })?;
                let n = serde_json::Number::from_f64(f)
                    .ok_or_else(|| format!("Non-finite number for '{key}': '{raw_val}'"))?;
                Ok(JsonValue::Number(n))
            }
        }
        JsonValue::String(_) => {
            // Strip optional surrounding quotes if passed
            let unquoted = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
                || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
            {
                &trimmed[1..trimmed.len() - 1]
            } else {
                trimmed
            };
            if key.ends_with("order_field_mode") {
                match unquoted.to_ascii_lowercase().as_str() {
                    "planar" | "height" | "heightonly" | "height_only" | "heightonlyplanar"
                    | "height_only_planar" => {
                        return Ok(JsonValue::String("HeightOnlyPlanar".to_string()));
                    }
                    "nonplanar" | "non_planar" | "eikonal" | "constrained"
                    | "constrainednonplanar" | "constrained_non_planar" => {
                        return Ok(JsonValue::String("ConstrainedNonPlanar".to_string()));
                    }
                    _ => {}
                }
            }
            if key.ends_with("support_style") {
                match unquoted.to_ascii_lowercase().as_str() {
                    "tree" | "nonplanartree" | "non_planar_tree" | "organic" => {
                        return Ok(JsonValue::String("NonPlanarTree".to_string()));
                    }
                    "grid" | "columnar" | "columnargrid" | "columnar_grid" | "normal" => {
                        return Ok(JsonValue::String("ColumnarGrid".to_string()));
                    }
                    _ => {}
                }
            }
            Ok(JsonValue::String(unquoted.to_string()))
        }
        JsonValue::Array(_) | JsonValue::Object(_) => serde_json::from_str(trimmed).map_err(|e| {
            format!("Expected valid JSON array/object for '{key}', got '{raw_val}' ({e})")
        }),
        JsonValue::Null => serde_json::from_str(trimmed)
            .or_else(|_| Ok(JsonValue::String(trimmed.to_string()))),
    }
}

/// Sets a single parameter on `cfg` using a CLI `key` and string `raw_val`.
pub fn set_config_key_value(
    cfg: &mut SlicerConfig,
    raw_key: &str,
    raw_val: &str,
) -> Result<String, String> {
    let mut cfg_val = serde_json::to_value(&*cfg).map_err(|e| e.to_string())?;
    let canonical = resolve_canonical_config_key(&cfg_val, raw_key)?;
    let target_slot = get_json_mut_by_dotted_path(&mut cfg_val, &canonical)
        .ok_or_else(|| format!("Key '{canonical}' not found"))?;
    let new_val = parse_cli_value_for_target(target_slot, raw_val, &canonical)?;
    *target_slot = new_val;

    let mut updated: SlicerConfig = serde_json::from_value(cfg_val)
        .map_err(|e| format!("Invalid value '{raw_val}' for '{canonical}': {e}"))?;

    if canonical == "machine.bed_tilt_rx_deg" || canonical == "machine.bed_tilt_ry_deg" {
        updated.machine.update_gravity_from_bed_tilt();
    }
    if canonical == "eikonal.voxel_size_mm" && raw_key.contains("voxel_size_mm") {
        // When the user explicitly sets `eikonal.voxel_size_mm`, keep their manual value
        // unless `auto_voxel_size` was explicitly requested.
        updated.eikonal.auto_voxel_size = false;
    }
    sanitize_and_sync_config(&mut updated);
    *cfg = updated;
    Ok(canonical)
}

fn set_config_json_value(
    cfg: &mut SlicerConfig,
    raw_key: &str,
    new_val: JsonValue,
) -> Result<String, String> {
    let mut cfg_val = serde_json::to_value(&*cfg).map_err(|e| e.to_string())?;
    let canonical = resolve_canonical_config_key(&cfg_val, raw_key)?;
    let target_slot = get_json_mut_by_dotted_path(&mut cfg_val, &canonical)
        .ok_or_else(|| format!("Key '{canonical}' not found"))?;
    *target_slot = new_val;

    let updated: SlicerConfig = serde_json::from_value(cfg_val)
        .map_err(|e| format!("Invalid value for '{canonical}': {e}"))?;
    *cfg = updated;
    Ok(canonical)
}

pub fn serialize_config(cfg: &SlicerConfig, format: ProfileFormat) -> Result<String, String> {
    match format {
        ProfileFormat::Json => serde_json::to_string_pretty(cfg).map_err(|e| e.to_string()),
        ProfileFormat::Toml => toml::to_string_pretty(cfg).map_err(|e| e.to_string()),
    }
}

fn parse_key_eq_val(arg: &str) -> Result<(String, String), String> {
    let Some((k, v)) = arg.split_once('=') else {
        return Err(format!(
            "Expected `KEY=VALUE` assignment, got '{arg}' (missing `=`)"
        ));
    };
    let key = k.trim();
    if key.is_empty() {
        return Err(format!("Empty parameter key in '{arg}'"));
    }
    Ok((key.to_string(), v.to_string()))
}

// ============================================================================
// GUI Launch Options Parser (`manifold-frontend gui ...`)
// ============================================================================

fn parse_gui_args(args: &[String]) -> Result<GuiLaunchOptions, String> {
    let mut opts = GuiLaunchOptions::default();
    let mut spec = ProfileBuildSpec::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => {
                print_top_level_help();
                std::process::exit(0);
            }
            "-p" | "--profile" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "Missing file path after `--profile`".to_string())?;
                spec.base_files.push(PathBuf::from(val));
            }
            "--printer" => {
                i += 1;
                spec.printer = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--printer`".to_string())?
                        .clone(),
                );
            }
            "--tool" => {
                i += 1;
                spec.tool = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--tool`".to_string())?
                        .clone(),
                );
            }
            "--filament" => {
                i += 1;
                spec.filament = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--filament`".to_string())?
                        .clone(),
                );
            }
            "--process" => {
                i += 1;
                spec.process = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--process`".to_string())?
                        .clone(),
                );
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                spec.overrides.push(parse_key_eq_val(kv)?);
            }
            _ if a.starts_with('-') => {
                return Err(format!("Unknown GUI option '{a}'"));
            }
            _ => {
                opts.initial_files.push(PathBuf::from(a));
            }
        }
        i += 1;
    }

    if spec.has_any() {
        let lib = PresetLibrary::default();
        let mut cfg = SlicerConfig::default();
        spec.apply_onto(&mut cfg, &lib)?;
        opts.initial_config = Some(cfg);
    }

    Ok(opts)
}

// ============================================================================
// Headless Slicing Command (`manifold-frontend slice ...`)
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct SliceCommandOptions {
    pub input_files: Vec<PathBuf>,
    pub profile_spec: ProfileBuildSpec,
    pub output_path: Option<PathBuf>,
    pub save_3mf_path: Option<PathBuf>,
    pub plate_filter: Option<String>,
    pub arrange_override: Option<bool>,
    pub place_on_face: Option<usize>,
    pub rotate_euler_deg: Option<[f32; 3]>,
    pub quiet: bool,
}

fn parse_slice_args(args: &[String]) -> Result<Option<SliceCommandOptions>, String> {
    let mut opts = SliceCommandOptions::default();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => {
                print_slice_help();
                return Ok(None);
            }
            "-p" | "--profile" => {
                i += 1;
                let p = args
                    .get(i)
                    .ok_or_else(|| "Missing file path after `--profile`".to_string())?;
                opts.profile_spec.base_files.push(PathBuf::from(p));
            }
            _ if a.starts_with("--profile=") => {
                opts.profile_spec
                    .base_files
                    .push(PathBuf::from(&a["--profile=".len()..]));
            }
            "--printer" => {
                i += 1;
                opts.profile_spec.printer = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--printer`".to_string())?
                        .clone(),
                );
            }
            "--tool" => {
                i += 1;
                opts.profile_spec.tool = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--tool`".to_string())?
                        .clone(),
                );
            }
            "--filament" => {
                i += 1;
                opts.profile_spec.filament = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--filament`".to_string())?
                        .clone(),
                );
            }
            "--process" => {
                i += 1;
                opts.profile_spec.process = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--process`".to_string())?
                        .clone(),
                );
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                opts.profile_spec.overrides.push(parse_key_eq_val(kv)?);
            }
            _ if a.starts_with("--set=") => {
                opts.profile_spec
                    .overrides
                    .push(parse_key_eq_val(&a["--set=".len()..])?);
            }
            _ if a.starts_with("-D") && a.len() > 2 => {
                opts.profile_spec.overrides.push(parse_key_eq_val(&a[2..])?);
            }
            "-o" | "--output" => {
                i += 1;
                let out = args
                    .get(i)
                    .ok_or_else(|| "Missing path after `--output`".to_string())?;
                opts.output_path = Some(PathBuf::from(out));
            }
            _ if a.starts_with("--output=") => {
                opts.output_path = Some(PathBuf::from(&a["--output=".len()..]));
            }
            "--save-3mf" => {
                i += 1;
                let out = args
                    .get(i)
                    .ok_or_else(|| "Missing path after `--save-3mf`".to_string())?;
                opts.save_3mf_path = Some(PathBuf::from(out));
            }
            "--plate" => {
                i += 1;
                let pl = args
                    .get(i)
                    .ok_or_else(|| "Missing plate index or name after `--plate`".to_string())?;
                opts.plate_filter = Some(pl.clone());
            }
            "--arrange" => {
                opts.arrange_override = Some(true);
            }
            "--no-arrange" => {
                opts.arrange_override = Some(false);
            }
            "--place-on-face" => {
                i += 1;
                let f_str = args
                    .get(i)
                    .ok_or_else(|| "Missing hull face index after `--place-on-face`".to_string())?;
                let idx: usize = f_str.parse().map_err(|_| {
                    format!("Invalid hull face index '{f_str}' for `--place-on-face`")
                })?;
                opts.place_on_face = Some(idx);
            }
            "--rotate" => {
                i += 1;
                let r_str = args
                    .get(i)
                    .ok_or_else(|| "Missing `RX,RY,RZ` degrees after `--rotate`".to_string())?;
                let parts: Vec<&str> = r_str.split(',').collect();
                if parts.len() != 3 {
                    return Err(format!(
                        "Expected 3 comma-separated angles `RX,RY,RZ` for `--rotate`, got '{r_str}'"
                    ));
                }
                let rx: f32 = parts[0]
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid X rotation '{}'", parts[0]))?;
                let ry: f32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid Y rotation '{}'", parts[1]))?;
                let rz: f32 = parts[2]
                    .trim()
                    .parse()
                    .map_err(|_| format!("Invalid Z rotation '{}'", parts[2]))?;
                opts.rotate_euler_deg = Some([rx, ry, rz]);
            }
            "-q" | "--quiet" => {
                opts.quiet = true;
            }
            _ if a.starts_with('-') => {
                return Err(format!(
                    "Unknown slice option '{a}'. Run `manifold-frontend slice --help` for usage."
                ));
            }
            _ => {
                opts.input_files.push(PathBuf::from(a));
            }
        }
        i += 1;
    }

    if opts.input_files.is_empty() {
        return Err(
            "No input 3D files specified. Pass one or more `.step`, `.stp`, `.stl`, or `.3mf` files."
                .to_string(),
        );
    }

    Ok(Some(opts))
}

fn cmd_slice(args: &[String]) -> Result<(), String> {
    let Some(opts) = parse_slice_args(args)? else {
        return Ok(());
    };
    execute_headless_slice(&opts).map(|_| ())
}

/// Summary returned by [`execute_headless_slice`].
#[derive(Clone, Debug)]
pub struct HeadlessSliceSummary {
    pub written_gcode_files: Vec<PathBuf>,
    pub total_layers: usize,
    pub total_extruding_moves: usize,
    pub total_filament_mm: f32,
    pub total_filament_grams: f32,
    pub total_time_s: f32,
}

/// Runs the full headless slicing pipeline on `opts`.
pub fn execute_headless_slice(opts: &SliceCommandOptions) -> Result<HeadlessSliceSummary, String> {
    let lib = PresetLibrary::default();
    if !opts.quiet {
        eprintln!(
            "[Manifold CLI] Loading {} input file(s)...",
            opts.input_files.len()
        );
    }

    let report = load_files_into_backend(&opts.input_files);
    for notice in &report.notices {
        if !opts.quiet {
            eprintln!("[Manifold CLI] Notice: {notice}");
        }
    }
    for err in &report.errors {
        eprintln!("[Manifold CLI] Import error: {err}");
    }
    if report.objects.is_empty() {
        if !report.errors.is_empty() {
            return Err(report.errors.join("; "));
        }
        return Err("No valid 3D objects found in the specified input file(s).".to_string());
    }

    // Start from restored `.3mf` project config (if any), then layer CLI profiles/presets/overrides on top
    let mut config = report.restored_config.unwrap_or_default();
    opts.profile_spec.apply_onto(&mut config, &lib)?;
    sync_config_to_backend(&config);

    let plate_names: Vec<String> = if report.plates.is_empty() {
        let max_p = report
            .objects
            .iter()
            .map(|o| o.plate_index)
            .max()
            .unwrap_or(0);
        (0..=max_p).map(|idx| format!("Plate {}", idx + 1)).collect()
    } else {
        report.plates.iter().map(|p| p.name.clone()).collect()
    };

    // Build initial world transforms for each loaded object
    let mut obj_transforms: Vec<Transform> = report
        .objects
        .iter()
        .map(|obj| Transform {
            translation: obj.initial_translation,
            rotation: obj.initial_rotation,
            scale: Vec3::ONE,
        })
        .collect();

    // Apply optional `--place-on-face` or `--rotate` and drop onto bed (`z_min = 0`)
    for (obj, tf) in report.objects.iter().zip(obj_transforms.iter_mut()) {
        let mut modified_rot = false;
        if let Some(face_idx) = opts.place_on_face {
            if let Some(face) = obj.hull.faces.get(face_idx) {
                tf.rotation = crate::hull::rotation_to_place_face_on_bed(tf.rotation, face.normal);
                modified_rot = true;
            } else {
                return Err(format!(
                    "Object '{}' has only {} simplified hull faces (requested `--place-on-face {}`)",
                    obj.name,
                    obj.hull.faces.len(),
                    face_idx
                ));
            }
        }
        if let Some([rx, ry, rz]) = opts.rotate_euler_deg {
            let euler_rot = Quat::from_euler(
                EulerRot::XYZ,
                rx.to_radians(),
                ry.to_radians(),
                rz.to_radians(),
            );
            tf.rotation = euler_rot * tf.rotation;
            modified_rot = true;
        }
        if modified_rot {
            let (min_p, _) = compute_loaded_object_world_aabb(obj, tf);
            tf.translation.z -= min_p.z;
        }
    }

    // Group object indices by plate_index
    let max_plate_idx = report
        .objects
        .iter()
        .map(|o| o.plate_index)
        .max()
        .unwrap_or(0)
        .max(plate_names.len().saturating_sub(1));

    let mut active_plates: Vec<usize> = Vec::new();
    for p_idx in 0..=max_plate_idx {
        let p_name = plate_names
            .get(p_idx)
            .cloned()
            .unwrap_or_else(|| format!("Plate {}", p_idx + 1));
        if let Some(ref filter) = opts.plate_filter {
            let matches_1based = filter
                .trim()
                .parse::<usize>()
                .is_ok_and(|n| n == p_idx + 1);
            let matches_name = p_name.eq_ignore_ascii_case(filter.trim());
            if !matches_1based && !matches_name {
                continue;
            }
        }
        let has_objs = report.objects.iter().any(|o| o.plate_index == p_idx);
        if has_objs {
            active_plates.push(p_idx);
        }
    }

    if active_plates.is_empty() {
        return Err(format!(
            "No objects matched plate filter {:?}",
            opts.plate_filter
        ));
    }

    // Run spiral auto-arrangement on plates when requested or when multiple objects overlap
    for &p_idx in &active_plates {
        let obj_indices: Vec<usize> = report
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.plate_index == p_idx)
            .map(|(i, _)| i)
            .collect();

        let should_arrange = match opts.arrange_override {
            Some(flag) => flag,
            None => {
                obj_indices.len() > 1
                    && objects_have_overlapping_xy_footprints(
                        &report.objects,
                        &obj_transforms,
                        &obj_indices,
                    )
            }
        };

        if should_arrange {
            arrange_loaded_objects_on_plate(
                &report.objects,
                &mut obj_transforms,
                &obj_indices,
                &config,
            );
        }
    }

    // Optionally save `.3mf` project archive if `--save-3mf` was passed
    if let Some(ref save_3mf) = opts.save_3mf_path {
        let export_list: Vec<ExportableObject> = report
            .objects
            .iter()
            .zip(obj_transforms.iter())
            .map(|(obj, tf)| ExportableObject {
                object_id: obj.object_id,
                name: obj.name.clone(),
                has_brep: obj.has_brep,
                plate_index: obj.plate_index,
                transform: *tf,
                vertices: obj.vertices.clone(),
                indices: obj.indices.clone(),
            })
            .collect();
        save_project_to_3mf(save_3mf, &plate_names, &export_list, &config)?;
        if !opts.quiet {
            eprintln!(
                "[Manifold CLI] Saved 3MF project archive to {}",
                save_3mf.display()
            );
        }
    }

    // Slice each active plate and emit G-code
    let mut summary = HeadlessSliceSummary {
        written_gcode_files: Vec::new(),
        total_layers: 0,
        total_extruding_moves: 0,
        total_filament_mm: 0.0,
        total_filament_grams: 0.0,
        total_time_s: 0.0,
    };

    let multiple_plates = active_plates.len() > 1;
    for &p_idx in &active_plates {
        let p_name = plate_names
            .get(p_idx)
            .cloned()
            .unwrap_or_else(|| format!("Plate {}", p_idx + 1));
        let obj_indices: Vec<usize> = report
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.plate_index == p_idx)
            .map(|(i, _)| i)
            .collect();

        if !opts.quiet {
            eprintln!(
                "[Manifold CLI] Slicing {} ({} object(s)) | Printer='{}' | Tool='{}' | Filament='{}' | Process='{}'...",
                p_name,
                obj_indices.len(),
                config.printer_preset_name,
                config.tool_preset_name,
                config.filament_preset_name,
                config.process_preset_name
            );
        }

        let mut planned_objects: Vec<(String, PlannedGcodeResult)> =
            Vec::with_capacity(obj_indices.len());

        for &obj_i in &obj_indices {
            let obj = &report.objects[obj_i];
            let tf = obj_transforms[obj_i];
            let planned = slice_single_loaded_object(obj, tf, &config, opts.quiet)?;
            summary.total_layers += planned.layers.len();
            summary.total_extruding_moves += planned.extruding_move_count;
            summary.total_filament_mm += planned.total_filament_mm;
            summary.total_filament_grams += planned.total_filament_grams;
            summary.total_time_s += planned.total_time_s;
            planned_objects.push((obj.name.clone(), planned));
        }

        let gcode_text = if planned_objects.len() == 1 {
            planned_objects.pop().unwrap().1.gcode_text
        } else {
            combine_planned_objects_into_plate_gcode(&planned_objects, &config)
        };

        let out_dest = resolve_plate_gcode_output_path(
            opts.output_path.as_deref(),
            &opts.input_files[0],
            p_idx,
            multiple_plates,
        );

        if out_dest.as_os_str() == "-" {
            io::stdout()
                .write_all(gcode_text.as_bytes())
                .map_err(|e| format!("Failed to write G-code to stdout: {e}"))?;
        } else {
            if let Some(parent) = out_dest.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            fs::write(&out_dest, gcode_text.as_bytes()).map_err(|e| {
                format!("Failed to write G-code to '{}': {e}", out_dest.display())
            })?;
            if !opts.quiet {
                eprintln!(
                    "[Manifold CLI] Wrote {} ({:.1} KB, est. {}, {:.2} m / {:.1} g filament)",
                    out_dest.display(),
                    (gcode_text.len() as f64) / 1024.0,
                    format_moonraker_estimated_time(summary.total_time_s),
                    summary.total_filament_mm / 1000.0,
                    summary.total_filament_grams
                );
            }
            summary.written_gcode_files.push(out_dest);
        }
    }

    Ok(summary)
}

fn slice_single_loaded_object(
    obj: &LoadedObjectData,
    obj_world_tf: Transform,
    config: &SlicerConfig,
    quiet: bool,
) -> Result<PlannedGcodeResult, String> {
    sync_config_to_backend(config);

    // Rotate bed-coordinate nozzle and gravity vectors into object-local OpenVDB coordinates
    // per AGENTS.md Rule 1 & Rule 2 (matching `spawn_field_compute_task` in `ui.rs`).
    let inv_rot = obj_world_tf.rotation.inverse();
    let local_nozzle =
        (inv_rot * Vec3::from_array(config.machine.nozzle_axis_bed)).normalize_or_zero();
    if local_nozzle.length_squared() > 1.0e-6 {
        ffi::set_nozzle_axis_bed(local_nozzle.to_array());
    }
    let local_gravity =
        (inv_rot * Vec3::from_array(config.machine.gravity_direction_bed)).normalize_or_zero();
    if local_gravity.length_squared() > 1.0e-6 {
        ffi::set_gravity_direction_bed(local_gravity.to_array());
    }

    // Push any configured spatial orientation hints for this object
    let mut ffi_hints = Vec::new();
    for sh in config
        .eikonal
        .surface_hints
        .iter()
        .filter(|h| h.object_id == obj.object_id)
    {
        if let Some(patch) = obj.surface_patches.get(sh.patch_id as usize) {
            let kind = match sh.mode {
                crate::config::SurfaceHintMode::NormalOrthogonalToSurface => 1u8,
                crate::config::SurfaceHintMode::NormalTangentToSurface => 0u8,
                crate::config::SurfaceHintMode::ConformalSurfaceSkin => 2u8,
            };
            let sample_count = patch.sample_points.len().max(1) as f32;
            let sample_spacing = (patch.area / sample_count).sqrt();
            let effective_radius = sh.influence_depth_mm.max(sample_spacing * 1.35);
            for &(pos, normal) in &patch.sample_points {
                ffi_hints.push(ffi::SpatialOrientationHint {
                    kind,
                    position_bed: [pos.x, pos.y, pos.z],
                    direction_bed: [normal.x, normal.y, normal.z],
                    influence_radius_mm: effective_radius,
                    weight: sh.weight,
                    apply_on_surface_only: true,
                    skin_layer_count: sh.skin_layer_count,
                });
            }
        }
    }
    for ph in config
        .eikonal
        .point_hints
        .iter()
        .filter(|h| h.object_id == obj.object_id)
    {
        ffi_hints.push(ffi::SpatialOrientationHint {
            kind: 1u8,
            position_bed: ph.local_position,
            direction_bed: ph.local_direction,
            influence_radius_mm: ph.influence_radius_mm,
            weight: ph.weight,
            apply_on_surface_only: false,
            skin_layer_count: 0,
        });
    }
    ffi::set_spatial_orientation_hints(&ffi_hints);

    let voxel_size = config.eikonal.voxel_size_mm;
    let narrow_band = config.eikonal.narrow_band_width_voxels as f32;
    if !quiet {
        eprintln!(
            "  -> Computing OpenVDB SDF & constrained Eikonal fields for '{}' (voxel {:.2} mm, {} sweeps)...",
            obj.name, voxel_size, config.eikonal.solver_iterations
        );
    }
    if !ffi::recompute_object_fields(obj.object_id, voxel_size, narrow_band) {
        return Err(format!(
            "C++ backend failed to compute OpenVDB scalar fields for '{}'",
            obj.name
        ));
    }

    if !quiet {
        eprintln!(
            "  -> Generating non-planar slicing isosurfaces & shell/infill/support toolpaths for '{}'...",
            obj.name
        );
    }
    let toolpaths = ffi::generate_object_shell_toolpaths(obj.object_id, 0.0);
    if toolpaths.layers.is_empty() {
        return Err(format!(
            "No non-planar toolpath layers were generated for '{}'",
            obj.name
        ));
    }

    if !quiet {
        eprintln!(
            "  -> Planning Klipper G-code across {} non-planar layers ({} paths)...",
            toolpaths.layers.len(),
            toolpaths.total_path_count
        );
    }
    let planned = plan_and_emit_gcode_for_object(
        &obj.name,
        obj.object_id,
        &toolpaths,
        obj_world_tf,
        config,
    );
    Ok(planned)
}

fn resolve_plate_gcode_output_path(
    explicit_out: Option<&Path>,
    first_input: &Path,
    plate_idx: usize,
    multiple_plates: bool,
) -> PathBuf {
    let default_stem = first_input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    if let Some(out) = explicit_out {
        if out.as_os_str() == "-" {
            return out.to_path_buf();
        }
        if out.is_dir() {
            let fname = if multiple_plates {
                format!("{default_stem}_plate{}.gcode", plate_idx + 1)
            } else {
                format!("{default_stem}.gcode")
            };
            return out.join(fname);
        }
        if multiple_plates {
            let parent = out.parent().unwrap_or_else(|| Path::new(""));
            let stem = out
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(default_stem);
            let ext = out
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("gcode");
            return parent.join(format!("{stem}_plate{}.{ext}", plate_idx + 1));
        }
        return out.to_path_buf();
    }

    let parent = first_input.parent().unwrap_or_else(|| Path::new(""));
    if multiple_plates {
        parent.join(format!("{default_stem}_plate{}.gcode", plate_idx + 1))
    } else {
        parent.join(format!("{default_stem}.gcode"))
    }
}

/// Combines multiple planned objects on the same build plate into a single Klipper G-code file
/// with all `EXCLUDE_OBJECT_DEFINE` declarations at the top, a unified Moonraker thumbnail,
/// start G-code, each object's body moves, end G-code, and combined metadata footer.
fn combine_planned_objects_into_plate_gcode(
    planned_objects: &[(String, PlannedGcodeResult)],
    config: &SlicerConfig,
) -> String {
    let mut gcfg = config.gcode.clone();
    gcfg.sanitize();
    let active_tool = config.active_tool();
    let tool_id = active_tool.map(|t| t.id).unwrap_or(0);
    let nozzle_diam_mm = active_tool
        .map(|t| t.nozzle_diameter_mm)
        .unwrap_or(0.40)
        .max(0.10);
    let nozzle_temp_c = active_tool.map(|t| t.temperature_celsius).unwrap_or(215.0);
    let gravity_bed = config.query_gravity_bed(Vec3::ZERO);
    let up_bed = (-gravity_bed).try_normalize().unwrap_or(Vec3::Z);

    let mut all_layers_for_thumb: Vec<PlannedLayerGcode> = Vec::new();
    let mut bbox_min = Vec3::splat(f32::MAX);
    let mut bbox_max = Vec3::splat(f32::MIN);
    let mut total_layers = 0usize;
    let mut total_time_s = 0.0f32;
    let mut total_filament_mm = 0.0f32;
    let mut total_filament_vol = 0.0f32;
    let mut total_filament_g = 0.0f32;

    for (_, p) in planned_objects {
        total_layers += p.layers.len();
        total_time_s += p.total_time_s;
        total_filament_mm += p.total_filament_mm;
        total_filament_vol += p.total_filament_volume_mm3;
        total_filament_g += p.total_filament_grams;
        for layer in &p.layers {
            for seg in &layer.segments {
                if seg.kind.is_extruding() && seg.delta_e_mm > 1.0e-6 {
                    bbox_min = bbox_min.min(seg.p0_bed).min(seg.p1_bed);
                    bbox_max = bbox_max.max(seg.p0_bed).max(seg.p1_bed);
                }
            }
            all_layers_for_thumb.push(layer.clone());
        }
    }
    if !bbox_min.is_finite() || bbox_min.x > bbox_max.x {
        bbox_min = Vec3::ZERO;
        bbox_max = Vec3::ZERO;
    }

    let mut out = String::with_capacity(128 * 1024);
    out.push_str("; generated by PrusaSlicer 2.8.0-Manifold2 on 2026-10-20 at 00:00:00 UTC\n");
    if gcfg.emit_thumbnails && !all_layers_for_thumb.is_empty() {
        out.push_str(&render_toolpath_preview_thumbnails(
            &all_layers_for_thumb,
            bbox_min,
            bbox_max,
            up_bed,
        ));
    }
    if gcfg.emit_exclude_object {
        for (_, p) in planned_objects {
            out.push_str(&format_exclude_object_define(
                &p.exclude_object_name,
                p.exclude_object_center_bed,
                &p.exclude_object_polygon_bed,
            ));
            out.push('\n');
        }
    }

    let start_expanded = expand_gcode_template(
        &gcfg.start_gcode_template,
        tool_id,
        nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        nozzle_diam_mm,
        bbox_min,
        bbox_max,
        total_layers,
    );
    out.push_str(&start_expanded);
    if !start_expanded.ends_with('\n') {
        out.push('\n');
    }
    out.push_str("M83 ; Ensure relative extrusion\n");

    // Extract the layer-body section from each object's `gcode_text`
    for (idx, (obj_name, p)) in planned_objects.iter().enumerate() {
        out.push_str(&format!("\n; === OBJECT {}: {} ===\n", idx + 1, obj_name));
        if idx > 0 {
            let safe_z = (bbox_max.z + gcfg.z_hop_mm.max(2.0)).min(config.bed.height_z);
            let z_feed = (gcfg.z_hop_speed_mms * 60.0).max(120.0);
            out.push_str(&format!(
                "G0 Z{safe_z:.3} F{z_feed:.0} ; Safe inter-object Z clearance\n"
            ));
        }
        if let Some(start_pos) = p.gcode_text.find("\n; --- LAYER ") {
            let body_slice = &p.gcode_text[start_pos..];
            let end_pos = body_slice
                .find("\n; --- Manifold2 Non-Planar Print End ---")
                .or_else(|| body_slice.find("\n; --- Moonraker / PrusaSlicer Metadata Footer ---"))
                .unwrap_or(body_slice.len());
            out.push_str(&body_slice[..end_pos]);
            out.push('\n');
        }
    }

    out.push('\n');
    let end_expanded = expand_gcode_template(
        &gcfg.end_gcode_template,
        tool_id,
        nozzle_temp_c,
        gcfg.bed_temperature_celsius,
        nozzle_diam_mm,
        bbox_min,
        bbox_max,
        total_layers,
    );
    out.push_str(&end_expanded);
    if !end_expanded.ends_with('\n') {
        out.push('\n');
    }

    let time_str = format_moonraker_estimated_time(total_time_s);
    out.push_str(&format!(
        concat!(
            "\n; --- Moonraker / PrusaSlicer Metadata Footer ---\n",
            "; filament used [mm] = {:.2}\n",
            "; filament used [cm3] = {:.2}\n",
            "; filament used [g] = {:.2}\n",
            "; total filament used [g] = {:.2}\n",
            "; estimated printing time (normal mode) = {}\n",
            "; first_layer_height = {:.3}\n",
            "; layer_height = {:.3}\n",
            "; max_print_height = {:.3}\n",
            "; total layers count = {}\n",
            "; nozzle_diameter = {:.2}\n",
            "; filament_diameter = {:.2}\n",
            "; filament_type = {}\n",
            "; filament_settings_id = \"Manifold2 Non-Planar\"\n",
            "; first_layer_bed_temperature = {:.0}\n",
            "; first_layer_temperature = {:.0}\n"
        ),
        total_filament_mm,
        total_filament_vol / 1000.0,
        total_filament_g,
        total_filament_g,
        time_str,
        config.slicing.layer_height.first_layer_mm,
        config.slicing.layer_height.target_mm,
        bbox_max.z.max(0.0),
        total_layers,
        nozzle_diam_mm,
        gcfg.filament_diameter_mm,
        gcfg.filament_type,
        gcfg.bed_temperature_celsius,
        nozzle_temp_c,
    ));
    out
}

fn compute_loaded_object_world_aabb(obj: &LoadedObjectData, tf: &Transform) -> (Vec3, Vec3) {
    let mut min_pt = Vec3::splat(f32::INFINITY);
    let mut max_pt = Vec3::splat(f32::NEG_INFINITY);
    for face in &obj.hull.faces {
        for &p in &face.polygon {
            let wp = tf.transform_point(p);
            min_pt = min_pt.min(wp);
            max_pt = max_pt.max(wp);
        }
    }
    let step = (obj.vertices.len() / 512).max(1);
    for v in obj.vertices.iter().step_by(step) {
        let wp = tf.transform_point(Vec3::new(v.x, v.y, v.z));
        min_pt = min_pt.min(wp);
        max_pt = max_pt.max(wp);
    }
    if min_pt.x.is_infinite() {
        (tf.translation, tf.translation)
    } else {
        (min_pt, max_pt)
    }
}

fn objects_have_overlapping_xy_footprints(
    objects: &[LoadedObjectData],
    transforms: &[Transform],
    indices: &[usize],
) -> bool {
    let bounds: Vec<(Vec3, Vec3)> = indices
        .iter()
        .map(|&i| compute_loaded_object_world_aabb(&objects[i], &transforms[i]))
        .collect();
    for a in 0..bounds.len() {
        for b in (a + 1)..bounds.len() {
            let (min_a, max_a) = bounds[a];
            let (min_b, max_b) = bounds[b];
            if min_a.x < max_b.x - 0.5
                && max_a.x > min_b.x + 0.5
                && min_a.y < max_b.y - 0.5
                && max_a.y > min_b.y + 0.5
            {
                return true;
            }
        }
    }
    false
}

fn arrange_loaded_objects_on_plate(
    objects: &[LoadedObjectData],
    transforms: &mut [Transform],
    indices: &[usize],
    config: &SlicerConfig,
) {
    let mut items = Vec::with_capacity(indices.len());
    for &i in indices {
        let obj = &objects[i];
        let tf = &transforms[i];
        let (min_p, max_p) = compute_loaded_object_world_aabb(obj, tf);
        let half_ext = 0.5 * (max_p.xy() - min_p.xy());
        let center_xy = 0.5 * (min_p.xy() + max_p.xy());
        let center_offset = center_xy - tf.translation.xy();
        let height_z = (max_p.z - min_p.z).max(0.0);
        items.push(ArrangeItem {
            object_id: obj.object_id,
            half_extents_xy: half_ext,
            center_offset_xy: center_offset,
            height_z,
        });
    }

    let placements = auto_arrange_spiral(&items, &config.bed, &config.arrangement);
    for p in placements {
        if let Some(&idx) = indices
            .iter()
            .find(|&&i| objects[i].object_id == p.object_id)
        {
            transforms[idx].translation.x = p.target_x;
            transforms[idx].translation.y = p.target_y;
            let (min_p, _) = compute_loaded_object_world_aabb(&objects[idx], &transforms[idx]);
            transforms[idx].translation.z -= min_p.z;
        }
    }
}

// ============================================================================
// Profile & Preset Verbs (`manifold-frontend profile <VERB> ...`)
// ============================================================================

fn cmd_profile(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        print_profile_help();
        return Ok(());
    }
    let verb = args[0].as_str();
    let rest = &args[1..];
    match verb {
        "-h" | "--help" | "help" => {
            print_profile_help();
            Ok(())
        }
        "init" | "create" | "dump" | "export" => cmd_profile_init(rest),
        "show" | "view" | "cat" => cmd_profile_show(rest),
        "get" => cmd_profile_get(rest),
        "set" | "edit" => cmd_profile_set(rest),
        "diff" => cmd_profile_diff(rest),
        "keys" | "params" | "fields" => cmd_profile_keys(rest),
        "list" | "ls" => cmd_profile_list(rest),
        "save-preset" => cmd_profile_save_preset(rest),
        "delete-preset" | "rm-preset" => cmd_profile_delete_preset(rest),
        "validate" | "check" => cmd_profile_validate(rest),
        other => Err(format!(
            "Unknown profile verb '{other}'. Run `manifold-frontend profile --help` for usage."
        )),
    }
}

fn cmd_profile_init(args: &[String]) -> Result<(), String> {
    let mut out_path: Option<PathBuf> = None;
    let mut explicit_format: Option<ProfileFormat> = None;
    let mut category: Option<PresetCategory> = None;
    let mut spec = ProfileBuildSpec::default();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => {
                print_profile_help();
                return Ok(());
            }
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| "Missing path after `--output`".to_string())?,
                ));
            }
            "--from" | "-p" | "--profile" => {
                i += 1;
                spec.base_files.push(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| format!("Missing file path after `{a}`"))?,
                ));
            }
            "--printer" => {
                i += 1;
                spec.printer = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--printer`".to_string())?
                        .clone(),
                );
            }
            "--tool" => {
                i += 1;
                spec.tool = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--tool`".to_string())?
                        .clone(),
                );
            }
            "--filament" => {
                i += 1;
                spec.filament = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--filament`".to_string())?
                        .clone(),
                );
            }
            "--process" => {
                i += 1;
                spec.process = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--process`".to_string())?
                        .clone(),
                );
            }
            "--category" | "-c" => {
                i += 1;
                let cat_str = args
                    .get(i)
                    .ok_or_else(|| "Missing category after `--category`".to_string())?;
                category = Some(parse_preset_category(cat_str)?);
            }
            "--format" | "-f" => {
                i += 1;
                let fmt_str = args
                    .get(i)
                    .ok_or_else(|| "Missing format (`json` or `toml`) after `--format`".to_string())?;
                explicit_format = Some(ProfileFormat::parse_str(fmt_str)?);
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                spec.overrides.push(parse_key_eq_val(kv)?);
            }
            _ if a.starts_with('-') => {
                return Err(format!("Unknown option '{a}' for `profile init`"));
            }
            _ => {
                if out_path.is_none() {
                    out_path = Some(PathBuf::from(a));
                } else {
                    spec.overrides.push(parse_key_eq_val(a)?);
                }
            }
        }
        i += 1;
    }

    let lib = PresetLibrary::default();
    let mut cfg = SlicerConfig::default();
    spec.apply_onto(&mut cfg, &lib)?;

    let fmt = ProfileFormat::from_path_or_default(out_path.as_deref(), explicit_format);
    let serialized = if let Some(cat) = category {
        serialize_category_from_config(&cfg, cat, fmt)?
    } else {
        serialize_config(&cfg, fmt)?
    };

    if let Some(path) = out_path {
        if path.as_os_str() == "-" {
            println!("{serialized}");
        } else {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    let _ = fs::create_dir_all(parent);
                }
            }
            fs::write(&path, format!("{serialized}\n"))
                .map_err(|e| format!("Failed to write '{}': {e}", path.display()))?;
            eprintln!("Wrote profile to {}", path.display());
        }
    } else {
        println!("{serialized}");
    }
    Ok(())
}

fn serialize_category_from_config(
    cfg: &SlicerConfig,
    category: PresetCategory,
    format: ProfileFormat,
) -> Result<String, String> {
    match (category, format) {
        (PresetCategory::Printer, ProfileFormat::Json) => serde_json::to_string_pretty(
            &PrinterPreset::from_config(&cfg.printer_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Printer, ProfileFormat::Toml) => toml::to_string_pretty(
            &PrinterPreset::from_config(&cfg.printer_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Tool, ProfileFormat::Json) => serde_json::to_string_pretty(
            &ToolPreset::from_config(&cfg.tool_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Tool, ProfileFormat::Toml) => toml::to_string_pretty(
            &ToolPreset::from_config(&cfg.tool_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Filament, ProfileFormat::Json) => serde_json::to_string_pretty(
            &FilamentPreset::from_config(&cfg.filament_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Filament, ProfileFormat::Toml) => toml::to_string_pretty(
            &FilamentPreset::from_config(&cfg.filament_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Process, ProfileFormat::Json) => serde_json::to_string_pretty(
            &ProcessPreset::from_config(&cfg.process_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
        (PresetCategory::Process, ProfileFormat::Toml) => toml::to_string_pretty(
            &ProcessPreset::from_config(&cfg.process_preset_name, cfg, false),
        )
        .map_err(|e| e.to_string()),
    }
}

/// Loads a `SlicerConfig` from either a file path, `"default"`, or a named preset in `PresetLibrary`.
pub fn load_config_from_source(
    source: &str,
    category_hint: Option<PresetCategory>,
    lib: &PresetLibrary,
) -> Result<(SlicerConfig, Option<PresetCategory>), String> {
    if source.is_empty() || source.eq_ignore_ascii_case("default") {
        let mut cfg = SlicerConfig::default();
        sanitize_and_sync_config(&mut cfg);
        return Ok((cfg, category_hint));
    }

    let path = Path::new(source);
    if path.exists() && path.is_file() {
        let mut cfg = SlicerConfig::default();
        apply_profile_file_onto_config(&mut cfg, path)?;
        return Ok((cfg, category_hint));
    }

    // Otherwise search saved/built-in presets in `PresetLibrary`
    let cats: Vec<PresetCategory> = if let Some(c) = category_hint {
        vec![c]
    } else {
        PresetCategory::ALL.to_vec()
    };

    for cat in cats {
        let mut cfg = SlicerConfig::default();
        if apply_category_preset_or_file(&mut cfg, lib, cat, source).is_ok() {
            sanitize_and_sync_config(&mut cfg);
            return Ok((cfg, Some(cat)));
        }
    }

    Err(format!(
        "Profile source '{source}' was not found as a file or saved preset."
    ))
}

fn cmd_profile_show(args: &[String]) -> Result<(), String> {
    let mut source = "default".to_string();
    let mut key_filter: Option<String> = None;
    let mut category: Option<PresetCategory> = None;
    let mut explicit_format: Option<ProfileFormat> = None;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => {
                print_profile_help();
                return Ok(());
            }
            "-k" | "--key" => {
                i += 1;
                key_filter = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing key path after `--key`".to_string())?
                        .clone(),
                );
            }
            "-c" | "--category" => {
                i += 1;
                category = Some(parse_preset_category(
                    args.get(i)
                        .ok_or_else(|| "Missing category after `--category`".to_string())?,
                )?);
            }
            "-f" | "--format" => {
                i += 1;
                explicit_format = Some(ProfileFormat::parse_str(
                    args.get(i)
                        .ok_or_else(|| "Missing format after `--format`".to_string())?,
                )?);
            }
            _ if a.starts_with('-') => {
                return Err(format!("Unknown option '{a}' for `profile show`"));
            }
            _ => {
                source = a.to_string();
            }
        }
        i += 1;
    }

    let lib = PresetLibrary::default();
    let (cfg, matched_cat) = load_config_from_source(&source, category, &lib)?;
    let fmt = ProfileFormat::from_path_or_default(Some(Path::new(&source)), explicit_format);

    if let Some(ref raw_key) = key_filter {
        let cfg_val = serde_json::to_value(&cfg).map_err(|e| e.to_string())?;
        let canonical = resolve_canonical_config_key(&cfg_val, raw_key)?;
        let val = get_json_by_dotted_path(&cfg_val, &canonical)
            .ok_or_else(|| format!("Key '{canonical}' not found"))?;
        println!("{}", format_json_value_for_stdout(val));
        return Ok(());
    }

    let out = if let Some(cat) = category.or(matched_cat) {
        if Path::new(&source).exists() && category.is_none() {
            serialize_config(&cfg, fmt)?
        } else {
            serialize_category_from_config(&cfg, cat, fmt)?
        }
    } else {
        serialize_config(&cfg, fmt)?
    };
    println!("{out}");
    Ok(())
}

fn format_json_value_for_stdout(val: &JsonValue) -> String {
    match val {
        JsonValue::String(s) => s.clone(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.to_string(),
        _ => serde_json::to_string_pretty(val).unwrap_or_else(|_| val.to_string()),
    }
}

fn cmd_profile_get(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "USAGE: manifold-frontend profile get <FILE_OR_PRESET> <KEY>\n       manifold-frontend profile get <KEY> (queries default config)"
        );
        return Ok(());
    }

    let (source, raw_key) = if args.len() == 1 {
        ("default", args[0].as_str())
    } else {
        (args[0].as_str(), args[1].as_str())
    };

    let lib = PresetLibrary::default();
    let (cfg, _) = load_config_from_source(source, None, &lib)?;
    let cfg_val = serde_json::to_value(&cfg).map_err(|e| e.to_string())?;
    let canonical = resolve_canonical_config_key(&cfg_val, raw_key)?;
    let val = get_json_by_dotted_path(&cfg_val, &canonical)
        .ok_or_else(|| format!("Key '{canonical}' not found"))?;
    println!("{}", format_json_value_for_stdout(val));
    Ok(())
}

fn cmd_profile_set(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "USAGE: manifold-frontend profile set <FILE_OR_PRESET> <KEY=VALUE>... [-o <OUTPUT_FILE>] [--category <CAT>]"
        );
        return Ok(());
    }

    let mut target: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut category: Option<PresetCategory> = None;
    let mut explicit_format: Option<ProfileFormat> = None;
    let mut assignments: Vec<(String, String)> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-o" | "--output" => {
                i += 1;
                out_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| "Missing path after `--output`".to_string())?,
                ));
            }
            "-c" | "--category" => {
                i += 1;
                category = Some(parse_preset_category(
                    args.get(i)
                        .ok_or_else(|| "Missing category after `--category`".to_string())?,
                )?);
            }
            "-f" | "--format" => {
                i += 1;
                explicit_format = Some(ProfileFormat::parse_str(
                    args.get(i)
                        .ok_or_else(|| "Missing format after `--format`".to_string())?,
                )?);
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                assignments.push(parse_key_eq_val(kv)?);
            }
            _ if a.starts_with('-') => {
                return Err(format!("Unknown option '{a}' for `profile set`"));
            }
            _ => {
                if target.is_none() && !a.contains('=') {
                    target = Some(a.to_string());
                } else {
                    assignments.push(parse_key_eq_val(a)?);
                }
            }
        }
        i += 1;
    }

    let target_str = target.ok_or_else(|| {
        "Missing target profile file or preset name for `profile set`.".to_string()
    })?;
    if assignments.is_empty() {
        return Err("No `KEY=VALUE` assignments specified for `profile set`.".to_string());
    }

    let mut lib = PresetLibrary::default();
    let target_path = Path::new(&target_str);
    let is_file_target = target_path.exists()
        || target_str.ends_with(".json")
        || target_str.ends_with(".toml")
        || target_str.contains('/')
        || target_str.contains('\\');

    let (mut cfg, matched_cat) = if is_file_target && !target_path.exists() {
        (SlicerConfig::default(), category)
    } else {
        load_config_from_source(&target_str, category, &lib)?
    };

    let mut applied_keys = Vec::with_capacity(assignments.len());
    for (k, v) in &assignments {
        let canonical = set_config_key_value(&mut cfg, k, v)?;
        let cfg_val = serde_json::to_value(&cfg).map_err(|e| e.to_string())?;
        let new_val_str = get_json_by_dotted_path(&cfg_val, &canonical)
            .map(format_json_value_for_stdout)
            .unwrap_or_else(|| v.clone());
        applied_keys.push((canonical, new_val_str));
    }

    if let Some(out) = out_path {
        let fmt = ProfileFormat::from_path_or_default(Some(&out), explicit_format);
        let text = if let Some(cat) = category {
            serialize_category_from_config(&cfg, cat, fmt)?
        } else {
            serialize_config(&cfg, fmt)?
        };
        if out.as_os_str() == "-" {
            println!("{text}");
        } else {
            fs::write(&out, format!("{text}\n"))
                .map_err(|e| format!("Failed to write '{}': {e}", out.display()))?;
            for (k, v) in &applied_keys {
                eprintln!("  {k} = {v}");
            }
            eprintln!("Updated profile written to {}", out.display());
        }
        return Ok(());
    }

    if is_file_target {
        let fmt = ProfileFormat::from_path_or_default(Some(target_path), explicit_format);
        let text = if let Some(cat) = category {
            serialize_category_from_config(&cfg, cat, fmt)?
        } else {
            serialize_config(&cfg, fmt)?
        };
        if let Some(parent) = target_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        fs::write(target_path, format!("{text}\n"))
            .map_err(|e| format!("Failed to write '{}': {e}", target_path.display()))?;
        for (k, v) in &applied_keys {
            eprintln!("  {k} = {v}");
        }
        eprintln!("Updated profile {}", target_path.display());
    } else if let Some(cat) = matched_cat.or(category) {
        let saved_name = lib.save_user_preset(cat, &target_str, &mut cfg)?;
        for (k, v) in &applied_keys {
            eprintln!("  {k} = {v}");
        }
        eprintln!("Saved {} user preset '{saved_name}'", cat.label());
    } else {
        return Err(format!(
            "Target '{target_str}' is neither a file path nor a known preset."
        ));
    }

    Ok(())
}

fn cmd_profile_diff(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "USAGE: manifold-frontend profile diff <SOURCE_A> [SOURCE_B]\n  (If SOURCE_B is omitted, compares SOURCE_A against the default SlicerConfig)"
        );
        return Ok(());
    }

    let lib = PresetLibrary::default();
    let (cfg_a, _) = if args.len() == 1 {
        (SlicerConfig::default(), None)
    } else {
        load_config_from_source(&args[0], None, &lib)?
    };
    let (cfg_b, _) = if args.len() == 1 {
        load_config_from_source(&args[0], None, &lib)?
    } else {
        load_config_from_source(&args[1], None, &lib)?
    };

    let diffs = diff_configs(&cfg_a, &cfg_b);
    if diffs.is_empty() {
        println!("No differences.");
    } else {
        for (key, val_a, val_b) in diffs {
            println!("{key}: {val_a} -> {val_b}");
        }
    }
    Ok(())
}

pub fn diff_configs(a: &SlicerConfig, b: &SlicerConfig) -> Vec<(String, String, String)> {
    let val_a = serde_json::to_value(a).unwrap_or(JsonValue::Null);
    let val_b = serde_json::to_value(b).unwrap_or(JsonValue::Null);
    let mut leaves_a = Vec::new();
    let mut leaves_b = Vec::new();
    collect_leaf_dotted_paths(&val_a, "", &mut leaves_a);
    collect_leaf_dotted_paths(&val_b, "", &mut leaves_b);

    let mut diffs = Vec::new();
    for (key, va) in &leaves_a {
        if let Some((_, vb)) = leaves_b.iter().find(|(kb, _)| kb == key) {
            let equal = match (va.as_f64(), vb.as_f64()) {
                (Some(fa), Some(fb)) => (fa - fb).abs() <= 1.0e-5,
                _ => va == vb,
            };
            if !equal {
                diffs.push((
                    key.clone(),
                     serde_json::to_string(va).unwrap_or_default(),
                    serde_json::to_string(vb).unwrap_or_default(),
                ));
            }
        }
    }
    diffs
}

fn cmd_profile_keys(args: &[String]) -> Result<(), String> {
    let mut filter: Option<String> = None;
    let mut show_docs = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!(
                    "USAGE: manifold-frontend profile keys [--filter <SUBSTRING>] [--docs]"
                );
                return Ok(());
            }
            "--filter" | "-f" => {
                i += 1;
                filter = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing pattern after `--filter`".to_string())?
                        .to_ascii_lowercase(),
                );
            }
            "--docs" | "-d" => {
                show_docs = true;
            }
            other if !other.starts_with('-') && filter.is_none() => {
                filter = Some(other.to_ascii_lowercase());
            }
            other => {
                return Err(format!("Unknown option '{other}' for `profile keys`"));
            }
        }
        i += 1;
    }

    let cfg = SlicerConfig::default();
    let val = serde_json::to_value(&cfg).map_err(|e| e.to_string())?;
    let mut leaves = Vec::new();
    collect_leaf_dotted_paths(&val, "", &mut leaves);

    for (key, default_val) in leaves {
        let doc_opt = PARAMETER_DOCS.iter().find(|d| {
            d.key == key
                || key.starts_with(d.key)
                || (d.key.starts_with("tools.")
                    && key.starts_with("tools.0.")
                    && d.key["tools.".len()..] == key["tools.0.".len()..])
        });

        if let Some(ref pat) = filter {
            let key_matches = key.to_ascii_lowercase().contains(pat);
            let doc_matches = doc_opt.is_some_and(|d| {
                d.title.to_ascii_lowercase().contains(pat)
                    || d.summary.to_ascii_lowercase().contains(pat)
                    || d.category.to_ascii_lowercase().contains(pat)
            });
            if !key_matches && !doc_matches {
                continue;
            }
        }

        let def_str = serde_json::to_string(&default_val).unwrap_or_default();
        if let Some(doc) = doc_opt {
            if doc.units.is_empty() {
                println!("{:<46} = {:<18} # {}", key, def_str, doc.summary);
            } else {
                println!(
                    "{:<46} = {:<18} [{}] # {}",
                    key, def_str, doc.units, doc.summary
                );
            }
            if show_docs {
                println!("    {}", doc.details);
            }
        } else {
            println!("{:<46} = {}", key, def_str);
        }
    }
    Ok(())
}

fn cmd_profile_list(args: &[String]) -> Result<(), String> {
    let mut cat_filter: Option<PresetCategory> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                println!("USAGE: manifold-frontend profile list [--category printer|tool|filament|process]");
                return Ok(());
            }
            "-c" | "--category" => {
                i += 1;
                cat_filter = Some(parse_preset_category(
                    args.get(i)
                        .ok_or_else(|| "Missing category after `--category`".to_string())?,
                )?);
            }
            other if !other.starts_with('-') && cat_filter.is_none() => {
                cat_filter = Some(parse_preset_category(other)?);
            }
            other => {
                return Err(format!("Unknown option '{other}' for `profile list`"));
            }
        }
        i += 1;
    }

    let lib = PresetLibrary::default();
    let cats: Vec<PresetCategory> = if let Some(c) = cat_filter {
        vec![c]
    } else {
        PresetCategory::ALL.to_vec()
    };

    for cat in cats {
        println!("=== {} Presets ===", cat.label());
        for (name, is_builtin) in lib.preset_names(cat) {
            let badge = if is_builtin { "[factory]" } else { "[user]   " };
            println!("  {badge} {name}");
        }
    }
    if let Some(root) = PresetLibrary::presets_root_dir() {
        println!("\nUser presets directory: {}", root.display());
    }
    Ok(())
}

fn cmd_profile_save_preset(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "USAGE: manifold-frontend profile save-preset <printer|tool|filament|process> <NAME> [--from <FILE>] [-s KEY=VALUE...]"
        );
        return Ok(());
    }

    let cat = parse_preset_category(&args[0])?;
    let name = args[1].clone();
    let mut spec = ProfileBuildSpec::default();

    let mut i = 2;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--from" | "-p" | "--profile" => {
                i += 1;
                spec.base_files.push(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| format!("Missing file path after `{a}`"))?,
                ));
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                spec.overrides.push(parse_key_eq_val(kv)?);
            }
            _ if !a.starts_with('-') && a.contains('=') => {
                spec.overrides.push(parse_key_eq_val(a)?);
            }
            other => {
                return Err(format!(
                    "Unknown option '{other}' for `profile save-preset`"
                ));
            }
        }
        i += 1;
    }

    let mut lib = PresetLibrary::default();
    let mut cfg = SlicerConfig::default();
    spec.apply_onto(&mut cfg, &lib)?;
    let saved_name = lib.save_user_preset(cat, &name, &mut cfg)?;
    println!("Saved {} preset '{saved_name}'.", cat.label());
    Ok(())
}

fn cmd_profile_delete_preset(args: &[String]) -> Result<(), String> {
    if args.len() < 2 || args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "USAGE: manifold-frontend profile delete-preset <printer|tool|filament|process> <NAME>"
        );
        return Ok(());
    }
    let cat = parse_preset_category(&args[0])?;
    let name = args[1].as_str();

    let mut lib = PresetLibrary::default();
    let mut cfg = SlicerConfig::default();
    if !lib.select_and_apply(cat, name, &mut cfg) {
        return Err(format!("{} preset '{name}' not found.", cat.label()));
    }
    let removed = lib.delete_selected_user_preset(cat, &mut cfg)?;
    println!("Deleted {} user preset '{removed}'.", cat.label());
    Ok(())
}

fn cmd_profile_validate(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        println!("USAGE: manifold-frontend profile validate <FILE>...");
        return Ok(());
    }
    for arg in args {
        let path = Path::new(arg);
        let mut cfg = SlicerConfig::default();
        apply_profile_file_onto_config(&mut cfg, path)?;
        println!(
            "OK: '{}' is a valid Manifold2 profile (Printer='{}', Tool='{}', Filament='{}', Process='{}').",
            path.display(),
            cfg.printer_preset_name,
            cfg.tool_preset_name,
            cfg.filament_preset_name,
            cfg.process_preset_name
        );
    }
    Ok(())
}

// ============================================================================
// Headless Calibration Pattern Generator (`manifold-frontend calibrate ...`)
// ============================================================================

pub fn cmd_calibrate(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help" || a == "help") {
        print_calibrate_help();
        return Ok(());
    }

    let kind_str = args[0].trim().to_ascii_lowercase();
    let mut profile_spec = ProfileBuildSpec::default();
    let mut output_path: Option<PathBuf> = None;
    let mut tool_selector: Option<String> = None;
    let mut pass_num: Option<u32> = None;
    let mut mode_str: Option<String> = None;
    let mut start_val: Option<f32> = None;
    let mut end_val: Option<f32> = None;
    let mut step_val: Option<f32> = None;
    let mut speed_val: Option<f32> = None;
    let mut accel_val: Option<f32> = None;
    let mut quiet = false;

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-p" | "--profile" => {
                i += 1;
                profile_spec.base_files.push(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| "Missing file path after `--profile`".to_string())?,
                ));
            }
            "--printer" => {
                i += 1;
                profile_spec.printer = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--printer`".to_string())?
                        .clone(),
                );
            }
            "--tool" => {
                i += 1;
                profile_spec.tool = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--tool`".to_string())?
                        .clone(),
                );
            }
            "-t" | "--tool-id" | "--tool-index" => {
                i += 1;
                tool_selector = Some(
                    args.get(i)
                        .ok_or_else(|| format!("Missing tool ID/index after `{a}`"))?
                        .clone(),
                );
            }
            "--filament" => {
                i += 1;
                profile_spec.filament = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--filament`".to_string())?
                        .clone(),
                );
            }
            "--process" => {
                i += 1;
                profile_spec.process = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing value after `--process`".to_string())?
                        .clone(),
                );
            }
            "-s" | "--set" | "-D" => {
                i += 1;
                let kv = args
                    .get(i)
                    .ok_or_else(|| format!("Missing `KEY=VALUE` after `{a}`"))?;
                profile_spec.overrides.push(parse_key_eq_val(kv)?);
            }
            "--pass" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing pass number after `--pass`".to_string())?;
                pass_num = Some(
                    v.parse::<u32>()
                        .map_err(|_| format!("Invalid `--pass` number '{v}'"))?,
                );
            }
            "--mode" => {
                i += 1;
                mode_str = Some(
                    args.get(i)
                        .ok_or_else(|| "Missing mode after `--mode`".to_string())?
                        .to_ascii_lowercase(),
                );
            }
            "--start" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing value after `--start`".to_string())?;
                start_val = Some(
                    v.parse::<f32>()
                        .map_err(|_| format!("Invalid `--start` value '{v}'"))?,
                );
            }
            "--end" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing value after `--end`".to_string())?;
                end_val = Some(
                    v.parse::<f32>()
                        .map_err(|_| format!("Invalid `--end` value '{v}'"))?,
                );
            }
            "--step" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing value after `--step`".to_string())?;
                step_val = Some(
                    v.parse::<f32>()
                        .map_err(|_| format!("Invalid `--step` value '{v}'"))?,
                );
            }
            "--speed" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing value after `--speed`".to_string())?;
                speed_val = Some(
                    v.parse::<f32>()
                        .map_err(|_| format!("Invalid `--speed` value '{v}'"))?,
                );
            }
            "--accel" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| "Missing value after `--accel`".to_string())?;
                accel_val = Some(
                    v.parse::<f32>()
                        .map_err(|_| format!("Invalid `--accel` value '{v}'"))?,
                );
            }
            "-o" | "--output" => {
                i += 1;
                output_path = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| "Missing path after `--output`".to_string())?,
                ));
            }
            "-q" | "--quiet" => {
                quiet = true;
            }
            other => {
                return Err(format!(
                    "Unknown option '{other}' for `calibrate`. Run `manifold-frontend calibrate --help`."
                ));
            }
        }
        i += 1;
    }

    let lib = PresetLibrary::default();
    let mut cfg = SlicerConfig::default();
    profile_spec.apply_onto(&mut cfg, &lib)?;

    let resolved_tool_index: Option<usize> = if let Some(ref raw_sel) = tool_selector {
        let trimmed = raw_sel.trim().trim_start_matches(['t', 'T']);
        let num = trimmed
            .parse::<u32>()
            .map_err(|_| format!("Invalid tool selector '{raw_sel}' (expected tool ID like `0`, `1`, `T0`, `T1`)"))?;
        let idx = cfg
            .tools
            .iter()
            .position(|t| t.id == num)
            .or_else(|| {
                let as_usize = num as usize;
                (as_usize < cfg.tools.len()).then_some(as_usize)
            })
            .ok_or_else(|| {
                format!(
                    "Tool '{raw_sel}' not found in configuration (available tools: {})",
                    cfg.tools
                        .iter()
                        .map(|t| format!("T{} ({})", t.id, t.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        cfg.active_tool_index = idx;
        Some(idx)
    } else {
        None
    };
    sync_config_to_backend(&cfg);

    let (planned, default_filename) = match kind_str.as_str() {
        "em" | "flow" | "swell" | "extrusion-multiplier" => {
            let mut spec = EmCalibrationSpec::default();
            spec.tool_index = resolved_tool_index;
            let pass = match pass_num.unwrap_or(1) {
                1 => EmCalibrationPass::Pass1Coarse,
                2 => EmCalibrationPass::Pass2Fine,
                3 => EmCalibrationPass::Pass3HighFlowDieSwell,
                other => {
                    return Err(format!(
                        "Unsupported `--pass {other}` for `calibrate em` (expected 1, 2, or 3)"
                    ));
                }
            };
            spec.apply_pass_defaults(pass, cfg.gcode.extrusion_multiplier);
            if pass == EmCalibrationPass::Pass3HighFlowDieSwell {
                if let Some(s) = start_val {
                    spec.start_swell_high = s;
                }
                if let Some(e) = end_val {
                    spec.end_swell_high = e;
                }
                if let Some(st) = step_val {
                    spec.step_swell_high = st;
                }
            } else {
                if let Some(s) = start_val {
                    spec.start_delta_pct = s;
                }
                if let Some(e) = end_val {
                    spec.end_delta_pct = e;
                }
                if let Some(st) = step_val {
                    spec.step_delta_pct = st;
                }
            }
            let fname = match spec.pass {
                EmCalibrationPass::Pass1Coarse => "calib_em_pass1_coarse.gcode",
                EmCalibrationPass::Pass2Fine => "calib_em_pass2_fine.gcode",
                EmCalibrationPass::Pass3HighFlowDieSwell => "calib_em_pass3_dieswell.gcode",
            };
            (generate_em_calibration_gcode(&spec, &cfg), fname)
        }
        "pa" | "pressure-advance" => {
            let mut spec = PaCalibrationSpec::default();
            spec.tool_index = resolved_tool_index;
            if let Some(ref m) = mode_str {
                spec.mode = match m.as_str() {
                    "dual" | "dual-band" | "dualband" => PaCalibrationMode::DualBandShearThinning,
                    "single" | "single-band" | "singleband" => PaCalibrationMode::SingleBand,
                    other => {
                        return Err(format!(
                            "Unsupported `--mode {other}` for `calibrate pa` (expected `dual` or `single`)"
                        ));
                    }
                };
            }
            if let Some(s) = start_val {
                spec.pa_start_s = s;
            }
            if let Some(e) = end_val {
                spec.pa_end_s = e;
            }
            if let Some(st) = step_val {
                spec.pa_step_s = st;
            }
            if let Some(spd) = speed_val {
                spec.fast_speed_mms = spd;
            }
            if let Some(acc) = accel_val {
                spec.accel_mms2 = acc;
            }
            let fname = match spec.mode {
                PaCalibrationMode::DualBandShearThinning => "calib_pa_dualband.gcode",
                PaCalibrationMode::SingleBand => "calib_pa_singleband.gcode",
            };
            (generate_pa_calibration_gcode(&spec, &cfg), fname)
        }
        "transient" | "melt-pressure" | "transient-pressure" => {
            let mut spec = TransientCalibrationSpec::default();
            spec.tool_index = resolved_tool_index;
            spec.base_exponent_gamma = cfg.gcode.transient_pressure_exponent;
            spec.base_tau_s = cfg.gcode.transient_pressure_tau_s;
            spec.min_flow_multiplier = cfg.gcode.transient_min_flow_multiplier;
            spec.pass = match pass_num.unwrap_or(1) {
                1 => TransientCalibrationPass::Pass1ExponentGamma,
                2 => TransientCalibrationPass::Pass2RelaxationTau,
                other => {
                    return Err(format!(
                        "Unsupported `--pass {other}` for `calibrate transient` (expected 1 or 2)"
                    ));
                }
            };
            match spec.pass {
                TransientCalibrationPass::Pass1ExponentGamma => {
                    if let Some(s) = start_val {
                        spec.start_gamma = s;
                    }
                    if let Some(e) = end_val {
                        spec.end_gamma = e;
                    }
                    if let Some(st) = step_val {
                        spec.step_gamma = st;
                    }
                }
                TransientCalibrationPass::Pass2RelaxationTau => {
                    if let Some(s) = start_val {
                        spec.start_tau_s = s;
                    }
                    if let Some(e) = end_val {
                        spec.end_tau_s = e;
                    }
                    if let Some(st) = step_val {
                        spec.step_tau_s = st;
                    }
                }
            }
            if let Some(spd) = speed_val {
                spec.constant_test_speed_mms = spd;
            }
            let fname = match spec.pass {
                TransientCalibrationPass::Pass1ExponentGamma => "calib_transient_pass1_gamma.gcode",
                TransientCalibrationPass::Pass2RelaxationTau => "calib_transient_pass2_tau.gcode",
            };
            (
                generate_transient_pressure_calibration_gcode(&spec, &cfg),
                fname,
            )
        }
        other => {
            return Err(format!(
                "Unknown calibration kind '{other}' (expected `em`, `pa`, or `transient`)"
            ));
        }
    };

    let out_path = output_path.unwrap_or_else(|| PathBuf::from(default_filename));
    if out_path.as_os_str() == "-" {
        io::stdout()
            .write_all(planned.gcode_text.as_bytes())
            .map_err(|e| e.to_string())?;
    } else {
        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create directory {}: {e}", parent.display()))?;
            }
        }
        fs::write(&out_path, planned.gcode_text.as_bytes())
            .map_err(|e| format!("Failed to write {}: {e}", out_path.display()))?;
    }

    if !quiet {
        eprintln!(
            "[Manifold CLI] Generated calibration G-code '{}' ({} layers, {} extruding moves, {:.2} m / {:.1} g, est. {:.0} s)",
            out_path.display(),
            planned.layers.len(),
            planned.extruding_move_count,
            planned.total_filament_mm / 1000.0,
            planned.total_filament_grams,
            planned.total_time_s,
        );
    }

    Ok(())
}

// ============================================================================
// Unit Tests for CLI Profile Editing & Headless Slicing
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::TEST_BACKEND_LOCK;

    #[test]
    fn test_profile_json_and_toml_roundtrip_and_partial_merge() {
        let mut cfg = SlicerConfig::default();
        let lib = PresetLibrary::default();

        // Apply a partial JSON profile containing both nested sections and flat/shorthand keys
        let partial_json = r#"{
            "slicing": {
                "layer_height": {
                    "target_mm": 0.16
                }
            },
            "infill_density_percent": 35.0,
            "bed_temperature_celsius": 75.0,
            "nozzle_diameter_mm": 0.60,
            "bed_tilt_rx_deg": 20.0
        }"#;
        apply_profile_str_onto_config(&mut cfg, partial_json, None).unwrap();

        assert!((cfg.slicing.layer_height.target_mm - 0.16).abs() < 1.0e-5);
        assert!((cfg.slicing.infill_density_percent - 35.0).abs() < 1.0e-5);
        assert!((cfg.gcode.bed_temperature_celsius - 75.0).abs() < 1.0e-5);
        assert!((cfg.active_tool().unwrap().nozzle_diameter_mm - 0.60).abs() < 1.0e-5);
        // Rule 2: Updating `bed_tilt_rx_deg` in profile automatically updates `gravity_direction_bed`
        assert!(cfg.machine.gravity_direction_bed[1].abs() > 0.1);

        // Set dotted and shorthand keys via `set_config_key_value`
        let resolved1 =
            set_config_key_value(&mut cfg, "slicing.inner_wall_count", "4").unwrap();
        assert_eq!(resolved1, "slicing.inner_wall_count");
        assert_eq!(cfg.slicing.inner_wall_count, 4);

        let resolved2 =
            set_config_key_value(&mut cfg, "outer_wall_speed_mms", "65.5").unwrap();
        assert_eq!(resolved2, "gcode.outer_wall_speed_mms");
        assert!((cfg.gcode.outer_wall_speed_mms - 65.5).abs() < 1.0e-5);

        // Diff against default config
        let diffs = diff_configs(&SlicerConfig::default(), &cfg);
        assert!(diffs
            .iter()
            .any(|(k, _, _)| k == "slicing.layer_height.target_mm"));
        assert!(diffs
            .iter()
            .any(|(k, _, _)| k == "slicing.infill_density_percent"));
        assert!(diffs
            .iter()
            .any(|(k, _, _)| k == "gcode.outer_wall_speed_mms"));

        // Serialize to JSON and reload into a fresh config
        let full_json = serialize_config(&cfg, ProfileFormat::Json).unwrap();
        let mut cfg_reloaded = SlicerConfig::default();
        apply_profile_str_onto_config(&mut cfg_reloaded, &full_json, None).unwrap();
        assert!(diff_configs(&cfg, &cfg_reloaded).is_empty());

        // Apply named presets via `ProfileBuildSpec`
        let spec = ProfileBuildSpec {
            base_files: Vec::new(),
            printer: Some("5-Axis Tilting Build Plate (250×210)".to_string()),
            tool: Some("0.6mm High-Flow".to_string()),
            filament: Some("Generic PETG".to_string()),
            process: Some("0.12mm Fine Conformal".to_string()),
            overrides: vec![("infill_density_percent".to_string(), "40".to_string())],
        };
        let mut cfg_from_spec = SlicerConfig::default();
        spec.apply_onto(&mut cfg_from_spec, &lib).unwrap();
        assert!((cfg_from_spec.bed.width_x - 250.0).abs() < 1.0e-4);
        assert_eq!(cfg_from_spec.gcode.filament_type, "PETG");
        assert!((cfg_from_spec.slicing.layer_height.target_mm - 0.12).abs() < 1.0e-4);
        assert!((cfg_from_spec.slicing.infill_density_percent - 40.0).abs() < 1.0e-4);
    }

    #[test]
    fn test_cli_headless_slice_with_profile_json() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_dir = std::env::temp_dir().join("manifold2_cli_test");
        let _ = fs::create_dir_all(&tmp_dir);
        let profile_path = tmp_dir.join("test_cli_profile.json");
        let gcode_path = tmp_dir.join("test_cli_out.gcode");

        // 1. Create a profile JSON using `cmd_profile_init`
        cmd_profile_init(&[
            profile_path.to_string_lossy().into_owned(),
            "--filament".to_string(),
            "Generic PETG".to_string(),
            "-s".to_string(),
            "slicing.layer_height.target_mm=0.25".to_string(),
            "-s".to_string(),
            "slicing.infill_density_percent=15".to_string(),
        ])
        .unwrap();
        assert!(profile_path.exists());

        // 2. Edit the profile JSON using `cmd_profile_set`
        cmd_profile_set(&[
            profile_path.to_string_lossy().into_owned(),
            "gcode.bed_temperature_celsius=78".to_string(),
        ])
        .unwrap();

        let mut verify_cfg = SlicerConfig::default();
        apply_profile_file_onto_config(&mut verify_cfg, &profile_path).unwrap();
        assert_eq!(verify_cfg.gcode.filament_type, "PETG");
        assert!((verify_cfg.gcode.bed_temperature_celsius - 78.0).abs() < 1.0e-4);

        // 3. Slice `Test Objects/testobj1.step` headlessly using the edited profile JSON
        let step_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("Test Objects/testobj1.step");
        if step_path.exists() {
            let slice_opts = SliceCommandOptions {
                input_files: vec![step_path],
                profile_spec: ProfileBuildSpec {
                    base_files: vec![profile_path.clone()],
                    printer: None,
                    tool: None,
                    filament: None,
                    process: None,
                    overrides: Vec::new(),
                },
                output_path: Some(gcode_path.clone()),
                save_3mf_path: None,
                plate_filter: None,
                arrange_override: None,
                place_on_face: None,
                rotate_euler_deg: None,
                quiet: true,
            };
            let summary = execute_headless_slice(&slice_opts).unwrap();
            assert_eq!(summary.written_gcode_files.len(), 1);
            assert!(summary.total_layers > 0);
            assert!(summary.total_extruding_moves > 0);
            assert!(summary.total_filament_mm > 0.0);

            let gcode_text = fs::read_to_string(&gcode_path).unwrap();
            assert!(gcode_text.contains("EXCLUDE_OBJECT_DEFINE NAME="));
            assert!(gcode_text.contains("M140 S78"));
            assert!(gcode_text.contains("; filament_type = PETG"));
        }

        let _ = fs::remove_file(&profile_path);
        let _ = fs::remove_file(&gcode_path);
    }

    #[test]
    fn test_cli_calibrate_em_pa_and_transient_subcommands() {
        let _guard = TEST_BACKEND_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_dir = std::env::temp_dir().join("manifold2_cli_calib_test");
        let _ = fs::create_dir_all(&tmp_dir);

        let em_p3_path = tmp_dir.join("calib_em_p3.gcode");
        cmd_calibrate(&[
            "em".to_string(),
            "--pass".to_string(),
            "3".to_string(),
            "-s".to_string(),
            "bed_tilt_rx_deg=12.0".to_string(),
            "-q".to_string(),
            "-o".to_string(),
            em_p3_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let em_text = fs::read_to_string(&em_p3_path).unwrap();
        assert!(em_text.contains("Pass 3: High-Flow Die Swell"));

        let pa_dual_path = tmp_dir.join("calib_pa_dual.gcode");
        cmd_calibrate(&[
            "pa".to_string(),
            "--mode".to_string(),
            "dual".to_string(),
            "--start".to_string(),
            "0.0".to_string(),
            "--end".to_string(),
            "0.03".to_string(),
            "--step".to_string(),
            "0.01".to_string(),
            "-q".to_string(),
            "-o".to_string(),
            pa_dual_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let pa_text = fs::read_to_string(&pa_dual_path).unwrap();
        assert!(pa_text.contains("SET_PRESSURE_ADVANCE ADVANCE="));

        let trans_p2_path = tmp_dir.join("calib_trans_p2.gcode");
        cmd_calibrate(&[
            "transient".to_string(),
            "--pass".to_string(),
            "2".to_string(),
            "-s".to_string(),
            "transient_pressure_exponent=0.28".to_string(),
            "-q".to_string(),
            "-o".to_string(),
            trans_p2_path.to_string_lossy().into_owned(),
        ])
        .unwrap();
        let trans_text = fs::read_to_string(&trans_p2_path).unwrap();
        assert!(trans_text.contains("Calibration_Transient_Melt_Pressure"));

        let _ = fs::remove_file(&em_p3_path);
        let _ = fs::remove_file(&pa_dual_path);
        let _ = fs::remove_file(&trans_p2_path);
    }
}
