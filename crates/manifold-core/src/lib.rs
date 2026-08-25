//! Manifold slicing engine.
//!
//! `manifold-core` turns a mesh into non-planar toolpaths and emits Gcode.
//! It has no UI or CLI dependencies so it can run headless (e.g. embedded
//! in a service) or be driven by the `manifold-cli` front-end.

pub mod bounds;
pub mod error;
pub mod extrusion;
pub mod fluid_dynamics;
pub mod gcode;
pub mod ids;
pub mod infill;
pub mod kinematics;
pub mod machine;
pub mod material;
pub mod mesh;
pub mod object;
pub mod order_field;
pub mod ordering;
pub mod polygon2d;
pub mod slicing;
pub mod statistics;
pub mod stl;
pub mod threemf;
pub mod tool;
pub mod toolpath;
pub mod transform;
pub mod wave_overhang;
pub mod workspace;

pub use error::{Error, Result};
pub use statistics::{
    compute_print_statistics, compute_print_statistics_with_machine, PrintStatistics,
};
pub use workspace::Workspace;

/// Slicer configuration shared across the pipeline.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SlicerConfig {
    pub layer_height: f64,
    /// First layer height (mm) for the initial layer touching the print bed.
    /// When `None` (the default), defaults to `layer_height`.
    #[serde(default)]
    pub first_layer_height: Option<f64>,
    /// First layer print speed (mm/min), defaulting to 40% of `print_speed` when `None`.
    #[serde(default)]
    pub first_layer_print_speed: Option<f64>,
    /// First layer extrusion flow multiplier, defaulting to `1.0` when `None`.
    #[serde(default)]
    pub first_layer_extrusion_multiplier: Option<f64>,
    /// First layer line width (mm), defaulting to 130% of `nozzle_diameter` (or `wall_line_width`) when `None`.
    #[serde(default)]
    pub first_layer_line_width: Option<f64>,
    /// Part cooling fan speed percentage (0.0 to 100.0), defaulting to 100.0% when `None`.
    #[serde(default)]
    pub fan_speed_percent: Option<f64>,
    /// Part cooling fan speed percentage (0.0 to 100.0) for overhang/bridge moves, defaulting to 100.0% when `None`.
    #[serde(default)]
    pub overhang_fan_speed_percent: Option<f64>,
    /// Number of initial layers to keep part cooling fan disabled, defaulting to 1 (disabled on layer 0) when `None`.
    #[serde(default)]
    pub fan_layer_delay: Option<u32>,
    pub nozzle_diameter: f64,
    /// Strategy used to decide the order objects are printed in. See
    /// `ordering` module and ROADMAP.md open decision #2.
    pub object_ordering: ordering::ObjectOrderingKind,
    /// Nozzle-center line width for a single wall/perimeter pass, in mm.
    /// Also used as the spacing between successive wall passes.
    pub wall_line_width: f64,
    /// Total desired shell thickness (mm), i.e. how far inward the walls
    /// should extend from the outer surface. The actual number of wall
    /// passes is derived (see [`SlicerConfig::wall_count`]) by rounding
    /// `shell_thickness / wall_line_width` to the nearest whole wall,
    /// clamped to a minimum of one.
    pub shell_thickness: f64,
    /// Inset of the outermost wall's nozzle-center path from the true
    /// mesh surface, in mm. Defaults to half the nozzle diameter so the
    /// nozzle's outer edge (not its center) lands on the surface.
    pub wall_offset: f64,
    /// Infill pattern for sparse interior regions. Defaults to `InfillPatternKind::Cubic`.
    #[serde(default)]
    pub sparse_infill_pattern: Option<infill::InfillPatternKind>,
    /// Infill pattern for solid top/bottom layers. Defaults to `InfillPatternKind::AllWalls`.
    #[serde(default)]
    pub solid_infill_pattern: Option<infill::InfillPatternKind>,
    /// Legacy infill pattern field preserved for backwards compatibility.
    #[serde(default)]
    pub infill_pattern: infill::InfillPatternKind,
    /// Nozzle-center line width for infill passes, in mm. Also the scan-
    /// line spacing for the `Monotonic` pattern. Defaults to
    /// `nozzle_diameter`.
    pub infill_line_width: f64,
    /// Base infill scan-line angle (degrees), before per-layer
    /// alternation. `infill::MonotonicInfill` alternates ±this angle by
    /// layer index (even layers add it, odd layers subtract it),
    /// measured relative to the source object's own orientation (see
    /// `Transform::in_plane_rotation_angle`) so rotating an object
    /// rotates its infill with it. Defaults to `45.0`.
    pub infill_angle_deg: f64,
    /// Fraction of the *sparse* infill region (i.e. the part of
    /// `Layer::infill_boundary` that is not `Layer::solid_fill_boundary` —
    /// interior fill that isn't closing a top/bottom surface) to
    /// actually fill with material, in `0.0..=1.0`. Scales scan-line
    /// spacing inversely (`infill_line_width / infill_density`), so `1.0`
    /// packs lines at `infill_line_width` spacing (fully solid) and lower
    /// values space them further apart. `0.0` omits sparse infill
    /// entirely. Solid fill (`Layer::solid_fill_boundary`, e.g. top/bottom
    /// surface layers) always prints at full density regardless of this
    /// setting. Defaults to `0.2` (20%).
    pub infill_density: f64,
    /// Number of fully solid layers to generate at the top of each
    /// object (adjacent to any facing-up exterior surface), replacing
    /// what would otherwise be sparse infill with a solid fill pattern.
    /// Defaults to `3`.
    pub top_layers: usize,
    /// Number of fully solid layers to generate at the bottom of each
    /// object (adjacent to any facing-down exterior surface), replacing
    /// what would otherwise be sparse infill with a solid fill pattern.
    /// Defaults to `3`.
    pub bottom_layers: usize,
    /// Which order field slicing walks isosurfaces of. See
    /// `order_field::OrderFieldKind`. Defaults to `Height`, i.e. today's
    /// exact flat planar slicing along `slicing::BUILD_DIRECTION`.
    pub order_field: order_field::OrderFieldKind,
    /// Whether the `Eikonal` order field blends with the top surface to make
    /// layers lie parallel to top surfaces and follow upper curvature.
    /// Defaults to `false`.
    #[serde(default)]
    pub eikonal_conform_top_surfaces: bool,
    /// Apex point of the cone used when `order_field` is
    /// `OrderFieldKind::Conical`. Inert (unused) otherwise. Defaults to
    /// the origin.
    pub order_field_apex: glam::DVec3,
    /// Axis the cone opens along when `order_field` is
    /// `OrderFieldKind::Conical`. Inert (unused) otherwise. Defaults to
    /// `slicing::BUILD_DIRECTION`, matching the `Height` field's
    /// direction so switching kinds is a smooth transition.
    pub order_field_axis: glam::DVec3,
    /// Cone steepness used when `order_field` is `OrderFieldKind::Conical`
    /// (`0.0` degenerates to a flat height field). Inert (unused)
    /// otherwise. Defaults to `0.0`.
    pub order_field_slope: f64,
    /// Density of the printing filament material in g/cm³ (default 1.24 g/cm³ for PLA/PETG/ASA).
    #[serde(default)]
    pub filament_density_g_cm3: Option<f64>,
    /// Diameter of the filament being fed into the extruder, in
    /// millimeters — the cross-section `crate::extrusion` divides a
    /// segment's deposited bead volume by to get linear filament feed
    /// length (the Gcode `E` axis). Defaults to `1.75`, the common
    /// consumer-FDM standard (the other being `2.85`/`3.0`).
    pub filament_diameter: f64,
    /// Gcode template emitted once at the very start of the program,
    /// before any tool-select/path Gcode. Supports `{print_min_x}`,
    /// `{print_min_y}`, `{print_max_x}`, `{print_max_y}` placeholders (see
    /// `gcode::interpolate`), substituted with the first layer's XY
    /// bounding box at emit time -- mirrors the common Klipper/Moonraker
    /// `PRINT_START` macro-invocation convention (e.g. Voron configs),
    /// where the slicer's "start gcode" is just a macro call with
    /// bed/chamber temps and the print's footprint as named parameters.
    /// Defaults to a `PRINT_START` invocation with placeholder temps.
    pub start_gcode: String,
    /// Gcode template emitted once at the very end of the program, after
    /// all path Gcode. No placeholders are substituted by default.
    /// Defaults to a bare `PRINT_END` macro invocation.
    pub end_gcode: String,
    /// Feedrate (Gcode `F`, mm/min) for non-extruding travel moves
    /// (`toolpath::MoveKind::Travel`). Defaults to `9000.0` (150 mm/s), a
    /// common consumer-FDM travel speed. `#[serde(default)]` so a saved
    /// profile from before this field existed still deserializes.
    #[serde(default = "default_travel_speed")]
    pub travel_speed: f64,
    /// Feedrate (Gcode `F`, mm/min) for extruding moves (walls/infill).
    /// Defaults to `3000.0` (50 mm/s), a common consumer-FDM print speed.
    /// `#[serde(default)]` so a saved profile from before this field
    /// existed still deserializes.
    #[serde(default = "default_print_speed")]
    pub print_speed: f64,
    /// Outer wall print speed (mm/min), defaulting to 60% of `print_speed` when `None`.
    #[serde(default)]
    pub outer_wall_speed: Option<f64>,
    /// Inner wall print speed (mm/min), defaulting to `print_speed` when `None`.
    #[serde(default)]
    pub inner_wall_speed: Option<f64>,
    /// Sparse infill print speed (mm/min), defaulting to `print_speed` when `None`.
    #[serde(default)]
    pub infill_speed: Option<f64>,
    /// Solid infill print speed (mm/min), defaulting to 80% of `print_speed` when `None`.
    #[serde(default)]
    pub solid_infill_speed: Option<f64>,
    /// Bridge / overhang print speed (mm/min), defaulting to 50% of `print_speed` when `None`.
    #[serde(default)]
    pub bridge_speed: Option<f64>,
    /// Default printing acceleration (mm/s²), defaulting to 5000.0 when `None`.
    #[serde(default)]
    pub default_acceleration: Option<f64>,
    /// Outer wall acceleration (mm/s²), defaulting to 2500.0 when `None`.
    #[serde(default)]
    pub outer_wall_acceleration: Option<f64>,
    /// Inner wall acceleration (mm/s²), defaulting to 5000.0 when `None`.
    #[serde(default)]
    pub inner_wall_acceleration: Option<f64>,
    /// Infill acceleration (mm/s²), defaulting to 7000.0 when `None`.
    #[serde(default)]
    pub infill_acceleration: Option<f64>,
    /// Travel acceleration (mm/s²), defaulting to 10000.0 when `None`.
    #[serde(default)]
    pub travel_acceleration: Option<f64>,
    /// First layer acceleration (mm/s²), defaulting to 2000.0 when `None`.
    #[serde(default)]
    pub first_layer_acceleration: Option<f64>,
    /// Maximum volumetric extrusion speed (mm³/s). When set and > 0, linear feedrates
    /// are automatically capped per-segment based on bead cross-sectional area.
    #[serde(default)]
    pub max_volumetric_speed: Option<f64>,
    /// Pressure advance value (seconds / k_pa).
    #[serde(default)]
    pub pressure_advance: Option<f64>,
    /// Distance (mm) before a retraction over which extrusion rate is tapered down.
    #[serde(default)]
    pub pre_retract_taper_distance: Option<f64>,
    /// Retraction distance in mm (default 0.8 mm).
    #[serde(default)]
    pub retraction_length: Option<f64>,
    /// Retraction speed in mm/min (default 3000.0 mm/min = 50 mm/s).
    #[serde(default)]
    pub retraction_speed: Option<f64>,
    /// Unretract / prime speed in mm/min (defaulting to `retraction_speed`).
    #[serde(default)]
    pub unretract_speed: Option<f64>,
    /// Extra length in mm to extrude upon unretract (default 0.0 mm).
    #[serde(default)]
    pub unretract_extra_length: Option<f64>,
    /// Distance (mm) to wipe the nozzle along/inward during retraction (default 1.0 mm).
    #[serde(default)]
    pub wipe_distance: Option<f64>,
    /// Whether nozzle wipe on retraction is enabled (default false).
    #[serde(default)]
    pub wipe_enabled: bool,
    /// Whether to emit firmware G10/G11 retractions instead of explicit G1 E moves (default false).
    #[serde(default)]
    pub use_firmware_retraction: bool,
    /// Whether non-planar scarf joint perimeter seam ramping is enabled (default true).
    #[serde(default = "default_scarf_joint_enabled")]
    pub scarf_joint_enabled: bool,
    /// Length (mm) over which closed wall loops ramp extrusion up on entry and down on overlap (default 3.0 mm).
    #[serde(default)]
    pub scarf_joint_length: Option<f64>,
    /// Whether Z-hop (lift-before-travel / lower-after-arrival) is enabled.
    /// When `false`, `plan`/`emit` behave exactly as before this field
    /// existed: Z tracks `point.z` unmodified even across
    /// `toolpath::MoveKind::Travel` segments. `#[serde(default)]` so a
    /// saved profile from before this field existed still deserializes to
    /// `false`, preserving prior behavior/output exactly.
    #[serde(default)]
    pub z_hop_enabled: bool,
    /// Height (mm) to lift above the current print Z before a travel move,
    /// and to lower back down by on arrival, when `z_hop_enabled` is
    /// `true`. Ignored when `z_hop_enabled` is `false`. Defaults to `0.4`
    /// (one common nozzle diameter's worth), though the value is inert
    /// while disabled. `#[serde(default = "default_z_hop_height")]` so a
    /// saved profile from before this field existed still deserializes.
    #[serde(default = "default_z_hop_height")]
    pub z_hop_height: f64,
    /// Whether the toolpath-simplification pass (`toolpath::simplify_paths`)
    /// is enabled. When `true` (the default), wall-loop paths
    /// (`toolpath::MoveKind::WallOuter`/`WallInner`) are run through an
    /// RDP/Douglas-Peucker-style perpendicular-distance decimation after
    /// planning, reducing point count from dense curved-order-field contour
    /// extraction (e.g. `OrderFieldKind::Eikonal`) without materially
    /// changing printed geometry. Infill paths are never touched by this
    /// pass regardless of this setting. `#[serde(default = "...")]` so a
    /// saved profile from before this field existed still deserializes to
    /// `true`.
    #[serde(default = "default_path_simplify_enabled")]
    pub path_simplify_enabled: bool,
    /// Perpendicular-distance tolerance (mm) for the toolpath-simplification
    /// pass: a point is dropped only if the resulting simplified polyline
    /// still passes within this distance of it. Ignored while
    /// `path_simplify_enabled` is `false`. Defaults to `nozzle_diameter /
    /// 20.0` (e.g. `0.02` mm for the default `0.4` mm nozzle) -- a
    /// fraction small enough to be well under typical dimensional-accuracy
    /// requirements (a twentieth of the nozzle diameter) while still
    /// collapsing the near-collinear "staircase" point runs that curved
    /// order fields produce. Because this default depends on
    /// `nozzle_diameter`'s value rather than being a fixed constant, it
    /// cannot be expressed as a `#[serde(default = "fn")]` static default
    /// (that attribute only sees this field in isolation); the static
    /// fallback below assumes the default `nozzle_diameter` of `0.4`, while
    /// [`SlicerConfig::default`] derives the value directly from whatever
    /// `nozzle_diameter` it constructs.
    #[serde(default = "default_path_simplify_tolerance")]
    pub path_simplify_tolerance: f64,
    /// Diameter (mm) of the flat land around the nozzle tip's orifice --
    /// the physical flat face that can contact an already-printed sloped
    /// surface when the nozzle axis is tilted for non-planar printing
    /// (see `toolpath::compensate_flat_nozzle`). `None` defers to twice
    /// `nozzle_diameter` (see [`SlicerConfig::nozzle_flat_diameter`]) --
    /// this can't be a `#[serde(default = "fn")]` static default since it
    /// depends on another field's value, the same reason
    /// `path_simplify_tolerance` documents its own static fallback above.
    /// `#[serde(default)]` so a saved profile from before this field
    /// existed still deserializes to `None` (i.e. the derived default),
    /// not a hardcoded literal that could silently diverge from
    /// `nozzle_diameter`.
    #[serde(default)]
    pub nozzle_flat_diameter: Option<f64>,
    /// Whether the travel-order optimization pass
    /// (`toolpath::optimize_travel_order`) is enabled. When `true` (the
    /// default), the paths within each layer (walls, sparse infill, solid
    /// fill) are greedily reordered -- and individually reversed where
    /// beneficial -- by nearest-endpoint distance from wherever the
    /// nozzle currently is, starting from the end of the previous layer's
    /// last path. Reduces long travel moves (e.g. a scanline infill
    /// pattern jumping across the whole layer and back) without changing
    /// which geometry is printed, only the order/direction paths are
    /// visited in. `#[serde(default = "...")]` so a saved profile from
    /// before this field existed still deserializes to `true`.
    #[serde(default = "default_travel_order_optimization_enabled")]
    pub travel_order_optimization_enabled: bool,
    /// Whether travel-move collision avoidance
    /// (`toolpath::route_travel_moves`) is enabled. When `true` (the
    /// default), any inter-path travel move whose straight-line chord
    /// would cross solid material (checked against `Layer::mesh_sdf`) is
    /// replaced with a routed path found via a bounded local grid search,
    /// gated by feasibility against [`Machine::slope_profile`]
    /// (`crate::machine::Machine`) so a routed move never implies a
    /// steeper local climb than the machine can physically clear.
    /// `#[serde(default = "...")]` so a saved profile from before this
    /// field existed still deserializes to `true`.
    #[serde(default = "default_travel_collision_avoidance_enabled")]
    pub travel_collision_avoidance_enabled: bool,
    /// Cost multiplier applied to the vertical (Z) component of a routed
    /// travel step in `toolpath::route_travel_moves`'s local grid search,
    /// relative to a purely horizontal step of the same length. Larger
    /// values bias the search toward horizontal detours over vertical
    /// ones (lift-then-move-then-drop-shaped routes), while still
    /// permitting a genuinely necessary 3D diagonal route when it is
    /// cheaper than any horizontal-plus-vertical alternative. Defaults to
    /// `8.0`. `#[serde(default = "...")]` so a saved profile from before
    /// this field existed still deserializes to `8.0`.
    #[serde(default = "default_z_travel_penalty")]
    pub z_travel_penalty: f64,
    /// Whether wave overhang path planning (Huygens-propagation support-free overhangs) is enabled.
    #[serde(default = "default_wave_overhangs_enabled")]
    pub wave_overhangs_enabled: bool,
    /// Overlap distance (mm) between adjacent wave tracks, defaulting to 0.05 mm.
    #[serde(default)]
    pub wave_overhang_overlap: Option<f64>,
    /// Wave overhang printing speed (mm/min), defaulting to 1500 mm/min (25 mm/s).
    #[serde(default)]
    pub wave_overhang_speed: Option<f64>,
    /// Flow multiplier for wave overhang teardrop beads, defaulting to 1.05.
    #[serde(default)]
    pub wave_overhang_flow: Option<f64>,
    /// Speed deadband percentage (e.g. 10.0%) for compacting G-code feedrate commands.
    #[serde(default)]
    pub speed_deadband_percent: Option<f64>,
    /// Acceleration deadband percentage (e.g. 20.0%) for compacting Klipper acceleration commands.
    #[serde(default)]
    pub acceleration_deadband_percent: Option<f64>,
    /// Klipper square corner velocity limit (mm/s), defaulting to 5.0 mm/s.
    #[serde(default)]
    pub square_corner_velocity: Option<f64>,
    /// Thermodynamic and non-Newtonian fluid dynamics configuration for dynamic pressure advance
    /// and adaptive retraction. When present, enables dynamic fluid state modeling.
    #[serde(default)]
    pub fluid_dynamics: Option<fluid_dynamics::FluidDynamicsConfig>,
}

/// Static serde-deserialize fallback for [`SlicerConfig::wave_overhangs_enabled`]: `true`.
fn default_wave_overhangs_enabled() -> bool {
    true
}

/// Static serde-deserialize fallback for [`SlicerConfig::scarf_joint_enabled`]: `true`.
fn default_scarf_joint_enabled() -> bool {
    true
}

/// Default value for [`SlicerConfig::travel_speed`]: `9000.0` mm/min
/// (150 mm/s), a common consumer-FDM travel speed.
fn default_travel_speed() -> f64 {
    9000.0
}

/// Default value for [`SlicerConfig::print_speed`]: `3000.0` mm/min
/// (50 mm/s), a common consumer-FDM print speed.
fn default_print_speed() -> f64 {
    3000.0
}

/// Default value for [`SlicerConfig::z_hop_height`]: `0.4` mm, a common
/// slicer convention (one nozzle diameter). Inert while `z_hop_enabled` is
/// `false` (the default).
fn default_z_hop_height() -> f64 {
    0.4
}

/// Static serde-deserialize fallback for
/// [`SlicerConfig::path_simplify_enabled`]: `true`.
fn default_path_simplify_enabled() -> bool {
    true
}

/// Static serde-deserialize fallback for
/// [`SlicerConfig::travel_order_optimization_enabled`]: `true`.
fn default_travel_order_optimization_enabled() -> bool {
    true
}

/// Static serde-deserialize fallback for
/// [`SlicerConfig::travel_collision_avoidance_enabled`]: `true`.
fn default_travel_collision_avoidance_enabled() -> bool {
    true
}

/// Static serde-deserialize fallback for [`SlicerConfig::z_travel_penalty`]:
/// `8.0`.
fn default_z_travel_penalty() -> f64 {
    8.0
}

/// Static serde-deserialize fallback for
/// [`SlicerConfig::path_simplify_tolerance`]: `0.02` mm, i.e.
/// `nozzle_diameter / 20.0` evaluated at the default `nozzle_diameter` of
/// `0.4` mm. Only used when deserializing a saved profile that predates
/// this field; [`SlicerConfig::default`] instead derives the value from
/// whatever `nozzle_diameter` it actually constructs.
fn default_path_simplify_tolerance() -> f64 {
    0.4 / 20.0
}

impl Default for SlicerConfig {
    fn default() -> Self {
        let nozzle_diameter = 0.4;
        let wall_line_width = nozzle_diameter;
        Self {
            layer_height: 0.2,
            first_layer_height: None,
            first_layer_print_speed: None,
            first_layer_extrusion_multiplier: None,
            first_layer_line_width: None,
            fan_speed_percent: None,
            overhang_fan_speed_percent: None,
            fan_layer_delay: None,
            nozzle_diameter,
            object_ordering: ordering::ObjectOrderingKind::default(),
            wall_line_width,
            shell_thickness: wall_line_width,
            wall_offset: nozzle_diameter / 2.0,
            sparse_infill_pattern: Some(infill::InfillPatternKind::Cubic),
            solid_infill_pattern: Some(infill::InfillPatternKind::AllWalls),
            infill_pattern: infill::InfillPatternKind::Cubic,
            infill_line_width: nozzle_diameter,
            infill_angle_deg: 45.0,
            infill_density: 0.2,
            top_layers: 3,
            bottom_layers: 3,
            order_field: order_field::OrderFieldKind::default(),
            eikonal_conform_top_surfaces: false,
            order_field_apex: glam::DVec3::ZERO,
            order_field_axis: slicing::BUILD_DIRECTION,
            order_field_slope: 0.0,
            filament_density_g_cm3: None,
            filament_diameter: 1.75,
            start_gcode: "PRINT_START T_TOOL=240 T_BED=105 T_CHAMBER=45 PRINT_MIN={print_min_x},{print_min_y} PRINT_MAX={print_max_x},{print_max_y}".to_string(),
            end_gcode: "PRINT_END".to_string(),
            travel_speed: default_travel_speed(),
            print_speed: default_print_speed(),
            outer_wall_speed: None,
            inner_wall_speed: None,
            infill_speed: None,
            solid_infill_speed: None,
            bridge_speed: None,
            default_acceleration: None,
            outer_wall_acceleration: None,
            inner_wall_acceleration: None,
            infill_acceleration: None,
            travel_acceleration: None,
            first_layer_acceleration: None,
            max_volumetric_speed: None,
            pressure_advance: None,
            pre_retract_taper_distance: None,
            retraction_length: None,
            retraction_speed: None,
            unretract_speed: None,
            unretract_extra_length: None,
            wipe_distance: None,
            wipe_enabled: false,
            use_firmware_retraction: false,
            scarf_joint_enabled: true,
            scarf_joint_length: None,
            z_hop_enabled: false,
            z_hop_height: default_z_hop_height(),
            path_simplify_enabled: true,
            path_simplify_tolerance: nozzle_diameter / 20.0,
            nozzle_flat_diameter: None,
            travel_order_optimization_enabled: true,
            travel_collision_avoidance_enabled: true,
            z_travel_penalty: default_z_travel_penalty(),
            wave_overhangs_enabled: true,
            wave_overhang_overlap: None,
            wave_overhang_speed: None,
            wave_overhang_flow: None,
            speed_deadband_percent: None,
            acceleration_deadband_percent: None,
            square_corner_velocity: None,
            fluid_dynamics: None,
        }
    }
}

impl SlicerConfig {
    /// Infill pattern for sparse interior regions, defaulting to
    /// [`infill::InfillPatternKind::Cubic`] when not set.
    #[must_use]
    pub fn sparse_infill_pattern(&self) -> infill::InfillPatternKind {
        self.sparse_infill_pattern.unwrap_or(
            if self.infill_pattern != infill::InfillPatternKind::default() {
                self.infill_pattern
            } else {
                infill::InfillPatternKind::Cubic
            },
        )
    }

    /// Infill pattern for solid top/bottom layers, defaulting to
    /// [`infill::InfillPatternKind::AllWalls`] when not set.
    #[must_use]
    pub fn solid_infill_pattern(&self) -> infill::InfillPatternKind {
        self.solid_infill_pattern
            .unwrap_or(infill::InfillPatternKind::AllWalls)
    }

    /// Number of wall/perimeter passes derived from `shell_thickness` /
    /// `wall_line_width`, rounded to the nearest whole wall and clamped to
    /// a minimum of one (a `shell_thickness` smaller than `wall_line_width`
    /// still gets a single outer wall, never zero).
    #[must_use]
    pub fn wall_count(&self) -> usize {
        let line_width = self.wall_line_width.abs().max(f64::EPSILON);
        (self.shell_thickness / line_width).round().max(1.0) as usize
    }

    /// First layer height (mm), defaulting to [`SlicerConfig::layer_height`]
    /// when [`SlicerConfig::first_layer_height`] (the field) is `None`.
    #[must_use]
    pub fn first_layer_height(&self) -> f64 {
        self.first_layer_height
            .map(|h| h.abs().max(f64::EPSILON))
            .unwrap_or_else(|| self.layer_height.abs().max(f64::EPSILON))
    }

    /// First layer print speed (mm/min), defaulting to 40% of `print_speed`
    /// when `first_layer_print_speed` is `None`.
    #[must_use]
    pub fn first_layer_print_speed(&self) -> f64 {
        self.first_layer_print_speed
            .unwrap_or_else(|| (self.print_speed * 0.4).min(self.print_speed))
    }

    /// Returns the resolved [`kinematics::StandardMotionModel`] reflecting
    /// the configured speeds and accelerations.
    #[must_use]
    pub fn motion_model(&self) -> kinematics::StandardMotionModel {
        kinematics::StandardMotionModel {
            outer_wall_speed: self
                .outer_wall_speed
                .unwrap_or_else(|| (self.print_speed * 0.6).min(self.print_speed)),
            inner_wall_speed: self.inner_wall_speed.unwrap_or(self.print_speed),
            infill_speed: self.infill_speed.unwrap_or(self.print_speed),
            solid_infill_speed: self
                .solid_infill_speed
                .unwrap_or_else(|| (self.print_speed * 0.8).min(self.print_speed)),
            bridge_speed: self
                .bridge_speed
                .unwrap_or_else(|| (self.print_speed * 0.5).min(self.print_speed)),
            travel_speed: self.travel_speed,
            first_layer_speed: self.first_layer_print_speed(),
            default_acceleration: self.default_acceleration.unwrap_or(5000.0),
            outer_wall_acceleration: self.outer_wall_acceleration.unwrap_or(2500.0),
            inner_wall_acceleration: self.inner_wall_acceleration.unwrap_or(5000.0),
            infill_acceleration: self.infill_acceleration.unwrap_or(7000.0),
            travel_acceleration: self.travel_acceleration.unwrap_or(10000.0),
            first_layer_acceleration: self.first_layer_acceleration.unwrap_or(2000.0),
        }
    }

    /// Returns a boxed dynamic [`kinematics::MotionModel`], selecting [`kinematics::StepperDynamicModel`]
    /// when `use_stepper_dynamics` is enabled on `machine`, or [`kinematics::StandardMotionModel`] otherwise.
    #[must_use]
    pub fn resolved_motion_model(
        &self,
        machine: Option<&crate::machine::Machine>,
    ) -> Box<dyn kinematics::MotionModel> {
        let std_model = self.motion_model();
        if let Some(m) = machine {
            if m.use_stepper_dynamics {
                return Box::new(kinematics::StepperDynamicModel::new(
                    std_model,
                    m.zero_speed_acceleration(),
                    m.max_available_speed(),
                    m.acceleration_limit(),
                    m.speed_limit(),
                ));
            }
        }
        Box::new(std_model)
    }

    /// First layer extrusion multiplier, defaulting to `1.0` when `None`.
    #[must_use]
    pub fn first_layer_extrusion_multiplier(&self) -> f64 {
        self.first_layer_extrusion_multiplier.unwrap_or(1.0)
    }

    /// Retraction distance (mm), defaulting to `0.8` mm when `None`.
    #[must_use]
    pub fn retraction_length(&self) -> f64 {
        self.retraction_length.unwrap_or(0.8)
    }

    /// Retraction speed (mm/min), defaulting to `3000.0` mm/min (50 mm/s) when `None`.
    #[must_use]
    pub fn retraction_speed(&self) -> f64 {
        self.retraction_speed.unwrap_or(3000.0)
    }

    /// Unretract speed (mm/min), defaulting to [`SlicerConfig::retraction_speed`] when `None`.
    #[must_use]
    pub fn unretract_speed(&self) -> f64 {
        self.unretract_speed
            .unwrap_or_else(|| self.retraction_speed())
    }

    /// Extra unretract length (mm), defaulting to `0.0` mm when `None`.
    #[must_use]
    pub fn unretract_extra_length(&self) -> f64 {
        self.unretract_extra_length.unwrap_or(0.0)
    }

    /// Wipe distance (mm), defaulting to `1.0` mm when `None`.
    #[must_use]
    pub fn wipe_distance(&self) -> f64 {
        self.wipe_distance.unwrap_or(1.0)
    }

    /// Filament material density in g/cm³ (defaulting to `1.24` when `None`).
    #[must_use]
    pub fn filament_density(&self) -> f64 {
        self.filament_density_g_cm3.unwrap_or(1.24)
    }

    /// Scarf joint overlap length (mm), defaulting to `3.0` mm when `None`.
    #[must_use]
    pub fn scarf_joint_length(&self) -> f64 {
        self.scarf_joint_length.unwrap_or(3.0)
    }

    /// First layer line width (mm), defaulting to `1.3 * nozzle_diameter`
    /// (or `wall_line_width`, whichever is larger) when `None`.
    #[must_use]
    pub fn first_layer_line_width(&self) -> f64 {
        self.first_layer_line_width
            .unwrap_or_else(|| (self.nozzle_diameter * 1.3).max(self.wall_line_width))
    }

    /// Part cooling fan speed percentage (0.0 to 100.0), defaulting to `100.0` when `None`.
    #[must_use]
    pub fn fan_speed_percent(&self) -> f64 {
        self.fan_speed_percent.unwrap_or(100.0).clamp(0.0, 100.0)
    }

    /// Part cooling fan speed percentage (0.0 to 100.0) for overhang moves, defaulting to `100.0` when `None`.
    #[must_use]
    pub fn overhang_fan_speed_percent(&self) -> f64 {
        self.overhang_fan_speed_percent
            .unwrap_or(100.0)
            .clamp(0.0, 100.0)
    }

    /// Number of initial layers to keep part cooling fan disabled, defaulting to `1` when `None`.
    #[must_use]
    pub fn fan_layer_delay(&self) -> u32 {
        self.fan_layer_delay.unwrap_or(1)
    }

    /// Diameter (mm) of the nozzle tip's flat land, used by
    /// `toolpath::compensate_flat_nozzle`. Defaults to twice
    /// `nozzle_diameter` when [`SlicerConfig::nozzle_flat_diameter`] (the
    /// field) is `None`.
    #[must_use]
    pub fn nozzle_flat_diameter(&self) -> f64 {
        self.nozzle_flat_diameter
            .unwrap_or(2.0 * self.nozzle_diameter)
    }

    /// Returns whether wave overhang toolpaths are enabled.
    #[must_use]
    pub fn wave_overhangs_enabled(&self) -> bool {
        self.wave_overhangs_enabled
    }

    /// Returns the lateral overlap distance (mm) for wave overhangs, defaulting to 0.05 mm.
    #[must_use]
    pub fn wave_overhang_overlap(&self) -> f64 {
        self.wave_overhang_overlap.unwrap_or(0.05)
    }

    /// Returns the printing speed (mm/min) for wave overhang moves, defaulting to 1500 mm/min (25 mm/s).
    #[must_use]
    pub fn wave_overhang_speed(&self) -> f64 {
        self.wave_overhang_speed.unwrap_or(1500.0)
    }

    /// Returns the flow multiplier for wave overhang teardrop beads, defaulting to 1.05.
    #[must_use]
    pub fn wave_overhang_flow(&self) -> f64 {
        self.wave_overhang_flow.unwrap_or(1.05)
    }

    /// Returns the speed deadband percentage for G-code feedrate compaction, defaulting to 10.0%.
    #[must_use]
    pub fn speed_deadband_percent(&self) -> f64 {
        self.speed_deadband_percent
            .unwrap_or(10.0)
            .clamp(0.0, 100.0)
    }

    /// Returns the acceleration deadband percentage for acceleration command compaction, defaulting to 20.0%.
    #[must_use]
    pub fn acceleration_deadband_percent(&self) -> f64 {
        self.acceleration_deadband_percent
            .unwrap_or(20.0)
            .clamp(0.0, 100.0)
    }

    /// Returns the Klipper square corner velocity (mm/s), defaulting to 5.0 mm/s.
    #[must_use]
    pub fn square_corner_velocity(&self) -> f64 {
        self.square_corner_velocity.unwrap_or(5.0).max(0.1)
    }

    /// Returns whether the dynamic thermodynamic and non-Newtonian fluid dynamics model is enabled.
    #[must_use]
    pub fn use_fluid_dynamics(&self) -> bool {
        self.fluid_dynamics.is_some()
    }

    /// Returns the resolved fluid dynamics engine, if configured.
    #[must_use]
    pub fn fluid_dynamics_engine(&self) -> Option<fluid_dynamics::FluidDynamicsEngine> {
        self.fluid_dynamics
            .map(fluid_dynamics::FluidDynamicsEngine::new)
    }
}

/// Run the full pipeline: workspace -> order objects -> slice -> plan
/// toolpaths -> emit Gcode.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the slicing/toolpath stages produce.
pub fn slice_to_gcode(workspace: &Workspace) -> Result<String> {
    let paths = plan_toolpaths(workspace)?;
    Ok(gcode::emit_with_machine(
        &paths,
        &workspace.config,
        Some(&workspace.machine),
    ))
}

/// Same as [`slice_to_gcode`], but calls `on_progress` with a `0.0..=1.0`
/// progress fraction across slicing and toolpath planning.
pub fn slice_to_gcode_with_progress(
    workspace: &Workspace,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<String> {
    let paths = plan_toolpaths_with_progress(workspace, on_progress)?;
    Ok(gcode::emit_with_machine(
        &paths,
        &workspace.config,
        Some(&workspace.machine),
    ))
}

/// Run the pipeline up to (and including) toolpath planning, stopping short
/// of Gcode emission: workspace -> order objects -> slice -> plan toolpaths.
///
/// Exposes the intermediate `Vec<toolpath::Path>` for consumers (e.g. the
/// GUI's 3D preview) that need the planned geometry without re-parsing
/// emitted Gcode text. [`slice_to_gcode`] builds on top of this and emits
/// Gcode from the same planned paths.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects, or
/// whatever error the ordering/slicing/toolpath stages produce.
pub fn plan_toolpaths(workspace: &Workspace) -> Result<Vec<toolpath::Path>> {
    plan_toolpaths_with_progress(workspace, &mut |_| {})
}

/// Same as [`plan_toolpaths`], but calls `on_progress` with a `0.0..=1.0`
/// fraction of overall progress across both stages, so a caller running
/// this on a background thread (slicing and toolpath planning can both be
/// slow — e.g. `AllWallsInfill` on complex geometry) can show live progress
/// without needing to know anything about layers, order fields, or infill.
///
/// The `0.0..=1.0` range is split evenly: order-field domain slicing (see
/// [`slicing::slice_workspace_with_progress`]) reports into `0.0..=0.5`,
/// then toolpath planning (see [`toolpath::plan_with_progress`]) reports
/// into `0.5..=1.0`. This is a fixed 50/50 split, not a measured time
/// estimate — for infill patterns/geometry where one stage dominates, the
/// bar will move unevenly through each half, but it will keep moving
/// throughout the whole call rather than stalling at `1.0` while
/// toolpath planning (previously unreported) is still running.
///
/// # Errors
///
/// Returns [`Error::InvalidMesh`] if `workspace` has no objects,
/// [`Error::MoveOutOfBounds`] if any planned move lies outside
/// `workspace.machine.build_volume`, or whatever error the
/// ordering/slicing/toolpath stages produce.
pub fn plan_toolpaths_with_progress(
    workspace: &Workspace,
    on_progress: &mut (dyn FnMut(f64) + Send),
) -> Result<Vec<toolpath::Path>> {
    if workspace.objects.is_empty() {
        return Err(Error::InvalidMesh("workspace has no objects".to_string()));
    }

    let strategy = ordering::strategy_for(workspace.config.object_ordering);
    let order = strategy.order(&workspace.objects)?;

    let layers = slicing::slice_workspace_with_progress(
        &workspace.objects,
        &order,
        &workspace.config,
        &workspace.machine.slope_profile(),
        &mut |fraction: f64| on_progress(fraction * 0.5),
    )?;
    let paths = toolpath::plan_with_progress(
        &layers,
        &workspace.objects,
        &workspace.machine.tools,
        &workspace.config,
        Some(&workspace.machine),
        &workspace.machine.slope_profile(),
        &mut |fraction: f64| on_progress(0.5 + fraction * 0.5),
    )?;

    toolpath::validate_within_bounds(&paths, &workspace.machine.build_volume)?;

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sane() {
        let cfg = SlicerConfig::default();
        assert!(cfg.layer_height > 0.0);
        assert!(cfg.nozzle_diameter > 0.0);
        assert!(cfg.wall_line_width > 0.0);
        assert!(cfg.shell_thickness > 0.0);
        assert!(cfg.wall_offset > 0.0);
        assert_eq!(cfg.wall_count(), 1);
        assert!(cfg.infill_line_width > 0.0);
        assert!(cfg.infill_angle_deg > 0.0);
        assert!(cfg.filament_diameter > 0.0);
    }

    #[test]
    fn slice_to_gcode_rejects_empty_workspace() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let workspace = Workspace::new(Vec::new(), machine, SlicerConfig::default());

        let err = slice_to_gcode(&workspace).unwrap_err();
        assert!(matches!(err, Error::InvalidMesh(_)));
    }

    #[test]
    fn slice_to_gcode_slices_first_object() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let object = crate::object::Object::new(
            crate::ids::ObjectId(0),
            mesh::Mesh::default(),
            crate::ids::ToolId(0),
        );
        let workspace = Workspace::new(vec![object], machine, SlicerConfig::default());

        assert!(slice_to_gcode(&workspace).is_ok());
    }

    /// Unit cube spanning [0,1]^3 — same fixture pattern as
    /// `slicing.rs`'s `cube_mesh` (and `manifold-fidget`'s
    /// `mesh_sdf`/`contour` tests).
    fn cube_mesh() -> mesh::Mesh {
        let vertices = vec![
            glam::DVec3::new(0.0, 0.0, 0.0),
            glam::DVec3::new(1.0, 0.0, 0.0),
            glam::DVec3::new(1.0, 1.0, 0.0),
            glam::DVec3::new(0.0, 1.0, 0.0),
            glam::DVec3::new(0.0, 0.0, 1.0),
            glam::DVec3::new(1.0, 0.0, 1.0),
            glam::DVec3::new(1.0, 1.0, 1.0),
            glam::DVec3::new(0.0, 1.0, 1.0),
        ];
        let indices = vec![
            0, 2, 1, 0, 3, 2, // -Z
            4, 5, 6, 4, 6, 7, // +Z
            0, 1, 5, 0, 5, 4, // -Y
            3, 7, 6, 3, 6, 2, // +Y
            0, 4, 7, 0, 7, 3, // -X
            1, 2, 6, 1, 6, 5, // +X
        ];
        mesh::Mesh::new(vertices, indices)
    }

    /// End-to-end: slicing a real solid (unit cube) through
    /// `slice_to_gcode` must produce non-trivial Gcode with real
    /// extrusion moves — not just the placeholder header. This is the
    /// first test exercising the full pipeline (slicing -> toolpath ->
    /// gcode) with a mesh that actually has contour geometry, so it
    /// would meaningfully fail if the pipeline regressed to producing
    /// empty layers/paths again.
    #[test]
    fn slice_to_gcode_produces_real_extrusion_moves_for_a_solid_cube() {
        // Arrange: a unit cube, one tool, and a layer height that steps
        // evenly across the cube's Z extent [0, 1] (5 layers: 0.0, 0.25,
        // 0.5, 0.75, 1.0 — matching slicing.rs's equivalent fixture/test).
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Aabb {
                min: glam::DVec3::new(-10.0, -10.0, -10.0),
                max: glam::DVec3::new(10.0, 10.0, 10.0),
            },
            Vec::new(),
        );
        let object =
            crate::object::Object::new(crate::ids::ObjectId(0), cube_mesh(), crate::ids::ToolId(0));
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let workspace = Workspace::new(vec![object], machine, config);

        // Act.
        let gcode = slice_to_gcode(&workspace).unwrap();

        // Assert: exactly one tool-select line (single object/tool), and
        // at least 3 travel-to-loop-start moves ("G0 X") — one per
        // interior layer's single wall contour loop (the two exact-
        // boundary layers at Z=0 and Z=1 sample directly on the mesh
        // surface and produce no contour — see slicing.rs's
        // `slice_mesh_produces_nonempty_contour_loops_for_a_solid_cube`).
        // Infill (see `infill` module) appends further "G0 X" travel
        // moves between scan-line segments, so the count is a lower bound
        // rather than an exact match now that infill is generated by
        // default; a regressed/placeholder pipeline (empty layers or
        // empty paths) would still produce fewer than 3, so this remains
        // a meaningful assertion.
        assert_eq!(gcode.matches("T0\n").count(), 1);
        let path_starts = gcode.matches("G0 X").count();
        assert!(
            path_starts >= 3,
            "expected at least one travel move per interior contour loop (3 layers with geometry), got {path_starts}"
        );
        // Each contour loop has more than one vertex, so real extrusion
        // ("G1") moves must follow each path's initial travel move.
        let extrusion_moves = gcode.matches("G1 X").count();
        assert!(
            extrusion_moves > 0,
            "expected non-zero G1 extrusion moves, got gcode:\n{gcode}"
        );
    }

    #[test]
    fn slice_to_gcode_handles_multiple_objects_and_tools() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Sphere {
                center: glam::DVec3::ZERO,
                radius: 1.0,
            },
            Vec::new(),
        );
        let objects = vec![
            crate::object::Object::new(
                crate::ids::ObjectId(0),
                mesh::Mesh::default(),
                crate::ids::ToolId(0),
            ),
            crate::object::Object::new(
                crate::ids::ObjectId(1),
                mesh::Mesh::default(),
                crate::ids::ToolId(1),
            ),
        ];
        let workspace = Workspace::new(objects, machine, SlicerConfig::default());

        assert!(slice_to_gcode(&workspace).is_ok());
    }

    /// `plan_toolpaths` is the shared helper `slice_to_gcode` now builds on
    /// top of — verify it returns non-empty paths for a real solid, and
    /// that `slice_to_gcode`'s Gcode output is unaffected by routing
    /// through it (still produces the same real extrusion moves as before
    /// the refactor).
    #[test]
    fn plan_toolpaths_returns_paths_and_slice_to_gcode_output_is_unaffected() {
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Aabb {
                min: glam::DVec3::new(-10.0, -10.0, -10.0),
                max: glam::DVec3::new(10.0, 10.0, 10.0),
            },
            Vec::new(),
        );
        let object =
            crate::object::Object::new(crate::ids::ObjectId(0), cube_mesh(), crate::ids::ToolId(0));
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let workspace = Workspace::new(vec![object], machine, config);

        let paths = plan_toolpaths(&workspace).unwrap();
        assert!(!paths.is_empty(), "expected non-empty planned toolpaths");

        let gcode = slice_to_gcode(&workspace).unwrap();
        assert_eq!(gcode.matches("T0\n").count(), 1);
        assert!(gcode.matches("G0 X").count() >= 3);
        assert!(gcode.matches("G1 X").count() > 0);
    }

    /// A build volume too small to contain the sliced object must fail the
    /// pipeline with `Error::MoveOutOfBounds` (via `validate_within_bounds`)
    /// rather than silently emitting Gcode with moves the machine can't
    /// actually reach -- both `plan_toolpaths` and `slice_to_gcode` build
    /// on the same validated path, so both must reject it.
    #[test]
    fn plan_toolpaths_rejects_geometry_outside_the_build_volume() {
        // The cube spans Z in [0, 1]; a build volume capped at Z=0.5
        // guarantees some planned point (at minimum the top layer's wall
        // loop) lies outside it.
        let machine = crate::machine::Machine::new(
            crate::bounds::BoundingVolume::Aabb {
                min: glam::DVec3::new(-10.0, -10.0, -10.0),
                max: glam::DVec3::new(10.0, 10.0, 0.5),
            },
            Vec::new(),
        );
        let object =
            crate::object::Object::new(crate::ids::ObjectId(0), cube_mesh(), crate::ids::ToolId(0));
        let config = SlicerConfig {
            layer_height: 0.25,
            ..SlicerConfig::default()
        };
        let workspace = Workspace::new(vec![object], machine, config);

        let err = plan_toolpaths(&workspace).unwrap_err();
        assert!(matches!(err, Error::MoveOutOfBounds { .. }));

        let err = slice_to_gcode(&workspace).unwrap_err();
        assert!(matches!(err, Error::MoveOutOfBounds { .. }));
    }
}
