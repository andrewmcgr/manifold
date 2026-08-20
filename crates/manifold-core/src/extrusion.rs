//! Extrusion math: converts a toolpath segment's deposited bead volume
//! into linear filament feed length (the Gcode `E` axis).
//!
//! Kept as pure functions over plain `f64`s (no `Segment`/`Path`
//! dependency) per `CODE_STYLE.md` so the geometry math is independently
//! testable; `toolpath::plan` is the only caller, wiring these together
//! per segment once a path's points/kind are known.

use crate::{toolpath::MoveKind, SlicerConfig};

/// Cross-sectional area (mm^2) of a single deposited bead, modeled as the
/// standard "stadium" (rounded-rectangle) shape used by Slic3r/
/// PrusaSlicer: a `(width - height) x height` rectangle capped by a full
/// circle of diameter `height` at each end (the nozzle's circular profile
/// pressed flat into a layer of that height). `line_width` is clamped up
/// to at least `layer_height` first so a width narrower than the layer
/// height (a physically nonsensical bead) degenerates to a plain circular
/// bead instead of a negative area.
#[must_use]
pub fn bead_cross_section_area(line_width: f64, layer_height: f64) -> f64 {
    let height = layer_height.abs().max(f64::EPSILON);
    let width = line_width.abs().max(height);
    height * (width - height) + std::f64::consts::PI * (height / 2.0).powi(2)
}

/// Cross-sectional area (mm^2) of a bead squished against the flat rigid
/// build plate: a plain `width x height` rectangle with no rounded ends.
/// The plate (unlike previously deposited, still-soft material) does not
/// let the bead's ends curl under, so the full rectangular footprint is
/// filled -- this is the same first-layer model planar slicers use, and
/// under-feeding it with the stadium volume is a classic cause of
/// first-layer underextrusion (~12% for a 0.4x0.2 bead). Width is clamped
/// up to at least `layer_height` like [`bead_cross_section_area`].
#[must_use]
pub fn rectangular_bead_cross_section_area(line_width: f64, layer_height: f64) -> f64 {
    let height = layer_height.abs().max(f64::EPSILON);
    let width = line_width.abs().max(height);
    width * height
}

/// Cross-sectional area (mm^2) of a bead extruded into free air (no
/// supporting surface below at all -- a bridge/overhang): the filament
/// keeps the nozzle bore's circular profile (die swell aside) instead of
/// being squished into a stadium, so the full circle of `nozzle_diameter`
/// must be fed. Note this is *more* volume per mm than the stadium for
/// typical width/height ratios (0.1257 vs 0.0714 mm^2 at 0.4/0.2):
/// unsupported lines underextruded at stadium flow come out as thin,
/// saggy strands.
#[must_use]
pub fn circular_bead_cross_section_area(nozzle_diameter: f64) -> f64 {
    let radius = nozzle_diameter.abs().max(f64::EPSILON) / 2.0;
    std::f64::consts::PI * radius * radius
}

/// Support-aware bead cross-section area (mm^2): blends the three
/// physical bead shapes by how the segment is supported.
///
/// - `support_fraction` (0..=1): how much previously deposited material
///   sits directly under the bead (along the order-field's local "down").
///   `1.0` is the fully supported stadium ([`bead_cross_section_area`],
///   today's uniform model); `0.0` is free air
///   ([`circular_bead_cross_section_area`]); linear blend between.
/// - `bed_fraction` (0..=1): how much of the bead is squished directly
///   against the build plate. Takes precedence over the
///   stadium/circle blend ([`rectangular_bead_cross_section_area`] at
///   `1.0`), since the plate is beneath whatever the SDF probe said.
///
/// Both fractions are clamped to `[0, 1]` here so callers can pass raw
/// distance-derived ratios.
#[must_use]
pub fn blended_bead_cross_section_area(
    line_width: f64,
    layer_height: f64,
    nozzle_diameter: f64,
    support_fraction: f64,
    bed_fraction: f64,
) -> f64 {
    let stadium = bead_cross_section_area(line_width, layer_height);
    let circle = circular_bead_cross_section_area(nozzle_diameter);
    let rectangle = rectangular_bead_cross_section_area(line_width, layer_height);
    let support = support_fraction.clamp(0.0, 1.0);
    let bed = bed_fraction.clamp(0.0, 1.0);
    let airborne_blend = circle + (stadium - circle) * support;
    airborne_blend + (rectangle - airborne_blend) * bed
}

/// Cross-sectional area (mm^2) of the filament being fed, treated as a
/// perfect circle of `filament_diameter` mm (1.75mm by default -- see
/// [`SlicerConfig::filament_diameter`]).
#[must_use]
pub fn filament_cross_section_area(filament_diameter: f64) -> f64 {
    let radius = filament_diameter.abs().max(f64::EPSILON) / 2.0;
    std::f64::consts::PI * radius * radius
}

/// Linear filament feed length (mm) to extrude for one segment of
/// `distance` mm, conserving volume between the deposited bead
/// (`bead_area` mm^2 cross-section) and the filament pushed through
/// (`filament_area` mm^2 cross-section): `distance * bead_area ==
/// filament_length * filament_area`.
#[must_use]
pub fn segment_extrusion_length(distance: f64, bead_area: f64, filament_area: f64) -> f64 {
    let area = filament_area.abs().max(f64::EPSILON);
    distance * bead_area / area
}

/// Nozzle-center line width used for a segment of the given [`MoveKind`],
/// looked up from `config`. `WallOuter`/`WallInner` use
/// `config.wall_line_width`; `Infill` uses `config.infill_line_width`.
/// `Overhang` is emitted by `toolpath::plan` for stitched wall-gap points
/// and is clamped to `config.wall_line_width.min(config.nozzle_diameter)`:
/// an unsupported line must never be wider than the nozzle diameter,
/// since there's no supporting surface underneath for the extra
/// squish/spread a wider bead needs. `TopSurface` is a fully-supported
/// wall-0 point (see `MoveKind::TopSurface`'s docs), so it uses the
/// ordinary `wall_line_width` like `WallOuter`/`WallInner`. `Bridge`
/// remains a forward-compatible placeholder mapped to
/// `config.infill_line_width` (no detection logic currently emits it --
/// see `toolpath::plan`). `Travel` is never extruded and returns `0.0`.
#[must_use]
pub fn line_width_for_kind(kind: MoveKind, config: &SlicerConfig) -> f64 {
    match kind {
        MoveKind::WallOuter | MoveKind::WallInner | MoveKind::TopSurface => config.wall_line_width,
        MoveKind::Infill | MoveKind::Bridge => config.infill_line_width,
        MoveKind::Overhang => config.wall_line_width.min(config.nozzle_diameter),
        MoveKind::Travel => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bead_cross_section_area_matches_stadium_formula_for_a_wide_bead() {
        // width=0.4, height=0.2: rectangle 0.2*(0.4-0.2)=0.04, plus a
        // circle of diameter 0.2 (radius 0.1): pi*0.1^2 ~= 0.0314159.
        let area = bead_cross_section_area(0.4, 0.2);
        assert!((area - (0.04 + std::f64::consts::PI * 0.01)).abs() < 1e-9);
    }

    #[test]
    fn bead_cross_section_area_degenerates_to_a_circle_when_width_equals_height() {
        // Rectangle term vanishes (width - height == 0); area is exactly
        // the nozzle's circular cross-section.
        let area = bead_cross_section_area(0.2, 0.2);
        assert!((area - std::f64::consts::PI * 0.01).abs() < 1e-9);
    }

    #[test]
    fn bead_cross_section_area_clamps_width_narrower_than_height_to_a_circle() {
        // A width smaller than the layer height is physically nonsensical
        // (see doc comment); clamped to width == height, same result as
        // the exact-equal case above rather than a negative area.
        let narrow = bead_cross_section_area(0.05, 0.2);
        let equal = bead_cross_section_area(0.2, 0.2);
        assert!((narrow - equal).abs() < 1e-12);
    }

    #[test]
    fn filament_cross_section_area_matches_known_value_for_standard_175mm_filament() {
        let expected = std::f64::consts::PI * (1.75_f64 / 2.0).powi(2);
        let area = filament_cross_section_area(1.75);
        assert!((area - expected).abs() < 1e-12);
    }

    #[test]
    fn segment_extrusion_length_conserves_volume() {
        // distance * bead_area == filament_length * filament_area.
        let distance = 10.0;
        let bead_area = 0.1;
        let filament_area = filament_cross_section_area(1.75);
        let length = segment_extrusion_length(distance, bead_area, filament_area);
        assert!((length * filament_area - distance * bead_area).abs() < 1e-9);
    }

    #[test]
    fn segment_extrusion_length_is_zero_for_zero_bead_area() {
        assert_eq!(segment_extrusion_length(10.0, 0.0, 2.4), 0.0);
    }

    #[test]
    fn line_width_for_kind_maps_walls_to_wall_line_width() {
        let config = SlicerConfig {
            wall_line_width: 0.5,
            infill_line_width: 0.3,
            ..SlicerConfig::default()
        };
        assert_eq!(line_width_for_kind(MoveKind::WallOuter, &config), 0.5);
        assert_eq!(line_width_for_kind(MoveKind::WallInner, &config), 0.5);
    }

    #[test]
    fn line_width_for_kind_maps_infill_and_bridge_to_infill_line_width() {
        let config = SlicerConfig {
            wall_line_width: 0.5,
            infill_line_width: 0.3,
            ..SlicerConfig::default()
        };
        assert_eq!(line_width_for_kind(MoveKind::Infill, &config), 0.3);
        assert_eq!(line_width_for_kind(MoveKind::Bridge, &config), 0.3);
    }

    #[test]
    fn line_width_for_kind_clamps_overhang_to_nozzle_diameter_when_wall_is_wider() {
        let config = SlicerConfig {
            wall_line_width: 0.8,
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        assert_eq!(line_width_for_kind(MoveKind::Overhang, &config), 0.4);
    }

    #[test]
    fn line_width_for_kind_leaves_overhang_unclamped_when_wall_is_not_wider_than_nozzle() {
        let config = SlicerConfig {
            wall_line_width: 0.35,
            nozzle_diameter: 0.4,
            ..SlicerConfig::default()
        };
        assert_eq!(line_width_for_kind(MoveKind::Overhang, &config), 0.35);
    }

    #[test]
    fn line_width_for_kind_travel_is_zero() {
        assert_eq!(
            line_width_for_kind(MoveKind::Travel, &SlicerConfig::default()),
            0.0
        );
    }
}
