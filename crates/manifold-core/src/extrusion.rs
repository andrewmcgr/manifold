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

/// Adjusts nominal bead cross-section area for in-plane path curvature $R = 1 / \kappa$ (mm)
/// around tight corners or small circular loops:
///
/// $$A_{\text{curved}} = A_{\text{nominal}} \cdot \max\left(0.70, \, 1.0 - \frac{w}{2R}\right)$$
///
/// Prevents inner-edge melt over-packing on small circular bosses and sharp turns.
#[must_use]
pub fn curvature_compensated_bead_area(
    nominal_area: f64,
    line_width: f64,
    radius_of_curvature: f64,
) -> f64 {
    if radius_of_curvature <= 1e-4 || !radius_of_curvature.is_finite() {
        return nominal_area;
    }
    let r = radius_of_curvature.abs();
    let ratio = (line_width / (2.0 * r)).clamp(0.0, 0.30);
    nominal_area * (1.0 - ratio)
}

/// Adjusts nominal bead cross-section area for transverse surface concavity / V-groove pinch:
///
/// $$\Phi_{\text{concave}} = \max\left(0.40, \, 1.0 - \frac{\Delta z_{\text{flanks}}}{h_{\text{layer}}} \cdot \sin \theta_{\text{transverse}}\right)$$
///
/// Prevents the flat nozzle land from plowing molten plastic and forcing forward overextrusion waves
/// at the bottom of V-grooves and concave troughs.
#[must_use]
pub fn concavity_compensated_bead_area(
    nominal_area: f64,
    layer_height: f64,
    flank_rise: f64,
    sin_transverse: f64,
) -> f64 {
    if flank_rise <= 1e-4 || sin_transverse <= 1e-4 {
        return nominal_area;
    }
    let h = layer_height.max(1e-3);
    let pinch_ratio = (flank_rise / h * sin_transverse).clamp(0.0, 0.60);
    nominal_area * (1.0 - pinch_ratio)
}

/// Evaluates in-plane radius of curvature (mm) from three consecutive 3D path points $(P_{i-1}, P_i, P_{i+1})$.
///
/// Uses Menger curvature (the circumradius of the triangle formed by the three points).
/// Returns `f64::INFINITY` for collinear or degenerate points.
#[must_use]
pub fn in_plane_radius_of_curvature(
    p_prev: glam::DVec3,
    p_curr: glam::DVec3,
    p_next: glam::DVec3,
) -> f64 {
    let a = (p_curr - p_prev).length();
    let b = (p_next - p_curr).length();
    let c = (p_next - p_prev).length();
    if a <= 1e-6 || b <= 1e-6 || c <= 1e-6 {
        return f64::INFINITY;
    }
    // Heron's formula for triangle area:
    let s = (a + b + c) * 0.5;
    let area_sq = s * (s - a) * (s - b) * (s - c);
    if area_sq <= 1e-12 {
        return f64::INFINITY;
    }
    let area = area_sq.sqrt();
    (a * b * c) / (4.0 * area)
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
    fn curvature_compensated_bead_area_reduces_volume_on_tight_curves() {
        let nominal_area = 0.08;
        let line_width = 0.4;
        let r_tight = 1.5; // R = 1.5mm -> ratio = 0.4 / 3.0 ≈ 0.1333
        let curved_area = curvature_compensated_bead_area(nominal_area, line_width, r_tight);
        let expected = nominal_area * (1.0 - (0.4 / 3.0));
        assert!((curved_area - expected).abs() < 1e-6);

        // Infinite radius / straight line -> no reduction
        let straight_area =
            curvature_compensated_bead_area(nominal_area, line_width, f64::INFINITY);
        assert_eq!(straight_area, nominal_area);
    }

    #[test]
    fn in_plane_radius_of_curvature_computes_circumradius_correctly() {
        use glam::DVec3;
        // 90-degree circular arc of radius 2.0 at origin:
        // p0 = (2.0, 0.0), p1 = (sqrt(2), sqrt(2)), p2 = (0.0, 2.0)
        let p0 = DVec3::new(2.0, 0.0, 0.0);
        let p1 = DVec3::new(2.0f64.sqrt(), 2.0f64.sqrt(), 0.0);
        let p2 = DVec3::new(0.0, 2.0, 0.0);
        let r = in_plane_radius_of_curvature(p0, p1, p2);
        assert!((r - 2.0).abs() < 1e-4);

        // Collinear line -> infinity
        let r_collinear = in_plane_radius_of_curvature(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
        );
        assert!(!r_collinear.is_finite());
    }

    #[test]
    fn concavity_compensated_bead_area_scales_down_at_v_grooves() {
        let nominal_area = 0.08;
        let layer_height = 0.20;
        let flank_rise = 0.10;
        let sin_transverse = 0.50;
        let area =
            concavity_compensated_bead_area(nominal_area, layer_height, flank_rise, sin_transverse);
        // pinch_ratio = (0.10 / 0.20) * 0.50 = 0.25 -> 75% flow
        let expected = nominal_area * 0.75;
        assert!((area - expected).abs() < 1e-6);

        // Zero flank rise / planar -> no reduction
        let flat_area =
            concavity_compensated_bead_area(nominal_area, layer_height, 0.0, sin_transverse);
        assert_eq!(flat_area, nominal_area);
    }

    #[test]
    fn line_width_for_kind_travel_is_zero() {
        assert_eq!(
            line_width_for_kind(MoveKind::Travel, &SlicerConfig::default()),
            0.0
        );
    }
}
