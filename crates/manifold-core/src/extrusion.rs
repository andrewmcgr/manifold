//! Extrusion math: converts a toolpath segment's deposited bead volume
//! into linear filament feed length (the Gcode `E` axis).
//!
//! Kept as pure functions over plain `f64`s (no `Segment`/`Path`
//! dependency) per `CODE_STYLE.md` so the geometry math is independently
//! testable; `toolpath::plan` is the only caller, wiring these together
//! per segment once a path's points/kind are known.

use crate::{toolpath::MoveKind, SlicerConfig};
use glam::DVec3;

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

/// Clamps nominal bead width down to whatever room is actually there before
/// computing the support-aware blended cross-section ([`blended_bead_cross_section_area`]):
///
/// - `channel_width`: the local channel width (see `polygon2d::channel_widths_3d`) --
///   `line_width` is clamped down to it when finite, so a bead squeezed into a
///   narrow feature isn't fed as if it had the full nominal width.
///
/// Only ever shrinks the bead (never widens it), and is a no-op (`f64::INFINITY` /
/// `>= nominal`) when nothing constrains that axis.
#[must_use]
pub fn clamped_bead_cross_section_area(
    line_width: f64,
    layer_height: f64,
    nozzle_diameter: f64,
    support_fraction: f64,
    bed_fraction: f64,
    channel_width: f64,
) -> f64 {
    let clamped_width = if channel_width.is_finite() {
        line_width.min(channel_width)
    } else {
        line_width
    };
    blended_bead_cross_section_area(
        clamped_width,
        layer_height,
        nozzle_diameter,
        support_fraction,
        bed_fraction,
    )
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
        MoveKind::WallOuter
        | MoveKind::WallInner
        | MoveKind::TopSurface
        | MoveKind::DebugExcluded => config.wall_line_width,
        MoveKind::Infill | MoveKind::Bridge => config.infill_line_width,
        MoveKind::Overhang => config.wall_line_width.min(config.nozzle_diameter),
        MoveKind::Travel | MoveKind::Wipe => 0.0,
    }
}

/// Computes the adaptive in-surface line width for a wall segment on a non-planar layer.
///
/// In 3D, adjacent wall passes are separated by nominal CAD-normal distance `nominal_width`.
/// Along the layer's 3D print surface, the distance between passes expands to:
///
/// $$\Delta s = \frac{\text{nominal\_width}}{\sin\theta}$$
///
/// where $\sin\theta = \|\hat{n}_{\text{CAD}} \times \hat{n}_{\text{order}}\|$ is the sine of the
/// contact angle between the CAD surface normal and the slicing order field normal.
/// The resulting width is clamped to `[min_width, max_width]`.
#[must_use]
pub fn adaptive_wall_line_width(
    nominal_width: f64,
    min_width: f64,
    max_width: f64,
    n_cad: DVec3,
    n_order: DVec3,
) -> f64 {
    let cross_len = n_cad.cross(n_order).length();
    // Guard against parallel or degenerate normals
    if cross_len <= 1e-4 || !cross_len.is_finite() {
        return max_width;
    }
    let target = nominal_width / cross_len;
    target.clamp(min_width, max_width)
}

/// Evaluates the local layer thickness and surface unit normal at point `p`
/// from the order field.
///
/// Returns `(local_thickness_mm, surface_normal)`:
/// - `local_thickness_mm`: the REAL physical distance, measured along the
///   local surface normal, between `p` and the previous layer's isosurface
///   -- the isosurface at `order(p) - nominal_layer_height`, matching
///   exactly how `slicing::slice_workspace_with_progress` defines
///   consecutive layers (stepping `order_value` by `layer_height` in raw
///   order units). Found by ray-marching from `p` along `-normal` and
///   bisecting to the order-value crossing, rather than by inverting the
///   field's instantaneous gradient magnitude at `p` alone.
///
///   The one-line gradient-inversion shortcut used previously
///   ($h_{\text{local}} = h_{\text{nominal}} / \|\nabla \phi\|$) is only
///   physically correct when the order field is arc-length calibrated --
///   i.e. one order unit equals one millimeter along the build direction,
///   true for [`manifold_fidget::order::HeightOrderField`] by construction
///   and true in the unconstrained (isotropic-speed) regions of an Eikonal
///   field. It is NOT true for `AnisotropicFsm`'s boundary-metric-blended
///   regions (see `order_field::fsm_field_for`'s `top_tangency`/`wall_ortho`
///   aspect ratios), whose entire purpose is to distort front-propagation
///   speed near walls for toolpath-sequencing quality -- not to preserve a
///   physical distance calibration. Inverting that distorted gradient
///   magnitude produced wildly wrong bead heights (and therefore wrong
///   extrusion volume) specifically in the near-wall region the feature
///   targets. Ray-marching to the actual order-value crossing is exact
///   regardless of the field's local calibration, since it measures the
///   real geometric gap the bead needs to fill rather than linearly
///   extrapolating from an instantaneous, possibly-uncalibrated slope.
///
///   Bounded within $[0.1 \times h_{\text{nominal}}, 3.0 \times
///   h_{\text{nominal}}]$ as a physical safety clamp against a
///   degenerate/non-monotonic field search (e.g. no crossing found within
///   the search bound), not as a correction-suppression mechanism -- wide
///   enough to capture the several-times-nominal real gaps a heavily
///   metric-distorted `AnisotropicFsm` region can produce.
/// - `surface_normal`: outward/upward unit normal of the layer isosurface,
///   from the field's local gradient DIRECTION at `p` -- unaffected by the
///   magnitude-calibration issue above (already relied on this way by
///   `corner_flow::calculate_corner_excess`).
#[must_use]
pub fn local_layer_geometry(
    field: &dyn manifold_fidget::order::OrderField,
    p: DVec3,
    nominal_layer_height: f64,
) -> (f64, DVec3) {
    let h_nom = nominal_layer_height.abs().max(1e-4);
    let Some(grad) = crate::order_field::numeric_gradient(field, p) else {
        return (h_nom, DVec3::Z);
    };
    let grad_len = grad.length();
    if grad_len <= 1e-4 || !grad_len.is_finite() {
        return (h_nom, DVec3::Z);
    }
    let normal = grad / grad_len;

    let order_p = field.order(p);
    if !order_p.is_finite() {
        // Direction is still trustworthy even if the scalar value at `p`
        // itself is degenerate; fall back to the gradient-inversion
        // estimate only for the magnitude.
        let h_local = (h_nom / grad_len).clamp(0.1 * h_nom, 3.0 * h_nom);
        return (h_local, normal);
    }
    let order_target = order_p - h_nom;

    let h_local = probe_previous_layer_distance(field, p, normal, order_p, order_target, h_nom)
        .unwrap_or_else(|| h_nom / grad_len)
        .clamp(0.1 * h_nom, 3.0 * h_nom);
    (h_local, normal)
}

/// Ray-marches from `p` along `-normal`, sampling `field.order()` at
/// geometrically-growing distances (each 1.6x the last, starting at
/// `0.05 * h_nom`, capped at `32.0 * h_nom`), to bracket the distance at
/// which the field crosses `order_target` -- then bisects within that
/// bracket to refine the crossing distance. The geometric growth covers a
/// wide dynamic range (a real gap anywhere from a fraction of `h_nom` up to
/// tens of times `h_nom`, matching how far a heavily metric-distorted
/// `AnisotropicFsm` region can push the true physical spacing) in a bounded
/// number of samples. Returns `None` if no sign change is found within the
/// search bound (a degenerate or non-monotonic field locally), leaving the
/// caller to fall back to the gradient-based estimate.
fn probe_previous_layer_distance(
    field: &dyn manifold_fidget::order::OrderField,
    p: DVec3,
    normal: DVec3,
    order_p: f64,
    order_target: f64,
    h_nom: f64,
) -> Option<f64> {
    const GROWTH: f64 = 1.6;
    let max_search = 32.0 * h_nom;

    let signed = |v: f64| (v - order_target).signum();
    let mut prev_d = 0.0;
    let mut prev_sign = signed(order_p);

    let mut bracket: Option<(f64, f64, f64)> = None;
    let mut d = 0.05 * h_nom;
    while d <= max_search {
        let val = field.order(p - normal * d);
        if !val.is_finite() {
            break;
        }
        let sign = signed(val);
        if sign == 0.0 {
            return Some(d);
        }
        if sign != prev_sign && prev_sign != 0.0 {
            bracket = Some((prev_d, d, prev_sign));
            break;
        }
        prev_d = d;
        prev_sign = sign;
        d *= GROWTH;
    }

    let (mut d0, mut d1, sign_at_d0) = bracket?;
    for _ in 0..25 {
        if (d1 - d0) < 1e-4 {
            break;
        }
        let dm = 0.5 * (d0 + d1);
        let vm = field.order(p - normal * dm);
        if !vm.is_finite() {
            break;
        }
        if signed(vm) == sign_at_d0 {
            d0 = dm;
        } else {
            d1 = dm;
        }
    }
    Some(0.5 * (d0 + d1))
}

/// Evaluates the surface inclination flow modulation factor for a flat horizontal nozzle tip.
///
/// When a flat nozzle tip moves over a surface with unit normal $\mathbf{n}_{\text{surface}}$,
/// the horizontal projected capacity scales as $|\mathbf{n}_{\text{surface}} \cdot \mathbf{e}_z| = \cos\theta$.
/// As the surface inclines from horizontal ($\theta = 0^\circ$) toward steep slopes ($\theta \to 90^\circ$),
/// the cross-sectional capacity under the flat nozzle land contracts by $\cos\theta$.
///
/// Clamped to $[0.15, 1.0]$ to prevent complete flow starvation on near-vertical walls.
#[must_use]
pub fn surface_inclination_flow_factor(normal: DVec3) -> f64 {
    normal.dot(DVec3::Z).abs().clamp(0.15, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_layer_geometry_flat_field_returns_nominal_height_and_z_normal() {
        let field = manifold_fidget::order::HeightOrderField::new(DVec3::Z);
        let p = DVec3::new(10.0, 20.0, 5.0);
        let (h_local, normal) = local_layer_geometry(&field, p, 0.2);

        assert!((h_local - 0.2).abs() < 1e-4);
        assert!((normal.x).abs() < 1e-4);
        assert!((normal.y).abs() < 1e-4);
        assert!((normal.z - 1.0).abs() < 1e-4);
        assert!((surface_inclination_flow_factor(normal) - 1.0).abs() < 1e-4);
    }

    /// An order field whose calibration is NOT arc-length/mm-equal
    /// everywhere -- the physical build-direction (Z) unit here is split
    /// into a "steep" branch (order advances 2.0 units/mm, at/above
    /// z=5.0) and a much "shallower" branch (order advances only 0.4
    /// units/mm, below z=5.0), continuous at the z=5.0 seam. This models
    /// exactly the failure mode `AnisotropicFsm`'s boundary-metric
    /// blending produces: the SAME order field has genuinely different
    /// mm-per-order-unit calibration in different regions.
    struct KinkedCalibrationField;
    impl manifold_fidget::order::OrderField for KinkedCalibrationField {
        fn order(&self, p: glam::DVec3) -> f64 {
            if p.z >= 5.0 {
                2.0 * (p.z - 5.0)
            } else {
                0.4 * (p.z - 5.0)
            }
        }
    }

    #[test]
    fn local_layer_geometry_recovers_true_physical_gap_across_a_calibration_kink() {
        let field = KinkedCalibrationField;
        // Just 0.02mm above the seam, still in the steep branch.
        let p = DVec3::new(0.0, 0.0, 5.02);
        let h_nom = 0.2;

        let (h_local, normal) = local_layer_geometry(&field, p, h_nom);

        // True answer, solved by hand: walking down from z=5.02 to z=5.0
        // (0.02mm) drops order by 2.0*0.02 = 0.04 (order goes 0.04 -> 0.0).
        // The remaining order to drop to reach order_target = 0.04 - 0.2 =
        // -0.16 is (0.0 - (-0.16)) = 0.16, at the shallow branch's 0.4
        // units/mm, requiring 0.16 / 0.4 = 0.4mm more. Total: 0.02 + 0.4 =
        // 0.42mm -- a real physical gap 2.1x the nominal 0.2mm layer height.
        assert!(
            (h_local - 0.42).abs() < 1e-3,
            "expected h_local close to the hand-solved 0.42mm, got {h_local}"
        );
        assert!((normal.z - 1.0).abs() < 1e-4, "normal={normal:?}");

        // The removed gradient-inversion formula, evaluated at p (in the
        // steep branch, gradient magnitude exactly 2.0 there), would have
        // returned h_nom / 2.0 = 0.1mm -- off from the true 0.42mm gap by
        // more than 4x. Confirm the fix's answer is nowhere near that
        // wrong estimate.
        let old_wrong_estimate = h_nom / 2.0;
        assert!(
            (h_local - old_wrong_estimate).abs() > 0.2,
            "fix must diverge sharply from the old formula's wrong estimate \
             {old_wrong_estimate}, got h_local={h_local}"
        );
    }

    /// A field that plateaus (goes flat, zero gradient) below z=5.0 --
    /// walking backward from a point just above the plateau never reaches
    /// `order_target` no matter how far the search probes, since the
    /// field asymptotically stops decreasing. Exercises the "no crossing
    /// found" fallback path.
    struct PlateauField;
    impl manifold_fidget::order::OrderField for PlateauField {
        fn order(&self, p: glam::DVec3) -> f64 {
            if p.z >= 5.0 {
                p.z - 5.0
            } else {
                0.0
            }
        }
    }

    #[test]
    fn local_layer_geometry_falls_back_gracefully_when_no_crossing_is_found() {
        let field = PlateauField;
        let p = DVec3::new(0.0, 0.0, 5.05);
        let h_nom = 0.2;

        let (h_local, normal) = local_layer_geometry(&field, p, h_nom);

        // Gradient at p (steep branch, slope 1.0) gives the pre-existing
        // fallback estimate h_nom / 1.0 = h_nom exactly, clamped into
        // range -- a safe, finite, sane result rather than a panic, NaN,
        // or runaway value from an unbounded search.
        assert!(h_local.is_finite());
        assert!((h_local - h_nom).abs() < 1e-6, "h_local={h_local}");
        assert!((normal.z - 1.0).abs() < 1e-4);
    }

    #[test]
    fn surface_inclination_flow_factor_scales_with_cosine_of_tilt_angle() {
        // 45 degree tilted normal
        let n_45 = DVec3::new(1.0, 0.0, 1.0).normalize();
        let factor_45 = surface_inclination_flow_factor(n_45);
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((factor_45 - expected).abs() < 1e-4);

        // Vertical wall normal: clamped to minimum 0.15
        let n_wall = DVec3::new(1.0, 0.0, 0.0);
        let factor_wall = surface_inclination_flow_factor(n_wall);
        assert!((factor_wall - 0.15).abs() < 1e-4);
    }

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
    fn clamped_bead_cross_section_area_clamps_width() {
        let nozzle_diameter = 0.4;
        let full =
            clamped_bead_cross_section_area(0.4, 0.2, nozzle_diameter, 1.0, 0.0, f64::INFINITY);
        let narrow = clamped_bead_cross_section_area(
            0.4,
            0.2,
            nozzle_diameter,
            1.0,
            0.0,
            0.25, // channel narrower than nominal width
        );
        assert!(narrow < full, "narrow channel should reduce area");
    }

    #[test]
    fn line_width_for_kind_travel_is_zero() {
        assert_eq!(
            line_width_for_kind(MoveKind::Travel, &SlicerConfig::default()),
            0.0
        );
    }

    #[test]
    fn adaptive_wall_line_width_at_90_degrees_returns_nominal() {
        let nominal = 0.45;
        let min = 0.28;
        let max = 0.64;
        let n_cad = DVec3::X;
        let n_order = DVec3::Z;
        let width = adaptive_wall_line_width(nominal, min, max, n_cad, n_order);
        assert!((width - nominal).abs() < 1e-6);
    }

    #[test]
    fn adaptive_wall_line_width_at_45_degrees_expands_width() {
        let nominal = 0.45;
        let min = 0.28;
        let max = 0.80; // allow expansion to inspect value
        let n_cad = (DVec3::X + DVec3::Z).normalize();
        let n_order = DVec3::Z;
        let width = adaptive_wall_line_width(nominal, min, max, n_cad, n_order);
        let expected = nominal * std::f64::consts::SQRT_2;
        assert!((width - expected).abs() < 1e-4);
    }

    #[test]
    fn adaptive_wall_line_width_at_shallow_angle_clamps_to_max() {
        let nominal = 0.45;
        let min = 0.28;
        let max = 0.64;
        let n_cad = (DVec3::X * 0.1 + DVec3::Z).normalize();
        let n_order = DVec3::Z;
        let width = adaptive_wall_line_width(nominal, min, max, n_cad, n_order);
        assert_eq!(width, max);
    }

    #[test]
    fn adaptive_wall_line_width_with_parallel_normals_clamps_to_max() {
        let nominal = 0.45;
        let min = 0.28;
        let max = 0.64;
        let n_cad = DVec3::Z;
        let n_order = DVec3::Z;
        let width = adaptive_wall_line_width(nominal, min, max, n_cad, n_order);
        assert_eq!(width, max);
    }
}
