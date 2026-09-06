//! Extrusion math: converts a toolpath segment's deposited bead volume
//! into linear filament feed length (the Gcode `E` axis).
//!
//! Kept as pure functions over plain `f64`s (no `Segment`/`Path`
//! dependency) per `CODE_STYLE.md` so the geometry math is independently
//! testable; `toolpath::plan` is the only caller, wiring these together
//! per segment once a path's points/kind are known.

use glam::DVec3;
use manifold_fidget::ScalarField;

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

/// Measures the actually-achievable extrusion height (mm) at `p` after
/// checking for solid material intruding above this bead's nominal top
/// across the flat nozzle land's transverse footprint (perpendicular to
/// `travel_dir`, in the plane spanned by `travel_dir` and `build_dir`).
///
/// Replaces the old differential-normal concavity heuristic
/// (`concavity_compensated_bead_area`) with a direct measurement against
/// the real solid geometry: samples `land_radius`-wide across the land,
/// and at each transverse offset queries `mesh_sdf` at the bead's nominal
/// top (`p + build_dir * nominal_height`). A reading there that is both
/// shallowly negative (inside solid, but not deep bulk) *and* measurably
/// closer to the surface than the same lateral offset at the current layer
/// means a ceiling is genuinely converging/closing in from above -- the
/// achievable height is clamped down to how far below that intrusion the
/// land can actually reach. Returns `nominal_height` unclamped (a no-op)
/// when nothing intrudes, when the surface there is no closer than it is
/// at the current layer (e.g. an ordinary vertical, untapered wall, whose
/// cross-section doesn't change with Z), or when `mesh_sdf` is unavailable
/// (e.g. hand-built test layers) -- flat/convex terrain never gets clamped.
#[must_use]
pub fn z_land_clearance(
    p: DVec3,
    travel_dir: DVec3,
    build_dir: DVec3,
    nominal_height: f64,
    land_radius: f64,
    mesh_sdf: Option<&manifold_fidget::mesh_sdf::MeshSdf>,
) -> f64 {
    let Some(sdf) = mesh_sdf else {
        return nominal_height;
    };
    if land_radius <= 1e-6 || nominal_height <= 1e-6 {
        return nominal_height;
    }
    let Some(perp) = travel_dir.cross(build_dir).try_normalize() else {
        return nominal_height;
    };
    const SAMPLE_COUNT: usize = 7;
    let top = p + build_dir * nominal_height;
    let mut achievable_height = nominal_height;
    for i in 0..SAMPLE_COUNT {
        let t = (i as f64 / (SAMPLE_COUNT - 1) as f64).mul_add(2.0, -1.0); // [-1, 1]
        let lateral = perp * (t * land_radius);
        let probe = top + lateral;
        let distance = sdf.sample(probe).value;
        // A probe landing deep inside the solid bulk (far from any surface,
        // i.e. `distance <= -nominal_height`) means there is no nearby
        // intrusion above the land -- it's ordinary interior material, not
        // an overhang squeezing the land from above, so it must not affect
        // achievable_height at all. Only a shallow negative reading (a real
        // nearby surface within one nominal layer height) represents a
        // *candidate* land-clearance constraint.
        //
        // That candidate is only a genuine overhang/converging ceiling
        // closing in from above if the surface is measurably *closer* at
        // `top` than it is at the current layer's own point at the same
        // lateral offset (`p + lateral`) -- i.e. `distance` (at `top`) is
        // shallower than `distance_now` (at `p`) by more than a small
        // tolerance. A plain vertical (untapered) wall has an unchanging
        // cross-section as Z increases, so `distance` and `distance_now`
        // are the same value one nominal layer height apart on either side
        // of it -- both shallow negative (a bead's own centerline sits
        // `line_width / 2` inside its own solid, comparable in magnitude
        // to a typical layer height), which the old "just check `distance`
        // alone" logic couldn't distinguish from a real intrusion, and so
        // clamped achievable height on almost every wall segment in the
        // model. Comparing against the current layer's own reading at the
        // same lateral offset restores the intended meaning: the surface
        // has to actually be closing in, not just present, to constrain
        // this land.
        const CONVERGENCE_EPS: f64 = 1e-6;
        let distance_now = sdf.sample(p + lateral).value;
        if distance < 0.0 && distance > -nominal_height && distance > distance_now + CONVERGENCE_EPS
        {
            let clearance = (nominal_height + distance).max(0.0);
            achievable_height = achievable_height.min(clearance);
        }
    }
    achievable_height
}

/// Clamps nominal bead width/height down to whatever room is actually
/// there before computing the support-aware blended cross-section
/// ([`blended_bead_cross_section_area`]), replacing the old
/// curvature-radius and concavity heuristics with measured clamps:
///
/// - `xy_channel_width`: the local 2D channel width (see
///   `polygon2d::channel_widths`) -- `line_width` is clamped down to it
///   when finite, so a bead squeezed into a narrow feature isn't fed as
///   if it had the full nominal width.
/// - `z_achievable_height` ([`z_land_clearance`]): `layer_height` is
///   clamped down to it, so a land forced to float above nominal by a
///   real transverse obstruction isn't fed as if it fully compressed.
///
/// Both clamps only ever shrink the bead (never widen it), and are
/// no-ops (`f64::INFINITY` / `>= nominal`) when nothing constrains that
/// axis -- so this never needlessly cuts flow when room *is* available,
/// unlike the flat-percentage heuristics it replaces.
#[must_use]
pub fn clamped_bead_cross_section_area(
    line_width: f64,
    layer_height: f64,
    nozzle_diameter: f64,
    support_fraction: f64,
    bed_fraction: f64,
    xy_channel_width: f64,
    z_achievable_height: f64,
) -> f64 {
    let clamped_width = if xy_channel_width.is_finite() {
        line_width.min(xy_channel_width)
    } else {
        line_width
    };
    let clamped_height = layer_height.min(z_achievable_height);
    blended_bead_cross_section_area(
        clamped_width,
        clamped_height,
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

    /// Cube of side `size` spanning `[0,size]^3`, as a ready-to-use `MeshSdf`
    /// (parametrized version of the fixture pattern used by
    /// `slicing::tests::cube_mesh` / `toolpath::tests::cube_sdf_fixture`).
    fn cube_sdf_fixture_sized(size: f64) -> manifold_fidget::mesh_sdf::MeshSdf {
        use manifold_fidget::mesh_sdf::MeshSdf;
        let vertices = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(size, 0.0, 0.0),
            DVec3::new(size, size, 0.0),
            DVec3::new(0.0, size, 0.0),
            DVec3::new(0.0, 0.0, size),
            DVec3::new(size, 0.0, size),
            DVec3::new(size, size, size),
            DVec3::new(0.0, size, size),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        MeshSdf::new(vertices, faces)
    }

    /// Unit cube spanning [0,1]^3, as a ready-to-use `MeshSdf` (same fixture
    /// pattern as `slicing::tests::cube_mesh` / `toolpath::tests::cube_sdf_fixture`).
    fn cube_sdf_fixture() -> manifold_fidget::mesh_sdf::MeshSdf {
        cube_sdf_fixture_sized(1.0)
    }

    #[test]
    fn z_land_clearance_is_nominal_without_mesh_sdf() {
        let p = DVec3::new(0.5, 0.5, 0.5);
        let h = z_land_clearance(p, DVec3::X, DVec3::Z, 0.2, 0.2, None);
        assert_eq!(h, 0.2);
    }

    #[test]
    fn z_land_clearance_clamps_when_solid_intrudes_above_nominal_top() {
        // Unit cube spanning [0,1]^3: a bead whose nominal top sits just
        // barely inside solid material (the cube's actual top face is only
        // slightly above the nominal top) -- a genuine shallow intrusion,
        // as opposed to being buried deep in the bulk. Clamp should shrink
        // achievable height down slightly rather than reporting the full
        // nominal height as if the land were floating in open air.
        let sdf = cube_sdf_fixture();
        let p = DVec3::new(0.5, 0.5, 0.78);
        let nominal_height = 0.2;
        let land_radius = 0.1;
        let h = z_land_clearance(
            p,
            DVec3::X,
            DVec3::Z,
            nominal_height,
            land_radius,
            Some(&sdf),
        );
        assert!(h < nominal_height, "expected clamp, got {h}");
        assert!(h >= 0.0);
    }

    #[test]
    fn z_land_clearance_full_on_an_ordinary_untapered_vertical_wall() {
        // Regression test for the global-underextrusion bug: a bead sitting
        // right at the centerline near a *vertical* (untapered) face -- the
        // ordinary case for the overwhelming majority of wall segments in
        // any real print. The bead's own centerline is shallowly inside the
        // solid (bead half-width, e.g. ~0.05mm from the face here), which is
        // the same order of magnitude as a typical nominal_height -- exactly
        // the shallow-negative range this function otherwise treats as a
        // "candidate intrusion". Because the face is vertical, one nominal
        // layer height straight up lands at the *same* shallow distance from
        // the face, not a new, closer one -- this is the wall simply
        // continuing upward, not a ceiling closing in from above, and must
        // not clamp achievable height at all. Before the current-layer
        // baseline check was added, this exact scenario clamped >50% of all
        // wall segments across a real test mesh, crushing total extruded
        // volume to ~60% of nominal.
        let sdf = cube_sdf_fixture_sized(10.0);
        // Just inside the x=0 face, deep in Z away from the top/bottom faces
        // so only the vertical x=0 face is in play.
        let p = DVec3::new(0.05, 5.0, 5.0);
        let nominal_height = 0.2;
        let land_radius = 0.1;
        let h = z_land_clearance(
            p,
            DVec3::Y,
            DVec3::Z,
            nominal_height,
            land_radius,
            Some(&sdf),
        );
        assert_eq!(
            h, nominal_height,
            "an ordinary vertical wall must not self-clamp achievable height, got {h}"
        );
    }

    #[test]
    fn z_land_clearance_full_when_probe_is_deep_in_solid_bulk() {
        // A large cube (side 10) with the probe centered deep in the
        // interior: a full layer height above the sample point still lands
        // far from any face (>> nominal_height away in every direction).
        // This is ordinary interior solid, not an overhang squeezing the
        // land from above -- it must NOT collapse achievable_height to
        // near-zero, or every wall segment near the base of a solid part
        // would get its bead area crushed and dropped as unprintable.
        let sdf = cube_sdf_fixture_sized(10.0);
        let p = DVec3::new(5.0, 5.0, 5.0);
        let nominal_height = 0.2;
        let land_radius = 0.1;
        let h = z_land_clearance(
            p,
            DVec3::X,
            DVec3::Z,
            nominal_height,
            land_radius,
            Some(&sdf),
        );
        assert_eq!(
            h, nominal_height,
            "deep interior bulk must not clamp achievable height, got {h}"
        );
    }

    #[test]
    fn clamped_bead_cross_section_area_clamps_width_and_height() {
        let nozzle_diameter = 0.4;
        let full = clamped_bead_cross_section_area(
            0.4,
            0.2,
            nozzle_diameter,
            1.0,
            0.0,
            f64::INFINITY,
            0.2,
        );
        let narrow = clamped_bead_cross_section_area(
            0.4,
            0.2,
            nozzle_diameter,
            1.0,
            0.0,
            0.25, // channel narrower than nominal width
            0.2,
        );
        assert!(narrow < full, "narrow channel should reduce area");

        let shallow = clamped_bead_cross_section_area(
            0.4,
            0.2,
            nozzle_diameter,
            1.0,
            0.0,
            f64::INFINITY,
            0.1, // less Z room than nominal height
        );
        assert!(shallow < full, "reduced Z clearance should reduce area");
    }

    #[test]
    fn line_width_for_kind_travel_is_zero() {
        assert_eq!(
            line_width_for_kind(MoveKind::Travel, &SlicerConfig::default()),
            0.0
        );
    }
}
