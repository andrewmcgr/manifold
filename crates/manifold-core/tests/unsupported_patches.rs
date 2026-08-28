//! Regression tests for order-aware unsupported-bead patches
//! (`manifold_core::verification::find_unsupported_patches`).
//!
//! Motivated by a real defect: enabling
//! `eikonal_conform_bottom_surfaces` on a mesh with shallow underside
//! overhangs produced patches of beads extruded over material that the
//! conformal order field had scheduled to print *later* — solid
//! according to the mesh SDF, but air at deposition time. The detector
//! (and these tests) check support against the order field itself, not
//! just the SDF.

use glam::DVec3;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::mesh::Mesh;
use manifold_core::object::{self, Object};
use manifold_core::order_field::OrderFieldKind;
use manifold_core::verification::{find_unsupported_patches, UnsupportedPatchOptions};
use manifold_core::{slicing, toolpath, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;
use manifold_fidget::ScalarField;

/// Axis-aligned box `[min, max]` with outward-facing triangles.
fn box_mesh(min: DVec3, max: DVec3) -> Mesh {
    let v = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
    let vertices = vec![
        v(min.x, min.y, min.z), // 0
        v(max.x, min.y, min.z), // 1
        v(max.x, max.y, min.z), // 2
        v(min.x, max.y, min.z), // 3
        v(min.x, min.y, max.z), // 4
        v(max.x, min.y, max.z), // 5
        v(max.x, max.y, max.z), // 6
        v(min.x, max.y, max.z), // 7
    ];
    let indices = vec![
        0, 2, 1, 0, 3, 2, // -Z
        4, 5, 6, 4, 6, 7, // +Z
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
    ];
    Mesh::new(vertices, indices)
}

/// Inverted frustum: a small rectangular bed footprint flaring outward to
/// a larger top, so all four sides are shallow underside overhangs
/// (~15 degrees from horizontal with the dimensions used below) — the
/// geometry class that trips bottom-surface conformal order fields.
fn inverted_frustum_mesh(bottom_half: f64, top_half: f64, height: f64) -> Mesh {
    let b = bottom_half;
    let t = top_half;
    let v = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
    let vertices = vec![
        // bottom quad (z = 0)
        v(-b, -b, 0.0), // 0
        v(b, -b, 0.0),  // 1
        v(b, b, 0.0),   // 2
        v(-b, b, 0.0),  // 3
        // top quad (z = height)
        v(-t, -t, height), // 4
        v(t, -t, height),  // 5
        v(t, t, height),   // 6
        v(-t, t, height),  // 7
    ];
    let indices = vec![
        0, 2, 1, 0, 3, 2, // bottom (-Z outward)
        4, 5, 6, 4, 6, 7, // top (+Z outward)
        0, 1, 5, 0, 5, 4, // -Y side (outward-facing, tilted downward)
        3, 7, 6, 3, 6, 2, // +Y side
        0, 4, 7, 0, 7, 3, // -X side
        1, 2, 6, 1, 6, 5, // +X side
    ];
    Mesh::new(vertices, indices)
}

fn plan_and_find_patches(mesh: Mesh, config: &SlicerConfig) -> Vec<String> {
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);

    let slope_profile = SlopeProfile::new(Vec::new());
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        config,
        &slope_profile,
        &mut |_| {},
    )
    .expect("slicing succeeds");
    assert!(!layers.is_empty(), "expected a non-empty slice");

    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &[],
        config,
        None,
        &slope_profile,
        &mut |_| {},
    )
    .expect("toolpath planning succeeds");

    let options = UnsupportedPatchOptions::for_config(config);
    find_unsupported_patches(&paths, &layers, config, &options)
        .into_iter()
        .map(|p| {
            format!(
                "order {:.3}: {:.1} mm / {} segs (max seg {:.2} mm) at ({:.1}, {:.1}, {:.1}) kinds {:?}",
                p.order,
                p.total_length_mm,
                p.segment_count,
                p.max_segment_length_mm,
                p.centroid.x,
                p.centroid.y,
                p.centroid.z,
                p.length_by_kind
            )
        })
        .collect()
}

/// Soundness guard: a plain box sliced with the flat `Height` field has
/// every bead either on the bed or on the previous layer — the detector
/// must stay silent, or every other test here is meaningless.
#[test]
fn height_field_box_has_no_unsupported_patches() {
    let config = SlicerConfig {
        layer_height: 0.4,
        nozzle_diameter: 0.8,
        order_field: OrderFieldKind::Height,
        ..SlicerConfig::default()
    };
    let patches = plan_and_find_patches(
        box_mesh(DVec3::new(0.0, 0.0, 0.0), DVec3::new(12.0, 12.0, 4.0)),
        &config,
    );
    assert!(
        patches.is_empty(),
        "expected no unsupported patches on a plain box, found:\n{}",
        patches.join("\n")
    );
}

/// Regression: bottom-surface conformal Eikonal slicing of shallow
/// underside overhangs must not schedule beads over not-yet-printed
/// material. Reproduces the pug_v4_m_sop_85mm.stl "unsupported patches
/// around order 4.8" defect on a minimal inverted frustum whose four
/// flared sides are ~15 degree underside overhangs (well within the 30
/// degree default bottom detach angle, so they *are* conformed to).
#[test]
fn bottom_conformal_eikonal_frustum_has_no_unsupported_patches() {
    let config = SlicerConfig {
        layer_height: 0.4,
        nozzle_diameter: 0.8,
        order_field: OrderFieldKind::Eikonal,
        eikonal_conform_bottom_surfaces: true,
        ..SlicerConfig::default()
    };
    // Underside slope: atan(height / (top_half - bottom_half))
    //   = atan(3 / 11) ~= 15.3 degrees from horizontal.
    let patches = plan_and_find_patches(inverted_frustum_mesh(4.0, 15.0, 3.0), &config);
    assert!(
        patches.is_empty(),
        "expected no unsupported patches on bottom-conformal frustum, found:\n{}",
        patches.join("\n")
    );
}

/// Same frustum without bottom conforming: the plain Eikonal field must
/// already be clean here, pinning any failure of the test above on the
/// bottom-conformal path rather than on Eikonal slicing in general.
#[test]
fn plain_eikonal_frustum_has_no_unsupported_patches() {
    let config = SlicerConfig {
        layer_height: 0.4,
        nozzle_diameter: 0.8,
        order_field: OrderFieldKind::Eikonal,
        ..SlicerConfig::default()
    };
    let patches = plan_and_find_patches(inverted_frustum_mesh(4.0, 15.0, 3.0), &config);
    assert!(
        patches.is_empty(),
        "expected no unsupported patches on plain Eikonal frustum, found:\n{}",
        patches.join("\n")
    );
}

/// Diagnostic (not a regression gate): sample the bottom-conformal order
/// field along vertical columns inside the frustum solid and report any
/// order inversions (order not strictly increasing with Z), which would
/// pin the unsupported patches on the field rather than on toolpath
/// classification. Run with `--ignored --nocapture`.
#[test]
#[ignore = "diagnostic"]
fn diag_bottom_conformal_field_monotonicity() {
    let config = SlicerConfig {
        layer_height: 0.4,
        nozzle_diameter: 0.8,
        order_field: OrderFieldKind::Eikonal,
        eikonal_conform_bottom_surfaces: true,
        ..SlicerConfig::default()
    };
    let mesh = inverted_frustum_mesh(4.0, 15.0, 3.0);
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);

    let slope_profile = SlopeProfile::new(Vec::new());
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )
    .expect("slicing succeeds");
    let layer = &layers[0];
    let field = layer.order_field.as_ref();
    let sdf = layer.mesh_sdf.as_deref().expect("mesh sdf cached");

    // Frustum footprint is centered on (100, 100); top half-width 15.
    let mut inversions = 0usize;
    let mut samples = 0usize;
    for xi in 0..31 {
        for yi in 0..31 {
            let x = 100.0 - 15.0 + xi as f64;
            let y = 100.0 - 15.0 + yi as f64;
            let mut prev: Option<(f64, f64)> = None;
            let mut z = 0.05;
            while z < 3.0 {
                let p = DVec3::new(x, y, z);
                if sdf.sample(p).value <= -0.05 {
                    let o = field.order(p);
                    if o.is_finite() {
                        samples += 1;
                        if let Some((pz, po)) = prev {
                            if o < po - 1e-6 {
                                inversions += 1;
                                if inversions <= 20 {
                                    println!(
                                        "inversion at ({x:.1},{y:.1}): z {pz:.2} -> {z:.2}, order {po:.3} -> {o:.3}"
                                    );
                                }
                            }
                        }
                        prev = Some((z, o));
                    }
                }
                z += 0.1;
            }
        }
    }
    println!("checked {samples} in-solid samples, {inversions} vertical order inversions");
}

/// Diagnostic: reason breakdown for the bottom-conformal frustum's
/// unsupported beads — probe in air vs. probe in solid scheduled later.
/// Run with `--ignored --nocapture`.
#[test]
#[ignore = "diagnostic"]
fn diag_bottom_conformal_unsupported_reasons() {
    let config = SlicerConfig {
        layer_height: 0.4,
        nozzle_diameter: 0.8,
        order_field: OrderFieldKind::Eikonal,
        eikonal_conform_bottom_surfaces: true,
        ..SlicerConfig::default()
    };
    let mesh = inverted_frustum_mesh(4.0, 15.0, 3.0);
    let mut objects = vec![Object::new(ObjectId(0), mesh, ToolId(0))];
    let build_volume = manifold_core::bounds::BoundingVolume::Aabb {
        min: DVec3::ZERO,
        max: DVec3::new(200.0, 200.0, 200.0),
    };
    object::center_on_bed(&mut objects, &build_volume);
    let slope_profile = SlopeProfile::new(Vec::new());
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )
    .expect("slicing succeeds");
    let paths = toolpath::plan_with_progress(
        &layers,
        &objects,
        &[],
        &config,
        None,
        &slope_profile,
        &mut |_| {},
    )
    .expect("toolpath planning succeeds");

    let layer_height = config.layer_height;
    let bed_z = paths
        .iter()
        .flat_map(|p| p.points.iter())
        .map(|p| p.z)
        .fold(f64::INFINITY, f64::min);
    let sdf_tolerance = 0.5 * config.nozzle_diameter;
    let order_epsilon = 0.5 * layer_height;

    let mut in_air = 0usize;
    let mut solid_later = 0usize;
    let mut supported = 0usize;
    let mut printed = 0usize;
    for path in &paths {
        for (i, segment) in path.segments.iter().enumerate() {
            use manifold_core::toolpath::MoveKind;
            if !matches!(
                segment.kind,
                MoveKind::WallOuter | MoveKind::WallInner | MoveKind::Infill | MoveKind::TopSurface
            ) || segment.extrusion_length <= 0.0
            {
                continue;
            }
            let a = path.points[i];
            let b = path.points[(i + 1) % path.points.len()];
            let midpoint = (a + b) * 0.5;
            let layer = layers
                .iter()
                .min_by(|x, y| {
                    (x.order - segment.order)
                        .abs()
                        .total_cmp(&(y.order - segment.order).abs())
                })
                .unwrap();
            let field = layer.order_field.as_ref();
            let (gradient_dir, gradient_len) =
                match manifold_core::order_field::numeric_gradient(field, midpoint)
                    .filter(|g| g.length_squared() > 1e-12 && g.is_finite())
                {
                    Some(g) => (g / g.length(), g.length()),
                    None => (DVec3::Z, 1.0),
                };
            let step = (layer_height / gradient_len).clamp(layer_height, 4.0 * layer_height);
            let probe = midpoint - step * gradient_dir;
            if probe.z <= bed_z + 0.25 * layer_height {
                supported += 1;
                continue;
            }
            let sdf_val = layer
                .mesh_sdf
                .as_deref()
                .map_or(f64::INFINITY, |s| s.sample(probe).value);
            if sdf_val <= sdf_tolerance {
                let probe_order = field.order(probe);
                if probe_order.is_finite() && probe_order <= segment.order - order_epsilon {
                    supported += 1;
                } else {
                    solid_later += 1;
                    if printed < 12 {
                        printed += 1;
                        println!(
                            "SOLID-LATER mid ({:.1},{:.1},{:.2}) o {:.3} sf {:.2} | grad ({:.2},{:.2},{:.2}) len {:.2} step {:.2} | probe ({:.1},{:.1},{:.2}) sdf {:.2} probe_o {:.3}",
                            midpoint.x, midpoint.y, midpoint.z, segment.order,
                            segment.support_fraction,
                            gradient_dir.x, gradient_dir.y, gradient_dir.z, gradient_len, step,
                            probe.x, probe.y, probe.z, sdf_val, field.order(probe),
                        );
                    }
                }
            } else {
                in_air += 1;
                if printed < 24 && in_air <= 12 {
                    println!(
                        "IN-AIR      mid ({:.1},{:.1},{:.2}) o {:.3} sf {:.2} | grad ({:.2},{:.2},{:.2}) len {:.2} step {:.2} | probe ({:.1},{:.1},{:.2}) sdf {:.2}",
                        midpoint.x, midpoint.y, midpoint.z, segment.order,
                        segment.support_fraction,
                        gradient_dir.x, gradient_dir.y, gradient_dir.z, gradient_len, step,
                        probe.x, probe.y, probe.z, sdf_val,
                    );
                }
            }
        }
    }
    println!("supported {supported}, in-air {in_air}, solid-later {solid_later}");
}
