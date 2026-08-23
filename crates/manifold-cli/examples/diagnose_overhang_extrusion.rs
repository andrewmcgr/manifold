//! Scratch diagnostic for the "T-shaped overextruded overhang patches"
//! defect reported against the GUI preview: dense clusters of
//! `MoveKind::Overhang` segments (from `slicing::stitch_wall_gaps`'
//! `WallLoop::unsupported` tag) in narrow/thin regions of a non-planar
//! Eikonal print, suspected of both (a) over-depositing volume where
//! stitched rows are packed closer together than the raw nozzle diameter,
//! and (b) possibly being triggered by spurious wall-0 loop-matching
//! failures rather than a genuine topological gap in the mesh.
//!
//! Not part of the crate's public surface or test suite -- run manually:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example diagnose_overhang_extrusion -- pug_v4_l_sop_85mm.stl
//! ```
//!
//! Reproduces the reported GUI profile: Eikonal order field with the
//! slope-limit breakpoints shown in the screenshot (h:0mm -> max 15 deg,
//! h:4mm -> max 2 deg), 0.4mm nozzle/wall line width, 0.2mm layer height.
//!
//! For every maximal run of `unsupported == true` points in every wall-0
//! loop, reports:
//!  - the mesh SDF value at each stitched point (negative/near-zero =>
//!    genuinely inside solid material, i.e. a *false-positive* stitch
//!    trigger; clearly positive => a real gap with no material there);
//!  - the lateral hop distance to each immediate stitched neighbor (the
//!    real printed row spacing), compared against the raw
//!    `nozzle_diameter` used unconditionally by
//!    `circular_bead_cross_section_area`;
//!  - the actual `Segment::extrusion_length`-implied cross-section area
//!    for the corresponding `Overhang` segments in `toolpath::plan`'s
//!    output, compared against what a non-overlapping bead at the real
//!    row spacing would need.
use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::extrusion;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::slicing;
use manifold_core::tool::Tool;
use manifold_core::toolpath::{self, MoveKind};
use manifold_core::{stl, SlicerConfig};
use manifold_fidget::slope_profile::SlopeProfile;
use manifold_fidget::ScalarField;

/// Replicates `order_field::numeric_gradient`'s central-difference stencil
/// (GRAD_EPS = 1e-4) inline -- that function is `pub(crate)` in
/// manifold-core, unreachable from this example crate -- so we can see
/// whether `lateral_gap`'s climb-direction estimate is degenerate (falls
/// back to `BUILD_DIRECTION`) at the flagged wall-gap-stitch points.
fn numeric_gradient_probe(
    field: &dyn manifold_fidget::order::OrderField,
    p: DVec3,
) -> Option<DVec3> {
    const GRAD_EPS: f64 = 1e-4;
    let sample = |offset: DVec3| -> Option<f64> {
        let plus = field.order(p + offset);
        let minus = field.order(p - offset);
        if plus.is_finite() && minus.is_finite() {
            Some((plus - minus) / (2.0 * GRAD_EPS))
        } else {
            None
        }
    };
    Some(DVec3::new(
        sample(DVec3::X * GRAD_EPS)?,
        sample(DVec3::Y * GRAD_EPS)?,
        sample(DVec3::Z * GRAD_EPS)?,
    ))
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let nozzle_diameter = 0.4;
    let nozzle_radius = nozzle_diameter / 2.0;

    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        nozzle_diameter,
        wall_line_width: 0.4,
        layer_height: 0.2,
        path_simplify_enabled: false,
        ..SlicerConfig::default()
    };

    // Reproduces the screenshot's slope-limit profile: near-vertical
    // (15 deg) allowed right at the build plate, clamping down to
    // nearly flat (2 deg) by 4mm up.
    let slope_profile = if std::env::args().any(|a| a == "--no-slope-limit") {
        SlopeProfile::new(Vec::new())
    } else {
        SlopeProfile::from_angle(15.0)
    };

    println!("slicing {path} with Eikonal order field (slope profile: 0mm->15deg, 4mm->2deg)...");
    let layers = slicing::slice_workspace_with_progress(
        &objects,
        &[ObjectId(0)],
        &config,
        &slope_profile,
        &mut |_| {},
    )?;
    println!("sliced {} layer(s)", layers.len());

    let mut total_runs = 0usize;
    let mut false_positive_runs = 0usize; // every stitched point's SDF is comfortably inside solid
    let mut genuine_gap_runs = 0usize; // every stitched point's SDF is comfortably outside solid
    let mut mixed_runs = 0usize;
    let mut min_row_spacing = f64::INFINITY;
    let mut worst_overlap_ratio = 0.0f64; // nozzle_diameter / actual row spacing, capped display

    for (li, layer) in layers.iter().enumerate() {
        let Some(sdf) = layer.mesh_sdf.as_ref() else {
            continue;
        };
        for wall in layer.loops.iter().filter(|w| w.wall_index == 0) {
            let n = wall.points.len();
            if n == 0 {
                continue;
            }
            let mut i = 0usize;
            let mut visited = 0usize;
            while visited < n {
                if wall.unsupported[i] {
                    let run_start = i;
                    let mut run_len = 0usize;
                    while visited < n && wall.unsupported[(run_start + run_len) % n] {
                        run_len += 1;
                        visited += 1;
                    }
                    total_runs += 1;

                    let indices: Vec<usize> = (0..run_len).map(|k| (run_start + k) % n).collect();
                    let sdf_values: Vec<f64> = indices
                        .iter()
                        .map(|&idx| sdf.sample(wall.points[idx]).value)
                        .collect();

                    let all_inside = sdf_values.iter().all(|&v| v < -1e-3);
                    let all_outside = sdf_values.iter().all(|&v| v > 0.25 * nozzle_radius);
                    if all_inside {
                        false_positive_runs += 1;
                    } else if all_outside {
                        genuine_gap_runs += 1;
                    } else {
                        mixed_runs += 1;
                    }

                    // Real printed row spacing: lateral hop between
                    // consecutive stitched points in this run (the same
                    // metric stitch_wall_gaps itself uses to decide
                    // whether stitching is needed).
                    for w in indices.windows(2) {
                        let spacing = slicing::lateral_gap(
                            layer.order_field.as_ref(),
                            wall.points[w[0]],
                            wall.points[w[1]],
                        );
                        if spacing > 1e-6 && spacing < min_row_spacing {
                            min_row_spacing = spacing;
                        }
                        if spacing > 1e-6 {
                            let ratio = nozzle_diameter / spacing;
                            if ratio > worst_overlap_ratio {
                                worst_overlap_ratio = ratio;
                            }
                        }
                    }

                    if total_runs <= 5 {
                        let sdf_str: Vec<String> =
                            sdf_values.iter().map(|v| format!("{v:.4}")).collect();
                        println!(
                            "run #{total_runs}: layer={} layer.order={:.5} run_len={run_len} sdf_values=[{}] classification={}",
                            layer.index,
                            layer.order,
                            sdf_str.join(", "),
                            if all_inside {
                                "FALSE-POSITIVE (inside solid)"
                            } else if all_outside {
                                "genuine gap (outside solid)"
                            } else {
                                "mixed"
                            }
                        );

                        for &idx in indices.iter().take(8) {
                            let p = wall.points[idx];
                            let g = numeric_gradient_probe(layer.order_field.as_ref(), p);
                            match g {
                                Some(g) => println!(
                                    "  point[{idx}]={p:?} order={:.5} grad={g:?} |grad|={:.6}",
                                    layer.order_field.order(p),
                                    g.length()
                                ),
                                None => println!(
                                    "  point[{idx}]={p:?} order={:.5} grad=None (non-finite in stencil)",
                                    layer.order_field.order(p)
                                ),
                            }
                        }
                        // Full-run order-value trace: does every point in this
                        // single wall-0 loop's contiguous run actually sit on
                        // the SAME isosurface (order ~= layer.order), or does
                        // the run straddle two different order values (i.e.
                        // points that belong to a different layer's contour
                        // entirely)?
                        let order_trace: Vec<String> = indices
                            .iter()
                            .map(|&idx| format!("{:.3}", layer.order_field.order(wall.points[idx])))
                            .collect();
                        println!(
                            "  full order-value trace for this run: [{}]",
                            order_trace.join(", ")
                        );

                        // Topology + local-surface-normal check: is this run
                        // bounded by REAL current-layer wall points (order ~=
                        // layer.order, unsupported == false) that plausibly
                        // correspond to real previous-layer wall points
                        // nearby -- and is the true mesh surface (via the SDF
                        // gradient, the actual local normal, not the order
                        // field's build-direction fallback) locally
                        // near-horizontal there (which would legitimately
                        // cause large lateral drift between adjacent
                        // order-levels, independent of any correspondence
                        // bug)?
                        let pred_idx = (run_start + n - 1) % n;
                        let succ_idx = (run_start + run_len) % n;
                        let pred = wall.points[pred_idx];
                        let succ = wall.points[succ_idx];
                        let pred_normal = sdf.sample(pred).gradient;
                        let succ_normal = sdf.sample(succ).gradient;
                        println!(
                            "  run bounded by real points: pred={pred:?} (order={:.3}, unsupported={}, surface_normal={pred_normal:?}, normal.dot(Z)={:.3})",
                            layer.order_field.order(pred),
                            wall.unsupported[pred_idx],
                            pred_normal.z
                        );
                        println!(
                            "  run bounded by real points: succ={succ:?} (order={:.3}, unsupported={}, surface_normal={succ_normal:?}, normal.dot(Z)={:.3})",
                            layer.order_field.order(succ),
                            wall.unsupported[succ_idx],
                            succ_normal.z
                        );
                        if li > 0 {
                            let prev_layer = &layers[li - 1];
                            let cur_loop_count =
                                layer.loops.iter().filter(|w| w.wall_index == 0).count();
                            let prev_loop_count = prev_layer
                                .loops
                                .iter()
                                .filter(|w| w.wall_index == 0)
                                .count();
                            println!(
                                "  wall-0 loop count: current layer={cur_loop_count}, previous layer={prev_loop_count} (topology change if these differ)"
                            );
                            // Nearest REAL (unsupported==false) previous-layer
                            // wall-0 point to each boundary point, by
                            // lateral_gap -- this is exactly the metric
                            // stitch_wall_gaps' veto uses.
                            let mut best_pred = f64::INFINITY;
                            let mut best_succ = f64::INFINITY;
                            for prev_wall in prev_layer.loops.iter().filter(|w| w.wall_index == 0) {
                                for (&pp, &sup) in
                                    prev_wall.points.iter().zip(prev_wall.unsupported.iter())
                                {
                                    if sup {
                                        continue;
                                    }
                                    let dp =
                                        slicing::lateral_gap(layer.order_field.as_ref(), pp, pred);
                                    let ds =
                                        slicing::lateral_gap(layer.order_field.as_ref(), pp, succ);
                                    if dp < best_pred {
                                        best_pred = dp;
                                    }
                                    if ds < best_succ {
                                        best_succ = ds;
                                    }
                                }
                            }
                            let hop_limit = 0.9 * (nozzle_radius);
                            println!(
                                "  nearest REAL previous-layer wall-0 point: pred->{best_pred:.4}mm succ->{best_succ:.4}mm (hop_limit={hop_limit:.4}mm)"
                            );

                            // Root-cause probe: sample the Eikonal order
                            // field AND mesh SDF along a straight-line probe
                            // from the nearest real previous-layer point to
                            // `pred`, to distinguish two hypotheses:
                            //  (a) genuine geodesic routing -- the field
                            //      shows a plateau/near-INFINITY "moat" along
                            //      the straight path (an obstacle the front
                            //      had to detour around), meaning the large
                            //      lateral gap is real/expected Eikonal
                            //      behavior, not a solve bug; or
                            //  (b) a corrupted/inconsistent solve -- the
                            //      field is finite and reasonably smooth
                            //      along the straight path, meaning there is
                            //      no physical obstacle explaining the jump
                            //      from order~0.2 to order~0.4 over such a
                            //      short lateral distance, pointing at a
                            //      genuine bug in the grid solve near the
                            //      kink/valley feature.
                            if let Some(nearest_prev_point) = prev_layer
                                .loops
                                .iter()
                                .filter(|w| w.wall_index == 0)
                                .flat_map(|w| w.points.iter().zip(w.unsupported.iter()))
                                .filter(|(_, sup)| !**sup)
                                .map(|(p, _)| *p)
                                .min_by(|a, b| {
                                    slicing::lateral_gap(layer.order_field.as_ref(), *a, pred)
                                        .total_cmp(&slicing::lateral_gap(
                                            layer.order_field.as_ref(),
                                            *b,
                                            pred,
                                        ))
                                })
                            {
                                const PROBE_STEPS: usize = 10;
                                let order_probe: Vec<String> = (0..=PROBE_STEPS)
                                    .map(|k| {
                                        let t = k as f64 / PROBE_STEPS as f64;
                                        let p = nearest_prev_point.lerp(pred, t);
                                        let o = layer.order_field.order(p);
                                        if o.is_finite() {
                                            format!("{o:.3}")
                                        } else {
                                            "INF".to_string()
                                        }
                                    })
                                    .collect();
                                let sdf_probe: Vec<String> = (0..=PROBE_STEPS)
                                    .map(|k| {
                                        let t = k as f64 / PROBE_STEPS as f64;
                                        let p = nearest_prev_point.lerp(pred, t);
                                        format!("{:.3}", sdf.sample(p).value)
                                    })
                                    .collect();
                                println!(
                                    "  straight-line probe prev-real-point[{nearest_prev_point:?}] -> pred[{pred:?}]:"
                                );
                                println!("    order values : [{}]", order_probe.join(", "));
                                println!("    mesh sdf     : [{}]", sdf_probe.join(", "));
                            }
                        }
                    }
                    i = (run_start + run_len) % n;
                } else {
                    i = (i + 1) % n;
                    visited += 1;
                }
            }
        }
    }

    println!();
    println!("=== Wall-gap-stitch run classification (by mesh SDF at stitched points) ===");
    println!("total stitched runs: {total_runs}");
    println!("  false-positive (all points already inside solid material): {false_positive_runs}");
    println!("  genuine gap (all points outside solid material):           {genuine_gap_runs}");
    println!("  mixed (straddles the surface):                             {mixed_runs}");
    println!();
    println!("=== Overhang bead-area/spacing check ===");
    println!("nozzle_diameter={nozzle_diameter}");
    println!("closest observed lateral row spacing within any stitch run: {min_row_spacing:.4} mm");
    println!(
        "circular_bead_cross_section_area(nozzle_diameter) = {:.4} mm^2 (used for every fully-unsupported Overhang segment regardless of actual spacing)",
        extrusion::circular_bead_cross_section_area(nozzle_diameter)
    );
    println!(
        "worst implied overlap ratio (nozzle_diameter / actual row spacing): {worst_overlap_ratio:.2}x{}",
        if worst_overlap_ratio > 1.0 {
            " -- adjacent beads would need to overlap/merge to deposit this much material in this little space"
        } else {
            ""
        }
    );

    // --- Cross-check against toolpath::plan's actual Overhang segments ---
    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;
    let mut overhang_segments = 0usize;
    let mut total_overhang_extrusion = 0.0f64;
    let mut total_overhang_length = 0.0f64;
    for p in &paths {
        for (idx, seg) in p.segments.iter().enumerate() {
            if seg.kind != MoveKind::Overhang {
                continue;
            }
            overhang_segments += 1;
            let a = p.points[idx];
            let b = p.points[(idx + 1) % p.points.len()];
            let dist = a.distance(b);
            total_overhang_length += dist;
            total_overhang_extrusion += seg.extrusion_length;
        }
    }
    println!();
    println!("=== toolpath::plan Overhang segments ===");
    println!("Overhang-tagged segments: {overhang_segments}");
    println!("total Overhang geometric length: {total_overhang_length:.2} mm");
    println!("total Overhang filament extrusion length: {total_overhang_extrusion:.4} mm");

    println!();
    if total_runs == 0 {
        println!("RESULT: INCONCLUSIVE -- no stitched runs reproduced with this profile.");
    } else if false_positive_runs > 0 {
        println!(
            "RESULT: {false_positive_runs}/{total_runs} stitched run(s) sit entirely inside already-solid material -- \
             these are FALSE-POSITIVE wall-gap detections, not real overhangs. The wall-gap correspondence logic \
             is misfiring in this region rather than reflecting genuine topology."
        );
    } else {
        println!(
            "RESULT: All {total_runs} stitched run(s) sit outside solid material at every sampled point -- \
             these are genuine gaps in the wall-0 contour, not false positives."
        );
    }
    if worst_overlap_ratio > 1.0 {
        println!(
            "RESULT: Overhang bead area uses the raw nozzle_diameter unconditionally, but stitched rows can be \
             packed as tight as {min_row_spacing:.4} mm apart ({worst_overlap_ratio:.2}x tighter than nozzle_diameter) \
             -- this over-doses filament in these regions."
        );
    }

    Ok(())
}
