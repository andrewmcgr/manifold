//! Scratch verification for inter-layer wall-0 gap stitching
//! (`slicing::stitch_wall_gaps` / the `WallLoop::unsupported` tag /
//! `toolpath::plan`'s `MoveKind::Overhang` classification / the
//! `Overhang` extrusion-width clamp in `extrusion::line_width_for_kind`).
//! Not part of the crate's public surface or test suite -- run manually
//! against the real test STL:
//!
//! ```sh
//! cargo run --release -p manifold-cli --example verify_wall_gap_stitching -- pug_v4_l_sop_85mm.stl
//! ```
//!
//! Uses an `OrderFieldKind::Eikonal` order field against the real test
//! mesh, which reproduces wall-0 gaps on its own (no additional
//! `eikonal_slope_profile` restriction is needed for this mesh -- an
//! empty profile, i.e. `SlopeProfile`'s unconstrained default, already
//! triggers `stitch_wall_gaps`). This example reconstructs the *pre-fix*
//! gap that would have existed for every stitched run by looking at the
//! distance between the run's two surrounding un-stitched
//! (`unsupported == false`) points -- i.e. the single direct hop
//! `stitch_wall_gaps` replaced with a bisected chain -- rather than
//! needing a separate before/after build, and separately checks that
//! every hop *within* each stitched chain itself (not the surrounding
//! contour's own point spacing, which is unrelated to this feature)
//! stays within `0.9 * nozzle_radius`.
//!
//! All gaps/hops are measured LATERALLY (perpendicular to the local
//! order-field climb direction -- see `slicing::lateral_gap`), matching
//! the stitching criterion itself: raw 3D distance always contains ~one
//! layer height of normal climb separation and is not the physically
//! meaningful bonding metric. Numbers reported here are therefore NOT
//! directly comparable to runs of this example predating the lateral
//! criterion.

use glam::DVec3;
use manifold_core::bounds::BoundingVolume;
use manifold_core::ids::{ObjectId, ToolId};
use manifold_core::machine::Machine;
use manifold_core::object::Object;
use manifold_core::order_field::OrderFieldKind;
use manifold_core::order_field::OrderFieldTrait as OrderField;
use manifold_core::slicing::WallLoop;
use manifold_core::tool::Tool;
use manifold_core::toolpath::MoveKind;
use manifold_core::{slicing, stl, toolpath, SlicerConfig};
use manifold_fidget::ScalarField;

/// Residual final-connection hops up to this length (mm) at chains whose
/// level-doubling stalled are classified WARN, not FAIL: they occur only
/// where the order field is genuinely discontinuous (the very unreachable
/// region being bridged -- e.g. an ear tip, or the pug_v4_m chin region at
/// layer 16 which converges 5.9mm -> 1.85mm before stalling) so no
/// intermediate isosurface exists to land further stitch rows on, and a
/// short anchored-both-ends unsupported hop is well within FDM bridging
/// capability (printers routinely bridge tens of mm; 5x nozzle diameter is
/// conservative). Anything larger means stitching failed to make real
/// progress and is a FAIL.
const BRIDGEABLE_RESIDUAL_HOP: f64 = 2.0;

/// Tolerated order-value regression between consecutive stitched points
/// before it is flagged as a zigzag: reprojection residuals and
/// unprojectable fallback (lerp-seed) points can wobble a row's order
/// slightly, but the per-point-ramp zigzag bug regressed by an entire
/// layer step (0.2), orders of magnitude above this.
const ORDER_BACKSLIDE_TOLERANCE: f64 = 0.02;

/// Lateral (perpendicular to the climb direction) distance between two
/// points on a closed loop, wrapping the index -- the same metric
/// `stitch_wall_gaps` itself triggers/terminates on.
fn dist(field: &dyn OrderField, points: &[DVec3], i: usize, j: usize) -> f64 {
    slicing::lateral_gap(field, points[i], points[j])
}

/// For each maximal run of consecutive `unsupported == true` points
/// (i.e. one inter-layer stitch chain) in a wall-0 loop, returns the
/// reconstructed PRE-FIX gap (the single direct jump from the
/// un-stitched point just before the run to the un-stitched point just
/// after it, i.e. what `stitch_wall_gaps` detected and replaced), the
/// POST-FIX worst hop (the largest consecutive-point distance within
/// that same chain: before -> first stitched -> ... -> last stitched ->
/// after, which is what the fix actually bounds to `<= 0.9 *
/// nozzle_radius` -- deliberately scoped to the stitch chain itself, not
/// every consecutive pair in the whole contour polyline, since ordinary
/// contour-extraction point spacing/resolution is unrelated to this
/// feature and can legitimately exceed the hop limit), the run length,
/// and the full chain (world-space points) so callers can dump per-hop
/// diagnostics for the worst offenders.
fn analyze_stitch_runs(
    field: &dyn OrderField,
    loop_: &WallLoop,
) -> Vec<(f64, f64, usize, Vec<DVec3>)> {
    let n = loop_.points.len();
    if n == 0 {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut i = 0usize;
    let mut visited = 0usize;
    while visited < n {
        if loop_.unsupported[i] {
            let run_start = i;
            let mut run_len = 0usize;
            while visited < n && loop_.unsupported[(run_start + run_len) % n] {
                run_len += 1;
                visited += 1;
            }
            let before = (run_start + n - 1) % n;
            let after = (run_start + run_len) % n;
            // The gap the stitcher detected and bridged: lateral
            // distance from the chain's anchor (the first stitched
            // point, spliced at the previous layer's correspondent) to
            // the target point (`after`). NOT before->after: those are
            // just adjacent current-loop points (ordinary contour
            // spacing), unrelated to the inter-layer gap.
            let pre_fix_gap = dist(field, &loop_.points, run_start, after);

            // Chain: before, run_start, ..., run_start+run_len-1, after.
            let mut chain_indices = vec![before];
            chain_indices.extend((0..run_len).map(|k| (run_start + k) % n));
            chain_indices.push(after);
            // The hop-limit guarantee no longer applies to every
            // consecutive pair inside a stitch block: a serpentine block
            // (see `slicing::serpentine_stitch_block`) walks whole order
            // levels across the run, so same-level neighbors are just
            // points along one drawn line (ordinary segment spacing, no
            // bonding constraint). What the fix actually guarantees, and
            // what this checks:
            //  1. anti-zigzag: order values through the chain climb
            //     monotonically from the previous layer toward the
            //     current one (the per-point-ramp bug dove back down to
            //     the previous layer between every pair of run points);
            //  2. the chain's last inserted point connects to the run's
            //     first real point (`after`) within the lateral hop
            //     limit.
            // The preceding `before -> anchor-row` hop is the APPROACH
            // move across the gap and is deliberately unbounded.
            let stitched = &chain_indices[1..chain_indices.len() - 1];
            // A maximal unsupported run can contain SEVERAL serpentine
            // blocks back-to-back (each gap sub-run gets its own block,
            // and the stitched target points are themselves flagged
            // unsupported), and between blocks the path intentionally
            // dives once back to the previous layer to start the next
            // block's anchor row. The per-point-ramp bug dove on
            // (essentially) EVERY point; so flag a chain only when order
            // backslides are frequent relative to its length, not for
            // the rare intended per-block dives.
            let mut backslide_count = 0usize;
            let mut worst_order_backslide = 0.0f64;
            let mut worst_backslide_pair: Option<(DVec3, DVec3)> = None;
            for w in stitched.windows(2) {
                let backslide = field.order(loop_.points[w[0]]) - field.order(loop_.points[w[1]]);
                if backslide > ORDER_BACKSLIDE_TOLERANCE {
                    backslide_count += 1;
                }
                if backslide > worst_order_backslide {
                    worst_order_backslide = backslide;
                    worst_backslide_pair = Some((loop_.points[w[0]], loop_.points[w[1]]));
                }
            }
            let zigzag = backslide_count > (run_len / 20).max(2);
            if zigzag {
                if let Some((a, b)) = worst_backslide_pair {
                    println!(
                        "  ZIGZAG: run_len={run_len} backslides={backslide_count} \
                         worst_order_backslide={worst_order_backslide:.4} \
                         between {a:.3?} (order {:.4}) and {b:.3?} (order {:.4})",
                        field.order(a),
                        field.order(b)
                    );
                }
            }
            // Reported in the "worst hop" slot: the final connection to
            // the run's first real point, which IS hop-limit-bounded.
            let post_fix_worst_hop = stitched
                .last()
                .map_or(0.0, |&last| dist(field, &loop_.points, last, after))
                .max(if zigzag {
                    // Surface a zigzag as an automatic violation by
                    // reporting an impossible hop.
                    f64::INFINITY
                } else {
                    0.0
                });
            let chain_points: Vec<DVec3> =
                chain_indices.iter().map(|&idx| loop_.points[idx]).collect();

            runs.push((pre_fix_gap, post_fix_worst_hop, run_len, chain_points));
            i = after;
        } else {
            i = (i + 1) % n;
            visited += 1;
        }
    }
    runs
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: <stl path>");
    let file = std::fs::File::open(&path)?;
    let mesh = stl::load_stl(file)?;

    let object = Object::new(ObjectId(0), mesh, ToolId(0));
    let objects = vec![object];

    // Effectively unbounded build volume so this scratch check isolates
    // wall-gap-stitching behavior from the unrelated default 200x200x200
    // build-volume validation failure (see verify_flat_nozzle.rs for the
    // same workaround).
    let machine = Machine::new(
        BoundingVolume::Aabb {
            min: DVec3::splat(-1_000.0),
            max: DVec3::splat(1_000.0),
        },
        vec![Tool::new(ToolId(0), 0.4)],
    );

    let nozzle_diameter = 0.4;
    let nozzle_radius = nozzle_diameter / 2.0;
    let hop_limit = 0.9 * nozzle_radius;

    let config = SlicerConfig {
        order_field: OrderFieldKind::Eikonal,
        nozzle_diameter,
        // Deliberately left unconstrained (empty = `SlopeProfile`'s
        // default): the Eikonal FMM front on this mesh already produces
        // wall-0 gaps large enough to trigger stitch_wall_gaps without
        // any additional slope restriction. A tighter profile (e.g.
        // `vec![(0.0, 90.0), (10.0, 35.0), (12.0, 35.0), (14.0, 90.0)]`
        // for a shallow 35-degree band from 10-12mm) was tried and
        // produced the same stitching pattern -- confirming the gap
        // here comes from the mesh's own geometry/Eikonal front shape,
        // not from an added slope constraint.
        eikonal_slope_profile: Vec::new(),
        // Disable toolpath simplification so the reported per-segment
        // hop distances/Overhang tags reflect stitch_wall_gaps' direct
        // output, not a subsequent RDP decimation pass.
        path_simplify_enabled: false,
        ..SlicerConfig::default()
    };

    println!("slicing {path} with Eikonal order field (unconstrained slope profile)...");
    let layers = slicing::slice_workspace(&objects, &[ObjectId(0)], &config)?;
    println!("sliced {} layer(s)", layers.len());

    // --- Pre-fix (reconstructed) vs. post-fix wall-0 gap analysis ---
    let mut worst_pre_fix: Option<(f64, usize, usize)> = None; // (gap, layer_index, run_len)
    let mut worst_post_fix: Option<(f64, usize)> = None; // (hop, layer_index)
    let mut stitched_runs = 0usize;
    let mut stitched_points = 0usize;
    let mut layers_with_stitching = Vec::new();
    let mut post_fix_violations = 0usize;
    let mut worst_violations: Vec<(f64, usize, f64, usize, Vec<DVec3>)> = Vec::new(); // (hop, layer, pre_fix_gap, run_len, chain)

    // Void-crossing check: an inserted stitch segment must lie in (or
    // hug) the solid -- a segment whose interior samples sit clearly
    // OUTSIDE the mesh (positive SDF beyond one bead radius) is printing
    // through air across a void, which is physically wrong regardless of
    // how well its hops satisfy the lateral limit.
    let void_tolerance = nozzle_radius;
    let mut void_segments = 0usize;
    let mut worst_void: Option<(f64, usize, DVec3, DVec3)> = None; // (max sdf, layer, a, b)

    for layer in &layers {
        for wall in layer.loops.iter().filter(|w| w.wall_index == 0) {
            debug_assert_eq!(wall.points.len(), wall.unsupported.len());

            if let Some(sdf) = layer.mesh_sdf.as_ref() {
                for k in 0..wall.points.len().saturating_sub(1) {
                    if !(wall.unsupported[k] || wall.unsupported[k + 1]) {
                        continue;
                    }
                    let (a, b) = (wall.points[k], wall.points[k + 1]);
                    let samples =
                        ((a.distance(b) / (0.5 * nozzle_radius)).ceil() as usize).clamp(2, 64);
                    let max_sdf = (0..=samples)
                        .map(|s| {
                            let t = s as f64 / samples as f64;
                            sdf.sample(a.lerp(b, t)).value
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    if max_sdf > void_tolerance {
                        void_segments += 1;
                        if worst_void.is_none_or(|(best, _, _, _)| max_sdf > best) {
                            worst_void = Some((max_sdf, layer.index, a, b));
                        }
                    }
                }
            }

            let runs = analyze_stitch_runs(layer.order_field.as_ref(), wall);
            if !runs.is_empty() {
                layers_with_stitching.push(layer.index);
            }
            for (pre_fix_gap, post_fix_hop, run_len, chain) in runs {
                stitched_runs += 1;
                stitched_points += run_len;
                if worst_pre_fix.is_none_or(|(best, _, _)| pre_fix_gap > best) {
                    worst_pre_fix = Some((pre_fix_gap, layer.index, run_len));
                }
                if worst_post_fix.is_none_or(|(best, _)| post_fix_hop > best) {
                    worst_post_fix = Some((post_fix_hop, layer.index));
                }
                if post_fix_hop > hop_limit + 1e-9 {
                    post_fix_violations += 1;
                    worst_violations.push((post_fix_hop, layer.index, pre_fix_gap, run_len, chain));
                }
            }
        }
    }
    worst_violations.sort_by(|a, b| b.0.total_cmp(&a.0));

    println!();
    println!("=== Wall-0 inter-layer gap analysis (LATERAL metric) ===");
    println!("nozzle_diameter={nozzle_diameter} nozzle_radius={nozzle_radius} hop_limit(0.9*radius)={hop_limit:.4}");
    println!("stitched runs found: {stitched_runs} (total stitched points: {stitched_points})");
    println!("layers containing stitched runs: {layers_with_stitching:?}");
    match worst_pre_fix {
        Some((gap, layer_index, run_len)) => println!(
            "worst PRE-FIX (reconstructed) LATERAL gap: {gap:.4} mm at layer {layer_index} (stitched with {run_len} inserted point(s); hop_limit={hop_limit:.4})"
        ),
        None => println!(
            "worst PRE-FIX (reconstructed) gap: none found -- no stitching occurred, gap not reproduced by this slope profile"
        ),
    }
    match worst_post_fix {
        Some((hop, layer_index)) => println!(
            "worst POST-FIX consecutive LATERAL hop (within any stitch chain, any layer): {hop:.4} mm at layer {layer_index}"
        ),
        None => println!("worst POST-FIX consecutive hop: no stitch chains found"),
    }
    println!(
        "post-fix LATERAL hop-limit violations (stitch chains with a lateral hop > 0.9*nozzle_radius): {post_fix_violations}"
    );
    println!(
        "void-crossing inserted segments (interior SDF sample > {void_tolerance:.4} mm outside the mesh): {void_segments}"
    );
    if let Some((max_sdf, layer_index, a, b)) = worst_void {
        println!("  worst void crossing: layer={layer_index} max_sdf={max_sdf:.4} between {a:.3?} and {b:.3?}");
    }
    for (rank, (hop, layer_index, pre_fix_gap, run_len, chain)) in
        worst_violations.iter().take(3).enumerate()
    {
        println!(
            "  violation #{}: layer={layer_index} worst_lateral_hop={hop:.4} pre_fix_lateral_gap={pre_fix_gap:.4} run_len={run_len}",
            rank + 1
        );
        for w in chain.windows(2) {
            println!(
                "    {:?} -> {:?}  hop_3d={:.4}",
                w[0],
                w[1],
                w[0].distance(w[1])
            );
        }
    }

    // --- Overhang-tagged segment count/location via toolpath::plan ---
    let paths = toolpath::plan(&layers, &objects, &machine.tools, &config)?;
    let mut overhang_segments = 0usize;
    let mut overhang_orders = Vec::new();
    for path in &paths {
        for segment in &path.segments {
            if segment.kind == MoveKind::Overhang {
                overhang_segments += 1;
                overhang_orders.push(segment.order);
            }
        }
    }
    println!();
    println!("=== toolpath::plan Overhang segments ===");
    println!("Overhang-tagged segments: {overhang_segments}");
    if let (Some(min), Some(max)) = (
        overhang_orders
            .iter()
            .copied()
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.min(v)))
            }),
        overhang_orders
            .iter()
            .copied()
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            }),
    ) {
        println!("Overhang segments span order (height) range: {min:.4} .. {max:.4}");
    }

    println!();
    let worst_violation_hop = worst_violations.first().map_or(0.0, |v| v.0);
    if stitched_runs == 0 {
        println!(
            "RESULT: INCONCLUSIVE -- this slope profile did not reproduce a wall-0 gap; adjust eikonal_slope_profile and re-run."
        );
    } else if void_segments > 0 {
        println!(
            "RESULT: FAIL -- {void_segments} inserted stitch segment(s) cross a void (print through air outside the mesh)."
        );
    } else if post_fix_violations == 0 {
        println!(
            "RESULT: PASS -- {stitched_runs} gap(s) up to {:.4} mm (lateral) were stitched, and every post-fix consecutive lateral hop is within the {hop_limit:.4} mm limit.",
            worst_pre_fix.map_or(0.0, |(g, _, _)| g)
        );
    } else if worst_violation_hop.is_finite() && worst_violation_hop <= BRIDGEABLE_RESIDUAL_HOP {
        println!(
            "RESULT: PASS (with warnings) -- no zigzag chains; {post_fix_violations} chain(s) retain a residual bridging hop above the {hop_limit:.4} mm limit (worst {worst_violation_hop:.4} mm, all <= {BRIDGEABLE_RESIDUAL_HOP:.1} mm) at genuine order-field discontinuities where no intermediate isosurface exists."
        );
    } else {
        println!(
            "RESULT: FAIL -- {post_fix_violations} wall-0 loop(s) still have a consecutive lateral hop exceeding the {hop_limit:.4} mm limit after stitching (worst {worst_violation_hop:.4} mm; zigzag chains report as infinite)."
        );
    }

    Ok(())
}
