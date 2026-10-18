use bevy::prelude::*;
use crate::config::{ArrangementConfig, BedShape, PrintBedConfig};

/// Input footprint description for a single object to be arranged on the print bed.
#[derive(Clone, Debug)]
pub struct ArrangeItem {
    pub object_id: u64,
    /// Half-extents of the object's rotated world-space AABB in XY (`(half_width_x, half_depth_y)`).
    pub half_extents_xy: Vec2,
    /// Offset of the object's rotated world-space XY center from its `Transform::translation.xy()`.
    pub center_offset_xy: Vec2,
    /// Total world-space Z height ($z_{\max} - z_{\min}$) of the object in its current orientation.
    pub height_z: f32,
}

/// Output placement result from `auto_arrange_spiral`.
#[derive(Clone, Debug)]
pub struct ArrangePlacement {
    pub object_id: u64,
    /// Target `Transform::translation.x` on the print bed.
    pub target_x: f32,
    /// Target `Transform::translation.y` on the print bed.
    pub target_y: f32,
    /// True if the object fits completely inside the configured print bed boundaries.
    pub fits_on_bed: bool,
}

/// Arranges objects on the print bed in **descending Z height** order along an
/// Archimedean spiral ($r(\theta) = \frac{\text{pitch}}{2\pi}\theta$) starting at the bed center $(0, 0)$.
pub fn auto_arrange_spiral(
    items: &[ArrangeItem],
    bed: &PrintBedConfig,
    cfg: &ArrangementConfig,
) -> Vec<ArrangePlacement> {
    if items.is_empty() {
        return Vec::new();
    }

    // 1. Sort items by descending Z height (tie-breaking by largest XY footprint area)
    let mut sorted_indices: Vec<usize> = (0..items.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        items[b]
            .height_z
            .partial_cmp(&items[a].height_z)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let area_a = items[a].half_extents_xy.x * items[a].half_extents_xy.y;
                let area_b = items[b].half_extents_xy.x * items[b].half_extents_xy.y;
                area_b.partial_cmp(&area_a).unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let half_clearance = 0.5 * cfg.part_clearance_mm.max(0.0);
    let dtheta = cfg.spiral_angle_step_rad.max(0.05);
    let b_coeff = cfg.spiral_radial_pitch_mm.max(1.0) / std::f32::consts::TAU;

    // Placed 2D AABBs (min_xy, max_xy) inflated by half_clearance
    let mut placed_boxes: Vec<(Vec2, Vec2)> = Vec::with_capacity(items.len());
    let mut results = Vec::with_capacity(items.len());

    for &idx in &sorted_indices {
        let item = &items[idx];
        let padded_half = item.half_extents_xy + Vec2::splat(half_clearance);

        let mut theta = 0.0_f32;
        let chosen_center = loop {
            let r = b_coeff * theta;
            let candidate_center = Vec2::new(r * theta.cos(), r * theta.sin());
            let cand_min = candidate_center - padded_half;
            let cand_max = candidate_center + padded_half;

            let overlaps = placed_boxes.iter().any(|&(other_min, other_max)| {
                cand_min.x < other_max.x
                    && cand_max.x > other_min.x
                    && cand_min.y < other_max.y
                    && cand_max.y > other_min.y
            });

            if !overlaps {
                placed_boxes.push((cand_min, cand_max));
                break candidate_center;
            }

            theta += dtheta;
            // Adaptive step adjustment at larger radii so arc length steps stay uniform
            if r > 20.0 {
                let adaptive_dtheta = (cfg.spiral_radial_pitch_mm * 0.35 / r).clamp(0.02, dtheta);
                theta += adaptive_dtheta - dtheta;
            }
        };

        let fits = is_box_within_bed(
            chosen_center - item.half_extents_xy,
            chosen_center + item.half_extents_xy,
            bed,
            cfg.bed_edge_margin_mm,
        );

        let target_translation_xy = chosen_center - item.center_offset_xy;
        results.push(ArrangePlacement {
            object_id: item.object_id,
            target_x: target_translation_xy.x,
            target_y: target_translation_xy.y,
            fits_on_bed: fits,
        });
    }

    results
}

fn is_box_within_bed(min_xy: Vec2, max_xy: Vec2, bed: &PrintBedConfig, margin: f32) -> bool {
    match bed.shape {
        BedShape::Rectangular => {
            let half_x = (0.5 * bed.width_x - margin).max(1.0);
            let half_y = (0.5 * bed.depth_y - margin).max(1.0);
            min_xy.x >= -half_x && max_xy.x <= half_x && min_xy.y >= -half_y && max_xy.y <= half_y
        }
        BedShape::Circular => {
            let max_r = (0.5 * bed.width_x - margin).max(1.0);
            let max_r_sq = max_r * max_r;
            let corners = [
                min_xy,
                Vec2::new(max_xy.x, min_xy.y),
                max_xy,
                Vec2::new(min_xy.x, max_xy.y),
            ];
            corners.iter().all(|c| c.length_squared() <= max_r_sq)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiral_auto_arrange_descending_z_and_non_overlapping() {
        let items = vec![
            ArrangeItem {
                object_id: 1,
                half_extents_xy: Vec2::new(15.0, 15.0),
                center_offset_xy: Vec2::ZERO,
                height_z: 20.0,
            },
            ArrangeItem {
                object_id: 2,
                half_extents_xy: Vec2::new(20.0, 20.0),
                center_offset_xy: Vec2::ZERO,
                height_z: 85.0, // Tallest object -> must be placed first at origin (0, 0)
            },
            ArrangeItem {
                object_id: 3,
                half_extents_xy: Vec2::new(12.0, 18.0),
                center_offset_xy: Vec2::ZERO,
                height_z: 45.0,
            },
        ];

        let bed = PrintBedConfig::default();
        let cfg = ArrangementConfig::default();
        let placements = auto_arrange_spiral(&items, &bed, &cfg);

        assert_eq!(placements.len(), 3);
        // Tallest object (id 2, height 85.0) is first and placed at (0, 0)
        assert_eq!(placements[0].object_id, 2);
        assert!(placements[0].target_x.abs() < 1e-4);
        assert!(placements[0].target_y.abs() < 1e-4);
        assert!(placements.iter().all(|p| p.fits_on_bed));

        // Verify no pair of placed AABBs overlap
        for i in 0..placements.len() {
            for j in (i + 1)..placements.len() {
                let pa = &placements[i];
                let pb = &placements[j];
                let ia = items.iter().find(|x| x.object_id == pa.object_id).unwrap();
                let ib = items.iter().find(|x| x.object_id == pb.object_id).unwrap();

                let min_a = Vec2::new(pa.target_x, pa.target_y) - ia.half_extents_xy;
                let max_a = Vec2::new(pa.target_x, pa.target_y) + ia.half_extents_xy;
                let min_b = Vec2::new(pb.target_x, pb.target_y) - ib.half_extents_xy;
                let max_b = Vec2::new(pb.target_x, pb.target_y) + ib.half_extents_xy;

                let overlap = min_a.x < max_b.x
                    && max_a.x > min_b.x
                    && min_a.y < max_b.y
                    && max_a.y > min_b.y;
                assert!(!overlap, "Objects {} and {} overlapped!", pa.object_id, pb.object_id);
            }
        }
    }
}
