//! CPU-side text rasterization for baking object name labels into a shared
//! GPU texture atlas (see `scene::build_object_labels` / `render::UploadedScene`).
//! Uses egui's bundled "Hack" font via `ab_glyph` — no extra font asset needed,
//! and visually consistent with the rest of the GUI's default text.

use ab_glyph::{Font as _, FontRef, Glyph, Point, PxScale, ScaleFont};

/// Rasterized glyph height in the atlas, in pixels. World-space label size is
/// derived from this via a fixed mm-per-pixel scale in `scene::LABEL_HEIGHT_MM`.
const ATLAS_PIXEL_HEIGHT: f32 = 48.0;
/// Horizontal gap between adjacent labels packed into the shared atlas row,
/// preventing bilinear sampling from bleeding one label's glyphs into the next.
const ATLAS_PADDING_PX: u32 = 4;

/// One label's placement within the shared [`TextAtlas`]'s single texture row.
#[derive(Debug, Clone, Copy)]
pub struct LabelMetrics {
    pub x_offset_px: u32,
    pub width_px: u32,
}

/// A single-channel (coverage-as-alpha) texture atlas packing every requested
/// label's rasterized text left-to-right in one row, plus each label's
/// placement within it. Index-aligned with the `names` slice passed to
/// [`build_atlas`].
pub struct TextAtlas {
    pub width_px: u32,
    pub height_px: u32,
    /// Row-major, single-channel (R8) coverage buffer: `pixels[y * width + x]`.
    pub pixels: Vec<u8>,
    pub labels: Vec<LabelMetrics>,
}

/// Rasterizes `names` (in order) into one shared [`TextAtlas`] using egui's
/// bundled "Hack" monospace font. Empty `names` yields a 1x1 empty atlas.
pub fn build_atlas(names: &[String]) -> TextAtlas {
    if names.is_empty() {
        return TextAtlas {
            width_px: 1,
            height_px: 1,
            pixels: vec![0],
            labels: Vec::new(),
        };
    }

    let font_bytes = egui::FontDefinitions::default()
        .font_data
        .get("Hack")
        .expect("egui bundles a \"Hack\" default font")
        .font
        .to_vec();
    let font = FontRef::try_from_slice(&font_bytes).expect("egui's bundled Hack font is valid");
    let scaled_font = font.as_scaled(PxScale::from(ATLAS_PIXEL_HEIGHT));
    let height_px = (scaled_font.ascent() - scaled_font.descent())
        .ceil()
        .max(1.0) as u32;

    let rasters: Vec<(u32, Vec<u8>)> = names
        .iter()
        .map(|name| rasterize_one(&font, &scaled_font, name, height_px))
        .collect();

    let width_px = rasters
        .iter()
        .map(|(w, _)| w + ATLAS_PADDING_PX)
        .sum::<u32>()
        .saturating_sub(ATLAS_PADDING_PX)
        .max(1);

    let mut pixels = vec![0u8; (width_px * height_px) as usize];
    let mut labels = Vec::with_capacity(rasters.len());
    let mut x_offset_px = 0u32;
    for (w, raster) in rasters {
        for y in 0..height_px {
            let src_start = (y * w) as usize;
            let src = &raster[src_start..src_start + w as usize];
            let dst_start = (y * width_px + x_offset_px) as usize;
            pixels[dst_start..dst_start + w as usize].copy_from_slice(src);
        }
        labels.push(LabelMetrics {
            x_offset_px,
            width_px: w,
        });
        x_offset_px += w + ATLAS_PADDING_PX;
    }

    TextAtlas {
        width_px,
        height_px,
        pixels,
        labels,
    }
}

/// Rasterizes a single line of `text` at `height_px` tall, returning its
/// pixel width and a row-major single-channel coverage buffer sized
/// `width_px * height_px`.
fn rasterize_one<'a>(
    font: &FontRef<'a>,
    scaled_font: &impl ScaleFont<&'a FontRef<'a>>,
    text: &str,
    height_px: u32,
) -> (u32, Vec<u8>) {
    let mut glyphs: Vec<Glyph> = Vec::new();
    let mut caret_x = 0.0f32;
    let mut prev: Option<ab_glyph::GlyphId> = None;
    for c in text.chars() {
        let glyph_id = scaled_font.glyph_id(c);
        if let Some(prev_id) = prev {
            caret_x += scaled_font.kern(prev_id, glyph_id);
        }
        let position = Point {
            x: caret_x,
            y: scaled_font.ascent(),
        };
        glyphs.push(glyph_id.with_scale_and_position(scaled_font.scale(), position));
        caret_x += scaled_font.h_advance(glyph_id);
        prev = Some(glyph_id);
    }

    let width_px = caret_x.ceil().max(1.0) as u32;
    let mut pixels = vec![0u8; (width_px * height_px) as usize];
    for glyph in glyphs {
        let Some(outlined) = font.outline_glyph(glyph) else {
            continue;
        };
        let bounds = outlined.px_bounds();
        outlined.draw(|dx, dy, coverage| {
            let x = dx as i32 + bounds.min.x as i32;
            let y = dy as i32 + bounds.min.y as i32;
            if x < 0 || y < 0 || x as u32 >= width_px || y as u32 >= height_px {
                return;
            }
            let idx = (y as u32 * width_px + x as u32) as usize;
            pixels[idx] = pixels[idx].max((coverage * 255.0).round() as u8);
        });
    }
    (width_px, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_names_yields_empty_atlas() {
        let atlas = build_atlas(&[]);
        assert_eq!(atlas.width_px, 1);
        assert_eq!(atlas.height_px, 1);
        assert!(atlas.labels.is_empty());
    }

    #[test]
    fn atlas_packs_one_label_per_name_left_to_right() {
        let names = vec!["Object 1".to_string(), "benchy".to_string()];
        let atlas = build_atlas(&names);
        assert_eq!(atlas.labels.len(), 2);
        assert_eq!(atlas.labels[0].x_offset_px, 0);
        assert!(atlas.labels[1].x_offset_px >= atlas.labels[0].width_px);
        assert_eq!(
            atlas.pixels.len(),
            (atlas.width_px * atlas.height_px) as usize
        );
        // Some rasterized coverage should be non-zero for non-empty text.
        assert!(atlas.pixels.iter().any(|&p| p > 0));
    }
}
