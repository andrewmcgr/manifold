use crate::bridge::ffi::{
    ExtrusionPath, ObjectToolpathsResult, SurfaceOpticalSample, ToolpathPoint,
};
use serde::{Deserialize, Serialize};

/// Bit flag in [`ExtrusionPath::filament_slot`] indicating that the lower 7 bits
/// (`slot & 0x7F`) directly address `SlicerConfig::tools[tool_idx]` rather than
/// `SlicerConfig::filament_slots`.
pub const DIRECT_TOOL_SLOT_BIT: u8 = 0x80;
pub const DIRECT_TOOL_SLOT_MASK: u8 = 0x7F;

/// Researched baseline Transmission Distance (mm) for standard opaque pigmented
/// PLA / PETG / ABS / ASA filaments when no explicit TD is provided.
///
/// In HueForge / Kubelka-Munk filament characterization (where TD is the physical
/// thickness required to attenuate ~99% of transmitted light, $T(\text{TD}) \approx 0.01$),
/// standard mid-tone pigmented filaments cluster around `2.0..3.0 mm`.
pub const DEFAULT_TRANSMISSION_DISTANCE_MM: f32 = 2.5;

/// Encodes a 0-based machine tool index (`0..127`) into an [`ExtrusionPath::filament_slot`].
#[inline]
pub fn encode_direct_tool_slot(tool_index: usize) -> u8 {
    DIRECT_TOOL_SLOT_BIT | ((tool_index.min(DIRECT_TOOL_SLOT_MASK as usize)) as u8)
}

/// Decodes a direct machine tool index from an [`ExtrusionPath::filament_slot`], if bit 7 is set.
#[inline]
pub fn decode_direct_tool_slot(filament_slot: u8) -> Option<usize> {
    if (filament_slot & DIRECT_TOOL_SLOT_BIT) != 0 {
        Some((filament_slot & DIRECT_TOOL_SLOT_MASK) as usize)
    } else {
        None
    }
}

// ============================================================================
// Standard Color Encodings (Hex, CSS rgb/rgba, sRGB IEC 61966-2-1, XYZ, Oklab)
// ============================================================================

/// Packs `[R, G, B, A]` (`u8`) into `0xRRGGBBAA` (`u32`).
#[inline]
pub const fn pack_rgba8(rgba: [u8; 4]) -> u32 {
    ((rgba[0] as u32) << 24)
        | ((rgba[1] as u32) << 16)
        | ((rgba[2] as u32) << 8)
        | (rgba[3] as u32)
}

/// Unpacks `0xRRGGBBAA` (`u32`) into `[R, G, B, A]` (`u8`).
#[inline]
pub const fn unpack_rgba8(packed: u32) -> [u8; 4] {
    [
        ((packed >> 24) & 0xFF) as u8,
        ((packed >> 16) & 0xFF) as u8,
        ((packed >> 8) & 0xFF) as u8,
        (packed & 0xFF) as u8,
    ]
}

#[inline]
pub fn rgba8_to_srgba(rgba: [u8; 4]) -> [f32; 4] {
    [
        rgba[0] as f32 / 255.0,
        rgba[1] as f32 / 255.0,
        rgba[2] as f32 / 255.0,
        rgba[3] as f32 / 255.0,
    ]
}

#[inline]
pub fn srgba_to_rgba8(srgba: [f32; 4]) -> [u8; 4] {
    [
        (srgba[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (srgba[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (srgba[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (srgba[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Formats `[u8; 4]` as `#RRGGBB` (when opaque) or `#RRGGBBAA`.
pub fn format_hex_rgba(rgba: [u8; 4]) -> String {
    if rgba[3] == 255 {
        format!("#{:02X}{:02X}{:02X}", rgba[0], rgba[1], rgba[2])
    } else {
        format!(
            "#{:02X}{:02X}{:02X}{:02X}",
            rgba[0], rgba[1], rgba[2], rgba[3]
        )
    }
}

/// Formats `[f32; 4]` sRGBA as `#RRGGBB` or `#RRGGBBAA`.
pub fn srgba_to_hex(srgba: [f32; 4]) -> String {
    format_hex_rgba(srgba_to_rgba8(srgba))
}

/// Parses standard color encodings:
/// - Hex: `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA` (also `0x...` or bare hex digits)
/// - Functional CSS: `rgb(255, 128, 0)`, `rgb(100%, 50%, 0%)`, `rgba(255, 128, 0, 0.8)`
/// - Common named colors (`white`, `black`, `cyan`, `magenta`, `yellow`, `red`, `green`, `blue`, etc.)
pub fn parse_color_encoding_rgba8(raw: &str) -> Option<[u8; 4]> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "white" => return Some([255, 255, 255, 255]),
        "black" => return Some([20, 20, 20, 255]),
        "cyan" | "aqua" => return Some([0, 134, 214, 255]),
        "magenta" | "fuchsia" => return Some([236, 0, 140, 255]),
        "yellow" => return Some([252, 227, 0, 255]),
        "red" => return Some([220, 38, 38, 255]),
        "green" => return Some([22, 163, 74, 255]),
        "blue" => return Some([37, 99, 235, 255]),
        "orange" => return Some([249, 115, 22, 255]),
        "purple" | "violet" => return Some([147, 51, 234, 255]),
        "gray" | "grey" => return Some([140, 140, 140, 255]),
        "silver" => return Some([192, 192, 192, 255]),
        "clear" | "natural" | "transparent" => return Some([245, 245, 245, 128]),
        _ => {}
    }

    if let Some(inner) = lower
        .strip_prefix("rgba(")
        .or_else(|| lower.strip_prefix("rgb("))
        .and_then(|s| s.strip_suffix(')'))
    {
        let parts: Vec<&str> = inner
            .split(|c: char| c == ',' || c == '/' || c.is_ascii_whitespace())
            .filter(|p| !p.is_empty())
            .collect();
        if parts.len() == 3 || parts.len() == 4 {
            let parse_chan = |tok: &str| -> Option<u8> {
                if let Some(pct) = tok.strip_suffix('%') {
                    let v: f32 = pct.trim().parse().ok()?;
                    Some((v.clamp(0.0, 100.0) * 2.55).round() as u8)
                } else {
                    let v: f32 = tok.trim().parse().ok()?;
                    Some(v.clamp(0.0, 255.0).round() as u8)
                }
            };
            let r = parse_chan(parts[0])?;
            let g = parse_chan(parts[1])?;
            let b = parse_chan(parts[2])?;
            let a = if parts.len() == 4 {
                let tok = parts[3].trim();
                if let Some(pct) = tok.strip_suffix('%') {
                    let v: f32 = pct.trim().parse().ok()?;
                    (v.clamp(0.0, 100.0) * 2.55).round() as u8
                } else {
                    let v: f32 = tok.parse().ok()?;
                    if v <= 1.0 && (tok.contains('.') || v == 0.0 || v == 1.0) {
                        (v.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        v.clamp(0.0, 255.0).round() as u8
                    }
                }
            } else {
                255
            };
            return Some([r, g, b, a]);
        }
    }

    let hex = trimmed
        .strip_prefix('#')
        .or_else(|| trimmed.strip_prefix("0x"))
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);

    match hex.len() {
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some([r, g, b, 255])
        }
        4 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            let a = u8::from_str_radix(&hex[3..4], 16).ok()? * 17;
            Some([r, g, b, a])
        }
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some([r, g, b, 255])
        }
        8 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            let a = u8::from_str_radix(&hex[6..8], 16).ok()?;
            Some([r, g, b, a])
        }
        _ => None,
    }
}

pub fn parse_color_encoding_srgba(raw: &str) -> Option<[f32; 4]> {
    parse_color_encoding_rgba8(raw).map(rgba8_to_srgba)
}

/// Exact IEC 61966-2-1 inverse companding (sRGB `[0, 1]` -> linear light `[0, 1]`).
#[inline]
pub fn srgb_channel_to_linear(c: f64) -> f64 {
    let x = c.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Exact IEC 61966-2-1 forward companding (linear light `[0, 1]` -> sRGB `[0, 1]`).
#[inline]
pub fn linear_channel_to_srgb(l: f64) -> f64 {
    let x = l.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub fn srgb_to_linear_rgb(srgb: [f32; 3]) -> [f64; 3] {
    [
        srgb_channel_to_linear(srgb[0] as f64),
        srgb_channel_to_linear(srgb[1] as f64),
        srgb_channel_to_linear(srgb[2] as f64),
    ]
}

#[inline]
pub fn linear_rgb_to_srgb(lin: [f64; 3]) -> [f32; 3] {
    [
        linear_channel_to_srgb(lin[0]) as f32,
        linear_channel_to_srgb(lin[1]) as f32,
        linear_channel_to_srgb(lin[2]) as f32,
    ]
}

/// Converts linear sRGB (D65) to CIE 1931 XYZ (D65 white point).
#[inline]
pub fn linear_rgb_to_xyz_d65(lin: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = lin;
    [
        0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
        0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
        0.0193339 * r + 0.1191920 * g + 0.9503041 * b,
    ]
}

/// Converts linear sRGB (D65) to perceptual Oklab `[L, a, b]` (Björn Ottosson, 2020).
#[inline]
pub fn linear_rgb_to_oklab(lin: [f64; 3]) -> [f64; 3] {
    let r = lin[0].max(0.0);
    let g = lin[1].max(0.0);
    let b = lin[2].max(0.0);

    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();

    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// Converts perceptual Oklab `[L, a, b]` back to linear sRGB (clamped to `[0, 1]`).
#[inline]
pub fn oklab_to_linear_rgb(lab: [f64; 3]) -> [f64; 3] {
    let [l_cap, a, b] = lab;
    let l_ = l_cap + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l_cap - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l_cap - 0.0894841775 * a - 1.2914855480 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s).clamp(0.0, 1.0),
        (-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s).clamp(0.0, 1.0),
        (-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s).clamp(0.0, 1.0),
    ]
}

#[inline]
pub fn srgb_to_oklab(srgb: [f32; 3]) -> [f64; 3] {
    linear_rgb_to_oklab(srgb_to_linear_rgb(srgb))
}

#[inline]
pub fn oklab_to_srgb(lab: [f64; 3]) -> [f32; 3] {
    linear_rgb_to_srgb(oklab_to_linear_rgb(lab))
}

#[inline]
pub fn oklab_distance_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    dl * dl + da * da + db * db
}

#[inline]
pub fn oklab_delta_e(a: [f64; 3], b: [f64; 3]) -> f64 {
    oklab_distance_sq(a, b).sqrt()
}

// ============================================================================
// Researched Default Transmission Distance (TD) & Built-in CMYKW Palettes
// ============================================================================

/// Estimates a physically realistic Transmission Distance (`TD`, mm for 99% light attenuation)
/// when the user leaves a filament's TD on Automatic (`None`), accounting for both material
/// modifiers (e.g. CF/GF, Translucent/Clear) and pigment chemistry (carbon-black vs TiO2 white
/// vs diarylide yellow vs standard organic pigments).
pub fn estimate_default_td_mm(filament_type: &str, color_srgb: [f32; 3]) -> f32 {
    let ft = filament_type.trim().to_ascii_lowercase();

    // Carbon-fiber, glass-fiber, or metal-filled polymers are heavily opaque (TD ~ 0.3 - 0.5 mm).
    if ft.contains("-cf")
        || ft.contains(" cf")
        || ft.contains("carbon")
        || ft.contains("-gf")
        || ft.contains("glass")
        || ft.contains("metal")
        || ft.contains("bronze")
        || ft.contains("copper")
        || ft.contains("steel")
    {
        return 0.4;
    }

    // Clear / Natural / Translucent / PVA polymers have minimal pigment scattering (TD ~ 10 mm).
    if ft.contains("translucent")
        || ft.contains("transparent")
        || ft.contains("clear")
        || ft.contains("natural")
        || ft == "pva"
    {
        return 10.0;
    }

    let lab = srgb_to_oklab(color_srgb);
    let l = lab[0];
    let a = lab[1];
    let b = lab[2];
    let chroma = (a * a + b * b).sqrt();

    // Carbon-black pigments absorb strongly across the visible spectrum (typical PLA Black TD ~ 0.5 - 0.7 mm).
    if l < 0.24 && chroma < 0.05 {
        return 0.6;
    }
    // Dark shades smoothly ramp from 0.6 mm up toward the 2.5 mm mid-tone pigment baseline.
    if l < 0.42 {
        let t = ((l - 0.24) / (0.42 - 0.24)).clamp(0.0, 1.0) as f32;
        return 0.6 + t * (2.2 - 0.6);
    }
    // Titanium-dioxide (TiO2) Jade/Basic White scatters strongly with low absorption (typical TD ~ 4.0 - 5.0 mm).
    if l > 0.88 && chroma < 0.045 {
        return 4.0;
    }
    // Bright organic yellow pigments have low blue-only absorption and higher transmission (TD ~ 3.5 - 5.0 mm).
    if l > 0.82 && b > 0.12 && a.abs() < 0.09 {
        return 3.5;
    }

    DEFAULT_TRANSMISSION_DISTANCE_MM
}

/// Descriptor for a single calibrated filament in a CMYKW or multi-color optical palette.
#[derive(Clone, Debug, PartialEq)]
pub struct PaletteFilamentEntry {
    pub name: &'static str,
    pub filament_type: &'static str,
    pub color_hex: &'static str,
    pub color_srgba: [f32; 4],
    pub transmission_distance_mm: f32,
}

/// Standard 5-tool PLA CMYKW Lithophane / Color-Mixing palette (modeled on calibrated
/// Bambu Lab PLA CMYK Lithophane + Basic Black values).
pub fn bambu_pla_cmykw_palette() -> [PaletteFilamentEntry; 5] {
    [
        PaletteFilamentEntry {
            name: "Cyan (CMYKW)",
            filament_type: "PLA",
            color_hex: "#0086D6",
            color_srgba: [0.0, 134.0 / 255.0, 214.0 / 255.0, 1.0],
            transmission_distance_mm: 3.5,
        },
        PaletteFilamentEntry {
            name: "Magenta (CMYKW)",
            filament_type: "PLA",
            color_hex: "#EC008C",
            color_srgba: [236.0 / 255.0, 0.0, 140.0 / 255.0, 1.0],
            transmission_distance_mm: 3.2,
        },
        PaletteFilamentEntry {
            name: "Yellow (CMYKW)",
            filament_type: "PLA",
            color_hex: "#FCE300",
            color_srgba: [252.0 / 255.0, 227.0 / 255.0, 0.0, 1.0],
            transmission_distance_mm: 6.0,
        },
        PaletteFilamentEntry {
            name: "Black (CMYKW)",
            filament_type: "PLA",
            color_hex: "#161616",
            color_srgba: [22.0 / 255.0, 22.0 / 255.0, 22.0 / 255.0, 1.0],
            transmission_distance_mm: 0.6,
        },
        PaletteFilamentEntry {
            name: "Jade White (CMYKW)",
            filament_type: "PLA",
            color_hex: "#F4F4F4",
            color_srgba: [244.0 / 255.0, 244.0 / 255.0, 244.0 / 255.0, 1.0],
            transmission_distance_mm: 5.0,
        },
    ]
}

/// High-transmission Translucent CMYKW palette (ideal for thicker perimeter shells `0.40 mm+`
/// and backlit multi-layer color lithophanes).
pub fn translucent_cmykw_palette() -> [PaletteFilamentEntry; 5] {
    [
        PaletteFilamentEntry {
            name: "Translucent Cyan",
            filament_type: "PETG",
            color_hex: "#08ABFB",
            color_srgba: [8.0 / 255.0, 171.0 / 255.0, 251.0 / 255.0, 1.0],
            transmission_distance_mm: 8.0,
        },
        PaletteFilamentEntry {
            name: "Translucent Magenta",
            filament_type: "PETG",
            color_hex: "#D93B90",
            color_srgba: [217.0 / 255.0, 59.0 / 255.0, 144.0 / 255.0, 1.0],
            transmission_distance_mm: 7.8,
        },
        PaletteFilamentEntry {
            name: "Translucent Yellow",
            filament_type: "PETG",
            color_hex: "#F9ED3D",
            color_srgba: [249.0 / 255.0, 237.0 / 255.0, 61.0 / 255.0, 1.0],
            transmission_distance_mm: 14.0,
        },
        PaletteFilamentEntry {
            name: "Blending Black",
            filament_type: "PETG",
            color_hex: "#181818",
            color_srgba: [24.0 / 255.0, 24.0 / 255.0, 24.0 / 255.0, 1.0],
            transmission_distance_mm: 2.3,
        },
        PaletteFilamentEntry {
            name: "Diffuser White",
            filament_type: "PETG",
            color_hex: "#F5F5F5",
            color_srgba: [245.0 / 255.0, 245.0 / 255.0, 245.0 / 255.0, 1.0],
            transmission_distance_mm: 4.5,
        },
    ]
}

// ============================================================================
// Kubelka-Munk Two-Flux Optical Model (Front-Lit Reflectance & Back-Lit Lithophane)
// ============================================================================

/// Asymmetric diffuse optical response of a single layer or a multi-layer stack of
/// heterogeneous filaments in linear RGB channels.
///
/// Supports both:
/// - **Front-lit 3D prints / HueForge paintings**: observed linear RGB is `r_front * I_front`.
/// - **Back-lit monochrome & CMYKW lithophanes**: observed linear RGB is `t * I_back + r_front * I_front`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpticalSlab {
    /// Diffuse reflectance from the observer-facing (+normal / front) side.
    pub r_front: [f64; 3],
    /// Diffuse reflectance from the core-facing (-normal / back) side.
    pub r_back: [f64; 3],
    /// Direct + scattered diffuse transmittance through the slab (symmetric by reciprocity).
    pub t: [f64; 3],
}

impl Default for OpticalSlab {
    fn default() -> Self {
        Self::transparent()
    }
}

impl OpticalSlab {
    #[inline]
    pub const fn transparent() -> Self {
        Self {
            r_front: [0.0; 3],
            r_back: [0.0; 3],
            t: [1.0; 3],
        }
    }

    #[inline]
    pub fn opaque_reflector(linear_rgb: [f64; 3]) -> Self {
        let clamped = [
            linear_rgb[0].clamp(0.0, 1.0),
            linear_rgb[1].clamp(0.0, 1.0),
            linear_rgb[2].clamp(0.0, 1.0),
        ];
        Self {
            r_front: clamped,
            r_back: clamped,
            t: [0.0; 3],
        }
    }

    /// Associative Redheffer star-product stacking `self` (closer to the viewer / outer surface)
    /// over `back` (deeper toward the core / backlight).
    ///
    /// Exact two-flux inter-reflection between asymmetric slabs:
    /// $$R_{1 \otimes 2}^{\text{front}} = R_1^{\text{front}} + \frac{T_1^2 R_2^{\text{front}}}{1 - R_1^{\text{back}} R_2^{\text{front}}}, \qquad
    ///   T_{1 \otimes 2} = \frac{T_1 T_2}{1 - R_1^{\text{back}} R_2^{\text{front}}}$$
    #[inline]
    pub fn stack_over(&self, back: &OpticalSlab) -> OpticalSlab {
        let mut r_front = [0.0; 3];
        let mut r_back = [0.0; 3];
        let mut t = [0.0; 3];
        for c in 0..3 {
            let denom = (1.0 - self.r_back[c] * back.r_front[c]).max(1.0e-9);
            r_front[c] =
                (self.r_front[c] + (self.t[c] * self.t[c] * back.r_front[c]) / denom).clamp(0.0, 1.0);
            r_back[c] =
                (back.r_back[c] + (back.t[c] * back.t[c] * self.r_back[c]) / denom).clamp(0.0, 1.0);
            t[c] = ((self.t[c] * back.t[c]) / denom).clamp(0.0, 1.0);
        }
        OpticalSlab {
            r_front,
            r_back,
            t,
        }
    }

    /// Observed linear RGB under front illumination `illuminant_linear_rgb` (no backlight).
    #[inline]
    pub fn front_lit_linear_rgb(&self, illuminant_linear_rgb: [f64; 3]) -> [f64; 3] {
        [
            (self.r_front[0] * illuminant_linear_rgb[0]).clamp(0.0, 1.0),
            (self.r_front[1] * illuminant_linear_rgb[1]).clamp(0.0, 1.0),
            (self.r_front[2] * illuminant_linear_rgb[2]).clamp(0.0, 1.0),
        ]
    }

    /// Observed linear RGB under rear transillumination `backlight_linear_rgb` (lithophane mode).
    #[inline]
    pub fn back_lit_linear_rgb(&self, backlight_linear_rgb: [f64; 3]) -> [f64; 3] {
        [
            (self.t[0] * backlight_linear_rgb[0]).clamp(0.0, 1.0),
            (self.t[1] * backlight_linear_rgb[1]).clamp(0.0, 1.0),
            (self.t[2] * backlight_linear_rgb[2]).clamp(0.0, 1.0),
        ]
    }

    /// Observed linear RGB under simultaneous front ambient illumination and rear lithophane backlight.
    #[inline]
    pub fn evaluate_lighting(&self, lighting: &LithophaneLighting) -> [f64; 3] {
        [
            (self.r_front[0] * lighting.front_illuminant_linear_rgb[0]
                + self.t[0] * lighting.backlight_linear_rgb[0])
                .clamp(0.0, 1.0),
            (self.r_front[1] * lighting.front_illuminant_linear_rgb[1]
                + self.t[1] * lighting.backlight_linear_rgb[1])
                .clamp(0.0, 1.0),
            (self.r_front[2] * lighting.front_illuminant_linear_rgb[2]
                + self.t[2] * lighting.backlight_linear_rgb[2])
                .clamp(0.0, 1.0),
        ]
    }
}

/// Illumination environment for both standard front-lit 3D prints and back-lit lithophanes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LithophaneLighting {
    pub front_illuminant_linear_rgb: [f64; 3],
    pub backlight_linear_rgb: [f64; 3],
}

impl LithophaneLighting {
    pub const fn default_frontlit_d65() -> Self {
        Self {
            front_illuminant_linear_rgb: [1.0, 1.0, 1.0],
            backlight_linear_rgb: [0.0, 0.0, 0.0],
        }
    }

    pub const fn default_backlit_d65() -> Self {
        Self {
            front_illuminant_linear_rgb: [0.05, 0.05, 0.05],
            backlight_linear_rgb: [1.0, 1.0, 1.0],
        }
    }

    pub const fn pure_backlit_d65() -> Self {
        Self {
            front_illuminant_linear_rgb: [0.0, 0.0, 0.0],
            backlight_linear_rgb: [1.0, 1.0, 1.0],
        }
    }

    pub const fn mixed(front_intensity: f64, back_intensity: f64) -> Self {
        Self {
            front_illuminant_linear_rgb: [front_intensity, front_intensity, front_intensity],
            backlight_linear_rgb: [back_intensity, back_intensity, back_intensity],
        }
    }
}

/// Calibrated Kubelka-Munk optical parameters $(R_\infty, S, K_c)$ for a single filament.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilamentOpticalModel {
    pub color_srgb: [f32; 3],
    pub r_inf_linear: [f64; 3],
    pub td_mm: f64,
    /// Isotropic polymer+pigment scattering coefficient $S$ ($\text{mm}^{-1}$).
    pub scatter_s_per_mm: f64,
    /// Per-channel Kubelka-Munk absorption coefficients $K_c$ ($\text{mm}^{-1}$).
    pub absorb_k_per_mm: [f64; 3],
    /// Precomputed attenuation factors $b_c = \frac{1 - R_{\infty, c}^2}{2 R_{\infty, c}}$.
    km_b: [f64; 3],
}

impl FilamentOpticalModel {
    /// Calibrates $(S, K_R, K_G, K_B)$ from a filament's opaque color `color_srgb` and
    /// physical Transmission Distance `td_mm` (~92% Kubelka-Munk hiding contrast / `8%` passband
    /// transmittance at $d = \text{TD}$, matching empirical HueForge step-wedge behavior).
    ///
    /// Uses the overflow-free exact exponential reformulation of Kubelka-Munk:
    /// $$R_c(d) = \frac{R_{\infty,c}(1 - e^{-2 u_c})}{1 - R_{\infty,c}^2 e^{-2 u_c}}, \qquad
    ///   T_c(d) = \frac{(1 - R_{\infty,c}^2) e^{-u_c}}{1 - R_{\infty,c}^2 e^{-2 u_c}}, \qquad u_c = b_c S d$$
    /// which remains exact from $\text{TiO}_2$ white scatterers through subtractive CMY dyes
    /// down to Beer-Lambert carbon-black absorbers ($R_\infty \to 0$).
    pub fn from_srgb_and_td(color_srgb: [f32; 3], td_mm: f32) -> Self {
        let raw_lin = srgb_to_linear_rgb(color_srgb);
        // Real thermoplastics lose ~4-7% at the air-polymer Fresnel interface + matrix absorption,
        // preventing the lossless non-absorbing singularity R_inf -> 1 where T(d) = 1/(1 + S d).
        let r_inf_linear = [
            raw_lin[0].clamp(0.004, 0.93),
            raw_lin[1].clamp(0.004, 0.93),
            raw_lin[2].clamp(0.004, 0.93),
        ];
        let td = (td_mm as f64).clamp(0.05, 100.0);

        let mut km_b = [0.0; 3];
        for c in 0..3 {
            let r = r_inf_linear[c];
            km_b[c] = (1.0 - r * r) / (2.0 * r);
        }

        let eval_t_eff = |s_val: f64| -> f64 {
            let mut tc = [0.0; 3];
            for c in 0..3 {
                let r = r_inf_linear[c];
                let u = (km_b[c] * s_val * td).min(80.0);
                let eu = (-u).exp();
                let e2u = eu * eu;
                let denom = (1.0 - r * r * e2u).max(1.0e-12);
                tc[c] = ((1.0 - r * r) * eu) / denom;
            }
            let lum_t = 0.2126729 * tc[0] + 0.7151522 * tc[1] + 0.0721750 * tc[2];
            let max_t = tc[0].max(tc[1]).max(tc[2]);
            // Blending luminance transmittance with peak passband transmittance ensures saturated
            // primaries (Cyan, Magenta, Blue) retain realistic glaze translucency across thin layers.
            lum_t.max(0.75 * max_t)
        };

        // Monotone bisection on S > 0 since T_eff(S) decreases strictly from 1 (S -> 0) to 0 (S -> inf).
        let target_t = 0.08;
        let mut s_low = 1.0e-5_f64;
        let mut s_high = 1.0_f64;
        while eval_t_eff(s_high) > target_t && s_high < 1.0e7 {
            s_high *= 2.0;
        }
        for _ in 0..48 {
            let s_mid = 0.5 * (s_low + s_high);
            if eval_t_eff(s_mid) > target_t {
                s_low = s_mid;
            } else {
                s_high = s_mid;
            }
        }
        let scatter_s_per_mm = 0.5 * (s_low + s_high);

        let mut absorb_k_per_mm = [0.0; 3];
        for c in 0..3 {
            let r = r_inf_linear[c];
            absorb_k_per_mm[c] = scatter_s_per_mm * ((1.0 - r) * (1.0 - r)) / (2.0 * r);
        }

        Self {
            color_srgb,
            r_inf_linear,
            td_mm: td,
            scatter_s_per_mm,
            absorb_k_per_mm,
            km_b,
        }
    }

    /// Computes the symmetric single-layer diffuse reflectance and transmittance slab for `thickness_mm`.
    #[inline]
    pub fn slab_response(&self, thickness_mm: f64) -> OpticalSlab {
        let d = thickness_mm.max(0.0);
        if d <= 1.0e-7 {
            return OpticalSlab::transparent();
        }
        let mut r_ch = [0.0; 3];
        let mut t_ch = [0.0; 3];
        for c in 0..3 {
            let r = self.r_inf_linear[c];
            let u = (self.km_b[c] * self.scatter_s_per_mm * d).min(80.0);
            let eu = (-u).exp();
            let e2u = eu * eu;
            let denom = (1.0 - r * r * e2u).max(1.0e-12);
            r_ch[c] = (r * (1.0 - e2u) / denom).clamp(0.0, 1.0);
            t_ch[c] = (((1.0 - r * r) * eu) / denom).clamp(0.0, 1.0);
        }
        OpticalSlab {
            r_front: r_ch,
            r_back: r_ch,
            t: t_ch,
        }
    }

    /// Semi-infinite opaque core backing slab ($d \to \infty \implies R = R_\infty, T = 0$).
    #[inline]
    pub fn opaque_backing_slab(&self) -> OpticalSlab {
        OpticalSlab::opaque_reflector(self.r_inf_linear)
    }

    /// Computes the physical single-slab hiding opacity $\alpha(d, \text{TD}) = 1 - (0.08)^{d / \text{TD}}$
    /// for a filament bead of physical thickness `thickness_mm` (`d = TD => α = 0.92`).
    ///
    /// Satisfies exact Beer–Lambert / HueForge additivity under Porter–Duff alpha compositing:
    /// stacking $N$ beads of thicknesses $d_1, \dots, d_N$ yields
    /// $1 - \prod_i (1 - \alpha_i) = 1 - (0.08)^{(\sum_i d_i) / \text{TD}}$, and is strictly
    /// monotonic in `td_mm` across all filament colours.
    #[inline]
    pub fn opacity_at_thickness_mm(&self, thickness_mm: f32) -> f32 {
        td_opacity_for_thickness_mm(self.td_mm as f32, thickness_mm)
    }
}

/// Computes the physical single-slab hiding opacity $\alpha(d, \text{TD}) = 1 - (0.08)^{d / \text{TD}}$
/// (`[0.06, 1.0]`) for a filament with Transmission Distance `td_mm` at physical bead thickness `thickness_mm`.
#[inline]
pub fn td_opacity_for_thickness_mm(td_mm: f32, thickness_mm: f32) -> f32 {
    let td = (td_mm as f64).max(0.05);
    let d = (thickness_mm as f64).max(0.02);
    // -ln(0.08) = 2.5257286443082556 (~92% hiding contrast at thickness == TD)
    let transmittance = (-2.525_728_644_308_255_6 * (d / td)).exp();
    (1.0 - transmittance).clamp(0.06, 1.0) as f32
}

/// Evaluates an arbitrary multi-layer filament stack ordered from **outermost** (viewer-facing, index 0)
/// to **innermost** (core/backlight-facing), optionally terminated by an opaque `backing` filament.
pub fn evaluate_optical_stack(
    slabs_outer_to_inner: &[(usize, f64)],
    models: &[FilamentOpticalModel],
    backing: Option<&FilamentOpticalModel>,
) -> OpticalSlab {
    let mut composite = OpticalSlab::transparent();
    for &(model_idx, thickness_mm) in slabs_outer_to_inner {
        if let Some(m) = models.get(model_idx) {
            let layer_slab = m.slab_response(thickness_mm);
            composite = composite.stack_over(&layer_slab);
        }
    }
    if let Some(back_model) = backing {
        composite = composite.stack_over(&back_model.opaque_backing_slab());
    }
    composite
}

/// Evaluates the observed sRGB `[0, 1]^3` of a multi-layer stack under arbitrary `LithophaneLighting`
/// (supports front-lit prints, back-lit color lithophanes, and mixed ambient+backlight).
pub fn evaluate_stack_srgb(
    slabs_outer_to_inner: &[(usize, f64)],
    models: &[FilamentOpticalModel],
    backing: Option<&FilamentOpticalModel>,
    lighting: &LithophaneLighting,
) -> [f32; 3] {
    let slab = evaluate_optical_stack(slabs_outer_to_inner, models, backing);
    linear_rgb_to_srgb(slab.evaluate_lighting(lighting))
}

/// Solves for the physical filament thickness $d \in [d_{\min}, d_{\max}]$ (mm) of a single
/// monochrome lithophane filament that achieves `target_perceptual_lightness_01` ($L_{\text{oklab}} \in [0, 1]$)
/// under `lighting` (e.g. `LithophaneLighting::default_backlit_d65()`).
///
/// Ready for direct use by monochrome thickness-modulated lithophane mesh/field generators.
pub fn solve_monochrome_lithophane_thickness(
    filament: &FilamentOpticalModel,
    target_perceptual_lightness_01: f64,
    min_thickness_mm: f64,
    max_thickness_mm: f64,
    lighting: &LithophaneLighting,
) -> f64 {
    let d_min = min_thickness_mm.max(0.02);
    let d_max = max_thickness_mm.max(d_min);
    if (d_max - d_min) <= 1.0e-6 {
        return d_min;
    }

    let lightness_at = |d: f64| -> f64 {
        let lin = filament.slab_response(d).evaluate_lighting(lighting);
        linear_rgb_to_oklab(lin)[0]
    };

    let l_at_min = lightness_at(d_min);
    let l_at_max = lightness_at(d_max);

    // Map normalized [0, 1] target across the achievable [darkest, brightest] span of [d_min, d_max]
    let l_dark = l_at_min.min(l_at_max);
    let l_bright = l_at_min.max(l_at_max);
    let target_l = l_dark + target_perceptual_lightness_01.clamp(0.0, 1.0) * (l_bright - l_dark);

    let decreasing_with_thickness = l_at_min >= l_at_max;
    let mut lo = d_min;
    let mut hi = d_max;
    for _ in 0..44 {
        let mid = 0.5 * (lo + hi);
        let l_mid = lightness_at(mid);
        if decreasing_with_thickness {
            if l_mid > target_l {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if l_mid < target_l {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ============================================================================
// Multi-Filament Optical Color Approximation Configuration & Solver
// ============================================================================

/// Target viewing illumination for optical color mixing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpticalLightingMode {
    /// Front-lit 3D object / HueForge reflection print over an internal core backing filament.
    FrontLitReflectance,
    /// Rear-transilluminated color lithophane (open backlight with no opaque core backing).
    BackLitTransmittance,
    /// Combined ambient front light (35%) + rear backlight (100%).
    MixedFrontAndBack,
}

impl Default for OpticalLightingMode {
    fn default() -> Self {
        Self::FrontLitReflectance
    }
}

impl OpticalLightingMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::FrontLitReflectance => "Front-Lit 3D Object (Reflectance)",
            Self::BackLitTransmittance => "Back-Lit Lithophane (Transmittance)",
            Self::MixedFrontAndBack => "Mixed Ambient + Back-Lit",
        }
    }

    pub fn lighting(self) -> LithophaneLighting {
        match self {
            Self::FrontLitReflectance => LithophaneLighting::default_frontlit_d65(),
            Self::BackLitTransmittance => LithophaneLighting::default_backlit_d65(),
            Self::MixedFrontAndBack => LithophaneLighting::mixed(0.35, 1.0),
        }
    }

    pub fn uses_opaque_core_backing(self) -> bool {
        matches!(self, Self::FrontLitReflectance)
    }
}

/// Configuration for optical color approximation (CMYKW & arbitrary $N$-filament palettes).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorMixingConfig {
    /// Master toggle enabling Kubelka-Munk optical filament mixing across layers & extrusions.
    pub enabled: bool,
    /// Viewing illumination mode (`FrontLitReflectance` for 3D objects, `BackLitTransmittance` for color lithophanes).
    pub lighting_mode: OpticalLightingMode,
    /// Blend colors across consecutive/adjacent layers (`layer_index` spatial dithering + multi-layer optical stacking).
    pub use_adjacent_layers: bool,
    /// Subdivide perimeters and skin extrusions into short color-diffused segments along the path.
    pub use_short_extrusions: bool,
    /// Minimum physical length (mm) of each subdivided color segment when `use_short_extrusions` is active.
    pub min_short_extrusion_mm: f32,
    /// Number of outer shell slabs ($M \in 1..=6$) participating in through-depth optical stacking over the core.
    pub optical_shell_layers: u32,
    /// Optional explicit core/backing tool index (`None` = auto-select brightest / White filament).
    pub backing_tool_index: Option<usize>,
    /// Oklab error-diffusion / spatial dithering strength (`0.0` = nearest optical stack, `1.0` = full dithering).
    pub dither_strength: f32,
    /// Oklab hysteresis penalty (`0.0..=0.10`) discouraging marginal tool switches along continuous paths.
    pub tool_switch_penalty: f32,
}

impl Default for ColorMixingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lighting_mode: OpticalLightingMode::FrontLitReflectance,
            use_adjacent_layers: true,
            use_short_extrusions: true,
            min_short_extrusion_mm: 3.0,
            optical_shell_layers: 4,
            backing_tool_index: None,
            dither_strength: 0.85,
            tool_switch_penalty: 0.012,
        }
    }
}

impl ColorMixingConfig {
    pub fn sanitize(&mut self) {
        if !self.min_short_extrusion_mm.is_finite() {
            self.min_short_extrusion_mm = 3.0;
        }
        self.min_short_extrusion_mm = self.min_short_extrusion_mm.clamp(0.8, 50.0);
        self.optical_shell_layers = self.optical_shell_layers.clamp(1, 6);
        if !self.dither_strength.is_finite() {
            self.dither_strength = 0.85;
        }
        self.dither_strength = self.dither_strength.clamp(0.0, 1.0);
        if !self.tool_switch_penalty.is_finite() {
            self.tool_switch_penalty = 0.012;
        }
        self.tool_switch_penalty = self.tool_switch_penalty.clamp(0.0, 0.25);
    }
}

/// Lightweight summary of a machine tool's optical properties passed into the solver.
#[derive(Clone, Copy, Debug)]
pub struct ToolOpticalInput {
    pub tool_index: usize,
    pub color_srgb: [f32; 3],
    pub td_mm: f32,
}

/// A precomputed $M$-layer optical stack candidate (`tools_outer_to_inner[0]` = outermost surface slab).
#[derive(Clone, Debug)]
pub struct CandidateOpticalStack {
    /// Tool index in `0..tools.len()` for each depth slab $m \in 0..M$ (`0` = outer surface).
    pub tools_outer_to_inner: Vec<usize>,
    /// Simulated composite linear RGB at nominal slab thickness.
    pub linear_rgb: [f64; 3],
    /// Simulated composite Oklab `[L, a, b]` at nominal slab thickness.
    pub oklab: [f64; 3],
    /// Number of distinct internal layer transitions inside the stack (used as a mild tie-breaker).
    pub transition_count: u8,
    /// Mean Oklab deviation of the individual shell slabs from the composite `oklab`
    /// (encourages carrying the surface colour through inner shell layers when optical reflectance ties).
    pub shell_color_spread: f64,
}

/// Precomputed Kubelka-Munk optical lookup table for any user-loaded set of filaments.
#[derive(Clone, Debug)]
pub struct OpticalPaletteLut {
    pub models: Vec<FilamentOpticalModel>,
    pub tool_indices: Vec<usize>,
    pub backing_model_idx: Option<usize>,
    pub shell_layers: usize,
    pub nominal_slab_mm: f64,
    pub lighting: LithophaneLighting,
    pub stacks: Vec<CandidateOpticalStack>,
}

impl OpticalPaletteLut {
    /// Builds an optical stack LUT for arbitrary `tools` ($N \ge 1$), shell depth $M \in 1..=6$,
    /// and nominal single-slab thickness `nominal_slab_mm`.
    pub fn build(
        tools: &[ToolOpticalInput],
        shell_layers: u32,
        nominal_slab_mm: f32,
        lighting_mode: OpticalLightingMode,
        explicit_backing_tool: Option<usize>,
    ) -> Self {
        let models: Vec<FilamentOpticalModel> = tools
            .iter()
            .map(|t| FilamentOpticalModel::from_srgb_and_td(t.color_srgb, t.td_mm))
            .collect();
        let tool_indices: Vec<usize> = tools.iter().map(|t| t.tool_index).collect();

        let backing_model_idx = if lighting_mode.uses_opaque_core_backing() && !models.is_empty() {
            if let Some(exp_tool) = explicit_backing_tool {
                tool_indices
                    .iter()
                    .position(|&ti| ti == exp_tool)
                    .or_else(|| Some(select_auto_backing_model_index(&models)))
            } else {
                Some(select_auto_backing_model_index(&models))
            }
        } else {
            None
        };

        let m_layers = (shell_layers.clamp(1, 6)) as usize;
        let slab_d = (nominal_slab_mm as f64).clamp(0.04, 2.0);
        let lighting = lighting_mode.lighting();

        if models.is_empty() {
            return Self {
                models,
                tool_indices,
                backing_model_idx: None,
                shell_layers: m_layers,
                nominal_slab_mm: slab_d,
                lighting,
                stacks: Vec::new(),
            };
        }

        let single_slabs: Vec<OpticalSlab> =
            models.iter().map(|m| m.slab_response(slab_d)).collect();
        let model_oklabs: Vec<[f64; 3]> = models
            .iter()
            .map(|m| linear_rgb_to_oklab(m.r_inf_linear))
            .collect();

        // Build stacks from innermost (m = M - 1) outward to outermost (m = 0).
        // When N^M > 1536 (e.g. 8+ filaments across 5 layers), prune perceptually redundant
        // intermediate states in Oklab voxel bins so construction stays < 2 ms for any palette size.
        #[derive(Clone)]
        struct PartialStack {
            tools_inner_to_outer: Vec<usize>,
            slab: OpticalSlab,
        }

        let initial_slab = if let Some(b_idx) = backing_model_idx {
            models[b_idx].opaque_backing_slab()
        } else {
            OpticalSlab::transparent()
        };

        let mut partials = vec![PartialStack {
            tools_inner_to_outer: Vec::with_capacity(m_layers),
            slab: initial_slab,
        }];

        const MAX_PARTIAL_STATES: usize = 1536;
        for _depth in 0..m_layers {
            let mut next_partials = Vec::with_capacity(partials.len() * models.len());
            for p in &partials {
                for (f_idx, f_slab) in single_slabs.iter().enumerate() {
                    let mut seq = p.tools_inner_to_outer.clone();
                    seq.push(f_idx);
                    // Prepend `f_slab` on the viewer-facing side of `p.slab`
                    let combined = f_slab.stack_over(&p.slab);
                    next_partials.push(PartialStack {
                        tools_inner_to_outer: seq,
                        slab: combined,
                    });
                }
            }

            if next_partials.len() > MAX_PARTIAL_STATES {
                // Keep diverse representatives across a 12x12x12 Oklab lattice (preferring fewer transitions)
                let mut bin_best: std::collections::HashMap<(i32, i32, i32), PartialStack> =
                    std::collections::HashMap::with_capacity(MAX_PARTIAL_STATES);
                for cand in next_partials {
                    let lin = cand.slab.evaluate_lighting(&lighting);
                    let lab = linear_rgb_to_oklab(lin);
                    let key = (
                        (lab[0] * 16.0).round() as i32,
                        ((lab[1] + 0.4) * 20.0).round() as i32,
                        ((lab[2] + 0.4) * 20.0).round() as i32,
                    );
                    let cand_trans = count_transitions(&cand.tools_inner_to_outer);
                    match bin_best.entry(key) {
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert(cand);
                        }
                        std::collections::hash_map::Entry::Occupied(mut o) => {
                            if cand_trans < count_transitions(&o.get().tools_inner_to_outer) {
                                o.insert(cand);
                            }
                        }
                    }
                }
                partials = bin_best.into_values().collect();
            } else {
                partials = next_partials;
            }
        }

        let mut stacks = Vec::with_capacity(partials.len());
        for mut p in partials {
            p.tools_inner_to_outer.reverse(); // Now ordered outermost (0) -> innermost (M - 1)
            let transitions = count_transitions(&p.tools_inner_to_outer);
            let linear_rgb = p.slab.evaluate_lighting(&lighting);
            let oklab = linear_rgb_to_oklab(linear_rgb);
            let spread = if p.tools_inner_to_outer.is_empty() {
                0.0
            } else {
                let sum_dev: f64 = p
                    .tools_inner_to_outer
                    .iter()
                    .map(|&fi| oklab_delta_e(model_oklabs[fi], oklab))
                    .sum();
                sum_dev / (p.tools_inner_to_outer.len() as f64)
            };
            stacks.push(CandidateOpticalStack {
                tools_outer_to_inner: p.tools_inner_to_outer,
                linear_rgb,
                oklab,
                transition_count: transitions,
                shell_color_spread: spread,
            });
        }

        Self {
            models,
            tool_indices,
            backing_model_idx,
            shell_layers: m_layers,
            nominal_slab_mm: slab_d,
            lighting,
            stacks,
        }
    }

    /// Evaluates a candidate stack at a specific local physical slab thickness `local_slab_mm`
    /// (accounting for variable layer height and non-planar conformal vs wall angles).
    #[inline]
    pub fn evaluate_stack_oklab_at_thickness(&self, stack_idx: usize, local_slab_mm: f64) -> [f64; 3] {
        let stack = &self.stacks[stack_idx];
        if (local_slab_mm - self.nominal_slab_mm).abs() <= 0.015 {
            return stack.oklab;
        }
        let d = local_slab_mm.clamp(0.03, 3.0);
        let mut comp = OpticalSlab::transparent();
        for &m_idx in &stack.tools_outer_to_inner {
            comp = comp.stack_over(&self.models[m_idx].slab_response(d));
        }
        if let Some(b_idx) = self.backing_model_idx {
            comp = comp.stack_over(&self.models[b_idx].opaque_backing_slab());
        }
        linear_rgb_to_oklab(comp.evaluate_lighting(&self.lighting))
    }

    /// Finds the optimal `M`-layer optical stack for `target_oklab` at `local_slab_mm`,
    /// optionally applying a gentle hysteresis `switch_penalty` if the filament at `active_depth`
    /// differs from `preferred_model_idx`.
    pub fn find_best_stack(
        &self,
        target_oklab: [f64; 3],
        local_slab_mm: f64,
        active_depth: usize,
        preferred_model_idx: Option<usize>,
        switch_penalty: f64,
    ) -> usize {
        if self.stacks.is_empty() {
            return 0;
        }

        // Fast pass over nominal Oklab coordinates to shortlist top 12 candidates,
        // then refine with exact `local_slab_mm` Kubelka-Munk evaluation.
        let use_local_refinement = (local_slab_mm - self.nominal_slab_mm).abs() > 0.015;
        let depth_clamped = active_depth.min(self.shell_layers.saturating_sub(1));

        if !use_local_refinement {
            let mut best_idx = 0usize;
            let mut best_cost = f64::INFINITY;
            for (i, st) in self.stacks.iter().enumerate() {
                let mut cost = oklab_delta_e(st.oklab, target_oklab)
                    + 0.0020 * (st.transition_count as f64)
                    + 0.0150 * st.shell_color_spread;
                if let Some(pref) = preferred_model_idx {
                    if st.tools_outer_to_inner[depth_clamped] != pref {
                        cost += switch_penalty;
                    }
                }
                if cost < best_cost {
                    best_cost = cost;
                    best_idx = i;
                }
            }
            return best_idx;
        }

        const K_SHORTLIST: usize = 16;
        let mut top_k: [(usize, f64); K_SHORTLIST] = [(0, f64::INFINITY); K_SHORTLIST];
        for (i, st) in self.stacks.iter().enumerate() {
            let approx_d = oklab_distance_sq(st.oklab, target_oklab)
                + 0.0004 * st.shell_color_spread;
            if approx_d < top_k[K_SHORTLIST - 1].1 {
                top_k[K_SHORTLIST - 1] = (i, approx_d);
                let mut k = K_SHORTLIST - 1;
                while k > 0 && top_k[k].1 < top_k[k - 1].1 {
                    top_k.swap(k, k - 1);
                    k -= 1;
                }
            }
        }

        let mut best_idx = top_k[0].0;
        let mut best_cost = f64::INFINITY;
        for &(cand_idx, approx_d) in &top_k {
            if !approx_d.is_finite() {
                continue;
            }
            let exact_lab = self.evaluate_stack_oklab_at_thickness(cand_idx, local_slab_mm);
            let st = &self.stacks[cand_idx];
            let mut cost = oklab_delta_e(exact_lab, target_oklab)
                + 0.0020 * (st.transition_count as f64)
                + 0.0150 * st.shell_color_spread;
            if let Some(pref) = preferred_model_idx {
                if st.tools_outer_to_inner[depth_clamped] != pref {
                    cost += switch_penalty;
                }
            }
            if cost < best_cost {
                best_cost = cost;
                best_idx = cand_idx;
            }
        }
        best_idx
    }

    /// Selects a pair of complementary stacks `(stack_a, stack_b, ratio_b_num_6)` whose spatial
    /// average `(1 - w) * lab(A) + w * lab(B)` ($w \in \{0, \frac{1}{3}, \frac{1}{2}, \frac{2}{3}\}$)
    /// best matches `target_oklab` for adjacent-layer dithering.
    pub fn find_adjacent_layer_pair(
        &self,
        target_oklab: [f64; 3],
        local_slab_mm: f64,
        dither_strength: f64,
    ) -> (usize, usize, u8) {
        let primary = self.find_best_stack(target_oklab, local_slab_mm, 0, None, 0.0);
        if self.stacks.len() < 2 || dither_strength <= 1.0e-3 {
            return (primary, primary, 0);
        }
        let lab_a = self.evaluate_stack_oklab_at_thickness(primary, local_slab_mm);
        let base_err = oklab_delta_e(lab_a, target_oklab);
        if base_err <= 0.012 {
            return (primary, primary, 0);
        }

        // Aim past `target_oklab` in the direction of the residual to find an opposing gamut neighbor.
        let aim_lab = [
            target_oklab[0] + (target_oklab[0] - lab_a[0]) * dither_strength,
            target_oklab[1] + (target_oklab[1] - lab_a[1]) * dither_strength,
            target_oklab[2] + (target_oklab[2] - lab_a[2]) * dither_strength,
        ];
        let partner = self.find_best_stack(aim_lab, local_slab_mm, 0, None, 0.0);
        if partner == primary {
            return (primary, primary, 0);
        }
        let lab_b = self.evaluate_stack_oklab_at_thickness(partner, local_slab_mm);

        // Test clean rational layer ratios w = num / 6 for num in {2, 3} (1/3 or 1/2).
        let mut best_num = 0u8;
        let mut best_err = base_err;
        for num in [2u8, 3u8] {
            let w = (num as f64) / 6.0;
            let mix = [
                (1.0 - w) * lab_a[0] + w * lab_b[0],
                (1.0 - w) * lab_a[1] + w * lab_b[1],
                (1.0 - w) * lab_a[2] + w * lab_b[2],
            ];
            // Slight penalty against high-contrast flicker between very dark and very bright filaments
            let flicker_pen = 0.12 * (lab_a[0] - lab_b[0]).abs();
            let err = oklab_delta_e(mix, target_oklab) + flicker_pen * (1.0 - 0.5 * dither_strength);
            if err + 0.004 < best_err {
                best_err = err;
                best_num = num;
            }
        }
        if best_num == 0 {
            (primary, primary, 0)
        } else {
            (primary, partner, best_num)
        }
    }
}

#[inline]
fn count_transitions(seq: &[usize]) -> u8 {
    let mut t = 0u8;
    for w in seq.windows(2) {
        if w[0] != w[1] {
            t = t.saturating_add(1);
        }
    }
    t
}

/// Automatically picks the highest-reflectance neutral/white filament as the internal core backing
/// (scoring $L_{\text{oklab}} - 0.35 C_{\text{oklab}}$ so White beats bright Yellow, while still
/// picking the brightest available color when White is not loaded).
fn select_auto_backing_model_index(models: &[FilamentOpticalModel]) -> usize {
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (i, m) in models.iter().enumerate() {
        let lab = linear_rgb_to_oklab(m.r_inf_linear);
        let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
        let score = lab[0] - 0.35 * chroma;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

// ============================================================================
// Non-Planar Kinematic-Invariant Toolpath Optical Color Assignment & Subdivision
// ============================================================================

/// Computes the directional physical bead thickness (mm) along the outward surface normal
/// $\hat{\mathbf{n}}_{\text{surf}}$ using the point's local `nozzle_axis` $\hat{\mathbf{z}}_{\text{nozzle}}$
/// (`(pt.nx, pt.ny, pt.nz)` — **never** assuming nozzle = `+Z`, per `AGENTS.md` Rule 1) and
/// local variable `width_mm` / `height_mm` (`AGENTS.md` Rule 3).
#[inline]
pub fn directional_slab_thickness_mm(
    pt: &ToolpathPoint,
    surf_normal_unit: [f32; 3],
) -> (f32, f32) {
    let w = pt.width_mm.max(0.08);
    let h = pt.height_mm.max(0.03);
    let n_len = (pt.nx * pt.nx + pt.ny * pt.ny + pt.nz * pt.nz).sqrt();
    let (znx, zny, znz) = if n_len > 1.0e-5 {
        (pt.nx / n_len, pt.ny / n_len, pt.nz / n_len)
    } else {
        (surf_normal_unit[0], surf_normal_unit[1], surf_normal_unit[2])
    };
    let cos_theta = (surf_normal_unit[0] * znx
        + surf_normal_unit[1] * zny
        + surf_normal_unit[2] * znz)
        .abs()
        .clamp(0.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let denom = ((w * cos_theta).powi(2) + (h * sin_theta).powi(2))
        .sqrt()
        .max(1.0e-4);
    ((w * h) / denom, cos_theta)
}

#[inline]
fn normalize_surf_normal(sample: &SurfaceOpticalSample, pt: &ToolpathPoint) -> [f32; 3] {
    let len = (sample.surf_nx * sample.surf_nx
        + sample.surf_ny * sample.surf_ny
        + sample.surf_nz * sample.surf_nz)
        .sqrt();
    if len > 1.0e-5 {
        [
            sample.surf_nx / len,
            sample.surf_ny / len,
            sample.surf_nz / len,
        ]
    } else {
        let n_len = (pt.nx * pt.nx + pt.ny * pt.ny + pt.nz * pt.nz).sqrt();
        if n_len > 1.0e-5 {
            [pt.nx / n_len, pt.ny / n_len, pt.nz / n_len]
        } else {
            [1.0, 0.0, 0.0]
        }
    }
}

#[inline]
fn interpolate_point(a: &ToolpathPoint, b: &ToolpathPoint, t: f32) -> ToolpathPoint {
    let u = t.clamp(0.0, 1.0);
    let nx_raw = a.nx + u * (b.nx - a.nx);
    let ny_raw = a.ny + u * (b.ny - a.ny);
    let nz_raw = a.nz + u * (b.nz - a.nz);
    let n_len = (nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw)
        .sqrt()
        .max(1.0e-6);
    ToolpathPoint {
        x: a.x + u * (b.x - a.x),
        y: a.y + u * (b.y - a.y),
        z: a.z + u * (b.z - a.z),
        nx: nx_raw / n_len,
        ny: ny_raw / n_len,
        nz: nz_raw / n_len,
        width_mm: a.width_mm + u * (b.width_mm - a.width_mm),
        height_mm: a.height_mm + u * (b.height_mm - a.height_mm),
        tool_iso_angle_deg: a.tool_iso_angle_deg + u * (b.tool_iso_angle_deg - a.tool_iso_angle_deg),
        uses_surface_normal: if u < 0.5 {
            a.uses_surface_normal
        } else {
            b.uses_surface_normal
        },
        sdf_mm: a.sdf_mm + u * (b.sdf_mm - a.sdf_mm),
    }
}

#[inline]
fn dist3(a: &ToolpathPoint, b: &ToolpathPoint) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dz = b.z - a.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Deterministic spatial hash of a surface footpoint (quantized at `1.2 mm` macro-cells so
/// outer wall `m=0` and inner walls `m=1..M-1` share the exact same phase!).
#[inline]
fn footpoint_phase_hash(footpoint_xyz: [f32; 3], layer_index: u32) -> u32 {
    let qx = (footpoint_xyz[0] / 1.25).round() as i32 as u32;
    let qy = (footpoint_xyz[1] / 1.25).round() as i32 as u32;
    let qz = (footpoint_xyz[2] / 1.25).round() as i32 as u32;
    let mut h = qx
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(qy.wrapping_mul(0x85EB_CA77))
        .wrapping_add(qz.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h % 6).wrapping_add(layer_index) % 6
}

/// Resolves the target surface `[R, G, B]` sRGB (`[f32; 3]`) for a query sample.
#[inline]
fn resolve_sample_target_srgb(
    sample: &SurfaceOpticalSample,
    path_filament_slot: u8,
    imported_slot_colors_srgba: &[[f32; 4]],
) -> Option<[f32; 3]> {
    if sample.rgba_packed != 0 {
        let [r, g, b, a] = unpack_rgba8(sample.rgba_packed);
        if a > 0 {
            return Some([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
        }
    }
    let slot_idx = sample.filament_slot as usize;
    if let Some(c) = imported_slot_colors_srgba.get(slot_idx) {
        return Some([c[0], c[1], c[2]]);
    }
    if decode_direct_tool_slot(path_filament_slot).is_none() {
        if let Some(c) = imported_slot_colors_srgba.get(path_filament_slot as usize) {
            return Some([c[0], c[1], c[2]]);
        }
    }
    None
}

#[derive(Clone, Copy)]
struct PreparedSubSeg {
    p0: ToolpathPoint,
    p1: ToolpathPoint,
    len_mm: f32,
    sample_idx: usize,
}

/// Applies Kubelka-Munk optical color approximation across layers and/or short extrusions
/// to an `ObjectToolpathsResult`, invoking `sample_surface_optics_fn(flat_xyz)` once in batch
/// to query SDF depth, outward surface normal, and nearest-surface RGBA / filament slot.
pub fn apply_optical_color_mixing_to_toolpaths<F>(
    raw_toolpaths: &ObjectToolpathsResult,
    mix_cfg: &ColorMixingConfig,
    tools: &[ToolOpticalInput],
    imported_slot_colors_srgba: &[[f32; 4]],
    nominal_layer_height_mm: f32,
    support_tool_index: Option<usize>,
    mut sample_surface_optics_fn: F,
) -> ObjectToolpathsResult
where
    F: FnMut(&[f32]) -> Vec<SurfaceOpticalSample>,
{
    if !raw_toolpaths.valid || !mix_cfg.enabled || tools.len() < 2 {
        return raw_toolpaths.clone();
    }

    let lut = OpticalPaletteLut::build(
        tools,
        mix_cfg.optical_shell_layers,
        nominal_layer_height_mm.max(0.08),
        mix_cfg.lighting_mode,
        mix_cfg.backing_tool_index,
    );
    if lut.stacks.is_empty() {
        return raw_toolpaths.clone();
    }

    let backing_tool_slot = lut
        .backing_model_idx
        .map(|mi| encode_direct_tool_slot(lut.tool_indices[mi]))
        .unwrap_or_else(|| encode_direct_tool_slot(lut.tool_indices[0]));

    let min_seg_mm = mix_cfg.min_short_extrusion_mm.clamp(0.8, 50.0);
    let allow_short = mix_cfg.use_short_extrusions;

    // 1. Sub-discretize eligible shell/skin paths so long straight segments (> min_seg_mm)
    //    have sample midpoints ready for short-extrusion color transitions, and collect all
    //    midpoint coordinates for a single vectorized C++ OpenVDB batch query.
    enum PreparedPath {
        Passthrough(ExtrusionPath),
        CoreFixedTool(ExtrusionPath, u8),
        OpticalShell {
            template: ExtrusionPath,
            sub_segs: Vec<PreparedSubSeg>,
        },
    }

    let mut flat_query_xyz: Vec<f32> = Vec::new();
    let mut prepared_layers: Vec<(u32, u32, f32, Vec<PreparedPath>)> =
        Vec::with_capacity(raw_toolpaths.layers.len());

    for layer in &raw_toolpaths.layers {
        let mut prep_paths = Vec::with_capacity(layer.paths.len());
        for path in &layer.paths {
            if path.points.len() < 2 || decode_direct_tool_slot(path.filament_slot).is_some() {
                prep_paths.push(PreparedPath::Passthrough(path.clone()));
                continue;
            }
            // Supports (7, 8): keep on configured support tool (or passthrough)
            if path.feature_kind == 7 || path.feature_kind == 8 {
                if let Some(sup_t) = support_tool_index {
                    prep_paths.push(PreparedPath::CoreFixedTool(
                        path.clone(),
                        encode_direct_tool_slot(sup_t),
                    ));
                } else {
                    prep_paths.push(PreparedPath::Passthrough(path.clone()));
                }
                continue;
            }
            // SparseInfill (4): always print in the internal backing filament (e.g. White)
            // so translucent outer shells reflect cleanly off a uniform bright core with zero extra toolchanges!
            // All 3D shell features (OuterWall=0, InnerWall=1, SolidInfill=2, GapFill=3, Bridge=5, WaveOverhang=6)
            // carry colour through the entire 3D shell.
            if path.feature_kind == 4 {
                prep_paths.push(PreparedPath::CoreFixedTool(path.clone(), backing_tool_slot));
                continue;
            }

            let mut sub_segs = Vec::new();
            let num_raw_edges = if path.is_closed {
                let d_close = dist3(path.points.last().unwrap(), &path.points[0]);
                if d_close > 1.0e-3 {
                    path.points.len()
                } else {
                    path.points.len() - 1
                }
            } else {
                path.points.len() - 1
            };

            let step_target = if allow_short {
                (0.75 * min_seg_mm).max(0.8)
            } else {
                25.0
            };

            for edge_i in 0..num_raw_edges {
                let a = &path.points[edge_i];
                let b = &path.points[(edge_i + 1) % path.points.len()];
                let edge_len = dist3(a, b);
                if edge_len <= 1.0e-5 {
                    continue;
                }
                let n_div = if allow_short && edge_len > 1.35 * step_target {
                    ((edge_len / step_target).ceil() as usize).clamp(1, 256)
                } else {
                    1
                };
                for k in 0..n_div {
                    let t0 = (k as f32) / (n_div as f32);
                    let t1 = ((k + 1) as f32) / (n_div as f32);
                    let p0 = if k == 0 {
                        *a
                    } else {
                        interpolate_point(a, b, t0)
                    };
                    let p1 = if k + 1 == n_div {
                        *b
                    } else {
                        interpolate_point(a, b, t1)
                    };
                    let seg_len = edge_len / (n_div as f32);
                    let mid = interpolate_point(&p0, &p1, 0.5);
                    let sample_idx = flat_query_xyz.len() / 3;
                    flat_query_xyz.push(mid.x);
                    flat_query_xyz.push(mid.y);
                    flat_query_xyz.push(mid.z);
                    sub_segs.push(PreparedSubSeg {
                        p0,
                        p1,
                        len_mm: seg_len,
                        sample_idx,
                    });
                }
            }

            if sub_segs.is_empty() {
                prep_paths.push(PreparedPath::Passthrough(path.clone()));
            } else {
                prep_paths.push(PreparedPath::OpticalShell {
                    template: path.clone(),
                    sub_segs,
                });
            }
        }
        prepared_layers.push((
            layer.layer_index,
            layer.branch_id,
            layer.isovalue,
            prep_paths,
        ));
    }

    let samples = if flat_query_xyz.is_empty() {
        Vec::new()
    } else {
        sample_surface_optics_fn(&flat_query_xyz)
    };

    // 2. Assign filaments to every path / sub-segment and rebuild clean toolpath geometry.
    let mut out = ObjectToolpathsResult {
        valid: true,
        total_path_count: 0,
        outer_wall_count: 0,
        inner_wall_count: 0,
        solid_infill_count: 0,
        sparse_infill_count: 0,
        bridge_count: 0,
        wave_overhang_count: 0,
        support_count: 0,
        support_interface_count: 0,
        layers: Vec::with_capacity(prepared_layers.len()),
    };

    let dither_gain = (mix_cfg.dither_strength as f64).clamp(0.0, 1.0);
    let switch_pen = (mix_cfg.tool_switch_penalty as f64).clamp(0.0, 0.25);
    let foot_cell_mm = (0.65 * min_seg_mm).clamp(0.75, 4.0);
    let quant_foot = |foot: [f32; 3]| -> (i32, i32, i32) {
        (
            (foot[0] / foot_cell_mm).round() as i32,
            (foot[1] / foot_cell_mm).round() as i32,
            (foot[2] / foot_cell_mm).round() as i32,
        )
    };

    for (layer_index, branch_id, isovalue, prep_paths) in prepared_layers {
        let mut out_paths: Vec<ExtrusionPath> = Vec::with_capacity(prep_paths.len());
        // Cache of `(footpoint_cell -> chosen_stack_idx)` populated by OuterWall (`feature_kind == 0`)
        // segments on this layer so that InnerWall (`1`) and SolidInfill (`2`) segments directly behind
        // them along `-n_surf` execute the exact same M-slab optical stack through the whole shell.
        let mut layer_footpoint_stacks: std::collections::HashMap<(i32, i32, i32), usize> =
            std::collections::HashMap::new();

        // Pre-pass: solve and cache `OuterWall` (`feature_kind == 0`) footpoint stacks first
        // even when `wall_print_order` places inner walls earlier in `prep_paths`.
        if allow_short {
            for prep in &prep_paths {
                let PreparedPath::OpticalShell { template, sub_segs } = prep else {
                    continue;
                };
                if template.feature_kind != 0 {
                    continue;
                }
                let mut err_lab = [0.0_f64; 3];
                let mut prev_target_lab: Option<[f64; 3]> = None;
                let mut prev_model_idx: Option<usize> = None;

                for seg in sub_segs {
                    let sample = samples.get(seg.sample_idx).copied().unwrap_or_default();
                    let mid = interpolate_point(&seg.p0, &seg.p1, 0.5);
                    let n_surf = normalize_surf_normal(&sample, &mid);
                    let (t_slab_mm, cos_theta) = directional_slab_thickness_mm(&mid, n_surf);
                    let Some(target_srgb) = resolve_sample_target_srgb(
                        &sample,
                        template.filament_slot,
                        imported_slot_colors_srgba,
                    ) else {
                        continue;
                    };
                    let target_lab = srgb_to_oklab(target_srgb);
                    if let Some(prev_t) = prev_target_lab {
                        if oklab_delta_e(prev_t, target_lab) > 0.10 {
                            err_lab = [0.0; 3];
                        }
                    }
                    prev_target_lab = Some(target_lab);

                    let foot = [
                        mid.x - sample.sdf_mm * n_surf[0],
                        mid.y - sample.sdf_mm * n_surf[1],
                        mid.z - sample.sdf_mm * n_surf[2],
                    ];
                    let chosen_stack_idx = select_optical_stack_for_sample(
                        &lut,
                        target_lab,
                        t_slab_mm as f64,
                        cos_theta,
                        0,
                        foot,
                        layer_index,
                        mix_cfg.use_adjacent_layers,
                        dither_gain,
                        err_lab,
                        prev_model_idx,
                        switch_pen,
                    );
                    let chosen_lab =
                        lut.evaluate_stack_oklab_at_thickness(chosen_stack_idx, t_slab_mm as f64);
                    prev_model_idx = Some(lut.stacks[chosen_stack_idx].tools_outer_to_inner[0]);
                    for c in 0..3 {
                        let raw_e = (target_lab[c] + dither_gain * err_lab[c]) - chosen_lab[c];
                        err_lab[c] = (0.85 * raw_e).clamp(-0.16, 0.16);
                    }
                    layer_footpoint_stacks.insert(quant_foot(foot), chosen_stack_idx);
                }
            }
        }

        for prep in prep_paths {
            match prep {
                PreparedPath::Passthrough(p) => {
                    increment_feature_counter(&mut out, p.feature_kind);
                    out_paths.push(p);
                }
                PreparedPath::CoreFixedTool(mut p, slot) => {
                    p.filament_slot = slot;
                    increment_feature_counter(&mut out, p.feature_kind);
                    out_paths.push(p);
                }
                PreparedPath::OpticalShell { template, sub_segs } => {
                    if !allow_short {
                        // Whole-path optical assignment (layers + walls, without splitting along paths)
                        let whole_slot = choose_whole_path_tool_slot(
                            &template,
                            &sub_segs,
                            &samples,
                            &lut,
                            imported_slot_colors_srgba,
                            layer_index,
                            mix_cfg.use_adjacent_layers,
                            dither_gain,
                            backing_tool_slot,
                        );
                        let mut p = template;
                        p.filament_slot = whole_slot;
                        increment_feature_counter(&mut out, p.feature_kind);
                        out_paths.push(p);
                        continue;
                    }

                    // Short-extrusion along-path optical solver with Oklab error diffusion
                    // and radial surface-footpoint locking across the entire 3D shell.
                    let mut seg_model_choices: Vec<usize> = Vec::with_capacity(sub_segs.len());
                    let mut err_lab = [0.0_f64; 3];
                    let mut prev_target_lab: Option<[f64; 3]> = None;
                    let mut prev_model_idx: Option<usize> = None;

                    for seg in &sub_segs {
                        let sample = samples
                            .get(seg.sample_idx)
                            .copied()
                            .unwrap_or_default();
                        let mid = interpolate_point(&seg.p0, &seg.p1, 0.5);
                        let n_surf = normalize_surf_normal(&sample, &mid);
                        let (t_slab_mm, cos_theta) = directional_slab_thickness_mm(&mid, n_surf);

                        // Determine depth index `m` inside the 3D shell (`0` = outer surface) and
                        // clamp to `0..lut.shell_layers - 1` so deeper inner walls and SolidInfill
                        // carry the innermost shell pigment all the way to `-shell_thickness_mm`.
                        let raw_depth_m = compute_shell_depth_index(
                            template.feature_kind,
                            template.wall_index,
                            sample.sdf_mm,
                            t_slab_mm,
                            cos_theta,
                        );
                        let depth_m = raw_depth_m.min(lut.shell_layers.saturating_sub(1));

                        let Some(target_srgb) = resolve_sample_target_srgb(
                            &sample,
                            template.filament_slot,
                            imported_slot_colors_srgba,
                        ) else {
                            let fallback_m = prev_model_idx
                                .or(lut.backing_model_idx)
                                .unwrap_or(0);
                            seg_model_choices.push(fallback_m);
                            continue;
                        };

                        let target_lab = srgb_to_oklab(target_srgb);
                        // Reset error diffusion across crisp painted color boundaries so edges never bleed
                        if let Some(prev_t) = prev_target_lab {
                            if oklab_delta_e(prev_t, target_lab) > 0.10 {
                                err_lab = [0.0; 3];
                            }
                        }
                        prev_target_lab = Some(target_lab);

                        let foot = [
                            mid.x - sample.sdf_mm * n_surf[0],
                            mid.y - sample.sdf_mm * n_surf[1],
                            mid.z - sample.sdf_mm * n_surf[2],
                        ];
                        let qf = quant_foot(foot);

                        // For inner walls and solid shell infill (`depth_m > 0`), check if the outer wall
                        // directly in front of this surface footpoint already locked an optical stack.
                        let cached_outer_stack = if template.feature_kind != 0 {
                            layer_footpoint_stacks.get(&qf).copied().or_else(|| {
                                for &(dx, dy, dz) in &[
                                    (1, 0, 0),
                                    (-1, 0, 0),
                                    (0, 1, 0),
                                    (0, -1, 0),
                                    (0, 0, 1),
                                    (0, 0, -1),
                                ] {
                                    if let Some(&st) =
                                        layer_footpoint_stacks.get(&(qf.0 + dx, qf.1 + dy, qf.2 + dz))
                                    {
                                        return Some(st);
                                    }
                                }
                                None
                            })
                        } else {
                            layer_footpoint_stacks.get(&qf).copied()
                        };

                        let chosen_stack_idx = if let Some(st_idx) = cached_outer_stack {
                            st_idx
                        } else {
                            select_optical_stack_for_sample(
                                &lut,
                                target_lab,
                                t_slab_mm as f64,
                                cos_theta,
                                depth_m,
                                foot,
                                layer_index,
                                mix_cfg.use_adjacent_layers,
                                dither_gain,
                                err_lab,
                                prev_model_idx,
                                switch_pen,
                            )
                        };

                        let chosen_lab = lut.evaluate_stack_oklab_at_thickness(
                            chosen_stack_idx,
                            t_slab_mm as f64,
                        );
                        let m_choice = lut.stacks[chosen_stack_idx].tools_outer_to_inner[depth_m];
                        seg_model_choices.push(m_choice);
                        prev_model_idx = Some(m_choice);

                        // Propagate bounded quantization error along the contour
                        for c in 0..3 {
                            let raw_e = (target_lab[c] + dither_gain * err_lab[c]) - chosen_lab[c];
                            err_lab[c] = (0.85 * raw_e).clamp(-0.16, 0.16);
                        }
                    }

                    // Enforce `min_short_extrusion_mm` by merging micro-runs shorter than `min_seg_mm`
                    // into their adjacent run, and emit clean contiguous `ExtrusionPath`s.
                    let emitted = emit_segmented_extrusion_paths(
                        &template,
                        &sub_segs,
                        &seg_model_choices,
                        &lut.tool_indices,
                        min_seg_mm,
                    );
                    for ep in emitted {
                        increment_feature_counter(&mut out, ep.feature_kind);
                        out_paths.push(ep);
                    }
                }
            }
        }

        out.layers.push(crate::bridge::ffi::LayerToolpaths {
            layer_index,
            branch_id,
            isovalue,
            paths: out_paths,
        });
    }

    out.total_path_count = out
        .layers
        .iter()
        .map(|l| l.paths.len() as u32)
        .sum();
    out
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn select_optical_stack_for_sample(
    lut: &OpticalPaletteLut,
    target_lab: [f64; 3],
    t_slab_mm: f64,
    cos_theta: f32,
    depth_m: usize,
    foot: [f32; 3],
    layer_index: u32,
    use_adjacent_layers: bool,
    dither_gain: f64,
    err_lab: [f64; 3],
    prev_model_idx: Option<usize>,
    switch_pen: f64,
) -> usize {
    if use_adjacent_layers && cos_theta < 0.85 {
        let (st_a, st_b, ratio_num_6) =
            lut.find_adjacent_layer_pair(target_lab, t_slab_mm, dither_gain);
        if ratio_num_6 > 0 && st_a != st_b {
            let phase = footpoint_phase_hash(foot, layer_index);
            let use_b = match ratio_num_6 {
                2 => phase == 1 || phase == 4,
                3 => (phase & 1) == 1,
                _ => false,
            };
            let layer_stack = if use_b { st_b } else { st_a };
            let layer_lab = lut.evaluate_stack_oklab_at_thickness(layer_stack, t_slab_mm);
            let avg_w = (ratio_num_6 as f64) / 6.0;
            let pair_avg_lab = [
                (1.0 - avg_w) * lut.stacks[st_a].oklab[0] + avg_w * lut.stacks[st_b].oklab[0],
                (1.0 - avg_w) * lut.stacks[st_a].oklab[1] + avg_w * lut.stacks[st_b].oklab[1],
                (1.0 - avg_w) * lut.stacks[st_a].oklab[2] + avg_w * lut.stacks[st_b].oklab[2],
            ];
            let desired_lab = [
                (layer_lab[0] + (target_lab[0] - pair_avg_lab[0]) + dither_gain * err_lab[0])
                    .clamp(0.0, 1.0),
                (layer_lab[1] + (target_lab[1] - pair_avg_lab[1]) + dither_gain * err_lab[1])
                    .clamp(-0.5, 0.5),
                (layer_lab[2] + (target_lab[2] - pair_avg_lab[2]) + dither_gain * err_lab[2])
                    .clamp(-0.5, 0.5),
            ];
            return lut.find_best_stack(
                desired_lab,
                t_slab_mm,
                depth_m,
                prev_model_idx,
                switch_pen,
            );
        }
    }
    let desired_lab = [
        (target_lab[0] + dither_gain * err_lab[0]).clamp(0.0, 1.0),
        (target_lab[1] + dither_gain * err_lab[1]).clamp(-0.5, 0.5),
        (target_lab[2] + dither_gain * err_lab[2]).clamp(-0.5, 0.5),
    ];
    lut.find_best_stack(desired_lab, t_slab_mm, depth_m, prev_model_idx, switch_pen)
}

#[inline]
fn increment_feature_counter(res: &mut ObjectToolpathsResult, feature_kind: u8) {
    match feature_kind {
        0 => res.outer_wall_count += 1,
        1 => res.inner_wall_count += 1,
        2 | 3 => res.solid_infill_count += 1,
        4 => res.sparse_infill_count += 1,
        5 => res.bridge_count += 1,
        6 => res.wave_overhang_count += 1,
        7 => res.support_count += 1,
        8 => res.support_interface_count += 1,
        _ => {}
    }
}

#[inline]
fn compute_shell_depth_index(
    feature_kind: u8,
    wall_index: u32,
    sdf_mm: f32,
    t_slab_mm: f32,
    cos_theta: f32,
) -> usize {
    let inward_depth_mm = (-sdf_mm).max(0.0);
    // On steep/orthogonal walls (cos_theta < 0.55), OuterWall (0) and InnerWall (1..=W)
    // form discrete concentric optical slabs directly indexed by `wall_index`, EXCEPT when
    // an outer wall is buried beneath a conformal skin cap (`inward_depth_mm` much deeper).
    if cos_theta < 0.55 && (feature_kind == 0 || feature_kind == 1) {
        let w_idx = if feature_kind == 0 {
            0usize
        } else {
            wall_index as usize
        };
        let sdf_depth_idx = ((inward_depth_mm - 0.55 * t_slab_mm).max(0.0) / t_slab_mm.max(0.05))
            .floor() as usize;
        return w_idx.max(sdf_depth_idx);
    }
    ((inward_depth_mm - 0.45 * t_slab_mm).max(0.0) / t_slab_mm.max(0.05)).floor() as usize
}

fn choose_whole_path_tool_slot(
    template: &ExtrusionPath,
    sub_segs: &[PreparedSubSeg],
    samples: &[SurfaceOpticalSample],
    lut: &OpticalPaletteLut,
    imported_slot_colors_srgba: &[[f32; 4]],
    layer_index: u32,
    use_adjacent_layers: bool,
    dither_gain: f64,
    backing_tool_slot: u8,
) -> u8 {
    let mut sum_lab = [0.0_f64; 3];
    let mut sum_slab = 0.0_f64;
    let mut sum_cos = 0.0_f64;
    let mut sum_sdf = 0.0_f32;
    let mut sum_foot = [0.0_f32; 3];
    let mut total_w = 0.0_f64;

    for seg in sub_segs {
        let Some(sample) = samples.get(seg.sample_idx) else {
            continue;
        };
        let mid = interpolate_point(&seg.p0, &seg.p1, 0.5);
        let n_surf = normalize_surf_normal(sample, &mid);
        let (t_slab, cos_t) = directional_slab_thickness_mm(&mid, n_surf);
        let Some(srgb) =
            resolve_sample_target_srgb(sample, template.filament_slot, imported_slot_colors_srgba)
        else {
            continue;
        };
        let w = (seg.len_mm.max(0.05)) as f64;
        let lab = srgb_to_oklab(srgb);
        sum_lab[0] += w * lab[0];
        sum_lab[1] += w * lab[1];
        sum_lab[2] += w * lab[2];
        sum_slab += w * (t_slab as f64);
        sum_cos += w * (cos_t as f64);
        sum_sdf += (w as f32) * sample.sdf_mm;
        sum_foot[0] += (w as f32) * (mid.x - sample.sdf_mm * n_surf[0]);
        sum_foot[1] += (w as f32) * (mid.y - sample.sdf_mm * n_surf[1]);
        sum_foot[2] += (w as f32) * (mid.z - sample.sdf_mm * n_surf[2]);
        total_w += w;
    }

    if total_w <= 1.0e-6 {
        return backing_tool_slot;
    }
    let avg_lab = [
        sum_lab[0] / total_w,
        sum_lab[1] / total_w,
        sum_lab[2] / total_w,
    ];
    let avg_slab = sum_slab / total_w;
    let avg_cos = (sum_cos / total_w) as f32;
    let avg_sdf = sum_sdf / (total_w as f32);
    let raw_depth_m = compute_shell_depth_index(
        template.feature_kind,
        template.wall_index,
        avg_sdf,
        avg_slab as f32,
        avg_cos,
    );
    let depth_m = raw_depth_m.min(lut.shell_layers.saturating_sub(1));

    let stack_idx = if use_adjacent_layers && avg_cos < 0.85 {
        let (st_a, st_b, ratio_num_6) =
            lut.find_adjacent_layer_pair(avg_lab, avg_slab, dither_gain);
        if ratio_num_6 > 0 && st_a != st_b {
            let avg_foot = [
                sum_foot[0] / (total_w as f32),
                sum_foot[1] / (total_w as f32),
                sum_foot[2] / (total_w as f32),
            ];
            let phase = footpoint_phase_hash(avg_foot, layer_index);
            let use_b = match ratio_num_6 {
                2 => phase == 1 || phase == 4,
                3 => (phase & 1) == 1,
                _ => false,
            };
            if use_b {
                st_b
            } else {
                st_a
            }
        } else {
            st_a
        }
    } else {
        lut.find_best_stack(avg_lab, avg_slab, depth_m, None, 0.0)
    };

    let m_idx = lut.stacks[stack_idx].tools_outer_to_inner[depth_m];
    encode_direct_tool_slot(lut.tool_indices[m_idx])
}

fn emit_segmented_extrusion_paths(
    template: &ExtrusionPath,
    sub_segs: &[PreparedSubSeg],
    seg_model_choices: &[usize],
    tool_indices: &[usize],
    min_seg_mm: f32,
) -> Vec<ExtrusionPath> {
    if sub_segs.is_empty() {
        return Vec::new();
    }

    #[derive(Clone)]
    struct Run {
        model_idx: usize,
        start_seg: usize,
        end_seg: usize, // exclusive
        length_mm: f32,
    }

    // Group contiguous sub-segments with the same filament model index.
    let mut runs: Vec<Run> = Vec::new();
    for (i, seg) in sub_segs.iter().enumerate() {
        let m_idx = seg_model_choices[i];
        if let Some(last) = runs.last_mut() {
            if last.model_idx == m_idx {
                last.end_seg = i + 1;
                last.length_mm += seg.len_mm;
                continue;
            }
        }
        runs.push(Run {
            model_idx: m_idx,
            start_seg: i,
            end_seg: i + 1,
            length_mm: seg.len_mm,
        });
    }

    // If the entire path (open or closed) uses a single filament, keep it completely intact!
    if runs.len() == 1 {
        let mut intact = template.clone();
        intact.filament_slot = encode_direct_tool_slot(tool_indices[runs[0].model_idx]);
        return vec![intact];
    }

    // For closed loops where the first and last run share the same filament, rotate the loop seam
    // to `runs[0].end_seg` so the wrap-around run is a single unbroken extrusion!
    let mut ordered_indices: Vec<usize> = (0..sub_segs.len()).collect();
    let mut ordered_models: Vec<usize> = seg_model_choices.to_vec();
    if template.is_closed
        && runs.len() >= 2
        && runs.first().unwrap().model_idx == runs.last().unwrap().model_idx
    {
        let split_at = runs[0].end_seg;
        ordered_indices.rotate_left(split_at);
        ordered_models.rotate_left(split_at);
    }

    // Rebuild runs on the seam-rotated sequence
    runs.clear();
    for (pos, &orig_idx) in ordered_indices.iter().enumerate() {
        let seg = &sub_segs[orig_idx];
        let m_idx = ordered_models[pos];
        if let Some(last) = runs.last_mut() {
            if last.model_idx == m_idx {
                last.end_seg = pos + 1;
                last.length_mm += seg.len_mm;
                continue;
            }
        }
        runs.push(Run {
            model_idx: m_idx,
            start_seg: pos,
            end_seg: pos + 1,
            length_mm: seg.len_mm,
        });
    }

    // Iteratively absorb any run shorter than `min_seg_mm` into its neighbor so every emitted
    // short extrusion strictly meets `min_short_extrusion_mm` (when total path length allows).
    loop {
        if runs.len() <= 1 {
            break;
        }
        let mut shortest_pos: Option<usize> = None;
        let mut shortest_len = min_seg_mm;
        for (ri, r) in runs.iter().enumerate() {
            if r.length_mm + 1.0e-4 < shortest_len {
                shortest_len = r.length_mm;
                shortest_pos = Some(ri);
            }
        }
        let Some(idx) = shortest_pos else {
            break;
        };

        let merge_into = if idx == 0 {
            1
        } else if idx + 1 == runs.len() {
            idx - 1
        } else if runs[idx - 1].model_idx == runs[idx + 1].model_idx
            || runs[idx - 1].length_mm >= runs[idx + 1].length_mm
        {
            idx - 1
        } else {
            idx + 1
        };

        let removed = runs.remove(idx);
        let target = if merge_into > idx {
            merge_into - 1
        } else {
            merge_into
        };
        runs[target].start_seg = runs[target].start_seg.min(removed.start_seg);
        runs[target].end_seg = runs[target].end_seg.max(removed.end_seg);
        runs[target].length_mm += removed.length_mm;

        // Coalesce newly adjacent identical-filament runs
        let mut coalesced: Vec<Run> = Vec::with_capacity(runs.len());
        for r in runs {
            if let Some(prev) = coalesced.last_mut() {
                if prev.model_idx == r.model_idx {
                    prev.end_seg = r.end_seg;
                    prev.length_mm += r.length_mm;
                    continue;
                }
            }
            coalesced.push(r);
        }
        runs = coalesced;
    }

    if runs.len() == 1 {
        let mut intact = template.clone();
        intact.filament_slot = encode_direct_tool_slot(tool_indices[runs[0].model_idx]);
        return vec![intact];
    }

    let mut out_paths = Vec::with_capacity(runs.len());
    for r in runs {
        let mut pts = Vec::with_capacity(r.end_seg - r.start_seg + 1);
        let mut sum_w = 0.0_f32;
        let mut total_len = 0.0_f32;
        for pos in r.start_seg..r.end_seg {
            let seg = &sub_segs[ordered_indices[pos]];
            if pts.is_empty() {
                pts.push(seg.p0);
            }
            pts.push(seg.p1);
            sum_w += 0.5 * (seg.p0.width_mm + seg.p1.width_mm) * seg.len_mm;
            total_len += seg.len_mm;
        }
        if pts.len() < 2 || total_len <= 1.0e-4 {
            continue;
        }
        let mean_w = if total_len > 1.0e-5 {
            sum_w / total_len
        } else {
            template.mean_width_mm
        };
        out_paths.push(ExtrusionPath {
            feature_kind: template.feature_kind,
            filament_slot: encode_direct_tool_slot(tool_indices[r.model_idx]),
            wall_index: template.wall_index,
            is_closed: false,
            total_length_mm: total_len,
            mean_width_mm: mean_w,
            points: pts,
        });
    }
    out_paths
}

/// Evaluates the simulated Kubelka–Munk composite sRGBA `[r, g, b, 1.0]` for every segment in
/// `planned` (`layers[layer_idx][seg_idx]`), stacking concentric inner walls (`wall_index = 1, 2`)
/// behind outer perimeters and stepping along $-\hat{\mathbf{z}}_{\text{nozzle}}$ (`AGENTS.md` Rule 1)
/// beneath top/conformal skin extrusions under `config.slicing.color_mixing.lighting_mode`.
pub fn compute_planned_optical_composite_colors(
    planned: &crate::gcode::PlannedGcodeResult,
    config: &crate::config::SlicerConfig,
) -> Vec<Vec<[f32; 4]>> {
    use std::collections::HashMap;

    if planned.layers.is_empty() || config.tools.is_empty() {
        return Vec::new();
    }

    let models: Vec<FilamentOpticalModel> = config
        .tools
        .iter()
        .map(|t| {
            FilamentOpticalModel::from_srgb_and_td(
                [
                    t.filament_color_srgba[0],
                    t.filament_color_srgba[1],
                    t.filament_color_srgba[2],
                ],
                t.effective_transmission_distance_mm(),
            )
        })
        .collect();

    let lighting_mode = config.slicing.color_mixing.lighting_mode;
    let lighting = lighting_mode.lighting();
    let backing_idx = if lighting_mode.uses_opaque_core_backing() {
        Some(
            config
                .slicing
                .color_mixing
                .backing_tool_index
                .filter(|&i| i < models.len())
                .unwrap_or_else(|| select_auto_backing_model_index(&models)),
        )
    } else {
        None
    };
    let backing_ref = backing_idx.map(|i| &models[i]);

    const CELL_MM: f32 = 1.6;
    let quant = |p: bevy::math::Vec3| -> (i32, i32, i32) {
        (
            (p.x / CELL_MM).round() as i32,
            (p.y / CELL_MM).round() as i32,
            (p.z / CELL_MM).round() as i32,
        )
    };

    // Index inner walls (tier 1 = wall 1, tier 2 = wall >= 2) and skin/solid infill (tier 3)
    let mut wall_grid: HashMap<(i32, i32, i32, u8), (usize, f32)> = HashMap::new();
    let mut skin_grid: HashMap<(i32, i32, i32), (usize, f32)> = HashMap::new();

    for layer in &planned.layers {
        for seg in &layer.segments {
            if !seg.kind.is_extruding() || seg.kind.is_support() {
                continue;
            }
            let mid = 0.5 * (seg.p0_local + seg.p1_local);
            let t_idx = config
                .tool_index_for_filament_slot(seg.filament_slot)
                .min(models.len() - 1);
            let (qx, qy, qz) = quant(mid);
            match seg.kind {
                crate::gcode::PlannedMoveKind::InnerWall => {
                    let tier = if seg.wall_index <= 1 { 1u8 } else { 2u8 };
                    wall_grid
                        .entry((qx, qy, qz, tier))
                        .or_insert((t_idx, seg.width_mm.max(0.10)));
                }
                crate::gcode::PlannedMoveKind::SolidInfill
                | crate::gcode::PlannedMoveKind::GapFill => {
                    skin_grid
                        .entry((qx, qy, qz))
                        .or_insert((t_idx, seg.height_mm.max(0.05)));
                }
                _ => {}
            }
        }
    }

    let lookup_wall = |p: bevy::math::Vec3, tier: u8| -> Option<(usize, f32)> {
        let (qx, qy, qz) = quant(p);
        if let Some(&hit) = wall_grid.get(&(qx, qy, qz, tier)) {
            return Some(hit);
        }
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(&hit) = wall_grid.get(&(qx + dx, qy + dy, qz + dz, tier)) {
                        return Some(hit);
                    }
                }
            }
        }
        None
    };

    let lookup_skin = |p: bevy::math::Vec3| -> Option<(usize, f32)> {
        let (qx, qy, qz) = quant(p);
        if let Some(&hit) = skin_grid.get(&(qx, qy, qz)) {
            return Some(hit);
        }
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(&hit) = skin_grid.get(&(qx + dx, qy + dy, qz)) {
                    return Some(hit);
                }
            }
        }
        None
    };

    let mut out_layers = Vec::with_capacity(planned.layers.len());
    let mut stack_buf: Vec<(usize, f64)> = Vec::with_capacity(4);

    for layer in &planned.layers {
        let mut layer_cols = Vec::with_capacity(layer.segments.len());
        for seg in &layer.segments {
            if matches!(
                seg.kind,
                crate::gcode::PlannedMoveKind::Travel | crate::gcode::PlannedMoveKind::ZHop
            ) {
                layer_cols.push([0.28, 0.86, 0.98, 1.0]);
                continue;
            }
            if seg.kind.is_support() {
                layer_cols.push([0.22, 0.88, 0.62, 1.0]);
                continue;
            }
            let m0 = config
                .tool_index_for_filament_slot(seg.filament_slot)
                .min(models.len() - 1);
            let mid = 0.5 * (seg.p0_local + seg.p1_local);
            let z_nozzle = (seg.nozzle_local_0 + seg.nozzle_local_1)
                .try_normalize()
                .unwrap_or_else(|| bevy::math::Vec3::from_array(config.machine.nozzle_axis_bed));

            stack_buf.clear();
            match seg.kind {
                crate::gcode::PlannedMoveKind::OuterWall
                | crate::gcode::PlannedMoveKind::ScarfSeam
                | crate::gcode::PlannedMoveKind::WaveOverhang
                | crate::gcode::PlannedMoveKind::Bridge => {
                    stack_buf.push((m0, seg.width_mm.max(0.10) as f64));
                    if let Some((m1, w1)) = lookup_wall(mid, 1) {
                        stack_buf.push((m1, w1 as f64));
                    }
                    if let Some((m2, w2)) = lookup_wall(mid, 2) {
                        stack_buf.push((m2, w2 as f64));
                    }
                }
                crate::gcode::PlannedMoveKind::InnerWall => {
                    stack_buf.push((m0, seg.width_mm.max(0.10) as f64));
                    if seg.wall_index <= 1 {
                        if let Some((m2, w2)) = lookup_wall(mid, 2) {
                            stack_buf.push((m2, w2 as f64));
                        }
                    }
                }
                crate::gcode::PlannedMoveKind::SolidInfill
                | crate::gcode::PlannedMoveKind::GapFill => {
                    let h = seg.height_mm.max(0.05);
                    stack_buf.push((m0, h as f64));
                    // Step inward opposite local `nozzle_axis` (never assuming -Z, AGENTS.md Rule 1)
                    for step in 1..=2 {
                        let probe = mid - z_nozzle * (h * (step as f32) * 1.15);
                        if let Some((mk, hk)) = lookup_skin(probe) {
                            stack_buf.push((mk, hk as f64));
                        }
                    }
                }
                _ => {
                    stack_buf.push((m0, seg.height_mm.max(0.10) as f64));
                }
            }

            let srgb = evaluate_stack_srgb(&stack_buf, &models, backing_ref, &lighting);
            layer_cols.push([srgb[0], srgb[1], srgb[2], 1.0]);
        }
        out_layers.push(layer_cols);
    }

    out_layers
}

/// Maps each packed `0xRRGGBBAA` triangle colour in `triangle_colors_rgba` to the index (`0..slot_colors_srgba.len()`)
/// of the perceptually closest slot colour in **Oklab** space.
pub fn map_triangle_colors_to_slot_colors(
    triangle_colors_rgba: &[u32],
    slot_colors_srgba: &[[f32; 4]],
) -> Vec<u8> {
    if triangle_colors_rgba.is_empty() || slot_colors_srgba.is_empty() {
        return Vec::new();
    }
    let slot_labs: Vec<[f64; 3]> = slot_colors_srgba
        .iter()
        .map(|&c| srgb_to_oklab([c[0], c[1], c[2]]))
        .collect();

    triangle_colors_rgba
        .iter()
        .map(|&packed| {
            if packed == 0 {
                return 0u8;
            }
            let [r, g, b, a] = unpack_rgba8(packed);
            if a == 0 {
                return 0u8;
            }
            let lab = srgb_to_oklab([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
            let mut best_idx = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (i, s_lab) in slot_labs.iter().enumerate() {
                let d2 = oklab_distance_sq(lab, *s_lab);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_idx = i;
                }
            }
            best_idx.min(127) as u8
        })
        .collect()
}

/// Quantizes continuous per-triangle `0xRRGGBBAA` surface colours (from a UV-mapped `.obj`/`.3mf`
/// texture or vertex-colour scan) into discrete [`crate::config::ThreeMfFilamentSlot`] entries
/// using deterministic **Oklab k-means++** clustering (`max_new_slots` $\in 2..=8$, default `6`),
/// and returns the per-triangle 0-based `filament_slot` assignment (`Vec<u8>`) so the C++ OpenVDB
/// backend builds `filamentSlotGrid` across the entire 3D shell.
pub fn quantize_texture_triangles_to_filament_slots(
    triangle_colors_rgba: &[u32],
    filament_slots: &mut Vec<crate::config::ThreeMfFilamentSlot>,
    max_new_slots: usize,
) -> Option<Vec<u8>> {
    if triangle_colors_rgba.is_empty() {
        return None;
    }

    // If filament slots are already populated, map triangles directly to those slots.
    if !filament_slots.is_empty() {
        let slot_colors: Vec<[f32; 4]> = filament_slots.iter().map(|s| s.color_srgba).collect();
        return Some(map_triangle_colors_to_slot_colors(
            triangle_colors_rgba,
            &slot_colors,
        ));
    }

    // Sample up to 16,384 non-zero triangles deterministically for fast (< 2 ms) Oklab k-means++.
    const MAX_SAMPLES: usize = 16_384;
    let step = (triangle_colors_rgba.len() / MAX_SAMPLES).max(1);
    let mut samples_lab: Vec<[f64; 3]> = Vec::with_capacity(MAX_SAMPLES.min(triangle_colors_rgba.len()));
    for &packed in triangle_colors_rgba.iter().step_by(step) {
        if packed == 0 {
            continue;
        }
        let [r, g, b, a] = unpack_rgba8(packed);
        if a == 0 {
            continue;
        }
        samples_lab.push(srgb_to_oklab([
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
        ]));
    }
    if samples_lab.is_empty() {
        return None;
    }

    let k_target = max_new_slots.clamp(2, 8);
    const MIN_CLUSTER_DELTA_E: f64 = 0.042;
    const MIN_CLUSTER_D2: f64 = MIN_CLUSTER_DELTA_E * MIN_CLUSTER_DELTA_E;

    // 1. Deterministic farthest-point (k-means++) seeding in Oklab space:
    //    Seed 0 is the sample closest to the global mean colour (dominant body tone).
    let inv_n = 1.0 / (samples_lab.len() as f64);
    let mut mean_lab = [0.0_f64; 3];
    for s in &samples_lab {
        mean_lab[0] += s[0] * inv_n;
        mean_lab[1] += s[1] * inv_n;
        mean_lab[2] += s[2] * inv_n;
    }
    let mut seed0_idx = 0usize;
    let mut seed0_d2 = f64::INFINITY;
    for (i, s) in samples_lab.iter().enumerate() {
        let d2 = oklab_distance_sq(*s, mean_lab);
        if d2 < seed0_d2 {
            seed0_d2 = d2;
            seed0_idx = i;
        }
    }

    let mut centroids: Vec<[f64; 3]> = vec![samples_lab[seed0_idx]];
    let mut min_dist_sq: Vec<f64> = samples_lab
        .iter()
        .map(|s| oklab_distance_sq(*s, centroids[0]))
        .collect();

    while centroids.len() < k_target {
        let mut farthest_idx = 0usize;
        let mut farthest_d2 = -1.0_f64;
        for (i, &d2) in min_dist_sq.iter().enumerate() {
            if d2 > farthest_d2 {
                farthest_d2 = d2;
                farthest_idx = i;
            }
        }
        if farthest_d2 < MIN_CLUSTER_D2 {
            break;
        }
        let next_c = samples_lab[farthest_idx];
        for (s, d2) in samples_lab.iter().zip(min_dist_sq.iter_mut()) {
            let new_d2 = oklab_distance_sq(*s, next_c);
            if new_d2 < *d2 {
                *d2 = new_d2;
            }
        }
        centroids.push(next_c);
    }

    // 2. Run 10 Lloyd refinement iterations on `samples_lab`.
    let mut counts = vec![0usize; centroids.len()];
    let mut sums = vec![[0.0_f64; 3]; centroids.len()];
    for _iter in 0..10 {
        counts.fill(0);
        sums.fill([0.0; 3]);
        for s in &samples_lab {
            let mut best_c = 0usize;
            let mut best_d2 = f64::INFINITY;
            for (ci, c) in centroids.iter().enumerate() {
                let d2 = oklab_distance_sq(*s, *c);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = ci;
                }
            }
            counts[best_c] += 1;
            sums[best_c][0] += s[0];
            sums[best_c][1] += s[1];
            sums[best_c][2] += s[2];
        }
        let mut max_shift_d2 = 0.0_f64;
        for ci in 0..centroids.len() {
            if counts[ci] > 0 {
                let inv = 1.0 / (counts[ci] as f64);
                let next = [sums[ci][0] * inv, sums[ci][1] * inv, sums[ci][2] * inv];
                max_shift_d2 = max_shift_d2.max(oklab_distance_sq(centroids[ci], next));
                centroids[ci] = next;
            }
        }
        if max_shift_d2 < 1.0e-7 {
            break;
        }
    }

    // 3. Filter out empty clusters, sort by descending population, and merge perceptually close centroids.
    let mut ranked: Vec<([f64; 3], usize)> = centroids
        .into_iter()
        .zip(counts)
        .filter(|&(_, cnt)| cnt > 0)
        .collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));

    let mut merged_centroids: Vec<[f64; 3]> = Vec::with_capacity(ranked.len());
    for (c, _cnt) in ranked {
        if merged_centroids
            .iter()
            .all(|existing| oklab_distance_sq(*existing, c) >= MIN_CLUSTER_D2)
        {
            merged_centroids.push(c);
        }
    }
    if merged_centroids.is_empty() {
        merged_centroids.push(mean_lab);
    }

    // 4. Populate `filament_slots` and map all triangles.
    let mut slot_colors_srgba = Vec::with_capacity(merged_centroids.len());
    for (idx, &lab) in merged_centroids.iter().enumerate() {
        let srgb = oklab_to_srgb(lab);
        let rgba8 = [
            (srgb[0].clamp(0.0, 1.0) * 255.0).round() as u8,
            (srgb[1].clamp(0.0, 1.0) * 255.0).round() as u8,
            (srgb[2].clamp(0.0, 1.0) * 255.0).round() as u8,
            255,
        ];
        let hex = format_hex_rgba(rgba8);
        let srgba = rgba8_to_srgba(rgba8);
        slot_colors_srgba.push(srgba);
        filament_slots.push(crate::config::ThreeMfFilamentSlot {
            slot_index: idx as u32,
            name: format!("Filament #{} (Texture {})", idx + 1, hex),
            filament_type: "PLA".to_string(),
            color_hex: hex,
            color_srgba: srgba,
            tool_index: idx,
            ..Default::default()
        });
    }

    Some(map_triangle_colors_to_slot_colors(
        triangle_colors_rgba,
        &slot_colors_srgba,
    ))
}

// ============================================================================
// Virtual Filaments: Integer-Ratio Dithering & Optical Blending
// ============================================================================

#[inline]
pub fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

/// Consolidates duplicate `tool_index` entries and divides all integer ratio weights by their
/// greatest common divisor ($\gcd$), clamping `tool_index` to `0..num_tools`.
pub fn normalize_virtual_filament_components(
    components: &[crate::config::VirtualFilamentComponent],
    num_tools: usize,
) -> Vec<crate::config::VirtualFilamentComponent> {
    if components.is_empty() || num_tools == 0 {
        return Vec::new();
    }
    let max_tool = num_tools - 1;
    let mut merged: Vec<crate::config::VirtualFilamentComponent> = Vec::with_capacity(components.len());
    for c in components {
        let ti = c.tool_index.min(max_tool);
        let r = c.ratio.clamp(1, 256);
        if let Some(existing) = merged.iter_mut().find(|m| m.tool_index == ti) {
            existing.ratio = existing.ratio.saturating_add(r).min(256);
        } else {
            merged.push(crate::config::VirtualFilamentComponent {
                tool_index: ti,
                ratio: r,
            });
        }
    }
    if merged.len() < 2 {
        return merged;
    }
    let mut g = merged[0].ratio;
    for m in &merged[1..] {
        g = gcd_u32(g, m.ratio);
    }
    let g = g.max(1);
    for m in &mut merged {
        m.ratio = (m.ratio / g).max(1);
    }
    merged
}

/// Builds a maximally interleaved Bresenham tool-index sequence of length $R = \sum (r_m / \gcd)$
/// where each component tool `c_m.tool_index` appears exact integer count $r_m / \gcd$ times.
pub fn build_bresenham_ratio_schedule(
    components: &[crate::config::VirtualFilamentComponent],
    num_tools: usize,
) -> Vec<usize> {
    let norm = normalize_virtual_filament_components(components, num_tools);
    if norm.is_empty() {
        return vec![0];
    }
    if norm.len() == 1 {
        return vec![norm[0].tool_index];
    }
    let total_steps: usize = norm.iter().map(|c| c.ratio as usize).sum();
    let total_steps = total_steps.clamp(2, 256);
    let mut counts: Vec<u32> = vec![0; norm.len()];
    let mut schedule = Vec::with_capacity(total_steps);

    for step in 0..total_steps {
        let mut best_k = 0usize;
        let mut best_deficit = f64::NEG_INFINITY;
        for (k, comp) in norm.iter().enumerate() {
            if counts[k] >= comp.ratio {
                continue;
            }
            // Target cumulative count at (step + 0.5) is (step + 0.5) * r_k / R
            let target = ((step as f64) + 0.5) * (comp.ratio as f64) / (total_steps as f64);
            let deficit = target - (counts[k] as f64);
            if deficit > best_deficit + 1.0e-9 {
                best_deficit = deficit;
                best_k = k;
            }
        }
        counts[best_k] += 1;
        schedule.push(norm[best_k].tool_index);
    }
    schedule
}

/// Builds a run-grouped integer-ratio schedule `[T_0 (×r_0), T_1 (×r_1), ...]` of length
/// $R = \sum (r_m / \gcd)$ for along-path short-segment dithering (`AlongPathSegments`),
/// minimizing toolchanges per cycle while preserving the exact integer length ratio.
pub fn build_grouped_ratio_schedule(
    components: &[crate::config::VirtualFilamentComponent],
    num_tools: usize,
) -> Vec<usize> {
    let norm = normalize_virtual_filament_components(components, num_tools);
    if norm.is_empty() {
        return vec![0];
    }
    let mut schedule = Vec::new();
    for comp in &norm {
        for _ in 0..comp.ratio {
            schedule.push(comp.tool_index);
        }
    }
    if schedule.is_empty() {
        schedule.push(0);
    }
    schedule
}

/// Computes the optically blended sRGBA colour `[r, g, b, 1.0]` of a Virtual Filament from its
/// constituent real tools and integer ratios, combining Kubelka-Munk stacked interlayer transmittance
/// and side-by-side linear-light ratio averaging.
pub fn blend_virtual_filament_srgba(
    components: &[crate::config::VirtualFilamentComponent],
    tools: &[crate::config::ToolConfig],
) -> [f32; 4] {
    if tools.is_empty() {
        return [0.85, 0.85, 0.85, 1.0];
    }
    let norm = normalize_virtual_filament_components(components, tools.len());
    if norm.is_empty() {
        return tools[0].filament_color_srgba;
    }
    if norm.len() == 1 {
        return tools[norm[0].tool_index].filament_color_srgba;
    }

    let total_r: f64 = norm.iter().map(|c| c.ratio as f64).sum();
    let mut lin_avg = [0.0_f64; 3];
    for comp in &norm {
        let w = (comp.ratio as f64) / total_r;
        let tc = tools[comp.tool_index].filament_color_srgba;
        let lin = srgb_to_linear_rgb([tc[0], tc[1], tc[2]]);
        lin_avg[0] += w * lin[0];
        lin_avg[1] += w * lin[1];
        lin_avg[2] += w * lin[2];
    }

    // Also evaluate Kubelka-Munk stacked slab reflectance over the Bresenham schedule (up to 6 slabs)
    // to capture subtractive dye filtering across translucent layers.
    let schedule = build_bresenham_ratio_schedule(&norm, tools.len());
    let models: Vec<FilamentOpticalModel> = tools
        .iter()
        .map(|t| {
            let c = t.filament_color_srgba;
            FilamentOpticalModel::from_srgb_and_td(
                [c[0], c[1], c[2]],
                t.effective_transmission_distance_mm(),
            )
        })
        .collect();
    let n_slabs = schedule.len().clamp(2, 6);
    let lighting = LithophaneLighting::default_frontlit_d65();
    let mut km_avg = [0.0_f64; 3];
    for phase in 0..schedule.len() {
        let mut comp_slab = OpticalSlab::transparent();
        for s in 0..n_slabs {
            let ti = schedule[(phase + s) % schedule.len()];
            comp_slab = comp_slab.stack_over(&models[ti].slab_response(0.20));
        }
        let back_ti = schedule[(phase + n_slabs - 1) % schedule.len()];
        comp_slab = comp_slab.stack_over(&models[back_ti].opaque_backing_slab());
        let rgb = comp_slab.evaluate_lighting(&lighting);
        km_avg[0] += rgb[0] / (schedule.len() as f64);
        km_avg[1] += rgb[1] / (schedule.len() as f64);
        km_avg[2] += rgb[2] / (schedule.len() as f64);
    }

    let blended_lin = [
        0.55 * km_avg[0] + 0.45 * lin_avg[0],
        0.55 * km_avg[1] + 0.45 * lin_avg[1],
        0.55 * km_avg[2] + 0.45 * lin_avg[2],
    ];
    let srgb = linear_rgb_to_srgb(blended_lin);
    [
        srgb[0].clamp(0.0, 1.0),
        srgb[1].clamp(0.0, 1.0),
        srgb[2].clamp(0.0, 1.0),
        1.0,
    ]
}

/// Searches 2-tool and 3-tool integer ratios ($\sum r_k \le \text{max\_total\_parts}$, $\gcd = 1$)
/// across `tools` to find the Virtual Filament composition that best approximates `target_srgba`
/// in perceptual Oklab space.
pub fn find_best_integer_ratio_for_target_srgba(
    target_srgba: [f32; 4],
    tools: &[crate::config::ToolConfig],
    max_total_parts: u32,
) -> Vec<crate::config::VirtualFilamentComponent> {
    if tools.len() < 2 {
        return vec![crate::config::VirtualFilamentComponent {
            tool_index: 0,
            ratio: 1,
        }];
    }
    let max_parts = max_total_parts.clamp(2, 12);
    let target_lab = srgb_to_oklab([target_srgba[0], target_srgba[1], target_srgba[2]]);
    let mut best_components = vec![
        crate::config::VirtualFilamentComponent {
            tool_index: 0,
            ratio: 1,
        },
        crate::config::VirtualFilamentComponent {
            tool_index: 1,
            ratio: 1,
        },
    ];
    let mut best_cost = f64::INFINITY;

    // 1. Evaluate all 2-tool integer ratios (r_a : r_b) with gcd(r_a, r_b) == 1 and r_a + r_b <= max_parts
    for a in 0..tools.len() {
        for b in (a + 1)..tools.len() {
            for ra in 1..max_parts {
                for rb in 1..=(max_parts - ra) {
                    if gcd_u32(ra, rb) != 1 {
                        continue;
                    }
                    let cand = [
                        crate::config::VirtualFilamentComponent {
                            tool_index: a,
                            ratio: ra,
                        },
                        crate::config::VirtualFilamentComponent {
                            tool_index: b,
                            ratio: rb,
                        },
                    ];
                    let srgba = blend_virtual_filament_srgba(&cand, tools);
                    let lab = srgb_to_oklab([srgba[0], srgba[1], srgba[2]]);
                    // Slight preference for smaller total parts when perceptual error is tied
                    let cost = oklab_delta_e(lab, target_lab) + 0.0012 * ((ra + rb) as f64);
                    if cost < best_cost {
                        best_cost = cost;
                        best_components = cand.to_vec();
                    }
                }
            }
        }
    }

    // 2. Evaluate 3-tool integer ratios (r_a : r_b : r_c) up to sum <= 6 when >= 3 tools exist
    if tools.len() >= 3 {
        let max_3 = max_parts.min(6);
        for a in 0..tools.len() {
            for b in (a + 1)..tools.len() {
                for c in (b + 1)..tools.len() {
                    for ra in 1..max_3 {
                        for rb in 1..(max_3 - ra) {
                            for rc in 1..=(max_3 - ra - rb) {
                                if gcd_u32(gcd_u32(ra, rb), rc) != 1 {
                                    continue;
                                }
                                let cand = [
                                    crate::config::VirtualFilamentComponent {
                                        tool_index: a,
                                        ratio: ra,
                                    },
                                    crate::config::VirtualFilamentComponent {
                                        tool_index: b,
                                        ratio: rb,
                                    },
                                    crate::config::VirtualFilamentComponent {
                                        tool_index: c,
                                        ratio: rc,
                                    },
                                ];
                                let srgba = blend_virtual_filament_srgba(&cand, tools);
                                let lab = srgb_to_oklab([srgba[0], srgba[1], srgba[2]]);
                                let cost = oklab_delta_e(lab, target_lab)
                                    + 0.0025 * ((ra + rb + rc) as f64);
                                if cost < best_cost {
                                    best_cost = cost;
                                    best_components = cand.to_vec();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort components in descending ratio order so primary component comes first
    best_components.sort_by(|x, y| y.ratio.cmp(&x.ratio).then(x.tool_index.cmp(&y.tool_index)));
    best_components
}

/// Dithers toolpaths assigned to any Virtual Filament (`ThreeMfFilamentSlot::is_virtual_mix()`)
/// into direct real-tool assignments (`encode_direct_tool_slot(tool_idx)`) according to each
/// virtual filament's integer ratio and `VirtualFilamentDitherMode`.
pub fn apply_virtual_filament_dithering_to_toolpaths(
    raw_toolpaths: &ObjectToolpathsResult,
    filament_slots: &[crate::config::ThreeMfFilamentSlot],
    num_tools: usize,
    unit_segment_mm: f32,
    support_tool_index: Option<usize>,
) -> ObjectToolpathsResult {
    if !raw_toolpaths.valid
        || num_tools < 2
        || !filament_slots.iter().any(|s| s.is_virtual_mix())
    {
        return raw_toolpaths.clone();
    }

    let find_virtual_slot = |slot_u8: u8| -> Option<&crate::config::ThreeMfFilamentSlot> {
        if decode_direct_tool_slot(slot_u8).is_some() {
            return None;
        }
        let s = filament_slots
            .iter()
            .find(|s| s.slot_index == slot_u8 as u32)
            .or_else(|| filament_slots.get(slot_u8 as usize))?;
        if s.is_virtual_mix() {
            Some(s)
        } else {
            None
        }
    };

    let unit_len = unit_segment_mm.clamp(1.0, 50.0);
    let mut out = ObjectToolpathsResult {
        valid: true,
        total_path_count: 0,
        outer_wall_count: 0,
        inner_wall_count: 0,
        solid_infill_count: 0,
        sparse_infill_count: 0,
        bridge_count: 0,
        wave_overhang_count: 0,
        support_count: 0,
        support_interface_count: 0,
        layers: Vec::with_capacity(raw_toolpaths.layers.len()),
    };

    for layer in &raw_toolpaths.layers {
        let mut out_paths: Vec<ExtrusionPath> = Vec::with_capacity(layer.paths.len());
        let mut solid_infill_seq = 0usize;

        for path in &layer.paths {
            if path.points.len() < 2 {
                increment_feature_counter(&mut out, path.feature_kind);
                out_paths.push(path.clone());
                continue;
            }
            if path.feature_kind == 7 || path.feature_kind == 8 {
                let mut p = path.clone();
                if let Some(sup_t) = support_tool_index {
                    p.filament_slot = encode_direct_tool_slot(sup_t);
                }
                increment_feature_counter(&mut out, p.feature_kind);
                out_paths.push(p);
                continue;
            }

            let Some(vslot) = find_virtual_slot(path.filament_slot) else {
                increment_feature_counter(&mut out, path.feature_kind);
                out_paths.push(path.clone());
                continue;
            };

            // SparseInfill (`feature_kind == 4`) stays on the virtual filament's primary real tool
            // so the printer never wastes toolchanges inside the hidden interior core.
            if path.feature_kind == 4 {
                let mut p = path.clone();
                p.filament_slot = encode_direct_tool_slot(vslot.primary_tool_index(num_tools));
                increment_feature_counter(&mut out, p.feature_kind);
                out_paths.push(p);
                continue;
            }

            match vslot.dither_mode {
                crate::config::VirtualFilamentDitherMode::ByLayer => {
                    let schedule =
                        build_bresenham_ratio_schedule(&vslot.dither_components, num_tools);
                    let tool_idx = schedule[(layer.layer_index as usize) % schedule.len()];
                    let mut p = path.clone();
                    p.filament_slot = encode_direct_tool_slot(tool_idx);
                    increment_feature_counter(&mut out, p.feature_kind);
                    out_paths.push(p);
                }
                crate::config::VirtualFilamentDitherMode::ByWallAndLayer => {
                    let schedule =
                        build_bresenham_ratio_schedule(&vslot.dither_components, num_tools);
                    let depth_offset = match path.feature_kind {
                        0 => 0usize,
                        1 | 5 | 6 => path.wall_index as usize,
                        2 | 3 => {
                            let idx = 2 + solid_infill_seq;
                            solid_infill_seq = solid_infill_seq.wrapping_add(1);
                            idx
                        }
                        _ => path.wall_index as usize,
                    };
                    let step_idx =
                        ((layer.layer_index as usize) + depth_offset) % schedule.len();
                    let tool_idx = schedule[step_idx];
                    let mut p = path.clone();
                    p.filament_slot = encode_direct_tool_slot(tool_idx);
                    increment_feature_counter(&mut out, p.feature_kind);
                    out_paths.push(p);
                }
                crate::config::VirtualFilamentDitherMode::AlongPathSegments => {
                    let schedule =
                        build_grouped_ratio_schedule(&vslot.dither_components, num_tools);
                    let phase_offset =
                        ((layer.layer_index as usize) + (path.wall_index as usize)) % schedule.len();

                    let num_raw_edges = if path.is_closed {
                        let d_close = dist3(path.points.last().unwrap(), &path.points[0]);
                        if d_close > 1.0e-3 {
                            path.points.len()
                        } else {
                            path.points.len() - 1
                        }
                    } else {
                        path.points.len() - 1
                    };

                    let mut sub_segs: Vec<PreparedSubSeg> = Vec::new();
                    let mut seg_tool_choices: Vec<usize> = Vec::new();
                    let mut accum_len = 0.0f32;

                    for edge_i in 0..num_raw_edges {
                        let a = &path.points[edge_i];
                        let b = &path.points[(edge_i + 1) % path.points.len()];
                        let edge_len = dist3(a, b);
                        if edge_len <= 1.0e-5 {
                            continue;
                        }
                        let n_div = if edge_len > 1.25 * unit_len {
                            ((edge_len / unit_len).round() as usize).clamp(1, 512)
                        } else {
                            1
                        };
                        let seg_len = edge_len / (n_div as f32);
                        for k in 0..n_div {
                            let t0 = (k as f32) / (n_div as f32);
                            let t1 = ((k + 1) as f32) / (n_div as f32);
                            let p0 = if k == 0 {
                                *a
                            } else {
                                interpolate_point(a, b, t0)
                            };
                            let p1 = if k + 1 == n_div {
                                *b
                            } else {
                                interpolate_point(a, b, t1)
                            };
                            let mid_dist = accum_len + 0.5 * seg_len;
                            accum_len += seg_len;
                            let unit_idx = (mid_dist / unit_len).floor().max(0.0) as usize;
                            let tool_idx = schedule[(phase_offset + unit_idx) % schedule.len()];
                            sub_segs.push(PreparedSubSeg {
                                p0,
                                p1,
                                len_mm: seg_len,
                                sample_idx: 0,
                            });
                            seg_tool_choices.push(tool_idx);
                        }
                    }

                    if sub_segs.is_empty() {
                        let mut p = path.clone();
                        p.filament_slot = encode_direct_tool_slot(schedule[phase_offset]);
                        increment_feature_counter(&mut out, p.feature_kind);
                        out_paths.push(p);
                    } else {
                        let identity_tools: Vec<usize> = (0..num_tools).collect();
                        let emitted = emit_segmented_extrusion_paths(
                            path,
                            &sub_segs,
                            &seg_tool_choices,
                            &identity_tools,
                            unit_len * 0.45,
                        );
                        for sub_path in emitted {
                            increment_feature_counter(&mut out, sub_path.feature_kind);
                            out_paths.push(sub_path);
                        }
                    }
                }
            }
        }

        out.total_path_count += out_paths.len() as u32;
        out.layers.push(crate::bridge::ffi::LayerToolpaths {
            layer_index: layer.layer_index,
            branch_id: layer.branch_id,
            isovalue: layer.isovalue,
            paths: out_paths,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_color_encodings_and_roundtrips_oklab() {
        assert_eq!(
            parse_color_encoding_rgba8("#08F"),
            Some([0x00, 0x88, 0xFF, 0xFF])
        );
        assert_eq!(
            parse_color_encoding_rgba8("#EC008C80"),
            Some([0xEC, 0x00, 0x8C, 0x80])
        );
        assert_eq!(
            parse_color_encoding_rgba8("rgb(252, 227, 0)"),
            Some([252, 227, 0, 255])
        );
        assert_eq!(
            parse_color_encoding_rgba8("rgba(100%, 0%, 50%, 0.5)"),
            Some([255, 0, 128, 128])
        );

        for hex in ["#0086D6", "#EC008C", "#FCE300", "#161616", "#F4F4F4"] {
            let srgba = parse_color_encoding_srgba(hex).unwrap();
            let srgb = [srgba[0], srgba[1], srgba[2]];
            let lab = srgb_to_oklab(srgb);
            let round = oklab_to_srgb(lab);
            for c in 0..3 {
                assert!(
                    (srgb[c] - round[c]).abs() < 1.5e-3,
                    "Oklab roundtrip mismatch for {hex}: {srgb:?} vs {round:?}"
                );
            }
        }
    }

    #[test]
    fn kubelka_munk_td_and_star_product_associativity() {
        let cyan = FilamentOpticalModel::from_srgb_and_td([0.0, 0.525, 0.839], 3.5);
        let mag = FilamentOpticalModel::from_srgb_and_td([0.925, 0.0, 0.549], 3.2);
        let yel = FilamentOpticalModel::from_srgb_and_td([0.988, 0.890, 0.0], 6.0);
        let wht = FilamentOpticalModel::from_srgb_and_td([0.96, 0.96, 0.96], 5.0);

        // At thickness = TD, effective passband transmittance is calibrated to 0.08 (~92% hiding contrast)
        let slab_td = cyan.slab_response(3.5);
        let lum_t = 0.2126729 * slab_td.t[0] + 0.7151522 * slab_td.t[1] + 0.0721750 * slab_td.t[2];
        let max_t = slab_td.t[0].max(slab_td.t[1]).max(slab_td.t[2]);
        let eff_t = lum_t.max(0.75 * max_t);
        assert!(
            (eff_t - 0.08).abs() < 1.0e-4,
            "Expected 8% passband transmittance (~92% hiding contrast) at TD, got {eff_t}"
        );

        // Associativity: (A * B) * C == A * (B * C) for both reflectance AND lithophane transmittance
        let s1 = cyan.slab_response(0.20);
        let s2 = yel.slab_response(0.20);
        let s3 = mag.slab_response(0.16);
        let s4 = wht.slab_response(0.40);

        let lhs = s1.stack_over(&s2).stack_over(&s3).stack_over(&s4);
        let rhs = s1.stack_over(&(s2.stack_over(&(s3.stack_over(&s4)))));
        for c in 0..3 {
            assert!((lhs.r_front[c] - rhs.r_front[c]).abs() < 1.0e-10);
            assert!((lhs.r_back[c] - rhs.r_back[c]).abs() < 1.0e-10);
            assert!((lhs.t[c] - rhs.t[c]).abs() < 1.0e-10);
        }

        // Subtractive CMY physics: Cyan + Yellow over White produces Green (G > R and G > B) in both orders,
        // with Yellow-over-Cyan-over-White absorbing Blue on both passes for vivid emerald Green!
        let green_cy = evaluate_stack_srgb(
            &[(0, 0.20), (1, 0.28)],
            &[cyan, yel],
            Some(&wht),
            &LithophaneLighting::default_frontlit_d65(),
        );
        let green_yc = evaluate_stack_srgb(
            &[(1, 0.24), (0, 0.24)],
            &[cyan, yel],
            Some(&wht),
            &LithophaneLighting::default_frontlit_d65(),
        );
        assert!(
            green_cy[1] > green_cy[0] + 0.20 && green_cy[1] > green_cy[2] + 0.12,
            "Cyan over Yellow over White should produce Green, got {green_cy:?}"
        );
        assert!(
            green_yc[1] > green_yc[0] + 0.15 && green_yc[1] > green_yc[2] + 0.30,
            "Yellow over Cyan over White should produce vivid Green, got {green_yc:?}"
        );
    }

    #[test]
    fn monochrome_lithophane_thickness_solver_is_monotone_and_accurate() {
        let white_pla = FilamentOpticalModel::from_srgb_and_td([0.96, 0.96, 0.95], 4.5);
        let lighting = LithophaneLighting::default_backlit_d65();

        let d_bright = solve_monochrome_lithophane_thickness(&white_pla, 0.85, 0.4, 3.2, &lighting);
        let d_mid = solve_monochrome_lithophane_thickness(&white_pla, 0.50, 0.4, 3.2, &lighting);
        let d_dark = solve_monochrome_lithophane_thickness(&white_pla, 0.15, 0.4, 3.2, &lighting);

        assert!(
            d_bright < d_mid && d_mid < d_dark,
            "Backlit lithophane must be thinner for brighter pixels: {d_bright}, {d_mid}, {d_dark}"
        );
    }

    #[test]
    fn cmykw_and_arbitrary_palette_short_extrusion_mixing() {
        let pal = bambu_pla_cmykw_palette();
        let tools: Vec<ToolOpticalInput> = pal
            .iter()
            .enumerate()
            .map(|(idx, p)| ToolOpticalInput {
                tool_index: idx,
                color_srgb: [p.color_srgba[0], p.color_srgba[1], p.color_srgba[2]],
                td_mm: p.transmission_distance_mm,
            })
            .collect();

        // Build a 24 mm outer wall whose left half is bright Green and right half is bright Red
        let raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 1,
            outer_wall_count: 1,
            layers: vec![crate::bridge::ffi::LayerToolpaths {
                layer_index: 2,
                branch_id: 0,
                isovalue: 0.6,
                paths: vec![ExtrusionPath {
                    feature_kind: 0,
                    filament_slot: 0,
                    wall_index: 0,
                    is_closed: false,
                    total_length_mm: 24.0,
                    mean_width_mm: 0.4,
                    points: vec![
                        ToolpathPoint {
                            x: 0.0,
                            y: 0.0,
                            z: 0.6,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                        ToolpathPoint {
                            x: 24.0,
                            y: 0.0,
                            z: 0.6,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                    ],
                }],
            }],
            ..Default::default()
        };

        let cfg = ColorMixingConfig {
            enabled: true,
            lighting_mode: OpticalLightingMode::FrontLitReflectance,
            use_adjacent_layers: true,
            use_short_extrusions: true,
            min_short_extrusion_mm: 3.0,
            optical_shell_layers: 4,
            backing_tool_index: None,
            dither_strength: 0.85,
            tool_switch_penalty: 0.01,
        };

        let mixed = apply_optical_color_mixing_to_toolpaths(
            &raw,
            &cfg,
            &tools,
            &[],
            0.2,
            None,
            |xyz| {
                xyz.chunks_exact(3)
                    .map(|p| {
                        let rgba = if p[0] < 12.0 {
                            pack_rgba8([20, 190, 50, 255]) // Green
                        } else {
                            pack_rgba8([230, 35, 20, 255]) // Warm Red-Orange
                        };
                        SurfaceOpticalSample {
                            sdf_mm: -0.2,
                            surf_nx: 0.0,
                            surf_ny: -1.0,
                            surf_nz: 0.0,
                            rgba_packed: rgba,
                            filament_slot: 0,
                        }
                    })
                    .collect()
            },
        );

        assert!(
            mixed.layers[0].paths.len() >= 2,
            "24mm bi-color wall should subdivide into multiple short extrusions"
        );
        for p in &mixed.layers[0].paths {
            assert!(
                p.total_length_mm >= 2.99,
                "Every subdivided extrusion must respect min_short_extrusion_mm (got {})",
                p.total_length_mm
            );
            assert!(
                decode_direct_tool_slot(p.filament_slot).is_some(),
                "Expected direct tool slot encoding"
            );
        }

        // Verify that colour carries through the ENTIRE shell (OuterWall, InnerWalls 1..4, and SolidInfill),
        // while SparseInfill remains on the backing/core filament.
        let make_line = |feature_kind: u8, wall_index: u32, y_depth_mm: f32| ExtrusionPath {
            feature_kind,
            filament_slot: 0,
            wall_index,
            is_closed: false,
            total_length_mm: 12.0,
            mean_width_mm: 0.4,
            points: vec![
                ToolpathPoint {
                    x: 0.0,
                    y: y_depth_mm,
                    z: 0.6,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    width_mm: 0.4,
                    height_mm: 0.2,
                    ..Default::default()
                },
                ToolpathPoint {
                    x: 12.0,
                    y: y_depth_mm,
                    z: 0.6,
                    nx: 0.0,
                    ny: 0.0,
                    nz: 1.0,
                    width_mm: 0.4,
                    height_mm: 0.2,
                    ..Default::default()
                },
            ],
        };

        let full_shell_raw = ObjectToolpathsResult {
            valid: true,
            total_path_count: 5,
            outer_wall_count: 1,
            layers: vec![crate::bridge::ffi::LayerToolpaths {
                layer_index: 1,
                branch_id: 0,
                isovalue: 0.4,
                paths: vec![
                    make_line(0, 0, 0.2), // OuterWall
                    make_line(1, 1, 0.6), // InnerWall 1
                    make_line(1, 4, 1.4), // Deep InnerWall 4 (wall_index >= optical_shell_layers)
                    make_line(2, 0, 1.5), // SolidInfill inside the 1.6mm shell
                    make_line(4, 0, 4.0), // SparseInfill deep inside the core
                ],
            }],
            ..Default::default()
        };

        let full_shell_mixed = apply_optical_color_mixing_to_toolpaths(
            &full_shell_raw,
            &cfg,
            &tools,
            &[],
            0.2,
            None,
            |xyz| {
                xyz.chunks_exact(3)
                    .map(|p| SurfaceOpticalSample {
                        sdf_mm: -p[1],
                        surf_nx: 0.0,
                        surf_ny: -1.0,
                        surf_nz: 0.0,
                        // Pure Cyan surface colour -> Tool 0 (Cyan), never Tool 3 (White backing)
                        rgba_packed: pack_rgba8([0, 134, 214, 255]),
                        filament_slot: 0,
                    })
                    .collect()
            },
        );

        for p in &full_shell_mixed.layers[0].paths {
            let tool = decode_direct_tool_slot(p.filament_slot).expect("direct tool slot");
            if p.feature_kind == 4 {
                assert_eq!(
                    tool, 4,
                    "SparseInfill (feature_kind == 4) should use White backing/core tool (T4)"
                );
            } else {
                assert_eq!(
                    tool, 0,
                    "Full shell feature_kind={} (wall_index={}) should carry Cyan surface colour (T0) through the entire shell, got T{}",
                    p.feature_kind, p.wall_index, tool
                );
            }
        }
    }

    #[test]
    fn virtual_filament_integer_ratio_dithering_and_blending() {
        use crate::config::{
            SlicerConfig, ThreeMfFilamentSlot, VirtualFilamentComponent, VirtualFilamentDitherMode,
        };

        // 1. GCD reduction (4:2 -> 2:1) and balanced Bresenham schedule for 3:2 -> [0, 1, 0, 1, 0]
        let sched_4_2 = build_bresenham_ratio_schedule(
            &[
                VirtualFilamentComponent {
                    tool_index: 0,
                    ratio: 4,
                },
                VirtualFilamentComponent {
                    tool_index: 1,
                    ratio: 2,
                },
            ],
            3,
        );
        assert_eq!(sched_4_2.len(), 3, "4:2 ratio must reduce via GCD to length 3");
        assert_eq!(sched_4_2.iter().filter(|&&t| t == 0).count(), 2);
        assert_eq!(sched_4_2.iter().filter(|&&t| t == 1).count(), 1);

        let sched_3_2 = build_bresenham_ratio_schedule(
            &[
                VirtualFilamentComponent {
                    tool_index: 0,
                    ratio: 3,
                },
                VirtualFilamentComponent {
                    tool_index: 1,
                    ratio: 2,
                },
            ],
            2,
        );
        assert_eq!(sched_3_2, vec![0, 1, 0, 1, 0]);

        // 2. Optical blending & Auto-Fit integer ratio on CMYKW tools
        let mut cfg = SlicerConfig::default();
        cfg.apply_optical_palette_to_tools(&bambu_pla_cmykw_palette());
        let green_1_1 = blend_virtual_filament_srgba(
            &[
                VirtualFilamentComponent {
                    tool_index: 0, // Cyan
                    ratio: 1,
                },
                VirtualFilamentComponent {
                    tool_index: 2, // Yellow
                    ratio: 1,
                },
            ],
            &cfg.tools,
        );
        assert!(
            green_1_1[1] > green_1_1[0] + 0.12 && green_1_1[1] > green_1_1[2] + 0.05,
            "1:1 Cyan:Yellow Virtual Filament should blend to Green, got {green_1_1:?}"
        );

        let fitted = find_best_integer_ratio_for_target_srgba(green_1_1, &cfg.tools, 8);
        assert!(
            fitted.len() >= 2
                && fitted.iter().any(|c| c.tool_index == 0)
                && fitted.iter().any(|c| c.tool_index == 2),
            "Auto-Fit on Green target should select Cyan (T0) and Yellow (T2), got {fitted:?}"
        );

        // 3. Verify `ByLayer`, `ByWallAndLayer`, and `AlongPathSegments` dithering in 2:1 integer ratio
        let mut vslot = ThreeMfFilamentSlot::default_for_slot(0, 0);
        vslot.dither_components = vec![
            VirtualFilamentComponent {
                tool_index: 0,
                ratio: 2,
            },
            VirtualFilamentComponent {
                tool_index: 1,
                ratio: 1,
            },
        ];

        let make_layer = |layer_index: u32| crate::bridge::ffi::LayerToolpaths {
            layer_index,
            branch_id: 0,
            isovalue: (layer_index + 1) as f32 * 0.2,
            paths: vec![
                ExtrusionPath {
                    feature_kind: 0,
                    filament_slot: 0,
                    wall_index: 0,
                    is_closed: false,
                    total_length_mm: 30.0,
                    mean_width_mm: 0.4,
                    points: vec![
                        ToolpathPoint {
                            x: 0.0,
                            y: 0.0,
                            z: 0.2,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                        ToolpathPoint {
                            x: 30.0,
                            y: 0.0,
                            z: 0.2,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                    ],
                },
                ExtrusionPath {
                    feature_kind: 4, // SparseInfill (must remain on primary tool T0)
                    filament_slot: 0,
                    wall_index: 0,
                    is_closed: false,
                    total_length_mm: 10.0,
                    mean_width_mm: 0.4,
                    points: vec![
                        ToolpathPoint {
                            x: 0.0,
                            y: 5.0,
                            z: 0.2,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                        ToolpathPoint {
                            x: 10.0,
                            y: 5.0,
                            z: 0.2,
                            nx: 0.0,
                            ny: 0.0,
                            nz: 1.0,
                            width_mm: 0.4,
                            height_mm: 0.2,
                            ..Default::default()
                        },
                    ],
                },
            ],
        };

        let raw_3_layers = ObjectToolpathsResult {
            valid: true,
            total_path_count: 6,
            outer_wall_count: 3,
            sparse_infill_count: 3,
            layers: vec![make_layer(0), make_layer(1), make_layer(2)],
            ..Default::default()
        };

        // 3a. ByLayer mode: across 3 layers, outer wall uses T0 on 2 layers and T1 on 1 layer,
        // while SparseInfill uses T0 on all 3 layers!
        vslot.dither_mode = VirtualFilamentDitherMode::ByLayer;
        let dithered_layer = apply_virtual_filament_dithering_to_toolpaths(
            &raw_3_layers,
            &[vslot.clone()],
            cfg.tools.len(),
            5.0,
            None,
        );
        let mut wall_t0 = 0usize;
        let mut wall_t1 = 0usize;
        for l in &dithered_layer.layers {
            for p in &l.paths {
                let t = decode_direct_tool_slot(p.filament_slot).expect("direct tool");
                if p.feature_kind == 0 {
                    if t == 0 {
                        wall_t0 += 1;
                    } else if t == 1 {
                        wall_t1 += 1;
                    }
                } else if p.feature_kind == 4 {
                    assert_eq!(t, 0, "SparseInfill must stay on primary tool T0");
                }
            }
        }
        assert_eq!((wall_t0, wall_t1), (2, 1), "Expected 2:1 layer ratio for T0:T1");

        // 3b. AlongPathSegments mode: 30mm outer wall with 5mm unit segment length in 2:1 ratio
        // extrudes 20mm of T0 and 10mm of T1!
        vslot.dither_mode = VirtualFilamentDitherMode::AlongPathSegments;
        let dithered_path = apply_virtual_filament_dithering_to_toolpaths(
            &raw_3_layers,
            std::slice::from_ref(&vslot),
            cfg.tools.len(),
            5.0,
            None,
        );
        let mut len_t0 = 0.0f32;
        let mut len_t1 = 0.0f32;
        for p in &dithered_path.layers[0].paths {
            if p.feature_kind == 0 {
                let t = decode_direct_tool_slot(p.filament_slot).expect("direct tool");
                if t == 0 {
                    len_t0 += p.total_length_mm;
                } else if t == 1 {
                    len_t1 += p.total_length_mm;
                }
            }
        }
        assert!(
            (len_t0 - 20.0).abs() < 0.05 && (len_t1 - 10.0).abs() < 0.05,
            "Expected 20mm T0 and 10mm T1 (2:1 ratio) on 30mm wall, got T0={len_t0}, T1={len_t1}"
        );

        // 4. Per-Object `ObjectFilamentMode` overrides (`SingleTool` and `FilamentSlot`)
        let mut slot0 = ThreeMfFilamentSlot::default_for_slot(0, 0);
        slot0.tool_index = 0;
        let mut slot1 = vslot.clone();
        slot1.slot_index = 1;
        cfg.filament_slots = vec![slot0, slot1];
        cfg.slicing.color_mixing.min_short_extrusion_mm = 5.0;

        let single_t3 = crate::field_vis::build_mixed_toolpaths_for_object_with_mode(
            0,
            &raw_3_layers,
            &cfg,
            crate::config::ObjectFilamentMode::SingleTool(3),
            None,
        );
        for l in &single_t3.layers {
            for p in &l.paths {
                assert_eq!(
                    decode_direct_tool_slot(p.filament_slot),
                    Some(3),
                    "SingleTool(3) must assign all non-support paths to T3"
                );
            }
        }

        let virtual_slot1 = crate::field_vis::build_mixed_toolpaths_for_object_with_mode(
            0,
            &raw_3_layers,
            &cfg,
            crate::config::ObjectFilamentMode::FilamentSlot(1),
            None,
        );
        let mut v_len_t0 = 0.0f32;
        let mut v_len_t1 = 0.0f32;
        for p in &virtual_slot1.layers[0].paths {
            if p.feature_kind == 0 {
                let t = decode_direct_tool_slot(p.filament_slot).expect("direct tool");
                if t == 0 {
                    v_len_t0 += p.total_length_mm;
                } else if t == 1 {
                    v_len_t1 += p.total_length_mm;
                }
            }
        }
        assert!(
            (v_len_t0 - 20.0).abs() < 0.05 && (v_len_t1 - 10.0).abs() < 0.05,
            "FilamentSlot(1) must dither unpainted object paths using Virtual Slot #2's 2:1 ratio"
        );
    }
}
