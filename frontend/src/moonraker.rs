use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy::tasks::Task;
use reqwest::blocking::{multipart, Body, Client};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, WWW_AUTHENTICATE};
use reqwest::{Certificate, Identity, StatusCode};
use serde_json::Value;

use crate::config::{MoonrakerNetworkConfig, PrinterNetworkProtocol};

/// Result of testing connectivity to a Moonraker (Klipper) or PrusaLink (Prusa Buddy) printer without printing anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoonrakerConnectionReport {
    pub protocol: PrinterNetworkProtocol,
    pub normalized_uri: String,
    pub moonraker_version: String,
    pub klippy_state: String,
    pub hostname: String,
    pub state_message: String,
}

impl MoonrakerConnectionReport {
    pub fn summary_line(&self) -> String {
        let host_part = if self.hostname.is_empty() {
            String::new()
        } else {
            format!(" on '{}'", self.hostname)
        };
        let msg_part = if self.state_message.is_empty()
            || self.state_message.eq_ignore_ascii_case("Printer is ready")
        {
            String::new()
        } else {
            format!(" — {}", self.state_message)
        };
        match self.protocol {
            PrinterNetworkProtocol::Moonraker => format!(
                "Connected{host_part}: Moonraker {}, Klippy '{}'{}",
                self.moonraker_version, self.klippy_state, msg_part
            ),
            PrinterNetworkProtocol::PrusaLink => format!(
                "Connected{host_part}: PrusaLink FW {}, State '{}'{}",
                self.moonraker_version, self.klippy_state, msg_part
            ),
            PrinterNetworkProtocol::BambuLan => format!(
                "Connected{host_part}: Bambu LAN ({}) State '{}'{}",
                self.moonraker_version, self.klippy_state, msg_part
            ),
        }
    }
}

/// Live printer status queried from Moonraker (`/printer/objects/query`) or PrusaLink (`/api/v1/status` & `/api/v1/job`).
#[derive(Clone, Debug, PartialEq)]
pub struct MoonrakerPrintStatus {
    /// Printer state: `"standby"`, `"printing"`, `"paused"`, `"complete"`, `"cancelled"`, `"error"`, or `"offline"`.
    pub state: String,
    /// Active G-code filename (empty if idle).
    pub filename: String,
    /// Normalized print progress in `[0.0 ..= 1.0]`.
    pub progress: f32,
    /// Active print duration in seconds.
    pub print_duration_s: f64,
    /// Total elapsed duration in seconds.
    pub total_duration_s: f64,
    /// Slicer estimated total print time in seconds, if known.
    pub slicer_estimated_time_s: Option<f64>,
    /// Computed remaining time (ETA) in seconds when printing or paused.
    pub eta_remaining_s: Option<f64>,
    /// Optional status message or telemetry summary.
    pub message: String,
}

impl MoonrakerPrintStatus {
    pub fn is_printing_or_paused(&self) -> bool {
        self.state.eq_ignore_ascii_case("printing") || self.state.eq_ignore_ascii_case("paused")
    }

    pub fn formatted_eta(&self) -> Option<String> {
        self.eta_remaining_s.map(format_duration_hms)
    }

    pub fn formatted_elapsed(&self) -> String {
        format_duration_hms(self.print_duration_s.max(0.0))
    }
}

/// Result of uploading a G-code file to Moonraker (`POST /server/files/upload`) or PrusaLink (`PUT /api/v1/files/{storage}/{path}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoonrakerUploadReport {
    pub protocol: PrinterNetworkProtocol,
    pub remote_path: String,
    pub bytes_uploaded: usize,
    pub print_started: bool,
}

impl MoonrakerUploadReport {
    pub fn summary_line(&self) -> String {
        let kb = (self.bytes_uploaded as f64) / 1024.0;
        let proto_name = self.protocol.short_label();
        if self.print_started {
            format!(
                "Uploaded '{}' ({kb:.1} KB) to {proto_name} and started print.",
                self.remote_path
            )
        } else {
            format!(
                "Uploaded '{}' ({kb:.1} KB) to {proto_name}.",
                self.remote_path
            )
        }
    }
}

/// Formats a byte count cleanly in B, KB, or MB.
pub fn format_byte_size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", (bytes as f64) / 1024.0)
    } else {
        format!("{:.2} MB", (bytes as f64) / (1024.0 * 1024.0))
    }
}

/// Formats transferred bytes against total bytes (e.g. `"48.0 KB / 128.0 KB"`).
pub fn format_byte_progress(sent: u64, total: u64) -> String {
    format!("{} / {}", format_byte_size(sent), format_byte_size(total))
}

/// Snapshot of an active network printer upload's progress and status stage.
#[derive(Clone, Debug, PartialEq)]
pub struct UploadProgressSnapshot {
    pub fraction: f32,
    pub bytes_sent: u64,
    pub total_bytes: u64,
    pub stage: String,
}

impl UploadProgressSnapshot {
    /// Formats concise progress bar overlay text combining percentage, transferred bytes, or stage.
    pub fn bar_text(&self) -> String {
        let pct = (self.fraction * 100.0).clamp(0.0, 100.0);
        if self.total_bytes > 0 && self.bytes_sent > 0 {
            format!(
                "{pct:.0}% ({})",
                format_byte_progress(self.bytes_sent, self.total_bytes)
            )
        } else if self.stage.is_empty() {
            format!("{pct:.0}%")
        } else {
            format!("{pct:.0}% — {}", self.stage)
        }
    }
}

#[derive(Default)]
struct UploadProgressInner {
    bytes_sent: AtomicU64,
    total_bytes: AtomicU64,
    fraction_bits: AtomicU32,
    stage: Mutex<String>,
}

/// Thread-safe handle shared between background network upload tasks and the immediate-mode UI.
#[derive(Clone, Default)]
pub struct UploadProgressTracker {
    inner: Arc<UploadProgressInner>,
}

impl UploadProgressTracker {
    /// Initializes upload progress state before spawning a background transfer task.
    pub fn reset(&self, total_bytes: u64, initial_stage: impl Into<String>) {
        self.inner.bytes_sent.store(0, Ordering::Relaxed);
        self.inner.total_bytes.store(total_bytes, Ordering::Relaxed);
        self.inner
            .fraction_bits
            .store(0.02f32.to_bits(), Ordering::Relaxed);
        if let Ok(mut st) = self.inner.stage.lock() {
            *st = initial_stage.into();
        }
    }

    /// Updates the fractional progress and stage text without altering byte counters.
    pub fn set_stage(&self, fraction: f32, stage: impl Into<String>) {
        self.inner
            .fraction_bits
            .store(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        if let Ok(mut st) = self.inner.stage.lock() {
            *st = stage.into();
        }
    }

    /// Updates transferred byte counters and maps `[0..total]` across `[frac_start..frac_end]`.
    pub fn set_bytes_progress(
        &self,
        sent: u64,
        total: u64,
        frac_start: f32,
        frac_end: f32,
        prefix: &str,
    ) {
        self.inner.bytes_sent.store(sent, Ordering::Relaxed);
        self.inner.total_bytes.store(total, Ordering::Relaxed);
        let ratio = if total > 0 {
            (sent as f32 / total as f32).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let frac = (frac_start + ratio * (frac_end - frac_start)).clamp(0.0, 1.0);
        self.inner
            .fraction_bits
            .store(frac.to_bits(), Ordering::Relaxed);
        if let Ok(mut st) = self.inner.stage.lock() {
            *st = format!("{prefix} ({})...", format_byte_progress(sent, total));
        }
    }

    /// Returns a consistent point-in-time snapshot of upload progress.
    pub fn snapshot(&self) -> UploadProgressSnapshot {
        let bytes_sent = self.inner.bytes_sent.load(Ordering::Relaxed);
        let total_bytes = self.inner.total_bytes.load(Ordering::Relaxed);
        let fraction = f32::from_bits(self.inner.fraction_bits.load(Ordering::Relaxed));
        let stage = self
            .inner
            .stage
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();
        UploadProgressSnapshot {
            fraction: if fraction.is_finite() {
                fraction.clamp(0.0, 1.0)
            } else {
                0.0
            },
            bytes_sent,
            total_bytes,
            stage,
        }
    }
}

/// Chunked `Read` wrapper reporting HTTP streaming progress cleanly without buffering chunks on retry.
struct UploadProgressReader {
    cursor: Cursor<Arc<[u8]>>,
    total: u64,
    sent: u64,
    tracker: Option<UploadProgressTracker>,
    frac_start: f32,
    frac_end: f32,
    prefix: String,
    waiting_stage: String,
}

impl UploadProgressReader {
    fn new(
        payload: Arc<[u8]>,
        tracker: Option<&UploadProgressTracker>,
        frac_start: f32,
        frac_end: f32,
        prefix: impl Into<String>,
        waiting_stage: impl Into<String>,
    ) -> Self {
        let total = payload.len() as u64;
        Self {
            cursor: Cursor::new(payload),
            total,
            sent: 0,
            tracker: tracker.cloned(),
            frac_start,
            frac_end,
            prefix: prefix.into(),
            waiting_stage: waiting_stage.into(),
        }
    }
}

impl Read for UploadProgressReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let max_chunk = buf.len().min(16 * 1024);
        let n = self.cursor.read(&mut buf[..max_chunk])?;
        if n > 0 {
            self.sent += n as u64;
            if let Some(ref tr) = self.tracker {
                tr.set_bytes_progress(
                    self.sent,
                    self.total,
                    self.frac_start,
                    self.frac_end,
                    &self.prefix,
                );
            }
        } else if self.total > 0 {
            if let Some(ref tr) = self.tracker {
                tr.set_stage(self.frac_end, self.waiting_stage.clone());
            }
        }
        Ok(n)
    }
}

/// Bevy resource holding non-blocking background tasks and latest network printer state for the UI.
#[derive(Resource, Default)]
pub struct MoonrakerUiState {
    pub show_api_key: bool,
    pub show_password: bool,
    pub last_status: Option<Result<MoonrakerPrintStatus, String>>,
    pub last_test_result: Option<Result<MoonrakerConnectionReport, String>>,
    pub last_upload_result: Option<Result<MoonrakerUploadReport, String>>,
    pub pending_status_task: Option<Task<(Result<MoonrakerPrintStatus, String>, Option<(String, Option<f64>)>)>>,
    pub pending_test_task: Option<Task<Result<MoonrakerConnectionReport, String>>>,
    pub pending_upload_task: Option<Task<Result<MoonrakerUploadReport, String>>>,
    pub upload_progress: UploadProgressTracker,
    pub upload_start_print: bool,
    pub pending_cert_pick_task: Option<(CertPickKind, Task<Option<String>>)>,
    pub last_poll_instant: Option<Instant>,
    pub cached_metadata_estimate: Option<(String, Option<f64>)>,
}

impl MoonrakerUiState {
    /// Returns the active upload's progress snapshot when a network upload task is running.
    pub fn active_upload_progress(&self) -> Option<UploadProgressSnapshot> {
        self.pending_upload_task
            .as_ref()
            .map(|_| self.upload_progress.snapshot())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertPickKind {
    CaCert,
    ClientCert,
    ClientKey,
}

/// Formats a duration in seconds as `HH:MM:SS` (or `MM:SS` when `< 1 hour`).
pub fn format_duration_hms(seconds: f64) -> String {
    let total_secs = seconds.max(0.0).round() as u64;
    let hrs = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    if hrs > 0 {
        format!("{hrs:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}

/// Normalizes a user-configured Moonraker or PrusaLink base URI:
/// - Trims whitespace and trailing `/` characters
/// - Prepends `http://` if no `http://` or `https://` scheme is present
pub fn normalize_moonraker_base_uri(raw_uri: &str) -> Result<String, String> {
    let trimmed = raw_uri.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(
            "Printer URI is empty. Enter a Printer URL such as http://printer.local:7125 or http://prusa-core-one.local"
                .to_string(),
        );
    }
    let with_scheme = if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        let after_scheme = trimmed
            .strip_prefix("http://")
            .or_else(|| trimmed.strip_prefix("https://"))
            .unwrap_or("");
        if after_scheme.trim().is_empty() {
            return Err("Printer URI is missing a hostname after http:// or https://".to_string());
        }
        trimmed.to_string()
    } else if trimmed.contains("://") {
        return Err(format!(
            "Unsupported URI scheme in '{trimmed}' (expected http:// or https://)"
        ));
    } else {
        format!("http://{trimmed}")
    };
    Ok(with_scheme)
}

// ============================================================================
// Pure-Rust Cryptographic Helpers for PrusaLink HTTP Digest & Basic Auth
// (RFC 1321 MD5, FIPS 180-4 SHA-256, RFC 4648 Base64, RFC 2617 / RFC 7616)
// ============================================================================

/// Computes the lowercase hexadecimal RFC 1321 MD5 digest of `data`.
pub fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ];

    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let off = i * 4;
            *word = u32::from_le_bytes([
                chunk[off],
                chunk[off + 1],
                chunk[off + 2],
                chunk[off + 3],
            ]);
        }
        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;

        for i in 0..64 {
            let (f, g) = if i < 16 {
                ((b & c) | (!b & d), i)
            } else if i < 32 {
                ((d & b) | (!d & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | !d), (7 * i) % 16)
            };
            let temp = f
                .wrapping_add(a)
                .wrapping_add(K[i])
                .wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(temp.rotate_left(S[i]));
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = String::with_capacity(32);
    for word in [a0, b0, c0, d0] {
        for byte in word.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}

/// Computes the lowercase hexadecimal FIPS 180-4 SHA-256 digest of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.extend_from_slice(&bit_len.to_be_bytes());

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            let off = i * 4;
            w[i] = u32::from_be_bytes([
                chunk[off],
                chunk[off + 1],
                chunk[off + 2],
                chunk[off + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for word in h {
        for byte in word.to_be_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}

/// Standard RFC 4648 Base64 encoder for HTTP Basic authentication fallback.
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(triple & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Parsed HTTP `WWW-Authenticate: Digest ...` challenge (RFC 2617 / RFC 7616 / RFC 2069 Buddy `nhttp`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigestChallenge {
    pub realm: String,
    pub nonce: String,
    pub opaque: Option<String>,
    pub algorithm: Option<String>,
    pub qop_auth: bool,
    pub stale: bool,
}

impl DigestChallenge {
    /// Parses a `WWW-Authenticate` header value starting with `Digest `.
    pub fn parse(header_val: &str) -> Option<Self> {
        let trimmed = header_val.trim();
        let rest = if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("digest ") {
            &trimmed[7..]
        } else {
            return None;
        };

        let params = parse_auth_key_values(rest);
        let realm = params.get("realm")?.clone();
        let nonce = params.get("nonce")?.clone();
        let opaque = params.get("opaque").cloned();
        let algorithm = params.get("algorithm").cloned();
        let qop_auth = params
            .get("qop")
            .is_some_and(|q| q.split(',').any(|tok| tok.trim().eq_ignore_ascii_case("auth")));
        let stale = params
            .get("stale")
            .is_some_and(|s| s.trim().eq_ignore_ascii_case("true"));

        Some(Self {
            realm,
            nonce,
            opaque,
            algorithm,
            qop_auth,
            stale,
        })
    }

    /// Computes an RFC 2617 / RFC 7616 / RFC 2069 `Authorization: Digest ...` header value.
    pub fn build_authorization_header(
        &self,
        username: &str,
        password: &str,
        method: &str,
        uri_path_and_query: &str,
        nc: u32,
        cnonce: &str,
    ) -> String {
        let alg_upper = self
            .algorithm
            .as_deref()
            .unwrap_or("MD5")
            .to_ascii_uppercase();
        let use_sha256 = alg_upper.starts_with("SHA-256");
        let is_sess = alg_upper.ends_with("-SESS");

        let hash_fn = |input: &str| -> String {
            if use_sha256 {
                sha256_hex(input.as_bytes())
            } else {
                md5_hex(input.as_bytes())
            }
        };

        let ha1_base = hash_fn(&format!("{username}:{}:{password}", self.realm));
        let ha1 = if is_sess {
            hash_fn(&format!("{ha1_base}:{}:{cnonce}", self.nonce))
        } else {
            ha1_base
        };
        let ha2 = hash_fn(&format!("{method}:{uri_path_and_query}"));

        let nc_hex = format!("{nc:08x}");
        let response = if self.qop_auth {
            hash_fn(&format!(
                "{ha1}:{}:{nc_hex}:{cnonce}:auth:{ha2}",
                self.nonce
            ))
        } else {
            hash_fn(&format!("{ha1}:{}:{ha2}", self.nonce))
        };

        let esc_user = username.replace('\\', "\\\\").replace('"', "\\\"");
        let esc_realm = self.realm.replace('\\', "\\\\").replace('"', "\\\"");
        let esc_nonce = self.nonce.replace('\\', "\\\\").replace('"', "\\\"");
        let esc_uri = uri_path_and_query.replace('"', "%22");

        let mut hdr = format!(
            "Digest username=\"{esc_user}\", realm=\"{esc_realm}\", nonce=\"{esc_nonce}\", uri=\"{esc_uri}\", response=\"{response}\""
        );
        if let Some(ref alg) = self.algorithm {
            hdr.push_str(&format!(", algorithm={alg}"));
        }
        if let Some(ref opq) = self.opaque {
            let esc_opq = opq.replace('\\', "\\\\").replace('"', "\\\"");
            hdr.push_str(&format!(", opaque=\"{esc_opq}\""));
        }
        if self.qop_auth {
            let esc_cnonce = cnonce.replace('\\', "\\\\").replace('"', "\\\"");
            hdr.push_str(&format!(", qop=auth, nc={nc_hex}, cnonce=\"{esc_cnonce}\""));
        }
        hdr
    }
}

/// Parses comma-separated `key="quoted value"` / `key=token` parameters from an HTTP `WWW-Authenticate` header.
fn parse_auth_key_values(input: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let bytes = input.as_bytes();
    let n = bytes.len();
    let mut i = 0;

    while i < n {
        while i < n && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        let key_start = i;
        while i < n && bytes[i] != b'=' && bytes[i] != b',' && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let key = input[key_start..i].trim().to_ascii_lowercase();
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if key.is_empty() || i >= n || bytes[i] != b'=' {
            while i < n && bytes[i] != b',' {
                i += 1;
            }
            continue;
        }
        i += 1; // skip '='
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }

        let val = if i < n && bytes[i] == b'"' {
            i += 1; // skip opening '"'
            let mut buf = String::new();
            while i < n {
                if bytes[i] == b'\\' && i + 1 < n {
                    i += 1;
                    buf.push(bytes[i] as char);
                    i += 1;
                } else if bytes[i] == b'"' {
                    i += 1;
                    break;
                } else {
                    let ch = input[i..].chars().next().unwrap();
                    buf.push(ch);
                    i += ch.len_utf8();
                }
            }
            buf
        } else {
            let val_start = i;
            while i < n && bytes[i] != b',' {
                i += 1;
            }
            input[val_start..i].trim().to_string()
        };

        map.insert(key, val);
    }

    map
}

fn generate_digest_cnonce(nc: u32) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let digest = md5_hex(format!("manifold2:{nanos}:{nc}").as_bytes());
    digest[..16].to_string()
}

/// Cached HTTP Digest challenges per `(normalized_base_uri, username)` so polling & `PUT` file uploads
/// attach a valid preemptive `Authorization: Digest ...` header on the very first transmission.
static CACHED_DIGEST_CHALLENGES: OnceLock<
    Mutex<HashMap<(String, String), (DigestChallenge, u32)>>,
> = OnceLock::new();

fn next_preemptive_digest_header(
    base_uri: &str,
    username: &str,
    password: &str,
    method: &str,
    uri_path_and_query: &str,
) -> Option<String> {
    if password.is_empty() {
        return None;
    }
    let cache = CACHED_DIGEST_CHALLENGES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().ok()?;
    let (challenge, nc) = guard.get_mut(&(base_uri.to_string(), username.to_string()))?;
    *nc = nc.wrapping_add(1).max(1);
    let cur_nc = *nc;
    let cnonce = generate_digest_cnonce(cur_nc);
    Some(challenge.build_authorization_header(
        username,
        password,
        method,
        uri_path_and_query,
        cur_nc,
        &cnonce,
    ))
}

fn store_digest_challenge_and_build_header(
    base_uri: &str,
    username: &str,
    password: &str,
    method: &str,
    uri_path_and_query: &str,
    challenge: DigestChallenge,
) -> String {
    let nc: u32 = 1;
    let cnonce = generate_digest_cnonce(nc);
    let hdr = challenge.build_authorization_header(
        username,
        password,
        method,
        uri_path_and_query,
        nc,
        &cnonce,
    );
    let cache = CACHED_DIGEST_CHALLENGES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(mut guard) = cache.lock() {
        guard.insert(
            (base_uri.to_string(), username.to_string()),
            (challenge, nc),
        );
    }
    hdr
}

fn extract_uri_path_and_query(full_url: &str) -> String {
    if let Ok(parsed) = reqwest::Url::parse(full_url) {
        match parsed.query() {
            Some(q) => format!("{}?{q}", parsed.path()),
            None => parsed.path().to_string(),
        }
    } else {
        "/".to_string()
    }
}

/// Extracts either a `Digest` challenge or indicates `Basic` challenge from response `WWW-Authenticate` headers.
enum ParsedWwwAuth {
    Digest(DigestChallenge),
    Basic,
}

fn parse_response_www_authenticate(headers: &HeaderMap) -> Option<ParsedWwwAuth> {
    let mut saw_basic = false;
    for val in headers.get_all(WWW_AUTHENTICATE) {
        if let Ok(s) = val.to_str() {
            if let Some(dig) = DigestChallenge::parse(s) {
                return Some(ParsedWwwAuth::Digest(dig));
            }
            if s.trim().len() >= 5 && s.trim()[..5].eq_ignore_ascii_case("basic") {
                saw_basic = true;
            }
        }
    }
    if saw_basic {
        Some(ParsedWwwAuth::Basic)
    } else {
        None
    }
}

/// Executes an HTTP request against PrusaLink with automatic HTTP Digest (RFC 2617 / 7616 / 2069)
/// and HTTP Basic challenge-response negotiation, preemptive Digest nonce reuse, and transient connect retries.
fn send_prusalink_request<F>(
    client: &Client,
    cfg: &MoonrakerNetworkConfig,
    base_uri: &str,
    method_name: &str,
    full_url: &str,
    max_connect_attempts: usize,
    mut build_req: F,
) -> Result<reqwest::blocking::Response, reqwest::Error>
where
    F: FnMut(&Client, &str) -> reqwest::blocking::RequestBuilder,
{
    let username = cfg.effective_prusa_username();
    let password = cfg.password.as_str();
    let uri_pq = extract_uri_path_and_query(full_url);

    let preemptive_auth =
        next_preemptive_digest_header(base_uri, username, password, method_name, &uri_pq);

    let attempts = max_connect_attempts.max(1);
    let mut first_resp = None;
    let mut last_err = None;
    for attempt in 0..attempts {
        let mut rb = build_req(client, full_url);
        if let Some(ref auth_hdr) = preemptive_auth {
            rb = rb.header(AUTHORIZATION, auth_hdr);
        }
        match rb.send() {
            Ok(resp) => {
                first_resp = Some(resp);
                break;
            }
            Err(err) => {
                last_err = Some(err);
                if attempt + 1 < attempts {
                    std::thread::sleep(Duration::from_millis(250));
                }
            }
        }
    }

    let resp = match first_resp {
        Some(r) => r,
        None => return Err(last_err.expect("at least one connect attempt executed")),
    };

    if resp.status() != StatusCode::UNAUTHORIZED || password.is_empty() {
        return Ok(resp);
    }

    // Complete HTTP 401 WWW-Authenticate Digest or Basic challenge
    if let Some(challenge) = parse_response_www_authenticate(resp.headers()) {
        let auth_value = match challenge {
            ParsedWwwAuth::Digest(dig) => store_digest_challenge_and_build_header(
                base_uri, username, password, method_name, &uri_pq, dig,
            ),
            ParsedWwwAuth::Basic => {
                let encoded = base64_encode(format!("{username}:{password}").as_bytes());
                format!("Basic {encoded}")
            }
        };
        let retry_rb = build_req(client, full_url).header(AUTHORIZATION, auth_value);
        return retry_rb.send();
    }

    Ok(resp)
}

fn format_prusalink_http_error(
    status: StatusCode,
    cfg: &MoonrakerNetworkConfig,
    endpoint_desc: &str,
    body_snippet: &str,
) -> String {
    if status == StatusCode::UNAUTHORIZED {
        if cfg.password.is_empty() && cfg.api_key.trim().is_empty() {
            return format!(
                "PrusaLink requires authentication (HTTP 401 on {endpoint_desc}): enter your PrusaLink Password (from Printer Settings -> Network -> PrusaLink; default username is 'maker') or API Key."
            );
        }
        return format!(
            "PrusaLink authentication failed (HTTP 401 on {endpoint_desc}): verify Username ('{}') and Password (from Printer Settings -> Network -> PrusaLink).",
            cfg.effective_prusa_username()
        );
    }
    if body_snippet.trim().is_empty() {
        format!("PrusaLink returned HTTP {status} from {endpoint_desc}")
    } else {
        format!(
            "PrusaLink returned HTTP {status} from {endpoint_desc}: {}",
            body_snippet.trim().chars().take(200).collect::<String>()
        )
    }
}

/// Resolves the Web UI URL (Mainsail / Fluidd / PrusaLink) to open in the system browser:
/// - If `cfg.web_ui_uri` is non-empty, normalizes and returns it.
/// - Otherwise derives it from `cfg.printer_uri` (stripping `:7125` for Moonraker so it points to
///   the standard HTTP/HTTPS web interface port on the printer host).
pub fn resolve_printer_web_ui_url(cfg: &MoonrakerNetworkConfig) -> Result<String, String> {
    if !cfg.web_ui_uri.trim().is_empty() {
        return normalize_moonraker_base_uri(&cfg.web_ui_uri);
    }
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    if cfg.protocol == PrinterNetworkProtocol::Moonraker {
        if let Some(stripped) = base.strip_suffix(":7125") {
            return Ok(stripped.to_string());
        } else if base.contains(":7125/") {
            return Ok(base.replacen(":7125/", "/", 1));
        }
    }
    Ok(base)
}

/// Launches the resolved printer Web UI URL in the operating system's default web browser.
pub fn open_printer_web_ui_in_browser(cfg: &MoonrakerNetworkConfig) -> Result<String, String> {
    let url = resolve_printer_web_ui_url(cfg)?;
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(format!("Refusing to open non-HTTP/HTTPS URL: {url}"));
    }

    #[cfg(target_os = "macos")]
    let spawn_res = std::process::Command::new("open").arg(&url).spawn();

    #[cfg(target_os = "linux")]
    let spawn_res = std::process::Command::new("xdg-open").arg(&url).spawn();

    #[cfg(target_os = "windows")]
    let spawn_res = std::process::Command::new("rundll32")
        .arg("url.dll,FileProtocolHandler")
        .arg(&url)
        .spawn();

    spawn_res
        .map(|_| url)
        .map_err(|e| format!("Failed to launch system browser: {e}"))
}

/// Formats a `reqwest::Error` with its full underlying `std::error::Error::source()` chain
/// so the user sees the actual OS/socket/TLS error rather than just `"error sending request"`.
pub fn format_reqwest_error(context: &str, err: &reqwest::Error) -> String {
    use std::error::Error as _;
    let mut chain = vec![err.to_string()];
    let mut curr = err.source();
    while let Some(src) = curr {
        let s = src.to_string();
        if !chain.iter().any(|prev| prev.contains(&s)) {
            chain.push(s);
        }
        curr = src.source();
    }
    let joined = chain.join(": ");
    if joined.contains("No route to host") || joined.contains("os error 65") {
        format!(
            "{context}: {joined} (On macOS, ensure Local Network access is allowed in System Settings -> Privacy & Security -> Local Network, or check printer IP/Wi-Fi)"
        )
    } else {
        format!("{context}: {joined}")
    }
}

/// Resolves `(host, port)` via OS DNS/mDNS (`ToSocketAddrs`), prioritizing IPv4 ahead of IPv6,
/// and probes TCP connectivity (with a brief retry to wake sleeping Wi-Fi / ARP caches on Raspberry Pi hosts)
/// so `reqwest` is handed a known-reachable `SocketAddr`.
fn resolve_and_probe_host_addrs(host: &str, port: u16) -> Vec<SocketAddr> {
    if host.parse::<IpAddr>().is_ok() {
        return Vec::new();
    }
    let mut all_addrs: Vec<SocketAddr> = match (host, port).to_socket_addrs() {
        Ok(iter) => iter.collect(),
        Err(_) => return Vec::new(),
    };
    if all_addrs.is_empty() {
        return Vec::new();
    }
    // Order IPv4 addresses before IPv6 addresses
    all_addrs.sort_by_key(|a| a.is_ipv6());

    // Probe TCP reachability (up to 2 passes so a sleeping Wi-Fi printer has time to answer ARP/NDP)
    for pass in 0..2 {
        for &addr in &all_addrs {
            if TcpStream::connect_timeout(&addr, Duration::from_millis(800)).is_ok() {
                return vec![addr];
            }
        }
        if pass == 0 {
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    let ipv4_addrs: Vec<SocketAddr> = all_addrs.iter().copied().filter(|a| a.is_ipv4()).collect();
    if !ipv4_addrs.is_empty() {
        ipv4_addrs
    } else {
        all_addrs
    }
}

/// Cached shared `reqwest::blocking::Client` so periodic status polling reuses HTTP keep-alive
/// connections and a single background runtime thread rather than spawning a new runtime every poll.
static CACHED_CLIENT: OnceLock<Mutex<Option<(MoonrakerNetworkConfig, Instant, Client)>>> =
    OnceLock::new();

fn invalidate_cached_moonraker_client() {
    if let Some(lock) = CACHED_CLIENT.get() {
        if let Ok(mut guard) = lock.lock() {
            *guard = None;
        }
    }
}

/// Returns a cached `reqwest::blocking::Client` if `cfg` matches and the client was created within
/// the last 60 seconds, or builds a fresh client otherwise.
pub fn get_or_build_moonraker_client(
    cfg: &MoonrakerNetworkConfig,
    timeout: Duration,
) -> Result<Client, String> {
    let cache_mutex = CACHED_CLIENT.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = cache_mutex.lock() {
        if let Some((ref cached_cfg, created_at, ref client)) = *guard {
            if cached_cfg == cfg && created_at.elapsed() < Duration::from_secs(60) {
                return Ok(client.clone());
            }
        }
    }

    let client = build_moonraker_client(cfg, timeout)?;
    if let Ok(mut guard) = cache_mutex.lock() {
        *guard = Some((cfg.clone(), Instant::now(), client.clone()));
    }
    Ok(client)
}

/// Builds a blocking `reqwest::blocking::Client` configured with optional `X-Api-Key`,
/// custom CA root certificate PEM, and mTLS client certificate + key PEM.
/// Also resolves `.local` / LAN hostnames upfront and orders IPv4 addresses ahead of IPv6
/// so dual-stack mDNS hosts where Moonraker only listens on IPv4 (or where IPv6 routes fail)
/// connect immediately without stalling or failing on unreachable IPv6 addresses.
pub fn build_moonraker_client(
    cfg: &MoonrakerNetworkConfig,
    timeout: Duration,
) -> Result<Client, String> {
    let mut builder = Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(5))
        .timeout(timeout)
        .user_agent("Manifold-Slicer/0.1");

    if !cfg.api_key.trim().is_empty() {
        let mut headers = HeaderMap::new();
        let val = HeaderValue::from_str(cfg.api_key.trim())
            .map_err(|e| format!("Invalid characters in Moonraker API key: {e}"))?;
        headers.insert("X-Api-Key", val);
        builder = builder.default_headers(headers);
    }

    // Load optional custom Root / CA certificate(s) from PEM file
    let ca_path = cfg.ca_cert_pem_path.trim();
    if !ca_path.is_empty() {
        let pem_bytes = std::fs::read(ca_path)
            .map_err(|e| format!("Failed to read CA certificate PEM file '{ca_path}': {e}"))?;
        let certs = Certificate::from_pem_bundle(&pem_bytes)
            .or_else(|_| Certificate::from_pem(&pem_bytes).map(|c| vec![c]))
            .map_err(|e| format!("Invalid CA certificate PEM in '{ca_path}': {e}"))?;
        for cert in certs {
            builder = builder.add_root_certificate(cert);
        }
    }

    // Load optional mTLS client certificate + private key from PEM file(s)
    let client_cert_path = cfg.client_cert_pem_path.trim();
    let client_key_path = cfg.client_key_pem_path.trim();
    if !client_cert_path.is_empty() {
        let mut combined_pem = std::fs::read(client_cert_path).map_err(|e| {
            format!("Failed to read client certificate PEM file '{client_cert_path}': {e}")
        })?;
        if !client_key_path.is_empty() && client_key_path != client_cert_path {
            if !combined_pem.ends_with(b"\n") {
                combined_pem.push(b'\n');
            }
            let key_bytes = std::fs::read(client_key_path).map_err(|e| {
                format!("Failed to read client private key PEM file '{client_key_path}': {e}")
            })?;
            combined_pem.extend_from_slice(&key_bytes);
        }
        let identity = Identity::from_pem(&combined_pem).map_err(|e| {
            format!(
                "Invalid mTLS client certificate/key PEM ('{client_cert_path}'): {e}. Ensure both the certificate and private key are PEM-encoded."
            )
        })?;
        builder = builder.identity(identity);
    } else if !client_key_path.is_empty() {
        return Err(
            "Client Private Key PEM is set, but Client Certificate PEM path is empty.".to_string(),
        );
    }

    if let Ok(base) = normalize_moonraker_base_uri(&cfg.printer_uri) {
        if let Ok(parsed_url) = reqwest::Url::parse(&base) {
            if let (Some(host), Some(port)) =
                (parsed_url.host_str(), parsed_url.port_or_known_default())
            {
                let chosen = resolve_and_probe_host_addrs(host, port);
                if !chosen.is_empty() {
                    builder = builder.resolve_to_addrs(host, &chosen);
                }
            }
        }
    }

    builder
        .build()
        .map_err(|e| format!("Failed to initialize HTTP/TLS client: {e}"))
}

/// Parses Moonraker `/server/info` and `/printer/info` JSON payloads into a `MoonrakerConnectionReport`.
pub fn parse_connection_info_json(
    normalized_uri: &str,
    server_json: &Value,
    printer_json: Option<&Value>,
) -> Result<MoonrakerConnectionReport, String> {
    let srv_res = server_json
        .get("result")
        .ok_or_else(|| "Moonraker /server/info response missing 'result' object".to_string())?;
    let moonraker_version = srv_res
        .get("moonraker_version")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let mut klippy_state = srv_res
        .get("klippy_state")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let mut hostname = String::new();
    let mut state_message = String::new();

    if let Some(p_res) = printer_json.and_then(|v| v.get("result")) {
        if let Some(st) = p_res.get("state").and_then(Value::as_str) {
            klippy_state = st.to_string();
        }
        if let Some(h) = p_res.get("hostname").and_then(Value::as_str) {
            hostname = h.to_string();
        }
        if let Some(msg) = p_res.get("state_message").and_then(Value::as_str) {
            state_message = msg.to_string();
        }
    }

    Ok(MoonrakerConnectionReport {
        protocol: PrinterNetworkProtocol::Moonraker,
        normalized_uri: normalized_uri.to_string(),
        moonraker_version,
        klippy_state,
        hostname,
        state_message,
    })
}

/// Parses PrusaLink `/api/v1/info` (or `/api/version`) and `/api/v1/status` JSON payloads
/// into a `MoonrakerConnectionReport`.
pub fn parse_prusalink_info_and_status_json(
    normalized_uri: &str,
    info_json: &Value,
    status_json: Option<&Value>,
) -> MoonrakerConnectionReport {
    let firmware = info_json
        .get("firmware")
        .or_else(|| info_json.get("server"))
        .or_else(|| info_json.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("PrusaLink v1")
        .to_string();

    let hostname = info_json
        .get("hostname")
        .or_else(|| info_json.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let mut printer_state = "ready".to_string();
    let mut telemetry_parts = Vec::new();

    if let Some(printer_obj) = status_json.and_then(|v| v.get("printer")) {
        if let Some(raw_st) = printer_obj.get("state").and_then(Value::as_str) {
            printer_state = raw_st.to_ascii_lowercase();
        }
        if let Some(t_noz) = printer_obj.get("temp_nozzle").and_then(Value::as_f64) {
            telemetry_parts.push(format!("Nozzle {t_noz:.1}°C"));
        }
        if let Some(t_bed) = printer_obj.get("temp_bed").and_then(Value::as_f64) {
            telemetry_parts.push(format!("Bed {t_bed:.1}°C"));
        }
    }

    if let Some(serial) = info_json
        .get("serial")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        if telemetry_parts.is_empty() {
            telemetry_parts.push(format!("SN {serial}"));
        }
    }

    MoonrakerConnectionReport {
        protocol: PrinterNetworkProtocol::PrusaLink,
        normalized_uri: normalized_uri.to_string(),
        moonraker_version: firmware,
        klippy_state: printer_state,
        hostname,
        state_message: telemetry_parts.join(", "),
    }
}

/// Sends a GET request with automatic retry on transient connect/ARP/mDNS wakeup errors.
fn send_get_with_retry(
    client: &Client,
    url: &str,
    max_attempts: usize,
) -> Result<reqwest::blocking::Response, reqwest::Error> {
    let attempts = max_attempts.max(1);
    let mut last_err = None;
    for attempt in 0..attempts {
        match client.get(url).send() {
            Ok(resp) => return Ok(resp),
            Err(err) => {
                last_err = Some(err);
                if attempt + 1 < attempts {
                    std::thread::sleep(Duration::from_millis(250));
                }
            }
        }
    }
    Err(last_err.expect("at least one attempt executed"))
}

/// Tests connectivity to PrusaLink (`/api/v1/info` and `/api/v1/status`) using HTTP Digest / Basic / X-Api-Key.
pub fn test_prusalink_connection(
    cfg: &MoonrakerNetworkConfig,
) -> Result<MoonrakerConnectionReport, String> {
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    let client = get_or_build_moonraker_client(cfg, Duration::from_secs(8))?;

    let info_url = format!("{base}/api/v1/info");
    let mut info_resp = send_prusalink_request(&client, cfg, &base, "GET", &info_url, 3, |c, u| {
        c.get(u)
    })
    .map_err(|e| {
        invalidate_cached_moonraker_client();
        format_reqwest_error(&format!("Failed to reach {info_url}"), &e)
    })?;

    let mut active_info_url = info_url;
    if info_resp.status() == StatusCode::NOT_FOUND {
        let fallback_url = format!("{base}/api/version");
        if let Ok(fb_resp) =
            send_prusalink_request(&client, cfg, &base, "GET", &fallback_url, 1, |c, u| {
                c.get(u)
            })
        {
            info_resp = fb_resp;
            active_info_url = fallback_url;
        }
    }

    let info_status = info_resp.status();
    if !info_status.is_success() {
        let body = info_resp.text().unwrap_or_default();
        return Err(format_prusalink_http_error(
            info_status,
            cfg,
            &active_info_url,
            &body,
        ));
    }

    let info_json: Value = info_resp
        .json()
        .map_err(|e| format_reqwest_error(&format!("Invalid JSON from {active_info_url}"), &e))?;

    let status_url = format!("{base}/api/v1/status");
    let status_json: Option<Value> =
        send_prusalink_request(&client, cfg, &base, "GET", &status_url, 2, |c, u| c.get(u))
            .ok()
            .filter(|r| r.status().is_success())
            .and_then(|r| r.json().ok());

    Ok(parse_prusalink_info_and_status_json(
        &base,
        &info_json,
        status_json.as_ref(),
    ))
}

/// Tests connectivity to the configured printer (`Moonraker`, `PrusaLink`, or `BambuLan`) without uploading or printing anything.
pub fn test_moonraker_connection(
    cfg: &MoonrakerNetworkConfig,
) -> Result<MoonrakerConnectionReport, String> {
    if cfg.protocol == PrinterNetworkProtocol::BambuLan {
        return test_bambu_lan_connection(cfg);
    }
    if cfg.protocol == PrinterNetworkProtocol::PrusaLink {
        return test_prusalink_connection(cfg);
    }

    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    let client = get_or_build_moonraker_client(cfg, Duration::from_secs(8))?;

    let server_url = format!("{base}/server/info");
    let srv_resp = send_get_with_retry(&client, &server_url, 3).map_err(|e| {
        invalidate_cached_moonraker_client();
        format_reqwest_error(&format!("Failed to reach {server_url}"), &e)
    })?;
    if !srv_resp.status().is_success() {
        return Err(format!(
            "Moonraker returned HTTP {} from {server_url}",
            srv_resp.status()
        ));
    }
    let srv_json: Value = srv_resp
        .json()
        .map_err(|e| format_reqwest_error(&format!("Invalid JSON from {server_url}"), &e))?;

    let printer_url = format!("{base}/printer/info");
    let printer_json: Option<Value> = send_get_with_retry(&client, &printer_url, 2)
        .ok()
        .filter(|r| r.status().is_success())
        .and_then(|r| r.json().ok());

    parse_connection_info_json(&base, &srv_json, printer_json.as_ref())
}

/// Computes remaining print time (ETA in seconds) from current progress `[0.0 ..= 1.0]`,
/// active print duration `print_duration_s`, and optional slicer metadata `estimated_time`.
pub fn compute_print_eta_seconds(
    state: &str,
    progress: f32,
    print_duration_s: f64,
    slicer_estimated_time_s: Option<f64>,
) -> Option<f64> {
    if !state.eq_ignore_ascii_case("printing") && !state.eq_ignore_ascii_case("paused") {
        return None;
    }
    let p = (progress as f64).clamp(0.0, 1.0);
    if p >= 0.9999 {
        return Some(0.0);
    }
    let rem_fraction = 1.0 - p;
    let linear_eta = if p >= 0.01 && print_duration_s > 1.0 {
        Some((print_duration_s * rem_fraction) / p)
    } else {
        None
    };
    let metadata_eta = slicer_estimated_time_s
        .filter(|&est| est > 1.0)
        .map(|est| (est * rem_fraction).max(0.0));

    match (metadata_eta, linear_eta) {
        (Some(m_eta), Some(l_eta)) => {
            // Smoothly blend from slicer metadata estimate early in the print toward empirical progress rate
            let w_linear = p.clamp(0.0, 1.0);
            Some((1.0 - w_linear) * m_eta + w_linear * l_eta)
        }
        (Some(m_eta), None) => Some(m_eta),
        (None, Some(l_eta)) => Some(l_eta),
        (None, None) => None,
    }
}

/// Parses Moonraker's `/printer/objects/query?print_stats&virtual_sdcard&display_status` JSON response.
pub fn parse_printer_objects_status_json(
    query_json: &Value,
    slicer_estimated_time_s: Option<f64>,
) -> Result<MoonrakerPrintStatus, String> {
    let status_obj = query_json
        .get("result")
        .and_then(|r| r.get("status"))
        .ok_or_else(|| {
            "Moonraker /printer/objects/query response missing 'result.status'".to_string()
        })?;

    let print_stats = status_obj.get("print_stats");
    let vsd = status_obj.get("virtual_sdcard");
    let disp = status_obj.get("display_status");

    let state = print_stats
        .and_then(|ps| ps.get("state"))
        .and_then(Value::as_str)
        .unwrap_or("standby")
        .to_string();
    let filename = print_stats
        .and_then(|ps| ps.get("filename"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let print_duration_s = print_stats
        .and_then(|ps| ps.get("print_duration"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let total_duration_s = print_stats
        .and_then(|ps| ps.get("total_duration"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let vsd_prog = vsd
        .and_then(|v| v.get("progress"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let disp_prog = disp
        .and_then(|d| d.get("progress"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    let progress = if state.eq_ignore_ascii_case("complete") {
        1.0
    } else {
        vsd_prog.max(disp_prog).clamp(0.0, 1.0)
    };

    let message = disp
        .and_then(|d| d.get("message"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            print_stats
                .and_then(|ps| ps.get("message"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("")
        .to_string();

    let eta_remaining_s =
        compute_print_eta_seconds(&state, progress, print_duration_s, slicer_estimated_time_s);

    Ok(MoonrakerPrintStatus {
        state,
        filename,
        progress,
        print_duration_s,
        total_duration_s,
        slicer_estimated_time_s,
        eta_remaining_s,
        message,
    })
}

/// Percent-encodes a filename query parameter for `/server/files/metadata?filename=...`.
fn url_encode_query_value(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Percent-encodes a single URL path segment (for PrusaLink `PUT /api/v1/files/{storage}/{filename}`).
fn url_encode_path_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Maps PrusaLink v1 `printer.state` strings (`IDLE`, `READY`, `PRINTING`, `PAUSED`, `FINISHED`, `STOPPED`, `ERROR`, `ATTENTION`)
/// into Manifold2's normalized status badge states.
fn map_prusalink_state(raw_state: &str) -> String {
    match raw_state.trim().to_ascii_uppercase().as_str() {
        "IDLE" | "READY" | "BUSY" => "standby".to_string(),
        "PRINTING" => "printing".to_string(),
        "PAUSED" => "paused".to_string(),
        "FINISHED" => "complete".to_string(),
        "STOPPED" => "cancelled".to_string(),
        "ERROR" | "ATTENTION" => "error".to_string(),
        other => other.to_ascii_lowercase(),
    }
}

/// Parses PrusaLink `/api/v1/status` and optional `/api/v1/job` JSON payloads into `MoonrakerPrintStatus`.
pub fn parse_prusalink_status_and_job_json(
    status_json: &Value,
    job_json: Option<&Value>,
) -> Result<MoonrakerPrintStatus, String> {
    let printer_obj = status_json
        .get("printer")
        .ok_or_else(|| "PrusaLink /api/v1/status response missing 'printer' object".to_string())?;

    let raw_state = printer_obj
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or("IDLE");
    let state = map_prusalink_state(raw_state);

    let status_job = status_json.get("job");
    let progress_pct = status_job
        .and_then(|j| j.get("progress"))
        .and_then(Value::as_f64)
        .or_else(|| job_json.and_then(|j| j.get("progress")).and_then(Value::as_f64))
        .unwrap_or(0.0);

    let progress = if state == "complete" {
        1.0
    } else if progress_pct > 1.0001 {
        (progress_pct / 100.0).clamp(0.0, 1.0) as f32
    } else {
        // Note: PrusaLink reports `progress` as 0..100 (e.g. 42.5 -> 42.5%, 1.0 -> 1.0%).
        // Only treat <= 1.0 as percentage too since PrusaLink OpenAPI defines progress in [0..100].
        (progress_pct / 100.0).clamp(0.0, 1.0) as f32
    };

    let print_duration_s = status_job
        .and_then(|j| j.get("time_printing"))
        .and_then(Value::as_f64)
        .or_else(|| {
            job_json
                .and_then(|j| j.get("time_printing"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(0.0);

    let reported_remaining_s = status_job
        .and_then(|j| j.get("time_remaining"))
        .and_then(Value::as_f64)
        .or_else(|| {
            job_json
                .and_then(|j| j.get("time_remaining"))
                .and_then(Value::as_f64)
        });

    let filename = job_json
        .and_then(|j| j.get("file"))
        .and_then(|f| {
            f.get("display_name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .or_else(|| f.get("name").and_then(Value::as_str))
        })
        .unwrap_or("")
        .to_string();

    let eta_remaining_s = if state == "printing" || state == "paused" {
        reported_remaining_s
            .filter(|&rem| rem >= 0.0)
            .or_else(|| compute_print_eta_seconds(&state, progress, print_duration_s, None))
    } else {
        None
    };

    let mut msg_parts = Vec::new();
    if let Some(t_noz) = printer_obj.get("temp_nozzle").and_then(Value::as_f64) {
        if let Some(tgt_noz) = printer_obj
            .get("target_nozzle")
            .and_then(Value::as_f64)
            .filter(|&t| t > 0.5)
        {
            msg_parts.push(format!("Nozzle {t_noz:.0}/{tgt_noz:.0}°C"));
        } else {
            msg_parts.push(format!("Nozzle {t_noz:.0}°C"));
        }
    }
    if let Some(t_bed) = printer_obj.get("temp_bed").and_then(Value::as_f64) {
        if let Some(tgt_bed) = printer_obj
            .get("target_bed")
            .and_then(Value::as_f64)
            .filter(|&t| t > 0.5)
        {
            msg_parts.push(format!("Bed {t_bed:.0}/{tgt_bed:.0}°C"));
        } else {
            msg_parts.push(format!("Bed {t_bed:.0}°C"));
        }
    }

    Ok(MoonrakerPrintStatus {
        state,
        filename,
        progress,
        print_duration_s,
        total_duration_s: print_duration_s,
        slicer_estimated_time_s: reported_remaining_s.map(|rem| print_duration_s + rem),
        eta_remaining_s,
        message: msg_parts.join(" | "),
    })
}

/// Queries PrusaLink (`/api/v1/status` + `/api/v1/job`) for live print status, temperatures, and active filename.
pub fn query_prusalink_print_status(
    cfg: &MoonrakerNetworkConfig,
    cached_metadata: Option<(String, Option<f64>)>,
) -> (
    Result<MoonrakerPrintStatus, String>,
    Option<(String, Option<f64>)>,
) {
    let base = match normalize_moonraker_base_uri(&cfg.printer_uri) {
        Ok(u) => u,
        Err(e) => return (Err(e), cached_metadata),
    };
    let client = match get_or_build_moonraker_client(cfg, Duration::from_secs(8)) {
        Ok(c) => c,
        Err(e) => return (Err(e), cached_metadata),
    };

    let status_url = format!("{base}/api/v1/status");
    let resp = match send_prusalink_request(&client, cfg, &base, "GET", &status_url, 2, |c, u| {
        c.get(u)
    }) {
        Ok(r) => r,
        Err(e) => {
            invalidate_cached_moonraker_client();
            return (
                Err(format_reqwest_error("Offline", &e)),
                cached_metadata,
            );
        }
    };

    let status_code = resp.status();
    if !status_code.is_success() {
        let body = resp.text().unwrap_or_default();
        return (
            Err(format_prusalink_http_error(
                status_code,
                cfg,
                &status_url,
                &body,
            )),
            cached_metadata,
        );
    }

    let status_json: Value = match resp.json() {
        Ok(j) => j,
        Err(e) => {
            return (
                Err(format_reqwest_error("Invalid PrusaLink status JSON", &e)),
                cached_metadata,
            )
        }
    };

    let has_active_job = status_json.get("job").is_some_and(|j| !j.is_null());
    let job_json: Option<Value> = if has_active_job {
        let job_url = format!("{base}/api/v1/job");
        send_prusalink_request(&client, cfg, &base, "GET", &job_url, 1, |c, u| c.get(u))
            .ok()
            .filter(|r| r.status() == StatusCode::OK)
            .and_then(|r| r.json::<Value>().ok())
    } else {
        None
    };

    let parsed = parse_prusalink_status_and_job_json(&status_json, job_json.as_ref());
    (parsed, cached_metadata)
}

/// Queries Moonraker, PrusaLink, or Bambu LAN for live print status.
pub fn query_moonraker_print_status(
    cfg: &MoonrakerNetworkConfig,
    cached_metadata: Option<(String, Option<f64>)>,
) -> (
    Result<MoonrakerPrintStatus, String>,
    Option<(String, Option<f64>)>,
) {
    if cfg.protocol == PrinterNetworkProtocol::BambuLan {
        return query_bambu_lan_print_status(cfg, cached_metadata);
    }
    if cfg.protocol == PrinterNetworkProtocol::PrusaLink {
        return query_prusalink_print_status(cfg, cached_metadata);
    }

    let base = match normalize_moonraker_base_uri(&cfg.printer_uri) {
        Ok(u) => u,
        Err(e) => return (Err(e), cached_metadata),
    };
    let client = match get_or_build_moonraker_client(cfg, Duration::from_secs(8)) {
        Ok(c) => c,
        Err(e) => return (Err(e), cached_metadata),
    };

    let query_url =
        format!("{base}/printer/objects/query?print_stats&virtual_sdcard&display_status");
    let resp = match send_get_with_retry(&client, &query_url, 2) {
        Ok(r) => r,
        Err(e) => {
            invalidate_cached_moonraker_client();
            return (
                Err(format_reqwest_error("Offline", &e)),
                cached_metadata,
            );
        }
    };
    if !resp.status().is_success() {
        return (
            Err(format!("Moonraker HTTP {}", resp.status())),
            cached_metadata,
        );
    }
    let query_json: Value = match resp.json() {
        Ok(j) => j,
        Err(e) => {
            return (
                Err(format_reqwest_error("Invalid status JSON", &e)),
                cached_metadata,
            )
        }
    };

    let prelim = match parse_printer_objects_status_json(&query_json, None) {
        Ok(s) => s,
        Err(e) => return (Err(e), cached_metadata),
    };

    let mut updated_cache = cached_metadata;
    let mut est_time_s: Option<f64> = None;

    if !prelim.filename.is_empty() {
        if let Some((ref cached_name, cached_est)) = updated_cache {
            if cached_name == &prelim.filename {
                est_time_s = cached_est;
            }
        }
        if updated_cache
            .as_ref()
            .is_none_or(|(name, _)| name != &prelim.filename)
        {
            let encoded = url_encode_query_value(&prelim.filename);
            let meta_url = format!("{base}/server/files/metadata?filename={encoded}");
            let fetched_est = client
                .get(&meta_url)
                .send()
                .ok()
                .filter(|r| r.status().is_success())
                .and_then(|r| r.json::<Value>().ok())
                .and_then(|v| {
                    v.get("result")
                        .and_then(|res| res.get("estimated_time"))
                        .and_then(Value::as_f64)
                });
            est_time_s = fetched_est;
            updated_cache = Some((prelim.filename.clone(), fetched_est));
        }
    }

    let final_status = parse_printer_objects_status_json(&query_json, est_time_s);
    (final_status, updated_cache)
}

/// Selects the preferred writable storage target from PrusaLink `/api/v1/storage` JSON
/// (e.g. `"usb"` on Prusa CORE One / MK4 / XL Buddy boards, or `"local"` on RPi PrusaLink).
fn select_prusalink_writable_storage(storage_json: &Value) -> String {
    let entries = storage_json
        .get("storage_list")
        .and_then(Value::as_array)
        .or_else(|| storage_json.as_array());

    if let Some(list) = entries {
        for item in list {
            let available = item
                .get("available")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let read_only = item
                .get("read_only")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !available || read_only {
                continue;
            }
            if let Some(raw_path) = item
                .get("path")
                .and_then(Value::as_str)
                .or_else(|| item.get("type").and_then(Value::as_str))
            {
                let clean = raw_path.trim_matches('/');
                if !clean.is_empty() {
                    return clean.to_string();
                }
            }
        }
    }
    "usb".to_string()
}

fn build_streaming_multipart_part(
    payload: Arc<[u8]>,
    clean_filename: String,
    tracker: Option<&UploadProgressTracker>,
    frac_start: f32,
    frac_end: f32,
    prefix: String,
    waiting_stage: String,
) -> Result<multipart::Part, String> {
    let total = payload.len() as u64;
    let reader = UploadProgressReader::new(
        payload,
        tracker,
        frac_start,
        frac_end,
        prefix,
        waiting_stage,
    );
    multipart::Part::reader_with_length(reader, total)
        .file_name(clean_filename)
        .mime_str("application/octet-stream")
        .map_err(|e| format!("Failed to construct multipart file payload: {e}"))
}

/// Uploads G-code to a PrusaLink printer via `PUT /api/v1/files/{storage}/{filename}`
/// (`Print-After-Upload: ?1` / `?0`, `Overwrite: ?1`), pre-negotiating HTTP Digest auth
/// via `GET /api/v1/storage` and falling back to legacy `POST /api/files/local` on HTTP 404/405.
pub fn upload_gcode_to_prusalink(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
) -> Result<MoonrakerUploadReport, String> {
    upload_gcode_to_prusalink_with_progress(cfg, filename, gcode_text, start_print, None)
}

/// Streaming variant of [`upload_gcode_to_prusalink`] that reports transfer progress to `progress`.
pub fn upload_gcode_to_prusalink_with_progress(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
    progress: Option<&UploadProgressTracker>,
) -> Result<MoonrakerUploadReport, String> {
    if let Some(tr) = progress {
        tr.set_stage(0.04, "Connecting to PrusaLink...");
    }
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    let client = build_moonraker_client(cfg, Duration::from_secs(120))?;

    let basename = filename
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("manifold2_print.gcode");
    let clean_filename = if basename.ends_with(".gcode")
        || basename.ends_with(".bgcode")
        || basename.ends_with(".gco")
    {
        basename.to_string()
    } else {
        format!("{basename}.gcode")
    };

    // Step 1: Query `/api/v1/storage` to discover writable storage ("usb" / "local") AND
    // pre-negotiate the HTTP Digest `nonce` before streaming the G-code payload.
    let storage_url = format!("{base}/api/v1/storage");
    let mut storage_target = "usb".to_string();
    if let Ok(st_resp) =
        send_prusalink_request(&client, cfg, &base, "GET", &storage_url, 2, |c, u| c.get(u))
    {
        if st_resp.status() == StatusCode::UNAUTHORIZED {
            let body = st_resp.text().unwrap_or_default();
            return Err(format_prusalink_http_error(
                StatusCode::UNAUTHORIZED,
                cfg,
                &storage_url,
                &body,
            ));
        }
        if st_resp.status().is_success() {
            if let Ok(st_json) = st_resp.json::<Value>() {
                storage_target = select_prusalink_writable_storage(&st_json);
            }
        }
    }

    let payload: Arc<[u8]> = gcode_text.into_bytes().into();
    let bytes_uploaded = payload.len();
    let encoded_name = url_encode_path_segment(&clean_filename);
    let put_url = format!("{base}/api/v1/files/{storage_target}/{encoded_name}");
    let print_after_hdr = if start_print { "?1" } else { "?0" };
    let upload_prefix = format!("Uploading '{clean_filename}' to PrusaLink");
    let waiting_stage = "Waiting for PrusaLink confirmation...".to_string();

    let put_resp = send_prusalink_request(&client, cfg, &base, "PUT", &put_url, 1, |c, u| {
        let reader = UploadProgressReader::new(
            Arc::clone(&payload),
            progress,
            0.08,
            0.95,
            upload_prefix.clone(),
            waiting_stage.clone(),
        );
        c.put(u)
            .header("Content-Type", "text/x.gcode")
            .header("Overwrite", "?1")
            .header("Print-After-Upload", print_after_hdr)
            .body(Body::sized(reader, bytes_uploaded as u64))
    })
    .map_err(|e| format_reqwest_error(&format!("Failed to upload G-code to {put_url}"), &e))?;

    let put_status = put_resp.status();
    if put_status.is_success() {
        if let Some(tr) = progress {
            tr.set_stage(1.0, "Upload complete.");
        }
        return Ok(MoonrakerUploadReport {
            protocol: PrinterNetworkProtocol::PrusaLink,
            remote_path: format!("{storage_target}/{clean_filename}"),
            bytes_uploaded,
            print_started: start_print,
        });
    }

    // Legacy OctoPrint-compatible PrusaLink fallback (`POST /api/files/local`) on 404 / 405
    if put_status == StatusCode::NOT_FOUND || put_status == StatusCode::METHOD_NOT_ALLOWED {
        let legacy_url = format!("{base}/api/files/local");
        let legacy_resp =
            send_prusalink_request(&client, cfg, &base, "POST", &legacy_url, 1, |c, u| {
                let part = build_streaming_multipart_part(
                    Arc::clone(&payload),
                    clean_filename.clone(),
                    progress,
                    0.08,
                    0.95,
                    upload_prefix.clone(),
                    waiting_stage.clone(),
                )
                .expect("valid mime literal");
                let form = multipart::Form::new()
                    .text("select", "true")
                    .text("print", if start_print { "true" } else { "false" })
                    .part("file", part);
                c.post(u).multipart(form)
            })
            .map_err(|e| {
                format_reqwest_error(&format!("Failed to upload G-code to {legacy_url}"), &e)
            })?;

        let legacy_status = legacy_resp.status();
        if legacy_status.is_success() {
            if let Some(tr) = progress {
                tr.set_stage(1.0, "Upload complete.");
            }
            return Ok(MoonrakerUploadReport {
                protocol: PrinterNetworkProtocol::PrusaLink,
                remote_path: clean_filename,
                bytes_uploaded,
                print_started: start_print,
            });
        }
        let body = legacy_resp.text().unwrap_or_default();
        return Err(format_prusalink_http_error(
            legacy_status,
            cfg,
            &legacy_url,
            &body,
        ));
    }

    let body = put_resp.text().unwrap_or_default();
    Err(format_prusalink_http_error(put_status, cfg, &put_url, &body))
}

/// Uploads G-code to a Moonraker (`POST /server/files/upload`), PrusaLink (`PUT /api/v1/files/{storage}/{path}`),
/// or Bambu Lab LAN Mode (`FTPS :990` `.gcode.3mf` + `MQTT-TLS :8883` `project_file`) printer and optionally starts printing immediately (`start_print = true`).
pub fn upload_gcode_to_moonraker(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
) -> Result<MoonrakerUploadReport, String> {
    upload_gcode_to_moonraker_with_progress(cfg, filename, gcode_text, start_print, None)
}

/// Streaming variant of [`upload_gcode_to_moonraker`] that updates `progress` across Moonraker,
/// PrusaLink, and Bambu LAN transfers.
pub fn upload_gcode_to_moonraker_with_progress(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
    progress: Option<&UploadProgressTracker>,
) -> Result<MoonrakerUploadReport, String> {
    if cfg.protocol == PrinterNetworkProtocol::BambuLan {
        return upload_gcode_to_bambu_lan_with_progress(
            cfg, filename, gcode_text, start_print, progress,
        );
    }
    if cfg.protocol == PrinterNetworkProtocol::PrusaLink {
        return upload_gcode_to_prusalink_with_progress(
            cfg, filename, gcode_text, start_print, progress,
        );
    }

    if let Some(tr) = progress {
        tr.set_stage(0.04, "Connecting to Moonraker...");
    }
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    let client = build_moonraker_client(cfg, Duration::from_secs(90))?;

    let clean_filename = if filename.ends_with(".gcode") || filename.ends_with(".nc") {
        filename.to_string()
    } else {
        format!("{filename}.gcode")
    };
    let payload: Arc<[u8]> = gcode_text.into_bytes().into();
    let bytes_uploaded = payload.len();

    let file_part = build_streaming_multipart_part(
        payload,
        clean_filename.clone(),
        progress,
        0.06,
        0.95,
        format!("Uploading '{clean_filename}' to Moonraker"),
        "Waiting for Moonraker confirmation...".to_string(),
    )?;

    let form = multipart::Form::new()
        .text("root", "gcodes")
        .text("print", if start_print { "true" } else { "false" })
        .part("file", file_part);

    let upload_url = format!("{base}/server/files/upload");
    let resp = client
        .post(&upload_url)
        .multipart(form)
        .send()
        .map_err(|e| format_reqwest_error(&format!("Failed to upload G-code to {upload_url}"), &e))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().unwrap_or_default();
        return Err(format!(
            "Moonraker upload failed (HTTP {status}): {}",
            body.chars().take(240).collect::<String>()
        ));
    }

    let resp_json: Value = resp.json().unwrap_or(Value::Null);
    let remote_path = resp_json
        .get("item")
        .and_then(|item| item.get("path"))
        .and_then(Value::as_str)
        .or_else(|| {
            resp_json
                .get("result")
                .and_then(|r| r.get("item"))
                .and_then(|item| item.get("path"))
                .and_then(Value::as_str)
        })
        .unwrap_or(&clean_filename)
        .to_string();

    if let Some(tr) = progress {
        tr.set_stage(1.0, "Upload complete.");
    }

    Ok(MoonrakerUploadReport {
        protocol: PrinterNetworkProtocol::Moonraker,
        remote_path,
        bytes_uploaded,
        print_started: start_print,
    })
}

// ============================================================================
// Bambu Lab LAN Mode Support (Implicit FTPS :990 + MQTT v3.1.1 over TLS :8883)
// ============================================================================

/// TLS server certificate verifier that accepts Bambu Lab printers' self-signed device certificates
/// on the local LAN (`BambuLab` CA / per-printer self-signed X.509 certificates on ports 990 and 8883).
#[derive(Debug)]
struct BambuLanSelfSignedCertVerifier;

impl rustls::client::danger::ServerCertVerifier for BambuLanSelfSignedCertVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls_pki_types::CertificateDer<'_>,
        _intermediates: &[rustls_pki_types::CertificateDer<'_>],
        _server_name: &rustls_pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls_pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls_pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls_pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &rustls::crypto::ring::default_provider().signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn build_bambu_lan_tls_config() -> std::sync::Arc<rustls::ClientConfig> {
    let provider = std::sync::Arc::new(rustls::crypto::ring::default_provider());
    let mut cfg = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("default TLS protocol versions")
        .dangerous()
        .with_custom_certificate_verifier(std::sync::Arc::new(BambuLanSelfSignedCertVerifier))
        .with_no_client_auth();
    // Enable TLS session resumption so implicit FTPS data channels on Bambu vsftpd reuse the control session
    cfg.resumption = rustls::client::Resumption::in_memory_sessions(32);
    std::sync::Arc::new(cfg)
}

/// Extracts `(host, optional_port)` from `cfg.printer_uri` for Bambu Lab LAN Mode.
pub fn parse_bambu_lan_host_and_port(raw_uri: &str, default_port: u16) -> Result<(String, u16), String> {
    let trimmed = raw_uri.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(
            "Bambu Lab Printer IP/Host is empty. Enter the printer's LAN IP address (e.g. 192.168.1.50)."
                .to_string(),
        );
    }
    let without_scheme = trimmed
        .strip_prefix("ftps://")
        .or_else(|| trimmed.strip_prefix("mqtts://"))
        .or_else(|| trimmed.strip_prefix("https://"))
        .or_else(|| trimmed.strip_prefix("http://"))
        .unwrap_or(trimmed);
    let host_port = without_scheme.split('/').next().unwrap_or(without_scheme).trim();
    if host_port.is_empty() {
        return Err("Bambu Lab Printer IP/Host is empty.".to_string());
    }
    if let Some((h, p_str)) = host_port.rsplit_once(':') {
        if !h.contains(':') {
            if let Ok(p) = p_str.parse::<u16>() {
                return Ok((h.trim().to_string(), p));
            }
        }
    }
    Ok((host_port.to_string(), default_port))
}

fn connect_bambu_tls_stream(
    host: &str,
    port: u16,
    tls_cfg: std::sync::Arc<rustls::ClientConfig>,
    timeout: Duration,
) -> Result<rustls::StreamOwned<rustls::ClientConnection, TcpStream>, String> {
    let mut addrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("Failed to resolve Bambu printer host '{host}:{port}': {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!("Could not resolve any IP address for '{host}:{port}'"));
    }
    addrs.sort_by_key(|a| a.is_ipv6());

    let mut last_err = None;
    let mut tcp_opt = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, Duration::from_secs(5)) {
            Ok(s) => {
                let _ = s.set_read_timeout(Some(timeout));
                let _ = s.set_write_timeout(Some(timeout));
                let _ = s.set_nodelay(true);
                tcp_opt = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let tcp = tcp_opt.ok_or_else(|| {
        format!(
            "Failed to connect to Bambu printer at {host}:{port}: {}",
            last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unreachable".to_string())
        )
    })?;

    let server_name = rustls_pki_types::ServerName::try_from(host.to_string())
        .or_else(|_| rustls_pki_types::ServerName::try_from("bambulab.local".to_string()))
        .expect("valid fallback server name");
    let conn = rustls::ClientConnection::new(tls_cfg, server_name)
        .map_err(|e| format!("Failed to initialize TLS session to {host}:{port}: {e}"))?;
    Ok(rustls::StreamOwned::new(conn, tcp))
}

/// Reads an FTP response status code (e.g. `220`, `331`, `230`, `227`, `150`, `226`) and full message text
/// from a buffered FTP control stream, handling multi-line (`XYZ-...` to `XYZ ...`) replies.
fn read_ftp_response<R: std::io::BufRead>(reader: &mut R) -> Result<(u16, String), String> {
    let mut full_msg = String::new();
    let mut expected_code: Option<u16> = None;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .map_err(|e| format!("FTPS control channel read error: {e}"))?;
        if n == 0 {
            return Err(format!(
                "FTPS server closed control connection unexpectedly (received: '{}')",
                full_msg.trim()
            ));
        }
        full_msg.push_str(&line);
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.len() >= 3 {
            if let Ok(code) = trimmed[..3].parse::<u16>() {
                let sep = trimmed.as_bytes().get(3).copied().unwrap_or(b' ');
                if expected_code.is_none() {
                    if sep == b' ' {
                        return Ok((code, full_msg));
                    } else if sep == b'-' {
                        expected_code = Some(code);
                    }
                } else if expected_code == Some(code) && sep == b' ' {
                    return Ok((code, full_msg));
                }
            }
        }
    }
}

fn send_ftp_command<S: std::io::Read + std::io::Write>(
    reader: &mut std::io::BufReader<S>,
    cmd: &str,
) -> Result<(u16, String), String> {
    let line = format!("{cmd}\r\n");
    reader
        .get_mut()
        .write_all(line.as_bytes())
        .map_err(|e| format!("FTPS write error sending '{}': {e}", cmd.split_whitespace().next().unwrap_or("")))?;
    reader
        .get_mut()
        .flush()
        .map_err(|e| format!("FTPS flush error: {e}"))?;
    read_ftp_response(reader)
}

/// Parses an FTP `227 Entering Passive Mode (h1,h2,h3,h4,p1,p2)` response to extract the passive data port.
pub fn parse_ftp_pasv_port(resp: &str) -> Result<u16, String> {
    let start = resp
        .find('(')
        .ok_or_else(|| format!("Malformed FTPS PASV response (missing '('): {}", resp.trim()))?;
    let end = resp[start..]
        .find(')')
        .map(|i| start + i)
        .ok_or_else(|| format!("Malformed FTPS PASV response (missing ')'): {}", resp.trim()))?;
    let nums: Vec<u16> = resp[start + 1..end]
        .split(',')
        .filter_map(|t| t.trim().parse::<u16>().ok())
        .collect();
    if nums.len() < 6 {
        return Err(format!("Malformed FTPS PASV address tuple: {}", resp.trim()));
    }
    Ok((nums[4] << 8) | (nums[5] & 0xFF))
}

/// Uploads `file_bytes` to a Bambu Lab printer over implicit FTPS (port 990 by default)
/// authenticating as `bblp` + `<8-char Access Code>`.
pub fn upload_bytes_to_bambu_ftps(
    cfg: &MoonrakerNetworkConfig,
    remote_filename: &str,
    file_bytes: &[u8],
) -> Result<String, String> {
    upload_bytes_to_bambu_ftps_with_progress(cfg, remote_filename, file_bytes, 0.95, None)
}

/// Chunked FTPS uploader that reports progress across `[0.16 ..= frac_end]`.
pub fn upload_bytes_to_bambu_ftps_with_progress(
    cfg: &MoonrakerNetworkConfig,
    remote_filename: &str,
    file_bytes: &[u8],
    frac_end: f32,
    progress: Option<&UploadProgressTracker>,
) -> Result<String, String> {
    use std::io::Write as _;
    let (host, port) = parse_bambu_lan_host_and_port(&cfg.printer_uri, 990)?;
    let access_code = cfg.effective_bambu_access_code();
    if access_code.is_empty() {
        return Err(
            "Bambu Lab Access Code is empty. Enter the 8-character LAN Access Code from the printer screen (Settings -> LAN Only Mode)."
                .to_string(),
        );
    }
    let username = cfg.effective_bambu_username();
    let tls_cfg = build_bambu_lan_tls_config();

    let ctrl_stream = connect_bambu_tls_stream(&host, port, tls_cfg.clone(), Duration::from_secs(20))?;
    let mut ctrl_reader = std::io::BufReader::new(ctrl_stream);

    // 1. Read 220 greeting
    let (greet_code, greet_msg) = read_ftp_response(&mut ctrl_reader)?;
    if greet_code != 220 {
        return Err(format!("Bambu FTPS unexpected greeting ({greet_code}): {}", greet_msg.trim()));
    }

    // 2. Authenticate USER bblp / PASS <access_code>
    let (u_code, u_msg) = send_ftp_command(&mut ctrl_reader, &format!("USER {username}"))?;
    if u_code == 331 {
        let (p_code, p_msg) = send_ftp_command(&mut ctrl_reader, &format!("PASS {access_code}"))?;
        if p_code != 230 {
            return Err(format!(
                "Bambu FTPS authentication failed ({p_code}): {}. Check the 8-character LAN Access Code.",
                p_msg.trim()
            ));
        }
    } else if u_code != 230 {
        return Err(format!("Bambu FTPS USER rejected ({u_code}): {}", u_msg.trim()));
    }

    // 3. Negotiate TLS data channel protection (PBSZ 0, PROT P, TYPE I, PASV)
    let _ = send_ftp_command(&mut ctrl_reader, "PBSZ 0")?;
    let _ = send_ftp_command(&mut ctrl_reader, "PROT P")?;
    let (t_code, t_msg) = send_ftp_command(&mut ctrl_reader, "TYPE I")?;
    if t_code != 200 {
        return Err(format!("Bambu FTPS TYPE I failed ({t_code}): {}", t_msg.trim()));
    }

    let (pasv_code, pasv_msg) = send_ftp_command(&mut ctrl_reader, "PASV")?;
    if pasv_code != 227 {
        return Err(format!("Bambu FTPS PASV failed ({pasv_code}): {}", pasv_msg.trim()));
    }
    let data_port = parse_ftp_pasv_port(&pasv_msg)?;

    // 4. Issue STOR and open implicit TLS data stream (reusing TLS session cache)
    let clean_name = remote_filename.trim_start_matches('/');
    let (stor_code, stor_msg) = send_ftp_command(&mut ctrl_reader, &format!("STOR {clean_name}"))?;
    if stor_code != 150 && stor_code != 125 {
        return Err(format!(
            "Bambu FTPS STOR '{clean_name}' rejected ({stor_code}): {}",
            stor_msg.trim()
        ));
    }

    {
        let mut data_stream =
            connect_bambu_tls_stream(&host, data_port, tls_cfg, Duration::from_secs(90))?;
        let total = file_bytes.len() as u64;
        let mut sent = 0u64;
        let prefix = format!("Uploading '{clean_name}' via FTPS");
        for chunk in file_bytes.chunks(16 * 1024) {
            data_stream
                .write_all(chunk)
                .map_err(|e| format!("Bambu FTPS data transfer error: {e}"))?;
            sent += chunk.len() as u64;
            if let Some(tr) = progress {
                tr.set_bytes_progress(sent, total, 0.16, frac_end, &prefix);
            }
        }
        if let Some(tr) = progress {
            tr.set_stage(frac_end, "Finalizing FTPS transfer...");
        }
        data_stream
            .flush()
            .map_err(|e| format!("Bambu FTPS data flush error: {e}"))?;
        data_stream.conn.send_close_notify();
        let _ = data_stream.conn.complete_io(&mut data_stream.sock);
    }

    let (done_code, done_msg) = read_ftp_response(&mut ctrl_reader)?;
    if done_code != 226 && done_code != 250 {
        return Err(format!(
            "Bambu FTPS transfer did not complete cleanly ({done_code}): {}",
            done_msg.trim()
        ));
    }
    let _ = send_ftp_command(&mut ctrl_reader, "QUIT");
    Ok(clean_name.to_string())
}

// ============================================================================
// Pure-Rust MQTT v3.1.1 over TLS (:8883) Client for Bambu Lab LAN Mode
// ============================================================================

fn encode_mqtt_var_len(mut len: usize, out: &mut Vec<u8>) {
    loop {
        let mut byte = (len % 128) as u8;
        len /= 128;
        if len > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if len == 0 {
            break;
        }
    }
}

fn encode_mqtt_utf8(s: &str, out: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    let len = bytes.len().min(u16::MAX as usize) as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&bytes[..len as usize]);
}

fn build_mqtt_connect_packet(client_id: &str, username: &str, password: &str) -> Vec<u8> {
    let mut payload = Vec::with_capacity(128);
    // Protocol Name "MQTT", Level 4 (v3.1.1)
    encode_mqtt_utf8("MQTT", &mut payload);
    payload.push(4);
    // Connect Flags: Username (0x80) | Password (0x40) | Clean Session (0x02) = 0xC2
    payload.push(0xC2);
    // Keep Alive = 30 seconds
    payload.extend_from_slice(&30u16.to_be_bytes());
    encode_mqtt_utf8(client_id, &mut payload);
    encode_mqtt_utf8(username, &mut payload);
    encode_mqtt_utf8(password, &mut payload);

    let mut pkt = Vec::with_capacity(payload.len() + 5);
    pkt.push(0x10); // CONNECT
    encode_mqtt_var_len(payload.len(), &mut pkt);
    pkt.extend_from_slice(&payload);
    pkt
}

fn build_mqtt_subscribe_packet(packet_id: u16, topic: &str) -> Vec<u8> {
    let mut payload = Vec::with_capacity(topic.len() + 6);
    payload.extend_from_slice(&packet_id.to_be_bytes());
    encode_mqtt_utf8(topic, &mut payload);
    payload.push(0x00); // QoS 0

    let mut pkt = Vec::with_capacity(payload.len() + 5);
    pkt.push(0x82); // SUBSCRIBE
    encode_mqtt_var_len(payload.len(), &mut pkt);
    pkt.extend_from_slice(&payload);
    pkt
}

fn build_mqtt_publish_packet(topic: &str, message_bytes: &[u8]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(topic.len() + 2 + message_bytes.len());
    encode_mqtt_utf8(topic, &mut payload);
    payload.extend_from_slice(message_bytes);

    let mut pkt = Vec::with_capacity(payload.len() + 5);
    pkt.push(0x30); // PUBLISH QoS 0
    encode_mqtt_var_len(payload.len(), &mut pkt);
    pkt.extend_from_slice(&payload);
    pkt
}

fn read_mqtt_packet<R: std::io::Read>(reader: &mut R) -> Result<(u8, Vec<u8>), String> {
    let mut hdr = [0u8; 1];
    reader
        .read_exact(&mut hdr)
        .map_err(|e| format!("MQTT read header error: {e}"))?;
    let pkt_type = hdr[0];

    let mut mult: usize = 1;
    let mut rem_len: usize = 0;
    loop {
        let mut b = [0u8; 1];
        reader
            .read_exact(&mut b)
            .map_err(|e| format!("MQTT read length error: {e}"))?;
        rem_len += ((b[0] & 0x7F) as usize) * mult;
        if (b[0] & 0x80) == 0 {
            break;
        }
        mult *= 128;
        if mult > 128 * 128 * 128 {
            return Err("Malformed MQTT variable length".to_string());
        }
    }

    let mut body = vec![0u8; rem_len];
    if rem_len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("MQTT read body error: {e}"))?;
    }
    Ok((pkt_type, body))
}

/// Parses a Bambu Lab MQTT `device/<SERIAL>/report` JSON payload (`{"print": {...}}`)
/// into `MoonrakerPrintStatus`.
pub fn parse_bambu_mqtt_report_json(report_json: &Value) -> Option<MoonrakerPrintStatus> {
    let print_obj = report_json.get("print").unwrap_or(report_json);
    if !print_obj.is_object() {
        return None;
    }

    let gcode_state = print_obj
        .get("gcode_state")
        .and_then(Value::as_str)
        .unwrap_or("IDLE");
    let state = match gcode_state.trim().to_ascii_uppercase().as_str() {
        "RUNNING" | "PREPARE" => "printing".to_string(),
        "PAUSE" => "paused".to_string(),
        "FINISH" => "complete".to_string(),
        "FAILED" => "error".to_string(),
        _ => "standby".to_string(),
    };

    let mc_percent = print_obj
        .get("mc_percent")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let progress = if state == "complete" {
        1.0
    } else {
        (mc_percent / 100.0).clamp(0.0, 1.0) as f32
    };

    // `mc_remaining_time` in Bambu MQTT reports is in minutes!
    let rem_s = print_obj
        .get("mc_remaining_time")
        .and_then(Value::as_f64)
        .filter(|&m| m >= 0.0)
        .map(|mins| mins * 60.0);

    let filename = print_obj
        .get("subtask_name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| print_obj.get("gcode_file").and_then(Value::as_str))
        .unwrap_or("")
        .to_string();

    let mut msg_parts = Vec::new();
    if let Some(t_noz) = print_obj.get("nozzle_temper").and_then(Value::as_f64) {
        if let Some(tgt_noz) = print_obj
            .get("nozzle_target_temper")
            .and_then(Value::as_f64)
            .filter(|&t| t > 0.5)
        {
            msg_parts.push(format!("Nozzle {t_noz:.0}/{tgt_noz:.0}°C"));
        } else {
            msg_parts.push(format!("Nozzle {t_noz:.0}°C"));
        }
    }
    if let Some(t_bed) = print_obj.get("bed_temper").and_then(Value::as_f64) {
        if let Some(tgt_bed) = print_obj
            .get("bed_target_temper")
            .and_then(Value::as_f64)
            .filter(|&t| t > 0.5)
        {
            msg_parts.push(format!("Bed {t_bed:.0}/{tgt_bed:.0}°C"));
        } else {
            msg_parts.push(format!("Bed {t_bed:.0}°C"));
        }
    }
    if let (Some(cur_l), Some(tot_l)) = (
        print_obj.get("layer_num").and_then(Value::as_u64),
        print_obj
            .get("total_layer_num")
            .and_then(Value::as_u64)
            .filter(|&t| t > 0),
    ) {
        msg_parts.push(format!("Layer {cur_l}/{tot_l}"));
    }

    let est_total_s = match (rem_s, progress > 0.01 && progress < 0.999) {
        (Some(r), true) => Some(r / (1.0 - progress as f64)),
        _ => None,
    };
    let print_duration_s = match (est_total_s, rem_s) {
        (Some(tot), Some(r)) => (tot - r).max(0.0),
        _ => 0.0,
    };

    Some(MoonrakerPrintStatus {
        state,
        filename,
        progress,
        print_duration_s,
        total_duration_s: print_duration_s,
        slicer_estimated_time_s: est_total_s,
        eta_remaining_s: if gcode_state.eq_ignore_ascii_case("RUNNING")
            || gcode_state.eq_ignore_ascii_case("PAUSE")
        {
            rem_s
        } else {
            None
        },
        message: msg_parts.join(" | "),
    })
}

/// Connects to the Bambu Lab printer's local MQTT-TLS broker (`:8883`), authenticates with
/// `bblp` + Access Code, optionally publishes a command to `device/<SERIAL>/request`, and
/// optionally reads status telemetry from `device/<SERIAL>/report`.
fn run_bambu_mqtt_session(
    cfg: &MoonrakerNetworkConfig,
    publish_request_json: Option<&str>,
    wait_for_report: bool,
) -> Result<Option<MoonrakerPrintStatus>, String> {
    use std::io::Write as _;
    let (host, port) = parse_bambu_lan_host_and_port(&cfg.printer_uri, 8883)?;
    let access_code = cfg.effective_bambu_access_code();
    if access_code.is_empty() {
        return Err(
            "Bambu Lab Access Code is empty. Enter the 8-character LAN Access Code from the printer screen (Settings -> LAN Only Mode)."
                .to_string(),
        );
    }
    let username = cfg.effective_bambu_username();
    let serial = cfg.serial_number.trim();
    if (publish_request_json.is_some() || wait_for_report) && serial.is_empty() {
        return Err(
            "Bambu Lab Printer Serial Number is required for MQTT print start and live telemetry (e.g. 01P00A... or 00M00A... from the printer screen)."
                .to_string(),
        );
    }

    let tls_cfg = build_bambu_lan_tls_config();
    let mut stream = connect_bambu_tls_stream(&host, port, tls_cfg, Duration::from_secs(6))?;

    let client_id = format!("manifold2-{}", std::process::id());
    let connect_pkt = build_mqtt_connect_packet(&client_id, username, access_code);
    stream
        .write_all(&connect_pkt)
        .map_err(|e| format!("Bambu MQTT write CONNECT error: {e}"))?;
    stream
        .flush()
        .map_err(|e| format!("Bambu MQTT flush error: {e}"))?;

    let (pkt_type, body) = read_mqtt_packet(&mut stream)?;
    if (pkt_type & 0xF0) != 0x20 || body.len() < 2 {
        return Err(format!("Unexpected Bambu MQTT CONNACK packet (0x{pkt_type:02x})"));
    }
    let return_code = body[1];
    if return_code != 0 {
        return Err(format!(
            "Bambu MQTT authentication failed (CONNACK return code {return_code}). Verify the 8-character LAN Access Code."
        ));
    }

    if !serial.is_empty() && wait_for_report {
        let report_topic = format!("device/{serial}/report");
        let sub_pkt = build_mqtt_subscribe_packet(1, &report_topic);
        stream
            .write_all(&sub_pkt)
            .map_err(|e| format!("Bambu MQTT SUBSCRIBE error: {e}"))?;
        // Request a full `pushall` status snapshot
        let req_topic = format!("device/{serial}/request");
        let pushall_json = r#"{"pushing":{"sequence_id":"0","command":"pushall"}}"#;
        let pub_pkt = build_mqtt_publish_packet(&req_topic, pushall_json.as_bytes());
        let _ = stream.write_all(&pub_pkt);
        let _ = stream.flush();
    }

    if let Some(req_json) = publish_request_json {
        let req_topic = format!("device/{serial}/request");
        let pub_pkt = build_mqtt_publish_packet(&req_topic, req_json.as_bytes());
        stream
            .write_all(&pub_pkt)
            .map_err(|e| format!("Bambu MQTT PUBLISH request error: {e}"))?;
        stream
            .flush()
            .map_err(|e| format!("Bambu MQTT flush request error: {e}"))?;
    }

    let mut latest_status = None;
    if !serial.is_empty() && wait_for_report {
        let deadline = Instant::now() + Duration::from_millis(2500);
        while Instant::now() < deadline {
            match read_mqtt_packet(&mut stream) {
                Ok((ptype, pbody)) => {
                    if (ptype & 0xF0) == 0x30 && pbody.len() > 2 {
                        let topic_len = u16::from_be_bytes([pbody[0], pbody[1]]) as usize;
                        let qos = (ptype >> 1) & 0x03;
                        let pkt_id_len = if qos > 0 { 2 } else { 0 };
                        let payload_start = 2 + topic_len + pkt_id_len;
                        if payload_start < pbody.len() {
                            if let Ok(json_val) =
                                serde_json::from_slice::<Value>(&pbody[payload_start..])
                            {
                                if json_val.get("print").is_some() {
                                    if let Some(st) = parse_bambu_mqtt_report_json(&json_val) {
                                        latest_status = Some(st);
                                        if latest_status
                                            .as_ref()
                                            .is_some_and(|s| !s.message.is_empty() || s.progress > 0.0)
                                        {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
    }

    // Send MQTT DISCONNECT (0xE0, 0x00)
    let _ = stream.write_all(&[0xE0, 0x00]);
    let _ = stream.flush();
    Ok(latest_status)
}

/// Builds the Bambu Lab MQTT `project_file` JSON command to start printing an uploaded `.gcode.3mf` archive.
pub fn build_bambu_project_file_command_json(
    remote_3mf_filename: &str,
    subtask_name: &str,
    use_ams: bool,
    bed_leveling: bool,
) -> String {
    let clean_file = remote_3mf_filename.trim_start_matches('/');
    let payload = serde_json::json!({
        "print": {
            "sequence_id": "1",
            "command": "project_file",
            "param": "Metadata/plate_1.gcode",
            "project_id": "0",
            "profile_id": "0",
            "task_id": "0",
            "subtask_id": "0",
            "subtask_name": subtask_name,
            "file": "",
            "url": format!("ftp:///{clean_file}"),
            "md5": "",
            "timelapse": false,
            "bed_type": "auto",
            "bed_levelling": bed_leveling,
            "bed_leveling": bed_leveling,
            "flow_cali": false,
            "vibration_cali": false,
            "layer_inspect": false,
            "ams_mapping": [0, 1, 2, 3],
            "use_ams": use_ams
        }
    });
    payload.to_string()
}

/// Tests Bambu Lab LAN connectivity:
/// - Verifies MQTT-TLS (`:8883`) `CONNACK` authentication with `bblp` + Access Code
/// - If `serial_number` is provided, subscribes to `device/<SERIAL>/report` and reads live temperatures & state
pub fn test_bambu_lan_connection(
    cfg: &MoonrakerNetworkConfig,
) -> Result<MoonrakerConnectionReport, String> {
    let (host, _) = parse_bambu_lan_host_and_port(&cfg.printer_uri, 8883)?;
    let has_serial = !cfg.serial_number.trim().is_empty();
    let status_opt = run_bambu_mqtt_session(cfg, None, has_serial)?;

    let (state, msg) = if let Some(st) = status_opt {
        (st.state, st.message)
    } else if has_serial {
        (
            "ready".to_string(),
            format!("SN {}", cfg.serial_number.trim()),
        )
    } else {
        (
            "authenticated".to_string(),
            "Enter Serial Number for live MQTT status & print start".to_string(),
        )
    };

    Ok(MoonrakerConnectionReport {
        protocol: PrinterNetworkProtocol::BambuLan,
        normalized_uri: format!("mqtts://{host}:8883"),
        moonraker_version: "FTPS :990 + MQTT :8883".to_string(),
        klippy_state: state,
        hostname: host,
        state_message: msg,
    })
}

/// Queries live print status from a Bambu Lab printer via MQTT-TLS (`:8883`).
pub fn query_bambu_lan_print_status(
    cfg: &MoonrakerNetworkConfig,
    cached_metadata: Option<(String, Option<f64>)>,
) -> (
    Result<MoonrakerPrintStatus, String>,
    Option<(String, Option<f64>)>,
) {
    if cfg.serial_number.trim().is_empty() {
        return (
            Err("Enter Printer Serial Number for Bambu live status".to_string()),
            cached_metadata,
        );
    }
    match run_bambu_mqtt_session(cfg, None, true) {
        Ok(Some(st)) => (Ok(st), cached_metadata),
        Ok(None) => (
            Ok(MoonrakerPrintStatus {
                state: "standby".to_string(),
                filename: String::new(),
                progress: 0.0,
                print_duration_s: 0.0,
                total_duration_s: 0.0,
                slicer_estimated_time_s: None,
                eta_remaining_s: None,
                message: format!("SN {}", cfg.serial_number.trim()),
            }),
            cached_metadata,
        ),
        Err(e) => (Err(e), cached_metadata),
    }
}

/// Packages `gcode_text` into a Bambu Lab `.gcode.3mf` container (`Metadata/plate_1.gcode`,
/// `Metadata/plate_1.gcode.md5`, `Metadata/slice_info.config`, `Metadata/plate_1.png`),
/// uploads it over implicit FTPS (`:990`), and when `start_print == true`, publishes the
/// `project_file` start command over MQTT-TLS (`:8883`).
pub fn upload_gcode_to_bambu_lan(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
) -> Result<MoonrakerUploadReport, String> {
    upload_gcode_to_bambu_lan_with_progress(cfg, filename, gcode_text, start_print, None)
}

/// Streaming variant of [`upload_gcode_to_bambu_lan`] that reports archive packaging,
/// chunked FTPS progress, and MQTT start command progress to `progress`.
pub fn upload_gcode_to_bambu_lan_with_progress(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
    progress: Option<&UploadProgressTracker>,
) -> Result<MoonrakerUploadReport, String> {
    let basename = filename
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("manifold2_print");
    let stem = basename
        .strip_suffix(".gcode.3mf")
        .or_else(|| basename.strip_suffix(".3mf"))
        .or_else(|| basename.strip_suffix(".gcode"))
        .or_else(|| basename.strip_suffix(".nc"))
        .unwrap_or(basename);
    let remote_3mf_name = format!("{stem}.gcode.3mf");

    if let Some(tr) = progress {
        tr.set_stage(0.04, "Packaging Bambu .gcode.3mf archive...");
    }
    let archive_bytes = crate::io::threemf::build_bambu_gcode_3mf_bytes(&gcode_text, None)?;
    let bytes_uploaded = archive_bytes.len();

    if let Some(tr) = progress {
        tr.set_stage(0.10, "Connecting to Bambu FTPS (:990)...");
    }
    let ftps_end = if start_print { 0.88 } else { 0.95 };
    let uploaded_path = upload_bytes_to_bambu_ftps_with_progress(
        cfg,
        &remote_3mf_name,
        &archive_bytes,
        ftps_end,
        progress,
    )?;

    if start_print {
        if let Some(tr) = progress {
            tr.set_stage(0.93, "Starting print via Bambu MQTT (:8883)...");
        }
        let cmd_json = build_bambu_project_file_command_json(
            &uploaded_path,
            stem,
            cfg.bambu_use_ams,
            cfg.bambu_bed_leveling,
        );
        run_bambu_mqtt_session(cfg, Some(&cmd_json), false)?;
    }

    if let Some(tr) = progress {
        tr.set_stage(1.0, "Upload complete.");
    }

    Ok(MoonrakerUploadReport {
        protocol: PrinterNetworkProtocol::BambuLan,
        remote_path: uploaded_path,
        bytes_uploaded,
        print_started: start_print,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn test_normalize_moonraker_uri_and_resolve_web_ui_url() {
        assert_eq!(
            normalize_moonraker_base_uri("http://voron.local:7125/").unwrap(),
            "http://voron.local:7125"
        );
        assert_eq!(
            normalize_moonraker_base_uri("192.168.1.42:7125").unwrap(),
            "http://192.168.1.42:7125"
        );
        assert_eq!(
            normalize_moonraker_base_uri("https://printer.example.com/").unwrap(),
            "https://printer.example.com"
        );
        assert!(normalize_moonraker_base_uri("   ").is_err());

        let mut cfg = MoonrakerNetworkConfig {
            printer_uri: "http://voron.local:7125/".to_string(),
            ..Default::default()
        };
        // Default Web UI strips :7125 so Mainsail/Fluidd opens on standard port 80
        assert_eq!(
            resolve_printer_web_ui_url(&cfg).unwrap(),
            "http://voron.local"
        );

        // Explicit Web UI override takes precedence
        cfg.web_ui_uri = "https://mainsail.example.com:8443/".to_string();
        assert_eq!(
            resolve_printer_web_ui_url(&cfg).unwrap(),
            "https://mainsail.example.com:8443"
        );
    }

    #[test]
    fn test_parse_status_and_eta_calculation() {
        let sample_json: Value = serde_json::from_str(
            r#"{
                "result": {
                    "status": {
                        "print_stats": {
                            "state": "printing",
                            "filename": "manifold_part.gcode",
                            "print_duration": 600.0,
                            "total_duration": 620.0,
                            "message": ""
                        },
                        "virtual_sdcard": {
                            "progress": 0.25
                        },
                        "display_status": {
                            "progress": 0.24,
                            "message": "Layer 12/48"
                        }
                    }
                }
            }"#,
        )
        .unwrap();

        // With slicer metadata estimated_time = 2000s at 25% progress (600s elapsed):
        // metadata_eta = 2000 * 0.75 = 1500s
        // linear_eta   = 600 * 0.75 / 0.25 = 1800s
        // blended      = 0.75 * 1500 + 0.25 * 1800 = 1575s
        let status = parse_printer_objects_status_json(&sample_json, Some(2000.0)).unwrap();
        assert_eq!(status.state, "printing");
        assert_eq!(status.filename, "manifold_part.gcode");
        assert!((status.progress - 0.25).abs() < 1.0e-5);
        assert_eq!(status.message, "Layer 12/48");
        let eta = status.eta_remaining_s.expect("Expected ETA while printing");
        assert!((eta - 1575.0).abs() < 1.0e-3, "Expected 1575s ETA, got {eta}");
        assert_eq!(status.formatted_eta().as_deref(), Some("26:15"));
        assert_eq!(status.formatted_elapsed(), "10:00");
    }

    #[test]
    fn test_tls_pem_validation_errors_are_clear() {
        let cfg_missing_ca = MoonrakerNetworkConfig {
            printer_uri: "https://printer.local:7125".to_string(),
            ca_cert_pem_path: "/nonexistent/path/ca_root.pem".to_string(),
            ..Default::default()
        };
        let err = build_moonraker_client(&cfg_missing_ca, Duration::from_secs(5)).unwrap_err();
        assert!(
            err.contains("Failed to read CA certificate PEM file"),
            "Unexpected error: {err}"
        );

        let cfg_orphan_key = MoonrakerNetworkConfig {
            printer_uri: "https://printer.local:7125".to_string(),
            client_key_pem_path: "/some/client.key".to_string(),
            ..Default::default()
        };
        let err_key = build_moonraker_client(&cfg_orphan_key, Duration::from_secs(5)).unwrap_err();
        assert!(err_key.contains("Client Certificate PEM path is empty"));
    }

    #[test]
    fn test_mock_moonraker_server_test_connection_status_and_upload() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Bind ephemeral port");
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            // Handle 5 sequential HTTP requests:
            // 1. GET /server/info
            // 2. GET /printer/info
            // 3. GET /printer/objects/query...
            // 4. GET /server/files/metadata...
            // 5. POST /server/files/upload
            for req_idx in 0..5 {
                let (mut stream, _) = listener.accept().expect("Accept connection");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut buf = vec![0u8; 65536];
                let mut total_read = 0usize;
                loop {
                    match stream.read(&mut buf[total_read..]) {
                        Ok(0) => break,
                        Ok(n) => {
                            total_read += n;
                            let req_str = String::from_utf8_lossy(&buf[..total_read]);
                            if let Some(hdr_end) = req_str.find("\r\n\r\n") {
                                let headers_part = &req_str[..hdr_end];
                                let content_len = headers_part
                                    .lines()
                                    .find_map(|l| {
                                        let lower = l.to_ascii_lowercase();
                                        lower
                                            .strip_prefix("content-length:")
                                            .and_then(|v| v.trim().parse::<usize>().ok())
                                    })
                                    .unwrap_or(0);
                                if total_read >= hdr_end + 4 + content_len {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
                let req_text = String::from_utf8_lossy(&buf[..total_read]);
                assert!(
                    req_text.to_ascii_lowercase().contains("x-api-key: secret-test-key"),
                    "Request #{req_idx} missing X-Api-Key header:\n{req_text}"
                );

                let body_json = match req_idx {
                    0 => {
                        assert!(req_text.starts_with("GET /server/info "));
                        r#"{"result":{"klippy_connected":true,"klippy_state":"ready","moonraker_version":"v0.8.0-320"}}"#
                    }
                    1 => {
                        assert!(req_text.starts_with("GET /printer/info "));
                        r#"{"result":{"state":"ready","state_message":"Printer is ready","hostname":"voron-5axis"}}"#
                    }
                    2 => {
                        assert!(req_text.starts_with("GET /printer/objects/query?print_stats&virtual_sdcard&display_status "));
                        r#"{"result":{"status":{"print_stats":{"state":"printing","filename":"test_part.gcode","print_duration":300.0,"total_duration":310.0,"message":""},"virtual_sdcard":{"progress":0.5},"display_status":{"progress":0.5,"message":"Printing"}}}}"#
                    }
                    3 => {
                        assert!(req_text.starts_with("GET /server/files/metadata?filename=test_part.gcode "));
                        r#"{"result":{"filename":"test_part.gcode","estimated_time":600.0}}"#
                    }
                    4 => {
                        assert!(req_text.starts_with("POST /server/files/upload "));
                        assert!(req_text.contains("name=\"print\""));
                        assert!(req_text.contains("true"));
                        assert!(req_text.contains("G1 X10 Y20 Z0.2 E0.5"));
                        r#"{"item":{"path":"test_upload.gcode","root":"gcodes"},"print_started":true}"#
                    }
                    _ => unreachable!(),
                };

                let http_resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body_json.len(),
                    body_json
                );
                stream.write_all(http_resp.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let cfg = MoonrakerNetworkConfig {
            printer_uri: format!("http://{addr}"),
            api_key: "secret-test-key".to_string(),
            ..Default::default()
        };

        // 1. Test connection
        let conn = test_moonraker_connection(&cfg).expect("Connection test should succeed");
        assert_eq!(conn.moonraker_version, "v0.8.0-320");
        assert_eq!(conn.klippy_state, "ready");
        assert_eq!(conn.hostname, "voron-5axis");

        // 2. Query status + metadata ETA
        let (status_res, cache) = query_moonraker_print_status(&cfg, None);
        let status = status_res.expect("Status query should succeed");
        assert_eq!(status.state, "printing");
        assert_eq!(status.filename, "test_part.gcode");
        assert!((status.progress - 0.5).abs() < 1.0e-5);
        assert!((status.eta_remaining_s.unwrap() - 300.0).abs() < 1.0e-3);
        assert_eq!(cache, Some(("test_part.gcode".to_string(), Some(600.0))));

        // 3. Upload & start print
        let upload = upload_gcode_to_moonraker(
            &cfg,
            "test_upload.gcode",
            "; Manifold2 test\nG1 X10 Y20 Z0.2 E0.5\n".to_string(),
            true,
        )
        .expect("Upload should succeed");
        assert_eq!(upload.protocol, PrinterNetworkProtocol::Moonraker);
        assert_eq!(upload.remote_path, "test_upload.gcode");
        assert!(upload.print_started);

        server_thread.join().unwrap();
    }

    #[test]
    fn test_md5_sha256_and_rfc2617_digest_auth_vectors() {
        // RFC 1321 MD5 standard vectors
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"),
            "d174ab98d277d9f5a5611c2c9f419d9f"
        );

        // FIPS 180-4 SHA-256 standard vectors
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        // RFC 2617 Section 3.5 canonical HTTP Digest vector:
        // username="Mufasa", password="Circle Of Life", method="GET", uri="/dir/index.html"
        let www_auth = r#"Digest realm="testrealm@host.com", qop="auth,auth-int", nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", opaque="5ccc069c403ebaf9f0171e9517f40e41""#;
        let challenge = DigestChallenge::parse(www_auth).expect("Should parse Digest header");
        assert_eq!(challenge.realm, "testrealm@host.com");
        assert_eq!(challenge.nonce, "dcd98b7102dd2f0e8b11d0f600bfb0c093");
        assert!(challenge.qop_auth);

        let auth_hdr = challenge.build_authorization_header(
            "Mufasa",
            "Circle Of Life",
            "GET",
            "/dir/index.html",
            1,
            "0a4f113b",
        );
        assert!(
            auth_hdr.contains("response=\"6629fae49393a05397450978507c4ef1\""),
            "Unexpected RFC 2617 Digest response: {auth_hdr}"
        );
        assert!(auth_hdr.contains("qop=auth, nc=00000001, cnonce=\"0a4f113b\""));

        // Prusa Buddy `nhttp` RFC 2069 no-qop Digest challenge
        let buddy_www = r#"Digest realm="Printer API", nonce="65f8a21b0042c901""#;
        let buddy_ch = DigestChallenge::parse(buddy_www).expect("Should parse Buddy Digest header");
        assert!(!buddy_ch.qop_auth);
        let buddy_hdr = buddy_ch.build_authorization_header(
            "maker",
            "prusa12345",
            "GET",
            "/api/v1/info",
            1,
            "unused",
        );
        let expected_ha1 = md5_hex(b"maker:Printer API:prusa12345");
        let expected_ha2 = md5_hex(b"GET:/api/v1/info");
        let expected_resp =
            md5_hex(format!("{expected_ha1}:65f8a21b0042c901:{expected_ha2}").as_bytes());
        assert!(buddy_hdr.contains(&format!("response=\"{expected_resp}\"")));
        assert!(!buddy_hdr.contains("qop="));
    }

    #[test]
    fn test_mock_prusalink_v1_digest_auth_connection_status_and_put_upload() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("Bind ephemeral port");
        let addr = listener.local_addr().unwrap();

        let server_thread = std::thread::spawn(move || {
            let realm = "Printer API";
            let nonce = "a1b2c3d4e5f60718";
            let expected_user = "maker";
            let expected_pass = "coreone-secret-pw";

            let mut handled_authenticated = 0usize;
            while handled_authenticated < 6 {
                let (mut stream, _) = listener.accept().expect("Accept PrusaLink connection");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut buf = vec![0u8; 65536];
                let mut total_read = 0usize;
                loop {
                    match stream.read(&mut buf[total_read..]) {
                        Ok(0) => break,
                        Ok(n) => {
                            total_read += n;
                            let req_str = String::from_utf8_lossy(&buf[..total_read]);
                            if let Some(hdr_end) = req_str.find("\r\n\r\n") {
                                let headers_part = &req_str[..hdr_end];
                                let content_len = headers_part
                                    .lines()
                                    .find_map(|l| {
                                        let lower = l.to_ascii_lowercase();
                                        lower
                                            .strip_prefix("content-length:")
                                            .and_then(|v| v.trim().parse::<usize>().ok())
                                    })
                                    .unwrap_or(0);
                                if total_read >= hdr_end + 4 + content_len {
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }

                let req_text = String::from_utf8_lossy(&buf[..total_read]);
                let first_line = req_text.lines().next().unwrap_or("");
                let mut parts = first_line.split_whitespace();
                let method = parts.next().unwrap_or("");
                let path = parts.next().unwrap_or("");

                // Verify Digest Authorization header; if missing, issue 401 WWW-Authenticate: Digest
                let auth_line = req_text
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("authorization:"));

                let is_authorized = if let Some(line) = auth_line {
                    let val = line["authorization:".len()..].trim();
                    if let Some(rest) = val.strip_prefix("Digest ") {
                        let params = parse_auth_key_values(rest);
                        let u = params.get("username").map(String::as_str).unwrap_or("");
                        let nc_val = params.get("nc").map(String::as_str).unwrap_or("");
                        let cnonce_val = params.get("cnonce").map(String::as_str).unwrap_or("");
                        let resp_val = params.get("response").map(String::as_str).unwrap_or("");
                        let ha1 = md5_hex(format!("{expected_user}:{realm}:{expected_pass}").as_bytes());
                        let ha2 = md5_hex(format!("{method}:{path}").as_bytes());
                        let expected_digest = md5_hex(
                            format!("{ha1}:{nonce}:{nc_val}:{cnonce_val}:auth:{ha2}").as_bytes(),
                        );
                        u == expected_user && resp_val == expected_digest
                    } else {
                        false
                    }
                } else {
                    false
                };

                if !is_authorized {
                    let unauth_resp = format!(
                        "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Digest realm=\"{realm}\", nonce=\"{nonce}\", qop=\"auth\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    stream.write_all(unauth_resp.as_bytes()).unwrap();
                    stream.flush().unwrap();
                    continue;
                }

                let (status_line, body_json) = match (method, path) {
                    ("GET", "/api/v1/info") => (
                        "200 OK",
                        r#"{"firmware":"6.3.2+8912","hostname":"prusa-core-one-indx","serial":"CZPX123456","nozzle_diameter":0.4,"mmu":false}"#,
                    ),
                    ("GET", "/api/v1/status") => (
                        "200 OK",
                        r#"{"printer":{"state":"PRINTING","temp_nozzle":225.0,"target_nozzle":225.0,"temp_bed":60.0,"target_bed":60.0},"job":{"id":42,"progress":40.0,"time_remaining":900,"time_printing":600}}"#,
                    ),
                    ("GET", "/api/v1/job") => (
                        "200 OK",
                        r#"{"id":42,"state":"PRINTING","progress":40.0,"time_remaining":900,"time_printing":600,"file":{"name":"BENCHY~1.GCO","display_name":"core_one_benchy.gcode"}}"#,
                    ),
                    ("GET", "/api/v1/storage") => (
                        "200 OK",
                        r#"{"storage_list":[{"path":"/usb/","name":"usb","type":"USB","read_only":false,"available":true}]}"#,
                    ),
                    ("PUT", "/api/v1/files/usb/test_upload.gcode") => {
                        let lower_req = req_text.to_ascii_lowercase();
                        assert!(
                            lower_req.contains("overwrite: ?1"),
                            "Missing Overwrite: ?1 header:\n{req_text}"
                        );
                        assert!(
                            lower_req.contains("print-after-upload: ?1"),
                            "Missing Print-After-Upload: ?1 header:\n{req_text}"
                        );
                        assert!(
                            req_text.contains("G1 X50 Y50 Z0.2 E1.25"),
                            "Missing G-code payload in PUT body:\n{req_text}"
                        );
                        ("201 Created", r#"{"name":"TEST_U~1.GCO","display_name":"test_upload.gcode"}"#)
                    }
                    other => panic!("Unexpected PrusaLink request: {other:?}"),
                };

                handled_authenticated += 1;
                let http_resp = format!(
                    "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body_json.len(),
                    body_json
                );
                stream.write_all(http_resp.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let cfg = MoonrakerNetworkConfig {
            protocol: PrinterNetworkProtocol::PrusaLink,
            printer_uri: format!("http://{addr}"),
            username: String::new(), // blank -> should automatically default to "maker"
            password: "coreone-secret-pw".to_string(),
            ..Default::default()
        };

        // 1. Test PrusaLink connection (performs initial 401 -> Digest auth handshake + /api/v1/info + /api/v1/status)
        let conn = test_moonraker_connection(&cfg).expect("PrusaLink connection test should succeed");
        assert_eq!(conn.protocol, PrinterNetworkProtocol::PrusaLink);
        assert_eq!(conn.moonraker_version, "6.3.2+8912");
        assert_eq!(conn.klippy_state, "printing");
        assert_eq!(conn.hostname, "prusa-core-one-indx");
        assert!(conn.summary_line().contains("PrusaLink FW 6.3.2+8912"));

        // 2. Query PrusaLink live status + job filename + ETA (/api/v1/status + /api/v1/job)
        let (status_res, _) = query_moonraker_print_status(&cfg, None);
        let status = status_res.expect("PrusaLink status query should succeed");
        assert_eq!(status.state, "printing");
        assert_eq!(status.filename, "core_one_benchy.gcode");
        assert!((status.progress - 0.40).abs() < 1.0e-5);
        assert_eq!(status.eta_remaining_s, Some(900.0));
        assert_eq!(status.formatted_eta().as_deref(), Some("15:00"));
        assert!(status.message.contains("Nozzle 225/225°C"));

        // 3. Upload G-code & start print (/api/v1/storage + PUT /api/v1/files/usb/test_upload.gcode)
        let upload = upload_gcode_to_moonraker(
            &cfg,
            "test_upload.gcode",
            "; Prusa CORE One INDX test\nG1 X50 Y50 Z0.2 E1.25\n".to_string(),
            true,
        )
        .expect("PrusaLink PUT upload should succeed");
        assert_eq!(upload.protocol, PrinterNetworkProtocol::PrusaLink);
        assert_eq!(upload.remote_path, "usb/test_upload.gcode");
        assert!(upload.print_started);
        assert!(upload.summary_line().contains("to PrusaLink and started print"));

        server_thread.join().unwrap();
    }

    #[test]
    fn test_bambu_lan_parsing_ftps_pasv_mqtt_report_and_gcode_3mf_container() {
        // 1. Host & port parsing
        assert_eq!(
            parse_bambu_lan_host_and_port("192.168.1.88", 990).unwrap(),
            ("192.168.1.88".to_string(), 990)
        );
        assert_eq!(
            parse_bambu_lan_host_and_port("ftps://192.168.1.88:9990/", 990).unwrap(),
            ("192.168.1.88".to_string(), 9990)
        );

        // 2. FTPS PASV port parsing: (192,168,1,88,195,80) -> 195*256 + 80 = 50000
        let pasv_port =
            parse_ftp_pasv_port("227 Entering Passive Mode (192,168,1,88,195,80).\r\n").unwrap();
        assert_eq!(pasv_port, 50000);

        // 3. Bambu MQTT report JSON parsing
        let report_json: Value = serde_json::from_str(
            r#"{
                "print": {
                    "gcode_state": "RUNNING",
                    "mc_percent": 60,
                    "mc_remaining_time": 12,
                    "subtask_name": "bambu_benchy",
                    "nozzle_temper": 220.4,
                    "nozzle_target_temper": 220.0,
                    "bed_temper": 55.1,
                    "bed_target_temper": 55.0,
                    "layer_num": 150,
                    "total_layer_num": 250
                }
            }"#,
        )
        .unwrap();
        let status = parse_bambu_mqtt_report_json(&report_json).expect("Should parse Bambu MQTT report");
        assert_eq!(status.state, "printing");
        assert_eq!(status.filename, "bambu_benchy");
        assert!((status.progress - 0.60).abs() < 1.0e-5);
        assert_eq!(status.eta_remaining_s, Some(720.0)); // 12 minutes = 720s
        assert!(status.message.contains("Nozzle 220/220°C"));
        assert!(status.message.contains("Bed 55/55°C"));
        assert!(status.message.contains("Layer 150/250"));

        // 4. Bambu MQTT `project_file` command JSON generation
        let cmd_str = build_bambu_project_file_command_json("part.gcode.3mf", "part", true, true);
        let cmd_json: Value = serde_json::from_str(&cmd_str).unwrap();
        let p = cmd_json.get("print").unwrap();
        assert_eq!(p.get("command").and_then(Value::as_str), Some("project_file"));
        assert_eq!(p.get("param").and_then(Value::as_str), Some("Metadata/plate_1.gcode"));
        assert_eq!(p.get("url").and_then(Value::as_str), Some("ftp:///part.gcode.3mf"));
        assert_eq!(p.get("use_ams").and_then(Value::as_bool), Some(true));

        // 5. `.gcode.3mf` packaging and MD5 verification
        let sample_gcode = "; generated by PrusaSlicer 2.8.0-Manifold2 on 2026-11-20\nG28\nG1 X10 Y10 Z0.2 E1.0 F3000\n; filament used [mm] = 1500.0\n; total filament used [g] = 4.50\n; estimated printing time (normal mode) = 12m 30s\n";
        let archive_bytes =
            crate::io::threemf::build_bambu_gcode_3mf_bytes(sample_gcode, None).unwrap();
        let cursor = std::io::Cursor::new(archive_bytes);
        let mut zip = zip::ZipArchive::new(cursor).expect("Valid .gcode.3mf zip");
        let mut plate_gcode = String::new();
        zip.by_name("Metadata/plate_1.gcode")
            .expect("plate_1.gcode present")
            .read_to_string(&mut plate_gcode)
            .unwrap();
        assert_eq!(plate_gcode, sample_gcode);

        let mut plate_md5 = String::new();
        zip.by_name("Metadata/plate_1.gcode.md5")
            .expect("plate_1.gcode.md5 present")
            .read_to_string(&mut plate_md5)
            .unwrap();
        assert_eq!(plate_md5, md5_hex(sample_gcode.as_bytes()));

        let mut slice_info = String::new();
        zip.by_name("Metadata/slice_info.config")
            .expect("slice_info.config present")
            .read_to_string(&mut slice_info)
            .unwrap();
        assert!(slice_info.contains("key=\"prediction\" value=\"750\""));
        assert!(slice_info.contains("key=\"weight\" value=\"4.50\""));
    }

    fn read_full_http_request(stream: &mut std::net::TcpStream) -> (String, Vec<u8>) {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut raw = Vec::with_capacity(8192);
        let mut chunk = [0u8; 8192];
        let mut hdr_end_opt = None;
        let mut content_len = 0usize;
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    raw.extend_from_slice(&chunk[..n]);
                    if hdr_end_opt.is_none() {
                        if let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                            hdr_end_opt = Some(pos);
                            let headers_part = String::from_utf8_lossy(&raw[..pos]);
                            content_len = headers_part
                                .lines()
                                .find_map(|l| {
                                    let lower = l.to_ascii_lowercase();
                                    lower
                                        .strip_prefix("content-length:")
                                        .and_then(|v| v.trim().parse::<usize>().ok())
                                })
                                .unwrap_or(0);
                        }
                    }
                    if let Some(hdr_end) = hdr_end_opt {
                        if raw.len() >= hdr_end + 4 + content_len {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
        let hdr_end = hdr_end_opt.expect("Valid HTTP request headers");
        let headers = String::from_utf8_lossy(&raw[..hdr_end]).into_owned();
        let body = raw[(hdr_end + 4)..].to_vec();
        (headers, body)
    }

    #[test]
    fn test_upload_progress_tracker_and_streaming_http_uploads() {
        assert_eq!(format_byte_size(512), "512 B");
        assert_eq!(format_byte_size(2048), "2.0 KB");
        assert_eq!(format_byte_size(2 * 1024 * 1024), "2.00 MB");

        // 1. Verify chunked UploadProgressReader updates tracker monotonically to completion
        let payload_48k: Arc<[u8]> = vec![b'G'; 48 * 1024].into();
        let tracker = UploadProgressTracker::default();
        tracker.reset(payload_48k.len() as u64, "Starting...");
        let mut reader = UploadProgressReader::new(
            Arc::clone(&payload_48k),
            Some(&tracker),
            0.10,
            0.90,
            "Uploading 'test.gcode' to Mock",
            "Waiting for confirmation...",
        );
        let mut scratch = vec![0u8; 8192];
        let mut prev_sent = 0u64;
        let mut prev_frac = 0.0f32;
        let mut seen_kb_text = false;
        loop {
            let n = reader.read(&mut scratch).unwrap();
            let snap = tracker.snapshot();
            assert!(snap.bytes_sent >= prev_sent);
            assert!(snap.fraction >= prev_frac);
            if n > 0 {
                assert!(snap.fraction >= 0.10 && snap.fraction <= 0.90 + 1.0e-5);
                if snap.bar_text().contains("48.0 KB") {
                    seen_kb_text = true;
                }
            } else {
                assert!((snap.fraction - 0.90).abs() < 1.0e-5);
                assert_eq!(snap.bytes_sent, (48 * 1024) as u64);
                break;
            }
            prev_sent = snap.bytes_sent;
            prev_frac = snap.fraction;
        }
        assert!(seen_kb_text);

        // 2. Streaming Moonraker multipart POST with a >64 KB payload
        let gcode_64k: String = (0..3072)
            .map(|i| format!("G1 X{:.2} Y{:.2} E0.123\n", (i % 100) as f32, i as f32 * 0.01))
            .collect();
        assert!(gcode_64k.len() >= 64 * 1024);
        let expected_len = gcode_64k.len();
        let expected_body_check = gcode_64k.clone();

        let moon_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let moon_addr = moon_listener.local_addr().unwrap();
        let moon_server = std::thread::spawn(move || {
            let (mut stream, _) = moon_listener.accept().unwrap();
            let (headers, body) = read_full_http_request(&mut stream);
            assert!(headers.starts_with("POST /server/files/upload "));
            assert!(
                headers.to_ascii_lowercase().contains("content-length:"),
                "Streaming multipart upload must include Content-Length:\n{headers}"
            );
            let body_str = String::from_utf8_lossy(&body);
            assert!(body_str.contains(&expected_body_check));
            let resp_json = r#"{"item":{"path":"progress_moon.gcode","root":"gcodes"},"print_started":false}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                resp_json.len(),
                resp_json
            );
            stream.write_all(resp.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        let moon_cfg = MoonrakerNetworkConfig {
            protocol: PrinterNetworkProtocol::Moonraker,
            printer_uri: format!("http://{moon_addr}"),
            ..Default::default()
        };
        let moon_tracker = UploadProgressTracker::default();
        moon_tracker.reset(expected_len as u64, "Starting...");
        let moon_report = upload_gcode_to_moonraker_with_progress(
            &moon_cfg,
            "progress_moon.gcode",
            gcode_64k.clone(),
            false,
            Some(&moon_tracker),
        )
        .expect("Streaming Moonraker upload should succeed");
        assert_eq!(moon_report.bytes_uploaded, expected_len);
        let moon_snap = moon_tracker.snapshot();
        assert_eq!(moon_snap.fraction, 1.0);
        assert_eq!(moon_snap.bytes_sent, expected_len as u64);
        moon_server.join().unwrap();

        // 3. Streaming PrusaLink PUT forcing an uncached HTTP 401 Digest retry on the PUT itself
        let prusa_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let prusa_addr = prusa_listener.local_addr().unwrap();
        let expected_prusa_gcode = gcode_64k.clone();
        let prusa_server = std::thread::spawn(move || {
            let realm = "Printer API";
            let nonce = "retry9876543210";
            let expected_user = "maker";
            let expected_pass = "stream-digest-pw";

            // Request 1: GET /api/v1/storage (allowed without 401 so PUT hits an uncached 401)
            {
                let (mut stream, _) = prusa_listener.accept().unwrap();
                let (headers, _) = read_full_http_request(&mut stream);
                assert!(headers.starts_with("GET /api/v1/storage "));
                let st_json = r#"{"storage_list":[{"path":"/usb/","name":"usb","type":"USB","read_only":false,"available":true}]}"#;
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    st_json.len(),
                    st_json
                );
                stream.write_all(resp.as_bytes()).unwrap();
                stream.flush().unwrap();
            }

            // Request 2: Initial unauthenticated PUT -> challenge with 401 Digest
            {
                let (mut stream, _) = prusa_listener.accept().unwrap();
                let (headers, body) = read_full_http_request(&mut stream);
                assert!(headers.starts_with("PUT /api/v1/files/usb/progress_prusa.gcode "));
                assert!(headers.to_ascii_lowercase().contains("content-length:"));
                assert_eq!(body.len(), expected_prusa_gcode.len());
                let resp = format!(
                    "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Digest realm=\"{realm}\", nonce=\"{nonce}\", qop=\"auth\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
                stream.write_all(resp.as_bytes()).unwrap();
                stream.flush().unwrap();
            }

            // Request 3: Retried authenticated PUT -> verify rewound full 64 KB body & Digest auth
            {
                let (mut stream, _) = prusa_listener.accept().unwrap();
                let (headers, body) = read_full_http_request(&mut stream);
                assert!(headers.starts_with("PUT /api/v1/files/usb/progress_prusa.gcode "));
                let auth_line = headers
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("authorization:"))
                    .expect("Retried PUT must include Authorization: Digest header");
                let rest = auth_line["authorization:".len()..]
                    .trim()
                    .strip_prefix("Digest ")
                    .unwrap();
                let params = parse_auth_key_values(rest);
                let nc_val = params.get("nc").map(String::as_str).unwrap_or("");
                let cnonce_val = params.get("cnonce").map(String::as_str).unwrap_or("");
                let resp_val = params.get("response").map(String::as_str).unwrap_or("");
                let ha1 = md5_hex(format!("{expected_user}:{realm}:{expected_pass}").as_bytes());
                let ha2 = md5_hex(b"PUT:/api/v1/files/usb/progress_prusa.gcode");
                let expected_digest = md5_hex(
                    format!("{ha1}:{nonce}:{nc_val}:{cnonce_val}:auth:{ha2}").as_bytes(),
                );
                assert_eq!(resp_val, expected_digest);
                assert_eq!(body, expected_prusa_gcode.as_bytes());

                let ok_json = r#"{"name":"PROGRESS.GCO","display_name":"progress_prusa.gcode"}"#;
                let resp = format!(
                    "HTTP/1.1 201 Created\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    ok_json.len(),
                    ok_json
                );
                stream.write_all(resp.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        let prusa_cfg = MoonrakerNetworkConfig {
            protocol: PrinterNetworkProtocol::PrusaLink,
            printer_uri: format!("http://{prusa_addr}"),
            password: "stream-digest-pw".to_string(),
            ..Default::default()
        };
        let prusa_tracker = UploadProgressTracker::default();
        prusa_tracker.reset(expected_len as u64, "Starting...");
        let prusa_report = upload_gcode_to_moonraker_with_progress(
            &prusa_cfg,
            "progress_prusa.gcode",
            gcode_64k,
            false,
            Some(&prusa_tracker),
        )
        .expect("Streaming PrusaLink PUT with 401 Digest retry should rewind and succeed");
        assert_eq!(prusa_report.bytes_uploaded, expected_len);
        assert_eq!(prusa_report.remote_path, "usb/progress_prusa.gcode");
        let prusa_snap = prusa_tracker.snapshot();
        assert_eq!(prusa_snap.fraction, 1.0);
        assert_eq!(prusa_snap.bytes_sent, expected_len as u64);
        prusa_server.join().unwrap();
    }
}
