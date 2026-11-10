use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::tasks::Task;
use reqwest::blocking::{multipart, Client};
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{Certificate, Identity};
use serde_json::Value;

use crate::config::MoonrakerNetworkConfig;

/// Result of testing connectivity to a Moonraker / Klipper printer without printing anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoonrakerConnectionReport {
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
        format!(
            "Connected{host_part}: Moonraker {}, Klippy '{}'{}",
            self.moonraker_version, self.klippy_state, msg_part
        )
    }
}

/// Live printer status queried from Moonraker (`/printer/objects/query?print_stats&virtual_sdcard&display_status`).
#[derive(Clone, Debug, PartialEq)]
pub struct MoonrakerPrintStatus {
    /// Printer state: `"standby"`, `"printing"`, `"paused"`, `"complete"`, `"cancelled"`, `"error"`, or `"offline"`.
    pub state: String,
    /// Active G-code filename from `print_stats.filename` (empty if idle).
    pub filename: String,
    /// Normalized print progress in `[0.0 ..= 1.0]`.
    pub progress: f32,
    /// Active print duration in seconds (`print_stats.print_duration`).
    pub print_duration_s: f64,
    /// Total elapsed duration in seconds (`print_stats.total_duration`).
    pub total_duration_s: f64,
    /// Slicer estimated total print time in seconds (`estimated_time` from `/server/files/metadata`), if known.
    pub slicer_estimated_time_s: Option<f64>,
    /// Computed remaining time (ETA) in seconds when printing or paused.
    pub eta_remaining_s: Option<f64>,
    /// Optional status message (`display_status.message` or `print_stats.message`).
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

/// Result of uploading a G-code file to Moonraker (`POST /server/files/upload`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoonrakerUploadReport {
    pub remote_path: String,
    pub bytes_uploaded: usize,
    pub print_started: bool,
}

impl MoonrakerUploadReport {
    pub fn summary_line(&self) -> String {
        let kb = (self.bytes_uploaded as f64) / 1024.0;
        if self.print_started {
            format!(
                "Uploaded '{}' ({kb:.1} KB) to Moonraker and started print.",
                self.remote_path
            )
        } else {
            format!("Uploaded '{}' ({kb:.1} KB) to Moonraker.", self.remote_path)
        }
    }
}

/// Bevy resource holding non-blocking background tasks and latest Moonraker state for the UI.
#[derive(Resource, Default)]
pub struct MoonrakerUiState {
    pub show_api_key: bool,
    pub last_status: Option<Result<MoonrakerPrintStatus, String>>,
    pub last_test_result: Option<Result<MoonrakerConnectionReport, String>>,
    pub last_upload_result: Option<Result<MoonrakerUploadReport, String>>,
    pub pending_status_task: Option<Task<(Result<MoonrakerPrintStatus, String>, Option<(String, Option<f64>)>)>>,
    pub pending_test_task: Option<Task<Result<MoonrakerConnectionReport, String>>>,
    pub pending_upload_task: Option<Task<Result<MoonrakerUploadReport, String>>>,
    pub pending_cert_pick_task: Option<(CertPickKind, Task<Option<String>>)>,
    pub last_poll_instant: Option<Instant>,
    pub cached_metadata_estimate: Option<(String, Option<f64>)>,
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

/// Normalizes a user-configured Moonraker base URI:
/// - Trims whitespace and trailing `/` characters
/// - Prepends `http://` if no `http://` or `https://` scheme is present
pub fn normalize_moonraker_base_uri(raw_uri: &str) -> Result<String, String> {
    let trimmed = raw_uri.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("Printer URI is empty. Enter a Moonraker URL such as http://printer.local:7125".to_string());
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

/// Resolves the Web UI URL (Mainsail / Fluidd) to open in the system browser:
/// - If `cfg.web_ui_uri` is non-empty, normalizes and returns it.
/// - Otherwise derives it from `cfg.printer_uri`, stripping `:7125` if present so it points to
///   the standard HTTP/HTTPS web interface port on the printer host.
pub fn resolve_printer_web_ui_url(cfg: &MoonrakerNetworkConfig) -> Result<String, String> {
    if !cfg.web_ui_uri.trim().is_empty() {
        return normalize_moonraker_base_uri(&cfg.web_ui_uri);
    }
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    if let Some(stripped) = base.strip_suffix(":7125") {
        Ok(stripped.to_string())
    } else if base.contains(":7125/") {
        Ok(base.replacen(":7125/", "/", 1))
    } else {
        Ok(base)
    }
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
        normalized_uri: normalized_uri.to_string(),
        moonraker_version,
        klippy_state,
        hostname,
        state_message,
    })
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

/// Tests connectivity to Moonraker (`/server/info` and `/printer/info`) without uploading or printing anything.
pub fn test_moonraker_connection(
    cfg: &MoonrakerNetworkConfig,
) -> Result<MoonrakerConnectionReport, String> {
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

/// Queries Moonraker for live print status and fetches slicer `estimated_time` metadata once per active filename.
pub fn query_moonraker_print_status(
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

/// Uploads G-code to a Moonraker printer via `POST /server/files/upload` (`multipart/form-data`)
/// and optionally starts printing immediately (`start_print = true`).
pub fn upload_gcode_to_moonraker(
    cfg: &MoonrakerNetworkConfig,
    filename: &str,
    gcode_text: String,
    start_print: bool,
) -> Result<MoonrakerUploadReport, String> {
    let base = normalize_moonraker_base_uri(&cfg.printer_uri)?;
    let client = build_moonraker_client(cfg, Duration::from_secs(90))?;

    let clean_filename = if filename.ends_with(".gcode") || filename.ends_with(".nc") {
        filename.to_string()
    } else {
        format!("{filename}.gcode")
    };
    let bytes_uploaded = gcode_text.len();

    let file_part = multipart::Part::bytes(gcode_text.into_bytes())
        .file_name(clean_filename.clone())
        .mime_str("application/octet-stream")
        .map_err(|e| format!("Failed to construct multipart file payload: {e}"))?;

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

    Ok(MoonrakerUploadReport {
        remote_path,
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
        assert_eq!(upload.remote_path, "test_upload.gcode");
        assert!(upload.print_started);

        server_thread.join().unwrap();
    }
}
