use crate::model::{PrintState, PrinterTelemetry, TemperatureState};
use serde_json::Value;

pub fn build_subscribe_request(id: u64) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": "printer.objects.subscribe",
        "params": {
            "objects": {
                "print_stats": null,
                "display_status": null,
                "toolhead": null,
                "extruder": null,
                "heater_bed": null,
                "virtual_sdcard": null
            }
        },
        "id": id
    })
}

pub fn apply_status_delta(telemetry: &mut PrinterTelemetry, delta: &Value) {
    if let Some(extruder) = delta.get("extruder") {
        let current = extruder
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let target = extruder
            .get("target")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        if let (Some(cur), Some(tgt)) = (current, target) {
            telemetry.hotend = Some(TemperatureState {
                current: cur,
                target: tgt,
            });
        } else if let Some(ref mut hotend) = telemetry.hotend {
            if let Some(c) = current {
                hotend.current = c;
            }
            if let Some(t) = target {
                hotend.target = t;
            }
        }
    }

    if let Some(bed) = delta.get("heater_bed") {
        let current = bed
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let target = bed.get("target").and_then(|v| v.as_f64()).map(|v| v as f32);
        if let (Some(cur), Some(tgt)) = (current, target) {
            telemetry.bed = Some(TemperatureState {
                current: cur,
                target: tgt,
            });
        } else if let Some(ref mut bed_state) = telemetry.bed {
            if let Some(c) = current {
                bed_state.current = c;
            }
            if let Some(t) = target {
                bed_state.target = t;
            }
        }
    }

    if let Some(stats) = delta.get("print_stats") {
        if let Some(state_str) = stats.get("state").and_then(|v| v.as_str()) {
            telemetry.print_state = PrintState::parse(state_str);
        }
        if let Some(filename) = stats.get("filename").and_then(Value::as_str) {
            let filename = (!filename.is_empty()).then(|| filename.to_string());
            if telemetry.filename != filename {
                telemetry.display_progress = None;
                telemetry.sd_progress = None;
                telemetry.current_layer = None;
                telemetry.total_layers = None;
                telemetry.klipper_message = None;
                telemetry.print_duration_secs = 0;
                telemetry.total_duration_secs = 0;
            }
            telemetry.filename = filename;
        }
        if let Some(info) = stats.get("info") {
            if let Some(layer) = info.get("current_layer") {
                telemetry.current_layer = layer.as_u64().and_then(|n| n.try_into().ok());
            }
            if let Some(layer) = info.get("total_layer") {
                telemetry.total_layers = layer.as_u64().and_then(|n| n.try_into().ok());
            }
        }
        if let Some(dur) = stats.get("print_duration").and_then(|v| v.as_f64()) {
            telemetry.print_duration_secs = dur.max(0.0) as u64;
        }
        if let Some(total) = stats.get("total_duration").and_then(|v| v.as_f64()) {
            telemetry.total_duration_secs = total.max(0.0) as u64;
        }
        if let Some(msg) = stats.get("message").and_then(|v| v.as_str()) {
            telemetry.klipper_message = if msg.is_empty() {
                None
            } else {
                Some(msg.to_string())
            };
        }
    }

    if let Some(toolhead) = delta.get("toolhead") {
        if let Some(pos) = toolhead.get("position").and_then(|v| v.as_array()) {
            if pos.len() >= 3 {
                if let Some(z) = pos[2].as_f64() {
                    telemetry.toolhead_z = Some(z);
                }
            }
        }
    }
    for (object, target) in [
        ("display_status", &mut telemetry.display_progress),
        ("virtual_sdcard", &mut telemetry.sd_progress),
    ] {
        if let Some(value) = delta.get(object).and_then(|v| v.get("progress")) {
            *target = value
                .as_f64()
                .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
                .map(|p| p as f32);
        }
    }
    telemetry.progress_fraction = telemetry
        .display_progress
        .or(telemetry.sd_progress)
        .unwrap_or(0.0);
    let p = f64::from(telemetry.progress_fraction);
    // Progress-based approximation, never Klipper's MCU motion clock.
    telemetry.estimated_remaining_secs = if telemetry.print_state == PrintState::Printing
        && p > 0.0
        && p <= 1.0
        && telemetry.print_duration_secs > 0
    {
        Some((telemetry.print_duration_secs as f64 * (1.0 - p) / p).round() as u64)
    } else {
        None
    };
}
