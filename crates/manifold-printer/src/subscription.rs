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

    if let Some(display) = delta.get("display_status") {
        if let Some(progress) = display.get("progress").and_then(|v| v.as_f64()) {
            telemetry.progress_fraction = progress.clamp(0.0, 1.0) as f32;
        }
    }

    if let Some(stats) = delta.get("print_stats") {
        if let Some(state_str) = stats.get("state").and_then(|v| v.as_str()) {
            telemetry.print_state = match state_str.to_lowercase().as_str() {
                "printing" => PrintState::Printing,
                "paused" => PrintState::Paused,
                "complete" => PrintState::Complete,
                "error" => PrintState::Error,
                _ => PrintState::Standby,
            };
        }
        if let Some(filename) = stats.get("filename").and_then(|v| v.as_str()) {
            if !filename.is_empty() {
                telemetry.filename = Some(filename.to_string());
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
        if let Some(est) = toolhead
            .get("estimated_print_time")
            .and_then(|v| v.as_f64())
        {
            if est > 0.0 && telemetry.print_duration_secs > 0 {
                let remaining = (est as u64).saturating_sub(telemetry.print_duration_secs);
                telemetry.estimated_remaining_secs = Some(remaining);
            }
        }
    }

    if let Some(sdcard) = delta.get("virtual_sdcard") {
        if telemetry.progress_fraction == 0.0 {
            if let Some(progress) = sdcard.get("progress").and_then(|v| v.as_f64()) {
                telemetry.progress_fraction = progress.clamp(0.0, 1.0) as f32;
            }
        }
    }
}
