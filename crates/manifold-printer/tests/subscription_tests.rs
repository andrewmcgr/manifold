use manifold_printer::model::{PrintState, PrinterTelemetry};
use manifold_printer::subscription::apply_status_delta;

#[test]
fn test_apply_status_delta_temperatures_and_progress() {
    let mut telemetry = PrinterTelemetry::default();
    let delta = serde_json::json!({
        "extruder": { "temperature": 215.2, "target": 215.0 },
        "heater_bed": { "temperature": 60.1, "target": 60.0 },
        "display_status": { "progress": 0.45 },
        "print_stats": {
            "state": "printing",
            "filename": "benchy.gcode",
            "print_duration": 120.0,
            "total_duration": 140.0
        },
        "toolhead": {
            "position": [100.0, 100.0, 12.4, 45.0]
        }
    });

    apply_status_delta(&mut telemetry, &delta);

    assert_eq!(telemetry.print_state, PrintState::Printing);
    assert_eq!(telemetry.filename.as_deref(), Some("benchy.gcode"));
    assert!((telemetry.progress_fraction - 0.45).abs() < 1e-4);
    assert_eq!(telemetry.print_duration_secs, 120);
    assert_eq!(telemetry.total_duration_secs, 140);
    assert_eq!(telemetry.toolhead_z, Some(12.4));
    assert_eq!(telemetry.hotend.as_ref().unwrap().current, 215.2);
    assert_eq!(telemetry.hotend.as_ref().unwrap().target, 215.0);
    assert_eq!(telemetry.bed.as_ref().unwrap().current, 60.1);
    assert_eq!(telemetry.bed.as_ref().unwrap().target, 60.0);
}

#[test]
fn sd_progress_advances_on_successive_deltas() {
    let mut t = PrinterTelemetry::default();
    apply_status_delta(
        &mut t,
        &serde_json::json!({"virtual_sdcard":{"progress":0.1}}),
    );
    apply_status_delta(
        &mut t,
        &serde_json::json!({"virtual_sdcard":{"progress":0.2}}),
    );
    assert_eq!(t.progress_fraction, 0.2);
}

#[test]
fn cleared_filename_and_layers_do_not_retain_old_job() {
    let mut t = PrinterTelemetry::default();
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"filename":"old.gcode","message":"oops","info":{"current_layer":3,"total_layer":10}}}),
    );
    assert_eq!(t.current_layer, Some(3));
    assert_eq!(t.total_layers, Some(10));
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"filename":"","message":""}}),
    );
    assert_eq!(t.filename, None);
    assert_eq!(t.current_layer, None);
    assert_eq!(t.klipper_message, None);
}

#[test]
fn progress_sources_null_fallback_pause_eta_and_cancelled_unknown() {
    let mut t = PrinterTelemetry::default();
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"filename":"one.gcode","state":"printing","print_duration":100},
        "display_status":{"progress":0.5},"virtual_sdcard":{"progress":0.2},"toolhead":{"estimated_print_time":999999}}),
    );
    assert_eq!(t.estimated_remaining_secs, Some(100));
    apply_status_delta(
        &mut t,
        &serde_json::json!({"virtual_sdcard":{"progress":0.4}}),
    );
    assert_eq!(t.progress_fraction, 0.5);
    apply_status_delta(
        &mut t,
        &serde_json::json!({"display_status":{"progress":null}}),
    );
    assert_eq!(t.progress_fraction, 0.4);
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"state":"paused"}}),
    );
    assert_eq!(t.estimated_remaining_secs, None);
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"state":"cancelled"}}),
    );
    assert_eq!(t.print_state, PrintState::Cancelled);
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"state":"future-state"}}),
    );
    assert_eq!(t.print_state, PrintState::Unknown);
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"filename":"two.gcode","state":"printing"}}),
    );
    assert_eq!(t.progress_fraction, 0.0);
    assert_eq!(t.estimated_remaining_secs, None);
}

#[test]
fn same_filename_restart_resets_job_estimates() {
    let mut t = PrinterTelemetry::default();
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"filename":"same.gcode","state":"complete","print_duration":1000,"info":{"current_layer":100}},"display_status":{"progress":1.0}}),
    );
    apply_status_delta(
        &mut t,
        &serde_json::json!({"print_stats":{"state":"printing"}}),
    );
    assert_eq!(t.progress_fraction, 0.0);
    assert_eq!(t.current_layer, None);
    assert_eq!(t.estimated_remaining_secs, None);
}
