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
