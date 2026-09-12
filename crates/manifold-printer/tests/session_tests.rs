use manifold_printer::model::{ConnectionState, MoonrakerConfig};
use manifold_printer::session::{PrinterAction, PrinterSessionHandle};
use std::time::Duration;

#[tokio::test]
async fn test_printer_session_spawns_and_handles_disconnect() {
    let config = MoonrakerConfig {
        url: "http://127.0.0.1:9".to_string(), // Unreachable port
        api_key: None,
        auto_connect: true,
    };
    let session = PrinterSessionHandle::spawn(config);
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Should indicate connecting or reconnecting, without panicking
    let telemetry = session.latest_telemetry();
    assert!(matches!(
        telemetry.connection_state,
        ConnectionState::Connecting
            | ConnectionState::Reconnecting { .. }
            | ConnectionState::Error(_)
    ));

    session.send_action(PrinterAction::Disconnect).unwrap();
}
