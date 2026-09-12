use super::*;
use futures_util::StreamExt;
use serde_json::json;
use std::time::Duration;

#[path = "../../manifold-printer/tests/support/mod.rs"]
mod support;
use support::{subscribe, Server};

async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
fn rendered(panel: &mut PrinterPanel, active: &mut Option<PrinterSessionHandle>) -> String {
    let ctx = egui::Context::default();
    let output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| panel.show(ui, active, |_| {}));
    });
    format!("{:?}", output.shapes)
}
async fn held_upload_retirement(connect_draft: bool, configured: bool) {
    let mut server = Server::start().await;
    let config = MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    };
    let mut panel = PrinterPanel::new(Some(&config));
    let mut active = Some(PrinterSessionHandle::spawn(config));
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| active.as_ref().unwrap().latest_telemetry().fresh).await;
    let session = active.as_ref().unwrap();
    let endpoint = session.endpoint().to_owned();
    let id = session
        .send_action(PrinterAction::UploadAndPrint {
            filename: "part.gcode".into(),
            gcode: vec![1],
        })
        .unwrap();
    server
        .request()
        .await
        .respond
        .send(json!({"result":{"klippy_state":"ready"}}))
        .unwrap();
    server
        .request()
        .await
        .respond
        .send(json!({"result":{"status":{"print_stats":{"state":"standby","filename":""}}}}))
        .unwrap();
    let mut held = server.request().await;
    assert_eq!(held.path, "/server/files/upload");
    session.send_action(PrinterAction::Pause).unwrap();
    let mut next = Server::start().await;
    let next_config = MoonrakerConfig {
        url: next.url.clone(),
        auto_connect: false,
        api_key: None,
    };
    if connect_draft {
        panel.connect_draft(&mut active, next_config);
        let _new_ws = next.socket().await;
        assert_ne!(active.as_ref().unwrap().endpoint(), endpoint);
    } else {
        panel.replace_profile(&mut active, configured.then_some(&next_config));
        assert!(active.is_none());
    }
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(Ok(_)) = ws.next().await {}
    })
    .await
    .expect("retired worker must close WS, not be retained for history");
    tokio::time::timeout(Duration::from_secs(3), &mut held.client_closed)
        .await
        .expect("retired worker must drop held HTTP request")
        .unwrap();
    let text = rendered(&mut panel, &mut active);
    assert!(text.contains(&endpoint), "retired target disappeared");
    assert!(text.contains(&format!("#{id}")));
    assert!(text.contains("outcome unknown"));
    assert!(text.contains("Unsent action rejected"));
    // Repeated profile resets, including missing configuration, do not acknowledge evidence.
    panel.replace_profile(&mut active, None);
    let text = rendered(&mut panel, &mut active);
    assert!(text.contains(&endpoint) && text.contains("outcome unknown"));
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), server.sockets.recv())
            .await
            .is_err()
    );
    assert!(
        server.requests.is_empty(),
        "unsent pause must never dispatch"
    );
    drop(held);
}

#[tokio::test]
async fn profile_retirement_preserves_transmitted_upload_uncertainty_without_worker() {
    held_upload_retirement(false, false).await;
    held_upload_retirement(false, true).await;
}
#[tokio::test]
async fn connect_draft_preserves_old_target_uncertainty_without_worker() {
    held_upload_retirement(true, false).await;
}
#[tokio::test]
async fn repeated_retirements_compact_results_visibly_until_acknowledged() {
    let mut panel = PrinterPanel::new(None);
    for _ in 0..10 {
        let mut server = Server::start().await;
        let mut active = Some(PrinterSessionHandle::spawn(MoonrakerConfig {
            url: server.url.clone(),
            auto_connect: false,
            api_key: None,
        }));
        active
            .as_ref()
            .unwrap()
            .send_action(PrinterAction::EmergencyStop)
            .unwrap();
        let held = server.request().await;
        panel.replace_profile(&mut active, None);
        assert!(active.is_none());
        assert!(panel.retired.len() <= 8);
        drop(held);
    }
    assert_eq!(panel.retired.len(), 8);
    assert_eq!(panel.retired_summary.failed, 2);
    assert_eq!(panel.retired_summary.outcome_unknown, 2);
    // Compacted warnings must also render when the newest profile has no connection.
    let text = rendered(&mut panel, &mut None);
    assert!(text.contains("Older retired targets (details compacted)"));
    assert!(text.contains("2 outcome unknown"));
}
#[tokio::test]
async fn normal_disconnect_keeps_operation_specific_uncertainty_visible() {
    let mut server = Server::start().await;
    let mut active = Some(PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        auto_connect: false,
        api_key: None,
    }));
    let mut panel = PrinterPanel::new(None);
    panel.send(active.as_ref().unwrap(), PrinterAction::EmergencyStop);
    let held = server.request().await;
    panel.send(active.as_ref().unwrap(), PrinterAction::Disconnect);
    let text = rendered(&mut panel, &mut active);
    assert!(text.contains("remote outcome unknown"));
    assert!(text.contains("emergency stop"));
    assert!(text.contains(&server.url));
    drop(held);
}
