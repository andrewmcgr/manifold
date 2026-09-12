use futures_util::{SinkExt, StreamExt};
use manifold_printer::{ConnectionState, MoonrakerConfig, PrinterAction, PrinterSessionHandle};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[tokio::test]
async fn authenticated_subscription_and_final_handle_drop_close_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: format!("http://{}", listener.local_addr().unwrap()),
        api_key: Some("key".into()),
        auto_connect: true,
    });
    let (socket, _) = tokio::time::timeout(Duration::from_secs(3), listener.accept())
        .await
        .unwrap()
        .unwrap();
    let mut ws = accept_async(socket).await.unwrap();
    let first = tokio::time::timeout(Duration::from_secs(3), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let identify: Value = serde_json::from_str(first.to_text().unwrap()).unwrap();
    assert_eq!(identify["method"], "server.connection.identify");
    assert_eq!(identify["params"]["api_key"], "key");
    assert_eq!(identify["params"]["type"], "desktop");
    ws.send(Message::Text(
        json!({"jsonrpc":"2.0", "id":identify["id"], "result":{"connection_id":12}}).to_string(),
    ))
    .await
    .unwrap();
    let info: Value =
        serde_json::from_str(ws.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
    assert_eq!(info["method"], "server.info");
    ws.send(Message::Text(
        json!({"id":info["id"],"result":{"klippy_state":"ready"}}).to_string(),
    ))
    .await
    .unwrap();
    let sub: Value =
        serde_json::from_str(ws.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
    assert_eq!(sub["method"], "printer.objects.subscribe");
    ws.send(Message::Text(json!({"id":sub["id"], "result":{"status":{"print_stats":{"state":"printing", "filename":"actual.gcode"}}}}).to_string())).await.unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if session.latest_telemetry().connection_state == ConnectionState::Connected {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        session.latest_telemetry().filename.as_deref(),
        Some("actual.gcode")
    );
    drop(session);
    tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(Ok(message)) = ws.next().await {
            if matches!(message, Message::Close(_)) {
                break;
            }
        }
    })
    .await
    .expect("last handle must terminate socket");
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), listener.accept())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn unavailable_commands_are_rejected_not_silently_consumed() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: format!("http://{}", listener.local_addr().unwrap()),
        auto_connect: false,
        api_key: None,
    });
    assert!(session.send_action(PrinterAction::Pause).is_err());
}

mod support;
use manifold_printer::OperationState;
use support::{reply, rpc, subscribe, Server};

#[tokio::test]
async fn subscription_rejection_is_terminal_and_redacted() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        api_key: Some("secret".into()),
        auto_connect: true,
    });
    let mut ws = server.socket().await;
    let identify = rpc(&mut ws).await;
    reply(&mut ws, &identify["id"], json!({"connection_id":1})).await;
    let info = rpc(&mut ws).await;
    reply(&mut ws, &info["id"], json!({"klippy_state":"ready"})).await;
    let sub = rpc(&mut ws).await;
    ws.send(Message::Text(
        json!({"id":sub["id"],"error":{"code":401,"message":"bad secret"}}).to_string(),
    ))
    .await
    .unwrap();
    until(|| {
        matches!(
            session.latest_telemetry().connection_state,
            ConnectionState::Error(_)
        )
    })
    .await;
    let t = session.latest_telemetry();
    assert!(!t.fresh);
    assert!(!format!("{t:?}").contains("secret"));
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), server.sockets.recv())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn reconnect_replaces_snapshot_and_disconnect_cancels_backoff() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(&mut ws,json!({"print_stats":{"state":"printing","filename":"old.gcode"},"extruder":{"temperature":200,"target":210}})).await;
    until(|| session.latest_telemetry().fresh).await;
    let generation = session.latest_telemetry().generation;
    drop(ws); // Bare EOF, not a graceful close frame.
    until(|| !session.latest_telemetry().fresh).await;
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    let t = session.latest_telemetry();
    assert!(t.generation > generation);
    assert_eq!(t.filename, None);
    assert_eq!(t.hotend, None);
    session.send_action(PrinterAction::Disconnect).unwrap();
    assert!(!session.latest_telemetry().fresh);
    assert_eq!(
        session.latest_telemetry().connection_state,
        ConnectionState::Disconnected
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), server.sockets.recv())
            .await
            .is_err()
    );
    session.send_action(PrinterAction::Reconnect).unwrap();
    let _ws = server.socket().await;
}

#[tokio::test]
async fn emergency_stop_runs_during_unanswered_identify() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    let _identify = rpc(&mut ws).await;
    assert!(session.send_action(PrinterAction::Pause).is_err());
    let id = session.send_action(PrinterAction::EmergencyStop).unwrap();
    let request = server.request().await;
    assert_eq!(request.path, "/printer/emergency_stop");
    request.respond.send(json!({"result":"ok"})).unwrap();
    until(|| {
        session
            .operations()
            .iter()
            .any(|o| o.id == id && matches!(o.state, OperationState::Succeeded(_)))
    })
    .await;
}

#[tokio::test]
async fn atomic_upload_admission_stop_bypasses_upload_and_invalidates_queued_controls() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    let upload = session
        .send_action(PrinterAction::UploadAndPrint {
            filename: "part.gcode".into(),
            gcode: vec![b'G'; 140000],
        })
        .unwrap();
    assert!(session.is_uploading());
    assert!(session
        .send_action(PrinterAction::UploadOnly {
            filename: "other.gcode".into(),
            gcode: vec![1]
        })
        .is_err());
    let info = server.request().await;
    assert_eq!(info.path, "/server/info");
    info.respond
        .send(json!({"result":{"klippy_state":"ready"}}))
        .unwrap();
    let query = server.request().await;
    assert!(query.path.starts_with("/printer/objects/query?"));
    query
        .respond
        .send(json!({"result":{"status":{"print_stats":{"state":"standby","filename":""}}}}))
        .unwrap();
    let held_upload = server.request().await;
    assert_eq!(held_upload.path, "/server/files/upload");
    until(|| {
        session.operations().iter().any(|o| {
            matches!(
                o.state,
                OperationState::Running {
                    bytes_read: 140000,
                    ..
                }
            )
        })
    })
    .await;
    let pause = session.send_action(PrinterAction::Pause).unwrap();
    // Fill the normal queue while the response is held: stop must still be admitted.
    for _ in 0..7 {
        session.send_action(PrinterAction::Resume).unwrap();
    }
    assert!(session.send_action(PrinterAction::Cancel).is_err());
    let stop = session.send_action(PrinterAction::EmergencyStop).unwrap();
    let emergency = server.request().await;
    assert_eq!(emergency.path, "/printer/emergency_stop");
    emergency.respond.send(json!({"result":"ok"})).unwrap();
    until(|| {
        session
            .operations()
            .iter()
            .any(|o| o.id == stop && matches!(o.state, OperationState::Succeeded(_)))
    })
    .await;
    let ops = session.operations();
    assert!(ops.iter().any(|o| o.id == upload
        && matches!(
            o.state,
            OperationState::Failed {
                outcome_unknown: true,
                ..
            }
        )));
    assert!(ops.iter().any(|o| o.id == pause
        && matches!(
            o.state,
            OperationState::Failed {
                outcome_unknown: false,
                ..
            }
        )));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.requests.recv())
            .await
            .is_err()
    );
    drop(held_upload);
    session.acknowledge(upload);
    assert!(!session.operations().iter().any(|o| o.id == upload));
}

#[tokio::test]
async fn normal_controls_are_ordered() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"printing","filename":"one.gcode"}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    session.send_action(PrinterAction::Pause).unwrap();
    session.send_action(PrinterAction::Resume).unwrap();
    let pause = server.request().await;
    assert_eq!(pause.path, "/printer/print/pause");
    assert!(
        tokio::time::timeout(Duration::from_millis(100), server.requests.recv())
            .await
            .is_err()
    );
    pause.respond.send(json!({"result":"ok"})).unwrap();
    let resume = server.request().await;
    assert_eq!(resume.path, "/printer/print/resume");
    resume.respond.send(json!({"result":"ok"})).unwrap();
}

#[tokio::test]
async fn missing_pong_expires_live_snapshot() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    // Do not poll the server socket: tungstenite would auto-answer a ping when polled.
    tokio::time::timeout(Duration::from_secs(23), async {
        loop {
            if !session.latest_telemetry().fresh {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    assert!(session
        .latest_telemetry()
        .klipper_message
        .unwrap()
        .contains("pong"));
}

#[tokio::test]
async fn started_upload_keeps_admission_reserved_until_job_is_observed() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
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
    server.request().await.respond.send(json!({"item":{"path":"canonical.gcode","root":"gcodes"},"print_started":true,"print_queued":false})).unwrap();
    until(|| {
        session
            .operations()
            .iter()
            .any(|o| o.id == id && matches!(o.state, OperationState::Succeeded(_)))
    })
    .await;
    assert!(
        session.is_uploading(),
        "start must remain reserved while telemetry still shows old standby"
    );
    assert!(session
        .send_action(PrinterAction::UploadAndPrint {
            filename: "other.gcode".into(),
            gcode: vec![1]
        })
        .is_err());
    ws.send(Message::Text(json!({"method":"notify_status_update","params":[{"print_stats":{"state":"printing","filename":"canonical.gcode"}}]}).to_string())).await.unwrap();
    until(|| !session.is_uploading()).await;
}

pub async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while !predicate() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn accept_close_is_backed_off_and_drop_cancels_retry() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let ws = server.socket().await;
    drop(ws);
    until(|| {
        matches!(
            session.latest_telemetry().connection_state,
            ConnectionState::Reconnecting { attempt: 1 }
        )
    })
    .await;
    assert!(
        tokio::time::timeout(Duration::from_millis(700), server.sockets.recv())
            .await
            .is_err()
    );
    let ws = server.socket().await;
    drop(ws);
    until(|| {
        matches!(
            session.latest_telemetry().connection_state,
            ConnectionState::Reconnecting { attempt: 2 }
        )
    })
    .await;
    drop(session);
    assert!(
        tokio::time::timeout(Duration::from_millis(2200), server.sockets.recv())
            .await
            .is_err()
    );
}

#[tokio::test]
async fn unanswered_subscription_has_deadline_and_lifecycle_invalidates_snapshot() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    let identify = rpc(&mut ws).await;
    reply(&mut ws, &identify["id"], json!({"connection_id":1})).await;
    let info = rpc(&mut ws).await;
    reply(&mut ws, &info["id"], json!({"klippy_state":"ready"})).await;
    let sub = rpc(&mut ws).await;
    assert_eq!(sub["method"], "printer.objects.subscribe");
    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            if matches!(
                session.latest_telemetry().connection_state,
                ConnectionState::Reconnecting { .. }
            ) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(!session.latest_telemetry().fresh);
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"printing","filename":"a.gcode"}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    ws.send(Message::Text(
        json!({"method":"notify_klippy_shutdown","params":[]}).to_string(),
    ))
    .await
    .unwrap();
    until(|| !session.latest_telemetry().fresh).await;
    assert!(!session.latest_telemetry().klippy_ready);
    assert!(session.send_action(PrinterAction::Resume).is_err());
}

#[tokio::test]
async fn emergency_http_timeout_reports_unknown_without_mutation_replay() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        auto_connect: false,
        api_key: None,
    });
    let id = session.send_action(PrinterAction::EmergencyStop).unwrap();
    let held = server.request().await;
    assert_eq!(held.path, "/printer/emergency_stop");
    tokio::time::timeout(Duration::from_secs(12), async {
        loop {
            if session
                .operations()
                .iter()
                .any(|o| o.id == id && !o.state.is_pending())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(session.operations().iter().any(|o| o.id == id
        && matches!(
            o.state,
            OperationState::Failed {
                outcome_unknown: true,
                ..
            }
        )));
    assert!(server.requests.is_empty());
    drop(held);
}

#[tokio::test]
async fn repeated_emergency_http_dispatch_survives_saturated_failed_history() {
    let mut server = Server::start().await;
    let session = PrinterSessionHandle::spawn(MoonrakerConfig {
        url: server.url.clone(),
        ..Default::default()
    });
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"standby","filename":""}}),
    )
    .await;
    until(|| session.latest_telemetry().fresh).await;
    // Construct the review's exact imbalance through real admission and HTTP completion.
    for index in 0..63 {
        let id = session.send_action(PrinterAction::Pause).unwrap();
        let request = server.request().await;
        assert_eq!(request.path, "/printer/print/pause");
        request
            .respond
            .send(if index == 0 {
                json!({})
            } else {
                json!({"error":{"code":409,"message":"normal failure retained"}})
            })
            .unwrap();
        until(|| {
            session
                .operations()
                .iter()
                .any(|op| op.id == id && !op.state.is_pending())
        })
        .await;
    }
    assert_eq!(session.operations().len(), 63);
    assert!(session
        .operations()
        .iter()
        .all(|op| matches!(op.state, OperationState::Failed { .. })));
    assert!(session.send_action(PrinterAction::Pause).is_err());
    session.send_action(PrinterAction::Disconnect).unwrap();
    // No WS, no acknowledgment, and no automatic replay. Each attempt must reach HTTP.
    for attempt in 0..100 {
        let id = session
            .send_action(PrinterAction::EmergencyStop)
            .unwrap_or_else(|e| panic!("explicit stop attempt {attempt} blocked: {e}"));
        let held = server.request().await;
        assert_eq!(held.path, "/printer/emergency_stop");
        assert!(
            session.send_action(PrinterAction::EmergencyStop).is_err(),
            "coalesce/reject only while in flight"
        );
        assert!(server.requests.is_empty());
        held.respond
            .send(if attempt % 2 == 0 {
                json!({"error":{"code":409,"message":"stop failed"}})
            } else {
                json!({})
            })
            .unwrap();
        until(|| {
            session
                .operations()
                .iter()
                .any(|op| op.id == id && !op.state.is_pending())
        })
        .await;
        assert!(session.operations().iter().any(|op| op.id == id && matches!(op.state, OperationState::Failed { outcome_unknown, .. } if outcome_unknown == (attempt % 2 == 1))));
        assert!(
            session.operations().len() <= 64,
            "retention must stay bounded"
        );
    }
    assert_eq!(
        session
            .operations()
            .iter()
            .filter(|op| op.name == "pause")
            .count(),
        63
    );
    let summary = session.emergency_summary();
    assert_eq!(summary.failed, 99);
    assert_eq!(summary.outcome_unknown, 49);
    assert_eq!(summary.succeeded, 0);
    session.acknowledge_emergency_summary();
    assert!(session.emergency_summary().is_empty());
    assert_eq!(
        session.operations().len(),
        64,
        "summary acknowledgment must not erase current or normal evidence"
    );
    assert!(server.requests.is_empty());
}
