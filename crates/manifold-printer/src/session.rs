use crate::client::MoonrakerClient;
use crate::model::{ConnectionState, MoonrakerConfig, PrinterTelemetry};
use crate::subscription::{apply_status_delta, build_subscribe_request};
use futures_util::{SinkExt, StreamExt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};
use url::Url;

#[derive(Debug)]
pub enum PrinterAction {
    UploadAndPrint { filename: String, gcode: Vec<u8> },
    UploadOnly { filename: String, gcode: Vec<u8> },
    Pause,
    Resume,
    Cancel,
    EmergencyStop,
    Disconnect,
    Reconnect,
}

#[derive(Clone)]
pub struct PrinterSessionHandle {
    action_tx: mpsc::UnboundedSender<PrinterAction>,
    telemetry: Arc<RwLock<PrinterTelemetry>>,
    uploading: Arc<AtomicBool>,
}

impl PrinterSessionHandle {
    pub fn spawn(config: MoonrakerConfig) -> Self {
        let (action_tx, action_rx) = mpsc::unbounded_channel();
        let telemetry = Arc::new(RwLock::new(PrinterTelemetry::default()));
        let uploading = Arc::new(AtomicBool::new(false));

        let tele_clone = Arc::clone(&telemetry);
        let up_clone = Arc::clone(&uploading);

        std::thread::Builder::new()
            .name("manifold-printer-worker".to_string())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!("Failed to create printer Tokio runtime: {e}");
                        return;
                    }
                };

                rt.block_on(run_worker(config, action_rx, tele_clone, up_clone));
            })
            .expect("Failed to spawn printer worker thread");

        Self {
            action_tx,
            telemetry,
            uploading,
        }
    }

    pub fn send_action(
        &self,
        action: PrinterAction,
    ) -> Result<(), mpsc::error::SendError<PrinterAction>> {
        self.action_tx.send(action)
    }

    pub fn latest_telemetry(&self) -> PrinterTelemetry {
        self.telemetry.read().unwrap().clone()
    }

    pub fn is_uploading(&self) -> bool {
        self.uploading.load(Ordering::Relaxed)
    }
}

async fn run_worker(
    config: MoonrakerConfig,
    mut action_rx: mpsc::UnboundedReceiver<PrinterAction>,
    telemetry: Arc<RwLock<PrinterTelemetry>>,
    uploading: Arc<AtomicBool>,
) {
    let client = match MoonrakerClient::new(config.clone()) {
        Ok(c) => c,
        Err(e) => {
            if let Ok(mut t) = telemetry.write() {
                t.connection_state = ConnectionState::Error(e.to_string());
            }
            return;
        }
    };

    let mut auto_reconnect = config.auto_connect;
    let mut reconnect_attempt = 0;

    let ws_url = {
        let mut u = match Url::parse(&config.url) {
            Ok(u) => u,
            Err(e) => {
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Error(e.to_string());
                }
                return;
            }
        };
        let ws_scheme = if u.scheme() == "https" { "wss" } else { "ws" };
        let _ = u.set_scheme(ws_scheme);
        match u.join("websocket") {
            Ok(ws_u) => ws_u,
            Err(e) => {
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Error(e.to_string());
                }
                return;
            }
        }
    };

    'main_loop: loop {
        if !auto_reconnect {
            if let Ok(mut t) = telemetry.write() {
                t.connection_state = ConnectionState::Disconnected;
            }
            while let Some(action) = action_rx.recv().await {
                match action {
                    PrinterAction::Reconnect => {
                        auto_reconnect = true;
                        reconnect_attempt = 0;
                        break;
                    }
                    _ => warn!("Ignored action while disconnected: {action:?}"),
                }
            }
        }

        if let Ok(mut t) = telemetry.write() {
            if reconnect_attempt == 0 {
                t.connection_state = ConnectionState::Connecting;
            } else {
                t.connection_state = ConnectionState::Reconnecting {
                    attempt: reconnect_attempt,
                };
            }
        }

        let ws_stream_res = connect_async(ws_url.as_str()).await;
        let mut ws_stream = match ws_stream_res {
            Ok((stream, _)) => {
                reconnect_attempt = 0;
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Connected;
                }
                stream
            }
            Err(_err) => {
                reconnect_attempt += 1;
                let backoff_secs = (1u64 << reconnect_attempt.min(4)).min(10);
                if let Ok(mut t) = telemetry.write() {
                    t.connection_state = ConnectionState::Reconnecting {
                        attempt: reconnect_attempt,
                    };
                }
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(backoff_secs)) => {
                        continue 'main_loop;
                    }
                    Some(action) = action_rx.recv() => {
                        if let PrinterAction::Disconnect = action {
                            auto_reconnect = false;
                            continue 'main_loop;
                        }
                    }
                }
                continue 'main_loop;
            }
        };

        // Subscribe to objects
        let sub_req = build_subscribe_request(1);
        let _ = ws_stream.send(Message::Text(sub_req.to_string())).await;

        let mut ping_interval = tokio::time::interval(Duration::from_secs(15));

        'ws_loop: loop {
            tokio::select! {
                _ = ping_interval.tick() => {
                    if let Err(e) = ws_stream.send(Message::Ping(vec![])).await {
                        warn!("WebSocket ping failed: {e}");
                        break 'ws_loop;
                    }
                }
                Some(msg_res) = ws_stream.next() => {
                    match msg_res {
                        Ok(Message::Text(text)) => {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&text) {
                                if val.get("method").and_then(|v| v.as_str()) == Some("notify_status_update") {
                                    if let Some(params) = val.get("params").and_then(|v| v.as_array()) {
                                        if let Some(delta) = params.first() {
                                            if let Ok(mut t) = telemetry.write() {
                                                apply_status_delta(&mut t, delta);
                                            }
                                        }
                                    }
                                } else if let Some(res) = val.get("result").and_then(|v| v.get("status")) {
                                    if let Ok(mut t) = telemetry.write() {
                                        apply_status_delta(&mut t, res);
                                    }
                                }
                            }
                        }
                        Ok(Message::Close(_)) => {
                            info!("WebSocket closed by server");
                            break 'ws_loop;
                        }
                        Err(e) => {
                            warn!("WebSocket stream error: {e}");
                            break 'ws_loop;
                        }
                        _ => {}
                    }
                }
                Some(action) = action_rx.recv() => {
                    match action {
                        PrinterAction::UploadAndPrint { filename, gcode } => {
                            let client = client.clone();
                            let uploading = Arc::clone(&uploading);
                            tokio::spawn(async move {
                                uploading.store(true, Ordering::Relaxed);
                                let res = client.upload_gcode(&filename, gcode, true).await;
                                uploading.store(false, Ordering::Relaxed);
                                if let Err(e) = res {
                                    error!("Upload & Print failed: {e}");
                                }
                            });
                        }
                        PrinterAction::UploadOnly { filename, gcode } => {
                            let client = client.clone();
                            let uploading = Arc::clone(&uploading);
                            tokio::spawn(async move {
                                uploading.store(true, Ordering::Relaxed);
                                let res = client.upload_gcode(&filename, gcode, false).await;
                                uploading.store(false, Ordering::Relaxed);
                                if let Err(e) = res {
                                    error!("Upload Only failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Pause => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.pause_print().await {
                                    error!("Pause failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Resume => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.resume_print().await {
                                    error!("Resume failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Cancel => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.cancel_print().await {
                                    error!("Cancel failed: {e}");
                                }
                            });
                        }
                        PrinterAction::EmergencyStop => {
                            let client = client.clone();
                            tokio::spawn(async move {
                                if let Err(e) = client.emergency_stop().await {
                                    error!("Emergency stop failed: {e}");
                                }
                            });
                        }
                        PrinterAction::Disconnect => {
                            auto_reconnect = false;
                            break 'ws_loop;
                        }
                        PrinterAction::Reconnect => {
                            auto_reconnect = true;
                            break 'ws_loop;
                        }
                    }
                }
            }
        }
    }
}
