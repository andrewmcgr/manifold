//! One authenticated socket per session epoch; all futures are owned by the session.
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::time::{sleep, timeout, Instant};
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};

use crate::client::MoonrakerClient;
use crate::error::MoonrakerError;
use crate::model::{ConnectionState, PrinterTelemetry};
use crate::session::Shared;
use crate::subscription::{apply_status_delta, build_subscribe_request};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
const RPC_DEADLINE: Duration = Duration::from_secs(10);
const PING_INTERVAL: Duration = Duration::from_secs(15);
const PONG_DEADLINE: Duration = Duration::from_secs(5);

fn update(shared: &Mutex<Shared>, epoch: u64, f: impl FnOnce(&mut PrinterTelemetry)) {
    let mut state = shared.lock().unwrap();
    if state.epoch == epoch {
        f(&mut state.telemetry);
        state.reconcile_start();
    }
}
fn backoff(attempt: u32) -> Duration {
    Duration::from_secs((1u64 << attempt.saturating_sub(1).min(4)).min(10))
}

pub(crate) async fn run(client: &MoonrakerClient, shared: &Arc<Mutex<Shared>>, epoch: u64) {
    let mut attempt = 0;
    loop {
        update(shared, epoch, |t| {
            t.fresh = false;
            t.klippy_ready = false;
            t.estimated_remaining_secs = None;
            t.connection_state = if attempt == 0 {
                ConnectionState::Connecting
            } else {
                ConnectionState::Reconnecting { attempt }
            };
        });
        let mut usable_since = None;
        let result = connected(client, shared, epoch, &mut usable_since).await;
        let error = result.expect_err("socket loop ends only on failure");
        update(shared, epoch, |t| {
            t.fresh = false;
            t.klippy_ready = false;
            t.estimated_remaining_secs = None;
            t.klipper_message = Some(client.redact(&error.to_string()));
        });
        let terminal = error.is_auth()
            || matches!(&error, MoonrakerError::Api {operation, ..} if operation == "subscription" || operation == "identify");
        if terminal {
            update(shared, epoch, |t| {
                t.connection_state = ConnectionState::Error(client.redact(&error.to_string()))
            });
            return; // Explicit reconnect after correcting credentials/settings.
        }
        // A handshake or instant accept/close must not reset the retry budget.
        if usable_since
            .is_some_and(|since: Instant| since.elapsed() >= PING_INTERVAL + PONG_DEADLINE)
        {
            attempt = 0;
        }
        attempt = attempt.saturating_add(1);
        update(shared, epoch, |t| {
            t.connection_state = ConnectionState::Reconnecting { attempt }
        });
        sleep(backoff(attempt)).await;
    }
}

async fn rpc(
    socket: &mut Socket,
    request: Value,
    operation: &str,
) -> Result<Value, MoonrakerError> {
    let id = request["id"].clone();
    timeout(RPC_DEADLINE, async {
        socket.send(Message::Text(request.to_string())).await?;
        loop {
            match socket.next().await {
                Some(Ok(Message::Text(text))) => {
                    let value: Value = serde_json::from_str(&text)?;
                    if matches!(
                        value["method"].as_str(),
                        Some(
                            "notify_klippy_shutdown"
                                | "notify_klippy_disconnected"
                                | "notify_klippy_ready"
                        )
                    ) {
                        return Err(MoonrakerError::rejected(
                            "Klipper lifecycle",
                            "readiness changed during RPC",
                        ));
                    }
                    if value["id"] != id {
                        continue;
                    }
                    if let Some(error) = value.get("error") {
                        return Err(MoonrakerError::Api {
                            operation: operation.into(),
                            code: error["code"].as_i64().unwrap_or(-1),
                            message: error["message"]
                                .as_str()
                                .unwrap_or("request rejected")
                                .into(),
                        });
                    }
                    return value
                        .get("result")
                        .cloned()
                        .ok_or_else(|| MoonrakerError::rejected(operation, "missing RPC result"));
                }
                Some(Ok(Message::Ping(data))) => socket.send(Message::Pong(data)).await?,
                Some(Ok(Message::Close(_))) | None => {
                    return Err(MoonrakerError::rejected(operation, "socket EOF"))
                }
                Some(Err(e)) => return Err(e.into()),
                _ => {}
            }
        }
    })
    .await
    .map_err(|_| MoonrakerError::Timeout("WebSocket RPC"))?
}
async fn connected(
    client: &MoonrakerClient,
    shared: &Mutex<Shared>,
    epoch: u64,
    usable_since: &mut Option<Instant>,
) -> Result<(), MoonrakerError> {
    let connected = timeout(
        Duration::from_secs(5),
        connect_async(client.websocket_url().as_str()),
    )
    .await
    .map_err(|_| MoonrakerError::Timeout("WebSocket connect"))?;
    let (mut socket, _) = connected.map_err(|e| match &e {
        tokio_tungstenite::tungstenite::Error::Http(response)
            if matches!(response.status().as_u16(), 401 | 403) =>
        {
            MoonrakerError::Api {
                operation: "WebSocket handshake".into(),
                code: i64::from(response.status().as_u16()),
                message: "authentication rejected; correct settings and reconnect".into(),
            }
        }
        _ => e.into(),
    })?;
    let mut params = json!({"client_name":"Manifold", "version":env!("CARGO_PKG_VERSION"), "type":"desktop", "url":env!("CARGO_PKG_REPOSITORY")});
    if let Some(key) = &client.config().api_key {
        params["api_key"] = json!(key);
    }
    let identified = rpc(
        &mut socket,
        json!({"jsonrpc":"2.0", "id":1, "method":"server.connection.identify", "params":params}),
        "identify",
    )
    .await?;
    if !identified["connection_id"].is_u64() {
        return Err(MoonrakerError::rejected(
            "identify",
            "missing connection_id",
        ));
    }
    let info = rpc(
        &mut socket,
        json!({"jsonrpc":"2.0", "id":2, "method":"server.info"}),
        "server info",
    )
    .await?;
    if info["klippy_state"] != "ready" {
        return Err(MoonrakerError::rejected(
            "readiness",
            info["klippy_state"]
                .as_str()
                .unwrap_or("unknown Klipper state"),
        ));
    }
    let result = rpc(&mut socket, build_subscribe_request(3), "subscription").await?;
    let status = &result["status"];
    if !status["print_stats"]["state"].is_string() {
        return Err(MoonrakerError::rejected(
            "subscription",
            "missing print_stats snapshot",
        ));
    }
    update(shared, epoch, |t| {
        let generation = t.generation + 1;
        *t = PrinterTelemetry {
            connection_state: ConnectionState::Connected,
            fresh: true,
            klippy_ready: true,
            generation,
            ..Default::default()
        };
        apply_status_delta(t, status);
    });
    *usable_since = Some(Instant::now());
    let mut ping_at = Instant::now() + PING_INTERVAL;
    let mut awaiting_pong = false;
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(ping_at) => {
                if awaiting_pong { return Err(MoonrakerError::Timeout("WebSocket pong")); }
                timeout(PONG_DEADLINE, socket.send(Message::Ping(b"manifold".to_vec()))).await
                    .map_err(|_| MoonrakerError::Timeout("WebSocket ping write"))??;
                awaiting_pong = true;
                ping_at = Instant::now() + PONG_DEADLINE;
            }
            message = socket.next() => match message {
                Some(Ok(Message::Pong(data))) if data == b"manifold" => {
                    awaiting_pong = false; ping_at = Instant::now() + PING_INTERVAL;
                }
                Some(Ok(Message::Ping(data))) => {
                    timeout(PONG_DEADLINE, socket.send(Message::Pong(data))).await.map_err(|_| MoonrakerError::Timeout("WebSocket pong write"))??;
                }
                Some(Ok(Message::Text(text))) => {
                    let value: Value = serde_json::from_str(&text)?;
                    match value["method"].as_str() {
                        Some("notify_status_update") => {
                            if let Some(delta) = value["params"].get(0) { update(shared, epoch, |t| apply_status_delta(t, delta)); }
                        }
                        Some("notify_klippy_shutdown" | "notify_klippy_disconnected" | "notify_klippy_ready") => {
                            return Err(MoonrakerError::rejected("Klipper lifecycle", value["method"].as_str().unwrap_or("changed")));
                        }
                        _ => {}
                    }
                }
                Some(Ok(Message::Close(_))) | None => return Err(MoonrakerError::rejected("telemetry", "socket EOF")),
                Some(Err(e)) => return Err(e.into()),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retries_start_at_one_second_and_cap() {
        assert_eq!(
            (1..=7).map(|n| backoff(n).as_secs()).collect::<Vec<_>>(),
            vec![1, 2, 4, 8, 10, 10, 10]
        );
    }
}
