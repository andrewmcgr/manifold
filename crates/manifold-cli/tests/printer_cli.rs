#[path = "../../manifold-printer/tests/support/mod.rs"]
mod support;

use futures_util::SinkExt;
use serde_json::json;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use support::{rpc, subscribe, Server};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio_tungstenite::tungstenite::Message;

fn command(directory: &Path) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_manifold"));
    c.current_dir(directory)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    c
}
async fn status(child: &mut Child) -> std::process::ExitStatus {
    tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .unwrap()
        .unwrap()
}
async fn line_containing(
    lines: &mut tokio::io::Lines<BufReader<tokio::process::ChildStderr>>,
    needle: &str,
) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let line = lines
                .next_line()
                .await
                .unwrap()
                .expect("CLI exited before expected output");
            if line.contains(needle) {
                break;
            }
        }
    })
    .await
    .unwrap();
}
fn mesh(dir: &Path) {
    let vertices = [[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]];
    let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mut stl = String::from("solid tetra\n");
    for face in faces {
        stl.push_str("facet normal 0 0 0\nouter loop\n");
        for i in face {
            let [x, y, z] = vertices[i];
            stl.push_str(&format!("vertex {x} {y} {z}\n"));
        }
        stl.push_str("endloop\nendfacet\n");
    }
    stl.push_str("endsolid tetra\n");
    std::fs::write(dir.join("model.stl"), stl).unwrap();
}
#[tokio::test]
async fn url_alone_slices_offline_without_any_printer_requests() {
    let server = Server::start().await;
    let dir = tempfile::tempdir().unwrap();
    mesh(dir.path());
    let mut child = command(dir.path())
        .args(["model.stl", "--printer-url", &server.url])
        .spawn()
        .unwrap();
    assert!(status(&mut child).await.success());
    assert!(dir.path().join("out.gcode").exists());
    assert!(server.requests.is_empty());
    assert!(server.sockets.is_empty());
}
#[tokio::test]
async fn invalid_intent_exits_nonzero_before_loading_or_writing() {
    let dir = tempfile::tempdir().unwrap();
    for args in [
        vec!["missing.stl", "--upload"],
        vec!["--monitor"],
        vec![
            "missing.stl",
            "--upload",
            "--monitor",
            "--printer-url",
            "http://127.0.0.1:7125",
        ],
    ] {
        let output = command(dir.path()).args(args).output().await.unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(error.contains("require"));
        assert!(!error.contains("No such file"));
        assert!(!dir.path().join("out.gcode").exists());
    }
}
#[tokio::test]
async fn monitor_only_labels_actual_file_never_uploads_and_uses_terminal_exit_codes() {
    for (terminal, success) in [("complete", true), ("cancelled", false), ("error", false)] {
        let mut server = Server::start().await;
        let dir = tempfile::tempdir().unwrap();
        let mut child = command(dir.path())
            .args(["--monitor", "--printer-url", &server.url])
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
        let mut ws = server.socket().await;
        subscribe(
            &mut ws,
            json!({"print_stats":{"state":"printing","filename":"actual.gcode"}}),
        )
        .await;
        line_containing(&mut lines, "actual.gcode — Printing").await;
        ws.send(Message::Text(json!({"method":"notify_status_update","params":[{"print_stats":{"state":terminal,"message":"test terminal"}}]}).to_string())).await.unwrap();
        assert_eq!(status(&mut child).await.success(), success);
        assert!(!dir.path().join("out.gcode").exists());
        assert!(server.requests.is_empty());
    }
}
#[tokio::test]
async fn terminal_auth_failure_exits_nonzero_without_upload() {
    let mut server = Server::start().await;
    let dir = tempfile::tempdir().unwrap();
    let mut child = command(dir.path())
        .args([
            "--monitor",
            "--printer-url",
            &server.url,
            "--printer-api-key",
            "secret",
        ])
        .spawn()
        .unwrap();
    let mut ws = server.socket().await;
    let identify = rpc(&mut ws).await;
    ws.send(Message::Text(
        json!({"id":identify["id"],"error":{"code":401,"message":"bad secret"}}).to_string(),
    ))
    .await
    .unwrap();
    assert!(!status(&mut child).await.success());
    assert!(server.requests.is_empty());
}
#[tokio::test]
async fn print_monitor_uses_canonical_upload_name_and_waits_for_active_job() {
    let mut server = Server::start().await;
    let dir = tempfile::tempdir().unwrap();
    mesh(dir.path());
    let mut child = command(dir.path())
        .args([
            "model.stl",
            "--print",
            "--monitor",
            "--printer-url",
            &server.url,
        ])
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"complete","filename":"old.gcode"}}),
    )
    .await;
    let info = server.request().await;
    assert_eq!(info.path, "/server/info");
    info.respond
        .send(json!({"result":{"klippy_state":"ready"}}))
        .unwrap();
    let query = server.request().await;
    assert!(query.path.starts_with("/printer/objects/query?"));
    query.respond.send(json!({"result":{"status":{"print_stats":{"state":"complete","filename":"old.gcode"}}}})).unwrap();
    let upload = server.request().await;
    assert_eq!(upload.path, "/server/files/upload");
    upload.respond.send(json!({"item":{"path":"canonical.gcode","root":"gcodes"},"print_started":true,"print_queued":false})).unwrap();
    line_containing(&mut lines, "canonical.gcode — Complete").await;
    assert!(child.try_wait().unwrap().is_none());
    ws.send(Message::Text(json!({"method":"notify_status_update","params":[{"print_stats":{"state":"printing","filename":"canonical.gcode"}}]}).to_string())).await.unwrap();
    line_containing(&mut lines, "canonical.gcode — Printing").await;
    ws.send(Message::Text(
        json!({"method":"notify_status_update","params":[{"print_stats":{"state":"complete"}}]})
            .to_string(),
    ))
    .await
    .unwrap();
    assert!(status(&mut child).await.success());
    assert!(server.requests.is_empty());
}
#[cfg(unix)]
#[tokio::test]
async fn ctrl_c_exits_monitor_without_print_cancellation() {
    let mut server = Server::start().await;
    let dir = tempfile::tempdir().unwrap();
    let mut child = command(dir.path())
        .args(["--monitor", "--printer-url", &server.url])
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let mut ws = server.socket().await;
    subscribe(
        &mut ws,
        json!({"print_stats":{"state":"printing","filename":"actual.gcode"}}),
    )
    .await;
    line_containing(&mut lines, "actual.gcode — Printing").await;
    let signal = Command::new("kill")
        .args(["-INT", &child.id().unwrap().to_string()])
        .status()
        .await
        .unwrap();
    assert!(signal.success());
    assert!(!status(&mut child).await.success());
    assert!(server.requests.is_empty());
}
