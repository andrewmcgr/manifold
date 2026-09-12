use manifold_printer::{MoonrakerClient, MoonrakerConfig};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_check_connection_success() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/server/info"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "result": { "klippy_state": "ready" }
        })))
        .mount(&mock_server)
        .await;

    let config = MoonrakerConfig {
        url: mock_server.uri(),
        api_key: None,
        auto_connect: false,
    };
    let client = MoonrakerClient::new(config).unwrap();
    assert!(client.check_connection().await.is_ok());
}

#[tokio::test]
async fn test_upload_gcode_and_start_print() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/server/files/upload"))
        .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
            "item": { "path": "test.gcode", "root":"gcodes" }, "print_started":true, "print_queued":false
        })))
        .mount(&mock_server)
        .await;

    Mock::given(method("GET"))
        .and(path("/server/info"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"result":{"klippy_state":"ready"}})),
        )
        .mount(&mock_server)
        .await;
    Mock::given(method("GET")).and(path("/printer/objects/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"result":{"status":{"print_stats":{"state":"standby","filename":""}}}})))
        .mount(&mock_server).await;
    let config = MoonrakerConfig {
        url: mock_server.uri(),
        api_key: None,
        auto_connect: false,
    };
    let client = MoonrakerClient::new(config).unwrap();
    let result = client
        .upload_gcode("test.gcode", b"G28\nG1 Z10\n".to_vec(), true)
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn start_preserves_reserved_filename() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/server/info"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"result":{"klippy_state":"ready"}})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET")).and(path("/printer/objects/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"result":{"status":{"print_stats":{"state":"standby","filename":""}}}})))
        .mount(&server).await;
    Mock::given(method("POST"))
        .and(path("/printer/print/start"))
        .and(wiremock::matchers::query_param(
            "filename",
            "a&b#2+%雪.gcode",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"result":"ok"})))
        .expect(1)
        .mount(&server)
        .await;
    let client = MoonrakerClient::new(MoonrakerConfig {
        url: server.uri(),
        ..Default::default()
    })
    .unwrap();
    assert!(client.start_print("a&b#2+%雪.gcode").await.is_ok());
}

#[test]
fn config_debug_redacts_key_and_rejects_credentials_in_url() {
    let c = MoonrakerConfig {
        api_key: Some("secret-key".into()),
        ..Default::default()
    };
    assert!(!format!("{c:?}").contains("secret-key"));
    for url in [
        "ftp://localhost",
        "http://user:secret@localhost",
        "http://localhost?api_key=secret",
    ] {
        assert!(MoonrakerClient::new(MoonrakerConfig {
            url: url.into(),
            ..Default::default()
        })
        .is_err());
    }
}

async fn ready(server: &MockServer, state: &str, filename: &str) {
    Mock::given(method("GET"))
        .and(path("/server/info"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"result":{"klippy_state":"ready"}})),
        )
        .mount(server)
        .await;
    Mock::given(method("GET")).and(path("/printer/objects/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"result":{"status":{"print_stats":{"state":state,"filename":filename}}}})))
        .mount(server).await;
}

#[tokio::test]
async fn upload_direct_outcomes_multipart_auth_and_chunk_progress() {
    use manifold_printer::UploadDisposition::*;
    for (start, started, queued, want) in [
        (false, false, false, Uploaded),
        (true, true, false, Started),
        (true, false, true, Queued),
        (true, false, false, StartNotConfirmed),
    ] {
        let server = MockServer::start().await;
        ready(&server, "standby", "").await;
        Mock::given(method("POST")).and(path("/server/files/upload"))
            .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                "item":{"path":"canonical.gcode","root":"gcodes","modified":123,"size":140000,"permissions":"rw"},
                "print_started":started,"print_queued":queued,"action":"create_file"
            }))).expect(1).mount(&server).await;
        let client = MoonrakerClient::new(MoonrakerConfig {
            url: server.uri(),
            api_key: Some("secret".into()),
            auto_connect: false,
        })
        .unwrap();
        let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture = events.clone();
        let result = client
            .upload_gcode_with_progress(
                "part.gcode",
                vec![b'G'; 140000],
                start,
                Some(std::sync::Arc::new(move |sent, total| {
                    capture.lock().unwrap().push((sent, total))
                })),
            )
            .await
            .unwrap();
        assert_eq!(result.path, "canonical.gcode");
        assert_eq!(result.disposition, want);
        assert_eq!(
            *events.lock().unwrap(),
            vec![
                (0, 140000),
                (65536, 140000),
                (131072, 140000),
                (140000, 140000)
            ]
        );
        let requests = server.received_requests().await.unwrap();
        assert!(requests.iter().all(|r| r.headers["x-api-key"] == "secret"));
        let upload = requests
            .iter()
            .find(|r| r.url.path() == "/server/files/upload")
            .unwrap();
        let body = String::from_utf8_lossy(&upload.body);
        assert!(body.contains("filename=\"part.gcode\""));
        assert!(body.contains("name=\"root\"\r\n\r\ngcodes"));
        assert_eq!(body.contains("name=\"print\"\r\n\r\ntrue"), start);
        assert!(body.contains(&"G".repeat(140000)));
        assert!(!requests
            .iter()
            .any(|r| r.url.path() == "/printer/print/start"));
    }
}

#[tokio::test]
async fn active_paused_unknown_and_shutdown_guard_mutations() {
    for state in ["printing", "paused", "future-state", "error"] {
        let server = MockServer::start().await;
        ready(&server, state, "active.gcode").await;
        let client = MoonrakerClient::new(MoonrakerConfig {
            url: server.uri(),
            ..Default::default()
        })
        .unwrap();
        assert!(client
            .upload_gcode("active.gcode", vec![1], true)
            .await
            .is_err());
        if state == "printing" || state == "paused" {
            assert!(client
                .upload_gcode("active.gcode", vec![1], false)
                .await
                .is_err());
        }
        assert!(server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| r.method == "GET"));
    }
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"result":{"klippy_state":"shutdown"}})),
        )
        .mount(&server)
        .await;
    let client = MoonrakerClient::new(MoonrakerConfig {
        url: server.uri(),
        ..Default::default()
    })
    .unwrap();
    assert!(client
        .start_print("a.gcode")
        .await
        .unwrap_err()
        .to_string()
        .contains("shutdown"));
}

#[tokio::test]
async fn malformed_upload_is_unknown_and_control_errors_keep_operation_detail() {
    let server = MockServer::start().await;
    ready(&server, "standby", "").await;
    Mock::given(method("POST"))
        .and(path("/server/files/upload"))
        .respond_with(
            ResponseTemplate::new(201)
                .set_body_json(serde_json::json!({"result":{"item":{"path":"old-mock.gcode"}}})),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/printer/print/pause"))
        .respond_with(ResponseTemplate::new(409).set_body_json(
            serde_json::json!({"error":{"code":409,"message":"not printing secret"}}),
        ))
        .mount(&server)
        .await;
    let client = MoonrakerClient::new(MoonrakerConfig {
        url: server.uri(),
        api_key: Some("secret".into()),
        auto_connect: false,
    })
    .unwrap();
    assert!(client
        .upload_gcode("a.gcode", vec![1], true)
        .await
        .unwrap_err()
        .outcome_unknown());
    let error = client.pause_print().await.unwrap_err().to_string();
    assert!(error.contains("pause") && error.contains("409") && error.contains("not printing"));
    assert!(!error.contains("secret"));
}

#[tokio::test]
async fn normalized_proxy_paths_and_redirect_do_not_leak_key() {
    let destination = MockServer::start().await;
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/proxy/server/info"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("location", format!("{}/server/info", destination.uri())),
        )
        .mount(&server)
        .await;
    for suffix in ["/proxy", "/proxy/"] {
        let client = MoonrakerClient::new(MoonrakerConfig {
            url: format!("{}{suffix}", server.uri()),
            api_key: Some("secret".into()),
            auto_connect: false,
        })
        .unwrap();
        assert_eq!(client.websocket_url().path(), "/proxy/websocket");
        assert!(client.websocket_url().query().is_none());
        assert!(client.check_connection().await.is_err());
    }
    assert!(destination.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn malformed_control_success_is_not_an_acknowledgement() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/printer/print/cancel"))
        .respond_with(ResponseTemplate::new(200).set_body_string("truncated proxy response"))
        .mount(&server)
        .await;
    let client = MoonrakerClient::new(MoonrakerConfig {
        url: server.uri(),
        ..Default::default()
    })
    .unwrap();
    assert!(client.cancel_print().await.unwrap_err().outcome_unknown());
}

#[tokio::test]
async fn gateway_and_server_mutation_errors_are_unknown_not_definitive_rejections() {
    for (status, body) in [
        (502, "<html>Bad Gateway secret</html>"),
        (504, "upstream timed out after forwarding secret"),
        (
            500,
            r#"{"error":{"code":500,"message":"internal error secret"}}"#,
        ),
        (
            200,
            r#"{"error":{"code":500,"message":"internal error secret"}}"#,
        ),
    ] {
        let server = MockServer::start().await;
        ready(&server, "standby", "").await;
        Mock::given(method("POST"))
            .and(path("/server/files/upload"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body))
            .expect(1)
            .mount(&server)
            .await;
        let client = MoonrakerClient::new(MoonrakerConfig {
            url: server.uri(),
            api_key: Some("secret".into()),
            auto_connect: false,
        })
        .unwrap();
        let error = client
            .upload_gcode("part.gcode", vec![1], true)
            .await
            .unwrap_err();
        assert!(
            error.outcome_unknown(),
            "gateway/server response cannot prove nonexecution: {error:?}"
        );
        let display = error.to_string();
        assert!(display.contains("upload") && display.contains(&status.to_string()));
        assert!(!display.contains("secret") && !format!("{error:?}").contains("secret"));
        assert_eq!(
            server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .filter(|r| r.method == "POST")
                .count(),
            1
        );
    }
}
#[tokio::test]
async fn malformed_decode_errors_redact_credentials_in_display_and_debug() {
    for (upload, key) in [
        (false, "secret-key"),
        (true, "secret-key"),
        (true, "secret\"key\\suffix"),
    ] {
        let server = MockServer::start().await;
        if upload {
            ready(&server, "standby", "").await;
        }
        Mock::given(path(if upload { "/server/files/upload" } else { "/server/info" }))
            .respond_with(ResponseTemplate::new(200).set_body_json(if upload {
                serde_json::json!({"item":{"path":"part.gcode","root":"gcodes"},"print_started":key,"print_queued":false})
            } else {
                serde_json::json!({"result":key})
            })).mount(&server).await;
        let client = MoonrakerClient::new(MoonrakerConfig {
            url: server.uri(),
            api_key: Some(key.into()),
            auto_connect: false,
        })
        .unwrap();
        let error = if upload {
            client
                .upload_gcode("part.gcode", vec![1], true)
                .await
                .unwrap_err()
        } else {
            client.check_connection().await.unwrap_err()
        };
        assert!(!error.to_string().contains(key), "{error}");
        assert!(!format!("{error:?}").contains(key));
        let quoted = format!("{key:?}");
        let escaped = &quoted[1..quoted.len() - 1];
        assert!(
            !error.to_string().contains(escaped),
            "escaped key leaked: {error}"
        );
        let debug_quoted = format!("{escaped:?}");
        assert!(!format!("{error:?}").contains(&debug_quoted[1..debug_quoted.len() - 1]));
        assert!(error.to_string().contains("invalid"));
        assert!(error.to_string().contains("[REDACTED]"));
    }
}

#[tokio::test]
async fn fresh_query_failure_is_known_unsent_even_for_gateway_response() {
    let server = MockServer::start().await;
    Mock::given(path("/server/info"))
        .respond_with(ResponseTemplate::new(504).set_body_string("gateway unavailable"))
        .mount(&server)
        .await;
    let client = MoonrakerClient::new(MoonrakerConfig {
        url: server.uri(),
        ..Default::default()
    })
    .unwrap();
    let error = client
        .upload_gcode("part.gcode", vec![1], true)
        .await
        .unwrap_err();
    assert!(!error.outcome_unknown());
    assert!(server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .all(|request| request.method == "GET"));
}
