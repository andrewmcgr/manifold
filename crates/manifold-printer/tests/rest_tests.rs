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
            "result": { "item": { "path": "test.gcode" } }
        })))
        .mount(&mock_server)
        .await;

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
