use std::sync::Arc;
use std::time::Duration;

use futures_util::stream;
use reqwest::multipart::{Form, Part};
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::Mutex;
use url::Url;

use crate::error::MoonrakerError;
use crate::model::{
    MoonrakerConfig, PrintState, PrinterTelemetry, ServerInfo, UploadDisposition, UploadOutcome,
};
use crate::subscription::apply_status_delta;

pub type UploadProgress = Arc<dyn Fn(u64, u64) + Send + Sync>;

#[derive(Clone, Debug)]
pub struct MoonrakerClient {
    config: MoonrakerConfig,
    base_url: Url,
    http: Client,
    transfer: Arc<Mutex<()>>,
}

impl MoonrakerClient {
    pub fn new(mut config: MoonrakerConfig) -> Result<Self, MoonrakerError> {
        let mut base_url = Url::parse(&config.url)?;
        if !matches!(base_url.scheme(), "http" | "https")
            || base_url.host_str().is_none()
            || !base_url.username().is_empty()
            || base_url.password().is_some()
            || base_url.query().is_some()
            || base_url.fragment().is_some()
        {
            return Err(MoonrakerError::rejected(
                "endpoint",
                "use an HTTP(S) directory URL without credentials, query, or fragment",
            ));
        }
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }
        config.url = base_url.to_string();
        let http = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            // Never forward a custom credential header to a redirect destination.
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            config,
            base_url,
            http,
            transfer: Arc::new(Mutex::new(())),
        })
    }
    pub fn config(&self) -> &MoonrakerConfig {
        &self.config
    }
    pub fn base_url(&self) -> &Url {
        &self.base_url
    }
    pub fn websocket_url(&self) -> Url {
        let mut url = self
            .base_url
            .join("websocket")
            .expect("validated directory URL");
        url.set_scheme(if self.base_url.scheme() == "https" {
            "wss"
        } else {
            "ws"
        })
        .expect("compatible scheme");
        url
    }
    pub(crate) fn redact(&self, message: &str) -> String {
        match self.config.api_key.as_deref().filter(|k| !k.is_empty()) {
            Some(key) => message.replace(key, "[REDACTED]"),
            None => message.into(),
        }
    }
    fn apply_auth(&self, req: RequestBuilder) -> RequestBuilder {
        if let Some(key) = &self.config.api_key {
            req.header("X-Api-Key", key)
        } else {
            req
        }
    }
    async fn request(
        &self,
        operation: &str,
        req: RequestBuilder,
        mutating: bool,
    ) -> Result<Value, MoonrakerError> {
        let resp = self.apply_auth(req).send().await.map_err(|e| {
            if mutating && !e.is_connect() && !e.is_builder() {
                MoonrakerError::OutcomeUnknown {
                    operation: operation.into(),
                    message: self.redact(&e.to_string()),
                }
            } else {
                MoonrakerError::rejected(operation, &self.redact(&e.to_string()))
            }
        })?;
        let status = resp.status();
        let body = resp
            .bytes()
            .await
            .map_err(|e| MoonrakerError::OutcomeUnknown {
                operation: operation.into(),
                message: self.redact(&e.to_string()),
            })?;
        let value: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        if !status.is_success() || value.get("error").is_some() {
            let error = value.get("error").unwrap_or(&value);
            return Err(MoonrakerError::Api {
                operation: operation.into(),
                code: error
                    .get("code")
                    .and_then(Value::as_i64)
                    .unwrap_or(i64::from(status.as_u16())),
                message: self.redact(
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or(status.canonical_reason().unwrap_or("invalid response")),
                ),
            });
        }
        Ok(value)
    }
    pub async fn check_connection(&self) -> Result<ServerInfo, MoonrakerError> {
        let value = self
            .request(
                "server info",
                self.http
                    .get(self.base_url.join("server/info")?)
                    .timeout(Duration::from_secs(10)),
                false,
            )
            .await?;
        Ok(serde_json::from_value(value["result"].clone())?)
    }
    pub async fn query_job(&self) -> Result<PrinterTelemetry, MoonrakerError> {
        let info = self.check_connection().await?;
        if info.klippy_state != "ready" {
            return Err(MoonrakerError::rejected(
                "job query",
                &format!("Klipper is {}", info.klippy_state),
            ));
        }
        let value = self
            .request(
                "job query",
                self.http
                    .get(self.base_url.join("printer/objects/query")?)
                    .query(&[("print_stats", "")])
                    .timeout(Duration::from_secs(10)),
                false,
            )
            .await?;
        let status = &value["result"]["status"];
        if !status["print_stats"]["state"].is_string() {
            return Err(MoonrakerError::rejected(
                "job query",
                "missing print_stats state",
            ));
        }
        let mut t = PrinterTelemetry {
            fresh: true,
            klippy_ready: true,
            ..Default::default()
        };
        apply_status_delta(&mut t, status);
        Ok(t)
    }
    async fn guard(&self, filename: &str, start: bool) -> Result<(), MoonrakerError> {
        let job = self.query_job().await?;
        if job.print_state == PrintState::Unknown
            || (start && (job.print_state.is_active() || job.print_state == PrintState::Error))
        {
            return Err(MoonrakerError::rejected(
                "start/upload",
                "printer state does not permit this operation",
            ));
        }
        if job.print_state.is_active() && job.filename.as_deref() == Some(filename) {
            return Err(MoonrakerError::rejected(
                "upload",
                "refusing to overwrite the active print file",
            ));
        }
        Ok(())
    }
    pub async fn upload_gcode(
        &self,
        filename: &str,
        bytes: Vec<u8>,
        start: bool,
    ) -> Result<UploadOutcome, MoonrakerError> {
        self.upload_gcode_with_progress(filename, bytes, start, None)
            .await
    }
    pub async fn upload_gcode_with_progress(
        &self,
        filename: &str,
        bytes: Vec<u8>,
        start: bool,
        progress: Option<UploadProgress>,
    ) -> Result<UploadOutcome, MoonrakerError> {
        let _flight = self
            .transfer
            .try_lock()
            .map_err(|_| MoonrakerError::rejected("upload", "another upload/start is pending"))?;
        if filename.is_empty() || filename.contains(['/', '\\']) || matches!(filename, "." | "..") {
            return Err(MoonrakerError::rejected(
                "upload",
                "provide a file basename, not a directory",
            ));
        }
        self.guard(filename, start).await?;
        let total = bytes.len() as u64;
        let chunks = stream::unfold(
            (bytes, 0usize, progress),
            |(bytes, offset, progress)| async move {
                if offset == bytes.len() {
                    return None;
                }
                let end = (offset + 64 * 1024).min(bytes.len());
                let chunk = bytes[offset..end].to_vec();
                if let Some(callback) = &progress {
                    callback(end as u64, bytes.len() as u64);
                }
                Some((Ok::<_, std::io::Error>(chunk), (bytes, end, progress)))
            },
        );
        let part = Part::stream_with_length(reqwest::Body::wrap_stream(chunks), total)
            .file_name(filename.to_owned())
            .mime_str("application/octet-stream")?;
        let mut form = Form::new().text("root", "gcodes").part("file", part);
        if start {
            form = form.text("print", "true");
        }
        let value = self
            .request(
                "upload",
                self.http
                    .post(self.base_url.join("server/files/upload")?)
                    .multipart(form),
                true,
            )
            .await?;
        #[derive(Deserialize)]
        struct Item {
            path: String,
            root: String,
        }
        #[derive(Deserialize)]
        struct Response {
            item: Item,
            print_started: bool,
            print_queued: bool,
        }
        let response: Response =
            serde_json::from_value(value).map_err(|e| MoonrakerError::OutcomeUnknown {
                operation: "upload".into(),
                message: format!("invalid upload response: {e}"),
            })?;
        if response.item.path.is_empty() || response.item.root != "gcodes" {
            return Err(MoonrakerError::OutcomeUnknown {
                operation: "upload".into(),
                message: "invalid canonical file location".into(),
            });
        }
        let disposition = if response.print_started {
            UploadDisposition::Started
        } else if response.print_queued {
            UploadDisposition::Queued
        } else if start {
            UploadDisposition::StartNotConfirmed
        } else {
            UploadDisposition::Uploaded
        };
        Ok(UploadOutcome {
            path: response.item.path,
            root: response.item.root,
            disposition,
        })
    }
    pub async fn start_print(&self, filename: &str) -> Result<(), MoonrakerError> {
        let _flight = self
            .transfer
            .try_lock()
            .map_err(|_| MoonrakerError::rejected("start", "another upload/start is pending"))?;
        self.guard(filename, true).await?;
        self.request(
            "start",
            self.http
                .post(self.base_url.join("printer/print/start")?)
                .query(&[("filename", filename)])
                .timeout(Duration::from_secs(15)),
            true,
        )
        .await?;
        Ok(())
    }
    async fn control(&self, operation: &str, path: &str) -> Result<(), MoonrakerError> {
        self.request(
            operation,
            self.http
                .post(self.base_url.join(path)?)
                .timeout(Duration::from_secs(10)),
            true,
        )
        .await?;
        Ok(())
    }
    pub async fn pause_print(&self) -> Result<(), MoonrakerError> {
        self.control("pause", "printer/print/pause").await
    }
    pub async fn resume_print(&self) -> Result<(), MoonrakerError> {
        self.control("resume", "printer/print/resume").await
    }
    pub async fn cancel_print(&self) -> Result<(), MoonrakerError> {
        self.control("cancel", "printer/print/cancel").await
    }
    pub async fn emergency_stop(&self) -> Result<(), MoonrakerError> {
        self.control("emergency stop", "printer/emergency_stop")
            .await
    }
}
