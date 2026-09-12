use crate::error::MoonrakerError;
use crate::model::MoonrakerConfig;
use reqwest::multipart::{Form, Part};
use reqwest::{Client, StatusCode};
use url::Url;

#[derive(Debug, Clone)]
pub struct MoonrakerClient {
    config: MoonrakerConfig,
    base_url: Url,
    http: Client,
}

impl MoonrakerClient {
    pub fn new(config: MoonrakerConfig) -> Result<Self, MoonrakerError> {
        let mut base_url = Url::parse(&config.url)?;
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path().trim_end_matches('/')));
        }
        let http = Client::builder().build()?;
        Ok(Self {
            config,
            base_url,
            http,
        })
    }

    pub fn config(&self) -> &MoonrakerConfig {
        &self.config
    }

    pub fn base_url(&self) -> &Url {
        &self.base_url
    }

    fn apply_auth(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req = req.header("X-Api-Key", key);
        }
        req
    }

    pub async fn check_connection(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("server/info")?;
        let req = self.apply_auth(self.http.get(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Server returned status {}",
                resp.status()
            )))
        }
    }

    pub async fn upload_gcode(
        &self,
        filename: &str,
        gcode_bytes: Vec<u8>,
        start_print: bool,
    ) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("server/files/upload")?;
        let part = Part::bytes(gcode_bytes)
            .file_name(filename.to_string())
            .mime_str("application/octet-stream")
            .map_err(|e| MoonrakerError::UploadFailed(e.to_string()))?;

        let mut form = Form::new().part("file", part);
        if start_print {
            form = form.text("print", "true");
        }

        let req = self.apply_auth(self.http.post(url).multipart(form));
        let resp = req.send().await?;

        if resp.status().is_success() || resp.status() == StatusCode::CREATED {
            Ok(())
        } else {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            Err(MoonrakerError::UploadFailed(format!(
                "Upload failed (status {status}): {body}"
            )))
        }
    }

    pub async fn start_print(&self, filename: &str) -> Result<(), MoonrakerError> {
        let url = self
            .base_url
            .join(&format!("printer/print/start?filename={filename}"))?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Start print failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn pause_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/pause")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Pause failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn resume_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/resume")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Resume failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn cancel_print(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/print/cancel")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Cancel failed: {}",
                resp.status()
            )))
        }
    }

    pub async fn emergency_stop(&self) -> Result<(), MoonrakerError> {
        let url = self.base_url.join("printer/emergency_stop")?;
        let req = self.apply_auth(self.http.post(url));
        let resp = req.send().await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(MoonrakerError::UploadFailed(format!(
                "Emergency stop failed: {}",
                resp.status()
            )))
        }
    }
}
