use std::time::Duration;

use egui::{Color32, RichText, Ui};
use manifold_printer::{
    ActionOutcome, ConnectionState, MoonrakerClient, MoonrakerConfig, OperationState, PrintState,
    PrinterAction, PrinterSessionHandle, PrinterTelemetry,
};

pub struct PrinterPanel {
    pub collapsed: bool,
    pub confirming_cancel: bool,
    pub url_input: String,
    pub api_key_input: String,
    pub auto_connect: bool,
    pub action_error: Option<String>,
    has_settings: bool,
    confirmation_target: Option<(usize, u64, u64, Option<String>)>,
}
impl Default for PrinterPanel {
    fn default() -> Self {
        Self {
            collapsed: true,
            ..Self::new(None)
        }
    }
}
impl PrinterPanel {
    pub fn new(config: Option<&MoonrakerConfig>) -> Self {
        Self {
            collapsed: false,
            confirming_cancel: false,
            url_input: config
                .map(|c| c.url.clone())
                .unwrap_or_else(|| MoonrakerConfig::default().url),
            api_key_input: config.and_then(|c| c.api_key.clone()).unwrap_or_default(),
            auto_connect: config.is_some_and(|c| c.auto_connect),
            has_settings: config.is_some(),
            action_error: None,
            confirmation_target: None,
        }
    }
    fn validated_draft(&self) -> Result<MoonrakerConfig, String> {
        let config = MoonrakerConfig {
            url: self.url_input.trim().into(),
            api_key: (!self.api_key_input.is_empty()).then(|| self.api_key_input.clone()),
            auto_connect: self.auto_connect,
        };
        MoonrakerClient::new(config)
            .map(|c| c.config().clone())
            .map_err(|e| e.to_string())
    }
    /// Save Profile uses the visible draft, without needing to connect first.
    pub fn profile_config(&self) -> Result<Option<MoonrakerConfig>, String> {
        if !self.has_settings
            && self.url_input == MoonrakerConfig::default().url
            && self.api_key_input.is_empty()
            && !self.auto_connect
        {
            return Ok(None);
        }
        self.validated_draft().map(Some)
    }
    pub fn replace_profile(
        &mut self,
        session: &mut Option<PrinterSessionHandle>,
        config: Option<&MoonrakerConfig>,
    ) {
        if let Some(old) = session.take() {
            let _ = old.send_action(PrinterAction::Disconnect);
        }
        *self = Self::new(config);
        if let Some(config) = config.filter(|c| c.auto_connect) {
            *session = Some(PrinterSessionHandle::spawn(config.clone()));
        }
    }
    pub fn send(&mut self, session: &PrinterSessionHandle, action: PrinterAction) {
        self.action_error = session.send_action(action).err().map(|e| e.to_string());
    }
    pub fn request_repaint(ctx: &egui::Context, session: Option<&PrinterSessionHandle>) {
        if session.is_some() {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
    }
    fn sync_confirmation(&mut self, session: &PrinterSessionHandle, t: &PrinterTelemetry) {
        let target = (
            session.identity(),
            t.generation,
            t.job_generation,
            t.filename.clone(),
        );
        if self.confirmation_target.as_ref() != Some(&target)
            || !t.fresh
            || !t.print_state.is_active()
        {
            self.confirming_cancel = false;
        }
        self.confirmation_target = Some(target);
    }
    pub fn can_start(session: &PrinterSessionHandle) -> bool {
        let t = session.latest_telemetry();
        t.fresh
            && t.klippy_ready
            && !session.is_uploading()
            && matches!(
                t.print_state,
                PrintState::Standby | PrintState::Complete | PrintState::Cancelled
            )
    }
    pub fn show(
        &mut self,
        ui: &mut Ui,
        session_opt: &mut Option<PrinterSessionHandle>,
        on_save_config: impl FnOnce(MoonrakerConfig),
    ) {
        Self::request_repaint(ui.ctx(), session_opt.as_ref());
        if let Some(session) = session_opt.as_ref() {
            self.sync_confirmation(session, &session.latest_telemetry());
        } else {
            self.confirming_cancel = false;
            self.confirmation_target = None;
        }
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let title = if self.collapsed {"▶ Printer"} else {"▼ Printer"};
                if ui.button(RichText::new(title).strong()).clicked() {self.collapsed = !self.collapsed;}
                let state = session_opt.as_ref().map(|s|s.latest_telemetry().connection_state).unwrap_or_default();
                let color = match state {ConnectionState::Connected=>Color32::GREEN,ConnectionState::Error(_)=>Color32::RED,_=>Color32::YELLOW};
                ui.colored_label(color,format!("{state:?}"));
            });
            if let Some(session) = session_opt.as_ref() {
                ui.label(format!("Active target: {}",session.endpoint()));
                let t = session.latest_telemetry();
                if !t.fresh {ui.colored_label(Color32::YELLOW,"Telemetry stale / unavailable");}
                if let Some(message) = &t.klipper_message {ui.colored_label(Color32::RED,message);}
                if ui.button(RichText::new("⛔ Emergency Stop").color(Color32::RED)).clicked() {
                    self.send(session,PrinterAction::EmergencyStop);
                }
                ui.small("Network stop is best effort, not a hardware safety guarantee.");
                if session.is_start_pending() {
                    ui.colored_label(Color32::YELLOW,"Start pending observation / reconciliation. If uncertain, inspect the printer before explicitly reconnecting.");
                }
                for op in session.operations() {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!("#{} {}:",op.id,op.name));
                        match &op.state {
                            OperationState::Pending => {ui.label("Pending");}
                            OperationState::Running {bytes_read,total_bytes} if *total_bytes > 0 => {
                                ui.add(egui::ProgressBar::new(*bytes_read as f32 / *total_bytes as f32)
                                    .text(format!("Upload: {bytes_read}/{total_bytes} client bytes read")));
                                ui.small("Awaiting server result; transfer progress is not print progress.");
                            }
                            OperationState::Running {..} => {ui.label("In progress");}
                            OperationState::Succeeded(ActionOutcome::Upload(outcome)) => {
                                ui.label(format!("{:?}: {}/{}",outcome.disposition,outcome.root,outcome.path));
                            }
                            OperationState::Succeeded(ActionOutcome::Acknowledged) => {ui.label("Server acknowledged");}
                            OperationState::Failed {message,outcome_unknown} => {
                                ui.colored_label(Color32::RED,message);
                                if *outcome_unknown {ui.colored_label(Color32::YELLOW,"Outcome unknown — inspect printer before retrying");}
                            }
                        }
                        if !op.state.is_pending() && ui.small_button("Acknowledge").clicked() {session.acknowledge(op.id);}
                    });
                }
            }
            if let Some(error) = &self.action_error {ui.colored_label(Color32::RED,error);}
            if self.collapsed {return;}
            ui.separator();
            ui.label("Draft URL:");
            self.has_settings |= ui.text_edit_singleline(&mut self.url_input).changed();
            ui.label("API key (stored in profile as plaintext):");
            self.has_settings |= ui.add(egui::TextEdit::singleline(&mut self.api_key_input).password(true)).changed();
            self.has_settings |= ui.checkbox(&mut self.auto_connect,"Auto-connect when profile loads").changed();
            let mut save = false;
            let mut connect = false;
            ui.horizontal(|ui| {
                save = ui.button("Save connection settings").clicked();
                connect = ui.button("Connect draft").clicked();
                if let Some(session) = session_opt.as_ref() {
                    if ui.button("Disconnect").clicked() {self.send(session,PrinterAction::Disconnect);}
                    if ui.button("Reconnect active").clicked() {self.send(session,PrinterAction::Reconnect);}
                }
            });
            ui.small("Draft edits do not retarget active controls. Disconnect does not cancel a print or undo transmitted commands.");
            if save || connect {
                match self.validated_draft() {
                    Ok(config) => {
                        self.has_settings = true;
                        on_save_config(config.clone());
                        self.action_error = None;
                        if connect {
                            if let Some(old) = session_opt.take() {let _ = old.send_action(PrinterAction::Disconnect);}
                            let mut active = config;
                            active.auto_connect = true; // Explicit Connect is independent of the saved auto-connect preference.
                            *session_opt = Some(PrinterSessionHandle::spawn(active));
                            self.confirming_cancel = false;
                        }
                    }
                    Err(e) => self.action_error = Some(e),
                }
            }
            if let Some(session) = session_opt.as_ref() {
                let t = session.latest_telemetry();
                ui.separator();
                ui.label(format!("Status: {:?} — File: {}",t.print_state,t.filename.as_deref().unwrap_or("unknown")));
                ui.add(egui::ProgressBar::new(t.progress_fraction).text(format!("Print: {:.1}%",t.progress_fraction*100.0)));
                ui.label(format!("Elapsed: {} s — Approx. remaining: {}",t.print_duration_secs,
                    t.estimated_remaining_secs.filter(|_|t.fresh).map(|n|format!("{n} s (progress-based)")).unwrap_or_else(||"unknown".into())));
                if t.current_layer.is_some() || t.total_layers.is_some() {ui.label(format!("Reported layer: {} / {}",t.current_layer.map(|n|n.to_string()).unwrap_or_else(||"?".into()),t.total_layers.map(|n|n.to_string()).unwrap_or_else(||"?".into())));}
                if let Some(z) = t.toolhead_z {ui.label(format!("Z: {z:.2} mm"));}
                if let Some(h) = t.hotend {ui.label(format!("Hotend: {:.1} / {:.0} °C",h.current,h.target));}
                if let Some(b) = t.bed {ui.label(format!("Bed: {:.1} / {:.0} °C",b.current,b.target));}
                ui.add_enabled_ui(t.fresh, |ui| {
                    ui.horizontal(|ui| {
                        if t.print_state == PrintState::Printing && ui.button("⏸ Pause").clicked() {self.send(session,PrinterAction::Pause);}
                        if t.print_state == PrintState::Paused && ui.button("▶ Resume").clicked() {self.send(session,PrinterAction::Resume);}
                        if t.print_state.is_active() {
                            if !self.confirming_cancel {
                                if ui.button("⏹ Cancel").clicked() {self.confirming_cancel = true;}
                            } else {
                                ui.colored_label(Color32::RED,"Cancel this job?");
                                if ui.button("Yes, Cancel").clicked() {self.send(session,PrinterAction::Cancel);self.confirming_cancel=false;}
                                if ui.button("No").clicked() {self.confirming_cancel=false;}
                            }
                        }
                    });
                });
            }
        });
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_profile_replacement_retires_previous_target_even_with_retained_handle() {
        for new_config in [
            None,
            Some(MoonrakerConfig {
                url: "http://draft.invalid".into(),
                auto_connect: false,
                api_key: None,
            }),
        ] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let config = MoonrakerConfig {
                url: format!("http://{}", listener.local_addr().unwrap()),
                auto_connect: false,
                api_key: None,
            };
            let old = PrinterSessionHandle::spawn(config.clone());
            let mut active = Some(old.clone());
            let mut panel = PrinterPanel::new(Some(&config));
            panel.confirming_cancel = true;
            panel.replace_profile(&mut active, new_config.as_ref());
            assert!(active.is_none());
            assert!(!panel.confirming_cancel);
            assert_eq!(
                old.latest_telemetry().connection_state,
                ConnectionState::Disconnected
            );
            assert!(old.send_action(PrinterAction::Pause).is_err());
            assert_eq!(
                panel.profile_config().unwrap(),
                new_config.map(|mut c| {
                    c.url.push('/');
                    c
                })
            );
        }
    }

    #[test]
    fn visible_draft_persists_without_connecting_and_key_is_masked() {
        let mut panel = PrinterPanel::new(None);
        assert_eq!(panel.profile_config().unwrap(), None);
        panel.url_input = "https://example.invalid/proxy".into();
        panel.api_key_input = "secret-draft-key".into();
        panel.auto_connect = false;
        let saved = panel.profile_config().unwrap().unwrap();
        assert_eq!(saved.url, "https://example.invalid/proxy/");
        assert_eq!(saved.api_key.as_deref(), Some("secret-draft-key"));
        assert!(!saved.auto_connect);
        let ctx = egui::Context::default();
        let mut active = None;
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.show(ui, &mut active, |_| {}));
        });
        assert!(!format!("{:?}", output.shapes).contains("secret-draft-key"));
        assert!(active.is_none());
        panel.url_input = "ftp://example.invalid".into();
        assert!(panel.profile_config().is_err());
    }

    #[test]
    fn unavailable_action_feedback_and_repaint_are_visible_without_input() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let session = PrinterSessionHandle::spawn(MoonrakerConfig {
            url: format!("http://{}", listener.local_addr().unwrap()),
            auto_connect: false,
            api_key: None,
        });
        let mut panel = PrinterPanel::new(None);
        panel.send(&session, PrinterAction::Pause);
        let mut active = Some(session);
        let ctx = egui::Context::default();
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.show(ui, &mut active, |_| {}));
        });
        assert!(format!("{:?}", output.shapes).contains("subscription unavailable"));
        assert!(
            output.viewport_output[&egui::ViewportId::ROOT].repaint_delay
                <= Duration::from_millis(250)
        );
    }

    #[test]
    fn cancel_confirmation_resets_on_session_generation_and_job_changes() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let session = PrinterSessionHandle::spawn(MoonrakerConfig {
            url: format!("http://{}", listener.local_addr().unwrap()),
            auto_connect: false,
            api_key: None,
        });
        let mut panel = PrinterPanel::new(None);
        let mut t = PrinterTelemetry {
            fresh: true,
            print_state: PrintState::Printing,
            filename: Some("a.gcode".into()),
            ..Default::default()
        };
        panel.sync_confirmation(&session, &t);
        panel.confirming_cancel = true;
        panel.sync_confirmation(&session, &t);
        assert!(panel.confirming_cancel);
        t.job_generation += 1;
        panel.sync_confirmation(&session, &t);
        assert!(!panel.confirming_cancel);
        panel.confirming_cancel = true;
        t.generation += 1;
        panel.sync_confirmation(&session, &t);
        assert!(!panel.confirming_cancel);
        panel.confirming_cancel = true;
        t.filename = Some("b.gcode".into());
        panel.sync_confirmation(&session, &t);
        assert!(!panel.confirming_cancel);
        panel.confirming_cancel = true;
        t.fresh = false;
        panel.sync_confirmation(&session, &t);
        assert!(!panel.confirming_cancel);
    }

    #[test]
    fn panel_labels_draft_and_active_endpoint_separately() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let config = MoonrakerConfig {
            url: format!("http://{}", listener.local_addr().unwrap()),
            auto_connect: false,
            api_key: None,
        };
        let mut session = Some(PrinterSessionHandle::spawn(config));
        let mut panel = PrinterPanel::new(None);
        panel.url_input = "http://draft.invalid".into();
        let ctx = egui::Context::default();
        let output = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| panel.show(ui, &mut session, |_| {}));
        });
        let text = format!("{:?}", output.shapes);
        assert!(text.contains("Active target:"));
        assert!(text.contains("Draft URL:"));
    }

    #[test]
    fn test_printer_panel_defaults() {
        let panel = PrinterPanel::default();
        assert!(panel.collapsed);
        assert!(!panel.confirming_cancel);
    }

    #[test]
    fn test_printer_panel_new() {
        let panel = PrinterPanel::new(None);
        assert!(!panel.collapsed);
        assert_eq!(panel.url_input, "http://127.0.0.1:7125");
        assert_eq!(panel.api_key_input, "");

        let cfg = MoonrakerConfig {
            url: "http://myprinter.lan:7125".to_string(),
            api_key: Some("key123".to_string()),
            auto_connect: true,
        };
        let panel2 = PrinterPanel::new(Some(&cfg));
        assert!(!panel2.collapsed);
        assert_eq!(panel2.url_input, "http://myprinter.lan:7125");
        assert_eq!(panel2.api_key_input, "key123");
    }
}
