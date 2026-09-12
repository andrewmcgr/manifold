use egui::{Color32, RichText, Ui};
use manifold_printer::{
    ConnectionState, MoonrakerConfig, PrintState, PrinterAction, PrinterSessionHandle,
};

// Allow dead_code temporarily while Task 8 wires PrinterPanel into app.rs
#[allow(dead_code)]
pub struct PrinterPanel {
    pub collapsed: bool,
    pub confirming_cancel: bool,
    pub url_input: String,
    pub api_key_input: String,
}

impl Default for PrinterPanel {
    fn default() -> Self {
        Self {
            collapsed: true,
            confirming_cancel: false,
            url_input: "http://127.0.0.1:7125".to_string(),
            api_key_input: String::new(),
        }
    }
}

#[allow(dead_code)]
impl PrinterPanel {
    pub fn new(config: Option<&MoonrakerConfig>) -> Self {
        Self {
            collapsed: false,
            confirming_cancel: false,
            url_input: config
                .map(|c| c.url.clone())
                .unwrap_or_else(|| "http://127.0.0.1:7125".to_string()),
            api_key_input: config.and_then(|c| c.api_key.clone()).unwrap_or_default(),
        }
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        session_opt: &mut Option<PrinterSessionHandle>,
        on_save_config: impl FnOnce(MoonrakerConfig),
    ) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let header_text = if self.collapsed {
                    "▶ Printer"
                } else {
                    "▼ Printer"
                };
                if ui.button(RichText::new(header_text).strong()).clicked() {
                    self.collapsed = !self.collapsed;
                }

                if let Some(session) = session_opt.as_ref() {
                    let tele = session.latest_telemetry();
                    match &tele.connection_state {
                        ConnectionState::Connected => {
                            ui.colored_label(Color32::from_rgb(0, 200, 0), "● Connected");
                        }
                        ConnectionState::Connecting => {
                            ui.colored_label(Color32::YELLOW, "◌ Connecting...");
                        }
                        ConnectionState::Reconnecting { attempt } => {
                            ui.colored_label(
                                Color32::YELLOW,
                                format!("◌ Reconnecting ({attempt})..."),
                            );
                        }
                        ConnectionState::Disconnected => {
                            ui.colored_label(Color32::GRAY, "○ Disconnected");
                        }
                        ConnectionState::Error(err) => {
                            ui.colored_label(Color32::RED, format!("⚠ Error: {err}"));
                        }
                    }
                } else {
                    ui.colored_label(Color32::GRAY, "○ Disconnected");
                }
            });

            if self.collapsed {
                return;
            }

            ui.separator();

            // Connection settings
            ui.horizontal(|ui| {
                ui.label("URL:");
                ui.text_edit_singleline(&mut self.url_input);
                if session_opt.is_none() {
                    if ui.button("Connect").clicked() {
                        let config = MoonrakerConfig {
                            url: self.url_input.clone(),
                            api_key: if self.api_key_input.is_empty() {
                                None
                            } else {
                                Some(self.api_key_input.clone())
                            },
                            auto_connect: true,
                        };
                        on_save_config(config.clone());
                        *session_opt = Some(PrinterSessionHandle::spawn(config));
                    }
                } else if ui.button("Disconnect").clicked() {
                    if let Some(session) = session_opt.take() {
                        let _ = session.send_action(PrinterAction::Disconnect);
                    }
                }
            });

            if let Some(session) = session_opt.as_ref() {
                let tele = session.latest_telemetry();
                if matches!(tele.connection_state, ConnectionState::Connected) {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(format!("Status: {:?}", tele.print_state));
                        if let Some(ref name) = tele.filename {
                            ui.label(format!("File: {name}"));
                        }
                    });

                    // Progress bar
                    let progress = tele.progress_fraction;
                    ui.add(egui::ProgressBar::new(progress).show_percentage());

                    // Thermals & Z
                    ui.horizontal(|ui| {
                        if let Some(ref h) = tele.hotend {
                            ui.label(format!("Hotend: {:.1} / {:.0} °C", h.current, h.target));
                        }
                        if let Some(ref b) = tele.bed {
                            ui.label(format!("Bed: {:.1} / {:.0} °C", b.current, b.target));
                        }
                        if let Some(z) = tele.toolhead_z {
                            ui.label(format!("Z: {:.2} mm", z));
                        }
                    });

                    // Controls
                    ui.horizontal(|ui| {
                        match tele.print_state {
                            PrintState::Printing if ui.button("⏸ Pause").clicked() => {
                                let _ = session.send_action(PrinterAction::Pause);
                            }
                            PrintState::Paused if ui.button("▶ Resume").clicked() => {
                                let _ = session.send_action(PrinterAction::Resume);
                            }
                            _ => {}
                        }

                        if matches!(tele.print_state, PrintState::Printing | PrintState::Paused) {
                            if !self.confirming_cancel {
                                if ui.button("⏹ Cancel").clicked() {
                                    self.confirming_cancel = true;
                                }
                            } else {
                                ui.colored_label(Color32::RED, "Sure?");
                                if ui.button("Yes, Cancel").clicked() {
                                    let _ = session.send_action(PrinterAction::Cancel);
                                    self.confirming_cancel = false;
                                }
                                if ui.button("No").clicked() {
                                    self.confirming_cancel = false;
                                }
                            }
                        }

                        if ui
                            .button(RichText::new("⛔ Emergency Stop").color(Color32::RED))
                            .clicked()
                        {
                            let _ = session.send_action(PrinterAction::EmergencyStop);
                        }
                    });
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
