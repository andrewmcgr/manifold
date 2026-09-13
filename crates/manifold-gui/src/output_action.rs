//! Session-local output selection. Only the main button emits an action.

use egui::Ui;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OutputAction {
    #[default]
    Export,
    Upload,
    Print,
}

impl OutputAction {
    fn label(self) -> &'static str {
        match self {
            Self::Export => "Export…",
            Self::Upload => "Upload",
            Self::Print => "Print",
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Self::Export => "Export G-code to a local file",
            Self::Upload => "Upload G-code to the active printer without starting a print",
            Self::Print => "Upload G-code to the active printer and start printing",
        }
    }

    pub(crate) fn show(
        &mut self,
        ui: &mut Ui,
        can_export: bool,
        can_upload: bool,
        can_print: bool,
    ) -> Option<Self> {
        // A non-wrapping child keeps the main button and arrow together in the toolbar.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let enabled = match self {
                Self::Export => can_export,
                Self::Upload => can_upload,
                Self::Print => can_print,
            };
            let action = ui
                .add_enabled(enabled, egui::Button::new(self.label()))
                .on_hover_text(self.tooltip())
                .on_disabled_hover_text(self.tooltip())
                .clicked()
                .then_some(*self);
            // Selection stays available even when the selected action is disabled.
            ui.menu_button("▼", |ui| {
                for choice in [Self::Export, Self::Upload, Self::Print] {
                    if ui
                        .selectable_label(*self == choice, choice.label())
                        .on_hover_text(choice.tooltip())
                        .clicked()
                    {
                        *self = choice;
                        ui.close_menu();
                    }
                }
            })
            .response
            .on_hover_text("Choose output action (does not execute)");
            action
        })
        .inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Controls {
        ctx: egui::Context,
        selected: OutputAction,
        enabled: [bool; 3],
        actions: Vec<OutputAction>,
    }

    impl Controls {
        fn new(enabled: [bool; 3]) -> Self {
            Self {
                ctx: egui::Context::default(),
                selected: OutputAction::default(),
                enabled,
                actions: Vec::new(),
            }
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> egui::FullOutput {
            self.ctx.run(
                egui::RawInput {
                    events,
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(300.0, 200.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            if let Some(action) = self.selected.show(
                                ui,
                                self.enabled[0],
                                self.enabled[1],
                                self.enabled[2],
                            ) {
                                self.actions.push(action);
                            }
                        });
                    });
                },
            )
        }

        fn click(&mut self, label: &str) {
            // Menu areas need a sizing frame before painting.
            self.frame(vec![]);
            let output = self.frame(vec![]);
            let pos = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::epaint::Shape::Text(text) if text.galley.text() == label => {
                        Some(egui::Rect::from_min_size(text.pos, text.galley.size()).center())
                    }
                    _ => None,
                })
                .next_back()
                .unwrap_or_else(|| panic!("missing {label}"));
            for pressed in [true, false] {
                self.frame(vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    },
                ]);
            }
        }

        fn select(&mut self, action: OutputAction) {
            let count = self.actions.len();
            self.click("▼");
            self.click(action.label());
            assert_eq!(self.selected, action);
            assert_eq!(self.actions.len(), count, "selection must not execute");
        }
    }

    #[test]
    fn export_is_initial_and_main_click_emits_only_the_remembered_selection() {
        let mut controls = Controls::new([true; 3]);
        assert_eq!(controls.selected, OutputAction::Export);
        for action in [
            OutputAction::Export,
            OutputAction::Upload,
            OutputAction::Print,
            OutputAction::Export,
        ] {
            controls.select(action);
            for _ in 0..3 {
                controls.frame(vec![]);
                assert_eq!(controls.selected, action);
            }
            controls.click(action.label());
            assert_eq!(controls.actions, vec![action]);
            controls.actions.clear();
        }
    }

    #[test]
    fn disabled_matrix_never_executes_and_arrow_always_allows_switching() {
        // No G-code; offline/stale/uploading; connected but not startable; ready.
        for enabled in [
            [false, false, false],
            [true, false, false],
            [true, true, false],
            [true, true, true],
        ] {
            let mut controls = Controls::new(enabled);
            for (index, action) in [
                OutputAction::Export,
                OutputAction::Upload,
                OutputAction::Print,
            ]
            .into_iter()
            .enumerate()
            {
                controls.select(action);
                controls.click(action.label());
                assert_eq!(
                    controls.actions,
                    if enabled[index] { vec![action] } else { vec![] },
                    "{action:?}, availability {enabled:?}"
                );
                controls.actions.clear();
            }
            controls.select(OutputAction::Export);
            controls.click("Export…");
            assert_eq!(controls.actions.len(), usize::from(enabled[0]));
        }
    }

    #[test]
    fn main_and_arrow_stay_adjacent_even_in_a_narrow_wrapping_toolbar() {
        let ctx = egui::Context::default();
        let mut action = OutputAction::default();
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(80.0, 200.0),
                )),
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        assert_eq!(action.show(ui, true, false, false), None);
                    });
                });
            },
        );
        let rect = |label| {
            output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::epaint::Shape::Text(text) if text.galley.text() == label => {
                        Some(egui::Rect::from_min_size(text.pos, text.galley.size()))
                    }
                    _ => None,
                })
                .unwrap()
        };
        let main = rect("Export…");
        let arrow = rect("▼");
        assert!((main.center().y - arrow.center().y).abs() < 2.0);
        assert!((0.0..=15.0).contains(&(arrow.left() - main.right())));
    }

    #[test]
    fn losing_readiness_does_not_reset_selection_or_execute() {
        let mut controls = Controls::new([true; 3]);
        controls.select(OutputAction::Print);
        controls.enabled = [true, false, false];
        controls.click("Print");
        assert_eq!(controls.selected, OutputAction::Print);
        assert!(controls.actions.is_empty());
        controls.select(OutputAction::Export);
        controls.click("Export…");
        assert_eq!(controls.actions, vec![OutputAction::Export]);
    }
}
