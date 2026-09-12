//! Printer intent and job monitoring, separate from mesh loading and slicing.
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use manifold_printer::{
    ConnectionState, MoonrakerClient, MoonrakerConfig, PrintState, PrinterSessionHandle,
    PrinterTelemetry, UploadDisposition, UploadOutcome,
};

use crate::Cli;

#[derive(Debug, PartialEq, Eq)]
pub enum Intent {
    Slice,
    Upload { start: bool, monitor: bool },
    Monitor,
}

pub fn intent(cli: &Cli) -> Result<Intent> {
    if (cli.upload || cli.print || cli.monitor || cli.printer_api_key.is_some())
        && cli.printer_url.is_none()
    {
        bail!("printer actions and API keys require --printer-url");
    }
    if cli.upload && cli.monitor && !cli.print {
        bail!(
            "--upload --monitor requires --print; use --monitor alone to attach without uploading"
        );
    }
    if let Some(url) = &cli.printer_url {
        MoonrakerClient::new(MoonrakerConfig {
            url: url.clone(),
            api_key: cli.printer_api_key.clone(),
            auto_connect: true,
        })?;
    }
    if cli.monitor && !cli.upload && !cli.print {
        return Ok(Intent::Monitor);
    }
    if cli.inputs.is_empty() {
        bail!(
            "provide mesh input(s), or --monitor with --printer-url to attach to an existing job"
        );
    }
    if cli.upload || cli.print {
        Ok(Intent::Upload {
            start: cli.print,
            monitor: cli.monitor,
        })
    } else {
        Ok(Intent::Slice)
    }
}
pub fn config(cli: &Cli) -> MoonrakerConfig {
    MoonrakerConfig {
        url: cli.printer_url.clone().expect("validated printer intent"),
        api_key: cli.printer_api_key.clone(),
        auto_connect: true,
    }
}
pub fn require_started(outcome: &UploadOutcome) -> Result<()> {
    if outcome.disposition != UploadDisposition::Started {
        bail!("Uploaded {}/{}: {:?}; immediate printing was not confirmed. No additional start was sent. Inspect the printer before retrying.",outcome.root,outcome.path,outcome.disposition);
    }
    Ok(())
}
const WAIT_LIMIT: Duration = Duration::from_secs(30);

/// One signal listener spans readiness, guards, transmission and monitoring. Dropping
/// a Tokio signal future does not restore default SIGINT handling, so inner stages
/// must not install and then abandon their own listeners.
pub async fn interruptible(
    work: impl Future<Output = Result<()>>,
    mutation_in_flight: &AtomicBool,
) -> Result<()> {
    tokio::select! {
        biased;
        signal = tokio::signal::ctrl_c() => {
            signal?;
            if mutation_in_flight.load(Ordering::SeqCst) {
                bail!("Interrupted upload: remote outcome unknown after possible transmission. Inspect the printer before retrying; no compensating cancel/start was sent");
            }
            bail!("Interrupted; work pending before mutation transmission was discarded. Any active print was NOT cancelled");
        }
        result = work => result,
    }
}

pub async fn wait_ready(session: &PrinterSessionHandle) -> Result<()> {
    let began = Instant::now();
    loop {
        let t = session.latest_telemetry();
        if let ConnectionState::Error(message) = t.connection_state {
            bail!("Printer connection failed: {message}");
        }
        if t.fresh && t.klippy_ready {
            return Ok(());
        }
        if began.elapsed() > WAIT_LIMIT {
            bail!("Timed out waiting for authenticated, ready printer telemetry; check endpoint, API key and Klipper state");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

struct Monitor {
    target: Option<String>,
    observed_active: bool,
    began: Instant,
    last_fresh: Instant,
    observed_job: Option<(u64, u64)>,
    subscription_generation: Option<u64>,
    prior_job: Option<u64>,
}
impl Monitor {
    fn new(target: Option<String>, now: Instant) -> Self {
        Self {
            target,
            observed_active: false,
            began: now,
            last_fresh: now,
            observed_job: None,
            subscription_generation: None,
            prior_job: None,
        }
    }
    /// Completion requires a matching active job observation, not old terminal telemetry.
    fn observe(&mut self, t: &PrinterTelemetry, now: Instant) -> Result<bool> {
        if let ConnectionState::Error(message) = &t.connection_state {
            bail!("Printer connection failed: {message}");
        }
        if !t.fresh {
            if now.duration_since(self.last_fresh) >= WAIT_LIMIT {
                bail!("Printer telemetry disconnected for 30 seconds; remote print may still be running");
            }
            return Ok(false);
        }
        self.last_fresh = now;
        if self
            .subscription_generation
            .is_some_and(|generation| generation != t.generation)
        {
            bail!("Printer subscription changed; monitored run continuity is uncertain. Inspect the printer; no completion was assumed");
        }
        self.subscription_generation = Some(t.generation);
        // Validate identity before every decision, including terminal snapshots which the
        // polling loop may see without observing the intervening run's active transition.
        if let Some((_, job)) = self.observed_job {
            if job != t.job_generation {
                bail!("The monitored job was replaced by another run; completion not confirmed");
            }
        }
        if self.target.is_none() && t.print_state.is_active() {
            self.target = t.filename.clone();
        }
        let matches = self.target.is_some() && self.target == t.filename;
        if matches && t.print_state.is_active() && self.prior_job != Some(t.job_generation) {
            self.observed_job = Some((t.generation, t.job_generation));
            self.observed_active = true;
        }
        if self.observed_active && !matches {
            bail!("Monitored file changed or was cleared; refusing to monitor another job");
        }
        if matches && self.observed_active {
            match t.print_state {
                PrintState::Complete if self.observed_active => return Ok(true),
                PrintState::Cancelled => bail!(
                    "Print cancelled: {}",
                    self.target.as_deref().unwrap_or("unknown")
                ),
                PrintState::Error => bail!(
                    "Print failed: {}",
                    t.klipper_message
                        .as_deref()
                        .unwrap_or("Klipper reported an error")
                ),
                PrintState::Unknown | PrintState::Standby if self.observed_active => bail!(
                    "Monitored job became {:?}; completion not confirmed",
                    t.print_state
                ),
                _ => {}
            }
        }
        if !self.observed_active && now.duration_since(self.began) >= WAIT_LIMIT {
            bail!("Timed out waiting to observe the requested active job {}; no completion was assumed",self.target.as_deref().unwrap_or("(attach to existing job)"));
        }
        Ok(false)
    }
}

pub async fn monitor(
    session: &PrinterSessionHandle,
    target: Option<(String, PrinterTelemetry)>,
) -> Result<()> {
    let mut tracker = Monitor::new(
        target.as_ref().map(|(name, _)| name.clone()),
        Instant::now(),
    );
    if let Some((_, before_start)) = target {
        tracker.subscription_generation = Some(before_start.generation);
        tracker.prior_job = Some(before_start.job_generation);
    }
    let pb = indicatif::ProgressBar::new(100);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner} [{bar:30.cyan/blue}] {pos}% {msg}")?,
    );
    loop {
        let t = session.latest_telemetry();
        let done = tracker.observe(&t, Instant::now())?;
        pb.set_position((t.progress_fraction * 100.0) as u64);
        let thermals = format!(
            "Nozzle: {} Bed: {}",
            t.hotend
                .as_ref()
                .map(|h| format!("{:.0}/{:.0}C", h.current, h.target))
                .unwrap_or_else(|| "?".into()),
            t.bed
                .as_ref()
                .map(|b| format!("{:.0}/{:.0}C", b.current, b.target))
                .unwrap_or_else(|| "?".into())
        );
        let message = format!(
            "{} — {:?} {} elapsed {}s, approx. remaining {} {}",
            tracker
                .target
                .as_deref()
                .unwrap_or("waiting for active filename"),
            t.print_state,
            thermals,
            t.print_duration_secs,
            t.estimated_remaining_secs
                .filter(|_| t.fresh)
                .map(|n| format!("{n}s"))
                .unwrap_or_else(|| "unknown".into()),
            if t.fresh { "" } else { "STALE" }
        );
        pb.set_message(message.clone());
        // Progress bars are hidden when redirected; preserve target and status in logs too.
        if pb.is_hidden() {
            eprintln!("{message}");
        }
        if done {
            pb.finish_with_message("Matching print completed");
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[test]
    fn intent_matrix_validates_before_loading_meshes() {
        for (args, want) in [
            (vec!["model.stl"], Some(Intent::Slice)),
            (
                vec!["model.stl", "--printer-url", "http://localhost"],
                Some(Intent::Slice),
            ),
            (
                vec!["--monitor", "--printer-url", "http://localhost"],
                Some(Intent::Monitor),
            ),
            (
                vec!["model.stl", "--upload", "--printer-url", "http://localhost"],
                Some(Intent::Upload {
                    start: false,
                    monitor: false,
                }),
            ),
            (
                vec![
                    "model.stl",
                    "--upload",
                    "--print",
                    "--monitor",
                    "--printer-url",
                    "http://localhost",
                ],
                Some(Intent::Upload {
                    start: true,
                    monitor: true,
                }),
            ),
            (
                vec![
                    "model.stl",
                    "--upload",
                    "--monitor",
                    "--printer-url",
                    "http://localhost",
                ],
                None,
            ),
            (vec!["--monitor"], None),
            (vec!["model.stl", "--print"], None),
            (vec![], None),
        ] {
            let cli = Cli::try_parse_from(std::iter::once("manifold").chain(args)).unwrap();
            assert_eq!(intent(&cli).ok(), want);
        }
    }
    fn telemetry(filename: &str, state: PrintState) -> PrinterTelemetry {
        PrinterTelemetry {
            fresh: true,
            connection_state: ConnectionState::Connected,
            filename: Some(filename.into()),
            print_state: state,
            ..Default::default()
        }
    }
    #[test]
    fn monitor_requires_matching_active_observation_before_completion() {
        let now = Instant::now();
        let mut m = Monitor::new(Some("new.gcode".into()), now);
        assert!(!m
            .observe(&telemetry("old.gcode", PrintState::Complete), now)
            .unwrap());
        assert!(!m
            .observe(&telemetry("new.gcode", PrintState::Complete), now)
            .unwrap());
        assert!(!m
            .observe(&telemetry("new.gcode", PrintState::Printing), now)
            .unwrap());
        assert!(m
            .observe(&telemetry("new.gcode", PrintState::Complete), now)
            .unwrap());
    }
    #[test]
    fn monitor_attach_errors_cancellation_auth_and_timeouts_are_failures() {
        let now = Instant::now();
        let mut m = Monitor::new(None, now);
        m.observe(&telemetry("actual.gcode", PrintState::Paused), now)
            .unwrap();
        assert_eq!(m.target.as_deref(), Some("actual.gcode"));
        assert!(m
            .observe(&telemetry("actual.gcode", PrintState::Cancelled), now)
            .is_err());
        let mut failed = telemetry("actual.gcode", PrintState::Error);
        failed.klipper_message = Some("heater fault".into());
        assert!(m
            .observe(&failed, now)
            .unwrap_err()
            .to_string()
            .contains("heater fault"));
        failed.connection_state = ConnectionState::Error("unauthorized".into());
        assert!(m.observe(&failed, now).is_err());
        assert!(m
            .observe(&PrinterTelemetry::default(), now + WAIT_LIMIT)
            .is_err());
        let mut waiting = Monitor::new(Some("new.gcode".into()), now);
        assert!(waiting
            .observe(
                &telemetry("old.gcode", PrintState::Complete),
                now + WAIT_LIMIT
            )
            .is_err());
    }
    #[test]
    fn monitor_ignores_old_same_filename_terminals_until_new_active_run() {
        let now = Instant::now();
        for terminal in [
            PrintState::Cancelled,
            PrintState::Error,
            PrintState::Complete,
        ] {
            let mut monitor = Monitor::new(Some("same.gcode".into()), now);
            let old = telemetry("same.gcode", terminal);
            assert!(!monitor.observe(&old, now).unwrap());
            let mut active = telemetry("same.gcode", PrintState::Printing);
            active.job_generation = 1;
            assert!(!monitor.observe(&active, now).unwrap());
            active.print_state = PrintState::Complete;
            assert!(monitor.observe(&active, now).unwrap());
        }
    }
    #[test]
    fn monitor_validates_identity_before_every_terminal_decision() {
        let now = Instant::now();
        for terminal in [
            PrintState::Complete,
            PrintState::Cancelled,
            PrintState::Error,
        ] {
            for reconnect in [false, true] {
                let mut monitor = Monitor::new(Some("same.gcode".into()), now);
                let mut t = telemetry("same.gcode", PrintState::Printing);
                monitor.observe(&t, now).unwrap();
                if reconnect {
                    t.generation += 1;
                } else {
                    t.job_generation += 1;
                }
                t.print_state = terminal;
                let error = monitor.observe(&t, now).unwrap_err().to_string();
                assert!(
                    error.contains(if reconnect { "continuity" } else { "replaced" }),
                    "{error}"
                );
            }
        }
    }
    #[test]
    fn monitor_does_not_attach_after_unobserved_reconnect() {
        let now = Instant::now();
        let mut monitor = Monitor::new(Some("same.gcode".into()), now);
        let mut t = telemetry("same.gcode", PrintState::Cancelled);
        assert!(!monitor.observe(&t, now).unwrap());
        t.generation += 1;
        t.print_state = PrintState::Printing;
        assert!(monitor
            .observe(&t, now)
            .unwrap_err()
            .to_string()
            .contains("continuity"));
    }
    #[test]
    fn started_monitor_requires_new_job_in_pre_upload_subscription() {
        let now = Instant::now();
        let mut monitor = Monitor::new(Some("same.gcode".into()), now);
        monitor.prior_job = Some(4);
        monitor.subscription_generation = Some(2);
        let mut t = telemetry("same.gcode", PrintState::Printing);
        t.generation = 2;
        t.job_generation = 4;
        assert!(!monitor.observe(&t, now).unwrap());
        assert!(
            !monitor.observed_active,
            "old active telemetry cannot bind the requested start"
        );
        t.job_generation = 5;
        assert!(!monitor.observe(&t, now).unwrap());
        assert!(monitor.observed_active);
        let mut ambiguous = Monitor::new(Some("same.gcode".into()), now);
        ambiguous.subscription_generation = Some(1);
        assert!(ambiguous
            .observe(&t, now)
            .unwrap_err()
            .to_string()
            .contains("continuity"));
    }
    #[test]
    fn queued_or_unconfirmed_upload_does_not_claim_print_started() {
        for disposition in [
            UploadDisposition::Uploaded,
            UploadDisposition::Queued,
            UploadDisposition::StartNotConfirmed,
        ] {
            assert!(require_started(&UploadOutcome {
                path: "canonical.gcode".into(),
                root: "gcodes".into(),
                disposition
            })
            .is_err());
        }
    }
}
