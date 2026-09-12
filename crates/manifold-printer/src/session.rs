use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, watch};

use crate::client::MoonrakerClient;
use crate::error::MoonrakerError;
use crate::model::{ConnectionState, MoonrakerConfig, PrinterTelemetry, UploadDisposition};
pub use crate::operation::PrinterAction;
use crate::operation::{ActionOutcome, Operation, OperationState};

const HISTORY_LIMIT: usize = 64;

pub(crate) struct Shared {
    pub telemetry: PrinterTelemetry,
    pub epoch: u64,
    online: bool,
    operations: VecDeque<Operation>,
    next_id: u64,
    pending_start: Option<String>,
}
impl Shared {
    pub(crate) fn reconcile_start(&mut self) {
        if self.telemetry.fresh
            && self.telemetry.print_state.is_active()
            && self.pending_start == self.telemetry.filename
        {
            self.pending_start = None;
        }
    }
    fn invalidate(&mut self) {
        self.epoch += 1;
        self.pending_start = None;
        self.telemetry.connection_state = if self.online {
            ConnectionState::Connecting
        } else {
            ConnectionState::Disconnected
        };
        self.telemetry.fresh = false;
        self.telemetry.klippy_ready = false;
        self.telemetry.estimated_remaining_secs = None;
        for op in &mut self.operations {
            if op.state.is_pending() {
                let outcome_unknown = matches!(op.state, OperationState::Running { .. });
                op.state = OperationState::Failed {
                    outcome_unknown,
                    message: if outcome_unknown {
                        "Session work interrupted; remote outcome unknown. Disconnect does not cancel a print. Check the printer before retrying.".into()
                    } else {
                        "Unsent action rejected: session changed".into()
                    },
                };
            }
        }
    }
    fn update(&mut self, id: u64, state: OperationState) {
        if let Some(op) = self
            .operations
            .iter_mut()
            .find(|o| o.id == id && o.state.is_pending())
        {
            op.state = state;
        }
    }
}
struct Command {
    id: u64,
    epoch: u64,
    action: PrinterAction,
}
struct Owner {
    endpoint: String,
    normal: mpsc::Sender<Command>,
    emergency: mpsc::Sender<Command>,
    lifecycle: watch::Sender<u64>,
    shared: Arc<Mutex<Shared>>,
}
impl Drop for Owner {
    fn drop(&mut self) {
        let mut shared = self.shared.lock().unwrap();
        shared.invalidate();
        shared.online = false;
        shared.telemetry.connection_state = ConnectionState::Disconnected;
        self.lifecycle.send_replace(shared.epoch);
        // No join on the UI thread. Closing lifecycle cancels every owned future.
    }
}
#[derive(Clone)]
pub struct PrinterSessionHandle {
    owner: Arc<Owner>,
}

impl PrinterSessionHandle {
    pub fn spawn(config: MoonrakerConfig) -> Self {
        let (normal, normal_rx) = mpsc::channel(8);
        let (emergency, emergency_rx) = mpsc::channel(1);
        let (lifecycle, lifecycle_rx) = watch::channel(0);
        let shared = Arc::new(Mutex::new(Shared {
            telemetry: PrinterTelemetry::default(),
            epoch: 0,
            online: config.auto_connect,
            operations: VecDeque::new(),
            next_id: 1,
            pending_start: None,
        }));
        let client = MoonrakerClient::new(config.clone());
        let endpoint = client
            .as_ref()
            .map(|c| c.config().url.clone())
            .unwrap_or(config.url);
        let worker_shared = shared.clone();
        let spawn = std::thread::Builder::new()
            .name("manifold-printer-worker".into())
            .spawn(move || {
                let result = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();
                match (result, client) {
                    (Ok(rt), Ok(client)) => rt.block_on(run_worker(
                        client,
                        normal_rx,
                        emergency_rx,
                        lifecycle_rx,
                        worker_shared,
                    )),
                    (rt, client) => {
                        let message = match (rt, client) {
                            (Err(e), _) => e.to_string(),
                            (_, Err(e)) => e.to_string(),
                            _ => unreachable!(),
                        };
                        worker_shared.lock().unwrap().telemetry.connection_state =
                            ConnectionState::Error(message);
                    }
                }
            });
        if let Err(e) = spawn {
            shared.lock().unwrap().telemetry.connection_state =
                ConnectionState::Error(e.to_string());
        }
        Self {
            owner: Arc::new(Owner {
                endpoint,
                normal,
                emergency,
                lifecycle,
                shared,
            }),
        }
    }
    pub fn endpoint(&self) -> &str {
        &self.owner.endpoint
    }
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.owner) as usize
    }
    pub fn latest_telemetry(&self) -> PrinterTelemetry {
        self.owner.shared.lock().unwrap().telemetry.clone()
    }
    pub fn operations(&self) -> Vec<Operation> {
        self.owner
            .shared
            .lock()
            .unwrap()
            .operations
            .iter()
            .cloned()
            .collect()
    }
    pub fn acknowledge(&self, id: u64) {
        self.owner
            .shared
            .lock()
            .unwrap()
            .operations
            .retain(|op| op.id != id || op.state.is_pending());
    }
    pub fn is_uploading(&self) -> bool {
        let shared = self.owner.shared.lock().unwrap();
        shared.pending_start.is_some()
            || shared
                .operations
                .iter()
                .any(|op| op.upload && op.state.is_pending())
    }
    pub fn is_start_pending(&self) -> bool {
        self.owner.shared.lock().unwrap().pending_start.is_some()
    }
    pub fn send_action(&self, action: PrinterAction) -> Result<u64, MoonrakerError> {
        let mut shared = self.owner.shared.lock().unwrap();
        if matches!(action, PrinterAction::Disconnect | PrinterAction::Reconnect) {
            shared.invalidate();
            shared.online = matches!(action, PrinterAction::Reconnect);
            shared.telemetry.connection_state = if shared.online {
                ConnectionState::Connecting
            } else {
                ConnectionState::Disconnected
            };
            self.owner.lifecycle.send_replace(shared.epoch);
            return Ok(0);
        }
        let stop = matches!(action, PrinterAction::EmergencyStop);
        if !stop
            && (!shared.online
                || !shared.telemetry.fresh
                || shared.telemetry.connection_state != ConnectionState::Connected)
        {
            return Err(MoonrakerError::rejected(
                action.name(),
                "printer subscription unavailable; reconnect and try explicitly",
            ));
        }
        if action.is_upload()
            && (shared.pending_start.is_some()
                || shared
                    .operations
                    .iter()
                    .any(|o| o.upload && o.state.is_pending()))
        {
            return Err(MoonrakerError::rejected(
                action.name(),
                "another upload/start is pending",
            ));
        }
        if stop
            && shared
                .operations
                .iter()
                .any(|o| o.name == "emergency stop" && o.state.is_pending())
        {
            return Err(MoonrakerError::rejected(
                action.name(),
                "emergency stop already pending",
            ));
        }
        // Reserve one history slot for emergency stop, even when normal results need acknowledgment.
        let limit = if stop {
            HISTORY_LIMIT
        } else {
            HISTORY_LIMIT - 1
        };
        while shared.operations.len() >= limit {
            if let Some(index) = shared
                .operations
                .iter()
                .position(|o| matches!(o.state, OperationState::Succeeded(_)))
            {
                shared.operations.remove(index);
            } else {
                return Err(MoonrakerError::rejected(
                    action.name(),
                    "acknowledge completed errors before submitting more actions",
                ));
            }
        }
        let tx = if stop {
            &self.owner.emergency
        } else {
            &self.owner.normal
        };
        let permit = tx.try_reserve().map_err(|_| {
            MoonrakerError::rejected(action.name(), "command queue full or worker stopped")
        })?;
        if stop {
            shared.invalidate();
            self.owner.lifecycle.send_replace(shared.epoch);
        }
        if let PrinterAction::UploadAndPrint { filename, .. } = &action {
            shared.pending_start = Some(filename.clone());
        }
        let id = shared.next_id;
        shared.next_id += 1;
        shared.operations.push_back(Operation {
            id,
            name: action.name().into(),
            upload: action.is_upload(),
            state: OperationState::Pending,
        });
        permit.send(Command {
            id,
            epoch: shared.epoch,
            action,
        });
        Ok(id)
    }
}

async fn run_worker(
    client: MoonrakerClient,
    mut normal: mpsc::Receiver<Command>,
    mut emergency: mpsc::Receiver<Command>,
    mut lifecycle: watch::Receiver<u64>,
    shared: Arc<Mutex<Shared>>,
) {
    loop {
        let epoch = *lifecycle.borrow_and_update();
        let online = shared.lock().unwrap().online;
        tokio::select! {
            biased;
            changed = lifecycle.changed() => { if changed.is_err() { break; } }
            _ = actions(&client, &mut emergency, &shared, epoch, true) => break,
            _ = actions(&client, &mut normal, &shared, epoch, false) => break,
            _ = async {
                if online { crate::transport::run(&client, &shared, epoch).await; }
                std::future::pending::<()>().await;
            } => {}
        }
    }
}

async fn actions(
    client: &MoonrakerClient,
    rx: &mut mpsc::Receiver<Command>,
    shared: &Arc<Mutex<Shared>>,
    epoch: u64,
    emergency: bool,
) {
    while let Some(command) = rx.recv().await {
        {
            let mut state = shared.lock().unwrap();
            if command.epoch != epoch || state.epoch != epoch {
                continue;
            }
            if !emergency && !state.telemetry.fresh {
                state.update(
                    command.id,
                    OperationState::Failed {
                        message: "Unsent action rejected: subscription lost".into(),
                        outcome_unknown: false,
                    },
                );
                continue;
            }
            state.update(
                command.id,
                OperationState::Running {
                    bytes_read: 0,
                    total_bytes: 0,
                },
            );
        }
        let progress_state = shared.clone();
        let id = command.id;
        let progress = Arc::new(move |bytes_read, total_bytes| {
            progress_state.lock().unwrap().update(
                id,
                OperationState::Running {
                    bytes_read,
                    total_bytes,
                },
            );
        });
        let starts = matches!(command.action, PrinterAction::UploadAndPrint { .. });
        let result = match command.action {
            PrinterAction::UploadOnly { filename, gcode } => client
                .upload_gcode_with_progress(&filename, gcode, false, Some(progress))
                .await
                .map(ActionOutcome::Upload),
            PrinterAction::UploadAndPrint { filename, gcode } => client
                .upload_gcode_with_progress(&filename, gcode, true, Some(progress))
                .await
                .map(ActionOutcome::Upload),
            PrinterAction::Pause => client
                .pause_print()
                .await
                .map(|_| ActionOutcome::Acknowledged),
            PrinterAction::Resume => client
                .resume_print()
                .await
                .map(|_| ActionOutcome::Acknowledged),
            PrinterAction::Cancel => client
                .cancel_print()
                .await
                .map(|_| ActionOutcome::Acknowledged),
            PrinterAction::EmergencyStop => client
                .emergency_stop()
                .await
                .map(|_| ActionOutcome::Acknowledged),
            _ => unreachable!("lifecycle actions never queued"),
        };
        if starts {
            let mut shared = shared.lock().unwrap();
            if shared.epoch == epoch {
                match &result {
                    Ok(ActionOutcome::Upload(outcome))
                        if matches!(
                            outcome.disposition,
                            UploadDisposition::Started | UploadDisposition::Queued
                        ) =>
                    {
                        shared.pending_start = Some(outcome.path.clone());
                        shared.reconcile_start();
                    }
                    Err(error) if error.outcome_unknown() => {}
                    _ => shared.pending_start = None,
                }
            }
        }
        let state = match result {
            Ok(result) => OperationState::Succeeded(result),
            Err(e) => OperationState::Failed {
                outcome_unknown: e.outcome_unknown(),
                message: client.redact(&e.to_string()),
            },
        };
        shared.lock().unwrap().update(id, state);
    }
}
