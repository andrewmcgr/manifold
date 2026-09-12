use crate::model::UploadOutcome;

#[derive(Debug)]
pub enum PrinterAction {
    UploadAndPrint { filename: String, gcode: Vec<u8> },
    UploadOnly { filename: String, gcode: Vec<u8> },
    Pause,
    Resume,
    Cancel,
    EmergencyStop,
    Disconnect,
    Reconnect,
}
impl PrinterAction {
    pub fn name(&self) -> &'static str {
        match self {
            Self::UploadAndPrint { .. } => "upload & print",
            Self::UploadOnly { .. } => "upload",
            Self::Pause => "pause",
            Self::Resume => "resume",
            Self::Cancel => "cancel",
            Self::EmergencyStop => "emergency stop",
            Self::Disconnect => "disconnect",
            Self::Reconnect => "reconnect",
        }
    }
    pub(crate) fn is_upload(&self) -> bool {
        matches!(self, Self::UploadAndPrint { .. } | Self::UploadOnly { .. })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionOutcome {
    Acknowledged,
    Upload(UploadOutcome),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationState {
    Pending,
    Running {
        bytes_read: u64,
        total_bytes: u64,
    },
    Succeeded(ActionOutcome),
    Failed {
        message: String,
        outcome_unknown: bool,
    },
}
impl OperationState {
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending | Self::Running { .. })
    }
}
#[derive(Debug, Clone)]
pub struct Operation {
    pub id: u64,
    pub name: String,
    pub state: OperationState,
    pub(crate) upload: bool,
}

/// Bounded, deliberately acknowledged evidence when detailed results are compacted.
/// Counts saturate instead of wrapping; unknown outcomes are a subset of failures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResultSummary {
    pub succeeded: u64,
    pub failed: u64,
    pub outcome_unknown: u64,
}
impl ResultSummary {
    pub fn include(&mut self, state: &OperationState) {
        match state {
            OperationState::Succeeded(_) => self.succeeded = self.succeeded.saturating_add(1),
            OperationState::Failed {
                outcome_unknown, ..
            } => {
                self.failed = self.failed.saturating_add(1);
                if *outcome_unknown {
                    self.outcome_unknown = self.outcome_unknown.saturating_add(1);
                }
            }
            _ => {}
        }
    }
    pub fn merge(&mut self, other: &Self) {
        self.succeeded = self.succeeded.saturating_add(other.succeeded);
        self.failed = self.failed.saturating_add(other.failed);
        self.outcome_unknown = self.outcome_unknown.saturating_add(other.outcome_unknown);
    }
    pub fn is_empty(&self) -> bool {
        self.succeeded == 0 && self.failed == 0
    }
}
