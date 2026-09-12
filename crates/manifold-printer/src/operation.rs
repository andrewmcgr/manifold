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
