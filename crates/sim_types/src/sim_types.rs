use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub mod cancellation;
pub mod json_compat;
pub mod protocol;
pub mod worker_protocol;
pub mod workflow;

pub use cancellation::*;
pub use json_compat::*;
pub use protocol::*;
pub use worker_protocol::*;
pub use workflow::*;

pub const NATIVE_PROTOCOL_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    DirectMl,
    Xpu,
    Npu,
    Mlu,
    CoreX,
}

impl DeviceKind {
    pub const ALL: [Self; 9] = [
        Self::Cpu,
        Self::Cuda,
        Self::Rocm,
        Self::Metal,
        Self::DirectMl,
        Self::Xpu,
        Self::Npu,
        Self::Mlu,
        Self::CoreX,
    ];
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExecutionId(Uuid);

impl ExecutionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for ExecutionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[error("native {device:?} backend is unavailable: {reason}")]
pub struct BackendUnavailable {
    device: DeviceKind,
    reason: String,
}

impl BackendUnavailable {
    pub fn new(device: DeviceKind, reason: impl Into<String>) -> Self {
        Self {
            device,
            reason: reason.into(),
        }
    }

    pub const fn device(&self) -> DeviceKind {
        self.device
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "binding")]
pub enum NativeBackendBindingStatus {
    Bound { device: DeviceKind },
    Unbound { device: DeviceKind, reason: String },
}

impl NativeBackendBindingStatus {
    pub const fn bound(device: DeviceKind) -> Self {
        Self::Bound { device }
    }

    pub fn unbound(device: DeviceKind, reason: impl Into<String>) -> Self {
        Self::Unbound {
            device,
            reason: reason.into(),
        }
    }

    pub const fn device(&self) -> DeviceKind {
        match self {
            Self::Bound { device } | Self::Unbound { device, .. } => *device,
        }
    }
}

pub trait NativeBackendBinding: Send + Sync {
    fn binding_status(&self) -> NativeBackendBindingStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_ids_are_distinct() {
        assert_ne!(ExecutionId::new(), ExecutionId::new());
    }

    #[test]
    fn binding_status_carries_no_semantic_support_claim() {
        for device in DeviceKind::ALL {
            let status = NativeBackendBindingStatus::unbound(device, "fixture binding missing");
            assert_eq!(status.device(), device);
            assert!(matches!(status, NativeBackendBindingStatus::Unbound { .. }));
        }
    }
}
