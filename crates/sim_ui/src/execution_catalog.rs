use crate::generated_execution_catalog::GENERATED_EXECUTION_CATALOG;
use serde::Serialize;

pub const EXECUTION_UI_OWNER: &str = "comfy-parity-execution-ui";
pub const NATIVE_API_OWNER: &str = "comfy-parity-native-api-host";
pub const NATIVE_IMAGE_OWNER: &str = "comfy-parity-native-execution-e2e";
pub const NATIVE_MEMORY_OWNER: &str = "comfy-parity-native-memory-planner";
pub const WORKFLOW_FORMATS_OWNER: &str = "comfy-parity-workflow-formats";
pub const NATIVE_GRAPH_OWNER: &str = "comfy-parity-native-graph";
pub const WORKFLOW_EXPERIENCE_OWNER: &str = "comfy-parity-workflow-experience";
pub const ASSET_VIEWERS_OWNER: &str = "comfy-parity-assets-editors-viewers";
pub const SETTINGS_OWNER: &str = "comfy-parity-settings-localization-ui";
pub const DIAGNOSTICS_OWNER: &str = "comfy-parity-process-diagnostics";
pub const PERFORMANCE_OWNER: &str = "comfy-parity-performance";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "disposition", rename_all = "snake_case")]
pub enum ExecutionFeatureDisposition {
    Native,
    Foundation { owner: &'static str },
    LaterOwned { owner: &'static str },
    SharedClosure { later_owner: &'static str },
}

impl ExecutionFeatureDisposition {
    pub fn current_owner(self) -> Option<&'static str> {
        match self {
            Self::Native | Self::SharedClosure { .. } => Some(EXECUTION_UI_OWNER),
            Self::Foundation { owner } => Some(owner),
            Self::LaterOwned { .. } => None,
        }
    }

    pub fn closure_owner(self) -> &'static str {
        match self {
            Self::Native => EXECUTION_UI_OWNER,
            Self::Foundation { owner }
            | Self::LaterOwned { owner }
            | Self::SharedClosure { later_owner: owner } => owner,
        }
    }
}

pub fn execution_feature_disposition(feature_id: &str) -> Option<ExecutionFeatureDisposition> {
    GENERATED_EXECUTION_CATALOG
        .iter()
        .find(|row| row.feature_id == feature_id)
        .map(|row| row.disposition)
}
