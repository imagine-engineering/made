use crate::{
    GeneratedFrontendExtensionDisposition, GeneratedFrontendExtensionDispositionKind,
    LegacyExtensionPlaceholder,
};
use gpui::{App, Global};
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

pub const MAX_PLUGIN_CONTRIBUTIONS: usize = 1_024;
pub const MAX_PLUGIN_CONTRIBUTION_SCHEMA_BYTES: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PluginContributionSurface {
    Command,
    Keybinding,
    Menu,
    Setting,
    BottomPanel,
    NodePanel,
    AboutBadge,
    TopbarBadge,
    ActionBarButton,
    NodeWidget,
    SelectionToolbox,
    CanvasMenu,
    NodeMenu,
}

impl PluginContributionSurface {
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "command" => Self::Command,
            "keybinding" => Self::Keybinding,
            "menu" => Self::Menu,
            "setting" => Self::Setting,
            "bottom-panel" => Self::BottomPanel,
            "node-panel" => Self::NodePanel,
            "about-badge" => Self::AboutBadge,
            "topbar-badge" => Self::TopbarBadge,
            "action-bar-button" => Self::ActionBarButton,
            "node-widget" => Self::NodeWidget,
            "selection-toolbox" => Self::SelectionToolbox,
            "canvas-menu" => Self::CanvasMenu,
            "node-menu" => Self::NodeMenu,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::Keybinding => "keybinding",
            Self::Menu => "menu",
            Self::Setting => "setting",
            Self::BottomPanel => "bottom-panel",
            Self::NodePanel => "node-panel",
            Self::AboutBadge => "about-badge",
            Self::TopbarBadge => "topbar-badge",
            Self::ActionBarButton => "action-bar-button",
            Self::NodeWidget => "node-widget",
            Self::SelectionToolbox => "selection-toolbox",
            Self::CanvasMenu => "canvas-menu",
            Self::NodeMenu => "node-menu",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginContributionInput {
    plugin_identifier: Arc<str>,
    manifest_digest_sha256: Arc<str>,
    contribution_id: Arc<str>,
    surface: Arc<str>,
    state_schema: Arc<str>,
}

impl PluginContributionInput {
    pub fn from_verified_manifest(
        plugin_identifier: impl Into<Arc<str>>,
        manifest_digest_sha256: impl Into<Arc<str>>,
        contribution_id: impl Into<Arc<str>>,
        surface: impl Into<Arc<str>>,
        state_schema: impl Into<Arc<str>>,
    ) -> Result<Self, PluginContributionError> {
        let input = Self {
            plugin_identifier: plugin_identifier.into(),
            manifest_digest_sha256: manifest_digest_sha256.into(),
            contribution_id: contribution_id.into(),
            surface: surface.into(),
            state_schema: state_schema.into(),
        };
        if !bounded_projection_identifier(&input.plugin_identifier)
            || !bounded_projection_identifier(&input.contribution_id)
            || input.surface.is_empty()
            || input.surface.len() > 256
            || input.surface.chars().any(char::is_control)
        {
            return Err(PluginContributionError::InvalidVerifiedProjection);
        }
        if input.manifest_digest_sha256.len() != 64
            || !input
                .manifest_digest_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(PluginContributionError::InvalidVerifiedProjection);
        }
        if input.state_schema.len() > MAX_PLUGIN_CONTRIBUTION_SCHEMA_BYTES {
            return Err(PluginContributionError::SchemaTooLarge {
                actual: input.state_schema.len(),
                maximum: MAX_PLUGIN_CONTRIBUTION_SCHEMA_BYTES,
            });
        }
        Ok(input)
    }

    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn manifest_digest_sha256(&self) -> &str {
        &self.manifest_digest_sha256
    }

    pub fn contribution_id(&self) -> &str {
        &self.contribution_id
    }

    pub fn surface(&self) -> &str {
        &self.surface
    }

    pub fn state_schema(&self) -> &str {
        &self.state_schema
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeclarativePluginContribution {
    input: PluginContributionInput,
    surface: PluginContributionSurface,
    parsed_state_schema: Value,
}

impl DeclarativePluginContribution {
    pub fn input(&self) -> &PluginContributionInput {
        &self.input
    }

    pub fn surface(&self) -> PluginContributionSurface {
        self.surface
    }

    pub fn parsed_state_schema(&self) -> &Value {
        &self.parsed_state_schema
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PluginContributionDiagnostic {
    plugin_identifier: Arc<str>,
    contribution_id: Arc<str>,
    message: Arc<str>,
}

impl PluginContributionDiagnostic {
    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn contribution_id(&self) -> &str {
        &self.contribution_id
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PluginContributionSnapshot {
    declarative: Vec<DeclarativePluginContribution>,
    placeholders: Vec<LegacyExtensionPlaceholder>,
    diagnostics: Vec<PluginContributionDiagnostic>,
    source_error: Option<Arc<str>>,
}

impl PluginContributionSnapshot {
    pub fn from_verified_inputs(
        inputs: Vec<PluginContributionInput>,
    ) -> Result<Self, PluginContributionError> {
        if inputs.len() > MAX_PLUGIN_CONTRIBUTIONS {
            return Err(PluginContributionError::TooManyContributions {
                actual: inputs.len(),
                maximum: MAX_PLUGIN_CONTRIBUTIONS,
            });
        }
        let identity_counts = inputs.iter().fold(BTreeMap::new(), |mut counts, input| {
            *counts
                .entry((
                    input.plugin_identifier.clone(),
                    input.contribution_id.clone(),
                ))
                .or_insert(0usize) += 1;
            counts
        });
        let mut snapshot = Self::default();
        for input in inputs {
            let identity = (
                input.plugin_identifier.clone(),
                input.contribution_id.clone(),
            );
            if identity_counts.get(&identity).copied().unwrap_or_default() > 1 {
                snapshot.push_placeholder(
                    &input,
                    "duplicate signed declarative contribution identity",
                )?;
                continue;
            }
            let Some(surface) = PluginContributionSurface::parse(input.surface()) else {
                snapshot
                    .push_placeholder(&input, "unsupported declarative contribution surface")?;
                continue;
            };
            let parsed_state_schema = match serde_json::from_str::<Value>(input.state_schema()) {
                Ok(Value::Object(schema)) => Value::Object(schema),
                Ok(_) => {
                    snapshot.push_placeholder(
                        &input,
                        "declarative contribution schema must be a JSON object",
                    )?;
                    continue;
                }
                Err(_) => {
                    snapshot.push_placeholder(
                        &input,
                        "declarative contribution schema is malformed JSON",
                    )?;
                    continue;
                }
            };
            snapshot.declarative.push(DeclarativePluginContribution {
                input,
                surface,
                parsed_state_schema,
            });
        }
        snapshot.declarative.sort_by(|left, right| {
            (left.input.plugin_identifier(), left.input.contribution_id()).cmp(&(
                right.input.plugin_identifier(),
                right.input.contribution_id(),
            ))
        });
        Ok(snapshot)
    }

    fn push_placeholder(
        &mut self,
        input: &PluginContributionInput,
        message: &'static str,
    ) -> Result<(), PluginContributionError> {
        self.placeholders.push(LegacyExtensionPlaceholder::new(
            input.plugin_identifier.clone(),
            input.contribution_id.clone(),
            Arc::<[u8]>::from(input.state_schema.as_bytes()),
            message,
            [],
        )?);
        self.diagnostics.push(PluginContributionDiagnostic {
            plugin_identifier: input.plugin_identifier.clone(),
            contribution_id: input.contribution_id.clone(),
            message: Arc::from(message),
        });
        Ok(())
    }

    pub fn declarative(&self) -> &[DeclarativePluginContribution] {
        &self.declarative
    }

    pub fn placeholders(&self) -> &[LegacyExtensionPlaceholder] {
        &self.placeholders
    }

    pub fn diagnostics(&self) -> &[PluginContributionDiagnostic] {
        &self.diagnostics
    }

    pub fn source_error(&self) -> Option<&str> {
        self.source_error.as_deref()
    }

    fn with_source_error(message: String) -> Self {
        Self {
            source_error: Some(message.into()),
            ..Self::default()
        }
    }
}

pub trait PluginContributionSource: Send + Sync {
    fn verified_contributions(&self) -> anyhow::Result<Vec<PluginContributionInput>>;
}

struct GlobalPluginContributionSource(Arc<dyn PluginContributionSource>);

impl Global for GlobalPluginContributionSource {}

pub fn register_plugin_contribution_source(
    source: Arc<dyn PluginContributionSource>,
    cx: &mut App,
) {
    cx.set_global(GlobalPluginContributionSource(source));
}

pub fn plugin_contribution_snapshot(cx: &App) -> PluginContributionSnapshot {
    let Some(source) = cx.try_global::<GlobalPluginContributionSource>() else {
        return PluginContributionSnapshot::default();
    };
    let inputs = match source.0.verified_contributions() {
        Ok(inputs) => inputs,
        Err(error) => {
            return PluginContributionSnapshot::with_source_error(format!(
                "verified component inventory unavailable: {error}"
            ));
        }
    };
    match PluginContributionSnapshot::from_verified_inputs(inputs) {
        Ok(snapshot) => snapshot,
        Err(error) => PluginContributionSnapshot::with_source_error(format!(
            "verified contribution projection rejected: {error}"
        )),
    }
}

pub fn frontend_extension_dispositions() -> &'static [GeneratedFrontendExtensionDisposition] {
    crate::GENERATED_FRONTEND_EXTENSION_DISPOSITIONS
}

pub fn frontend_extension_disposition(
    feature_id: &str,
) -> Option<&'static GeneratedFrontendExtensionDisposition> {
    crate::GENERATED_FRONTEND_EXTENSION_DISPOSITIONS
        .iter()
        .find(|disposition| disposition.feature_id == feature_id)
}

pub fn declarative_legacy_hook_count() -> usize {
    crate::GENERATED_FRONTEND_EXTENSION_DISPOSITIONS
        .iter()
        .filter(|disposition| {
            disposition.classification
                == GeneratedFrontendExtensionDispositionKind::DeclarativeRustWasm
        })
        .count()
}

fn bounded_projection_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && !value.chars().any(char::is_control)
}

#[derive(Debug, thiserror::Error)]
pub enum PluginContributionError {
    #[error("verified plugin contribution projection is malformed")]
    InvalidVerifiedProjection,
    #[error("plugin contribution schema has {actual} bytes; maximum is {maximum}")]
    SchemaTooLarge { actual: usize, maximum: usize },
    #[error("plugin contribution inventory has {actual} entries; maximum is {maximum}")]
    TooManyContributions { actual: usize, maximum: usize },
    #[error(transparent)]
    Placeholder(#[from] crate::LegacyExtensionPlaceholderError),
}
