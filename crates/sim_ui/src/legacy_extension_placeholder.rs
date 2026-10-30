use gpui::{App, IntoElement, RenderOnce, Role, SharedString, Window};
use std::sync::Arc;
use ui::prelude::*;

pub const MAX_LEGACY_EXTENSION_PAYLOAD_BYTES: usize = 1024 * 1024;
const MAX_LEGACY_EXTENSION_IDENTITY_BYTES: usize = 512;
const MAX_REPLACEMENT_CHOICES: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyExtensionReplacementChoice {
    identifier: Arc<str>,
    owner: Arc<str>,
    label: Arc<str>,
}

impl LegacyExtensionReplacementChoice {
    pub fn new(
        identifier: impl Into<Arc<str>>,
        owner: impl Into<Arc<str>>,
        label: impl Into<Arc<str>>,
    ) -> Result<Self, LegacyExtensionPlaceholderError> {
        let identifier = identifier.into();
        let owner = owner.into();
        let label = label.into();
        validate_identity("replacement identifier", &identifier)?;
        validate_identity("replacement owner", &owner)?;
        validate_display_text("replacement label", &label)?;
        Ok(Self {
            identifier,
            owner,
            label,
        })
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

#[derive(Clone, Debug, Eq, IntoElement, PartialEq)]
pub struct LegacyExtensionPlaceholder {
    extension_identity: Arc<str>,
    hook_identity: Arc<str>,
    exact_payload: Arc<[u8]>,
    diagnostic: Arc<str>,
    replacement_choices: Arc<[LegacyExtensionReplacementChoice]>,
}

impl LegacyExtensionPlaceholder {
    pub fn new(
        extension_identity: impl Into<Arc<str>>,
        hook_identity: impl Into<Arc<str>>,
        exact_payload: impl Into<Arc<[u8]>>,
        diagnostic: impl Into<Arc<str>>,
        replacement_choices: impl IntoIterator<Item = LegacyExtensionReplacementChoice>,
    ) -> Result<Self, LegacyExtensionPlaceholderError> {
        let extension_identity = extension_identity.into();
        let hook_identity = hook_identity.into();
        let exact_payload = exact_payload.into();
        let diagnostic = diagnostic.into();
        validate_identity("extension identity", &extension_identity)?;
        validate_identity("hook identity", &hook_identity)?;
        validate_display_text("diagnostic", &diagnostic)?;
        if exact_payload.len() > MAX_LEGACY_EXTENSION_PAYLOAD_BYTES {
            return Err(LegacyExtensionPlaceholderError::PayloadTooLarge {
                actual: exact_payload.len(),
                maximum: MAX_LEGACY_EXTENSION_PAYLOAD_BYTES,
            });
        }
        let replacement_choices = replacement_choices.into_iter().collect::<Vec<_>>();
        if replacement_choices.len() > MAX_REPLACEMENT_CHOICES {
            return Err(LegacyExtensionPlaceholderError::TooManyReplacementChoices {
                actual: replacement_choices.len(),
                maximum: MAX_REPLACEMENT_CHOICES,
            });
        }
        let mut replacement_identifiers = std::collections::BTreeSet::new();
        for replacement in &replacement_choices {
            if !replacement_identifiers.insert(replacement.identifier()) {
                return Err(LegacyExtensionPlaceholderError::DuplicateReplacement(
                    replacement.identifier().to_owned(),
                ));
            }
        }
        Ok(Self {
            extension_identity,
            hook_identity,
            exact_payload,
            diagnostic,
            replacement_choices: replacement_choices.into(),
        })
    }

    pub fn extension_identity(&self) -> &str {
        &self.extension_identity
    }

    pub fn hook_identity(&self) -> &str {
        &self.hook_identity
    }

    pub fn exact_payload(&self) -> &[u8] {
        &self.exact_payload
    }

    pub fn diagnostic(&self) -> &str {
        &self.diagnostic
    }

    pub fn replacement_choices(&self) -> &[LegacyExtensionReplacementChoice] {
        &self.replacement_choices
    }
}

impl RenderOnce for LegacyExtensionPlaceholder {
    fn render(self, _window: &mut Window, cx: &mut App) -> impl IntoElement {
        let replacement_summary = if self.replacement_choices.is_empty() {
            "No native replacement is currently available".to_owned()
        } else {
            format!(
                "Native replacement choices: {}",
                self.replacement_choices
                    .iter()
                    .map(|choice| choice.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        v_flex()
            .id(SharedString::from(format!(
                "legacy-extension-placeholder-{}-{}",
                self.extension_identity, self.hook_identity
            )))
            .role(Role::Alert)
            .aria_label(format!(
                "Unavailable native extension {} hook {}. {}. {}",
                self.extension_identity, self.hook_identity, self.diagnostic, replacement_summary
            ))
            .w_full()
            .gap_1()
            .p_2()
            .rounded_sm()
            .bg(cx.theme().status().warning_background.opacity(0.16))
            .child(
                div()
                    .text_ui_sm(cx)
                    .child(format!("Missing extension: {}", self.extension_identity)),
            )
            .child(
                div()
                    .text_ui_sm(cx)
                    .child(format!("Unsupported hook: {}", self.hook_identity)),
            )
            .child(div().text_ui_sm(cx).child(self.diagnostic.to_string()))
            .child(div().text_ui_sm(cx).child(replacement_summary))
    }
}

fn validate_identity(
    field: &'static str,
    value: &str,
) -> Result<(), LegacyExtensionPlaceholderError> {
    if value.is_empty()
        || value.len() > MAX_LEGACY_EXTENSION_IDENTITY_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(LegacyExtensionPlaceholderError::InvalidText(field));
    }
    Ok(())
}

fn validate_display_text(
    field: &'static str,
    value: &str,
) -> Result<(), LegacyExtensionPlaceholderError> {
    if value.is_empty() || value.len() > 4_096 || value.chars().any(char::is_control) {
        return Err(LegacyExtensionPlaceholderError::InvalidText(field));
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LegacyExtensionPlaceholderError {
    #[error("invalid legacy extension {0}")]
    InvalidText(&'static str),
    #[error("legacy extension payload has {actual} bytes; maximum is {maximum}")]
    PayloadTooLarge { actual: usize, maximum: usize },
    #[error("legacy extension placeholder has {actual} replacement choices; maximum is {maximum}")]
    TooManyReplacementChoices { actual: usize, maximum: usize },
    #[error("duplicate legacy extension replacement `{0}`")]
    DuplicateReplacement(String),
}
