use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use crate::{
    NativeRuntimeProfile, PluginPolicy,
    legacy_connections::{contains_legacy_secret_key, sanitize_legacy_fields},
};

pub const CURRENT_LEGACY_INSTALLATION_MIGRATION_VERSION: u16 = 1;

const MAX_LEGACY_INSTALLATION_BYTES: usize = 16 * 1024 * 1024;
const MAX_LEGACY_INSTALLATION_NAME_BYTES: usize = 1_024;
const MAX_LEGACY_INSTALLATION_ITEMS: usize = 4_096;
const MAX_LEGACY_INSTALLATION_ITEM_BYTES: usize = 16 * 1_024;
const MAX_LEGACY_INSTALLATION_EXPLANATION_BYTES: usize = 4_096;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyLifecycleAction {
    Launch,
    Connect,
    Update,
    Delete,
    Reconfigure,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct LegacyLifecycleEvidence {
    #[serde(default)]
    pub python_environment_managed: bool,
    #[serde(default)]
    pub git_custom_nodes_managed: bool,
    #[serde(default)]
    pub sim_server_managed: bool,
    #[serde(default)]
    pub was_running: bool,
    #[serde(default)]
    pub automatic_updates_enabled: bool,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LegacyInstallationImport {
    pub name: String,
    pub installation_location: Option<String>,
    #[serde(default)]
    pub model_roots: Vec<String>,
    #[serde(default)]
    pub workflow_stores: Vec<String>,
    #[serde(default)]
    pub output_stores: Vec<String>,
    #[serde(default)]
    pub extension_references: Vec<String>,
    #[serde(default)]
    pub settings: BTreeMap<String, Value>,
    #[serde(default)]
    pub credentials: BTreeMap<String, Value>,
    #[serde(default)]
    pub lifecycle: LegacyLifecycleEvidence,
    #[serde(default)]
    pub requested_lifecycle_actions: Vec<LegacyLifecycleAction>,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LegacyInstallationRecord {
    pub name: String,
    pub installation_location_hint: Option<String>,
    pub location_removed_or_redacted: bool,
    pub model_roots: Vec<String>,
    pub workflow_stores: Vec<String>,
    pub output_stores: Vec<String>,
    pub extension_references: Vec<String>,
    pub settings: BTreeMap<String, Value>,
    pub lifecycle: LegacyLifecycleEvidence,
    pub active: bool,
    pub read_only: bool,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyInstallationField {
    ModelRoots,
    WorkflowStores,
    OutputStores,
    Settings,
    ExtensionReferences,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyInstallationConversionOwner {
    ArtifactRootAndAssetService,
    WorkflowFormatDocument,
    OutputCommitterAndAssetService,
    SettingsStore,
    LegacyMappingResolver,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LegacyInstallationConversionStep {
    pub field: LegacyInstallationField,
    pub canonical_owner: LegacyInstallationConversionOwner,
    pub retained_items: usize,
    pub requires_explicit_acceptance: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LegacyInstallationMigrationPresentation {
    pub migration_id: Uuid,
    pub inactive_installation_name: String,
    pub installation_location_hint: Option<String>,
    pub native_profile_id: Uuid,
    pub native_profile_name: String,
    pub refused_lifecycle_actions: Vec<LegacyLifecycleAction>,
    pub credentials_removed: bool,
    pub removed_secret_values: u32,
    pub conversion_steps: Vec<LegacyInstallationConversionStep>,
    pub explanation: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LegacyInstallationMigrationResult {
    pub schema_version: u16,
    pub migration_id: Uuid,
    pub inactive_legacy_installation: LegacyInstallationRecord,
    pub native_profile: NativeRuntimeProfile,
    pub refused_lifecycle_actions: Vec<LegacyLifecycleAction>,
    pub credentials_removed: bool,
    pub removed_secret_values: u32,
    pub explanation: String,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum LegacyInstallationMigrationError {
    #[error("legacy installation migration id cannot be nil")]
    NilMigrationId,
    #[error("native profile id cannot be nil")]
    NilNativeProfileId,
    #[error("legacy installation exceeds the {MAX_LEGACY_INSTALLATION_BYTES}-byte migration limit")]
    InstallationTooLarge,
    #[error(
        "legacy installation name exceeds the {MAX_LEGACY_INSTALLATION_NAME_BYTES}-byte migration limit"
    )]
    NameTooLarge,
    #[error("legacy installation has too many {0}")]
    TooManyItems(&'static str),
    #[error(
        "legacy installation {0} exceeds the {MAX_LEGACY_INSTALLATION_ITEM_BYTES}-byte item limit"
    )]
    ItemTooLarge(&'static str),
    #[error("legacy installation repeats lifecycle action {0:?}")]
    DuplicateLifecycleAction(LegacyLifecycleAction),
    #[error("legacy installation payload is invalid: {0}")]
    InvalidPayload(String),
}

impl LegacyInstallationMigrationResult {
    pub fn presentation(&self) -> LegacyInstallationMigrationPresentation {
        let inactive = &self.inactive_legacy_installation;
        LegacyInstallationMigrationPresentation {
            migration_id: self.migration_id,
            inactive_installation_name: inactive.name.clone(),
            installation_location_hint: inactive.installation_location_hint.clone(),
            native_profile_id: self.native_profile.id,
            native_profile_name: self.native_profile.name.clone(),
            refused_lifecycle_actions: self.refused_lifecycle_actions.clone(),
            credentials_removed: self.credentials_removed,
            removed_secret_values: self.removed_secret_values,
            conversion_steps: vec![
                conversion_step(
                    LegacyInstallationField::ModelRoots,
                    LegacyInstallationConversionOwner::ArtifactRootAndAssetService,
                    inactive.model_roots.len(),
                ),
                conversion_step(
                    LegacyInstallationField::WorkflowStores,
                    LegacyInstallationConversionOwner::WorkflowFormatDocument,
                    inactive.workflow_stores.len(),
                ),
                conversion_step(
                    LegacyInstallationField::OutputStores,
                    LegacyInstallationConversionOwner::OutputCommitterAndAssetService,
                    inactive.output_stores.len(),
                ),
                conversion_step(
                    LegacyInstallationField::Settings,
                    LegacyInstallationConversionOwner::SettingsStore,
                    inactive.settings.len(),
                ),
                conversion_step(
                    LegacyInstallationField::ExtensionReferences,
                    LegacyInstallationConversionOwner::LegacyMappingResolver,
                    inactive.extension_references.len(),
                ),
            ],
            explanation: self.explanation.clone(),
        }
    }

    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == CURRENT_LEGACY_INSTALLATION_MIGRATION_VERSION,
            "unsupported legacy installation migration schema {}",
            self.schema_version
        );
        anyhow::ensure!(!self.migration_id.is_nil(), "legacy migration id is nil");
        anyhow::ensure!(
            !self.native_profile.id.is_nil(),
            "migrated native profile id is nil"
        );
        let inactive = &self.inactive_legacy_installation;
        anyhow::ensure!(
            !inactive.active && inactive.read_only,
            "legacy installation is executable or mutable"
        );
        anyhow::ensure!(
            self.native_profile.model_roots.is_empty(),
            "legacy model roots bypass explicit ArtifactRoot review"
        );
        anyhow::ensure!(
            !self.native_profile.api_host.enabled && !self.native_profile.api_host.allow_remote,
            "legacy installation activates native API exposure"
        );
        anyhow::ensure!(
            self.native_profile.plugin_policy == PluginPolicy::Disabled,
            "legacy extension references activate plugin authority"
        );
        let expected_native_profile = NativeRuntimeProfile::disabled_migration_replacement(
            self.native_profile.id,
            &inactive.name,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            self.native_profile == expected_native_profile,
            "legacy installation overrides the canonical disabled native profile"
        );
        anyhow::ensure!(
            self.refused_lifecycle_actions
                .windows(2)
                .all(|actions| actions[0] < actions[1]),
            "refused lifecycle actions are duplicated or noncanonical"
        );
        validate_record(inactive)?;
        anyhow::ensure!(
            !contains_legacy_secret_key(&inactive.settings)
                && !contains_legacy_secret_key(&inactive.lifecycle.unknown_fields)
                && !contains_legacy_secret_key(&inactive.unknown_fields)
                && !contains_legacy_secret_key(&self.unknown_fields),
            "legacy installation retains a secret-bearing field"
        );
        anyhow::ensure!(
            !self.explanation.trim().is_empty()
                && self.explanation.len() <= MAX_LEGACY_INSTALLATION_EXPLANATION_BYTES,
            "legacy installation explanation is empty or oversized"
        );
        anyhow::ensure!(
            self.presentation()
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance),
            "legacy installation conversion bypasses explicit acceptance"
        );
        let serialized = serde_json::to_vec(self)?;
        anyhow::ensure!(
            serialized.len() <= MAX_LEGACY_INSTALLATION_BYTES,
            "legacy installation migration result exceeds its bound"
        );
        Ok(())
    }
}

pub fn migrate_legacy_installation_json(
    payload: &[u8],
    migration_id: Uuid,
    native_profile_id: Uuid,
) -> Result<LegacyInstallationMigrationResult, LegacyInstallationMigrationError> {
    if payload.len() > MAX_LEGACY_INSTALLATION_BYTES {
        return Err(LegacyInstallationMigrationError::InstallationTooLarge);
    }
    let input = serde_json::from_slice(payload)
        .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    migrate_legacy_installation(input, migration_id, native_profile_id)
}

pub fn migrate_legacy_installation(
    input: LegacyInstallationImport,
    migration_id: Uuid,
    native_profile_id: Uuid,
) -> Result<LegacyInstallationMigrationResult, LegacyInstallationMigrationError> {
    if migration_id.is_nil() {
        return Err(LegacyInstallationMigrationError::NilMigrationId);
    }
    if native_profile_id.is_nil() {
        return Err(LegacyInstallationMigrationError::NilNativeProfileId);
    }
    validate_input(&input)?;

    let LegacyInstallationImport {
        name,
        installation_location,
        model_roots,
        workflow_stores,
        output_stores,
        extension_references,
        settings,
        credentials,
        lifecycle,
        requested_lifecycle_actions,
        unknown_fields,
    } = input;
    let name = sanitized_display_text(&name, "Imported legacy ComfyUI installation");
    let (installation_location_hint, location_removed_or_redacted) =
        sanitize_location_hint(installation_location);
    let mut removed_secret_values = credentials.len().try_into().unwrap_or(u32::MAX);
    let settings = sanitize_legacy_fields(settings, &mut removed_secret_values);
    let lifecycle_unknown_fields =
        sanitize_legacy_fields(lifecycle.unknown_fields, &mut removed_secret_values);
    let unknown_fields = sanitize_legacy_fields(unknown_fields, &mut removed_secret_values);
    let lifecycle = LegacyLifecycleEvidence {
        unknown_fields: lifecycle_unknown_fields,
        ..lifecycle
    };
    let refused_lifecycle_actions = requested_lifecycle_actions
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let credentials_removed = !credentials.is_empty() || removed_secret_values > 0;
    let inactive_legacy_installation = LegacyInstallationRecord {
        name: name.clone(),
        installation_location_hint,
        location_removed_or_redacted,
        model_roots,
        workflow_stores,
        output_stores,
        extension_references,
        settings,
        lifecycle,
        active: false,
        read_only: true,
        unknown_fields,
    };
    let native_profile =
        NativeRuntimeProfile::disabled_migration_replacement(native_profile_id, &name)
            .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    let result = LegacyInstallationMigrationResult {
        schema_version: CURRENT_LEGACY_INSTALLATION_MIGRATION_VERSION,
        migration_id,
        inactive_legacy_installation,
        native_profile,
        refused_lifecycle_actions,
        credentials_removed,
        removed_secret_values,
        explanation: "The legacy Python/Git/ComfyUI lifecycle is read-only migration evidence. Every requested legacy lifecycle action is refused; a disabled native profile is offered, while model roots, workflows, outputs, settings, and extension references require explicit review by their canonical native owners before use.".into(),
        unknown_fields: BTreeMap::new(),
    };
    result
        .validate()
        .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    Ok(result)
}

pub fn decode_legacy_installation_migration(
    payload: &[u8],
) -> Result<LegacyInstallationMigrationResult, LegacyInstallationMigrationError> {
    if payload.len() > MAX_LEGACY_INSTALLATION_BYTES {
        return Err(LegacyInstallationMigrationError::InstallationTooLarge);
    }
    let result: LegacyInstallationMigrationResult = serde_json::from_slice(payload)
        .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    result
        .validate()
        .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    Ok(result)
}

fn conversion_step(
    field: LegacyInstallationField,
    canonical_owner: LegacyInstallationConversionOwner,
    retained_items: usize,
) -> LegacyInstallationConversionStep {
    LegacyInstallationConversionStep {
        field,
        canonical_owner,
        retained_items,
        requires_explicit_acceptance: true,
    }
}

fn validate_input(
    input: &LegacyInstallationImport,
) -> Result<(), LegacyInstallationMigrationError> {
    if input.name.len() > MAX_LEGACY_INSTALLATION_NAME_BYTES {
        return Err(LegacyInstallationMigrationError::NameTooLarge);
    }
    if let Some(location) = &input.installation_location {
        validate_item("location hint", location)?;
    }
    validate_items("model roots", &input.model_roots)?;
    validate_items("workflow stores", &input.workflow_stores)?;
    validate_items("output stores", &input.output_stores)?;
    validate_items("extension references", &input.extension_references)?;
    validate_map("settings", &input.settings)?;
    validate_map("credentials", &input.credentials)?;
    validate_map("lifecycle fields", &input.lifecycle.unknown_fields)?;
    validate_map("unknown fields", &input.unknown_fields)?;
    let mut actions = BTreeSet::new();
    for action in &input.requested_lifecycle_actions {
        if !actions.insert(*action) {
            return Err(LegacyInstallationMigrationError::DuplicateLifecycleAction(
                *action,
            ));
        }
    }
    let serialized = serde_json::to_vec(input)
        .map_err(|error| LegacyInstallationMigrationError::InvalidPayload(error.to_string()))?;
    if serialized.len() > MAX_LEGACY_INSTALLATION_BYTES {
        return Err(LegacyInstallationMigrationError::InstallationTooLarge);
    }
    Ok(())
}

fn validate_record(record: &LegacyInstallationRecord) -> anyhow::Result<()> {
    anyhow::ensure!(
        record.name.len() <= MAX_LEGACY_INSTALLATION_NAME_BYTES,
        "legacy installation name is oversized"
    );
    if let Some(location) = &record.installation_location_hint {
        validate_item("location hint", location)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            !location.contains("://")
                && sanitized_display_text(location, "").as_str() == location.as_str(),
            "legacy installation location hint is not canonical display-only evidence"
        );
    }
    validate_items("model roots", &record.model_roots)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_items("workflow stores", &record.workflow_stores)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_items("output stores", &record.output_stores)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_items("extension references", &record.extension_references)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_map("settings", &record.settings)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_map("lifecycle fields", &record.lifecycle.unknown_fields)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_map("unknown fields", &record.unknown_fields)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

fn validate_items(
    label: &'static str,
    values: &[String],
) -> Result<(), LegacyInstallationMigrationError> {
    if values.len() > MAX_LEGACY_INSTALLATION_ITEMS {
        return Err(LegacyInstallationMigrationError::TooManyItems(label));
    }
    for value in values {
        validate_item(label, value)?;
    }
    Ok(())
}

fn validate_item(label: &'static str, value: &str) -> Result<(), LegacyInstallationMigrationError> {
    if value.len() > MAX_LEGACY_INSTALLATION_ITEM_BYTES {
        return Err(LegacyInstallationMigrationError::ItemTooLarge(label));
    }
    Ok(())
}

fn validate_map(
    label: &'static str,
    values: &BTreeMap<String, Value>,
) -> Result<(), LegacyInstallationMigrationError> {
    if values.len() > MAX_LEGACY_INSTALLATION_ITEMS {
        return Err(LegacyInstallationMigrationError::TooManyItems(label));
    }
    for key in values.keys() {
        validate_item(label, key)?;
    }
    Ok(())
}

fn sanitized_display_text(value: &str, fallback: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim();
    if sanitized.is_empty() {
        fallback.into()
    } else {
        sanitized.into()
    }
}

fn sanitize_location_hint(location: Option<String>) -> (Option<String>, bool) {
    let Some(location) = location else {
        return (None, false);
    };
    if location.contains("://") {
        return (None, true);
    }
    let sanitized = sanitized_display_text(&location, "");
    if sanitized.is_empty() {
        (None, true)
    } else {
        let redacted = sanitized != location;
        (Some(sanitized), redacted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> LegacyInstallationImport {
        LegacyInstallationImport {
            name: "Legacy Desktop".into(),
            installation_location: Some("/legacy/ComfyUI".into()),
            model_roots: vec!["/legacy/ComfyUI/models".into()],
            workflow_stores: vec!["/legacy/ComfyUI/user/workflows".into()],
            output_stores: vec!["/legacy/ComfyUI/output".into()],
            extension_references: vec!["custom_nodes/legacy-node".into()],
            settings: BTreeMap::from([
                ("preview_method".into(), Value::String("auto".into())),
                ("api_token".into(), Value::String("hidden".into())),
            ]),
            credentials: BTreeMap::from([("manager".into(), Value::String("hidden".into()))]),
            lifecycle: LegacyLifecycleEvidence {
                python_environment_managed: true,
                git_custom_nodes_managed: true,
                sim_server_managed: true,
                was_running: true,
                automatic_updates_enabled: true,
                unknown_fields: BTreeMap::from([(
                    "nested".into(),
                    serde_json::json!({"password": "hidden", "retained": true}),
                )]),
            },
            requested_lifecycle_actions: vec![
                LegacyLifecycleAction::Launch,
                LegacyLifecycleAction::Connect,
                LegacyLifecycleAction::Update,
                LegacyLifecycleAction::Delete,
                LegacyLifecycleAction::Reconfigure,
            ],
            unknown_fields: BTreeMap::from([("revision".into(), Value::from(7))]),
        }
    }

    #[test]
    fn migration_is_read_only_and_routes_every_field_to_one_owner() -> anyhow::Result<()> {
        let result =
            migrate_legacy_installation(input(), Uuid::from_u128(0x3001), Uuid::from_u128(0x3002))?;
        assert!(!result.inactive_legacy_installation.active);
        assert!(result.inactive_legacy_installation.read_only);
        assert!(result.native_profile.model_roots.is_empty());
        assert!(!result.native_profile.api_host.enabled);
        assert_eq!(result.native_profile.plugin_policy, PluginPolicy::Disabled);
        assert_eq!(result.refused_lifecycle_actions.len(), 5);
        let presentation = result.presentation();
        assert_eq!(presentation.conversion_steps.len(), 5);
        assert!(
            presentation
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance)
        );
        let encoded = serde_json::to_vec(&result)?;
        assert_eq!(decode_legacy_installation_migration(&encoded)?, result);
        assert!(!String::from_utf8(encoded)?.contains("hidden"));
        Ok(())
    }

    #[test]
    fn malformed_and_oversized_imports_fail_closed() -> anyhow::Result<()> {
        let mut duplicate_action = input();
        duplicate_action
            .requested_lifecycle_actions
            .push(LegacyLifecycleAction::Launch);
        assert_eq!(
            migrate_legacy_installation(
                duplicate_action,
                Uuid::from_u128(0x3003),
                Uuid::from_u128(0x3004),
            ),
            Err(LegacyInstallationMigrationError::DuplicateLifecycleAction(
                LegacyLifecycleAction::Launch
            ))
        );

        let oversized = vec![b' '; MAX_LEGACY_INSTALLATION_BYTES + 1];
        assert_eq!(
            migrate_legacy_installation_json(
                &oversized,
                Uuid::from_u128(0x3005),
                Uuid::from_u128(0x3006),
            ),
            Err(LegacyInstallationMigrationError::InstallationTooLarge)
        );
        assert!(matches!(
            migrate_legacy_installation_json(
                b"{",
                Uuid::from_u128(0x3005),
                Uuid::from_u128(0x3006),
            ),
            Err(LegacyInstallationMigrationError::InvalidPayload(_))
        ));

        let mut remote_location = input();
        remote_location.installation_location = Some("https://example.invalid/ComfyUI".into());
        let result = migrate_legacy_installation(
            remote_location,
            Uuid::from_u128(0x3007),
            Uuid::from_u128(0x3008),
        )?;
        assert!(
            result
                .inactive_legacy_installation
                .location_removed_or_redacted
        );
        assert!(
            result
                .inactive_legacy_installation
                .installation_location_hint
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn tampered_evidence_cannot_activate_or_acquire_authority() -> anyhow::Result<()> {
        let result =
            migrate_legacy_installation(input(), Uuid::from_u128(0x3009), Uuid::from_u128(0x3010))?;
        let value = serde_json::to_value(result)?;
        for (pointer, replacement) in [
            ("/inactive_legacy_installation/active", Value::Bool(true)),
            (
                "/inactive_legacy_installation/read_only",
                Value::Bool(false),
            ),
            (
                "/native_profile/model_roots",
                serde_json::json!(["/unreviewed"]),
            ),
            ("/native_profile/api_host/enabled", Value::Bool(true)),
            (
                "/native_profile/plugin_policy",
                Value::String("approved_only".into()),
            ),
            (
                "/native_profile/provider_scope",
                Value::String("remote".into()),
            ),
            (
                "/native_profile/api_host/bind",
                Value::String("0.0.0.0:8188".into()),
            ),
            (
                "/native_profile/unknown_fields",
                serde_json::json!({"provider_token": "hidden"}),
            ),
            (
                "/inactive_legacy_installation/installation_location_hint",
                Value::String("https://example.invalid/ComfyUI".into()),
            ),
            (
                "/refused_lifecycle_actions",
                serde_json::json!(["update", "launch"]),
            ),
        ] {
            let mut tampered = value.clone();
            let field = tampered
                .pointer_mut(pointer)
                .ok_or_else(|| anyhow::anyhow!("missing migration field {pointer}"))?;
            *field = replacement;
            assert!(
                decode_legacy_installation_migration(&serde_json::to_vec(&tampered)?).is_err(),
                "accepted tampered field {pointer}"
            );
        }

        let mut secret_bearing = value;
        let settings = secret_bearing
            .pointer_mut("/inactive_legacy_installation/settings")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow::anyhow!("missing inactive settings"))?;
        settings.insert("api_token".into(), Value::String("hidden".into()));
        assert!(
            decode_legacy_installation_migration(&serde_json::to_vec(&secret_bearing)?).is_err()
        );
        Ok(())
    }
}
