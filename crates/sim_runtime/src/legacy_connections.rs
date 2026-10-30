use std::{collections::BTreeMap, net::SocketAddr};

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use thiserror::Error;
use url::Url;
use uuid::Uuid;

use crate::NativeRuntimeProfile;

pub const CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION: u16 = 2;

const MAX_LEGACY_PROFILE_BYTES: usize = 16 * 1024 * 1024;
const MAX_LEGACY_NAME_BYTES: usize = 1_024;
const MAX_LEGACY_ITEMS: usize = 4_096;
const MAX_LEGACY_ITEM_BYTES: usize = 16 * 1_024;
const MAX_MIGRATION_EXPLANATION_BYTES: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegacyComfyProfile {
    pub name: String,
    pub endpoint: Option<String>,
    pub credential: Option<String>,
    pub model_roots: Vec<String>,
    pub api_host_enabled: bool,
    pub plugin_mappings: Vec<String>,
    #[serde(default)]
    pub workflow_state: BTreeMap<String, Value>,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct InactiveLegacyOrigin(String);

impl InactiveLegacyOrigin {
    pub fn display(&self) -> &str {
        &self.0
    }

    fn checked(value: String) -> Result<Self, LegacyMigrationError> {
        let parsed = Url::parse(&value).map_err(|_| LegacyMigrationError::InvalidStoredOrigin)?;
        if !matches!(parsed.scheme(), "http" | "https")
            || parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
            || parsed.query().is_some()
            || parsed.fragment().is_some()
            || parsed.origin().ascii_serialization() != value
        {
            return Err(LegacyMigrationError::InvalidStoredOrigin);
        }
        Ok(Self(value))
    }
}

impl<'de> Deserialize<'de> for InactiveLegacyOrigin {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::checked(value).map_err(D::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegacyMigrationResult {
    pub schema_version: u16,
    pub migration_id: Uuid,
    pub inactive_legacy_profile: LegacyComfyProfileRecord,
    pub native_profile: NativeRuntimeProfile,
    pub credential_removed: bool,
    #[serde(default)]
    pub removed_secret_values: u32,
    #[serde(default)]
    pub endpoint_removed_or_redacted: bool,
    pub explanation: String,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LegacyComfyProfileRecord {
    pub name: String,
    pub former_endpoint: Option<InactiveLegacyOrigin>,
    pub model_roots: Vec<String>,
    pub plugin_mappings: Vec<String>,
    #[serde(default)]
    pub workflow_state: BTreeMap<String, Value>,
    #[serde(default)]
    pub api_host_was_enabled: bool,
    pub active: bool,
    #[serde(default, flatten)]
    pub unknown_fields: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyConnectionField {
    ModelRoots,
    ApiHostPolicy,
    PluginMappings,
    WorkflowState,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LegacyConversionOwner {
    ArtifactRootAndAssetService,
    NativeApiExposure,
    LegacyMappingResolver,
    WorkflowFormatDocument,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LegacyConversionStep {
    pub field: LegacyConnectionField,
    pub canonical_owner: LegacyConversionOwner,
    pub retained_items: usize,
    pub requires_explicit_acceptance: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LegacyMigrationPresentation {
    pub migration_id: Uuid,
    pub inactive_profile_name: String,
    pub former_origin: Option<String>,
    pub native_profile_id: Uuid,
    pub native_profile_name: String,
    pub credential_removed: bool,
    pub removed_secret_values: u32,
    pub endpoint_removed_or_redacted: bool,
    pub conversion_steps: Vec<LegacyConversionStep>,
    pub explanation: String,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum LegacyMigrationError {
    #[error("legacy migration id cannot be nil")]
    NilMigrationId,
    #[error("native profile id cannot be nil")]
    NilNativeProfileId,
    #[error("legacy profile exceeds the {MAX_LEGACY_PROFILE_BYTES}-byte migration limit")]
    ProfileTooLarge,
    #[error("legacy profile name exceeds the {MAX_LEGACY_NAME_BYTES}-byte migration limit")]
    NameTooLarge,
    #[error("legacy profile has too many {0}")]
    TooManyItems(&'static str),
    #[error("legacy {0} exceeds the {MAX_LEGACY_ITEM_BYTES}-byte item limit")]
    ItemTooLarge(&'static str),
    #[error("stored inactive legacy origin is not a canonical HTTP(S) origin")]
    InvalidStoredOrigin,
    #[error("legacy migration payload is invalid: {0}")]
    InvalidPayload(String),
}

impl LegacyMigrationResult {
    pub fn presentation(&self) -> LegacyMigrationPresentation {
        let inactive = &self.inactive_legacy_profile;
        LegacyMigrationPresentation {
            migration_id: self.migration_id,
            inactive_profile_name: inactive.name.clone(),
            former_origin: inactive
                .former_endpoint
                .as_ref()
                .map(|origin| origin.display().to_owned()),
            native_profile_id: self.native_profile.id,
            native_profile_name: self.native_profile.name.clone(),
            credential_removed: self.credential_removed,
            removed_secret_values: self.removed_secret_values,
            endpoint_removed_or_redacted: self.endpoint_removed_or_redacted,
            conversion_steps: vec![
                LegacyConversionStep {
                    field: LegacyConnectionField::ModelRoots,
                    canonical_owner: LegacyConversionOwner::ArtifactRootAndAssetService,
                    retained_items: inactive.model_roots.len(),
                    requires_explicit_acceptance: true,
                },
                LegacyConversionStep {
                    field: LegacyConnectionField::ApiHostPolicy,
                    canonical_owner: LegacyConversionOwner::NativeApiExposure,
                    retained_items: usize::from(inactive.api_host_was_enabled),
                    requires_explicit_acceptance: true,
                },
                LegacyConversionStep {
                    field: LegacyConnectionField::PluginMappings,
                    canonical_owner: LegacyConversionOwner::LegacyMappingResolver,
                    retained_items: inactive.plugin_mappings.len(),
                    requires_explicit_acceptance: true,
                },
                LegacyConversionStep {
                    field: LegacyConnectionField::WorkflowState,
                    canonical_owner: LegacyConversionOwner::WorkflowFormatDocument,
                    retained_items: inactive.workflow_state.len(),
                    requires_explicit_acceptance: true,
                },
            ],
            explanation: self.explanation.clone(),
        }
    }

    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION,
            "unsupported legacy connection migration schema {}",
            self.schema_version
        );
        anyhow::ensure!(!self.migration_id.is_nil(), "legacy migration id is nil");
        anyhow::ensure!(
            !self.native_profile.id.is_nil(),
            "migrated native profile id is nil"
        );
        anyhow::ensure!(
            !self.inactive_legacy_profile.active,
            "legacy profile migration is active"
        );
        anyhow::ensure!(
            self.native_profile.model_roots.is_empty(),
            "legacy model roots bypass explicit ArtifactRoot review"
        );
        anyhow::ensure!(
            !self.native_profile.api_host.enabled,
            "legacy API-host preference bypasses explicit native exposure review"
        );
        let bind = self
            .native_profile
            .api_host
            .bind
            .parse::<SocketAddr>()
            .map_err(|error| anyhow::anyhow!("invalid migrated native API bind: {error}"))?;
        anyhow::ensure!(
            bind.ip().is_loopback(),
            "migrated native API bind is remote"
        );
        anyhow::ensure!(
            !self.explanation.trim().is_empty()
                && self.explanation.len() <= MAX_MIGRATION_EXPLANATION_BYTES,
            "legacy migration explanation is empty or oversized"
        );
        validate_inactive_profile(&self.inactive_legacy_profile)?;
        ensure_unknown_fields_do_not_shadow(
            &self.unknown_fields,
            &[
                "schema_version",
                "migration_id",
                "inactive_legacy_profile",
                "native_profile",
                "credential_removed",
                "removed_secret_values",
                "endpoint_removed_or_redacted",
                "explanation",
            ],
        )?;
        anyhow::ensure!(
            !contains_legacy_secret_key(&self.unknown_fields)
                && !contains_legacy_secret_key(&self.inactive_legacy_profile.unknown_fields)
                && !contains_legacy_secret_key(&self.inactive_legacy_profile.workflow_state),
            "legacy migration retains a secret-bearing field"
        );
        let serialized = serde_json::to_vec(self)?;
        anyhow::ensure!(
            serialized.len() <= MAX_LEGACY_PROFILE_BYTES,
            "legacy migration result exceeds its persistence bound"
        );
        anyhow::ensure!(
            self.presentation()
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance),
            "legacy conversion step is executable without explicit acceptance"
        );
        Ok(())
    }
}

pub fn migrate_legacy_profile(
    profile: LegacyComfyProfile,
    migration_id: Uuid,
    native_profile_id: Uuid,
) -> Result<LegacyMigrationResult, LegacyMigrationError> {
    if migration_id.is_nil() {
        return Err(LegacyMigrationError::NilMigrationId);
    }
    if native_profile_id.is_nil() {
        return Err(LegacyMigrationError::NilNativeProfileId);
    }
    validate_legacy_input(&profile)?;

    let LegacyComfyProfile {
        name,
        endpoint,
        credential,
        model_roots,
        api_host_enabled,
        plugin_mappings,
        workflow_state,
        unknown_fields,
    } = profile;
    let name = sanitized_display_name(&name);
    let mut removed_secret_values = 0_u32;
    let unknown_fields = sanitize_legacy_fields(unknown_fields, &mut removed_secret_values);
    let workflow_state = sanitize_legacy_fields(workflow_state, &mut removed_secret_values);
    let (former_endpoint, endpoint_removed_or_redacted) = sanitize_legacy_endpoint(endpoint);
    let explicit_credential_removed = credential.is_some();
    if explicit_credential_removed {
        removed_secret_values = removed_secret_values.saturating_add(1);
    }
    let credential_removed =
        explicit_credential_removed || removed_secret_values > 0 || endpoint_removed_or_redacted;
    let inactive_legacy_profile = LegacyComfyProfileRecord {
        name: name.clone(),
        former_endpoint,
        model_roots,
        plugin_mappings,
        workflow_state,
        api_host_was_enabled: api_host_enabled,
        active: false,
        unknown_fields,
    };
    let native_profile =
        NativeRuntimeProfile::disabled_migration_replacement(native_profile_id, &name)
            .map_err(|error| LegacyMigrationError::InvalidPayload(error.to_string()))?;
    let result = LegacyMigrationResult {
        schema_version: CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION,
        migration_id,
        inactive_legacy_profile,
        native_profile,
        credential_removed,
        removed_secret_values,
        endpoint_removed_or_redacted,
        explanation: "External ComfyUI execution is unsupported. The source record is inactive; a disabled native profile is offered, while model roots, API-host policy, plugin mappings, and workflow state require explicit review by their canonical native owners before use.".into(),
        unknown_fields: BTreeMap::new(),
    };
    result
        .validate()
        .map_err(|error| LegacyMigrationError::InvalidPayload(error.to_string()))?;
    Ok(result)
}

pub(crate) fn decode_legacy_migration(json: &str) -> anyhow::Result<LegacyMigrationResult> {
    anyhow::ensure!(
        json.len() <= MAX_LEGACY_PROFILE_BYTES,
        "legacy migration payload exceeds the {MAX_LEGACY_PROFILE_BYTES}-byte migration limit"
    );
    let value: Value = serde_json::from_str(json)?;
    match value.get("schema_version").and_then(Value::as_u64) {
        Some(version) if version == u64::from(CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION) => {
            let result: LegacyMigrationResult = serde_json::from_value(value)?;
            result.validate()?;
            Ok(result)
        }
        Some(1) => upgrade_v1(serde_json::from_value(value)?),
        Some(version) => anyhow::bail!("unsupported legacy connection migration schema {version}"),
        None => anyhow::bail!("legacy connection migration has no schema version"),
    }
}

#[derive(Deserialize)]
struct LegacyMigrationResultV1 {
    #[serde(rename = "schema_version")]
    _schema_version: u16,
    migration_id: Uuid,
    inactive_legacy_profile: LegacyComfyProfileRecordV1,
    native_profile: NativeRuntimeProfile,
    credential_removed: bool,
    #[serde(rename = "explanation")]
    _explanation: String,
    #[serde(default, flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct LegacyComfyProfileRecordV1 {
    name: String,
    former_endpoint: Option<String>,
    model_roots: Vec<String>,
    plugin_mappings: Vec<String>,
    active: bool,
    #[serde(default, flatten)]
    unknown_fields: BTreeMap<String, Value>,
}

fn upgrade_v1(value: LegacyMigrationResultV1) -> anyhow::Result<LegacyMigrationResult> {
    let (former_endpoint, endpoint_removed_or_redacted) =
        sanitize_legacy_endpoint(value.inactive_legacy_profile.former_endpoint);
    let mut removed_secret_values = 0_u32;
    let inactive_unknown_fields = sanitize_legacy_fields(
        value.inactive_legacy_profile.unknown_fields,
        &mut removed_secret_values,
    );
    let unknown_fields = sanitize_legacy_fields(value.unknown_fields, &mut removed_secret_values);
    let native_profile = NativeRuntimeProfile::disabled_migration_replacement(
        value.native_profile.id,
        &value.native_profile.name,
    )?;
    let result = LegacyMigrationResult {
        schema_version: CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION,
        migration_id: value.migration_id,
        inactive_legacy_profile: LegacyComfyProfileRecord {
            name: value.inactive_legacy_profile.name,
            former_endpoint,
            model_roots: value.inactive_legacy_profile.model_roots,
            plugin_mappings: value.inactive_legacy_profile.plugin_mappings,
            workflow_state: BTreeMap::new(),
            api_host_was_enabled: value.native_profile.api_host.enabled,
            active: value.inactive_legacy_profile.active,
            unknown_fields: inactive_unknown_fields,
        },
        native_profile,
        credential_removed: value.credential_removed
            || removed_secret_values > 0
            || endpoint_removed_or_redacted,
        removed_secret_values,
        endpoint_removed_or_redacted,
        explanation: "External ComfyUI execution is unsupported. The stored v1 projection was upgraded in memory; reusable fields remain inactive pending explicit native-owner review.".into(),
        unknown_fields,
    };
    result.validate()?;
    Ok(result)
}

fn validate_legacy_input(profile: &LegacyComfyProfile) -> Result<(), LegacyMigrationError> {
    if profile.name.len() > MAX_LEGACY_NAME_BYTES {
        return Err(LegacyMigrationError::NameTooLarge);
    }
    validate_items("model roots", &profile.model_roots)?;
    validate_items("plugin mappings", &profile.plugin_mappings)?;
    validate_map("workflow records", &profile.workflow_state)?;
    validate_map("unknown fields", &profile.unknown_fields)?;
    let serialized = serde_json::to_vec(profile)
        .map_err(|error| LegacyMigrationError::InvalidPayload(error.to_string()))?;
    if serialized.len() > MAX_LEGACY_PROFILE_BYTES {
        return Err(LegacyMigrationError::ProfileTooLarge);
    }
    Ok(())
}

fn validate_inactive_profile(profile: &LegacyComfyProfileRecord) -> anyhow::Result<()> {
    anyhow::ensure!(
        profile.name.len() <= MAX_LEGACY_NAME_BYTES,
        "legacy name is oversized"
    );
    validate_items("model roots", &profile.model_roots)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_items("plugin mappings", &profile.plugin_mappings)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    validate_map("workflow records", &profile.workflow_state)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    ensure_unknown_fields_do_not_shadow(
        &profile.unknown_fields,
        &[
            "name",
            "former_endpoint",
            "model_roots",
            "plugin_mappings",
            "workflow_state",
            "api_host_was_enabled",
            "active",
        ],
    )
}

fn validate_items(label: &'static str, values: &[String]) -> Result<(), LegacyMigrationError> {
    if values.len() > MAX_LEGACY_ITEMS {
        return Err(LegacyMigrationError::TooManyItems(label));
    }
    if values
        .iter()
        .any(|value| value.len() > MAX_LEGACY_ITEM_BYTES)
    {
        return Err(LegacyMigrationError::ItemTooLarge(label));
    }
    Ok(())
}

fn validate_map(
    label: &'static str,
    values: &BTreeMap<String, Value>,
) -> Result<(), LegacyMigrationError> {
    if values.len() > MAX_LEGACY_ITEMS {
        return Err(LegacyMigrationError::TooManyItems(label));
    }
    if values.keys().any(|key| key.len() > MAX_LEGACY_ITEM_BYTES) {
        return Err(LegacyMigrationError::ItemTooLarge(label));
    }
    Ok(())
}

fn sanitized_display_name(name: &str) -> String {
    let sanitized = name
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
        "Imported legacy ComfyUI profile".into()
    } else {
        sanitized.into()
    }
}

fn sanitize_legacy_endpoint(endpoint: Option<String>) -> (Option<InactiveLegacyOrigin>, bool) {
    let Some(endpoint) = endpoint else {
        return (None, false);
    };
    let Ok(parsed) = Url::parse(&endpoint) else {
        return (None, true);
    };
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return (None, true);
    }
    let origin = parsed.origin().ascii_serialization();
    if origin == "null" {
        return (None, true);
    }
    match InactiveLegacyOrigin::checked(origin) {
        Ok(origin) => {
            let redacted = origin.display() != endpoint;
            (Some(origin), redacted)
        }
        Err(_) => (None, true),
    }
}

fn ensure_unknown_fields_do_not_shadow(
    fields: &BTreeMap<String, Value>,
    reserved: &[&str],
) -> anyhow::Result<()> {
    for key in fields.keys() {
        anyhow::ensure!(
            !reserved.contains(&key.as_str()),
            "unknown field `{key}` shadows a legacy migration field"
        );
    }
    Ok(())
}

pub(crate) fn sanitize_legacy_fields(
    fields: BTreeMap<String, Value>,
    removed_secret_values: &mut u32,
) -> BTreeMap<String, Value> {
    fields
        .into_iter()
        .filter_map(|(key, value)| {
            if is_secret_key(&key) {
                *removed_secret_values = removed_secret_values.saturating_add(1);
                return None;
            }
            Some((key, sanitize_value(value, removed_secret_values)))
        })
        .collect()
}

fn sanitize_value(value: Value, removed_secret_values: &mut u32) -> Value {
    match value {
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .filter_map(|(key, value)| {
                    if is_secret_key(&key) {
                        *removed_secret_values = removed_secret_values.saturating_add(1);
                        return None;
                    }
                    Some((key, sanitize_value(value, removed_secret_values)))
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| sanitize_value(value, removed_secret_values))
                .collect(),
        ),
        value => value,
    }
}

pub(crate) fn contains_legacy_secret_key(fields: &BTreeMap<String, Value>) -> bool {
    fields
        .iter()
        .any(|(key, value)| is_secret_key(key) || value_contains_secret_key(value))
}

fn value_contains_secret_key(value: &Value) -> bool {
    match value {
        Value::Object(values) => values
            .iter()
            .any(|(key, value)| is_secret_key(key) || value_contains_secret_key(value)),
        Value::Array(values) => values.iter().any(value_contains_secret_key),
        _ => false,
    }
}

fn is_secret_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    [
        "credential",
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
    ]
    .into_iter()
    .any(|marker| key.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_profile() -> LegacyComfyProfile {
        LegacyComfyProfile {
            name: "Old Server".into(),
            endpoint: Some("https://user:pass@example.invalid/private?access_token=hidden".into()),
            credential: Some("credential-value".into()),
            model_roots: vec!["models".into()],
            api_host_enabled: true,
            plugin_mappings: vec!["Old=Native".into()],
            workflow_state: BTreeMap::from([(
                "workflow-1".into(),
                serde_json::json!({"nodes": [], "api_token": "hidden"}),
            )]),
            unknown_fields: BTreeMap::from([
                ("future_mode".into(), serde_json::json!("preserved")),
                (
                    "nested".into(),
                    serde_json::json!({"access_token": "hidden", "kept": 1}),
                ),
            ]),
        }
    }

    #[test]
    fn legacy_migration_is_inactive_redacted_and_explicit() -> anyhow::Result<()> {
        let result =
            migrate_legacy_profile(legacy_profile(), Uuid::from_u128(1), Uuid::from_u128(2))?;
        assert!(result.credential_removed);
        assert_eq!(result.removed_secret_values, 3);
        assert!(result.endpoint_removed_or_redacted);
        assert!(!result.inactive_legacy_profile.active);
        assert_eq!(
            result
                .inactive_legacy_profile
                .former_endpoint
                .as_ref()
                .map(InactiveLegacyOrigin::display),
            Some("https://example.invalid")
        );
        assert!(result.native_profile.model_roots.is_empty());
        assert!(!result.native_profile.api_host.enabled);
        assert_eq!(result.native_profile.device, sim_types::DeviceKind::Cpu);
        let presentation = result.presentation();
        assert_eq!(presentation.conversion_steps.len(), 4);
        assert!(
            presentation
                .conversion_steps
                .iter()
                .all(|step| step.requires_explicit_acceptance)
        );
        assert_eq!(
            result
                .inactive_legacy_profile
                .unknown_fields
                .get("future_mode"),
            Some(&serde_json::json!("preserved"))
        );
        let serialized = serde_json::to_string(&result)?;
        for secret in [
            "credential-value",
            "hidden",
            "user:pass",
            "\"access_token\"",
            "\"api_token\"",
        ] {
            assert!(!serialized.contains(secret), "retained {secret}");
        }
        result.validate()?;
        Ok(())
    }

    #[test]
    fn v1_migration_is_upgraded_without_activating_reusable_fields() -> anyhow::Result<()> {
        let v1 = serde_json::json!({
            "schema_version": 1,
            "migration_id": Uuid::from_u128(3),
            "inactive_legacy_profile": {
                "name": "Old",
                "former_endpoint": "http://example.invalid",
                "model_roots": ["models"],
                "plugin_mappings": ["Old=Native"],
                "active": false
            },
            "native_profile": {
                "id": Uuid::from_u128(4),
                "name": "Old (Native)",
                "model_roots": ["models"],
                "device": "cpu",
                "memory_policy": "balanced",
                "api_host": {"enabled": true, "bind": "127.0.0.1:8188", "allow_remote": false},
                "plugin_policy": "approved_only",
                "provider_scope": "local",
                "compatibility_version": 1,
                "unknown_fields": {}
            },
            "credential_removed": true,
            "explanation": "legacy"
        });
        let upgraded = decode_legacy_migration(&serde_json::to_string(&v1)?)?;
        assert_eq!(
            upgraded.schema_version,
            CURRENT_LEGACY_CONNECTION_MIGRATION_VERSION
        );
        assert!(upgraded.native_profile.model_roots.is_empty());
        assert!(!upgraded.native_profile.api_host.enabled);
        assert!(upgraded.inactive_legacy_profile.api_host_was_enabled);
        assert_eq!(upgraded.inactive_legacy_profile.model_roots, ["models"]);
        Ok(())
    }

    #[test]
    fn malformed_and_oversized_legacy_data_fails_closed() -> anyhow::Result<()> {
        let oversized_payload = " ".repeat(MAX_LEGACY_PROFILE_BYTES + 1);
        let oversized_error = decode_legacy_migration(&oversized_payload)
            .expect_err("oversized persisted migration must fail before parsing");
        assert!(
            oversized_error
                .to_string()
                .contains("legacy migration payload exceeds")
        );

        let mut malformed_endpoint = legacy_profile();
        malformed_endpoint.endpoint = Some("file:///tmp/legacy".into());
        let migrated =
            migrate_legacy_profile(malformed_endpoint, Uuid::from_u128(5), Uuid::from_u128(6))?;
        assert!(migrated.endpoint_removed_or_redacted);
        assert!(migrated.inactive_legacy_profile.former_endpoint.is_none());

        let mut oversized_name = legacy_profile();
        oversized_name.name = "x".repeat(MAX_LEGACY_NAME_BYTES + 1);
        assert_eq!(
            migrate_legacy_profile(oversized_name, Uuid::from_u128(7), Uuid::from_u128(8)),
            Err(LegacyMigrationError::NameTooLarge)
        );

        let mut too_many_roots = legacy_profile();
        too_many_roots.model_roots = vec![String::from("models"); MAX_LEGACY_ITEMS + 1];
        assert_eq!(
            migrate_legacy_profile(too_many_roots, Uuid::from_u128(9), Uuid::from_u128(10)),
            Err(LegacyMigrationError::TooManyItems("model roots"))
        );

        let mut tampered = serde_json::to_value(migrated)?;
        let inactive = tampered
            .get_mut("inactive_legacy_profile")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow::anyhow!("inactive migration record is missing"))?;
        inactive.insert(
            "former_endpoint".into(),
            Value::String("https://example.invalid/executable-path".into()),
        );
        assert!(decode_legacy_migration(&serde_json::to_string(&tampered)?).is_err());
        Ok(())
    }
}
