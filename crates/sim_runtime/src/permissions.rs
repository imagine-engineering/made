use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
};

use serde::{Deserialize, Serialize};
use sim_plugin_sdk::{CapabilityKind, CapabilityRequest};
use thiserror::Error;

pub const PERMISSION_GRANT_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PermissionPolicyGeneration(NonZeroU64);

impl PermissionPolicyGeneration {
    pub fn new(value: u64) -> Result<Self, PermissionError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(PermissionError::InvalidPolicyGeneration)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}
pub const NATIVE_INPUT_READER_SUBJECT: &str = "native.input-reader";
pub const NATIVE_IMAGE_IMPORTER_SUBJECT: &str = "native.image-importer";
pub const NATIVE_IMAGE_OUTPUT_READER_SUBJECT: &str = "native.image-output-reader";
pub const NATIVE_API_ASSET_READER_SUBJECT: &str = "native.api-asset-reader";
pub const OUTPUT_COMMITTER_SUBJECT: &str = "native.output-committer";
pub const OUTPUT_UI_SUBJECT: &str = "native.output-ui";
pub const PLUGIN_ASSET_BROKER_SUBJECT: &str = "native.plugin-asset-broker";
pub const SUBGRAPH_LIBRARY_SUBJECT: &str = "native.subgraph-library";

fn native_input_reader_capabilities() -> CapabilitySet {
    CapabilitySet::new([Capability::Asset {
        namespace: "input".to_owned(),
        action: AssetOperation::Read,
    }])
}

fn native_image_importer_capabilities() -> CapabilitySet {
    CapabilitySet::new([Capability::Asset {
        namespace: "input".to_owned(),
        action: AssetOperation::Write,
    }])
}

fn native_image_output_reader_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        ["output", "temp"]
            .into_iter()
            .map(|namespace| Capability::Asset {
                namespace: namespace.to_owned(),
                action: AssetOperation::Read,
            }),
    )
}

fn native_api_asset_reader_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        ["input", "output", "temp", "model", "plugin"]
            .into_iter()
            .map(|namespace| Capability::Asset {
                namespace: namespace.to_owned(),
                action: AssetOperation::Read,
            }),
    )
}

fn output_committer_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        ["output", "temp"]
            .into_iter()
            .map(|namespace| Capability::Asset {
                namespace: namespace.to_owned(),
                action: AssetOperation::Write,
            }),
    )
}

fn output_ui_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        ["output", "temp"]
            .into_iter()
            .flat_map(|namespace| {
                [
                    Capability::Asset {
                        namespace: namespace.to_owned(),
                        action: AssetOperation::Read,
                    },
                    Capability::Asset {
                        namespace: namespace.to_owned(),
                        action: AssetOperation::Delete,
                    },
                ]
            })
            .chain([Capability::Asset {
                namespace: "temp".to_owned(),
                action: AssetOperation::Write,
            }]),
    )
}

fn plugin_asset_broker_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        ["input", "output", "temp", "model", "plugin"]
            .into_iter()
            .map(|namespace| Capability::Asset {
                namespace: namespace.to_owned(),
                action: AssetOperation::Read,
            }),
    )
}

fn subgraph_library_capabilities() -> CapabilitySet {
    CapabilitySet::new(
        [AssetOperation::Read, AssetOperation::Write]
            .into_iter()
            .map(|action| Capability::Asset {
                namespace: "plugin".to_owned(),
                action,
            }),
    )
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetOperation {
    Read,
    Write,
    Rename,
    Tag,
    Delete,
}

impl AssetOperation {
    fn wire_name(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Rename => "rename",
            Self::Tag => "tag",
            Self::Delete => "delete",
        }
    }

    fn parse_wire_name(value: &str) -> Option<Self> {
        match value {
            "read" => Some(Self::Read),
            "write" => Some(Self::Write),
            "rename" => Some(Self::Rename),
            "tag" => Some(Self::Tag),
            "delete" => Some(Self::Delete),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Asset {
        namespace: String,
        action: AssetOperation,
    },
    ProviderNetwork {
        provider: String,
        endpoint: String,
    },
    ProviderUpload {
        provider: String,
        endpoint: String,
    },
    ProviderCost {
        provider: String,
        endpoint: String,
    },
    Secret {
        secret_id: String,
    },
    Clock {
        clock_id: String,
    },
    Randomness {
        stream_id: String,
    },
    ModelHandle {
        model_id: String,
    },
    TransactionalOutput {
        namespace: String,
    },
    SanitizedLog {
        level: String,
    },
    DeclarativeUi {
        contribution_id: String,
    },
    NativeRoute {
        route_id: String,
    },
    Codec {
        codec_id: String,
    },
    ExternalNavigation,
    NativeFfi {
        library_id: String,
    },
}

impl Capability {
    pub fn from_plugin_request(request: &CapabilityRequest) -> Result<Self, PermissionError> {
        Self::from_plugin_scope(request.kind, &request.scope)
    }

    pub fn from_plugin_scope(kind: CapabilityKind, scope: &str) -> Result<Self, PermissionError> {
        validate_scope(scope).map_err(|()| PermissionError::InvalidPluginCapabilityScope {
            kind,
            scope: scope.to_owned(),
        })?;
        let capability = match kind {
            CapabilityKind::Filesystem => Self::Asset {
                namespace: crate::AssetNamespace::from_plugin_root(scope)
                    .map_err(|_| PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    })?
                    .locator_type()
                    .to_owned(),
                action: AssetOperation::Read,
            },
            CapabilityKind::NetworkProvider => {
                let (provider, endpoint) = scope.split_once('|').ok_or_else(|| {
                    PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    }
                })?;
                if endpoint.contains('|') {
                    return Err(PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    });
                }
                let endpoint = crate::ProviderEndpoint::new(provider, endpoint).map_err(|_| {
                    PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    }
                })?;
                Self::ProviderNetwork {
                    provider: endpoint.provider().as_str().to_owned(),
                    endpoint: endpoint.endpoint().to_owned(),
                }
            }
            CapabilityKind::ProviderUpload | CapabilityKind::ProviderCost => {
                let (provider, endpoint) = scope.split_once('|').ok_or_else(|| {
                    PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    }
                })?;
                if endpoint.contains('|') {
                    return Err(PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    });
                }
                let endpoint = crate::ProviderEndpoint::new(provider, endpoint).map_err(|_| {
                    PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    }
                })?;
                if kind == CapabilityKind::ProviderUpload {
                    Self::ProviderUpload {
                        provider: endpoint.provider().as_str().to_owned(),
                        endpoint: endpoint.endpoint().to_owned(),
                    }
                } else {
                    Self::ProviderCost {
                        provider: endpoint.provider().as_str().to_owned(),
                        endpoint: endpoint.endpoint().to_owned(),
                    }
                }
            }
            CapabilityKind::Secret => Self::Secret {
                secret_id: crate::SecretId::new(scope)
                    .map_err(|_| PermissionError::InvalidPluginCapabilityScope {
                        kind,
                        scope: scope.to_owned(),
                    })?
                    .as_str()
                    .to_owned(),
            },
            CapabilityKind::Clock => Self::Clock {
                clock_id: scope.to_owned(),
            },
            CapabilityKind::Randomness => Self::Randomness {
                stream_id: scope.to_owned(),
            },
            CapabilityKind::Model => Self::ModelHandle {
                model_id: scope.to_owned(),
            },
            CapabilityKind::TransactionalOutput => Self::TransactionalOutput {
                namespace: scope.to_owned(),
            },
            CapabilityKind::SanitizedLog => Self::SanitizedLog {
                level: scope.to_owned(),
            },
            CapabilityKind::DeclarativeUi => Self::DeclarativeUi {
                contribution_id: scope.to_owned(),
            },
            CapabilityKind::Route => Self::NativeRoute {
                route_id: scope.to_owned(),
            },
        };
        capability.validate()?;
        Ok(capability)
    }

    pub fn plugin_capability_key(&self) -> Option<(CapabilityKind, String)> {
        match self {
            Self::Asset {
                namespace,
                action: AssetOperation::Read,
            } => Some((CapabilityKind::Filesystem, namespace.clone())),
            Self::ProviderNetwork { provider, endpoint } => Some((
                CapabilityKind::NetworkProvider,
                format!("{provider}|{endpoint}"),
            )),
            Self::ProviderUpload { provider, endpoint } => Some((
                CapabilityKind::ProviderUpload,
                format!("{provider}|{endpoint}"),
            )),
            Self::ProviderCost { provider, endpoint } => Some((
                CapabilityKind::ProviderCost,
                format!("{provider}|{endpoint}"),
            )),
            Self::Secret { secret_id } => Some((CapabilityKind::Secret, secret_id.clone())),
            Self::Clock { clock_id } => Some((CapabilityKind::Clock, clock_id.clone())),
            Self::Randomness { stream_id } => Some((CapabilityKind::Randomness, stream_id.clone())),
            Self::ModelHandle { model_id } => Some((CapabilityKind::Model, model_id.clone())),
            Self::TransactionalOutput { namespace } => {
                Some((CapabilityKind::TransactionalOutput, namespace.clone()))
            }
            Self::SanitizedLog { level } => Some((CapabilityKind::SanitizedLog, level.clone())),
            Self::DeclarativeUi { contribution_id } => {
                Some((CapabilityKind::DeclarativeUi, contribution_id.clone()))
            }
            Self::NativeRoute { route_id } => Some((CapabilityKind::Route, route_id.clone())),
            Self::Asset { .. }
            | Self::Codec { .. }
            | Self::ExternalNavigation
            | Self::NativeFfi { .. } => None,
        }
    }

    pub fn wire_identifier(&self) -> String {
        match self {
            Self::Asset { namespace, action } => {
                format!("asset:{}:{namespace}", action.wire_name())
            }
            Self::ProviderNetwork { provider, endpoint } => {
                format!("provider_network:{provider}|{endpoint}")
            }
            Self::ProviderUpload { provider, endpoint } => {
                format!("provider_upload:{provider}|{endpoint}")
            }
            Self::ProviderCost { provider, endpoint } => {
                format!("provider_cost:{provider}|{endpoint}")
            }
            Self::Secret { secret_id } => format!("secret:{secret_id}"),
            Self::Clock { clock_id } => format!("clock:{clock_id}"),
            Self::Randomness { stream_id } => format!("randomness:{stream_id}"),
            Self::ModelHandle { model_id } => format!("model_handle:{model_id}"),
            Self::TransactionalOutput { namespace } => {
                format!("transactional_output:{namespace}")
            }
            Self::SanitizedLog { level } => format!("sanitized_log:{level}"),
            Self::DeclarativeUi { contribution_id } => {
                format!("declarative_ui:{contribution_id}")
            }
            Self::NativeRoute { route_id } => format!("native_route:{route_id}"),
            Self::Codec { codec_id } => format!("codec:{codec_id}"),
            Self::ExternalNavigation => "external_navigation".to_owned(),
            Self::NativeFfi { library_id } => format!("native_ffi:{library_id}"),
        }
    }

    pub fn parse_wire_identifier(value: &str) -> Result<Self, PermissionError> {
        if value == "external_navigation" {
            return Ok(Self::ExternalNavigation);
        }
        let (kind, scope) = value
            .split_once(':')
            .ok_or_else(|| PermissionError::InvalidCapabilityWireIdentifier(value.to_owned()))?;
        validate_scope(scope)
            .map_err(|()| PermissionError::InvalidCapabilityWireIdentifier(value.to_owned()))?;
        let capability = match kind {
            "asset" => {
                let (action, namespace) = scope.split_once(':').ok_or_else(|| {
                    PermissionError::InvalidCapabilityWireIdentifier(value.to_owned())
                })?;
                validate_scope(namespace).map_err(|()| {
                    PermissionError::InvalidCapabilityWireIdentifier(value.to_owned())
                })?;
                Self::Asset {
                    namespace: namespace.to_owned(),
                    action: AssetOperation::parse_wire_name(action).ok_or_else(|| {
                        PermissionError::InvalidCapabilityWireIdentifier(value.to_owned())
                    })?,
                }
            }
            "provider_network" => Self::from_plugin_scope(CapabilityKind::NetworkProvider, scope)
                .map_err(|_| {
                PermissionError::InvalidCapabilityWireIdentifier(value.to_owned())
            })?,
            "provider_upload" => Self::from_plugin_scope(CapabilityKind::ProviderUpload, scope)
                .map_err(|_| PermissionError::InvalidCapabilityWireIdentifier(value.to_owned()))?,
            "provider_cost" => Self::from_plugin_scope(CapabilityKind::ProviderCost, scope)
                .map_err(|_| PermissionError::InvalidCapabilityWireIdentifier(value.to_owned()))?,
            "secret" => Self::Secret {
                secret_id: scope.to_owned(),
            },
            "clock" => Self::Clock {
                clock_id: scope.to_owned(),
            },
            "randomness" => Self::Randomness {
                stream_id: scope.to_owned(),
            },
            "model_handle" => Self::ModelHandle {
                model_id: scope.to_owned(),
            },
            "transactional_output" => Self::TransactionalOutput {
                namespace: scope.to_owned(),
            },
            "sanitized_log" => Self::SanitizedLog {
                level: scope.to_owned(),
            },
            "declarative_ui" => Self::DeclarativeUi {
                contribution_id: scope.to_owned(),
            },
            "native_route" => Self::NativeRoute {
                route_id: scope.to_owned(),
            },
            "codec" => Self::Codec {
                codec_id: scope.to_owned(),
            },
            "native_ffi" => Self::NativeFfi {
                library_id: scope.to_owned(),
            },
            _ => {
                return Err(PermissionError::InvalidCapabilityWireIdentifier(
                    value.to_owned(),
                ));
            }
        };
        capability.validate()?;
        Ok(capability)
    }

    fn validate(&self) -> Result<(), PermissionError> {
        if let Self::ProviderNetwork { provider, endpoint }
        | Self::ProviderUpload { provider, endpoint }
        | Self::ProviderCost { provider, endpoint } = self
        {
            return crate::ProviderEndpoint::new(provider, endpoint)
                .map(|_| ())
                .map_err(|_| PermissionError::InvalidCapability(self.clone()));
        }
        if let Self::Secret { secret_id } = self {
            return crate::SecretId::new(secret_id)
                .map(|_| ())
                .map_err(|_| PermissionError::InvalidCapability(self.clone()));
        }
        let scopes: &[&str] = match self {
            Self::Asset { namespace, .. } | Self::TransactionalOutput { namespace } => &[namespace],
            Self::ProviderNetwork { .. }
            | Self::ProviderUpload { .. }
            | Self::ProviderCost { .. }
            | Self::Secret { .. } => &[],
            Self::Clock { clock_id } => &[clock_id],
            Self::Randomness { stream_id } => &[stream_id],
            Self::ModelHandle { model_id } => &[model_id],
            Self::SanitizedLog { level } => &[level],
            Self::DeclarativeUi { contribution_id } => &[contribution_id],
            Self::NativeRoute { route_id } => &[route_id],
            Self::Codec { codec_id } => &[codec_id],
            Self::NativeFfi { library_id } => &[library_id],
            Self::ExternalNavigation => &[],
        };
        if scopes.iter().any(|scope| validate_scope(scope).is_err()) {
            return Err(PermissionError::InvalidCapability(self.clone()));
        }
        Ok(())
    }
}

impl TryFrom<&str> for Capability {
    type Error = PermissionError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::parse_wire_identifier(value)
    }
}

fn validate_scope(scope: &str) -> Result<(), ()> {
    if scope.is_empty()
        || scope != scope.trim()
        || scope.len() > 1_024
        || scope.bytes().any(|byte| byte.is_ascii_control())
    {
        Err(())
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilitySet(BTreeSet<Capability>);

impl CapabilitySet {
    pub fn new(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        Self(capabilities.into_iter().collect())
    }

    pub fn contains(&self, capability: &Capability) -> bool {
        self.0.contains(capability)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Capability> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn require(&self, capability: &Capability) -> Result<(), PermissionDenied> {
        if self.contains(capability) {
            Ok(())
        } else {
            Err(PermissionDenied {
                capability: capability.clone(),
            })
        }
    }

    pub fn denied_from(&self, granted: &Self) -> Vec<Capability> {
        self.0.difference(&granted.0).cloned().collect()
    }

    fn validate(&self) -> Result<(), PermissionError> {
        for capability in &self.0 {
            capability.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[error("capability denied: {capability:?}")]
pub struct PermissionDenied {
    pub capability: Capability,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermissionGrant {
    profile_id: String,
    subject_id: String,
    capabilities: CapabilitySet,
    provenance: String,
    grant_version: u16,
}

impl PermissionGrant {
    pub fn new(
        profile_id: impl Into<String>,
        subject_id: impl Into<String>,
        capabilities: CapabilitySet,
        provenance: impl Into<String>,
    ) -> Result<Self, PermissionError> {
        let grant = Self {
            profile_id: profile_id.into(),
            subject_id: subject_id.into(),
            capabilities,
            provenance: provenance.into(),
            grant_version: PERMISSION_GRANT_VERSION,
        };
        grant.validate()?;
        Ok(grant)
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    pub fn grant_version(&self) -> u16 {
        self.grant_version
    }

    fn validate(&self) -> Result<(), PermissionError> {
        if !valid_permission_identifier(&self.profile_id) {
            return Err(PermissionError::InvalidProfile);
        }
        if !valid_permission_identifier(&self.subject_id) {
            return Err(PermissionError::InvalidSubject);
        }
        if self.provenance.trim().is_empty() {
            return Err(PermissionError::InvalidProvenance);
        }
        if self.grant_version != PERMISSION_GRANT_VERSION {
            return Err(PermissionError::UnsupportedGrantVersion(self.grant_version));
        }
        self.capabilities.validate()?;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorizedCapabilities {
    profile_id: String,
    subject_id: String,
    policy_generation: PermissionPolicyGeneration,
    capabilities: CapabilitySet,
    provenance: String,
    grant_version: u16,
}

impl AuthorizedCapabilities {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    pub const fn policy_generation(&self) -> PermissionPolicyGeneration {
        self.policy_generation
    }

    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    pub fn grant_version(&self) -> u16 {
        self.grant_version
    }

    pub fn require(&self, capability: &Capability) -> Result<(), PermissionDenied> {
        self.capabilities.require(capability)
    }

    pub(crate) fn validate_sealed(&self) -> Result<(), PermissionError> {
        if !valid_permission_identifier(&self.profile_id) {
            return Err(PermissionError::InvalidProfile);
        }
        if !valid_permission_identifier(&self.subject_id) {
            return Err(PermissionError::InvalidSubject);
        }
        if self.provenance.trim().is_empty() {
            return Err(PermissionError::InvalidProvenance);
        }
        if self.grant_version != PERMISSION_GRANT_VERSION {
            return Err(PermissionError::UnsupportedGrantVersion(self.grant_version));
        }
        self.capabilities.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PermissionPolicy {
    profile_id: String,
    generation: PermissionPolicyGeneration,
    grants: BTreeMap<String, PermissionGrant>,
}

impl PermissionPolicy {
    pub fn native_runtime_services(profile_id: impl Into<String>) -> Result<Self, PermissionError> {
        let profile_id = profile_id.into();
        let input_reader = PermissionGrant::new(
            profile_id.clone(),
            NATIVE_INPUT_READER_SUBJECT,
            native_input_reader_capabilities(),
            "native-runtime-profile",
        )?;
        let api_asset_reader = PermissionGrant::new(
            profile_id.clone(),
            NATIVE_API_ASSET_READER_SUBJECT,
            native_api_asset_reader_capabilities(),
            "native-runtime-profile",
        )?;
        let image_importer = PermissionGrant::new(
            profile_id.clone(),
            NATIVE_IMAGE_IMPORTER_SUBJECT,
            native_image_importer_capabilities(),
            "native-runtime-profile",
        )?;
        let image_output_reader = PermissionGrant::new(
            profile_id.clone(),
            NATIVE_IMAGE_OUTPUT_READER_SUBJECT,
            native_image_output_reader_capabilities(),
            "native-runtime-profile",
        )?;
        let output_committer = PermissionGrant::new(
            profile_id.clone(),
            OUTPUT_COMMITTER_SUBJECT,
            output_committer_capabilities(),
            "native-runtime-profile",
        )?;
        let output_ui = PermissionGrant::new(
            profile_id.clone(),
            OUTPUT_UI_SUBJECT,
            output_ui_capabilities(),
            "native-runtime-profile",
        )?;
        let plugin_asset_broker = PermissionGrant::new(
            profile_id.clone(),
            PLUGIN_ASSET_BROKER_SUBJECT,
            plugin_asset_broker_capabilities(),
            "native-runtime-profile",
        )?;
        let subgraph_library = PermissionGrant::new(
            profile_id.clone(),
            SUBGRAPH_LIBRARY_SUBJECT,
            subgraph_library_capabilities(),
            "native-runtime-profile",
        )?;
        Self::new(
            profile_id,
            [
                input_reader,
                image_importer,
                image_output_reader,
                api_asset_reader,
                output_committer,
                output_ui,
                plugin_asset_broker,
                subgraph_library,
            ],
        )
    }

    pub fn new(
        profile_id: impl Into<String>,
        grants: impl IntoIterator<Item = PermissionGrant>,
    ) -> Result<Self, PermissionError> {
        Self::new_with_generation(profile_id, PermissionPolicyGeneration::new(1)?, grants)
    }

    pub fn new_with_generation(
        profile_id: impl Into<String>,
        generation: PermissionPolicyGeneration,
        grants: impl IntoIterator<Item = PermissionGrant>,
    ) -> Result<Self, PermissionError> {
        let profile_id = profile_id.into();
        if !valid_permission_identifier(&profile_id) {
            return Err(PermissionError::InvalidProfile);
        }
        let mut checked_grants = BTreeMap::new();
        for grant in grants {
            grant.validate()?;
            if grant.profile_id != profile_id {
                return Err(PermissionError::ProfileMismatch);
            }
            let subject_id = grant.subject_id.clone();
            if checked_grants.insert(subject_id.clone(), grant).is_some() {
                return Err(PermissionError::DuplicateSubject(subject_id));
            }
        }
        Ok(Self {
            profile_id,
            generation,
            grants: checked_grants,
        })
    }

    pub fn with_additional_grants(
        mut self,
        grants: impl IntoIterator<Item = PermissionGrant>,
    ) -> Result<Self, PermissionError> {
        for grant in grants {
            grant.validate()?;
            if grant.profile_id != self.profile_id {
                return Err(PermissionError::ProfileMismatch);
            }
            let subject_id = grant.subject_id.clone();
            if self.grants.insert(subject_id.clone(), grant).is_some() {
                return Err(PermissionError::DuplicateSubject(subject_id));
            }
        }
        Ok(self)
    }

    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub const fn generation(&self) -> PermissionPolicyGeneration {
        self.generation
    }

    pub fn with_generation(mut self, generation: PermissionPolicyGeneration) -> Self {
        self.generation = generation;
        self
    }

    pub fn authorize(
        &self,
        subject_id: &str,
        requested: &CapabilitySet,
    ) -> Result<AuthorizedCapabilities, PermissionError> {
        let grant = self
            .grants
            .get(subject_id)
            .ok_or(PermissionError::UnknownSubject)?;
        let denied = requested.denied_from(&grant.capabilities);
        if !denied.is_empty() {
            return Err(PermissionError::Denied(denied));
        }
        Ok(AuthorizedCapabilities {
            profile_id: self.profile_id.clone(),
            subject_id: subject_id.to_owned(),
            policy_generation: self.generation,
            capabilities: requested.clone(),
            provenance: grant.provenance.clone(),
            grant_version: grant.grant_version,
        })
    }

    pub fn authorize_one(
        &self,
        subject_id: &str,
        capability: Capability,
    ) -> Result<AuthorizedCapabilities, PermissionError> {
        self.authorize(subject_id, &CapabilitySet::new([capability]))
    }
}

fn valid_permission_identifier(value: &str) -> bool {
    if value.is_empty() || value.len() > 1_024 || !value.is_ascii() {
        return false;
    }
    let bytes = value.as_bytes();
    if !bytes.first().is_some_and(u8::is_ascii_alphanumeric)
        || !bytes.last().is_some_and(u8::is_ascii_alphanumeric)
    {
        return false;
    }
    let mut previous_separator = false;
    for byte in bytes {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() {
            previous_separator = false;
        } else if matches!(byte, b'.' | b'-' | b'_') && !previous_separator {
            previous_separator = true;
        } else {
            return false;
        }
    }
    true
}

pub fn authorize_native_input_reader(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(
        NATIVE_INPUT_READER_SUBJECT,
        &native_input_reader_capabilities(),
    )
}

pub fn authorize_native_image_importer(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(
        NATIVE_IMAGE_IMPORTER_SUBJECT,
        &native_image_importer_capabilities(),
    )
}

pub fn authorize_native_image_output_reader(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(
        NATIVE_IMAGE_OUTPUT_READER_SUBJECT,
        &native_image_output_reader_capabilities(),
    )
}

pub fn authorize_native_api_asset_reader(
    policy: &PermissionPolicy,
) -> Result<AuthorizedCapabilities, PermissionError> {
    policy.authorize(
        NATIVE_API_ASSET_READER_SUBJECT,
        &native_api_asset_reader_capabilities(),
    )
}

pub fn authorize_native_plugin_asset_broker(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    authorize_native_plugin_asset_broker_from_policy(&policy)
}

pub fn authorize_native_plugin_asset_broker_from_policy(
    policy: &PermissionPolicy,
) -> Result<AuthorizedCapabilities, PermissionError> {
    policy.authorize(
        PLUGIN_ASSET_BROKER_SUBJECT,
        &plugin_asset_broker_capabilities(),
    )
}

pub fn authorize_native_output_committer(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(OUTPUT_COMMITTER_SUBJECT, &output_committer_capabilities())
}

pub fn authorize_native_output_ui(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(OUTPUT_UI_SUBJECT, &output_ui_capabilities())
}

pub fn authorize_native_subgraph_library(
    profile_id: impl Into<String>,
) -> Result<AuthorizedCapabilities, PermissionError> {
    let policy = PermissionPolicy::native_runtime_services(profile_id)?;
    policy.authorize(SUBGRAPH_LIBRARY_SUBJECT, &subgraph_library_capabilities())
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PermissionError {
    #[error("permission policy generation must be nonzero")]
    InvalidPolicyGeneration,
    #[error("permission profile is invalid")]
    InvalidProfile,
    #[error("permission subject is invalid")]
    InvalidSubject,
    #[error("permission grant provenance is invalid")]
    InvalidProvenance,
    #[error("permission grant version {0} is unsupported")]
    UnsupportedGrantVersion(u16),
    #[error("permission capability has an invalid scope: {0:?}")]
    InvalidCapability(Capability),
    #[error("plugin capability {kind:?} has an invalid scope: {scope}")]
    InvalidPluginCapabilityScope { kind: CapabilityKind, scope: String },
    #[error("capability wire identifier is invalid: {0}")]
    InvalidCapabilityWireIdentifier(String),
    #[error("permission grant belongs to a different runtime profile")]
    ProfileMismatch,
    #[error("permission subject is duplicated: {0}")]
    DuplicateSubject(String),
    #[error("permission subject has no grant")]
    UnknownSubject,
    #[error("capabilities are not granted: {0:?}")]
    Denied(Vec<Capability>),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> Result<PermissionPolicy, PermissionError> {
        PermissionPolicy::new(
            "profile-a",
            [PermissionGrant::new(
                "profile-a",
                "plugin-a",
                CapabilitySet::new([
                    Capability::Asset {
                        namespace: "input-root".to_owned(),
                        action: AssetOperation::Read,
                    },
                    Capability::ModelHandle {
                        model_id: "model-root".to_owned(),
                    },
                ]),
                "approved-settings",
            )?],
        )
    }

    #[test]
    fn grants_are_profile_and_subject_scoped() -> Result<(), PermissionError> {
        let grant = PermissionGrant::new(
            "profile-b",
            "plugin-a",
            CapabilitySet::new([Capability::ModelHandle {
                model_id: "model-root".to_owned(),
            }]),
            "approved-settings",
        )?;
        assert_eq!(
            PermissionPolicy::new("profile-a", [grant]),
            Err(PermissionError::ProfileMismatch)
        );
        assert_eq!(
            policy()?.authorize_one(
                "plugin-b",
                Capability::ModelHandle {
                    model_id: "model-root".to_owned(),
                }
            ),
            Err(PermissionError::UnknownSubject)
        );
        Ok(())
    }

    #[test]
    fn authorization_seals_only_the_requested_subset() -> Result<(), PermissionError> {
        let read_model = Capability::ModelHandle {
            model_id: "model-root".to_owned(),
        };
        let read_input = Capability::Asset {
            namespace: "input-root".to_owned(),
            action: AssetOperation::Read,
        };
        let authorization = policy()?.authorize_one("plugin-a", read_model.clone())?;
        assert_eq!(authorization.profile_id(), "profile-a");
        assert_eq!(authorization.subject_id(), "plugin-a");
        assert_eq!(authorization.capabilities().len(), 1);
        assert!(authorization.require(&read_model).is_ok());
        assert_eq!(
            authorization.require(&read_input),
            Err(PermissionDenied {
                capability: read_input
            })
        );
        Ok(())
    }

    #[test]
    fn undeclared_capability_fails_before_authorization() -> Result<(), PermissionError> {
        assert_eq!(
            policy()?.authorize_one(
                "plugin-a",
                Capability::Asset {
                    namespace: "output".to_owned(),
                    action: AssetOperation::Write,
                }
            ),
            Err(PermissionError::Denied(vec![Capability::Asset {
                namespace: "output".to_owned(),
                action: AssetOperation::Write,
            }]))
        );
        Ok(())
    }

    #[test]
    fn scoped_capability_identifiers_are_checked_before_grant() {
        assert_eq!(
            PermissionGrant::new(
                "profile-a",
                "plugin-a",
                CapabilitySet::new([Capability::ProviderNetwork {
                    provider: String::new(),
                    endpoint: "https://provider.invalid/v1/generate".to_owned(),
                }]),
                "approved-settings",
            ),
            Err(PermissionError::InvalidCapability(
                Capability::ProviderNetwork {
                    provider: String::new(),
                    endpoint: "https://provider.invalid/v1/generate".to_owned(),
                }
            ))
        );
    }

    #[test]
    fn every_plugin_capability_kind_preserves_its_exact_runtime_scope()
    -> Result<(), PermissionError> {
        let cases = [
            (CapabilityKind::Filesystem, "input"),
            (
                CapabilityKind::NetworkProvider,
                "provider.demo|https://provider.invalid/v1/generate",
            ),
            (CapabilityKind::Secret, "secret.demo"),
            (CapabilityKind::Clock, "clock.demo"),
            (CapabilityKind::Randomness, "random.demo"),
            (CapabilityKind::Model, "model.demo"),
            (CapabilityKind::TransactionalOutput, "output"),
            (CapabilityKind::SanitizedLog, "info"),
            (CapabilityKind::DeclarativeUi, "ui.demo"),
            (CapabilityKind::Route, "route.demo"),
        ];
        for (kind, scope) in cases {
            let request = CapabilityRequest {
                kind,
                scope: scope.to_owned(),
                quota: sim_plugin_sdk::CapabilityQuota {
                    maximum_operations: 1,
                    maximum_request_bytes: 1,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 1,
                    maximum_handles: 1,
                    timeout_milliseconds: 1,
                },
            };
            let capability = Capability::from_plugin_request(&request)?;
            assert_eq!(
                capability.plugin_capability_key(),
                Some((request.kind, request.scope.clone()))
            );
            assert_eq!(
                Capability::parse_wire_identifier(&capability.wire_identifier())?,
                capability
            );
        }
        Ok(())
    }

    #[test]
    fn plugin_network_capability_maps_to_provider_domain_type() -> Result<(), PermissionError> {
        let request = CapabilityRequest {
            kind: CapabilityKind::NetworkProvider,
            scope: "provider.demo|https://provider.invalid/v1/generate".to_owned(),
            quota: sim_plugin_sdk::CapabilityQuota {
                maximum_operations: 1,
                maximum_request_bytes: 1,
                maximum_response_bytes: 1,
                maximum_total_bytes: 1,
                maximum_handles: 1,
                timeout_milliseconds: 1,
            },
        };
        let capability = Capability::from_plugin_request(&request)?;
        assert_eq!(
            capability,
            Capability::ProviderNetwork {
                provider: "provider.demo".to_owned(),
                endpoint: "https://provider.invalid/v1/generate".to_owned(),
            }
        );
        assert_eq!(
            capability.plugin_capability_key(),
            Some((CapabilityKind::NetworkProvider, request.scope))
        );
        assert_eq!(
            Capability::parse_wire_identifier(&capability.wire_identifier())?,
            capability
        );
        Ok(())
    }

    #[test]
    fn malformed_plugin_and_wire_scopes_fail_closed() {
        for scope in [
            "provider-only",
            "|/v1/generate",
            "provider|",
            "provider|/v1/generate|extra",
            " provider|/v1/generate",
        ] {
            assert!(
                Capability::from_plugin_scope(CapabilityKind::NetworkProvider, scope).is_err(),
                "accepted malformed scope {scope:?}"
            );
        }
        for identifier in [
            "",
            "asset:read:",
            "asset:execute:input",
            "unknown:value",
            "external_navigation:anything",
            "provider_network:provider-only",
            "secret: secret",
        ] {
            assert!(
                Capability::parse_wire_identifier(identifier).is_err(),
                "accepted malformed identifier {identifier:?}"
            );
        }
    }

    #[test]
    fn native_service_policy_is_fixed_and_least_privilege() -> Result<(), PermissionError> {
        let policy = PermissionPolicy::native_runtime_services("profile-a")?;
        let output_write = Capability::Asset {
            namespace: "output".to_owned(),
            action: AssetOperation::Write,
        };
        let output_read = Capability::Asset {
            namespace: "output".to_owned(),
            action: AssetOperation::Read,
        };
        let output_delete = Capability::Asset {
            namespace: "output".to_owned(),
            action: AssetOperation::Delete,
        };
        let temporary_write = Capability::Asset {
            namespace: "temp".to_owned(),
            action: AssetOperation::Write,
        };
        assert!(
            policy
                .authorize_one(OUTPUT_COMMITTER_SUBJECT, output_write.clone())
                .is_ok()
        );
        assert!(
            policy
                .authorize_one(OUTPUT_COMMITTER_SUBJECT, output_read.clone())
                .is_err()
        );
        assert!(
            policy
                .authorize_one(OUTPUT_COMMITTER_SUBJECT, output_delete.clone())
                .is_err()
        );
        assert!(policy.authorize_one(OUTPUT_UI_SUBJECT, output_read).is_ok());
        assert!(
            policy
                .authorize_one(OUTPUT_UI_SUBJECT, output_delete)
                .is_ok()
        );
        assert!(
            policy
                .authorize_one(OUTPUT_UI_SUBJECT, output_write)
                .is_err()
        );
        assert!(
            policy
                .authorize_one(OUTPUT_UI_SUBJECT, temporary_write)
                .is_ok()
        );
        assert_eq!(
            authorize_native_output_committer("profile-a")?
                .capabilities()
                .len(),
            2
        );
        assert_eq!(
            authorize_native_output_ui("profile-a")?
                .capabilities()
                .len(),
            5
        );
        let subgraph_library = authorize_native_subgraph_library("profile-a")?;
        assert_eq!(subgraph_library.capabilities().len(), 2);
        assert!(
            subgraph_library
                .require(&Capability::Asset {
                    namespace: "plugin".to_owned(),
                    action: AssetOperation::Read,
                })
                .is_ok()
        );
        assert!(
            subgraph_library
                .require(&Capability::Asset {
                    namespace: "output".to_owned(),
                    action: AssetOperation::Read,
                })
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn native_image_importer_cannot_read_or_modify_other_asset_roots() -> Result<(), PermissionError>
    {
        let policy = PermissionPolicy::native_runtime_services("profile-a")?;
        let authorization = authorize_native_image_importer("profile-a")?;
        assert_eq!(authorization.capabilities().len(), 1);
        for namespace in ["input", "output", "temp", "model", "plugin"] {
            for action in [
                AssetOperation::Read,
                AssetOperation::Write,
                AssetOperation::Rename,
                AssetOperation::Tag,
                AssetOperation::Delete,
            ] {
                let allowed = namespace == "input" && action == AssetOperation::Write;
                let capability = Capability::Asset {
                    namespace: namespace.to_owned(),
                    action,
                };
                assert_eq!(
                    policy
                        .authorize_one(NATIVE_IMAGE_IMPORTER_SUBJECT, capability.clone())
                        .is_ok(),
                    allowed
                );
                assert_eq!(authorization.require(&capability).is_ok(), allowed);
            }
        }
        assert!(
            authorization
                .require(&Capability::ExternalNavigation)
                .is_err()
        );
        assert!(authorize_native_image_importer("").is_err());
        Ok(())
    }

    #[test]
    fn native_image_output_reader_has_no_input_or_mutation_authority() -> Result<(), PermissionError>
    {
        let policy = PermissionPolicy::native_runtime_services("profile-a")?;
        let authorization = authorize_native_image_output_reader("profile-a")?;
        assert_eq!(authorization.capabilities().len(), 2);
        for namespace in ["input", "output", "temp", "model", "plugin"] {
            for action in [
                AssetOperation::Read,
                AssetOperation::Write,
                AssetOperation::Rename,
                AssetOperation::Tag,
                AssetOperation::Delete,
            ] {
                let allowed =
                    matches!(namespace, "output" | "temp") && action == AssetOperation::Read;
                let capability = Capability::Asset {
                    namespace: namespace.to_owned(),
                    action,
                };
                assert_eq!(
                    policy
                        .authorize_one(NATIVE_IMAGE_OUTPUT_READER_SUBJECT, capability.clone())
                        .is_ok(),
                    allowed
                );
                assert_eq!(authorization.require(&capability).is_ok(), allowed);
            }
        }
        assert!(authorize_native_image_output_reader("").is_err());
        Ok(())
    }
}
