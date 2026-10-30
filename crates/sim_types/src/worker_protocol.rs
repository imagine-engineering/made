use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use thiserror::Error;
use uuid::Uuid;

use crate::{AttemptId, CancellationToken, DeviceKind, PromptId};

pub const WORKER_PROTOCOL_VERSION: u16 = 8;
pub const PREVIOUS_WORKER_PROTOCOL_VERSION: u16 = 7;
pub const LEGACY_WORKER_PROTOCOL_VERSION: u16 = 6;
pub const WORKER_REGISTRY_DEPLOYMENT_REJECTION_VERSION: u16 = 1;
pub const MAX_WORKER_FRAME_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_ENCODED_PREVIEW_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_WORKER_OUTPUT_METADATA_BYTES: usize = 64 * 1024;
pub const MAX_WORKER_OUTPUT_CONTENT_BYTES: usize = 12 * 1024 * 1024;
pub const MAX_WORKER_BACKEND_CAPABILITY_ENTRIES: usize = 4_096;
pub const MAX_WORKER_DEVICE_PROPERTY_BYTES: usize = 256;
pub const MAX_WORKER_COMPONENT_COUNT: usize = 256;
pub const MAX_WORKER_COMPONENT_IDENTITY_BYTES: usize = 256;
pub const MAX_WORKER_COMPONENT_VERSION_BYTES: usize = 128;
pub const MAX_WORKER_COMPONENT_MANIFEST_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_WORKER_COMPONENT_AUTHORIZATION_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_WORKER_COMPONENT_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_WORKER_COMPONENT_CHUNK_BYTES: usize = 1024 * 1024;
pub const MAX_WORKER_REGISTRY_DEPLOYMENT_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_WORKER_PLUGIN_INVOCATION_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES: usize = 72 * 1024 * 1024;
pub const MAX_WORKER_PLUGIN_RESULT_BYTES: usize = 12 * 1024 * 1024;
pub const MAX_WORKER_PLUGIN_DIAGNOSTIC_CHARS: usize = 4_096;
pub const MAX_WORKER_FATAL_CODE_BYTES: usize = 128;
pub const MAX_WORKER_FATAL_MESSAGE_BYTES: usize = 4_096;
pub const MAX_WORKER_PROVIDER_HEADERS: usize = 256;
pub const MAX_WORKER_PROVIDER_HEADER_NAME_BYTES: usize = 256;
pub const MAX_WORKER_PROVIDER_HEADER_VALUE_BYTES: usize = 8 * 1024;
pub const MAX_WORKER_PROVIDER_HEADER_BYTES: usize = 256 * 1024;
pub const MAX_WORKER_PROVIDER_BODY_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_WORKER_PROVIDER_CHUNK_BYTES: usize = 1024 * 1024;
pub const MAX_WORKER_PROVIDER_NDJSON_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_WORKER_PROVIDER_WAIT_MILLISECONDS: u64 = 60_000;
pub const MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS: u64 = 150_000;
pub const MAX_WORKER_PROVIDER_ATTEMPTS: u32 = 17;
pub const MAX_WORKER_PROVIDER_UPLOADS: u32 = 128;
pub const MAX_WORKER_PROVIDER_COST_REQUESTS: u32 = 64;
pub const MAX_WORKER_PROVIDER_PROGRESS_TOTAL: u64 = 1_000_000_000;
pub const MAX_WORKER_PROVIDER_PROGRESS_MESSAGE_BYTES: usize = 1024;
pub const MAX_WORKER_PROVIDER_RECEIPT_BYTES: usize = 32 * 1024;
pub const MAX_WORKER_PROVIDER_PENDING_CALLS: usize = 1;
pub const MAX_WORKER_PROVIDER_HTTP_REQUESTS: u32 = 8_192;
pub const MAX_WORKER_PROVIDER_ENDPOINT_BYTES: usize = 2_048;
pub const MAX_WORKER_PROVIDER_UPLOAD_URL_BYTES: usize = 2_048;
pub const MAX_WORKER_PROVIDER_SECRET_ID_BYTES: usize = 1_024;
pub const MAX_WORKER_PROVIDER_FINALIZATION_IDENTITY_BYTES: usize = 64;
pub const MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES: usize = 512 * 1024;
pub const MAX_WORKER_MODEL_SOURCE_SELECTIONS: usize = 4;
pub const MAX_WORKER_MODEL_SOURCE_NAME_BYTES: usize = 4 * 1024;
pub const MAX_WORKER_MODEL_SOURCE_CATEGORY_BYTES: usize = 128;
pub const MAX_WORKER_MODEL_SOURCE_NODE_ID_BYTES: usize = 256;
pub const MAX_WORKER_MODEL_SOURCE_ARTIFACTS: usize = 1_024;
pub const MAX_WORKER_MODEL_SOURCE_TENSORS: usize = 16_384;
pub const MAX_WORKER_MODEL_SOURCE_TENSOR_DIMENSIONS: usize = 64;
pub const MAX_WORKER_MODEL_SOURCE_DTYPE_BYTES: usize = 64;
pub const MAX_WORKER_MODEL_SOURCE_MANIFEST_WIRE_BYTES: usize = 12 * 1024 * 1024;
pub const WORKER_OPERATION_SUPPORT_VERSION: u16 = 2;
pub const LEGACY_WORKER_OPERATION_SUPPORT_VERSION: u16 = 1;
pub const WORKER_REGISTRY_DIGEST_DOMAIN: &[u8] = b"zed-sim-worker-registry-v1";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerRegistryGeneration(NonZeroU64);

impl WorkerRegistryGeneration {
    pub fn new(value: u64) -> Result<Self, WorkerComponentDeploymentError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(WorkerComponentDeploymentError::ZeroGeneration)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct WorkerSha256Digest(String);

impl WorkerSha256Digest {
    pub fn new(value: impl Into<String>) -> Result<Self, WorkerComponentDeploymentError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(WorkerComponentDeploymentError::InvalidSha256);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn bytes(&self) -> [u8; 32] {
        let mut digest = [0_u8; 32];
        for (index, pair) in self.0.as_bytes().chunks_exact(2).enumerate() {
            digest[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
        }
        digest
    }
}

impl<'de> Deserialize<'de> for WorkerSha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl TryFrom<String> for WorkerSha256Digest {
    type Error = WorkerComponentDeploymentError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerComponentContent {
    Manifest,
    Authorization,
    Component,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerComponentIdentityField {
    ExtensionId,
    ExtensionVersion,
    PluginIdentifier,
    PluginVersion,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerComponentDescriptor {
    extension_id: String,
    extension_version: String,
    plugin_identifier: String,
    plugin_version: String,
    authorization_generation: WorkerSha256Digest,
    manifest_digest_sha256: WorkerSha256Digest,
    component_digest_sha256: WorkerSha256Digest,
    manifest_bytes: u64,
    authorization_bytes: u64,
    component_bytes: u64,
    manifest_chunk_count: u32,
    authorization_chunk_count: u32,
    component_chunk_count: u32,
}

impl WorkerComponentDescriptor {
    pub fn new(
        extension_id: impl Into<String>,
        extension_version: impl Into<String>,
        plugin_identifier: impl Into<String>,
        plugin_version: impl Into<String>,
        authorization_generation: WorkerSha256Digest,
        manifest_digest_sha256: WorkerSha256Digest,
        component_digest_sha256: WorkerSha256Digest,
        manifest_bytes: u64,
        authorization_bytes: u64,
        component_bytes: u64,
    ) -> Result<Self, WorkerComponentDeploymentError> {
        let extension_id = checked_component_identity(
            extension_id.into(),
            MAX_WORKER_COMPONENT_IDENTITY_BYTES,
            WorkerComponentIdentityField::ExtensionId,
        )?;
        let extension_version = checked_component_identity(
            extension_version.into(),
            MAX_WORKER_COMPONENT_VERSION_BYTES,
            WorkerComponentIdentityField::ExtensionVersion,
        )?;
        let plugin_identifier = checked_component_identity(
            plugin_identifier.into(),
            MAX_WORKER_COMPONENT_IDENTITY_BYTES,
            WorkerComponentIdentityField::PluginIdentifier,
        )?;
        let plugin_version = checked_component_identity(
            plugin_version.into(),
            MAX_WORKER_COMPONENT_VERSION_BYTES,
            WorkerComponentIdentityField::PluginVersion,
        )?;
        let manifest_chunk_count = checked_chunk_count(
            manifest_bytes,
            MAX_WORKER_COMPONENT_MANIFEST_BYTES,
            WorkerComponentContent::Manifest,
        )?;
        let component_chunk_count = checked_chunk_count(
            component_bytes,
            MAX_WORKER_COMPONENT_BYTES,
            WorkerComponentContent::Component,
        )?;
        let authorization_chunk_count = checked_chunk_count(
            authorization_bytes,
            MAX_WORKER_COMPONENT_AUTHORIZATION_BYTES,
            WorkerComponentContent::Authorization,
        )?;
        Ok(Self {
            extension_id,
            extension_version,
            plugin_identifier,
            plugin_version,
            authorization_generation,
            manifest_digest_sha256,
            component_digest_sha256,
            manifest_bytes,
            authorization_bytes,
            component_bytes,
            manifest_chunk_count,
            authorization_chunk_count,
            component_chunk_count,
        })
    }

    pub fn extension_id(&self) -> &str {
        &self.extension_id
    }

    pub fn extension_version(&self) -> &str {
        &self.extension_version
    }

    pub fn plugin_identifier(&self) -> &str {
        &self.plugin_identifier
    }

    pub fn plugin_version(&self) -> &str {
        &self.plugin_version
    }

    pub fn authorization_generation(&self) -> &WorkerSha256Digest {
        &self.authorization_generation
    }

    pub fn manifest_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.manifest_digest_sha256
    }

    pub fn component_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.component_digest_sha256
    }

    pub const fn manifest_bytes(&self) -> u64 {
        self.manifest_bytes
    }

    pub const fn component_bytes(&self) -> u64 {
        self.component_bytes
    }

    pub const fn authorization_bytes(&self) -> u64 {
        self.authorization_bytes
    }

    pub const fn manifest_chunk_count(&self) -> u32 {
        self.manifest_chunk_count
    }

    pub const fn component_chunk_count(&self) -> u32 {
        self.component_chunk_count
    }

    pub const fn authorization_chunk_count(&self) -> u32 {
        self.authorization_chunk_count
    }

    fn validate(&self) -> Result<(), WorkerComponentDeploymentError> {
        let checked = Self::new(
            self.extension_id.clone(),
            self.extension_version.clone(),
            self.plugin_identifier.clone(),
            self.plugin_version.clone(),
            self.authorization_generation.clone(),
            self.manifest_digest_sha256.clone(),
            self.component_digest_sha256.clone(),
            self.manifest_bytes,
            self.authorization_bytes,
            self.component_bytes,
        )?;
        if checked.manifest_chunk_count != self.manifest_chunk_count
            || checked.authorization_chunk_count != self.authorization_chunk_count
            || checked.component_chunk_count != self.component_chunk_count
        {
            return Err(WorkerComponentDeploymentError::InvalidChunkCount);
        }
        Ok(())
    }
}

fn checked_component_identity(
    value: String,
    maximum_bytes: usize,
    field: WorkerComponentIdentityField,
) -> Result<String, WorkerComponentDeploymentError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || value.trim() != value
        || value.chars().any(char::is_control)
        || value.contains(['/', '\\'])
        || matches!(value.as_str(), "." | "..")
    {
        return Err(WorkerComponentDeploymentError::InvalidIdentity(field));
    }
    Ok(value)
}

fn checked_chunk_count(
    byte_length: u64,
    maximum: usize,
    content: WorkerComponentContent,
) -> Result<u32, WorkerComponentDeploymentError> {
    if byte_length == 0 {
        return Err(WorkerComponentDeploymentError::EmptyContent(content));
    }
    let maximum =
        u64::try_from(maximum).map_err(|_| WorkerComponentDeploymentError::LengthOverflow)?;
    if byte_length > maximum {
        return Err(WorkerComponentDeploymentError::ContentTooLarge(content));
    }
    let chunk_bytes = u64::try_from(MAX_WORKER_COMPONENT_CHUNK_BYTES)
        .map_err(|_| WorkerComponentDeploymentError::LengthOverflow)?;
    let count = byte_length
        .checked_add(chunk_bytes - 1)
        .ok_or(WorkerComponentDeploymentError::LengthOverflow)?
        / chunk_bytes;
    u32::try_from(count).map_err(|_| WorkerComponentDeploymentError::LengthOverflow)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRegistryDeploymentBegin {
    generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
    components: Vec<WorkerComponentDescriptor>,
}

impl WorkerRegistryDeploymentBegin {
    pub fn new(
        generation: WorkerRegistryGeneration,
        registry_digest_sha256: WorkerSha256Digest,
        components: Vec<WorkerComponentDescriptor>,
    ) -> Result<Self, WorkerComponentDeploymentError> {
        let value = Self {
            generation,
            registry_digest_sha256,
            components,
        };
        value.validate()?;
        Ok(value)
    }

    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }

    pub fn components(&self) -> &[WorkerComponentDescriptor] {
        &self.components
    }

    pub fn digest_material(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            WORKER_REGISTRY_DIGEST_DOMAIN.len() + 4 + self.components.len() * 256,
        );
        bytes.extend_from_slice(WORKER_REGISTRY_DIGEST_DOMAIN);
        let component_count = u32::try_from(self.components.len()).unwrap_or(u32::MAX);
        bytes.extend_from_slice(&component_count.to_le_bytes());
        for component in &self.components {
            extend_digest_string(&mut bytes, component.extension_id());
            extend_digest_string(&mut bytes, component.extension_version());
            extend_digest_string(&mut bytes, component.plugin_identifier());
            extend_digest_string(&mut bytes, component.plugin_version());
            bytes.extend_from_slice(&component.authorization_generation.bytes());
            bytes.extend_from_slice(&component.manifest_digest_sha256.bytes());
            bytes.extend_from_slice(&component.component_digest_sha256.bytes());
            bytes.extend_from_slice(&component.manifest_bytes.to_le_bytes());
            bytes.extend_from_slice(&component.authorization_bytes.to_le_bytes());
            bytes.extend_from_slice(&component.component_bytes.to_le_bytes());
        }
        bytes
    }

    fn validate(&self) -> Result<(), WorkerComponentDeploymentError> {
        if self.components.len() > MAX_WORKER_COMPONENT_COUNT {
            return Err(WorkerComponentDeploymentError::TooManyComponents);
        }
        let mut previous_key: Option<(&str, &WorkerSha256Digest)> = None;
        let mut total_bytes = 0_u64;
        for component in &self.components {
            component.validate()?;
            let current_key = (
                component.extension_id(),
                component.component_digest_sha256(),
            );
            if let Some(previous_key) = previous_key {
                if previous_key.0 == current_key.0 {
                    return Err(WorkerComponentDeploymentError::DuplicateExtensionIdentity);
                }
                if previous_key >= current_key {
                    return Err(WorkerComponentDeploymentError::NonCanonicalComponentOrder);
                }
            }
            previous_key = Some(current_key);
            total_bytes = total_bytes
                .checked_add(component.manifest_bytes)
                .and_then(|total| total.checked_add(component.authorization_bytes))
                .and_then(|total| total.checked_add(component.component_bytes))
                .ok_or(WorkerComponentDeploymentError::LengthOverflow)?;
        }
        let maximum = u64::try_from(MAX_WORKER_REGISTRY_DEPLOYMENT_BYTES)
            .map_err(|_| WorkerComponentDeploymentError::LengthOverflow)?;
        if total_bytes > maximum {
            return Err(WorkerComponentDeploymentError::DeploymentTooLarge);
        }
        Ok(())
    }
}

fn extend_digest_string(bytes: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).unwrap_or(u32::MAX);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRegistryDeploymentChunk {
    generation: WorkerRegistryGeneration,
    component_index: u32,
    content: WorkerComponentContent,
    chunk_index: u32,
    bytes: Vec<u8>,
}

impl WorkerRegistryDeploymentChunk {
    pub fn new(
        generation: WorkerRegistryGeneration,
        component_index: u32,
        content: WorkerComponentContent,
        chunk_index: u32,
        bytes: Vec<u8>,
    ) -> Result<Self, WorkerComponentDeploymentError> {
        let value = Self {
            generation,
            component_index,
            content,
            chunk_index,
            bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub const fn component_index(&self) -> u32 {
        self.component_index
    }

    pub const fn content(&self) -> WorkerComponentContent {
        self.content
    }

    pub const fn chunk_index(&self) -> u32 {
        self.chunk_index
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn validate(&self) -> Result<(), WorkerComponentDeploymentError> {
        if self.bytes.is_empty() {
            return Err(WorkerComponentDeploymentError::EmptyChunk);
        }
        if self.bytes.len() > MAX_WORKER_COMPONENT_CHUNK_BYTES {
            return Err(WorkerComponentDeploymentError::ChunkTooLarge);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRegistryDeploymentCommit {
    generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
}

impl WorkerRegistryDeploymentCommit {
    pub fn new(
        generation: WorkerRegistryGeneration,
        registry_digest_sha256: WorkerSha256Digest,
    ) -> Self {
        Self {
            generation,
            registry_digest_sha256,
        }
    }

    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRegistryDeploymentAck {
    generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
    component_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRegistryDeploymentRejectionReason {
    VerificationUnavailable,
    InvalidCandidate,
    InvalidAuthorization,
    ComponentCompilationFailed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerRegistryDeploymentRejection {
    version: u16,
    generation: WorkerRegistryGeneration,
    registry_digest_sha256: WorkerSha256Digest,
    reason: WorkerRegistryDeploymentRejectionReason,
}

impl WorkerRegistryDeploymentRejection {
    pub fn new(
        generation: WorkerRegistryGeneration,
        registry_digest_sha256: WorkerSha256Digest,
        reason: WorkerRegistryDeploymentRejectionReason,
    ) -> Self {
        Self {
            version: WORKER_REGISTRY_DEPLOYMENT_REJECTION_VERSION,
            generation,
            registry_digest_sha256,
            reason,
        }
    }

    pub const fn version(&self) -> u16 {
        self.version
    }

    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }

    pub const fn reason(&self) -> WorkerRegistryDeploymentRejectionReason {
        self.reason
    }

    fn validate(&self) -> Result<(), WorkerComponentDeploymentError> {
        if self.version != WORKER_REGISTRY_DEPLOYMENT_REJECTION_VERSION {
            return Err(
                WorkerComponentDeploymentError::UnsupportedDeploymentRejectionVersion {
                    expected: WORKER_REGISTRY_DEPLOYMENT_REJECTION_VERSION,
                    actual: self.version,
                },
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPluginExecutionFailure {
    Cancelled,
    TimedOut,
    Trap { diagnostic: String },
    InvalidInvocation,
    CapabilityDenied,
    HostFailure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPluginExecutionOutcome {
    Succeeded(Vec<u8>),
    Failed(WorkerPluginExecutionFailure),
}

impl WorkerPluginExecutionOutcome {
    pub fn result_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Succeeded(bytes) => Some(bytes),
            Self::Failed(_) => None,
        }
    }
}

impl WorkerRegistryDeploymentAck {
    pub fn new(
        generation: WorkerRegistryGeneration,
        registry_digest_sha256: WorkerSha256Digest,
        component_count: u32,
    ) -> Result<Self, WorkerComponentDeploymentError> {
        if usize::try_from(component_count)
            .map_err(|_| WorkerComponentDeploymentError::TooManyComponents)?
            > MAX_WORKER_COMPONENT_COUNT
        {
            return Err(WorkerComponentDeploymentError::TooManyComponents);
        }
        Ok(Self {
            generation,
            registry_digest_sha256,
            component_count,
        })
    }

    pub const fn generation(&self) -> WorkerRegistryGeneration {
        self.generation
    }

    pub fn registry_digest_sha256(&self) -> &WorkerSha256Digest {
        &self.registry_digest_sha256
    }

    pub const fn component_count(&self) -> u32 {
        self.component_count
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerComponentDeploymentError {
    #[error("worker registry generation must be nonzero")]
    ZeroGeneration,
    #[error(
        "worker registry deployment rejection version {actual} is unsupported; expected {expected}"
    )]
    UnsupportedDeploymentRejectionVersion { expected: u16, actual: u16 },
    #[error("worker content address must be exactly 64 lowercase hexadecimal characters")]
    InvalidSha256,
    #[error("worker component {0:?} is empty, oversized, malformed, or path-bearing")]
    InvalidIdentity(WorkerComponentIdentityField),
    #[error("worker {0:?} content is empty")]
    EmptyContent(WorkerComponentContent),
    #[error("worker {0:?} content exceeds its bound")]
    ContentTooLarge(WorkerComponentContent),
    #[error("worker component chunk is empty")]
    EmptyChunk,
    #[error("worker component chunk exceeds {MAX_WORKER_COMPONENT_CHUNK_BYTES} bytes")]
    ChunkTooLarge,
    #[error("worker component chunk count does not match its declared byte length")]
    InvalidChunkCount,
    #[error("worker component count exceeds {MAX_WORKER_COMPONENT_COUNT}")]
    TooManyComponents,
    #[error("worker component descriptors are not in canonical identity and digest order")]
    NonCanonicalComponentOrder,
    #[error("worker component deployment repeats a lifecycle extension identifier")]
    DuplicateExtensionIdentity,
    #[error("worker registry deployment exceeds {MAX_WORKER_REGISTRY_DEPLOYMENT_BYTES} bytes")]
    DeploymentTooLarge,
    #[error("worker component deployment length overflowed")]
    LengthOverflow,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProfileId(pub Uuid);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(pub Uuid);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub Uuid);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerOperationCategory {
    Allocation,
    Copy,
    Event,
    Scalar,
    Unary,
    Binary,
    Reduction,
    Indexing,
    Resize,
    Convolution,
    LinearAlgebra,
    CustomKernel,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerDType {
    F64,
    F32,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Complex64,
    Complex128,
    Float8E4m3Fn,
    Float8E5m2,
    #[serde(rename = "float8_e4m3fnuz", alias = "float8_e4m3_fnuz")]
    Float8E4m3Fnuz,
    #[serde(rename = "float8_e5m2fnuz", alias = "float8_e5m2_fnuz")]
    Float8E5m2Fnuz,
    #[serde(rename = "float8_e8m0fnu", alias = "float8_e8m0_fnu")]
    Float8E8m0Fnu,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLayout {
    Contiguous,
    ChannelsLast,
    ChannelsLast3d,
    Strided,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerUnaryOperationV1 {
    Absolute,
    Negate,
    Exponential,
    NaturalLogarithm,
    SquareRoot,
    Reciprocal,
    Sine,
    Cosine,
    HyperbolicTangent,
    Sigmoid,
    Round,
    Sinc,
    Log1p,
    ReciprocalSquareRoot,
    Relu,
    IsFinite,
    InvertUnitInterval,
    LogarithmBaseTwo,
    Signum,
    Tangent,
    ArcTangent,
    ArcHyperbolicTangent,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerBinaryOperationV1 {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Power,
    Minimum,
    Maximum,
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
    FloatingRemainder,
    Atan2,
    LogAddExp,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerReductionOperationV1 {
    Sum,
    Product,
    Mean,
    Minimum,
    Maximum,
    ArgMinimum,
    ArgMaximum,
    All,
    Any,
    Variance,
    StandardDeviation,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerResizeModeV1 {
    NearestExact,
    Bilinear,
    Area,
    Bicubic,
    Lanczos,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLinearAlgebraOperationV1 {
    MatrixMultiply,
    BatchMatrixMultiply,
    MatrixVectorMultiply,
    Dot,
    Outer,
    Solve,
    SingularValueDecomposition,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerTensorRoleV1 {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPrimitiveOperationV1 {
    Allocation,
    Copy,
    Fill,
    Unary(WorkerUnaryOperationV1),
    Binary(WorkerBinaryOperationV1),
    BinaryScalar(WorkerBinaryOperationV1),
    Reduction(WorkerReductionOperationV1),
    Select,
    Narrow,
    Resize(WorkerResizeModeV1),
    RecordEvent,
    WaitEvent,
}

impl WorkerPrimitiveOperationV1 {
    pub const fn category(self) -> WorkerOperationCategory {
        match self {
            Self::Allocation => WorkerOperationCategory::Allocation,
            Self::Copy => WorkerOperationCategory::Copy,
            Self::Fill => WorkerOperationCategory::Scalar,
            Self::Unary(_) => WorkerOperationCategory::Unary,
            Self::Binary(_) | Self::BinaryScalar(_) => WorkerOperationCategory::Binary,
            Self::Reduction(_) => WorkerOperationCategory::Reduction,
            Self::Select | Self::Narrow => WorkerOperationCategory::Indexing,
            Self::Resize(_) => WorkerOperationCategory::Resize,
            Self::RecordEvent | Self::WaitEvent => WorkerOperationCategory::Event,
        }
    }

    pub const fn is_event(self) -> bool {
        matches!(self, Self::RecordEvent | Self::WaitEvent)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPrimitiveOperationV2 {
    Allocation,
    Copy,
    Fill,
    Unary(WorkerUnaryOperationV1),
    Binary(WorkerBinaryOperationV1),
    BinaryScalar(WorkerBinaryOperationV1),
    Reduction(WorkerReductionOperationV1),
    Select,
    Narrow,
    Resize(WorkerResizeModeV1),
    RecordEvent,
    WaitEvent,
    LinearAlgebra(WorkerLinearAlgebraOperationV1),
    Gather,
    Scatter,
    MaskedSelect,
    Convolution,
    CustomKernel,
}

impl WorkerPrimitiveOperationV2 {
    pub const fn category(self) -> WorkerOperationCategory {
        match self {
            Self::Allocation => WorkerOperationCategory::Allocation,
            Self::Copy => WorkerOperationCategory::Copy,
            Self::Fill => WorkerOperationCategory::Scalar,
            Self::Unary(_) => WorkerOperationCategory::Unary,
            Self::Binary(_) | Self::BinaryScalar(_) => WorkerOperationCategory::Binary,
            Self::Reduction(_) => WorkerOperationCategory::Reduction,
            Self::Select | Self::Narrow | Self::Gather | Self::Scatter | Self::MaskedSelect => {
                WorkerOperationCategory::Indexing
            }
            Self::Resize(_) => WorkerOperationCategory::Resize,
            Self::LinearAlgebra(_) => WorkerOperationCategory::LinearAlgebra,
            Self::Convolution => WorkerOperationCategory::Convolution,
            Self::CustomKernel => WorkerOperationCategory::CustomKernel,
            Self::RecordEvent | Self::WaitEvent => WorkerOperationCategory::Event,
        }
    }

    pub const fn is_event(self) -> bool {
        matches!(self, Self::RecordEvent | Self::WaitEvent)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(into = "WorkerOperationSupportV1Wire")]
pub struct WorkerOperationSupportV1 {
    operation: WorkerPrimitiveOperationV1,
    role: Option<WorkerTensorRoleV1>,
    dtype: Option<WorkerDType>,
    layout: Option<WorkerLayout>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerOperationSupportV1Wire {
    version: u16,
    operation: WorkerPrimitiveOperationV1,
    role: Option<WorkerTensorRoleV1>,
    dtype: Option<WorkerDType>,
    layout: Option<WorkerLayout>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(into = "WorkerOperationSupportWire")]
pub struct WorkerOperationSupport {
    operation: WorkerPrimitiveOperationV2,
    role: Option<WorkerTensorRoleV1>,
    dtype: Option<WorkerDType>,
    layout: Option<WorkerLayout>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerOperationSupportWire {
    version: u16,
    operation: WorkerPrimitiveOperationV2,
    role: Option<WorkerTensorRoleV1>,
    dtype: Option<WorkerDType>,
    layout: Option<WorkerLayout>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerOperationSupportError {
    #[error("unsupported worker operation-support version {0}")]
    UnsupportedVersion(u16),
    #[error("tensor primitive capability requires role, dtype, and layout")]
    MissingTensorSignature,
    #[error("event primitive capability cannot declare a role, dtype, or layout")]
    EventTensorSignature,
}

impl WorkerOperationSupportV1 {
    pub const fn for_tensor(
        operation: WorkerPrimitiveOperationV1,
        role: WorkerTensorRoleV1,
        dtype: WorkerDType,
        layout: WorkerLayout,
    ) -> Result<Self, WorkerOperationSupportError> {
        if operation.is_event() {
            Err(WorkerOperationSupportError::EventTensorSignature)
        } else {
            Ok(Self {
                operation,
                role: Some(role),
                dtype: Some(dtype),
                layout: Some(layout),
            })
        }
    }

    pub const fn for_event(
        operation: WorkerPrimitiveOperationV1,
    ) -> Result<Self, WorkerOperationSupportError> {
        if operation.is_event() {
            Ok(Self {
                operation,
                role: None,
                dtype: None,
                layout: None,
            })
        } else {
            Err(WorkerOperationSupportError::MissingTensorSignature)
        }
    }

    pub const fn version(&self) -> u16 {
        LEGACY_WORKER_OPERATION_SUPPORT_VERSION
    }

    pub const fn operation(&self) -> WorkerPrimitiveOperationV1 {
        self.operation
    }
}

impl From<WorkerOperationSupportV1> for WorkerOperationSupportV1Wire {
    fn from(value: WorkerOperationSupportV1) -> Self {
        Self {
            version: LEGACY_WORKER_OPERATION_SUPPORT_VERSION,
            operation: value.operation,
            role: value.role,
            dtype: value.dtype,
            layout: value.layout,
        }
    }
}

impl TryFrom<WorkerOperationSupportV1Wire> for WorkerOperationSupportV1 {
    type Error = WorkerOperationSupportError;

    fn try_from(value: WorkerOperationSupportV1Wire) -> Result<Self, Self::Error> {
        if value.version != LEGACY_WORKER_OPERATION_SUPPORT_VERSION {
            return Err(WorkerOperationSupportError::UnsupportedVersion(
                value.version,
            ));
        }
        match (
            value.operation.is_event(),
            value.role,
            value.dtype,
            value.layout,
        ) {
            (true, None, None, None) => Self::for_event(value.operation),
            (true, _, _, _) => Err(WorkerOperationSupportError::EventTensorSignature),
            (false, Some(role), Some(dtype), Some(layout)) => {
                Self::for_tensor(value.operation, role, dtype, layout)
            }
            (false, _, _, _) => Err(WorkerOperationSupportError::MissingTensorSignature),
        }
    }
}

impl<'de> Deserialize<'de> for WorkerOperationSupportV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        WorkerOperationSupportV1Wire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl WorkerOperationSupport {
    pub const fn for_tensor_v2(
        operation: WorkerPrimitiveOperationV2,
        role: WorkerTensorRoleV1,
        dtype: WorkerDType,
        layout: WorkerLayout,
    ) -> Result<Self, WorkerOperationSupportError> {
        if operation.is_event() {
            Err(WorkerOperationSupportError::EventTensorSignature)
        } else {
            Ok(Self {
                operation,
                role: Some(role),
                dtype: Some(dtype),
                layout: Some(layout),
            })
        }
    }

    pub const fn for_event_v2(
        operation: WorkerPrimitiveOperationV2,
    ) -> Result<Self, WorkerOperationSupportError> {
        if operation.is_event() {
            Ok(Self {
                operation,
                role: None,
                dtype: None,
                layout: None,
            })
        } else {
            Err(WorkerOperationSupportError::MissingTensorSignature)
        }
    }

    pub const fn version(&self) -> u16 {
        WORKER_OPERATION_SUPPORT_VERSION
    }

    pub const fn operation(&self) -> WorkerPrimitiveOperationV2 {
        self.operation
    }

    pub const fn category(&self) -> WorkerOperationCategory {
        self.operation.category()
    }

    pub const fn role(&self) -> Option<WorkerTensorRoleV1> {
        self.role
    }

    pub const fn dtype(&self) -> Option<WorkerDType> {
        self.dtype
    }

    pub const fn layout(&self) -> Option<WorkerLayout> {
        self.layout
    }
}

impl From<WorkerOperationSupport> for WorkerOperationSupportWire {
    fn from(value: WorkerOperationSupport) -> Self {
        Self {
            version: WORKER_OPERATION_SUPPORT_VERSION,
            operation: value.operation,
            role: value.role,
            dtype: value.dtype,
            layout: value.layout,
        }
    }
}

impl TryFrom<WorkerOperationSupportWire> for WorkerOperationSupport {
    type Error = WorkerOperationSupportError;

    fn try_from(value: WorkerOperationSupportWire) -> Result<Self, Self::Error> {
        if value.version != WORKER_OPERATION_SUPPORT_VERSION {
            return Err(WorkerOperationSupportError::UnsupportedVersion(
                value.version,
            ));
        }
        match (
            value.operation.is_event(),
            value.role,
            value.dtype,
            value.layout,
        ) {
            (true, None, None, None) => Self::for_event_v2(value.operation),
            (true, _, _, _) => Err(WorkerOperationSupportError::EventTensorSignature),
            (false, Some(role), Some(dtype), Some(layout)) => {
                Self::for_tensor_v2(value.operation, role, dtype, layout)
            }
            (false, _, _, _) => Err(WorkerOperationSupportError::MissingTensorSignature),
        }
    }
}

impl<'de> Deserialize<'de> for WorkerOperationSupport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        WorkerOperationSupportWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "WorkerBackendCapabilitiesWire",
    into = "WorkerBackendCapabilitiesWire"
)]
pub struct WorkerBackendCapabilities {
    device: DeviceKind,
    ordinal: u32,
    supported: Vec<WorkerOperationSupport>,
    deterministic: Vec<WorkerOperationSupport>,
    properties: Option<WorkerNativeDeviceProperties>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WorkerBackendCapabilitiesWire {
    device: DeviceKind,
    ordinal: u32,
    supported: Vec<WorkerOperationSupport>,
    deterministic: Vec<WorkerOperationSupport>,
    properties: Option<WorkerNativeDeviceProperties>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "WorkerNativeDevicePropertiesWire",
    into = "WorkerNativeDevicePropertiesWire"
)]
pub struct WorkerNativeDeviceProperties {
    name: String,
    total_memory_bytes: u64,
    allocation_limit_bytes: u64,
    major: u32,
    minor: u32,
    architecture: Option<String>,
    has_fp16: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WorkerNativeDevicePropertiesWire {
    name: String,
    total_memory_bytes: u64,
    major: u32,
    minor: u32,
    architecture: Option<String>,
    has_fp16: bool,
    allocation_limit_bytes: u64,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerNativeDevicePropertiesError {
    #[error("worker native device name exceeds {MAX_WORKER_DEVICE_PROPERTY_BYTES} bytes")]
    NameOversized,
    #[error("worker native device architecture exceeds {MAX_WORKER_DEVICE_PROPERTY_BYTES} bytes")]
    ArchitectureOversized,
    #[error("worker native device memory and allocation limits are invalid")]
    InvalidMemoryLimit,
}

impl WorkerNativeDeviceProperties {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        total_memory_bytes: u64,
        major: u32,
        minor: u32,
        architecture: Option<String>,
        has_fp16: bool,
    ) -> Result<Self, WorkerNativeDevicePropertiesError> {
        Self::new_with_allocation_limit(
            name,
            total_memory_bytes,
            total_memory_bytes,
            major,
            minor,
            architecture,
            has_fp16,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_allocation_limit(
        name: impl Into<String>,
        total_memory_bytes: u64,
        allocation_limit_bytes: u64,
        major: u32,
        minor: u32,
        architecture: Option<String>,
        has_fp16: bool,
    ) -> Result<Self, WorkerNativeDevicePropertiesError> {
        let name = name.into();
        if name.len() > MAX_WORKER_DEVICE_PROPERTY_BYTES {
            return Err(WorkerNativeDevicePropertiesError::NameOversized);
        }
        if architecture
            .as_ref()
            .is_some_and(|value| value.len() > MAX_WORKER_DEVICE_PROPERTY_BYTES)
        {
            return Err(WorkerNativeDevicePropertiesError::ArchitectureOversized);
        }
        if total_memory_bytes == 0
            || allocation_limit_bytes == 0
            || allocation_limit_bytes > total_memory_bytes
        {
            return Err(WorkerNativeDevicePropertiesError::InvalidMemoryLimit);
        }
        Ok(Self {
            name,
            total_memory_bytes,
            allocation_limit_bytes,
            major,
            minor,
            architecture,
            has_fp16,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn total_memory_bytes(&self) -> u64 {
        self.total_memory_bytes
    }

    pub const fn allocation_limit_bytes(&self) -> u64 {
        self.allocation_limit_bytes
    }

    pub const fn major(&self) -> u32 {
        self.major
    }

    pub const fn minor(&self) -> u32 {
        self.minor
    }

    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    pub const fn has_fp16(&self) -> bool {
        self.has_fp16
    }
}

impl From<WorkerNativeDeviceProperties> for WorkerNativeDevicePropertiesWire {
    fn from(value: WorkerNativeDeviceProperties) -> Self {
        Self {
            name: value.name,
            total_memory_bytes: value.total_memory_bytes,
            allocation_limit_bytes: value.allocation_limit_bytes,
            major: value.major,
            minor: value.minor,
            architecture: value.architecture,
            has_fp16: value.has_fp16,
        }
    }
}

impl TryFrom<WorkerNativeDevicePropertiesWire> for WorkerNativeDeviceProperties {
    type Error = WorkerNativeDevicePropertiesError;

    fn try_from(value: WorkerNativeDevicePropertiesWire) -> Result<Self, Self::Error> {
        Self::new_with_allocation_limit(
            value.name,
            value.total_memory_bytes,
            value.allocation_limit_bytes,
            value.major,
            value.minor,
            value.architecture,
            value.has_fp16,
        )
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerBackendCapabilityError {
    #[error("worker backend capability declaration is empty")]
    Empty,
    #[error(
        "worker backend capability declaration exceeds {MAX_WORKER_BACKEND_CAPABILITY_ENTRIES} entries"
    )]
    Oversized,
    #[error("worker backend capability declaration contains a duplicate primitive support")]
    Duplicate,
}

impl WorkerBackendCapabilities {
    pub fn new(
        device: DeviceKind,
        ordinal: u32,
        supported: Vec<WorkerOperationSupport>,
        deterministic: Vec<WorkerOperationSupport>,
    ) -> Result<Self, WorkerBackendCapabilityError> {
        Self::new_with_properties(device, ordinal, supported, deterministic, None)
    }

    pub fn new_with_properties(
        device: DeviceKind,
        ordinal: u32,
        mut supported: Vec<WorkerOperationSupport>,
        mut deterministic: Vec<WorkerOperationSupport>,
        properties: Option<WorkerNativeDeviceProperties>,
    ) -> Result<Self, WorkerBackendCapabilityError> {
        if supported.is_empty() {
            return Err(WorkerBackendCapabilityError::Empty);
        }
        if supported.len() > MAX_WORKER_BACKEND_CAPABILITY_ENTRIES
            || deterministic.len() > MAX_WORKER_BACKEND_CAPABILITY_ENTRIES
        {
            return Err(WorkerBackendCapabilityError::Oversized);
        }
        canonicalize_worker_support(&mut supported)?;
        canonicalize_worker_support(&mut deterministic)?;
        Ok(Self {
            device,
            ordinal,
            supported,
            deterministic,
            properties,
        })
    }

    pub const fn device(&self) -> DeviceKind {
        self.device
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn supported(&self) -> &[WorkerOperationSupport] {
        &self.supported
    }

    pub fn deterministic(&self) -> &[WorkerOperationSupport] {
        &self.deterministic
    }

    pub fn properties(&self) -> Option<&WorkerNativeDeviceProperties> {
        self.properties.as_ref()
    }
}

fn canonicalize_worker_support(
    support: &mut [WorkerOperationSupport],
) -> Result<(), WorkerBackendCapabilityError> {
    support.sort_unstable();
    if support.windows(2).any(|entries| entries[0] == entries[1]) {
        Err(WorkerBackendCapabilityError::Duplicate)
    } else {
        Ok(())
    }
}

impl From<WorkerBackendCapabilities> for WorkerBackendCapabilitiesWire {
    fn from(value: WorkerBackendCapabilities) -> Self {
        Self {
            device: value.device,
            ordinal: value.ordinal,
            supported: value.supported,
            deterministic: value.deterministic,
            properties: value.properties,
        }
    }
}

impl TryFrom<WorkerBackendCapabilitiesWire> for WorkerBackendCapabilities {
    type Error = WorkerBackendCapabilityError;

    fn try_from(value: WorkerBackendCapabilitiesWire) -> Result<Self, Self::Error> {
        Self::new_with_properties(
            value.device,
            value.ordinal,
            value.supported,
            value.deterministic,
            value.properties,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkerEnvelope {
    pub version: u16,
    pub profile_id: ProfileId,
    pub worker_id: WorkerId,
    pub request_id: RequestId,
    pub prompt_id: Option<PromptId>,
    pub attempt_id: Option<AttemptId>,
    pub sequence: u64,
    pub registry_version: String,
    pub message: WorkerMessage,
    pub extensions: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycleEvent {
    ExecutionStarted,
    CancellationRequested { reason: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(into = "WorkerOutputProposalWire")]
pub struct WorkerOutputProposal {
    proposal_id: Uuid,
    metadata: Vec<u8>,
    content: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WorkerOutputProposalWire {
    proposal_id: Uuid,
    metadata: Vec<u8>,
    content: Vec<u8>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerOutputProposalError {
    #[error("worker output metadata exceeds {MAX_WORKER_OUTPUT_METADATA_BYTES} bytes")]
    MetadataTooLarge,
    #[error("worker output content exceeds {MAX_WORKER_OUTPUT_CONTENT_BYTES} bytes")]
    ContentTooLarge,
}

impl WorkerOutputProposal {
    pub fn new(
        proposal_id: Uuid,
        metadata: Vec<u8>,
        content: Vec<u8>,
    ) -> Result<Self, WorkerOutputProposalError> {
        if metadata.len() > MAX_WORKER_OUTPUT_METADATA_BYTES {
            return Err(WorkerOutputProposalError::MetadataTooLarge);
        }
        if content.len() > MAX_WORKER_OUTPUT_CONTENT_BYTES {
            return Err(WorkerOutputProposalError::ContentTooLarge);
        }
        Ok(Self {
            proposal_id,
            metadata,
            content,
        })
    }

    pub const fn proposal_id(&self) -> Uuid {
        self.proposal_id
    }

    pub fn metadata(&self) -> &[u8] {
        &self.metadata
    }

    pub fn content(&self) -> &[u8] {
        &self.content
    }

    pub fn into_parts(self) -> (Uuid, Vec<u8>, Vec<u8>) {
        (self.proposal_id, self.metadata, self.content)
    }
}

impl From<WorkerOutputProposal> for WorkerOutputProposalWire {
    fn from(value: WorkerOutputProposal) -> Self {
        Self {
            proposal_id: value.proposal_id,
            metadata: value.metadata,
            content: value.content,
        }
    }
}

impl TryFrom<WorkerOutputProposalWire> for WorkerOutputProposal {
    type Error = WorkerOutputProposalError;

    fn try_from(value: WorkerOutputProposalWire) -> Result<Self, Self::Error> {
        Self::new(value.proposal_id, value.metadata, value.content)
    }
}

impl<'de> Deserialize<'de> for WorkerOutputProposal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        WorkerOutputProposalWire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderHttpMethod {
    Delete,
    Get,
    Head,
    Options,
    Patch,
    Post,
    Put,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderHeader {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderInvocationContext {
    pub session_id: Uuid,
    pub session_generation: u64,
    pub invocation: u64,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderStreamHandle {
    pub session_id: Uuid,
    pub session_generation: u64,
    pub invocation: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderStreamingContract {
    pub methods: Vec<WorkerProviderHttpMethod>,
    pub maximum_headers: u16,
    pub maximum_header_bytes: u32,
    pub maximum_request_body_bytes: u64,
    pub maximum_response_body_bytes: u64,
    pub maximum_chunk_bytes: u32,
    pub maximum_ndjson_line_bytes: u32,
    pub maximum_wait_milliseconds: u64,
    pub maximum_uploads: u32,
    pub maximum_upload_body_bytes: u64,
    pub maximum_cost_requests: u32,
    pub maximum_progress_total: u64,
    pub uploads: bool,
    pub cost_requests: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderRequestHead {
    pub endpoint: String,
    pub secret_id: Option<String>,
    pub method: WorkerProviderHttpMethod,
    pub headers: Vec<WorkerProviderHeader>,
    pub declared_body_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderRequestChunk {
    pub handle: WorkerProviderStreamHandle,
    pub sequence: u64,
    pub bytes: Vec<u8>,
    pub end: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderResponseHead {
    pub status: u16,
    pub headers: Vec<WorkerProviderHeader>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderResponseChunk {
    Binary(Vec<u8>),
    Text(String),
    NdjsonLine(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderTerminal {
    Completed(Vec<u8>),
    Failed { code: String, message: String },
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderResponseFrameEvent {
    Head(WorkerProviderResponseHead),
    Chunk(WorkerProviderResponseChunk),
    Terminal(WorkerProviderTerminal),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderResponseFrame {
    pub handle: WorkerProviderStreamHandle,
    pub sequence: u64,
    pub event: WorkerProviderResponseFrameEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderWaitRequest {
    pub handle: WorkerProviderStreamHandle,
    pub after_sequence: Option<u64>,
    pub timeout_milliseconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderWaitOutcome {
    Frame(WorkerProviderResponseFrame),
    TimedOut,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderUploadRequest {
    pub handle: WorkerProviderStreamHandle,
    pub port_id: String,
    pub media_type: String,
    pub byte_length: u64,
    pub content_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderUploadCompletionRequest {
    pub handle: WorkerProviderStreamHandle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderUploadCompletion {
    pub download_url: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderRetryRequest {
    pub handle: WorkerProviderStreamHandle,
    pub delay_milliseconds: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderCostRequest {
    pub handle: WorkerProviderStreamHandle,
    pub operation: String,
    pub currency: String,
    pub maximum_microunits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderCostResponse {
    pub accepted: bool,
    pub approved_microunits: u64,
    pub receipt: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderProgress {
    pub handle: WorkerProviderStreamHandle,
    pub sequence: u64,
    pub completed: u64,
    pub total: u64,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderStreamError {
    #[error("provider stream operation was cancelled")]
    Cancelled,
    #[error("provider stream operation timed out")]
    TimedOut,
    #[error("provider stream host failed")]
    HostFailure,
    #[error("provider streaming contract is invalid")]
    InvalidContract,
    #[error("provider stream handle is invalid")]
    InvalidHandle,
    #[error("provider stream handle belongs to a different stream")]
    ForeignHandle,
    #[error("provider stream handle was revoked")]
    RevokedHandle,
    #[error("provider HTTP method is invalid")]
    InvalidMethod,
    #[error("provider HTTP headers are invalid")]
    InvalidHeaders,
    #[error("provider stream body exceeds its bound")]
    BodyLimit,
    #[error("provider stream chunk exceeds its bound")]
    ChunkLimit,
    #[error("provider stream NDJSON line is invalid")]
    InvalidNdjsonLine,
    #[error("provider stream sequence is invalid")]
    InvalidSequence,
    #[error("provider stream operation order is invalid")]
    InvalidOrder,
    #[error("provider stream wait exceeds its bound")]
    WaitLimit,
    #[error("provider stream upload is invalid")]
    InvalidUpload,
    #[error("provider stream cost request is invalid")]
    InvalidCostRequest,
    #[error("provider stream progress is invalid")]
    InvalidProgress,
    #[error("provider stream terminal is invalid")]
    InvalidTerminal,
    #[error("provider invocation result is invalid")]
    InvalidInvocationResult,
    #[error("provider request authority is invalid")]
    InvalidRequestAuthority,
    #[error("provider stream session is foreign")]
    ForeignSession,
    #[error("provider stream belongs to a stale session generation")]
    StaleSession,
    #[error("provider stream belongs to a different invocation")]
    ForeignInvocation,
    #[error("provider stream belongs to a stale invocation generation")]
    StaleGeneration,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderStreamRequest {
    StartRequest {
        context: WorkerProviderInvocationContext,
        head: WorkerProviderRequestHead,
    },
    WriteRequestChunk(WorkerProviderRequestChunk),
    WaitResponse(WorkerProviderWaitRequest),
    StartUpload(WorkerProviderUploadRequest),
    WriteUploadChunk(WorkerProviderRequestChunk),
    RequestCost(WorkerProviderCostRequest),
    ReportProgress(WorkerProviderProgress),
    CheckCancelled(WorkerProviderStreamHandle),
    CompleteUpload(WorkerProviderUploadCompletionRequest),
    RetryRequest(WorkerProviderRetryRequest),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProviderStreamResponse {
    Stream(Result<WorkerProviderStreamHandle, WorkerProviderStreamError>),
    Unit(Result<(), WorkerProviderStreamError>),
    Wait(Result<WorkerProviderWaitOutcome, WorkerProviderStreamError>),
    Cost(Result<WorkerProviderCostResponse, WorkerProviderStreamError>),
    UploadCompletion(Result<WorkerProviderUploadCompletion, WorkerProviderStreamError>),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderV2ProposalFinalization {
    pub context: WorkerProviderInvocationContext,
    pub handle: WorkerProviderStreamHandle,
    pub proposal_generation: u64,
    pub finalization_nonce: [u8; 32],
    pub receipt_identity_sha256: WorkerSha256Digest,
    pub materialization_identity_sha256: WorkerSha256Digest,
}

impl WorkerProviderV2ProposalFinalization {
    pub fn validate(&self) -> Result<(), WorkerProviderStreamError> {
        validate_provider_context(&self.context)?;
        validate_provider_handle_wire(self.handle)?;
        if self.handle.session_id != self.context.session_id
            || self.handle.session_generation != self.context.session_generation
            || self.handle.invocation != self.context.invocation
            || self.handle.generation != self.context.generation
            || self.proposal_generation == 0
            || self.finalization_nonce == [0; 32]
            || self.receipt_identity_sha256.as_str().len()
                != MAX_WORKER_PROVIDER_FINALIZATION_IDENTITY_BYTES
            || self.materialization_identity_sha256.as_str().len()
                != MAX_WORKER_PROVIDER_FINALIZATION_IDENTITY_BYTES
        {
            return Err(WorkerProviderStreamError::InvalidInvocationResult);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerProviderV2ProposalFinalizationAck {
    pub finalization: WorkerProviderV2ProposalFinalization,
    pub result: Result<(), WorkerProviderStreamError>,
}

impl WorkerProviderV2ProposalFinalizationAck {
    pub fn validate(&self) -> Result<(), WorkerProviderStreamError> {
        self.finalization.validate()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WorkerProviderPendingResponse {
    PrimaryStream,
    UploadStream {
        byte_length: u64,
        content_sha256: String,
    },
    UploadCompletion {
        handle: WorkerProviderStreamHandle,
    },
    RetryStream {
        old_handle: WorkerProviderStreamHandle,
    },
    FinalProgress,
    HttpProgress,
    Unit,
    Wait,
    Cost {
        maximum_microunits: u64,
    },
}

#[derive(Debug)]
struct WorkerProviderUploadState {
    expected_bytes: u64,
    expected_sha256: String,
    received_bytes: u64,
    next_sequence: u64,
    digest: Sha256,
    terminal: bool,
    completion_received: bool,
}

pub struct WorkerProviderHttpContinuationChallenge<'a> {
    context: &'a WorkerProviderInvocationContext,
    certified_binding_sha256: &'a WorkerSha256Digest,
    ordinal: u32,
    predecessor_handle: WorkerProviderStreamHandle,
    predecessor_receipt_sha256: &'a str,
    request_head: &'a WorkerProviderRequestHead,
}

impl WorkerProviderHttpContinuationChallenge<'_> {
    pub fn context(&self) -> &WorkerProviderInvocationContext {
        self.context
    }

    pub fn certified_binding_sha256(&self) -> &WorkerSha256Digest {
        self.certified_binding_sha256
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn predecessor_handle(&self) -> WorkerProviderStreamHandle {
        self.predecessor_handle
    }

    pub fn predecessor_receipt_sha256(&self) -> &str {
        self.predecessor_receipt_sha256
    }

    pub fn request_head(&self) -> &WorkerProviderRequestHead {
        self.request_head
    }
}

#[derive(Debug)]
struct WorkerProviderHttpInvocationState {
    certified_binding_sha256: WorkerSha256Digest,
    maximum_requests: u32,
    ordinal: u32,
    completed_receipt_sha256: Option<String>,
    armed_head: Option<WorkerProviderRequestHead>,
}

#[derive(Debug)]
pub struct WorkerProviderStreamTransportValidator {
    context: WorkerProviderInvocationContext,
    contract: WorkerProviderStreamingContract,
    handle: Option<WorkerProviderStreamHandle>,
    retired_handles: BTreeSet<WorkerProviderStreamHandle>,
    pending: BTreeMap<u64, WorkerProviderPendingResponse>,
    last_admitted_call_id: u64,
    request_method: Option<WorkerProviderHttpMethod>,
    declared_request_bytes: Option<u64>,
    request_sequence: u64,
    request_bytes: u64,
    total_request_bytes: u64,
    request_ended: bool,
    response_sequence: u64,
    response_bytes: u64,
    total_response_bytes: u64,
    http_invocation: Option<WorkerProviderHttpInvocationState>,
    response_head_seen: bool,
    response_status: Option<u16>,
    response_retry_after: Option<String>,
    last_response_sequence: Option<u64>,
    terminal: bool,
    final_progress_pending: bool,
    revoked: bool,
    progress_sequence: u64,
    progress_completed: u64,
    progress_total: Option<u64>,
    upload_count: u32,
    upload_bytes: u64,
    uploads: BTreeMap<WorkerProviderStreamHandle, WorkerProviderUploadState>,
    cost_request_count: u32,
    attempt_count: u32,
    cancellation: CancellationToken,
}

impl WorkerProviderStreamTransportValidator {
    pub const fn primary_handle(&self) -> Option<WorkerProviderStreamHandle> {
        self.handle
    }

    pub fn checked_for_host_session(
        expected_host_context: WorkerProviderInvocationContext,
        contract: WorkerProviderStreamingContract,
        cancellation: CancellationToken,
    ) -> Result<Self, WorkerProviderStreamError> {
        validate_provider_context(&expected_host_context)?;
        validate_provider_contract(&contract)?;
        Ok(Self {
            context: expected_host_context,
            contract,
            handle: None,
            retired_handles: BTreeSet::new(),
            pending: BTreeMap::new(),
            last_admitted_call_id: 0,
            request_method: None,
            declared_request_bytes: None,
            request_sequence: 0,
            request_bytes: 0,
            total_request_bytes: 0,
            request_ended: false,
            response_sequence: 0,
            response_bytes: 0,
            total_response_bytes: 0,
            http_invocation: None,
            response_head_seen: false,
            response_status: None,
            response_retry_after: None,
            last_response_sequence: None,
            terminal: false,
            final_progress_pending: false,
            revoked: false,
            progress_sequence: 0,
            progress_completed: 0,
            progress_total: None,
            upload_count: 0,
            upload_bytes: 0,
            uploads: BTreeMap::new(),
            cost_request_count: 0,
            attempt_count: 0,
            cancellation,
        })
    }

    pub fn bind_checked_http_invocation(
        &mut self,
        certified_binding_sha256: WorkerSha256Digest,
        maximum_requests: u32,
    ) -> Result<(), WorkerProviderStreamError> {
        self.check_active()?;
        if maximum_requests == 0
            || maximum_requests > MAX_WORKER_PROVIDER_HTTP_REQUESTS
            || self.http_invocation.is_some()
            || self.last_admitted_call_id != 0
            || self.handle.is_some()
            || self.request_method.is_some()
            || !self.pending.is_empty()
        {
            return Err(WorkerProviderStreamError::InvalidRequestAuthority);
        }
        self.http_invocation = Some(WorkerProviderHttpInvocationState {
            certified_binding_sha256,
            maximum_requests,
            ordinal: 0,
            completed_receipt_sha256: None,
            armed_head: None,
        });
        Ok(())
    }

    pub fn prepare_checked_http_continuation(
        &mut self,
        context: &WorkerProviderInvocationContext,
        certified_binding_sha256: &WorkerSha256Digest,
        ordinal: u32,
        head: &WorkerProviderRequestHead,
        verify: impl FnOnce(
            &WorkerProviderHttpContinuationChallenge<'_>,
        ) -> Result<(), WorkerProviderStreamError>,
    ) -> Result<(), WorkerProviderStreamError> {
        let result = (|| {
            self.check_invocation_active()?;
            self.validate_context(context)?;
            let invocation = self
                .http_invocation
                .as_ref()
                .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
            if &invocation.certified_binding_sha256 != certified_binding_sha256
                || invocation.armed_head.is_some()
                || !self.terminal
                || !self.pending.is_empty()
                || invocation.ordinal.checked_add(1) != Some(ordinal)
                || ordinal > invocation.maximum_requests
            {
                return Err(WorkerProviderStreamError::InvalidOrder);
            }
            let receipt_sha256 = invocation
                .completed_receipt_sha256
                .as_deref()
                .ok_or(WorkerProviderStreamError::InvalidTerminal)?;
            let handle = self
                .handle
                .ok_or(WorkerProviderStreamError::InvalidHandle)?;
            validate_provider_request_head(head, &self.contract)?;
            // The SHA selects a previously verified receipt in the trusted owner. Terminal
            // bytes themselves must never be promoted into another request's authority.
            verify(&WorkerProviderHttpContinuationChallenge {
                context: &self.context,
                certified_binding_sha256,
                ordinal,
                predecessor_handle: handle,
                predecessor_receipt_sha256: receipt_sha256,
                request_head: head,
            })?;
            self.check_invocation_active()?;
            let invocation = self
                .http_invocation
                .as_mut()
                .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
            invocation.armed_head = Some(head.clone());
            Ok(())
        })();
        if result.is_err() {
            self.revoke();
        }
        result
    }

    pub fn next_http_request_ordinal(&self) -> Result<u32, WorkerProviderStreamError> {
        self.check_invocation_active()?;
        let invocation = self
            .http_invocation
            .as_ref()
            .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
        invocation
            .ordinal
            .checked_add(1)
            .filter(|ordinal| *ordinal <= invocation.maximum_requests)
            .ok_or(WorkerProviderStreamError::InvalidOrder)
    }

    pub fn finish_checked_http_invocation(&mut self) -> Result<(), WorkerProviderStreamError> {
        self.check_invocation_active()?;
        if !self.terminal
            || !self.pending.is_empty()
            || self.http_invocation.as_ref().is_none_or(|invocation| {
                invocation.completed_receipt_sha256.is_none() || invocation.armed_head.is_some()
            })
        {
            return Err(WorkerProviderStreamError::InvalidTerminal);
        }
        self.revoke();
        Ok(())
    }

    fn check_invocation_active(&self) -> Result<(), WorkerProviderStreamError> {
        if self.revoked {
            return Err(WorkerProviderStreamError::RevokedHandle);
        }
        if self.cancellation.is_cancelled() {
            return Err(WorkerProviderStreamError::Cancelled);
        }
        Ok(())
    }

    pub fn validate_request(
        &mut self,
        call_id: u64,
        request: &WorkerProviderStreamRequest,
    ) -> Result<(), WorkerProviderStreamError> {
        let checked_invocation = self.http_invocation.is_some();
        let result = self.validate_request_active(call_id, request);
        if checked_invocation && result.is_err() {
            self.revoke();
        }
        result
    }

    fn validate_request_active(
        &mut self,
        call_id: u64,
        request: &WorkerProviderStreamRequest,
    ) -> Result<(), WorkerProviderStreamError> {
        let continuation = matches!(request, WorkerProviderStreamRequest::StartRequest { .. })
            && self
                .http_invocation
                .as_ref()
                .is_some_and(|invocation| invocation.armed_head.is_some());
        let http_progress = self.terminal
            && self
                .http_invocation
                .as_ref()
                .is_some_and(|invocation| invocation.completed_receipt_sha256.is_some())
            && matches!(request, WorkerProviderStreamRequest::ReportProgress(_));
        let final_progress = self.final_progress_pending
            && matches!(
                request,
                WorkerProviderStreamRequest::ReportProgress(progress)
                    if progress.completed == progress.total
            );
        if final_progress || continuation || http_progress {
            if self.revoked {
                return Err(WorkerProviderStreamError::RevokedHandle);
            }
            if self.cancellation.is_cancelled() {
                return Err(WorkerProviderStreamError::Cancelled);
            }
        } else {
            self.check_active()?;
        }
        if call_id == 0
            || call_id <= self.last_admitted_call_id
            || self.pending.len() >= MAX_WORKER_PROVIDER_PENDING_CALLS
        {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        let pending = match request {
            WorkerProviderStreamRequest::StartRequest { context, head } => {
                self.validate_context(context)?;
                if continuation {
                    let invocation = self
                        .http_invocation
                        .as_ref()
                        .ok_or(WorkerProviderStreamError::InvalidRequestAuthority)?;
                    if invocation.armed_head.as_ref() != Some(head) || !self.pending.is_empty() {
                        self.revoke();
                        return Err(WorkerProviderStreamError::InvalidRequestAuthority);
                    }
                    validate_provider_request_head(head, &self.contract)?;
                    if let Some(handle) = self.handle.take() {
                        self.retired_handles.insert(handle);
                    }
                    self.retired_handles.extend(self.uploads.keys().copied());
                    self.uploads.clear();
                    self.request_sequence = 0;
                    self.request_bytes = 0;
                    self.request_ended = false;
                    self.response_sequence = 0;
                    self.response_bytes = 0;
                    self.response_head_seen = false;
                    self.response_status = None;
                    self.response_retry_after = None;
                    self.last_response_sequence = None;
                    self.terminal = false;
                    self.final_progress_pending = false;
                } else if self.handle.is_some() || self.request_method.is_some() {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                validate_provider_request_head(head, &self.contract)?;
                if let Some(invocation) = self.http_invocation.as_mut() {
                    invocation.ordinal = invocation
                        .ordinal
                        .checked_add(1)
                        .filter(|ordinal| *ordinal <= invocation.maximum_requests)
                        .ok_or(WorkerProviderStreamError::InvalidOrder)?;
                    invocation.armed_head = None;
                    invocation.completed_receipt_sha256 = None;
                }
                self.request_method = Some(head.method);
                self.declared_request_bytes = head.declared_body_bytes;
                WorkerProviderPendingResponse::PrimaryStream
            }
            WorkerProviderStreamRequest::WriteRequestChunk(chunk) => {
                self.validate_request_chunk(chunk)?;
                WorkerProviderPendingResponse::Unit
            }
            WorkerProviderStreamRequest::WaitResponse(request) => {
                self.validate_handle(request.handle)?;
                if request.timeout_milliseconds == 0
                    || request.timeout_milliseconds > self.contract.maximum_wait_milliseconds
                    || request.timeout_milliseconds > MAX_WORKER_PROVIDER_WAIT_MILLISECONDS
                {
                    return Err(WorkerProviderStreamError::WaitLimit);
                }
                if request.after_sequence != self.last_response_sequence {
                    return Err(WorkerProviderStreamError::InvalidSequence);
                }
                WorkerProviderPendingResponse::Wait
            }
            WorkerProviderStreamRequest::StartUpload(request) => {
                self.validate_handle(request.handle)?;
                if !self.contract.uploads {
                    return Err(WorkerProviderStreamError::InvalidUpload);
                }
                validate_upload_request(request, &self.contract)?;
                let reserved_uploads = self
                    .pending
                    .values()
                    .filter(|pending| {
                        matches!(pending, WorkerProviderPendingResponse::UploadStream { .. })
                    })
                    .count();
                let reserved_bytes = self.pending.values().try_fold(0_u64, |total, pending| {
                    let bytes = match pending {
                        WorkerProviderPendingResponse::UploadStream { byte_length, .. } => {
                            *byte_length
                        }
                        _ => 0,
                    };
                    total
                        .checked_add(bytes)
                        .ok_or(WorkerProviderStreamError::InvalidUpload)
                })?;
                let total_uploads = usize::try_from(self.upload_count)
                    .map_err(|_| WorkerProviderStreamError::InvalidUpload)?
                    .checked_add(reserved_uploads)
                    .and_then(|count| count.checked_add(1))
                    .ok_or(WorkerProviderStreamError::InvalidUpload)?;
                let total_upload_bytes = self
                    .upload_bytes
                    .checked_add(reserved_bytes)
                    .and_then(|bytes| bytes.checked_add(request.byte_length))
                    .ok_or(WorkerProviderStreamError::InvalidUpload)?;
                if total_uploads > self.contract.maximum_uploads as usize
                    || total_uploads > MAX_WORKER_PROVIDER_UPLOADS as usize
                    || total_upload_bytes > self.contract.maximum_upload_body_bytes
                {
                    return Err(WorkerProviderStreamError::InvalidUpload);
                }
                WorkerProviderPendingResponse::UploadStream {
                    byte_length: request.byte_length,
                    content_sha256: request.content_sha256.clone(),
                }
            }
            WorkerProviderStreamRequest::WriteUploadChunk(chunk) => {
                self.validate_upload_chunk(chunk)?;
                WorkerProviderPendingResponse::Unit
            }
            WorkerProviderStreamRequest::CompleteUpload(request) => {
                self.validate_handle_identity(request.handle)?;
                let upload = self
                    .uploads
                    .get(&request.handle)
                    .ok_or(WorkerProviderStreamError::ForeignHandle)?;
                if !upload.terminal || upload.completion_received {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                WorkerProviderPendingResponse::UploadCompletion {
                    handle: request.handle,
                }
            }
            WorkerProviderStreamRequest::RetryRequest(request) => {
                self.validate_retry_request(request)?;
                WorkerProviderPendingResponse::RetryStream {
                    old_handle: request.handle,
                }
            }
            WorkerProviderStreamRequest::RequestCost(request) => {
                self.validate_handle(request.handle)?;
                if !self.contract.cost_requests {
                    return Err(WorkerProviderStreamError::InvalidCostRequest);
                }
                if !valid_dotted_identifier(&request.operation)
                    || request.currency.len() != 3
                    || !request
                        .currency
                        .bytes()
                        .all(|byte| byte.is_ascii_uppercase())
                    || request.maximum_microunits == 0
                {
                    return Err(WorkerProviderStreamError::InvalidCostRequest);
                }
                let reserved_costs = self
                    .pending
                    .values()
                    .filter(|pending| matches!(pending, WorkerProviderPendingResponse::Cost { .. }))
                    .count();
                let total_costs = usize::try_from(self.cost_request_count)
                    .map_err(|_| WorkerProviderStreamError::InvalidCostRequest)?
                    .checked_add(reserved_costs)
                    .and_then(|count| count.checked_add(1))
                    .ok_or(WorkerProviderStreamError::InvalidCostRequest)?;
                if total_costs > self.contract.maximum_cost_requests as usize
                    || total_costs > MAX_WORKER_PROVIDER_COST_REQUESTS as usize
                {
                    return Err(WorkerProviderStreamError::InvalidCostRequest);
                }
                WorkerProviderPendingResponse::Cost {
                    maximum_microunits: request.maximum_microunits,
                }
            }
            WorkerProviderStreamRequest::ReportProgress(progress) => {
                self.validate_progress(progress)?;
                if http_progress {
                    WorkerProviderPendingResponse::HttpProgress
                } else if final_progress {
                    WorkerProviderPendingResponse::FinalProgress
                } else {
                    WorkerProviderPendingResponse::Unit
                }
            }
            WorkerProviderStreamRequest::CheckCancelled(handle) => {
                self.validate_handle(*handle)?;
                WorkerProviderPendingResponse::Unit
            }
        };
        self.pending.insert(call_id, pending);
        self.last_admitted_call_id = call_id;
        Ok(())
    }

    pub fn validate_response(
        &mut self,
        call_id: u64,
        response: &WorkerProviderStreamResponse,
    ) -> Result<(), WorkerProviderStreamError> {
        let checked_invocation = self.http_invocation.is_some();
        let result = self.validate_response_active(call_id, response);
        if checked_invocation && (result.is_err() || self.revoked) {
            self.revoke();
        }
        result
    }

    fn validate_response_active(
        &mut self,
        call_id: u64,
        response: &WorkerProviderStreamResponse,
    ) -> Result<(), WorkerProviderStreamError> {
        let final_progress = matches!(
            self.pending.get(&call_id),
            Some(
                WorkerProviderPendingResponse::FinalProgress
                    | WorkerProviderPendingResponse::HttpProgress
            )
        );
        if final_progress {
            if self.revoked {
                return Err(WorkerProviderStreamError::RevokedHandle);
            }
            if self.cancellation.is_cancelled() {
                return Err(WorkerProviderStreamError::Cancelled);
            }
        } else {
            self.check_active()?;
        }
        let expected = self
            .pending
            .get(&call_id)
            .cloned()
            .ok_or(WorkerProviderStreamError::InvalidOrder)?;
        let kind_matches = matches!(
            (&expected, response),
            (
                WorkerProviderPendingResponse::PrimaryStream
                    | WorkerProviderPendingResponse::UploadStream { .. }
                    | WorkerProviderPendingResponse::RetryStream { .. },
                WorkerProviderStreamResponse::Stream(_)
            ) | (
                WorkerProviderPendingResponse::Unit
                    | WorkerProviderPendingResponse::FinalProgress
                    | WorkerProviderPendingResponse::HttpProgress,
                WorkerProviderStreamResponse::Unit(_)
            ) | (
                WorkerProviderPendingResponse::Wait,
                WorkerProviderStreamResponse::Wait(_)
            ) | (
                WorkerProviderPendingResponse::UploadCompletion { .. },
                WorkerProviderStreamResponse::UploadCompletion(_)
            ) | (
                WorkerProviderPendingResponse::Cost { .. },
                WorkerProviderStreamResponse::Cost(_)
            )
        );
        if !kind_matches {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        let response_failed = match response {
            WorkerProviderStreamResponse::Stream(result) => result.is_err(),
            WorkerProviderStreamResponse::Unit(result) => result.is_err(),
            WorkerProviderStreamResponse::Wait(result) => result.is_err(),
            WorkerProviderStreamResponse::UploadCompletion(result) => result.is_err(),
            WorkerProviderStreamResponse::Cost(result) => result.is_err(),
        };
        if response_failed {
            self.pending.remove(&call_id);
            self.revoked = true;
            return Ok(());
        }
        match (expected, response) {
            (
                WorkerProviderPendingResponse::PrimaryStream,
                WorkerProviderStreamResponse::Stream(Ok(handle)),
            ) => {
                self.validate_handle_identity(*handle)?;
                if handle.slot == 0
                    || self.handle.is_some()
                    || self.retired_handles.contains(handle)
                    || self.http_invocation.is_some()
                        && self
                            .retired_handles
                            .iter()
                            .any(|retired| retired.slot >= handle.slot)
                {
                    return Err(WorkerProviderStreamError::InvalidHandle);
                }
                self.handle = Some(*handle);
                self.attempt_count = 1;
            }
            (
                WorkerProviderPendingResponse::RetryStream { old_handle },
                WorkerProviderStreamResponse::Stream(Ok(handle)),
            ) => {
                self.validate_handle_identity(*handle)?;
                if handle.slot == 0
                    || *handle == old_handle
                    || self.retired_handles.contains(handle)
                    || self.uploads.contains_key(handle)
                {
                    return Err(WorkerProviderStreamError::InvalidHandle);
                }
                self.retired_handles.insert(old_handle);
                self.retired_handles.extend(self.uploads.keys().copied());
                self.uploads.clear();
                self.handle = Some(*handle);
                self.request_sequence = 0;
                self.request_bytes = 0;
                self.request_ended = false;
                self.response_sequence = 0;
                self.response_bytes = 0;
                self.response_head_seen = false;
                self.response_status = None;
                self.response_retry_after = None;
                self.last_response_sequence = None;
                self.terminal = false;
                self.final_progress_pending = false;
                self.attempt_count = self
                    .attempt_count
                    .checked_add(1)
                    .ok_or(WorkerProviderStreamError::InvalidOrder)?;
            }
            (
                WorkerProviderPendingResponse::UploadStream {
                    byte_length,
                    content_sha256,
                },
                WorkerProviderStreamResponse::Stream(Ok(handle)),
            ) => {
                self.validate_handle_identity(*handle)?;
                if handle.slot == 0
                    || self.handle == Some(*handle)
                    || self.uploads.contains_key(handle)
                    || self.http_invocation.is_some() && self.retired_handles.contains(handle)
                {
                    return Err(WorkerProviderStreamError::ForeignHandle);
                }
                let upload_count = self
                    .upload_count
                    .checked_add(1)
                    .ok_or(WorkerProviderStreamError::InvalidUpload)?;
                let upload_bytes = self
                    .upload_bytes
                    .checked_add(byte_length)
                    .ok_or(WorkerProviderStreamError::InvalidUpload)?;
                if upload_count > self.contract.maximum_uploads
                    || upload_count > MAX_WORKER_PROVIDER_UPLOADS
                    || upload_bytes > self.contract.maximum_upload_body_bytes
                {
                    return Err(WorkerProviderStreamError::InvalidUpload);
                }
                self.upload_count = upload_count;
                self.upload_bytes = upload_bytes;
                self.uploads.insert(
                    *handle,
                    WorkerProviderUploadState {
                        expected_bytes: byte_length,
                        expected_sha256: content_sha256,
                        received_bytes: 0,
                        next_sequence: 0,
                        digest: Sha256::new(),
                        terminal: false,
                        completion_received: false,
                    },
                );
            }
            (
                WorkerProviderPendingResponse::Wait,
                WorkerProviderStreamResponse::Wait(Ok(outcome)),
            ) => match outcome {
                WorkerProviderWaitOutcome::Frame(frame) => self.validate_response_frame(frame)?,
                WorkerProviderWaitOutcome::TimedOut => {}
                WorkerProviderWaitOutcome::Cancelled => {
                    self.terminal = true;
                    self.revoked = true;
                }
            },
            (
                WorkerProviderPendingResponse::UploadCompletion { handle },
                WorkerProviderStreamResponse::UploadCompletion(Ok(completion)),
            ) => {
                validate_provider_upload_url(&completion.download_url)?;
                let upload = self
                    .uploads
                    .get_mut(&handle)
                    .ok_or(WorkerProviderStreamError::ForeignHandle)?;
                if !upload.terminal || upload.completion_received {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                upload.completion_received = true;
            }
            (
                WorkerProviderPendingResponse::Cost { maximum_microunits },
                WorkerProviderStreamResponse::Cost(Ok(cost)),
            ) => {
                let accepted = cost.accepted
                    && cost.approved_microunits != 0
                    && cost.approved_microunits <= maximum_microunits
                    && !cost.receipt.is_empty()
                    && cost.receipt.len() <= MAX_WORKER_PROVIDER_RECEIPT_BYTES;
                let denied =
                    !cost.accepted && cost.approved_microunits == 0 && cost.receipt.is_empty();
                if !accepted && !denied {
                    return Err(WorkerProviderStreamError::InvalidCostRequest);
                }
                let cost_request_count = self
                    .cost_request_count
                    .checked_add(1)
                    .ok_or(WorkerProviderStreamError::InvalidCostRequest)?;
                if cost_request_count > self.contract.maximum_cost_requests
                    || cost_request_count > MAX_WORKER_PROVIDER_COST_REQUESTS
                {
                    return Err(WorkerProviderStreamError::InvalidCostRequest);
                }
                self.cost_request_count = cost_request_count;
            }
            (
                WorkerProviderPendingResponse::FinalProgress,
                WorkerProviderStreamResponse::Unit(Ok(())),
            ) => {
                self.final_progress_pending = false;
                self.revoked = self.http_invocation.is_none();
            }
            _ => {}
        }
        self.pending.remove(&call_id);
        Ok(())
    }

    pub fn restart(
        self,
        context: WorkerProviderInvocationContext,
        cancellation: CancellationToken,
    ) -> Result<Self, WorkerProviderStreamError> {
        validate_provider_context(&context)?;
        if context.session_id == self.context.session_id
            || context.session_generation <= self.context.session_generation
        {
            return Err(WorkerProviderStreamError::StaleSession);
        }
        if context.invocation == self.context.invocation
            || context.generation <= self.context.generation
        {
            return Err(WorkerProviderStreamError::StaleGeneration);
        }
        let last_admitted_call_id = self.last_admitted_call_id;
        let mut restarted = Self::checked_for_host_session(context, self.contract, cancellation)?;
        restarted.last_admitted_call_id = last_admitted_call_id;
        Ok(restarted)
    }

    pub fn validate_retry_request(
        &self,
        request: &WorkerProviderRetryRequest,
    ) -> Result<(), WorkerProviderStreamError> {
        self.check_active()?;
        self.validate_handle(request.handle)?;
        if !self.request_ended
            || self.response_bytes != 0
            || self.last_response_sequence != Some(0)
            || !self
                .response_status
                .is_some_and(is_retryable_provider_status)
            || request.delay_milliseconds > MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS
            || self.attempt_count >= MAX_WORKER_PROVIDER_ATTEMPTS
            || !self.pending.is_empty()
        {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        Ok(())
    }

    pub fn retry_response_head(
        &self,
        request: &WorkerProviderRetryRequest,
    ) -> Result<(u16, Option<&str>), WorkerProviderStreamError> {
        self.validate_retry_request(request)?;
        let status = self
            .response_status
            .ok_or(WorkerProviderStreamError::InvalidOrder)?;
        Ok((status, self.response_retry_after.as_deref()))
    }

    pub fn revoke(&mut self) {
        self.pending.clear();
        self.http_invocation = None;
        self.revoked = true;
    }

    fn check_active(&self) -> Result<(), WorkerProviderStreamError> {
        if self.revoked {
            Err(WorkerProviderStreamError::RevokedHandle)
        } else if self.cancellation.is_cancelled() {
            Err(WorkerProviderStreamError::Cancelled)
        } else if self.terminal {
            Err(WorkerProviderStreamError::InvalidOrder)
        } else {
            Ok(())
        }
    }

    fn validate_context(
        &self,
        context: &WorkerProviderInvocationContext,
    ) -> Result<(), WorkerProviderStreamError> {
        if context.session_id != self.context.session_id {
            return Err(WorkerProviderStreamError::ForeignSession);
        }
        if context.session_generation != self.context.session_generation {
            return Err(WorkerProviderStreamError::StaleSession);
        }
        if context.invocation != self.context.invocation {
            return Err(WorkerProviderStreamError::ForeignInvocation);
        }
        if context.generation != self.context.generation {
            return Err(WorkerProviderStreamError::StaleGeneration);
        }
        Ok(())
    }

    fn validate_handle_identity(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<(), WorkerProviderStreamError> {
        if handle.session_id != self.context.session_id {
            return Err(WorkerProviderStreamError::ForeignSession);
        }
        if handle.session_generation != self.context.session_generation {
            return Err(WorkerProviderStreamError::StaleSession);
        }
        if handle.invocation != self.context.invocation {
            return Err(WorkerProviderStreamError::ForeignInvocation);
        }
        if handle.generation != self.context.generation {
            return Err(WorkerProviderStreamError::StaleGeneration);
        }
        Ok(())
    }

    fn validate_handle(
        &self,
        handle: WorkerProviderStreamHandle,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validate_handle_identity(handle)?;
        if handle.slot == 0 {
            return Err(WorkerProviderStreamError::InvalidHandle);
        }
        if self.retired_handles.contains(&handle) {
            return Err(WorkerProviderStreamError::RevokedHandle);
        }
        if self.handle != Some(handle) {
            return Err(WorkerProviderStreamError::ForeignHandle);
        }
        Ok(())
    }

    fn validate_request_chunk(
        &mut self,
        chunk: &WorkerProviderRequestChunk,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validate_handle(chunk.handle)?;
        if self.request_ended {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        if chunk.sequence != self.request_sequence {
            return Err(WorkerProviderStreamError::InvalidSequence);
        }
        if chunk.bytes.is_empty() && !chunk.end {
            return Err(WorkerProviderStreamError::ChunkLimit);
        }
        if !chunk.bytes.is_empty() {
            validate_provider_chunk_bytes(&chunk.bytes, self.contract.maximum_chunk_bytes)?;
        }
        let next = self
            .request_bytes
            .checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| WorkerProviderStreamError::BodyLimit)?,
            )
            .ok_or(WorkerProviderStreamError::BodyLimit)?;
        let total = self
            .total_request_bytes
            .checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| WorkerProviderStreamError::BodyLimit)?,
            )
            .ok_or(WorkerProviderStreamError::BodyLimit)?;
        let maximum_total = self
            .contract
            .maximum_request_body_bytes
            .checked_mul(u64::from(MAX_WORKER_PROVIDER_ATTEMPTS))
            .ok_or(WorkerProviderStreamError::BodyLimit)?;
        if next > self.contract.maximum_request_body_bytes
            || total > maximum_total
            || (self.request_method == Some(WorkerProviderHttpMethod::Head) && next != 0)
            || self
                .declared_request_bytes
                .is_some_and(|declared| next > declared)
            || (chunk.end
                && self
                    .declared_request_bytes
                    .is_some_and(|declared| next != declared))
        {
            return Err(WorkerProviderStreamError::BodyLimit);
        }
        self.request_bytes = next;
        self.total_request_bytes = total;
        self.request_sequence = self
            .request_sequence
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidSequence)?;
        self.request_ended = chunk.end;
        Ok(())
    }

    fn validate_response_frame(
        &mut self,
        frame: &WorkerProviderResponseFrame,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validate_handle(frame.handle)?;
        if self.terminal {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        if frame.sequence != self.response_sequence {
            return Err(WorkerProviderStreamError::InvalidSequence);
        }
        let mut response_bytes = self.response_bytes;
        let mut total_response_bytes = self.total_response_bytes;
        let mut completed_receipt_sha256 = None;
        let mut response_head_seen = self.response_head_seen;
        let mut response_status = self.response_status;
        let mut response_retry_after = self.response_retry_after.clone();
        let mut terminal = self.terminal;
        let mut final_progress_pending = self.final_progress_pending;
        match &frame.event {
            WorkerProviderResponseFrameEvent::Head(head) => {
                if !self.request_ended
                    || response_head_seen
                    || frame.sequence != 0
                    || !(200..=599).contains(&head.status)
                {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                validate_provider_headers(&head.headers, &self.contract)?;
                response_retry_after = retry_after_header(&head.headers)?;
                response_head_seen = true;
                response_status = Some(head.status);
            }
            WorkerProviderResponseFrameEvent::Chunk(chunk) => {
                if !response_head_seen
                    || self.request_method == Some(WorkerProviderHttpMethod::Head)
                    || response_status.is_some_and(|status| matches!(status, 204 | 205 | 304))
                {
                    return Err(WorkerProviderStreamError::InvalidOrder);
                }
                let bytes = match chunk {
                    WorkerProviderResponseChunk::Binary(bytes) => bytes.len(),
                    WorkerProviderResponseChunk::Text(text) => text.len(),
                    WorkerProviderResponseChunk::NdjsonLine(line) => {
                        if line.len() > self.contract.maximum_ndjson_line_bytes as usize
                            || line.len() > MAX_WORKER_PROVIDER_NDJSON_LINE_BYTES
                            || line.contains('\n')
                            || line.contains('\r')
                            || serde_json::from_str::<serde_json::Value>(line).is_err()
                        {
                            return Err(WorkerProviderStreamError::InvalidNdjsonLine);
                        }
                        line.len()
                    }
                };
                if bytes == 0
                    || bytes > self.contract.maximum_chunk_bytes as usize
                    || bytes > MAX_WORKER_PROVIDER_CHUNK_BYTES
                {
                    return Err(WorkerProviderStreamError::ChunkLimit);
                }
                let next = response_bytes
                    .checked_add(
                        u64::try_from(bytes).map_err(|_| WorkerProviderStreamError::BodyLimit)?,
                    )
                    .ok_or(WorkerProviderStreamError::BodyLimit)?;
                if next > self.contract.maximum_response_body_bytes {
                    return Err(WorkerProviderStreamError::BodyLimit);
                }
                if self.http_invocation.is_some() {
                    total_response_bytes = total_response_bytes
                        .checked_add(
                            u64::try_from(bytes)
                                .map_err(|_| WorkerProviderStreamError::BodyLimit)?,
                        )
                        .filter(|total| *total <= self.contract.maximum_response_body_bytes)
                        .ok_or(WorkerProviderStreamError::BodyLimit)?;
                }
                response_bytes = next;
            }
            WorkerProviderResponseFrameEvent::Terminal(event_terminal) => {
                validate_provider_terminal(
                    event_terminal,
                    self.request_ended,
                    response_head_seen,
                    self.uploads
                        .values()
                        .all(|upload| upload.terminal && upload.completion_received)
                        && !self.pending.values().any(|pending| {
                            matches!(pending, WorkerProviderPendingResponse::UploadStream { .. })
                        }),
                )?;
                if self.http_invocation.is_some()
                    && let WorkerProviderTerminal::Completed(receipt) = event_terminal
                {
                    let mut digest = Sha256::new();
                    digest.update(b"zed-sim-provider-terminal-completed-receipt-v2\0");
                    digest.update((receipt.len() as u64).to_le_bytes());
                    digest.update(receipt);
                    completed_receipt_sha256 = Some(format!("{:x}", digest.finalize()));
                }
                terminal = true;
                final_progress_pending =
                    matches!(event_terminal, WorkerProviderTerminal::Completed(_));
            }
        }
        let response_sequence = frame
            .sequence
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidSequence)?;
        self.total_response_bytes = total_response_bytes;
        if let Some(receipt_sha256) = completed_receipt_sha256
            && let Some(invocation) = self.http_invocation.as_mut()
        {
            invocation.completed_receipt_sha256 = Some(receipt_sha256);
        }
        self.response_bytes = response_bytes;
        self.response_head_seen = response_head_seen;
        self.response_status = response_status;
        self.response_retry_after = response_retry_after;
        self.terminal = terminal;
        self.final_progress_pending = final_progress_pending;
        self.revoked = terminal && !final_progress_pending;
        self.last_response_sequence = Some(frame.sequence);
        self.response_sequence = response_sequence;
        Ok(())
    }

    fn validate_upload_chunk(
        &mut self,
        chunk: &WorkerProviderRequestChunk,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validate_handle_identity(chunk.handle)?;
        if self.retired_handles.contains(&chunk.handle) {
            return Err(WorkerProviderStreamError::RevokedHandle);
        }
        let upload = self
            .uploads
            .get(&chunk.handle)
            .ok_or(WorkerProviderStreamError::ForeignHandle)?;
        if upload.terminal {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        if chunk.sequence != upload.next_sequence {
            return Err(WorkerProviderStreamError::InvalidSequence);
        }
        if chunk.bytes.is_empty() && !chunk.end {
            return Err(WorkerProviderStreamError::InvalidOrder);
        }
        if !chunk.bytes.is_empty() {
            validate_provider_chunk_bytes(&chunk.bytes, self.contract.maximum_chunk_bytes)?;
        }
        let received_bytes = upload
            .received_bytes
            .checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| WorkerProviderStreamError::BodyLimit)?,
            )
            .ok_or(WorkerProviderStreamError::BodyLimit)?;
        if received_bytes > upload.expected_bytes
            || (chunk.end && received_bytes != upload.expected_bytes)
        {
            return Err(WorkerProviderStreamError::BodyLimit);
        }
        let next_sequence = chunk
            .sequence
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidSequence)?;
        let mut digest = upload.digest.clone();
        digest.update(&chunk.bytes);
        if chunk.end && format!("{:x}", digest.clone().finalize()) != upload.expected_sha256 {
            return Err(WorkerProviderStreamError::InvalidUpload);
        }
        let upload = self
            .uploads
            .get_mut(&chunk.handle)
            .ok_or(WorkerProviderStreamError::ForeignHandle)?;
        upload.received_bytes = received_bytes;
        upload.next_sequence = next_sequence;
        upload.digest = digest;
        upload.terminal = chunk.end;
        Ok(())
    }

    fn validate_progress(
        &mut self,
        progress: &WorkerProviderProgress,
    ) -> Result<(), WorkerProviderStreamError> {
        self.validate_handle(progress.handle)?;
        if progress.sequence != self.progress_sequence
            || progress.completed < self.progress_completed
            || progress.completed > progress.total
            || progress.total == 0
            || progress.total > self.contract.maximum_progress_total
            || progress.total > MAX_WORKER_PROVIDER_PROGRESS_TOTAL
            || self
                .progress_total
                .is_some_and(|total| total != progress.total)
            || progress.message.as_ref().is_some_and(|message| {
                message.is_empty() || message.len() > MAX_WORKER_PROVIDER_PROGRESS_MESSAGE_BYTES
            })
        {
            return Err(WorkerProviderStreamError::InvalidProgress);
        }
        self.progress_sequence = self
            .progress_sequence
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidProgress)?;
        self.progress_completed = progress.completed;
        self.progress_total = Some(progress.total);
        Ok(())
    }
}

fn validate_provider_context(
    context: &WorkerProviderInvocationContext,
) -> Result<(), WorkerProviderStreamError> {
    if context.session_id.is_nil()
        || context.session_generation == 0
        || context.invocation == 0
        || context.generation == 0
    {
        Err(WorkerProviderStreamError::InvalidHandle)
    } else {
        Ok(())
    }
}

fn validate_provider_contract(
    contract: &WorkerProviderStreamingContract,
) -> Result<(), WorkerProviderStreamError> {
    if contract.methods.is_empty()
        || contract.methods.len() > 7
        || contract
            .methods
            .windows(2)
            .any(|methods| methods[0] >= methods[1])
        || contract.maximum_headers == 0
        || usize::from(contract.maximum_headers) > MAX_WORKER_PROVIDER_HEADERS
        || contract.maximum_header_bytes == 0
        || contract.maximum_header_bytes as usize > MAX_WORKER_PROVIDER_HEADER_BYTES
        || contract.maximum_request_body_bytes == 0
        || contract.maximum_request_body_bytes > MAX_WORKER_PROVIDER_BODY_BYTES
        || contract.maximum_response_body_bytes == 0
        || contract.maximum_response_body_bytes > MAX_WORKER_PROVIDER_BODY_BYTES
        || contract.maximum_chunk_bytes == 0
        || contract.maximum_chunk_bytes as usize > MAX_WORKER_PROVIDER_CHUNK_BYTES
        || contract.maximum_ndjson_line_bytes == 0
        || contract.maximum_ndjson_line_bytes > contract.maximum_chunk_bytes
        || contract.maximum_ndjson_line_bytes as usize > MAX_WORKER_PROVIDER_NDJSON_LINE_BYTES
        || contract.maximum_wait_milliseconds == 0
        || contract.maximum_wait_milliseconds > MAX_WORKER_PROVIDER_WAIT_MILLISECONDS
        || contract.maximum_uploads > MAX_WORKER_PROVIDER_UPLOADS
        || contract.maximum_upload_body_bytes > MAX_WORKER_PROVIDER_BODY_BYTES
        || contract.maximum_cost_requests > MAX_WORKER_PROVIDER_COST_REQUESTS
        || contract.maximum_progress_total == 0
        || contract.maximum_progress_total > MAX_WORKER_PROVIDER_PROGRESS_TOTAL
        || contract.uploads
            != (contract.maximum_uploads != 0 && contract.maximum_upload_body_bytes != 0)
        || (!contract.uploads
            && (contract.maximum_uploads != 0 || contract.maximum_upload_body_bytes != 0))
        || contract.cost_requests != (contract.maximum_cost_requests != 0)
    {
        return Err(WorkerProviderStreamError::InvalidContract);
    }
    Ok(())
}

fn validate_provider_request_head(
    head: &WorkerProviderRequestHead,
    contract: &WorkerProviderStreamingContract,
) -> Result<(), WorkerProviderStreamError> {
    if !valid_provider_request_authority(&head.endpoint, MAX_WORKER_PROVIDER_ENDPOINT_BYTES)
        || head.secret_id.as_deref().is_some_and(|secret_id| {
            !valid_provider_request_authority(secret_id, MAX_WORKER_PROVIDER_SECRET_ID_BYTES)
        })
    {
        return Err(WorkerProviderStreamError::InvalidRequestAuthority);
    }
    if !contract.methods.contains(&head.method) {
        return Err(WorkerProviderStreamError::InvalidMethod);
    }
    validate_provider_headers(&head.headers, contract)?;
    if head
        .declared_body_bytes
        .is_some_and(|bytes| bytes > contract.maximum_request_body_bytes)
        || (head.method == WorkerProviderHttpMethod::Head
            && head.declared_body_bytes.is_some_and(|bytes| bytes != 0))
    {
        return Err(WorkerProviderStreamError::BodyLimit);
    }
    Ok(())
}

fn validate_provider_headers(
    headers: &[WorkerProviderHeader],
    contract: &WorkerProviderStreamingContract,
) -> Result<(), WorkerProviderStreamError> {
    if headers.len() > usize::from(contract.maximum_headers)
        || headers.len() > MAX_WORKER_PROVIDER_HEADERS
    {
        return Err(WorkerProviderStreamError::InvalidHeaders);
    }
    let mut total = 0_usize;
    for header in headers {
        if header.name.is_empty()
            || header.name.len() > MAX_WORKER_PROVIDER_HEADER_NAME_BYTES
            || !header.name.bytes().all(|byte| {
                byte.is_ascii_alphanumeric()
                    || matches!(
                        byte,
                        b'!' | b'#'
                            ..=b'\'' | b'*' | b'+' | b'-' | b'.' | b'^' | b'_' | b'`' | b'|' | b'~'
                    )
            })
            || header.value.len() > MAX_WORKER_PROVIDER_HEADER_VALUE_BYTES
            || header.value.bytes().any(|byte| {
                byte == b'\r' || byte == b'\n' || byte == 0x7f || (byte < 0x20 && byte != b'\t')
            })
        {
            return Err(WorkerProviderStreamError::InvalidHeaders);
        }
        total = total
            .checked_add(header.name.len())
            .and_then(|value| value.checked_add(header.value.len()))
            .ok_or(WorkerProviderStreamError::InvalidHeaders)?;
    }
    if total > contract.maximum_header_bytes as usize || total > MAX_WORKER_PROVIDER_HEADER_BYTES {
        return Err(WorkerProviderStreamError::InvalidHeaders);
    }
    Ok(())
}

fn validate_provider_chunk_bytes(
    bytes: &[u8],
    contract_maximum: u32,
) -> Result<(), WorkerProviderStreamError> {
    if bytes.is_empty()
        || bytes.len() > contract_maximum as usize
        || bytes.len() > MAX_WORKER_PROVIDER_CHUNK_BYTES
    {
        Err(WorkerProviderStreamError::ChunkLimit)
    } else {
        Ok(())
    }
}

fn validate_upload_request(
    request: &WorkerProviderUploadRequest,
    contract: &WorkerProviderStreamingContract,
) -> Result<(), WorkerProviderStreamError> {
    if !valid_dotted_identifier(&request.port_id)
        || !valid_media_type(&request.media_type)
        || request.byte_length == 0
        || request.byte_length > contract.maximum_request_body_bytes
        || request.byte_length > contract.maximum_upload_body_bytes
        || request.content_sha256.len() != 64
        || !request
            .content_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Err(WorkerProviderStreamError::InvalidUpload)
    } else {
        Ok(())
    }
}

fn validate_provider_terminal(
    terminal: &WorkerProviderTerminal,
    request_finished: bool,
    response_started: bool,
    uploads_finished: bool,
) -> Result<(), WorkerProviderStreamError> {
    match terminal {
        WorkerProviderTerminal::Completed(receipt)
            if !request_finished
                || !response_started
                || !uploads_finished
                || receipt.is_empty()
                || receipt.len() > MAX_WORKER_PROVIDER_RECEIPT_BYTES =>
        {
            Err(WorkerProviderStreamError::InvalidTerminal)
        }
        WorkerProviderTerminal::Failed { code, message }
            if !valid_dotted_identifier(code)
                || message.is_empty()
                || message.len() > MAX_WORKER_PROVIDER_HEADER_VALUE_BYTES =>
        {
            Err(WorkerProviderStreamError::InvalidTerminal)
        }
        _ => Ok(()),
    }
}

fn valid_provider_request_authority(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_bytes
        && value.is_ascii()
        && value.trim() == value
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn valid_dotted_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.split('.').all(|segment| {
            !segment.is_empty()
                && segment.len() <= 64
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
                && segment
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
        })
}

fn valid_media_type(value: &str) -> bool {
    let Some((kind, subtype)) = value.split_once('/') else {
        return false;
    };
    !kind.is_empty()
        && !subtype.is_empty()
        && value.len() <= 256
        && kind.bytes().all(valid_http_token_byte)
        && subtype.bytes().all(valid_http_token_byte)
}

fn valid_http_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceContext {
    pub session_id: Uuid,
    pub attempt_id: AttemptId,
    pub attempt_generation: u64,
    pub node_id: String,
    pub node_generation: u64,
    pub service_id: Uuid,
    pub service_generation: u64,
    pub ordered_source_identity_sha256: WorkerSha256Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelSourceOperation {
    Open {
        folder_category: String,
        source_names: Vec<String>,
    },
    Read {
        source_ordinal: u32,
        tensor_ordinal: u32,
        byte_offset: u64,
        byte_length: u32,
    },
    Close,
    ReadArtifact {
        source_ordinal: u32,
        artifact_ordinal: u32,
        artifact_sha256: WorkerSha256Digest,
        byte_offset: u64,
        byte_length: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceRequest {
    pub context: WorkerModelSourceContext,
    pub call_ordinal: u64,
    pub operation: WorkerModelSourceOperation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceManifest {
    pub source_ordinal: u32,
    pub model_identity_sha256: WorkerSha256Digest,
    pub artifacts: Vec<WorkerModelSourceArtifact>,
    pub tensors: Vec<WorkerModelSourceTensor>,
    pub aggregate_tensor_bytes: u64,
    pub maximum_read_bytes: u64,
    pub parser_limits: WorkerModelSourceParserLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelSourceFormat {
    Safetensors,
    PytorchArchive,
    Gguf,
    JsonConfig,
    JsonTokenizer,
    YamlConfig,
    SentencePiece,
    Tiktoken,
    SafetensorsMetadata { source_artifact_ordinal: u32 },
    ClipConfiguration { target_ordinal: u8 },
    ClipTokenizerJson { target_ordinal: u8 },
    ClipSentencePiece { target_ordinal: u8 },
    ClipTiktoken { target_ordinal: u8 },
}

impl WorkerModelSourceFormat {
    pub const fn supports_auxiliary_artifact_stream(self) -> bool {
        matches!(
            self,
            Self::JsonConfig
                | Self::JsonTokenizer
                | Self::YamlConfig
                | Self::SentencePiece
                | Self::Tiktoken
                | Self::SafetensorsMetadata { .. }
                | Self::ClipConfiguration { .. }
                | Self::ClipTokenizerJson { .. }
                | Self::ClipSentencePiece { .. }
                | Self::ClipTiktoken { .. }
        )
    }

    pub const fn valid(self) -> bool {
        match self {
            Self::ClipConfiguration { target_ordinal }
            | Self::ClipTokenizerJson { target_ordinal }
            | Self::ClipSentencePiece { target_ordinal }
            | Self::ClipTiktoken { target_ordinal } => target_ordinal < 4,
            Self::SafetensorsMetadata { .. }
            | Self::Safetensors
            | Self::PytorchArchive
            | Self::Gguf
            | Self::JsonConfig
            | Self::JsonTokenizer
            | Self::YamlConfig
            | Self::SentencePiece
            | Self::Tiktoken => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelSourceNestedStateDisposition {
    Flat,
    NestedStringToParam,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceArtifact {
    pub sha256: WorkerSha256Digest,
    pub byte_size: u64,
    pub format: WorkerModelSourceFormat,
    pub nested_state_disposition: WorkerModelSourceNestedStateDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceTensor {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<u64>,
    pub artifact_ordinal: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceParserLimits {
    pub version: u32,
    pub manifest_bytes: u64,
    pub maximum_depth: u32,
    pub maximum_tensors: u64,
    pub maximum_tensor_bytes: u64,
    pub maximum_aggregate_tensor_bytes: u64,
    pub maximum_name_bytes: u64,
    pub maximum_archive_entries: u64,
    pub maximum_metadata_values: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceOpened {
    pub session_id: Uuid,
    pub call_ordinal: u64,
    pub ordered_source_identity_sha256: WorkerSha256Digest,
    pub sources: Vec<WorkerModelSourceManifest>,
    pub maximum_chunk_bytes: u32,
    pub response_sha256: WorkerSha256Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceChunk {
    pub session_id: Uuid,
    pub call_ordinal: u64,
    pub source_ordinal: u32,
    pub tensor_ordinal: u32,
    pub byte_offset: u64,
    pub bytes: Vec<u8>,
    pub response_sha256: WorkerSha256Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceArtifactChunk {
    pub session_id: Uuid,
    pub call_ordinal: u64,
    pub source_ordinal: u32,
    pub artifact_ordinal: u32,
    pub artifact_sha256: WorkerSha256Digest,
    pub byte_offset: u64,
    pub bytes: Vec<u8>,
    pub response_sha256: WorkerSha256Digest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerModelSourceClosed {
    pub session_id: Uuid,
    pub call_ordinal: u64,
    pub response_sha256: WorkerSha256Digest,
}

#[derive(Clone, Debug, Error, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelSourceError {
    #[error("model source operation was cancelled")]
    Cancelled,
    #[error("model source call order is invalid")]
    InvalidOrder,
    #[error("model source call was replayed")]
    Replay,
    #[error("model source call is late")]
    Late,
    #[error("model source session is foreign")]
    ForeignSession,
    #[error("model source attempt identity or generation is stale")]
    StaleAttempt,
    #[error("model source node identity or generation is stale")]
    StaleNode,
    #[error("model source service identity or generation is stale")]
    StaleService,
    #[error("model source selection identity is invalid")]
    InvalidSourceSelection,
    #[error("model source ordinal is invalid")]
    InvalidSourceOrdinal,
    #[error("model source tensor ordinal is invalid")]
    InvalidTensorOrdinal,
    #[error("model source range exceeds its bound")]
    RangeLimit,
    #[error("model source chunk exceeds its fixed bound")]
    ChunkLimit,
    #[error("model source response digest is invalid")]
    DigestMismatch,
    #[error("model source bytes changed")]
    SourceChanged,
    #[error("model source session is closed")]
    Closed,
    #[error("model source host failed")]
    HostFailure,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerModelSourceResponse {
    Opened(WorkerModelSourceOpened),
    Chunk(WorkerModelSourceChunk),
    Closed(WorkerModelSourceClosed),
    Rejected {
        session_id: Uuid,
        call_ordinal: u64,
        error: WorkerModelSourceError,
        response_sha256: WorkerSha256Digest,
    },
    ArtifactChunk(WorkerModelSourceArtifactChunk),
}

impl WorkerModelSourceOpened {
    pub fn checked(
        session_id: Uuid,
        call_ordinal: u64,
        ordered_source_identity_sha256: WorkerSha256Digest,
        sources: Vec<WorkerModelSourceManifest>,
    ) -> Result<Self, WorkerModelSourceError> {
        let maximum_chunk_bytes = u32::try_from(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)
            .map_err(|_| WorkerModelSourceError::ChunkLimit)?;
        let response_sha256 = model_source_opened_digest(
            session_id,
            call_ordinal,
            &ordered_source_identity_sha256,
            &sources,
            maximum_chunk_bytes,
        )?;
        let opened = Self {
            session_id,
            call_ordinal,
            ordered_source_identity_sha256,
            sources,
            maximum_chunk_bytes,
            response_sha256,
        };
        opened.validate()?;
        Ok(opened)
    }

    pub fn validate(&self) -> Result<(), WorkerModelSourceError> {
        if self.session_id.is_nil()
            || self.call_ordinal == 0
            || self.sources.is_empty()
            || self.sources.len() > MAX_WORKER_MODEL_SOURCE_SELECTIONS
            || usize::try_from(self.maximum_chunk_bytes).ok()
                != Some(MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)
        {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let mut manifest_wire_bytes = 0_usize;
        for (source_ordinal, source) in self.sources.iter().enumerate() {
            let limits = source.parser_limits;
            if usize::try_from(source.source_ordinal).ok() != Some(source_ordinal)
                || source.artifacts.is_empty()
                || source.artifacts.len() > MAX_WORKER_MODEL_SOURCE_ARTIFACTS
                || source.tensors.len() > MAX_WORKER_MODEL_SOURCE_TENSORS
                || (source.tensors.is_empty()
                    && source
                        .artifacts
                        .iter()
                        .any(|artifact| !artifact.format.supports_auxiliary_artifact_stream()))
                || source.maximum_read_bytes < u64::from(self.maximum_chunk_bytes)
                || limits.version == 0
                || limits.manifest_bytes == 0
                || limits.maximum_depth == 0
                || limits.maximum_tensors == 0
                || limits.maximum_tensor_bytes == 0
                || limits.maximum_aggregate_tensor_bytes == 0
                || limits.maximum_name_bytes == 0
                || limits.maximum_archive_entries == 0
                || limits.maximum_metadata_values == 0
                || limits.maximum_tensor_bytes > limits.maximum_aggregate_tensor_bytes
                || limits.maximum_tensors > limits.maximum_archive_entries
                || limits.manifest_bytes.checked_mul(8).is_none()
                || source.maximum_read_bytes > limits.maximum_tensor_bytes
                || limits.maximum_tensors < u64::try_from(source.tensors.len()).unwrap_or(u64::MAX)
                || limits.maximum_archive_entries
                    < u64::try_from(source.artifacts.len()).unwrap_or(u64::MAX)
            {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
            if source.artifacts.iter().any(|artifact| {
                artifact.byte_size == 0
                    || !artifact.format.valid()
                    || (artifact.format.supports_auxiliary_artifact_stream()
                        && artifact.byte_size > limits.manifest_bytes)
            }) {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
            let mut safetensors_metadata_sources = BTreeSet::new();
            for artifact in &source.artifacts {
                let WorkerModelSourceFormat::SafetensorsMetadata {
                    source_artifact_ordinal,
                } = artifact.format
                else {
                    continue;
                };
                let source_artifact = usize::try_from(source_artifact_ordinal)
                    .ok()
                    .and_then(|ordinal| source.artifacts.get(ordinal));
                if !source_artifact.is_some_and(|source_artifact| {
                    source_artifact.format == WorkerModelSourceFormat::Safetensors
                }) || !safetensors_metadata_sources.insert(source_artifact_ordinal)
                {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
            }
            manifest_wire_bytes = manifest_wire_bytes
                .checked_add(source.artifacts.len().saturating_mul(128))
                .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
            let mut names = BTreeMap::new();
            let mut ranges = BTreeMap::<u32, Vec<(u64, u64)>>::new();
            let mut aggregate_tensor_bytes = 0_u64;
            for tensor in &source.tensors {
                if tensor.name.is_empty()
                    || u64::try_from(tensor.name.len()).unwrap_or(u64::MAX)
                        > limits.maximum_name_bytes
                    || tensor.data_type.is_empty()
                    || tensor.name.len() > MAX_WORKER_MODEL_SOURCE_NAME_BYTES
                    || tensor.data_type.len() > MAX_WORKER_MODEL_SOURCE_DTYPE_BYTES
                    || u64::try_from(tensor.data_type.len()).unwrap_or(u64::MAX)
                        > limits.maximum_name_bytes
                    || tensor.shape.len() > MAX_WORKER_MODEL_SOURCE_TENSOR_DIMENSIONS
                    || tensor.shape.len()
                        > usize::try_from(limits.maximum_depth).unwrap_or(usize::MAX)
                    || usize::try_from(tensor.artifact_ordinal)
                        .map_or(true, |ordinal| ordinal >= source.artifacts.len())
                    || tensor.byte_length == 0
                    || tensor.byte_length > limits.maximum_tensor_bytes
                    || names.insert(tensor.name.as_str(), ()).is_some()
                {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
                manifest_wire_bytes = manifest_wire_bytes
                    .checked_add(tensor.name.len())
                    .and_then(|bytes| bytes.checked_add(tensor.data_type.len()))
                    .and_then(|bytes| {
                        bytes.checked_add(tensor.shape.len().saturating_mul(size_of::<u64>()))
                    })
                    .and_then(|bytes| bytes.checked_add(64))
                    .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
                if manifest_wire_bytes > MAX_WORKER_MODEL_SOURCE_MANIFEST_WIRE_BYTES {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
                let range_end = tensor
                    .byte_offset
                    .checked_add(tensor.byte_length)
                    .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
                let artifact = source
                    .artifacts
                    .get(
                        usize::try_from(tensor.artifact_ordinal)
                            .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?,
                    )
                    .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
                if range_end > artifact.byte_size {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
                if !matches!(
                    artifact.format,
                    WorkerModelSourceFormat::Safetensors
                        | WorkerModelSourceFormat::PytorchArchive
                        | WorkerModelSourceFormat::Gguf
                ) {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
                aggregate_tensor_bytes = aggregate_tensor_bytes
                    .checked_add(tensor.byte_length)
                    .ok_or(WorkerModelSourceError::InvalidSourceSelection)?;
                ranges
                    .entry(tensor.artifact_ordinal)
                    .or_default()
                    .push((tensor.byte_offset, range_end));
            }
            if aggregate_tensor_bytes != source.aggregate_tensor_bytes
                || aggregate_tensor_bytes > limits.maximum_aggregate_tensor_bytes
            {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
            for artifact_ranges in ranges.values_mut() {
                artifact_ranges.sort_unstable();
                if artifact_ranges
                    .windows(2)
                    .any(|ranges| ranges[0].1 > ranges[1].0)
                {
                    return Err(WorkerModelSourceError::InvalidSourceSelection);
                }
            }
        }
        let expected = model_source_opened_digest(
            self.session_id,
            self.call_ordinal,
            &self.ordered_source_identity_sha256,
            &self.sources,
            self.maximum_chunk_bytes,
        )?;
        if expected != self.response_sha256 {
            return Err(WorkerModelSourceError::DigestMismatch);
        }
        Ok(())
    }
}

impl WorkerModelSourceChunk {
    pub fn checked(
        session_id: Uuid,
        call_ordinal: u64,
        source_ordinal: u32,
        tensor_ordinal: u32,
        byte_offset: u64,
        bytes: Vec<u8>,
    ) -> Result<Self, WorkerModelSourceError> {
        if bytes.is_empty() || bytes.len() > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES {
            return Err(WorkerModelSourceError::ChunkLimit);
        }
        let response_sha256 = model_source_chunk_digest(
            session_id,
            call_ordinal,
            source_ordinal,
            tensor_ordinal,
            byte_offset,
            &bytes,
        )?;
        Ok(Self {
            session_id,
            call_ordinal,
            source_ordinal,
            tensor_ordinal,
            byte_offset,
            bytes,
            response_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), WorkerModelSourceError> {
        if self.session_id.is_nil() || self.call_ordinal == 0 {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        if self.bytes.is_empty() || self.bytes.len() > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES {
            return Err(WorkerModelSourceError::ChunkLimit);
        }
        let expected = model_source_chunk_digest(
            self.session_id,
            self.call_ordinal,
            self.source_ordinal,
            self.tensor_ordinal,
            self.byte_offset,
            &self.bytes,
        )?;
        if expected != self.response_sha256 {
            return Err(WorkerModelSourceError::DigestMismatch);
        }
        Ok(())
    }
}

impl WorkerModelSourceArtifactChunk {
    pub fn checked(
        session_id: Uuid,
        call_ordinal: u64,
        source_ordinal: u32,
        artifact_ordinal: u32,
        artifact_sha256: WorkerSha256Digest,
        byte_offset: u64,
        bytes: Vec<u8>,
    ) -> Result<Self, WorkerModelSourceError> {
        if bytes.is_empty() || bytes.len() > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES {
            return Err(WorkerModelSourceError::ChunkLimit);
        }
        let response_sha256 = model_source_artifact_chunk_digest(
            session_id,
            call_ordinal,
            source_ordinal,
            artifact_ordinal,
            &artifact_sha256,
            byte_offset,
            &bytes,
        )?;
        Ok(Self {
            session_id,
            call_ordinal,
            source_ordinal,
            artifact_ordinal,
            artifact_sha256,
            byte_offset,
            bytes,
            response_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), WorkerModelSourceError> {
        if self.session_id.is_nil() || self.call_ordinal == 0 {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        if self.bytes.is_empty() || self.bytes.len() > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES {
            return Err(WorkerModelSourceError::ChunkLimit);
        }
        let expected = model_source_artifact_chunk_digest(
            self.session_id,
            self.call_ordinal,
            self.source_ordinal,
            self.artifact_ordinal,
            &self.artifact_sha256,
            self.byte_offset,
            &self.bytes,
        )?;
        if expected != self.response_sha256 {
            return Err(WorkerModelSourceError::DigestMismatch);
        }
        Ok(())
    }
}

impl WorkerModelSourceClosed {
    pub fn checked(session_id: Uuid, call_ordinal: u64) -> Result<Self, WorkerModelSourceError> {
        if session_id.is_nil() || call_ordinal == 0 {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        Ok(Self {
            session_id,
            call_ordinal,
            response_sha256: model_source_closed_digest(session_id, call_ordinal)?,
        })
    }

    pub fn validate(&self) -> Result<(), WorkerModelSourceError> {
        if self.session_id.is_nil() || self.call_ordinal == 0 {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        if self.response_sha256 != model_source_closed_digest(self.session_id, self.call_ordinal)? {
            return Err(WorkerModelSourceError::DigestMismatch);
        }
        Ok(())
    }
}

impl WorkerModelSourceResponse {
    pub fn rejected(
        session_id: Uuid,
        call_ordinal: u64,
        error: WorkerModelSourceError,
    ) -> Result<Self, WorkerModelSourceError> {
        if session_id.is_nil() || call_ordinal == 0 {
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let response_sha256 = model_source_rejected_digest(session_id, call_ordinal, &error)?;
        Ok(Self::Rejected {
            session_id,
            call_ordinal,
            error,
            response_sha256,
        })
    }

    pub fn validate(&self) -> Result<(), WorkerModelSourceError> {
        match self {
            Self::Opened(opened) => opened.validate(),
            Self::Chunk(chunk) => chunk.validate(),
            Self::Closed(closed) => closed.validate(),
            Self::Rejected {
                session_id,
                call_ordinal,
                error,
                response_sha256,
            } => {
                if session_id.is_nil() || *call_ordinal == 0 {
                    return Err(WorkerModelSourceError::InvalidOrder);
                }
                if *response_sha256
                    != model_source_rejected_digest(*session_id, *call_ordinal, error)?
                {
                    return Err(WorkerModelSourceError::DigestMismatch);
                }
                Ok(())
            }
            Self::ArtifactChunk(chunk) => chunk.validate(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum WorkerModelSourcePendingOperation {
    Open {
        source_count: usize,
    },
    Read {
        source_ordinal: u32,
        tensor_ordinal: u32,
        byte_offset: u64,
        byte_length: u32,
    },
    Close,
    ReadArtifact {
        source_ordinal: u32,
        artifact_ordinal: u32,
        artifact_sha256: WorkerSha256Digest,
        byte_offset: u64,
        byte_length: u32,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorkerModelSourceTransportState {
    Initial,
    Open,
    Closed,
}

#[derive(Debug)]
pub struct WorkerModelSourceTransportValidator {
    context: WorkerModelSourceContext,
    next_call_ordinal: u64,
    pending: Option<(u64, WorkerModelSourcePendingOperation)>,
    state: WorkerModelSourceTransportState,
}

impl WorkerModelSourceTransportValidator {
    pub fn checked(context: WorkerModelSourceContext) -> Result<Self, WorkerModelSourceError> {
        validate_model_source_context(&context)?;
        Ok(Self {
            context,
            next_call_ordinal: 1,
            pending: None,
            state: WorkerModelSourceTransportState::Initial,
        })
    }

    pub fn validate_request(
        &mut self,
        call_id: u64,
        request: &WorkerModelSourceRequest,
    ) -> Result<(), WorkerModelSourceError> {
        let result = self.validate_request_inner(call_id, request);
        if result.is_err() {
            self.revoke();
        }
        result
    }

    fn validate_request_inner(
        &mut self,
        call_id: u64,
        request: &WorkerModelSourceRequest,
    ) -> Result<(), WorkerModelSourceError> {
        if self.state == WorkerModelSourceTransportState::Closed {
            return Err(WorkerModelSourceError::Closed);
        }
        if self.pending.is_some() {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        validate_model_source_request_wire(request)?;
        validate_model_source_context_match(&self.context, &request.context)?;
        if request.call_ordinal < self.next_call_ordinal {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::Replay);
        }
        if request.call_ordinal != self.next_call_ordinal || call_id == 0 {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        let operation = match &request.operation {
            WorkerModelSourceOperation::Open { source_names, .. }
                if self.state == WorkerModelSourceTransportState::Initial =>
            {
                WorkerModelSourcePendingOperation::Open {
                    source_count: source_names.len(),
                }
            }
            WorkerModelSourceOperation::Read {
                source_ordinal,
                tensor_ordinal,
                byte_offset,
                byte_length,
            } if self.state == WorkerModelSourceTransportState::Open => {
                WorkerModelSourcePendingOperation::Read {
                    source_ordinal: *source_ordinal,
                    tensor_ordinal: *tensor_ordinal,
                    byte_offset: *byte_offset,
                    byte_length: *byte_length,
                }
            }
            WorkerModelSourceOperation::Close
                if self.state == WorkerModelSourceTransportState::Open =>
            {
                WorkerModelSourcePendingOperation::Close
            }
            WorkerModelSourceOperation::ReadArtifact {
                source_ordinal,
                artifact_ordinal,
                artifact_sha256,
                byte_offset,
                byte_length,
            } if self.state == WorkerModelSourceTransportState::Open => {
                WorkerModelSourcePendingOperation::ReadArtifact {
                    source_ordinal: *source_ordinal,
                    artifact_ordinal: *artifact_ordinal,
                    artifact_sha256: artifact_sha256.clone(),
                    byte_offset: *byte_offset,
                    byte_length: *byte_length,
                }
            }
            _ => {
                self.state = WorkerModelSourceTransportState::Closed;
                return Err(WorkerModelSourceError::InvalidOrder);
            }
        };
        self.pending = Some((call_id, operation));
        Ok(())
    }

    pub fn validate_response(
        &mut self,
        call_id: u64,
        response: &WorkerModelSourceResponse,
    ) -> Result<(), WorkerModelSourceError> {
        let result = self.validate_response_inner(call_id, response);
        if result.is_err() {
            self.revoke();
        }
        result
    }

    fn validate_response_inner(
        &mut self,
        call_id: u64,
        response: &WorkerModelSourceResponse,
    ) -> Result<(), WorkerModelSourceError> {
        if self.state == WorkerModelSourceTransportState::Closed {
            return Err(WorkerModelSourceError::Closed);
        }
        let Some((pending_call_id, pending_operation)) = self.pending.take() else {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::Late);
        };
        if call_id != pending_call_id {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        response.validate()?;
        let (session_id, call_ordinal) = model_source_response_identity(response);
        if session_id != self.context.session_id || call_ordinal != self.next_call_ordinal {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::ForeignSession);
        }
        if matches!(response, WorkerModelSourceResponse::Rejected { .. }) {
            self.state = WorkerModelSourceTransportState::Closed;
            return Ok(());
        }
        let matches_operation = match (&pending_operation, response) {
            (
                WorkerModelSourcePendingOperation::Open { source_count },
                WorkerModelSourceResponse::Opened(opened),
            ) => {
                opened.ordered_source_identity_sha256 == self.context.ordered_source_identity_sha256
                    && opened.sources.len() == *source_count
            }
            (
                WorkerModelSourcePendingOperation::Read {
                    source_ordinal,
                    tensor_ordinal,
                    byte_offset,
                    byte_length,
                },
                WorkerModelSourceResponse::Chunk(chunk),
            ) => {
                chunk.source_ordinal == *source_ordinal
                    && chunk.tensor_ordinal == *tensor_ordinal
                    && chunk.byte_offset == *byte_offset
                    && usize::try_from(*byte_length).ok() == Some(chunk.bytes.len())
            }
            (WorkerModelSourcePendingOperation::Close, WorkerModelSourceResponse::Closed(_)) => {
                true
            }
            (
                WorkerModelSourcePendingOperation::ReadArtifact {
                    source_ordinal,
                    artifact_ordinal,
                    artifact_sha256,
                    byte_offset,
                    byte_length,
                },
                WorkerModelSourceResponse::ArtifactChunk(chunk),
            ) => {
                chunk.source_ordinal == *source_ordinal
                    && chunk.artifact_ordinal == *artifact_ordinal
                    && chunk.artifact_sha256 == *artifact_sha256
                    && chunk.byte_offset == *byte_offset
                    && usize::try_from(*byte_length).ok() == Some(chunk.bytes.len())
            }
            _ => false,
        };
        if !matches_operation {
            self.state = WorkerModelSourceTransportState::Closed;
            return Err(WorkerModelSourceError::InvalidOrder);
        }
        self.next_call_ordinal = self
            .next_call_ordinal
            .checked_add(1)
            .ok_or(WorkerModelSourceError::InvalidOrder)?;
        self.state = match pending_operation {
            WorkerModelSourcePendingOperation::Open { .. }
            | WorkerModelSourcePendingOperation::Read { .. }
            | WorkerModelSourcePendingOperation::ReadArtifact { .. } => {
                WorkerModelSourceTransportState::Open
            }
            WorkerModelSourcePendingOperation::Close => WorkerModelSourceTransportState::Closed,
        };
        Ok(())
    }

    pub fn revoke(&mut self) {
        self.pending = None;
        self.state = WorkerModelSourceTransportState::Closed;
    }

    pub const fn is_closed(&self) -> bool {
        matches!(self.state, WorkerModelSourceTransportState::Closed)
    }
}

fn validate_model_source_context(
    context: &WorkerModelSourceContext,
) -> Result<(), WorkerModelSourceError> {
    if context.session_id.is_nil()
        || context.attempt_generation == 0
        || context.node_id.is_empty()
        || context.node_id.len() > MAX_WORKER_MODEL_SOURCE_NODE_ID_BYTES
        || context.node_id.chars().any(char::is_control)
        || context.node_generation == 0
        || context.service_id.is_nil()
        || context.service_generation == 0
    {
        return Err(WorkerModelSourceError::InvalidOrder);
    }
    Ok(())
}

fn validate_model_source_context_match(
    expected: &WorkerModelSourceContext,
    actual: &WorkerModelSourceContext,
) -> Result<(), WorkerModelSourceError> {
    if actual.session_id != expected.session_id {
        return Err(WorkerModelSourceError::ForeignSession);
    }
    if actual.attempt_id != expected.attempt_id
        || actual.attempt_generation != expected.attempt_generation
    {
        return Err(WorkerModelSourceError::StaleAttempt);
    }
    if actual.node_id != expected.node_id || actual.node_generation != expected.node_generation {
        return Err(WorkerModelSourceError::StaleNode);
    }
    if actual.service_id != expected.service_id
        || actual.service_generation != expected.service_generation
    {
        return Err(WorkerModelSourceError::StaleService);
    }
    if actual.ordered_source_identity_sha256 != expected.ordered_source_identity_sha256 {
        return Err(WorkerModelSourceError::InvalidSourceSelection);
    }
    Ok(())
}

fn validate_model_source_request_wire(
    request: &WorkerModelSourceRequest,
) -> Result<(), WorkerModelSourceError> {
    validate_model_source_context(&request.context)?;
    if request.call_ordinal == 0 {
        return Err(WorkerModelSourceError::InvalidOrder);
    }
    match &request.operation {
        WorkerModelSourceOperation::Open {
            folder_category,
            source_names,
        } => {
            if folder_category.is_empty()
                || folder_category.len() > MAX_WORKER_MODEL_SOURCE_CATEGORY_BYTES
                || !folder_category
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
                || source_names.is_empty()
                || source_names.len() > MAX_WORKER_MODEL_SOURCE_SELECTIONS
                || source_names.iter().collect::<BTreeSet<_>>().len() != source_names.len()
                || source_names.iter().any(|name| {
                    name.is_empty()
                        || name.len() > MAX_WORKER_MODEL_SOURCE_NAME_BYTES
                        || name.chars().any(char::is_control)
                        || name.starts_with('/')
                        || name.contains(['\\', ':'])
                        || name.split('/').any(|component| {
                            component.is_empty() || matches!(component, "." | "..")
                        })
                })
            {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
            if worker_model_source_selection_sha256(folder_category, source_names)?
                != request.context.ordered_source_identity_sha256
            {
                return Err(WorkerModelSourceError::InvalidSourceSelection);
            }
        }
        WorkerModelSourceOperation::Read { byte_length, .. }
        | WorkerModelSourceOperation::ReadArtifact { byte_length, .. } => {
            if *byte_length == 0
                || usize::try_from(*byte_length)
                    .map_or(true, |length| length > MAX_WORKER_MODEL_SOURCE_CHUNK_BYTES)
            {
                return Err(WorkerModelSourceError::RangeLimit);
            }
        }
        WorkerModelSourceOperation::Close => {}
    }
    Ok(())
}

pub fn worker_model_source_selection_sha256(
    folder_category: &str,
    source_names: &[String],
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    if folder_category.is_empty()
        || folder_category.len() > MAX_WORKER_MODEL_SOURCE_CATEGORY_BYTES
        || source_names.is_empty()
        || source_names.len() > MAX_WORKER_MODEL_SOURCE_SELECTIONS
    {
        return Err(WorkerModelSourceError::InvalidSourceSelection);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-selection-v1");
    hasher.update(
        u64::try_from(folder_category.len())
            .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?
            .to_le_bytes(),
    );
    hasher.update(folder_category.as_bytes());
    hasher.update(
        u64::try_from(source_names.len())
            .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?
            .to_le_bytes(),
    );
    for source_name in source_names {
        hasher.update(
            u64::try_from(source_name.len())
                .map_err(|_| WorkerModelSourceError::InvalidSourceSelection)?
                .to_le_bytes(),
        );
        hasher.update(source_name.as_bytes());
    }
    model_source_digest(hasher)
}

fn model_source_response_identity(response: &WorkerModelSourceResponse) -> (Uuid, u64) {
    match response {
        WorkerModelSourceResponse::Opened(opened) => (opened.session_id, opened.call_ordinal),
        WorkerModelSourceResponse::Chunk(chunk) => (chunk.session_id, chunk.call_ordinal),
        WorkerModelSourceResponse::Closed(closed) => (closed.session_id, closed.call_ordinal),
        WorkerModelSourceResponse::Rejected {
            session_id,
            call_ordinal,
            ..
        } => (*session_id, *call_ordinal),
        WorkerModelSourceResponse::ArtifactChunk(chunk) => (chunk.session_id, chunk.call_ordinal),
    }
}

fn model_source_digest(hasher: Sha256) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    WorkerSha256Digest::new(format!("{:x}", hasher.finalize()))
        .map_err(|_| WorkerModelSourceError::HostFailure)
}

fn model_source_opened_digest(
    session_id: Uuid,
    call_ordinal: u64,
    ordered_source_identity_sha256: &WorkerSha256Digest,
    sources: &[WorkerModelSourceManifest],
    maximum_chunk_bytes: u32,
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-opened-v1");
    hasher.update(session_id.as_bytes());
    hasher.update(call_ordinal.to_le_bytes());
    hasher.update(ordered_source_identity_sha256.as_str().as_bytes());
    hasher.update(maximum_chunk_bytes.to_le_bytes());
    hasher.update(
        u64::try_from(sources.len())
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .to_le_bytes(),
    );
    for source in sources {
        hasher.update(source.source_ordinal.to_le_bytes());
        hasher.update(source.model_identity_sha256.as_str().as_bytes());
        hasher.update(source.aggregate_tensor_bytes.to_le_bytes());
        hasher.update(source.maximum_read_bytes.to_le_bytes());
        hasher.update(source.parser_limits.version.to_le_bytes());
        hasher.update(source.parser_limits.manifest_bytes.to_le_bytes());
        hasher.update(source.parser_limits.maximum_depth.to_le_bytes());
        hasher.update(source.parser_limits.maximum_tensors.to_le_bytes());
        hasher.update(source.parser_limits.maximum_tensor_bytes.to_le_bytes());
        hasher.update(
            source
                .parser_limits
                .maximum_aggregate_tensor_bytes
                .to_le_bytes(),
        );
        hasher.update(source.parser_limits.maximum_name_bytes.to_le_bytes());
        hasher.update(source.parser_limits.maximum_archive_entries.to_le_bytes());
        hasher.update(source.parser_limits.maximum_metadata_values.to_le_bytes());
        hasher.update(
            u64::try_from(source.artifacts.len())
                .map_err(|_| WorkerModelSourceError::HostFailure)?
                .to_le_bytes(),
        );
        for artifact in &source.artifacts {
            hasher.update(artifact.sha256.as_str().as_bytes());
            hasher.update(artifact.byte_size.to_le_bytes());
            match artifact.format {
                WorkerModelSourceFormat::Safetensors => hasher.update([0]),
                WorkerModelSourceFormat::PytorchArchive => hasher.update([1]),
                WorkerModelSourceFormat::Gguf => hasher.update([2]),
                WorkerModelSourceFormat::JsonConfig => hasher.update([3]),
                WorkerModelSourceFormat::JsonTokenizer => hasher.update([4]),
                WorkerModelSourceFormat::YamlConfig => hasher.update([5]),
                WorkerModelSourceFormat::SentencePiece => hasher.update([6]),
                WorkerModelSourceFormat::Tiktoken => hasher.update([7]),
                WorkerModelSourceFormat::SafetensorsMetadata {
                    source_artifact_ordinal,
                } => {
                    hasher.update([8]);
                    hasher.update(source_artifact_ordinal.to_le_bytes());
                }
                WorkerModelSourceFormat::ClipConfiguration { target_ordinal } => {
                    hasher.update([9, target_ordinal]);
                }
                WorkerModelSourceFormat::ClipTokenizerJson { target_ordinal } => {
                    hasher.update([10, target_ordinal]);
                }
                WorkerModelSourceFormat::ClipSentencePiece { target_ordinal } => {
                    hasher.update([11, target_ordinal]);
                }
                WorkerModelSourceFormat::ClipTiktoken { target_ordinal } => {
                    hasher.update([12, target_ordinal]);
                }
            }
            hasher.update([artifact.nested_state_disposition as u8]);
        }
        hasher.update(
            u64::try_from(source.tensors.len())
                .map_err(|_| WorkerModelSourceError::HostFailure)?
                .to_le_bytes(),
        );
        for tensor in &source.tensors {
            hasher.update(
                u64::try_from(tensor.name.len())
                    .map_err(|_| WorkerModelSourceError::HostFailure)?
                    .to_le_bytes(),
            );
            hasher.update(tensor.name.as_bytes());
            hasher.update(
                u64::try_from(tensor.data_type.len())
                    .map_err(|_| WorkerModelSourceError::HostFailure)?
                    .to_le_bytes(),
            );
            hasher.update(tensor.data_type.as_bytes());
            hasher.update(
                u64::try_from(tensor.shape.len())
                    .map_err(|_| WorkerModelSourceError::HostFailure)?
                    .to_le_bytes(),
            );
            for dimension in &tensor.shape {
                hasher.update(dimension.to_le_bytes());
            }
            hasher.update(tensor.artifact_ordinal.to_le_bytes());
            hasher.update(tensor.byte_offset.to_le_bytes());
            hasher.update(tensor.byte_length.to_le_bytes());
        }
    }
    model_source_digest(hasher)
}

fn model_source_chunk_digest(
    session_id: Uuid,
    call_ordinal: u64,
    source_ordinal: u32,
    tensor_ordinal: u32,
    byte_offset: u64,
    bytes: &[u8],
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-chunk-v1");
    hasher.update(session_id.as_bytes());
    hasher.update(call_ordinal.to_le_bytes());
    hasher.update(source_ordinal.to_le_bytes());
    hasher.update(tensor_ordinal.to_le_bytes());
    hasher.update(byte_offset.to_le_bytes());
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    model_source_digest(hasher)
}

fn model_source_artifact_chunk_digest(
    session_id: Uuid,
    call_ordinal: u64,
    source_ordinal: u32,
    artifact_ordinal: u32,
    artifact_sha256: &WorkerSha256Digest,
    byte_offset: u64,
    bytes: &[u8],
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-artifact-chunk-v1");
    hasher.update(session_id.as_bytes());
    hasher.update(call_ordinal.to_le_bytes());
    hasher.update(source_ordinal.to_le_bytes());
    hasher.update(artifact_ordinal.to_le_bytes());
    hasher.update(artifact_sha256.as_str().as_bytes());
    hasher.update(byte_offset.to_le_bytes());
    hasher.update(
        u64::try_from(bytes.len())
            .map_err(|_| WorkerModelSourceError::HostFailure)?
            .to_le_bytes(),
    );
    hasher.update(bytes);
    model_source_digest(hasher)
}

fn model_source_closed_digest(
    session_id: Uuid,
    call_ordinal: u64,
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-closed-v1");
    hasher.update(session_id.as_bytes());
    hasher.update(call_ordinal.to_le_bytes());
    model_source_digest(hasher)
}

fn model_source_rejected_digest(
    session_id: Uuid,
    call_ordinal: u64,
    error: &WorkerModelSourceError,
) -> Result<WorkerSha256Digest, WorkerModelSourceError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zed-sim-model-source-rejected-v1");
    hasher.update(session_id.as_bytes());
    hasher.update(call_ordinal.to_le_bytes());
    hasher.update(format!("{error:?}").as_bytes());
    model_source_digest(hasher)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum WorkerMessage {
    Hello {
        backend: WorkerBackendCapabilities,
    },
    HelloAck {
        accepted_backend: WorkerBackendCapabilities,
    },
    Ready,
    Execute {
        plan: Vec<u8>,
    },
    Cancel {
        reason: String,
    },
    Event {
        event: Vec<u8>,
    },
    OutputProposal {
        proposal: WorkerOutputProposal,
    },
    Heartbeat,
    Shutdown,
    Fatal {
        code: String,
        message: String,
    },
    Lifecycle {
        event: WorkerLifecycleEvent,
    },
    RegistryDeploymentBegin {
        deployment: WorkerRegistryDeploymentBegin,
    },
    RegistryDeploymentChunk {
        chunk: WorkerRegistryDeploymentChunk,
    },
    RegistryDeploymentCommit {
        commit: WorkerRegistryDeploymentCommit,
    },
    RegistryDeploymentAck {
        acknowledgement: WorkerRegistryDeploymentAck,
    },
    RegistryDeploymentRejected {
        rejection: WorkerRegistryDeploymentRejection,
    },
    ExecutePlugin {
        invocation: Vec<u8>,
    },
    PluginCapabilityRequest {
        call_id: u64,
        request: Vec<u8>,
    },
    PluginCapabilityResponse {
        call_id: u64,
        response: Vec<u8>,
    },
    PluginResult {
        outcome: WorkerPluginExecutionOutcome,
    },
    ProviderStreamRequest {
        call_id: u64,
        request: WorkerProviderStreamRequest,
    },
    ProviderStreamResponse {
        call_id: u64,
        response: WorkerProviderStreamResponse,
    },
    ProviderV2ProposalFinalization {
        finalization: WorkerProviderV2ProposalFinalization,
    },
    ProviderV2ProposalFinalizationAck {
        acknowledgement: WorkerProviderV2ProposalFinalizationAck,
    },
    ModelSourceRequest {
        call_id: u64,
        request: WorkerModelSourceRequest,
    },
    ModelSourceResponse {
        call_id: u64,
        response: WorkerModelSourceResponse,
    },
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum WorkerProtocolError {
    #[error("worker frame exceeds {MAX_WORKER_FRAME_BYTES} bytes")]
    Oversized,
    #[error("encoded worker event exceeds {MAX_ENCODED_PREVIEW_BYTES} bytes")]
    OversizedEvent,
    #[error("worker frame length prefix does not match its payload")]
    LengthMismatch,
    #[error("unsupported worker protocol version {0}")]
    UnsupportedVersion(u16),
    #[error("invalid worker payload: {0}")]
    InvalidPayload(String),
    #[error("worker envelope contains unsupported opaque extensions")]
    OpaqueExtensions,
    #[error("worker component deployment is invalid: {0}")]
    InvalidComponentDeployment(String),
    #[error("worker plugin invocation exceeds {MAX_WORKER_PLUGIN_INVOCATION_BYTES} bytes")]
    OversizedPluginInvocation,
    #[error(
        "worker plugin capability payload exceeds {MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES} bytes"
    )]
    OversizedPluginCapabilityPayload,
    #[error("worker plugin result exceeds {MAX_WORKER_PLUGIN_RESULT_BYTES} bytes")]
    OversizedPluginResult,
    #[error("worker plugin capability call identifier must be nonzero")]
    InvalidPluginCapabilityCallId,
    #[error("invalid provider worker stream: {0}")]
    InvalidProviderStream(WorkerProviderStreamError),
    #[error("invalid model-source worker stream: {0}")]
    InvalidModelSource(WorkerModelSourceError),
}

pub fn encode_worker_frame(message: &WorkerEnvelope) -> Result<Vec<u8>, WorkerProtocolError> {
    if message.version != WORKER_PROTOCOL_VERSION {
        return Err(WorkerProtocolError::UnsupportedVersion(message.version));
    }
    validate_envelope_bounds(message)?;
    let payload = postcard::to_stdvec(message)
        .map_err(|error| WorkerProtocolError::InvalidPayload(error.to_string()))?;
    if payload.len() > MAX_WORKER_FRAME_BYTES {
        return Err(WorkerProtocolError::Oversized);
    }
    let length = u32::try_from(payload.len()).map_err(|_| WorkerProtocolError::Oversized)?;
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&length.to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

pub fn decode_worker_frame(frame: &[u8]) -> Result<WorkerEnvelope, WorkerProtocolError> {
    let length_bytes: [u8; 4] = frame
        .get(..4)
        .ok_or(WorkerProtocolError::LengthMismatch)?
        .try_into()
        .map_err(|_| WorkerProtocolError::LengthMismatch)?;
    let declared = u32::from_le_bytes(length_bytes) as usize;
    let payload = frame.get(4..).ok_or(WorkerProtocolError::LengthMismatch)?;
    if declared != payload.len() {
        return Err(WorkerProtocolError::LengthMismatch);
    }
    if declared > MAX_WORKER_FRAME_BYTES {
        return Err(WorkerProtocolError::Oversized);
    }
    let (version, _) = postcard::take_from_bytes::<u16>(payload)
        .map_err(|error| WorkerProtocolError::InvalidPayload(error.to_string()))?;
    if version != WORKER_PROTOCOL_VERSION {
        return Err(WorkerProtocolError::UnsupportedVersion(version));
    }
    let message: WorkerEnvelope = postcard::from_bytes(payload)
        .map_err(|error| WorkerProtocolError::InvalidPayload(error.to_string()))?;
    validate_envelope_bounds(&message)?;
    Ok(message)
}

fn validate_envelope_bounds(message: &WorkerEnvelope) -> Result<(), WorkerProtocolError> {
    if !message.extensions.is_empty() {
        return Err(WorkerProtocolError::OpaqueExtensions);
    }
    validate_message_bounds(&message.message)
}

fn validate_message_bounds(message: &WorkerMessage) -> Result<(), WorkerProtocolError> {
    match message {
        WorkerMessage::Fatal { code, message }
            if code.is_empty()
                || code.len() > MAX_WORKER_FATAL_CODE_BYTES
                || !code.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'_' | b'-' | b'.')
                })
                || message.is_empty()
                || message.len() > MAX_WORKER_FATAL_MESSAGE_BYTES
                || message
                    .chars()
                    .any(|character| character.is_control() && character != '\t') =>
        {
            Err(WorkerProtocolError::InvalidPayload(
                "worker fatal diagnostic is invalid".to_owned(),
            ))
        }
        WorkerMessage::Event { event } if event.len() > MAX_ENCODED_PREVIEW_BYTES => {
            Err(WorkerProtocolError::OversizedEvent)
        }
        WorkerMessage::RegistryDeploymentBegin { deployment } => deployment
            .validate()
            .map_err(|error| WorkerProtocolError::InvalidComponentDeployment(error.to_string())),
        WorkerMessage::RegistryDeploymentChunk { chunk } => chunk
            .validate()
            .map_err(|error| WorkerProtocolError::InvalidComponentDeployment(error.to_string())),
        WorkerMessage::RegistryDeploymentAck { acknowledgement }
            if usize::try_from(acknowledgement.component_count)
                .map_or(true, |count| count > MAX_WORKER_COMPONENT_COUNT) =>
        {
            Err(WorkerProtocolError::InvalidComponentDeployment(
                WorkerComponentDeploymentError::TooManyComponents.to_string(),
            ))
        }
        WorkerMessage::RegistryDeploymentRejected { rejection } => rejection
            .validate()
            .map_err(|error| WorkerProtocolError::InvalidComponentDeployment(error.to_string())),
        WorkerMessage::ExecutePlugin { invocation }
            if invocation.is_empty() || invocation.len() > MAX_WORKER_PLUGIN_INVOCATION_BYTES =>
        {
            Err(WorkerProtocolError::OversizedPluginInvocation)
        }
        WorkerMessage::PluginCapabilityRequest { call_id, request }
        | WorkerMessage::PluginCapabilityResponse {
            call_id,
            response: request,
        } if *call_id == 0 => Err(WorkerProtocolError::InvalidPluginCapabilityCallId),
        WorkerMessage::PluginCapabilityRequest { request, .. }
        | WorkerMessage::PluginCapabilityResponse {
            response: request, ..
        } if request.is_empty() || request.len() > MAX_WORKER_PLUGIN_CAPABILITY_PAYLOAD_BYTES => {
            Err(WorkerProtocolError::OversizedPluginCapabilityPayload)
        }
        WorkerMessage::PluginResult {
            outcome: WorkerPluginExecutionOutcome::Succeeded(result),
        } if result.is_empty() || result.len() > MAX_WORKER_PLUGIN_RESULT_BYTES => {
            Err(WorkerProtocolError::OversizedPluginResult)
        }
        WorkerMessage::PluginResult {
            outcome:
                WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::Trap { diagnostic }),
        } if diagnostic.is_empty()
            || diagnostic.chars().count() > MAX_WORKER_PLUGIN_DIAGNOSTIC_CHARS
            || diagnostic
                .chars()
                .any(|character| character.is_control() && !matches!(character, '\n' | '\t')) =>
        {
            Err(WorkerProtocolError::InvalidPayload(
                "worker plugin trap diagnostic is invalid".to_owned(),
            ))
        }
        WorkerMessage::ProviderStreamRequest { call_id, .. }
        | WorkerMessage::ProviderStreamResponse { call_id, .. }
            if *call_id == 0 =>
        {
            Err(WorkerProtocolError::InvalidProviderStream(
                WorkerProviderStreamError::InvalidOrder,
            ))
        }
        WorkerMessage::ProviderStreamRequest { request, .. } => {
            validate_provider_stream_request_wire(request)
                .map_err(WorkerProtocolError::InvalidProviderStream)
        }
        WorkerMessage::ProviderStreamResponse { response, .. } => {
            validate_provider_stream_response_wire(response)
                .map_err(WorkerProtocolError::InvalidProviderStream)
        }
        WorkerMessage::ProviderV2ProposalFinalization { finalization } => finalization
            .validate()
            .map_err(WorkerProtocolError::InvalidProviderStream),
        WorkerMessage::ProviderV2ProposalFinalizationAck { acknowledgement } => acknowledgement
            .validate()
            .map_err(WorkerProtocolError::InvalidProviderStream),
        WorkerMessage::ModelSourceRequest { call_id, request } => {
            if *call_id == 0 {
                return Err(WorkerProtocolError::InvalidModelSource(
                    WorkerModelSourceError::InvalidOrder,
                ));
            }
            validate_model_source_request_wire(request)
                .map_err(WorkerProtocolError::InvalidModelSource)
        }
        WorkerMessage::ModelSourceResponse { call_id, response } => {
            if *call_id == 0 {
                return Err(WorkerProtocolError::InvalidModelSource(
                    WorkerModelSourceError::InvalidOrder,
                ));
            }
            response
                .validate()
                .map_err(WorkerProtocolError::InvalidModelSource)
        }
        _ => Ok(()),
    }
}

fn validate_provider_stream_request_wire(
    request: &WorkerProviderStreamRequest,
) -> Result<(), WorkerProviderStreamError> {
    match request {
        WorkerProviderStreamRequest::StartRequest { context, head } => {
            validate_provider_context(context)?;
            validate_provider_request_head_global(head)
        }
        WorkerProviderStreamRequest::WriteRequestChunk(chunk) => {
            validate_provider_handle_wire(chunk.handle)?;
            if chunk.bytes.is_empty() && !chunk.end {
                return Err(WorkerProviderStreamError::ChunkLimit);
            }
            if chunk.bytes.len() > MAX_WORKER_PROVIDER_CHUNK_BYTES {
                return Err(WorkerProviderStreamError::ChunkLimit);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::WriteUploadChunk(chunk) => {
            validate_provider_handle_wire(chunk.handle)?;
            if chunk.bytes.is_empty() && !chunk.end {
                return Err(WorkerProviderStreamError::InvalidOrder);
            }
            if chunk.bytes.len() > MAX_WORKER_PROVIDER_CHUNK_BYTES {
                return Err(WorkerProviderStreamError::ChunkLimit);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::CompleteUpload(request) => {
            validate_provider_handle_wire(request.handle)
        }
        WorkerProviderStreamRequest::RetryRequest(request) => {
            validate_provider_handle_wire(request.handle)?;
            if request.delay_milliseconds > MAX_WORKER_PROVIDER_RETRY_DELAY_MILLISECONDS {
                return Err(WorkerProviderStreamError::WaitLimit);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::WaitResponse(request) => {
            validate_provider_handle_wire(request.handle)?;
            if request.timeout_milliseconds == 0
                || request.timeout_milliseconds > MAX_WORKER_PROVIDER_WAIT_MILLISECONDS
            {
                return Err(WorkerProviderStreamError::WaitLimit);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::StartUpload(request) => {
            validate_provider_handle_wire(request.handle)?;
            if !valid_dotted_identifier(&request.port_id)
                || !valid_media_type(&request.media_type)
                || request.byte_length == 0
                || request.byte_length > MAX_WORKER_PROVIDER_BODY_BYTES
                || request.content_sha256.len() != 64
                || !request
                    .content_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(WorkerProviderStreamError::InvalidUpload);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::RequestCost(request) => {
            validate_provider_handle_wire(request.handle)?;
            if !valid_dotted_identifier(&request.operation)
                || request.currency.len() != 3
                || !request
                    .currency
                    .bytes()
                    .all(|byte| byte.is_ascii_uppercase())
                || request.maximum_microunits == 0
            {
                return Err(WorkerProviderStreamError::InvalidCostRequest);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::ReportProgress(progress) => {
            validate_provider_handle_wire(progress.handle)?;
            if progress.total == 0
                || progress.total > MAX_WORKER_PROVIDER_PROGRESS_TOTAL
                || progress.completed > progress.total
                || progress.message.as_ref().is_some_and(|message| {
                    message.is_empty() || message.len() > MAX_WORKER_PROVIDER_PROGRESS_MESSAGE_BYTES
                })
            {
                return Err(WorkerProviderStreamError::InvalidProgress);
            }
            Ok(())
        }
        WorkerProviderStreamRequest::CheckCancelled(handle) => {
            validate_provider_handle_wire(*handle)
        }
    }
}

fn is_retryable_provider_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}

fn retry_after_header(
    headers: &[WorkerProviderHeader],
) -> Result<Option<String>, WorkerProviderStreamError> {
    let mut retry_after = None;
    for header in headers {
        if header.name.eq_ignore_ascii_case("retry-after") {
            if retry_after.is_some() {
                return Err(WorkerProviderStreamError::InvalidHeaders);
            }
            retry_after = Some(header.value.clone());
        }
    }
    Ok(retry_after)
}

fn validate_provider_stream_response_wire(
    response: &WorkerProviderStreamResponse,
) -> Result<(), WorkerProviderStreamError> {
    match response {
        WorkerProviderStreamResponse::Stream(Ok(handle)) => validate_provider_handle_wire(*handle),
        WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(frame))) => {
            validate_provider_handle_wire(frame.handle)?;
            match &frame.event {
                WorkerProviderResponseFrameEvent::Head(head) => {
                    if !(200..=599).contains(&head.status) {
                        return Err(WorkerProviderStreamError::InvalidOrder);
                    }
                    validate_provider_headers_global(&head.headers)
                }
                WorkerProviderResponseFrameEvent::Chunk(chunk) => {
                    validate_provider_response_chunk_global(chunk)
                }
                WorkerProviderResponseFrameEvent::Terminal(terminal) => {
                    validate_provider_terminal(terminal, true, true, true)
                }
            }
        }
        WorkerProviderStreamResponse::Cost(Ok(cost)) => {
            let accepted = cost.accepted
                && cost.approved_microunits != 0
                && !cost.receipt.is_empty()
                && cost.receipt.len() <= MAX_WORKER_PROVIDER_RECEIPT_BYTES;
            let denied = !cost.accepted && cost.approved_microunits == 0 && cost.receipt.is_empty();
            if accepted || denied {
                Ok(())
            } else {
                Err(WorkerProviderStreamError::InvalidCostRequest)
            }
        }
        WorkerProviderStreamResponse::UploadCompletion(Ok(completion)) => {
            validate_provider_upload_url(&completion.download_url)
        }
        _ => Ok(()),
    }
}

pub fn validate_provider_upload_url(value: &str) -> Result<(), WorkerProviderStreamError> {
    if value.is_empty()
        || value.len() > MAX_WORKER_PROVIDER_UPLOAD_URL_BYTES
        || !value.starts_with("https://")
        || value.chars().any(char::is_control)
        || value.bytes().any(|byte| byte.is_ascii_whitespace())
        || value.contains('?')
        || value.contains('#')
        || value.contains('\\')
    {
        return Err(WorkerProviderStreamError::InvalidUpload);
    }
    let authority_and_path = &value["https://".len()..];
    let Some((authority, path)) = authority_and_path.split_once('/') else {
        return Err(WorkerProviderStreamError::InvalidUpload);
    };
    if path.is_empty() || authority.is_empty() || authority.contains('@') {
        return Err(WorkerProviderStreamError::InvalidUpload);
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (host, Some(port)),
        None => (authority, None),
    };
    if host.contains(':') || host.len() > 253 || !host.is_ascii() || !host.contains('.') {
        return Err(WorkerProviderStreamError::InvalidUpload);
    }
    let host = host.to_ascii_lowercase();
    if host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host.ends_with(".lan")
        || host.ends_with(".home")
        || host.ends_with(".home.arpa")
    {
        return Err(WorkerProviderStreamError::InvalidUpload);
    }
    let mut labels = host.split('.').peekable();
    while let Some(label) = labels.next() {
        if label.is_empty()
            || label.len() > 63
            || label.starts_with('-')
            || label.ends_with('-')
            || !label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
            || (labels.peek().is_none() && label.bytes().all(|byte| byte.is_ascii_digit()))
        {
            return Err(WorkerProviderStreamError::InvalidUpload);
        }
    }
    if let Some(port) = port {
        if port.is_empty()
            || !port.bytes().all(|byte| byte.is_ascii_digit())
            || port.parse::<u16>().ok().filter(|port| *port != 0).is_none()
        {
            return Err(WorkerProviderStreamError::InvalidUpload);
        }
    }
    let bytes = path.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        let Some(high) = bytes
            .get(index + 1)
            .and_then(|byte| checked_hex_nibble(*byte))
        else {
            return Err(WorkerProviderStreamError::InvalidUpload);
        };
        let Some(low) = bytes
            .get(index + 2)
            .and_then(|byte| checked_hex_nibble(*byte))
        else {
            return Err(WorkerProviderStreamError::InvalidUpload);
        };
        let decoded = (high << 4) | low;
        if decoded.is_ascii_control() {
            return Err(WorkerProviderStreamError::InvalidUpload);
        }
        index += 3;
    }
    Ok(())
}

fn checked_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn validate_provider_handle_wire(
    handle: WorkerProviderStreamHandle,
) -> Result<(), WorkerProviderStreamError> {
    if handle.session_id.is_nil()
        || handle.session_generation == 0
        || handle.invocation == 0
        || handle.slot == 0
        || handle.generation == 0
    {
        Err(WorkerProviderStreamError::InvalidHandle)
    } else {
        Ok(())
    }
}

fn validate_provider_headers_global(
    headers: &[WorkerProviderHeader],
) -> Result<(), WorkerProviderStreamError> {
    validate_provider_headers(headers, &maximum_worker_provider_contract()?)
}

fn validate_provider_request_head_global(
    head: &WorkerProviderRequestHead,
) -> Result<(), WorkerProviderStreamError> {
    validate_provider_request_head(head, &maximum_worker_provider_contract()?)
}

fn maximum_worker_provider_contract()
-> Result<WorkerProviderStreamingContract, WorkerProviderStreamError> {
    Ok(WorkerProviderStreamingContract {
        methods: vec![
            WorkerProviderHttpMethod::Delete,
            WorkerProviderHttpMethod::Get,
            WorkerProviderHttpMethod::Head,
            WorkerProviderHttpMethod::Options,
            WorkerProviderHttpMethod::Patch,
            WorkerProviderHttpMethod::Post,
            WorkerProviderHttpMethod::Put,
        ],
        maximum_headers: u16::try_from(MAX_WORKER_PROVIDER_HEADERS)
            .map_err(|_| WorkerProviderStreamError::InvalidContract)?,
        maximum_header_bytes: u32::try_from(MAX_WORKER_PROVIDER_HEADER_BYTES)
            .map_err(|_| WorkerProviderStreamError::InvalidContract)?,
        maximum_request_body_bytes: MAX_WORKER_PROVIDER_BODY_BYTES,
        maximum_response_body_bytes: MAX_WORKER_PROVIDER_BODY_BYTES,
        maximum_chunk_bytes: u32::try_from(MAX_WORKER_PROVIDER_CHUNK_BYTES)
            .map_err(|_| WorkerProviderStreamError::InvalidContract)?,
        maximum_ndjson_line_bytes: u32::try_from(MAX_WORKER_PROVIDER_NDJSON_LINE_BYTES)
            .map_err(|_| WorkerProviderStreamError::InvalidContract)?,
        maximum_wait_milliseconds: MAX_WORKER_PROVIDER_WAIT_MILLISECONDS,
        maximum_uploads: 0,
        maximum_upload_body_bytes: 0,
        maximum_cost_requests: 0,
        maximum_progress_total: MAX_WORKER_PROVIDER_PROGRESS_TOTAL,
        uploads: false,
        cost_requests: false,
    })
}

fn validate_provider_response_chunk_global(
    chunk: &WorkerProviderResponseChunk,
) -> Result<(), WorkerProviderStreamError> {
    let bytes = match chunk {
        WorkerProviderResponseChunk::Binary(bytes) => bytes.len(),
        WorkerProviderResponseChunk::Text(text) => text.len(),
        WorkerProviderResponseChunk::NdjsonLine(line) => {
            if line.contains(['\n', '\r'])
                || line.len() > MAX_WORKER_PROVIDER_NDJSON_LINE_BYTES
                || serde_json::from_str::<serde_json::Value>(line).is_err()
            {
                return Err(WorkerProviderStreamError::InvalidNdjsonLine);
            }
            line.len()
        }
    };
    if bytes == 0 || bytes > MAX_WORKER_PROVIDER_CHUNK_BYTES {
        Err(WorkerProviderStreamError::ChunkLimit)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(message: WorkerMessage) -> WorkerEnvelope {
        WorkerEnvelope {
            version: WORKER_PROTOCOL_VERSION,
            profile_id: ProfileId(Uuid::nil()),
            worker_id: WorkerId(Uuid::nil()),
            request_id: RequestId(Uuid::nil()),
            prompt_id: None,
            attempt_id: None,
            sequence: 7,
            registry_version: "fixture-v1".into(),
            message,
            extensions: BTreeMap::new(),
        }
    }

    fn cpu_backend() -> WorkerBackendCapabilities {
        let allocation = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F32,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        WorkerBackendCapabilities::new(DeviceKind::Cpu, 0, vec![allocation], vec![])
            .expect("valid CPU fixture")
    }

    fn digest(byte: char) -> WorkerSha256Digest {
        WorkerSha256Digest::new(std::iter::repeat_n(byte, 64).collect::<String>())
            .expect("valid digest")
    }

    fn provider_context() -> WorkerProviderInvocationContext {
        WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(1),
            session_generation: 1,
            invocation: 7,
            generation: 3,
        }
    }

    fn provider_handle(slot: u32) -> WorkerProviderStreamHandle {
        let context = provider_context();
        WorkerProviderStreamHandle {
            session_id: context.session_id,
            session_generation: context.session_generation,
            invocation: context.invocation,
            slot,
            generation: context.generation,
        }
    }

    fn provider_finalization() -> WorkerProviderV2ProposalFinalization {
        WorkerProviderV2ProposalFinalization {
            context: provider_context(),
            handle: provider_handle(1),
            proposal_generation: 9,
            finalization_nonce: [0x5a; 32],
            receipt_identity_sha256: digest('a'),
            materialization_identity_sha256: digest('b'),
        }
    }

    fn provider_contract() -> WorkerProviderStreamingContract {
        WorkerProviderStreamingContract {
            methods: vec![
                WorkerProviderHttpMethod::Get,
                WorkerProviderHttpMethod::Post,
            ],
            maximum_headers: 4,
            maximum_header_bytes: 1_024,
            maximum_request_body_bytes: 1_024,
            maximum_response_body_bytes: 1_024,
            maximum_chunk_bytes: 128,
            maximum_ndjson_line_bytes: 128,
            maximum_wait_milliseconds: 1_000,
            maximum_uploads: 1,
            maximum_upload_body_bytes: 1_024,
            maximum_cost_requests: 1,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        }
    }

    fn provider_head(declared_body_bytes: Option<u64>) -> WorkerProviderRequestHead {
        WorkerProviderRequestHead {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: Some("provider.api-key".to_owned()),
            method: WorkerProviderHttpMethod::Post,
            headers: vec![WorkerProviderHeader {
                name: "content-type".to_owned(),
                value: "application/json".to_owned(),
            }],
            declared_body_bytes,
        }
    }

    fn started_provider_validator(
        cancellation: CancellationToken,
    ) -> WorkerProviderStreamTransportValidator {
        let mut validator = WorkerProviderStreamTransportValidator::checked_for_host_session(
            provider_context(),
            provider_contract(),
            cancellation,
        )
        .expect("host-issued provider session is valid");
        validator
            .validate_request(
                1,
                &WorkerProviderStreamRequest::StartRequest {
                    context: provider_context(),
                    head: provider_head(Some(3)),
                },
            )
            .expect("start request is valid");
        validator
            .validate_response(
                1,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1))),
            )
            .expect("host stream handle is valid");
        validator
    }

    fn component_descriptor(
        extension_id: &str,
        component_digest: char,
        manifest_bytes: u64,
        component_bytes: u64,
    ) -> Result<WorkerComponentDescriptor, WorkerComponentDeploymentError> {
        WorkerComponentDescriptor::new(
            extension_id,
            "1.2.3",
            "plugin.identifier",
            "4.5.6",
            digest('d'),
            digest('a'),
            digest(component_digest),
            manifest_bytes,
            1,
            component_bytes,
        )
    }

    fn deployment_begin() -> WorkerRegistryDeploymentBegin {
        WorkerRegistryDeploymentBegin::new(
            WorkerRegistryGeneration::new(1).expect("nonzero generation"),
            digest('c'),
            vec![component_descriptor("extension-a", 'b', 2, 3).expect("bounded descriptor")],
        )
        .expect("bounded deployment")
    }

    #[test]
    fn every_worker_message_variant_round_trips() {
        let messages = [
            WorkerMessage::Hello {
                backend: cpu_backend(),
            },
            WorkerMessage::HelloAck {
                accepted_backend: cpu_backend(),
            },
            WorkerMessage::Ready,
            WorkerMessage::Execute { plan: vec![1] },
            WorkerMessage::Cancel {
                reason: "test".into(),
            },
            WorkerMessage::Event { event: vec![2] },
            WorkerMessage::OutputProposal {
                proposal: WorkerOutputProposal::new(Uuid::nil(), vec![3], vec![4])
                    .expect("bounded proposal"),
            },
            WorkerMessage::Heartbeat,
            WorkerMessage::Shutdown,
            WorkerMessage::Fatal {
                code: "fault".into(),
                message: "fixture".into(),
            },
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted,
            },
            WorkerMessage::RegistryDeploymentBegin {
                deployment: deployment_begin(),
            },
            WorkerMessage::RegistryDeploymentChunk {
                chunk: WorkerRegistryDeploymentChunk::new(
                    WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                    0,
                    WorkerComponentContent::Manifest,
                    0,
                    vec![1, 2],
                )
                .expect("bounded chunk"),
            },
            WorkerMessage::RegistryDeploymentCommit {
                commit: WorkerRegistryDeploymentCommit::new(
                    WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                    digest('c'),
                ),
            },
            WorkerMessage::RegistryDeploymentAck {
                acknowledgement: WorkerRegistryDeploymentAck::new(
                    WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                    digest('c'),
                    1,
                )
                .expect("bounded acknowledgement"),
            },
            WorkerMessage::RegistryDeploymentRejected {
                rejection: WorkerRegistryDeploymentRejection::new(
                    WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                    digest('c'),
                    WorkerRegistryDeploymentRejectionReason::InvalidAuthorization,
                ),
            },
        ];
        for message in messages {
            let original = envelope(message);
            let frame = encode_worker_frame(&original).expect("encodable frame");
            assert_eq!(decode_worker_frame(&frame), Ok(original));
        }
    }

    #[test]
    fn protocol_v8_preserves_every_pre_v8_worker_message_discriminant() {
        assert_eq!(WORKER_PROTOCOL_VERSION, 8);
        assert_eq!(PREVIOUS_WORKER_PROTOCOL_VERSION, 7);
        assert_eq!(LEGACY_WORKER_PROTOCOL_VERSION, 6);

        let messages = vec![
            WorkerMessage::Hello {
                backend: cpu_backend(),
            },
            WorkerMessage::HelloAck {
                accepted_backend: cpu_backend(),
            },
            WorkerMessage::Ready,
            WorkerMessage::Execute { plan: vec![1] },
            WorkerMessage::Cancel {
                reason: "cancel".to_owned(),
            },
            WorkerMessage::Event { event: vec![1] },
            WorkerMessage::OutputProposal {
                proposal: WorkerOutputProposal::new(Uuid::from_u128(2), vec![1], vec![2])
                    .expect("bounded proposal"),
            },
            WorkerMessage::Heartbeat,
            WorkerMessage::Shutdown,
            WorkerMessage::Fatal {
                code: "fatal".to_owned(),
                message: "message".to_owned(),
            },
            WorkerMessage::Lifecycle {
                event: WorkerLifecycleEvent::ExecutionStarted,
            },
            WorkerMessage::RegistryDeploymentBegin {
                deployment: deployment_begin(),
            },
            WorkerMessage::RegistryDeploymentChunk {
                chunk: WorkerRegistryDeploymentChunk::new(
                    WorkerRegistryGeneration::new(1).expect("generation"),
                    0,
                    WorkerComponentContent::Manifest,
                    0,
                    vec![1],
                )
                .expect("chunk"),
            },
            WorkerMessage::RegistryDeploymentCommit {
                commit: WorkerRegistryDeploymentCommit::new(
                    WorkerRegistryGeneration::new(1).expect("generation"),
                    digest('c'),
                ),
            },
            WorkerMessage::RegistryDeploymentAck {
                acknowledgement: WorkerRegistryDeploymentAck::new(
                    WorkerRegistryGeneration::new(1).expect("generation"),
                    digest('c'),
                    1,
                )
                .expect("acknowledgement"),
            },
            WorkerMessage::RegistryDeploymentRejected {
                rejection: WorkerRegistryDeploymentRejection::new(
                    WorkerRegistryGeneration::new(1).expect("generation"),
                    digest('c'),
                    WorkerRegistryDeploymentRejectionReason::InvalidCandidate,
                ),
            },
            WorkerMessage::ExecutePlugin {
                invocation: vec![1],
            },
            WorkerMessage::PluginCapabilityRequest {
                call_id: 1,
                request: vec![1],
            },
            WorkerMessage::PluginCapabilityResponse {
                call_id: 1,
                response: vec![1],
            },
            WorkerMessage::PluginResult {
                outcome: WorkerPluginExecutionOutcome::Succeeded(vec![1]),
            },
        ];
        for (discriminant, message) in messages.into_iter().enumerate() {
            let bytes = postcard::to_stdvec(&message).expect("legacy variant serializes");
            assert_eq!(bytes.first().copied(), u8::try_from(discriminant).ok());
        }

        let request = WorkerMessage::ProviderStreamRequest {
            call_id: 1,
            request: WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
        };
        let response = WorkerMessage::ProviderStreamResponse {
            call_id: 1,
            response: WorkerProviderStreamResponse::Unit(Ok(())),
        };
        assert_eq!(postcard::to_stdvec(&request).expect("request")[0], 20);
        assert_eq!(postcard::to_stdvec(&response).expect("response")[0], 21);

        let finalization = WorkerMessage::ProviderV2ProposalFinalization {
            finalization: provider_finalization(),
        };
        let acknowledgement = WorkerMessage::ProviderV2ProposalFinalizationAck {
            acknowledgement: WorkerProviderV2ProposalFinalizationAck {
                finalization: provider_finalization(),
                result: Ok(()),
            },
        };
        assert_eq!(
            postcard::to_stdvec(&finalization).expect("finalization")[0],
            22
        );
        assert_eq!(
            postcard::to_stdvec(&acknowledgement).expect("acknowledgement")[0],
            23
        );
    }

    #[test]
    fn provider_v2_finalization_is_bounded_and_round_trips_independently() {
        let finalization = provider_finalization();
        finalization.validate().expect("finalization is valid");
        for message in [
            WorkerMessage::ProviderV2ProposalFinalization {
                finalization: finalization.clone(),
            },
            WorkerMessage::ProviderV2ProposalFinalizationAck {
                acknowledgement: WorkerProviderV2ProposalFinalizationAck {
                    finalization: finalization.clone(),
                    result: Ok(()),
                },
            },
        ] {
            let original = envelope(message);
            let frame = encode_worker_frame(&original).expect("finalization frame is bounded");
            assert_eq!(decode_worker_frame(&frame), Ok(original));
        }

        let mut invalid = finalization.clone();
        invalid.finalization_nonce = [0; 32];
        assert_eq!(
            invalid.validate(),
            Err(WorkerProviderStreamError::InvalidInvocationResult)
        );
        let mut foreign = finalization;
        foreign.handle.invocation = foreign.handle.invocation.saturating_add(1);
        assert_eq!(
            foreign.validate(),
            Err(WorkerProviderStreamError::InvalidInvocationResult)
        );
    }

    #[test]
    fn every_provider_stream_message_shape_round_trips_through_the_worker_envelope() {
        let upload_sha256 = format!("{:x}", Sha256::digest(b"abc"));
        let requests = vec![
            WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head: provider_head(Some(3)),
            },
            WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle: provider_handle(1),
                sequence: 0,
                bytes: b"abc".to_vec(),
                end: true,
            }),
            WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                handle: provider_handle(1),
                after_sequence: None,
                timeout_milliseconds: 10,
            }),
            WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                handle: provider_handle(1),
                port_id: "image.output".to_owned(),
                media_type: "application/octet-stream".to_owned(),
                byte_length: 3,
                content_sha256: upload_sha256,
            }),
            WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                handle: provider_handle(2),
                sequence: 0,
                bytes: b"abc".to_vec(),
                end: true,
            }),
            WorkerProviderStreamRequest::CompleteUpload(WorkerProviderUploadCompletionRequest {
                handle: provider_handle(2),
            }),
            WorkerProviderStreamRequest::RetryRequest(WorkerProviderRetryRequest {
                handle: provider_handle(1),
                delay_milliseconds: 10,
            }),
            WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                handle: provider_handle(1),
                operation: "image.generate".to_owned(),
                currency: "USD".to_owned(),
                maximum_microunits: 1,
            }),
            WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                handle: provider_handle(1),
                sequence: 0,
                completed: 1,
                total: 2,
                message: Some("working".to_owned()),
            }),
            WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
        ];
        for (index, request) in requests.into_iter().enumerate() {
            let original = envelope(WorkerMessage::ProviderStreamRequest {
                call_id: u64::try_from(index).expect("bounded index") + 1,
                request,
            });
            let frame = encode_worker_frame(&original).expect("provider request is encodable");
            assert_eq!(decode_worker_frame(&frame), Ok(original));
        }

        let responses = vec![
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(1))),
            WorkerProviderStreamResponse::Unit(Ok(())),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                        status: 200,
                        headers: Vec::new(),
                    }),
                },
            ))),
            WorkerProviderStreamResponse::UploadCompletion(Ok(WorkerProviderUploadCompletion {
                download_url: "https://fixture.invalid/storage/file.bin".to_owned(),
            })),
            WorkerProviderStreamResponse::Cost(Ok(WorkerProviderCostResponse {
                accepted: true,
                approved_microunits: 1,
                receipt: vec![1],
            })),
        ];
        for (index, response) in responses.into_iter().enumerate() {
            let original = envelope(WorkerMessage::ProviderStreamResponse {
                call_id: u64::try_from(index).expect("bounded index") + 1,
                response,
            });
            let frame = encode_worker_frame(&original).expect("provider response is encodable");
            assert_eq!(decode_worker_frame(&frame), Ok(original));
        }
    }

    #[test]
    fn provider_stream_inner_enum_postcard_bytes_are_append_only() {
        let zero_handle = WorkerProviderStreamHandle {
            session_id: Uuid::nil(),
            session_generation: 0,
            invocation: 0,
            slot: 0,
            generation: 0,
        };
        let legacy_requests = [
            (
                0,
                WorkerProviderStreamRequest::StartRequest {
                    context: provider_context(),
                    head: provider_head(None),
                },
            ),
            (
                1,
                WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: zero_handle,
                    sequence: 0,
                    bytes: Vec::new(),
                    end: false,
                }),
            ),
            (
                2,
                WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: zero_handle,
                    after_sequence: None,
                    timeout_milliseconds: 0,
                }),
            ),
            (
                3,
                WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: zero_handle,
                    port_id: String::new(),
                    media_type: String::new(),
                    byte_length: 0,
                    content_sha256: String::new(),
                }),
            ),
            (
                4,
                WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                    handle: zero_handle,
                    sequence: 0,
                    bytes: Vec::new(),
                    end: false,
                }),
            ),
            (
                5,
                WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                    handle: zero_handle,
                    operation: String::new(),
                    currency: String::new(),
                    maximum_microunits: 0,
                }),
            ),
            (
                6,
                WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                    handle: zero_handle,
                    sequence: 0,
                    completed: 0,
                    total: 0,
                    message: None,
                }),
            ),
            (7, WorkerProviderStreamRequest::CheckCancelled(zero_handle)),
        ];
        for (expected, request) in legacy_requests {
            assert_eq!(
                postcard::to_stdvec(&request)
                    .expect("legacy provider request serializes")
                    .first()
                    .copied(),
                Some(expected)
            );
        }
        let legacy_responses = [
            (
                0,
                WorkerProviderStreamResponse::Stream(Err(WorkerProviderStreamError::Cancelled)),
            ),
            (
                1,
                WorkerProviderStreamResponse::Unit(Err(WorkerProviderStreamError::Cancelled)),
            ),
            (
                2,
                WorkerProviderStreamResponse::Wait(Err(WorkerProviderStreamError::Cancelled)),
            ),
            (
                3,
                WorkerProviderStreamResponse::Cost(Err(WorkerProviderStreamError::Cancelled)),
            ),
        ];
        for (expected, response) in legacy_responses {
            assert_eq!(
                postcard::to_stdvec(&response)
                    .expect("legacy provider response serializes")
                    .first()
                    .copied(),
                Some(expected)
            );
        }
        let mut legacy_request_bytes = vec![7, 16];
        legacy_request_bytes.extend_from_slice(&[0; 20]);
        assert_eq!(
            postcard::to_stdvec(&WorkerProviderStreamRequest::CheckCancelled(zero_handle))
                .expect("legacy provider request serializes"),
            legacy_request_bytes
        );
        assert_eq!(
            postcard::to_stdvec(&WorkerProviderStreamResponse::Unit(Ok(())))
                .expect("legacy provider response serializes"),
            vec![1, 0]
        );

        let mut completion_request_bytes = vec![8, 16];
        completion_request_bytes.extend_from_slice(&[0; 20]);
        assert_eq!(
            postcard::to_stdvec(&WorkerProviderStreamRequest::CompleteUpload(
                WorkerProviderUploadCompletionRequest {
                    handle: zero_handle,
                },
            ))
            .expect("upload completion request serializes"),
            completion_request_bytes
        );
        assert_eq!(
            postcard::to_stdvec(&WorkerProviderStreamResponse::UploadCompletion(Ok(
                WorkerProviderUploadCompletion {
                    download_url: "x".to_owned(),
                },
            )))
            .expect("upload completion response serializes"),
            vec![4, 0, 1, b'x']
        );

        let mut retry_request_bytes = vec![9, 16];
        retry_request_bytes.extend_from_slice(&[0; 20]);
        retry_request_bytes.push(1);
        assert_eq!(
            postcard::to_stdvec(&WorkerProviderStreamRequest::RetryRequest(
                WorkerProviderRetryRequest {
                    handle: zero_handle,
                    delay_milliseconds: 1,
                },
            ))
            .expect("retry request serializes"),
            retry_request_bytes
        );
    }

    #[test]
    fn malformed_provider_stream_messages_fail_before_routing() {
        let cases = [
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 0,
                    request: WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
                },
                WorkerProviderStreamError::InvalidOrder,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::StartRequest {
                        context: provider_context(),
                        head: WorkerProviderRequestHead {
                            endpoint: "https://invalid.example/\u{7f}".to_owned(),
                            secret_id: None,
                            method: WorkerProviderHttpMethod::Post,
                            headers: Vec::new(),
                            declared_body_bytes: None,
                        },
                    },
                },
                WorkerProviderStreamError::InvalidRequestAuthority,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::CheckCancelled(
                        WorkerProviderStreamHandle {
                            slot: 0,
                            ..provider_handle(1)
                        },
                    ),
                },
                WorkerProviderStreamError::InvalidHandle,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::StartRequest {
                        context: provider_context(),
                        head: provider_head(Some(MAX_WORKER_PROVIDER_BODY_BYTES + 1)),
                    },
                },
                WorkerProviderStreamError::BodyLimit,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::WriteRequestChunk(
                        WorkerProviderRequestChunk {
                            handle: provider_handle(1),
                            sequence: 0,
                            bytes: vec![0; MAX_WORKER_PROVIDER_CHUNK_BYTES + 1],
                            end: true,
                        },
                    ),
                },
                WorkerProviderStreamError::ChunkLimit,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::RequestCost(WorkerProviderCostRequest {
                        handle: provider_handle(1),
                        operation: "invalid operation".to_owned(),
                        currency: "USD".to_owned(),
                        maximum_microunits: 1,
                    }),
                },
                WorkerProviderStreamError::InvalidCostRequest,
            ),
            (
                WorkerMessage::ProviderStreamRequest {
                    call_id: 1,
                    request: WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                        handle: provider_handle(1),
                        sequence: 0,
                        completed: 1,
                        total: 0,
                        message: None,
                    }),
                },
                WorkerProviderStreamError::InvalidProgress,
            ),
            (
                WorkerMessage::ProviderStreamResponse {
                    call_id: 1,
                    response: WorkerProviderStreamResponse::Wait(Ok(
                        WorkerProviderWaitOutcome::Frame(WorkerProviderResponseFrame {
                            handle: provider_handle(1),
                            sequence: 1,
                            event: WorkerProviderResponseFrameEvent::Chunk(
                                WorkerProviderResponseChunk::NdjsonLine("not-json".to_owned()),
                            ),
                        }),
                    )),
                },
                WorkerProviderStreamError::InvalidNdjsonLine,
            ),
        ];
        for (message, expected) in cases {
            assert_eq!(
                encode_worker_frame(&envelope(message)),
                Err(WorkerProtocolError::InvalidProviderStream(expected))
            );
        }
    }

    fn checked_http_validator(
        maximum_requests: u32,
        cancellation: CancellationToken,
    ) -> Result<WorkerProviderStreamTransportValidator, Box<dyn std::error::Error>> {
        let mut validator = WorkerProviderStreamTransportValidator::checked_for_host_session(
            provider_context(),
            provider_contract(),
            cancellation,
        )?;
        validator.bind_checked_http_invocation(digest('a'), maximum_requests)?;
        validator.validate_request(
            1,
            &WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head: provider_head(Some(3)),
            },
        )?;
        validator.validate_response(
            1,
            &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1))),
        )?;
        Ok(validator)
    }

    fn checked_http_round_trip(
        validator: &mut WorkerProviderStreamTransportValidator,
        call_id: &mut u64,
        request: WorkerProviderStreamRequest,
        response: WorkerProviderStreamResponse,
    ) -> Result<(), WorkerProviderStreamError> {
        validator.validate_request(*call_id, &request)?;
        validator.validate_response(*call_id, &response)?;
        *call_id = call_id
            .checked_add(1)
            .ok_or(WorkerProviderStreamError::InvalidOrder)?;
        Ok(())
    }

    fn complete_checked_http_validator_request(
        validator: &mut WorkerProviderStreamTransportValidator,
        call_id: &mut u64,
        handle: WorkerProviderStreamHandle,
        request_body: Vec<u8>,
        response_body: &[u8],
    ) -> Result<(), WorkerProviderStreamError> {
        checked_http_round_trip(
            validator,
            call_id,
            WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                handle,
                sequence: 0,
                bytes: request_body,
                end: true,
            }),
            WorkerProviderStreamResponse::Unit(Ok(())),
        )?;
        let mut after_sequence = None;
        let events = std::iter::once(WorkerProviderResponseFrameEvent::Head(
            WorkerProviderResponseHead {
                status: 200,
                headers: Vec::new(),
            },
        ))
        .chain(response_body.chunks(128).map(|bytes| {
            WorkerProviderResponseFrameEvent::Chunk(WorkerProviderResponseChunk::Binary(
                bytes.to_vec(),
            ))
        }))
        .chain(std::iter::once(WorkerProviderResponseFrameEvent::Terminal(
            WorkerProviderTerminal::Completed(b"verified-http-terminal".to_vec()),
        )));
        for (sequence, event) in events.enumerate() {
            let sequence =
                u64::try_from(sequence).map_err(|_| WorkerProviderStreamError::InvalidSequence)?;
            checked_http_round_trip(
                validator,
                call_id,
                WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle,
                    after_sequence,
                    timeout_milliseconds: 10,
                }),
                WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                    WorkerProviderResponseFrame {
                        handle,
                        sequence,
                        event,
                    },
                ))),
            )?;
            after_sequence = Some(sequence);
        }
        Ok(())
    }

    #[test]
    fn provider_http_protocol_requires_checked_predecessors_and_preserves_invocation_progress()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut validator = checked_http_validator(3, CancellationToken::default())?;
        let binding = digest('a');
        let mut call_id = 2;
        let mut handle = provider_handle(1);
        for ordinal in 1..=3 {
            let body = if ordinal == 1 {
                b"abc".to_vec()
            } else {
                Vec::new()
            };
            complete_checked_http_validator_request(
                &mut validator,
                &mut call_id,
                handle,
                body,
                b"reply",
            )?;
            checked_http_round_trip(
                &mut validator,
                &mut call_id,
                WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                    handle,
                    sequence: u64::from(ordinal - 1),
                    completed: u64::from(ordinal),
                    total: 3,
                    message: None,
                }),
                WorkerProviderStreamResponse::Unit(Ok(())),
            )?;
            assert!(!validator.revoked);
            assert_eq!(validator.progress_completed, u64::from(ordinal));
            if ordinal == 3 {
                assert_eq!(validator.total_response_bytes, 15);
                assert_eq!(validator.total_request_bytes, 3);
                validator.finish_checked_http_invocation()?;
                assert!(validator.revoked);
                assert!(validator.pending.is_empty());
                return Ok(());
            }
            let mut head = provider_head(Some(0));
            head.method = WorkerProviderHttpMethod::Get;
            let predecessor = handle;
            validator.prepare_checked_http_continuation(
                &provider_context(),
                &binding,
                ordinal + 1,
                &head,
                |challenge| {
                    assert_eq!(challenge.context(), &provider_context());
                    assert_eq!(challenge.certified_binding_sha256(), &binding);
                    assert_eq!(challenge.ordinal(), ordinal + 1);
                    assert_eq!(challenge.predecessor_handle(), predecessor);
                    assert_eq!(
                        challenge.predecessor_receipt_sha256(),
                        "0279fa674c7f6e8aedd6f3d0390a478f61b326b2a9282ab246fbb16633f64678"
                    );
                    assert_eq!(challenge.request_head(), &head);
                    Ok(())
                },
            )?;
            handle = provider_handle(ordinal + 1);
            checked_http_round_trip(
                &mut validator,
                &mut call_id,
                WorkerProviderStreamRequest::StartRequest {
                    context: provider_context(),
                    head,
                },
                WorkerProviderStreamResponse::Stream(Ok(handle)),
            )?;
            assert_eq!(
                validator.validate_handle(predecessor),
                Err(WorkerProviderStreamError::RevokedHandle)
            );
        }
        Err("three-request protocol did not finalize".into())
    }

    #[test]
    fn provider_http_protocol_denials_revoke_without_admitting_continuations()
    -> Result<(), Box<dyn std::error::Error>> {
        for case in 0..11 {
            let cancellation = CancellationToken::default();
            let mut validator =
                checked_http_validator(if case == 4 { 1 } else { 3 }, cancellation.clone())?;
            let mut call_id = 2;
            if case != 0 {
                complete_checked_http_validator_request(
                    &mut validator,
                    &mut call_id,
                    provider_handle(1),
                    b"abc".to_vec(),
                    &[],
                )?;
            }
            let mut context = provider_context();
            if case == 5 {
                context.generation += 1;
            }
            if case == 6 {
                context.session_generation += 1;
            }
            if case == 7 {
                context.session_id = Uuid::from_u128(99);
            }
            if case == 8 {
                context.invocation += 1;
            }
            if case == 9 {
                cancellation.cancel();
            }
            let binding = if case == 1 { digest('b') } else { digest('a') };
            let ordinal = match case {
                2 => 1,
                3 => 3,
                _ => 2,
            };
            let mut verifier_called = false;
            let result = validator.prepare_checked_http_continuation(
                &context,
                &binding,
                ordinal,
                &provider_head(Some(3)),
                |_| {
                    verifier_called = true;
                    Err(WorkerProviderStreamError::InvalidRequestAuthority)
                },
            );
            assert!(result.is_err(), "case {case}");
            assert_eq!(verifier_called, case == 10, "case {case}");
            assert!(validator.revoked);
            assert!(validator.http_invocation.is_none());
            assert!(validator.pending.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_http_protocol_consumes_only_one_exact_continuation_and_fresh_handle()
    -> Result<(), Box<dyn std::error::Error>> {
        for case in 0..4 {
            let mut validator = checked_http_validator(3, CancellationToken::default())?;
            let mut call_id = 2;
            complete_checked_http_validator_request(
                &mut validator,
                &mut call_id,
                provider_handle(1),
                b"abc".to_vec(),
                &[],
            )?;
            let head = provider_head(Some(3));
            validator.prepare_checked_http_continuation(
                &provider_context(),
                &digest('a'),
                2,
                &head,
                |_| Ok(()),
            )?;
            if case == 0 {
                let mut called = false;
                assert!(
                    validator
                        .prepare_checked_http_continuation(
                            &provider_context(),
                            &digest('a'),
                            2,
                            &head,
                            |_| {
                                called = true;
                                Ok(())
                            }
                        )
                        .is_err()
                );
                assert!(!called);
            } else if case == 1 {
                let mut changed = head;
                changed.endpoint.push_str("/changed");
                assert!(
                    validator
                        .validate_request(
                            call_id,
                            &WorkerProviderStreamRequest::StartRequest {
                                context: provider_context(),
                                head: changed,
                            }
                        )
                        .is_err()
                );
            } else if case == 2 {
                validator.validate_request(
                    call_id,
                    &WorkerProviderStreamRequest::StartRequest {
                        context: provider_context(),
                        head,
                    },
                )?;
                assert!(
                    validator
                        .validate_response(
                            call_id,
                            &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1)))
                        )
                        .is_err()
                );
            } else {
                checked_http_round_trip(
                    &mut validator,
                    &mut call_id,
                    WorkerProviderStreamRequest::StartRequest {
                        context: provider_context(),
                        head,
                    },
                    WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
                )?;
                assert!(
                    validator
                        .prepare_checked_http_continuation(
                            &provider_context(),
                            &digest('a'),
                            3,
                            &provider_head(Some(3)),
                            |_| Ok(())
                        )
                        .is_err()
                );
            }
            assert!(validator.revoked, "case {case}");
            assert!(validator.http_invocation.is_none());
            assert!(validator.pending.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_http_protocol_rejects_retired_upload_handles_and_erases_failed_authority()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut validator = checked_http_validator(2, CancellationToken::default())?;
        let mut call_id = 2;
        complete_checked_http_validator_request(
            &mut validator,
            &mut call_id,
            provider_handle(1),
            b"abc".to_vec(),
            &[],
        )?;
        let head = provider_head(Some(3));
        validator.prepare_checked_http_continuation(
            &provider_context(),
            &digest('a'),
            2,
            &head,
            |_| Ok(()),
        )?;
        checked_http_round_trip(
            &mut validator,
            &mut call_id,
            WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head,
            },
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
        )?;
        validator.validate_request(
            call_id,
            &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                handle: provider_handle(2),
                port_id: "image.input".to_owned(),
                media_type: "image/png".to_owned(),
                byte_length: 1,
                content_sha256: format!("{:x}", Sha256::digest([7])),
            }),
        )?;
        assert!(
            validator
                .validate_response(
                    call_id,
                    &WorkerProviderStreamResponse::Stream(Ok(provider_handle(1)))
                )
                .is_err()
        );
        assert!(validator.revoked);
        assert!(validator.http_invocation.is_none());
        for response in [
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Cancelled)),
            WorkerProviderStreamResponse::Wait(Err(WorkerProviderStreamError::HostFailure)),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Terminal(
                        WorkerProviderTerminal::Cancelled,
                    ),
                },
            ))),
            WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                WorkerProviderResponseFrame {
                    handle: provider_handle(1),
                    sequence: 0,
                    event: WorkerProviderResponseFrameEvent::Terminal(
                        WorkerProviderTerminal::Failed {
                            code: "fixture.failure".to_owned(),
                            message: "failed".to_owned(),
                        },
                    ),
                },
            ))),
        ] {
            let mut validator = checked_http_validator(2, CancellationToken::default())?;
            validator.validate_request(
                2,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 10,
                }),
            )?;
            validator.validate_response(2, &response)?;
            assert!(validator.revoked);
            assert!(validator.http_invocation.is_none());
            assert!(validator.pending.is_empty());
        }
        Ok(())
    }

    #[test]
    fn provider_http_protocol_keeps_aggregate_body_limits_and_unbound_default()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut validator = checked_http_validator(2, CancellationToken::default())?;
        let mut call_id = 2;
        complete_checked_http_validator_request(
            &mut validator,
            &mut call_id,
            provider_handle(1),
            b"abc".to_vec(),
            &[7; 600],
        )?;
        let head = provider_head(Some(3));
        validator.prepare_checked_http_continuation(
            &provider_context(),
            &digest('a'),
            2,
            &head,
            |_| Ok(()),
        )?;
        checked_http_round_trip(
            &mut validator,
            &mut call_id,
            WorkerProviderStreamRequest::StartRequest {
                context: provider_context(),
                head,
            },
            WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
        )?;
        assert!(
            complete_checked_http_validator_request(
                &mut validator,
                &mut call_id,
                provider_handle(2),
                b"abc".to_vec(),
                &[7; 600]
            )
            .is_err()
        );
        assert!(validator.revoked);
        let mut legacy = started_provider_validator(CancellationToken::default());
        let mut call_id = 2;
        complete_checked_http_validator_request(
            &mut legacy,
            &mut call_id,
            provider_handle(1),
            b"abc".to_vec(),
            &[],
        )?;
        assert!(
            legacy
                .prepare_checked_http_continuation(
                    &provider_context(),
                    &digest('a'),
                    2,
                    &provider_head(Some(3)),
                    |_| Ok(())
                )
                .is_err()
        );
        assert!(legacy.revoked);
        for maximum_requests in [0, MAX_WORKER_PROVIDER_HTTP_REQUESTS + 1] {
            let mut empty = WorkerProviderStreamTransportValidator::checked_for_host_session(
                provider_context(),
                provider_contract(),
                CancellationToken::default(),
            )?;
            assert!(
                empty
                    .bind_checked_http_invocation(digest('a'), maximum_requests)
                    .is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn provider_stream_transport_is_ordered_bounded_and_incremental() {
        let mut validator = started_provider_validator(CancellationToken::default());
        validator
            .validate_request(
                2,
                &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(1),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("declared request body completes");
        assert_eq!(
            validator.validate_request(
                3,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 1,
                }),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("request chunk acknowledgement");
        assert_eq!(
            validator.validate_request(
                3,
                &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(1),
                    sequence: 1,
                    bytes: Vec::new(),
                    end: true,
                },),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_request(
                3,
                &WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
                    handle: provider_handle(1),
                    sequence: 0,
                    completed: 1,
                    total: 2,
                    message: Some("line\nretained".to_owned()),
                }),
            )
            .expect("progress mirrors the SDK byte-only message bound");
        validator
            .validate_response(3, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("progress acknowledgement");

        let frames = [
            WorkerProviderResponseFrame {
                handle: provider_handle(1),
                sequence: 0,
                event: WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                    status: 200,
                    headers: vec![WorkerProviderHeader {
                        name: "content-type".to_owned(),
                        value: "application/x-ndjson".to_owned(),
                    }],
                }),
            },
            WorkerProviderResponseFrame {
                handle: provider_handle(1),
                sequence: 1,
                event: WorkerProviderResponseFrameEvent::Chunk(
                    WorkerProviderResponseChunk::NdjsonLine("{\"value\":1}".to_owned()),
                ),
            },
            WorkerProviderResponseFrame {
                handle: provider_handle(1),
                sequence: 2,
                event: WorkerProviderResponseFrameEvent::Terminal(
                    WorkerProviderTerminal::Completed(vec![7]),
                ),
            },
        ];
        for (index, frame) in frames.into_iter().enumerate() {
            let call_id = u64::try_from(index).expect("index fits") + 4;
            validator
                .validate_request(
                    call_id,
                    &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                        handle: provider_handle(1),
                        after_sequence: index
                            .checked_sub(1)
                            .and_then(|value| u64::try_from(value).ok()),
                        timeout_milliseconds: 10,
                    }),
                )
                .expect("wait sequence follows the last response");
            validator
                .validate_response(
                    call_id,
                    &WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                        frame,
                    ))),
                )
                .expect("response frame advances incrementally");
            if index == 0 {
                assert_eq!(
                    validator.validate_request(
                        8,
                        &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                            handle: provider_handle(1),
                            after_sequence: None,
                            timeout_milliseconds: 10,
                        },),
                    ),
                    Err(WorkerProviderStreamError::InvalidSequence)
                );
            }
        }
        let final_progress = WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
            handle: provider_handle(1),
            sequence: 1,
            completed: 2,
            total: 2,
            message: Some("Completed".to_owned()),
        });
        validator
            .validate_request(8, &final_progress)
            .expect("one final completed progress update follows a successful terminal");
        validator
            .validate_response(8, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("final completed progress is acknowledged before revocation");
        assert_eq!(
            validator.validate_request(9, &final_progress),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
    }

    #[test]
    fn provider_upload_sha_and_rejected_responses_are_atomic() {
        let mut validator = started_provider_validator(CancellationToken::default());
        validator
            .validate_request(
                2,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(1),
                    port_id: "image.output".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 3,
                    content_sha256: format!("{:x}", Sha256::digest(b"abc")),
                }),
            )
            .expect("upload reservation is bounded");
        assert_eq!(
            validator.validate_request(
                3,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(1),
                    port_id: "image.second".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 1,
                    content_sha256: format!("{:x}", Sha256::digest(b"x")),
                }),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(
                2,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
            )
            .expect("unique upload handle is issued");
        assert_eq!(
            validator.validate_request(
                3,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(1),
                    port_id: "image.second".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 1,
                    content_sha256: format!("{:x}", Sha256::digest(b"x")),
                }),
            ),
            Err(WorkerProviderStreamError::InvalidUpload)
        );
        assert_eq!(
            validator.validate_request(
                4,
                &WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(2),
                    sequence: 0,
                    bytes: Vec::new(),
                    end: false,
                },),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        let wrong = WorkerProviderRequestChunk {
            handle: provider_handle(2),
            sequence: 0,
            bytes: b"abd".to_vec(),
            end: true,
        };
        assert_eq!(
            validator.validate_request(4, &WorkerProviderStreamRequest::WriteUploadChunk(wrong),),
            Err(WorkerProviderStreamError::InvalidUpload)
        );
        validator
            .validate_request(
                4,
                &WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(2),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("failed digest attempt did not mutate upload state");
        assert_eq!(
            validator.validate_request(
                5,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 1,
                }),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(
                4,
                &WorkerProviderStreamResponse::Unit(Err(WorkerProviderStreamError::HostFailure)),
            )
            .expect("typed host failure is a valid response");
        assert_eq!(
            validator.validate_request(
                5,
                &WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
            ),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
    }

    #[test]
    fn provider_upload_completion_is_fresh_bounded_and_separate_from_primary_wait() {
        let mut validator = started_provider_validator(CancellationToken::default());
        validator
            .validate_request(
                2,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(1),
                    port_id: "image.output".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 3,
                    content_sha256: format!("{:x}", Sha256::digest(b"abc")),
                }),
            )
            .expect("upload reservation is valid");
        validator
            .validate_response(
                2,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
            )
            .expect("fresh upload handle is valid");
        assert_eq!(
            validator.validate_request(
                3,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest {
                        handle: provider_handle(2),
                    },
                ),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_request(
                3,
                &WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(2),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("upload bytes are valid");
        validator
            .validate_response(3, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("upload write is acknowledged");

        let mut foreign = provider_handle(2);
        foreign.slot = 9;
        assert_eq!(
            validator.validate_request(
                4,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest { handle: foreign },
                ),
            ),
            Err(WorkerProviderStreamError::ForeignHandle)
        );
        let mut stale = provider_handle(2);
        stale.generation = stale.generation.saturating_add(1);
        assert_eq!(
            validator.validate_request(
                4,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest { handle: stale },
                ),
            ),
            Err(WorkerProviderStreamError::StaleGeneration)
        );
        validator
            .validate_request(
                4,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest {
                        handle: provider_handle(2),
                    },
                ),
            )
            .expect("completed upload accepts one receipt request");
        for invalid in [
            "http://fixture.invalid/file",
            "https://",
            "https://fixture.invalid",
            "https://fixture.invalid/",
            "https://localhost/file",
            "https://service.local/file",
            "https://service.internal/file",
            "https://SERVICE.LOCAL/file",
            "https://service.home.arpa/file",
            "https://127.0.0.1/file",
            "https://[::1]/file",
            "https://user:secret@fixture.invalid/file",
            "https://fixture.invalid:/file",
            "https://fixture.invalid:0/file",
            "https://fixture.invalid:65536/file",
            "https://fixture.invalid:not-a-port/file",
            "https://fixture.invalid:443:444/file",
            "https://-fixture.invalid/file",
            "https://fixture..invalid/file",
            "https://fixture.invalid/file?token=secret",
            "https://fixture.invalid/file#fragment",
            "https://fixture.invalid/file\nnext",
            "https://fixture.invalid/file%",
            "https://fixture.invalid/file%0",
            "https://fixture.invalid/file%GG",
            "https://fixture.invalid/file%0Aembedded-control",
        ] {
            assert_eq!(
                validator.validate_response(
                    4,
                    &WorkerProviderStreamResponse::UploadCompletion(Ok(
                        WorkerProviderUploadCompletion {
                            download_url: invalid.to_owned(),
                        },
                    )),
                ),
                Err(WorkerProviderStreamError::InvalidUpload)
            );
        }
        assert_eq!(
            validator.validate_response(
                4,
                &WorkerProviderStreamResponse::UploadCompletion(Ok(
                    WorkerProviderUploadCompletion {
                        download_url: format!(
                            "https://fixture.invalid/{}",
                            "x".repeat(MAX_WORKER_PROVIDER_UPLOAD_URL_BYTES)
                        ),
                    },
                )),
            ),
            Err(WorkerProviderStreamError::InvalidUpload)
        );
        validator
            .validate_response(
                4,
                &WorkerProviderStreamResponse::UploadCompletion(Ok(
                    WorkerProviderUploadCompletion {
                        download_url: "https://fixture.invalid:443/storage/reference%20file.bin"
                            .to_owned(),
                    },
                )),
            )
            .expect("valid HTTPS upload receipt is admitted atomically");
        assert_eq!(
            validator.validate_request(
                5,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest {
                        handle: provider_handle(2),
                    },
                ),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        assert_eq!(
            validator.validate_request(
                5,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(2),
                    after_sequence: None,
                    timeout_milliseconds: 10,
                }),
            ),
            Err(WorkerProviderStreamError::ForeignHandle)
        );

        let mut revoked = started_provider_validator(CancellationToken::default());
        revoked.revoke();
        assert_eq!(
            revoked.validate_request(
                2,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest {
                        handle: provider_handle(2),
                    },
                ),
            ),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
    }

    #[test]
    fn provider_stream_retry_issues_a_fresh_handle_and_preserves_cumulative_quotas() {
        let mut validator = started_provider_validator(CancellationToken::default());
        validator
            .validate_request(
                2,
                &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(1),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("first request body is admitted");
        validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("first request body is acknowledged");
        validator
            .validate_request(
                3,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(1),
                    port_id: "image.output".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 3,
                    content_sha256: format!("{:x}", Sha256::digest(b"abc")),
                }),
            )
            .expect("upload is admitted");
        validator
            .validate_response(
                3,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(2))),
            )
            .expect("upload handle is issued");
        validator
            .validate_request(
                4,
                &WorkerProviderStreamRequest::WriteUploadChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(2),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("upload body is admitted");
        validator
            .validate_response(4, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("upload body is acknowledged");
        validator
            .validate_request(
                5,
                &WorkerProviderStreamRequest::CompleteUpload(
                    WorkerProviderUploadCompletionRequest {
                        handle: provider_handle(2),
                    },
                ),
            )
            .expect("upload completion is admitted");
        validator
            .validate_response(
                5,
                &WorkerProviderStreamResponse::UploadCompletion(Ok(
                    WorkerProviderUploadCompletion {
                        download_url: "https://fixture.invalid/storage/image.bin".to_owned(),
                    },
                )),
            )
            .expect("upload completion is acknowledged");
        validator
            .validate_request(
                6,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 10,
                }),
            )
            .expect("first response head is requested");
        validator
            .validate_response(
                6,
                &WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                    WorkerProviderResponseFrame {
                        handle: provider_handle(1),
                        sequence: 0,
                        event: WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                            status: 503,
                            headers: Vec::new(),
                        }),
                    },
                ))),
            )
            .expect("retryable head is admitted");
        let retry = WorkerProviderStreamRequest::RetryRequest(WorkerProviderRetryRequest {
            handle: provider_handle(1),
            delay_milliseconds: 0,
        });
        validator
            .validate_request(7, &retry)
            .expect("retry is admitted after a retryable head");
        validator
            .validate_response(
                7,
                &WorkerProviderStreamResponse::Stream(Ok(provider_handle(3))),
            )
            .expect("fresh retry handle is admitted");
        assert_eq!(
            validator.validate_request(
                8,
                &WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
            ),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
        assert_eq!(
            validator.validate_request(
                8,
                &WorkerProviderStreamRequest::CheckCancelled(provider_handle(2)),
            ),
            Err(WorkerProviderStreamError::RevokedHandle)
        );
        validator
            .validate_request(
                8,
                &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(3),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("request body can be replayed on the fresh handle");
        validator
            .validate_response(8, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("replayed request body is acknowledged");
        assert_eq!(
            validator.validate_request(
                9,
                &WorkerProviderStreamRequest::StartUpload(WorkerProviderUploadRequest {
                    handle: provider_handle(3),
                    port_id: "image.output".to_owned(),
                    media_type: "application/octet-stream".to_owned(),
                    byte_length: 1,
                    content_sha256: format!("{:x}", Sha256::digest(b"x")),
                }),
            ),
            Err(WorkerProviderStreamError::InvalidUpload)
        );
    }

    #[test]
    fn provider_stream_rejects_foreign_stale_cancelled_and_reused_sessions() {
        let mut validator = started_provider_validator(CancellationToken::default());
        let mut foreign = provider_handle(1);
        foreign.session_id = Uuid::from_u128(99);
        assert_eq!(
            validator.validate_request(2, &WorkerProviderStreamRequest::CheckCancelled(foreign),),
            Err(WorkerProviderStreamError::ForeignSession)
        );
        let mut stale = provider_handle(1);
        stale.generation += 1;
        assert_eq!(
            validator.validate_request(2, &WorkerProviderStreamRequest::CheckCancelled(stale),),
            Err(WorkerProviderStreamError::StaleGeneration)
        );
        assert!(matches!(
            validator.restart(provider_context(), CancellationToken::default()),
            Err(WorkerProviderStreamError::StaleSession)
        ));

        let rollback_generation = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(2),
            session_generation: 2,
            invocation: 8,
            generation: 2,
        };
        assert!(matches!(
            started_provider_validator(CancellationToken::default())
                .restart(rollback_generation, CancellationToken::default(),),
            Err(WorkerProviderStreamError::StaleGeneration)
        ));
        let fresh_context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(2),
            session_generation: 2,
            invocation: 8,
            generation: 4,
        };
        let mut restarted = started_provider_validator(CancellationToken::default())
            .restart(fresh_context.clone(), CancellationToken::default())
            .expect("a strictly fresh host-issued session can restart transport state");
        restarted
            .validate_request(
                2,
                &WorkerProviderStreamRequest::StartRequest {
                    context: fresh_context,
                    head: provider_head(None),
                },
            )
            .expect("fresh session preserves the call-id high-water mark");

        let mut revoked = started_provider_validator(CancellationToken::default());
        revoked.revoke();
        assert_eq!(
            revoked.validate_request(
                2,
                &WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
            ),
            Err(WorkerProviderStreamError::RevokedHandle)
        );

        let cancellation = CancellationToken::default();
        let mut cancelled = started_provider_validator(cancellation.clone());
        assert!(cancellation.cancel());
        assert_eq!(
            cancelled.validate_request(
                2,
                &WorkerProviderStreamRequest::CheckCancelled(provider_handle(1)),
            ),
            Err(WorkerProviderStreamError::Cancelled)
        );
    }

    #[test]
    fn provider_stream_rejections_do_not_advance_ordered_state() {
        let mut validator = started_provider_validator(CancellationToken::default());
        let out_of_order = WorkerProviderRequestChunk {
            handle: provider_handle(1),
            sequence: 1,
            bytes: b"abc".to_vec(),
            end: true,
        };
        assert_eq!(
            validator.validate_request(
                2,
                &WorkerProviderStreamRequest::WriteRequestChunk(out_of_order),
            ),
            Err(WorkerProviderStreamError::InvalidSequence)
        );
        validator
            .validate_request(
                2,
                &WorkerProviderStreamRequest::WriteRequestChunk(WorkerProviderRequestChunk {
                    handle: provider_handle(1),
                    sequence: 0,
                    bytes: b"abc".to_vec(),
                    end: true,
                }),
            )
            .expect("rejected sequence did not advance state");
        assert_eq!(
            validator.validate_response(
                2,
                &WorkerProviderStreamResponse::Cost(Ok(WorkerProviderCostResponse {
                    accepted: false,
                    approved_microunits: 0,
                    receipt: Vec::new(),
                })),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("wrong response kind did not consume the pending call");
        validator
            .validate_request(
                3,
                &WorkerProviderStreamRequest::WaitResponse(WorkerProviderWaitRequest {
                    handle: provider_handle(1),
                    after_sequence: None,
                    timeout_milliseconds: 1,
                }),
            )
            .expect("first wait");
        let chunk_before_head = WorkerProviderResponseFrame {
            handle: provider_handle(1),
            sequence: 0,
            event: WorkerProviderResponseFrameEvent::Chunk(WorkerProviderResponseChunk::Binary(
                vec![1],
            )),
        };
        assert_eq!(
            validator.validate_response(
                3,
                &WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                    chunk_before_head,
                ))),
            ),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(
                3,
                &WorkerProviderStreamResponse::Wait(Ok(WorkerProviderWaitOutcome::Frame(
                    WorkerProviderResponseFrame {
                        handle: provider_handle(1),
                        sequence: 0,
                        event: WorkerProviderResponseFrameEvent::Head(WorkerProviderResponseHead {
                            status: 200,
                            headers: Vec::new(),
                        }),
                    },
                ))),
            )
            .expect("rejected response did not advance state");
    }

    #[test]
    fn provider_call_ids_are_monotonic_and_delayed_responses_cannot_ack_new_calls() {
        let mut validator = started_provider_validator(CancellationToken::default());
        let progress = WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
            handle: provider_handle(1),
            sequence: 0,
            completed: 1,
            total: 2,
            message: None,
        });
        validator
            .validate_request(2, &progress)
            .expect("next call id is admitted");
        validator
            .validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("first progress completes");
        assert_eq!(
            validator.validate_request(2, &progress),
            Err(WorkerProviderStreamError::InvalidOrder)
        );

        let next_progress = WorkerProviderStreamRequest::ReportProgress(WorkerProviderProgress {
            handle: provider_handle(1),
            sequence: 1,
            completed: 2,
            total: 2,
            message: None,
        });
        validator
            .validate_request(3, &next_progress)
            .expect("fresh call id is admitted");
        assert_eq!(
            validator.validate_response(2, &WorkerProviderStreamResponse::Unit(Ok(()))),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        validator
            .validate_response(3, &WorkerProviderStreamResponse::Unit(Ok(())))
            .expect("delayed response did not consume the current call");

        let fresh_context = WorkerProviderInvocationContext {
            session_id: Uuid::from_u128(2),
            session_generation: 2,
            invocation: 8,
            generation: 4,
        };
        let mut restarted = started_provider_validator(CancellationToken::default())
            .restart(fresh_context.clone(), CancellationToken::default())
            .expect("fresh context restarts the call-id domain");
        let start = WorkerProviderStreamRequest::StartRequest {
            context: fresh_context,
            head: provider_head(None),
        };
        assert_eq!(
            restarted.validate_request(1, &start),
            Err(WorkerProviderStreamError::InvalidOrder)
        );
        restarted
            .validate_request(2, &start)
            .expect("restart preserves the prior high-water mark");
    }

    #[test]
    fn provider_contract_and_request_head_error_precedence_match_the_sdk() {
        for (maximum_uploads, maximum_upload_body_bytes) in [(1, 0), (0, 1)] {
            let mut contract = provider_contract();
            contract.uploads = false;
            contract.maximum_uploads = maximum_uploads;
            contract.maximum_upload_body_bytes = maximum_upload_body_bytes;
            assert!(matches!(
                WorkerProviderStreamTransportValidator::checked_for_host_session(
                    provider_context(),
                    contract,
                    CancellationToken::default(),
                ),
                Err(WorkerProviderStreamError::InvalidContract)
            ));
        }

        let mut invalid = provider_head(Some(MAX_WORKER_PROVIDER_BODY_BYTES + 1));
        invalid.endpoint.push('\u{7f}');
        invalid.method = WorkerProviderHttpMethod::Put;
        invalid.headers.push(WorkerProviderHeader {
            name: "bad header".to_owned(),
            value: "value".to_owned(),
        });
        assert_eq!(
            validate_provider_request_head(&invalid, &provider_contract()),
            Err(WorkerProviderStreamError::InvalidRequestAuthority)
        );
        invalid.endpoint = "https://api.provider.invalid/v1/generate".to_owned();
        assert_eq!(
            validate_provider_request_head(&invalid, &provider_contract()),
            Err(WorkerProviderStreamError::InvalidMethod)
        );
        invalid.method = WorkerProviderHttpMethod::Post;
        assert_eq!(
            validate_provider_request_head(&invalid, &provider_contract()),
            Err(WorkerProviderStreamError::InvalidHeaders)
        );
        invalid.headers.pop();
        assert_eq!(
            validate_provider_request_head(&invalid, &provider_contract()),
            Err(WorkerProviderStreamError::BodyLimit)
        );
    }

    #[test]
    fn provider_worker_wire_matches_v2_authority_shape_and_excludes_authority_material() {
        let source = include_str!("worker_protocol.rs");
        let wit = include_str!("../../sim_plugin_sdk/wit/provider-v2/sim-provider-plugin.wit");
        for declaration in [
            "endpoint: string",
            "secret-id: option<string>",
            "declared-body-bytes: option<u64>",
            "after-sequence: option<u64>",
            "timeout-milliseconds: u64",
            "message: option<string>",
            "content-sha256: string",
        ] {
            assert!(wit.contains(declaration));
        }
        let request_head = source
            .find("pub struct WorkerProviderRequestHead")
            .expect("request-head declaration");
        let request_chunk = source[request_head..]
            .find("pub struct WorkerProviderRequestChunk")
            .map(|offset| request_head + offset)
            .expect("request chunk declaration");
        let request_head = &source[request_head..request_chunk];
        let ordered_fields = [
            "pub endpoint: String",
            "pub secret_id: Option<String>",
            "pub method: WorkerProviderHttpMethod",
            "pub headers: Vec<WorkerProviderHeader>",
            "pub declared_body_bytes: Option<u64>",
        ];
        let mut cursor = 0;
        for field in ordered_fields {
            let offset = request_head[cursor..]
                .find(field)
                .expect("request-head field is present in WIT order");
            cursor += offset + field.len();
        }
        let provider_start = source
            .find("pub enum WorkerProviderHttpMethod")
            .expect("provider worker declarations");
        let message_start = source
            .find("pub enum WorkerMessage")
            .expect("worker message declaration");
        let wire = &source[provider_start..message_start];
        for prohibited in [
            "PathBuf",
            "NativeOpaqueHandle",
            "secret_bytes",
            "provider_id",
            "authorization_decision",
        ] {
            assert!(
                !wire.contains(prohibited),
                "prohibited wire field {prohibited}"
            );
        }
    }

    #[test]
    fn fatal_diagnostics_are_bounded_typed_and_control_free() {
        for (code, message) in [
            (String::new(), "message".to_owned()),
            ("UPPERCASE".to_owned(), "message".to_owned()),
            ("path/code".to_owned(), "message".to_owned()),
            (
                "a".repeat(MAX_WORKER_FATAL_CODE_BYTES + 1),
                "message".to_owned(),
            ),
            ("backend_unavailable".to_owned(), String::new()),
            (
                "backend_unavailable".to_owned(),
                "m".repeat(MAX_WORKER_FATAL_MESSAGE_BYTES + 1),
            ),
            ("backend_unavailable".to_owned(), "line\nfeed".to_owned()),
        ] {
            let message = envelope(WorkerMessage::Fatal { code, message });
            assert!(matches!(
                encode_worker_frame(&message),
                Err(WorkerProtocolError::InvalidPayload(reason))
                    if reason == "worker fatal diagnostic is invalid"
            ));
        }
        let bounded = envelope(WorkerMessage::Fatal {
            code: "backend_unavailable".to_owned(),
            message: "typed\tdiagnostic".to_owned(),
        });
        let frame = encode_worker_frame(&bounded).expect("bounded fatal diagnostic");
        assert_eq!(decode_worker_frame(&frame), Ok(bounded));
    }

    #[test]
    fn registry_deployment_rejection_requires_its_exact_wire_version() {
        let mut rejection = WorkerRegistryDeploymentRejection::new(
            WorkerRegistryGeneration::new(1).expect("nonzero generation"),
            digest('c'),
            WorkerRegistryDeploymentRejectionReason::InvalidCandidate,
        );
        rejection.version = WORKER_REGISTRY_DEPLOYMENT_REJECTION_VERSION + 1;
        let malformed = envelope(WorkerMessage::RegistryDeploymentRejected { rejection });
        assert!(matches!(
            encode_worker_frame(&malformed),
            Err(WorkerProtocolError::InvalidComponentDeployment(message))
                if message.contains("rejection version")
        ));
        let payload = postcard::to_stdvec(&malformed).expect("malformed fixture serializes");
        let mut frame = u32::try_from(payload.len())
            .expect("fixture length fits the frame prefix")
            .to_le_bytes()
            .to_vec();
        frame.extend_from_slice(&payload);
        assert!(matches!(
            decode_worker_frame(&frame),
            Err(WorkerProtocolError::InvalidComponentDeployment(message))
                if message.contains("rejection version")
        ));
    }

    #[test]
    fn component_deployment_contract_is_bounded_canonical_and_path_free() {
        assert_eq!(
            WorkerRegistryGeneration::new(0),
            Err(WorkerComponentDeploymentError::ZeroGeneration)
        );
        for invalid in ["a".repeat(63), "A".repeat(64), "g".repeat(64)] {
            assert_eq!(
                WorkerSha256Digest::new(invalid),
                Err(WorkerComponentDeploymentError::InvalidSha256)
            );
        }
        assert_eq!(
            component_descriptor("extension-a", 'b', 0, 1),
            Err(WorkerComponentDeploymentError::EmptyContent(
                WorkerComponentContent::Manifest
            ))
        );
        assert_eq!(
            component_descriptor(
                "extension-a",
                'b',
                1,
                u64::try_from(MAX_WORKER_COMPONENT_BYTES).expect("bounded maximum") + 1,
            ),
            Err(WorkerComponentDeploymentError::ContentTooLarge(
                WorkerComponentContent::Component
            ))
        );
        assert_eq!(
            component_descriptor("../escape", 'b', 2, 3),
            Err(WorkerComponentDeploymentError::InvalidIdentity(
                WorkerComponentIdentityField::ExtensionId
            ))
        );
        let descriptor =
            component_descriptor("extension-a", 'b', 2, 3).expect("bounded descriptor");
        assert_eq!(descriptor.extension_id(), "extension-a");
        assert_eq!(descriptor.extension_version(), "1.2.3");
        assert_eq!(descriptor.plugin_identifier(), "plugin.identifier");
        assert_eq!(descriptor.plugin_version(), "4.5.6");
        assert_eq!(descriptor.authorization_generation(), &digest('d'));
        assert_eq!(
            WorkerRegistryDeploymentBegin::new(
                WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                digest('c'),
                vec![descriptor.clone(), descriptor],
            ),
            Err(WorkerComponentDeploymentError::DuplicateExtensionIdentity)
        );
        assert_eq!(
            WorkerRegistryDeploymentBegin::new(
                WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                digest('c'),
                vec![
                    component_descriptor("extension-b", 'c', 2, 3).expect("bounded descriptor"),
                    component_descriptor("extension-a", 'b', 2, 3).expect("bounded descriptor"),
                ],
            ),
            Err(WorkerComponentDeploymentError::NonCanonicalComponentOrder)
        );
        assert_eq!(
            WorkerRegistryDeploymentChunk::new(
                WorkerRegistryGeneration::new(1).expect("nonzero generation"),
                0,
                WorkerComponentContent::Manifest,
                0,
                vec![0; MAX_WORKER_COMPONENT_CHUNK_BYTES + 1],
            ),
            Err(WorkerComponentDeploymentError::ChunkTooLarge)
        );

        let source = include_str!("worker_protocol.rs");
        let deployment_start = source
            .find("pub struct WorkerComponentDescriptor")
            .expect("deployment declaration");
        let envelope_start = source
            .find("pub struct WorkerEnvelope")
            .expect("worker envelope declaration");
        let deployment_wire = &source[deployment_start..envelope_start];
        for prohibited in ["Path", "PathBuf", "AssetRoot", "TypedRoot"] {
            assert!(!deployment_wire.contains(prohibited));
        }
    }

    #[test]
    fn registry_digest_material_binds_component_identity_and_authorization() {
        let generation = WorkerRegistryGeneration::new(1).expect("nonzero generation");
        let original = deployment_begin();
        let changed_identity = WorkerRegistryDeploymentBegin::new(
            generation,
            digest('c'),
            vec![
                WorkerComponentDescriptor::new(
                    "extension-a",
                    "1.2.3",
                    "different.plugin",
                    "4.5.6",
                    digest('d'),
                    digest('a'),
                    digest('b'),
                    2,
                    1,
                    3,
                )
                .expect("bounded descriptor"),
            ],
        )
        .expect("bounded deployment");
        let changed_authorization = WorkerRegistryDeploymentBegin::new(
            generation,
            digest('c'),
            vec![
                WorkerComponentDescriptor::new(
                    "extension-a",
                    "1.2.3",
                    "plugin.identifier",
                    "4.5.6",
                    digest('e'),
                    digest('a'),
                    digest('b'),
                    2,
                    1,
                    3,
                )
                .expect("bounded descriptor"),
            ],
        )
        .expect("bounded deployment");

        assert_ne!(
            original.digest_material(),
            changed_identity.digest_material()
        );
        assert_ne!(
            original.digest_material(),
            changed_authorization.digest_material()
        );
    }

    #[test]
    fn opaque_worker_extensions_are_rejected() {
        let mut message = envelope(WorkerMessage::Ready);
        message.extensions.insert("escape".into(), vec![1]);
        assert_eq!(
            encode_worker_frame(&message),
            Err(WorkerProtocolError::OpaqueExtensions)
        );
    }

    #[test]
    fn invalid_length_and_version_are_rejected() {
        assert_eq!(
            decode_worker_frame(&[0, 0, 0]),
            Err(WorkerProtocolError::LengthMismatch)
        );
        let mut unsupported = envelope(WorkerMessage::Ready);
        unsupported.version = WORKER_PROTOCOL_VERSION + 1;
        assert_eq!(
            encode_worker_frame(&unsupported),
            Err(WorkerProtocolError::UnsupportedVersion(
                WORKER_PROTOCOL_VERSION + 1
            ))
        );

        let legacy = envelope(WorkerMessage::Ready);
        let mut payload = postcard::to_stdvec(&legacy).expect("current envelope serializes");
        assert_eq!(
            payload.first().copied(),
            Some(WORKER_PROTOCOL_VERSION as u8)
        );
        for unsupported_version in [
            PREVIOUS_WORKER_PROTOCOL_VERSION,
            LEGACY_WORKER_PROTOCOL_VERSION,
        ] {
            payload[0] = unsupported_version as u8;
            let mut frame = u32::try_from(payload.len())
                .expect("fixture length fits")
                .to_le_bytes()
                .to_vec();
            frame.extend_from_slice(&payload);
            assert_eq!(
                decode_worker_frame(&frame),
                Err(WorkerProtocolError::UnsupportedVersion(unsupported_version))
            );
        }
    }

    #[test]
    fn legacy_protocol_is_rejected_before_changed_payload_decode() {
        for unsupported_version in [
            0,
            1,
            LEGACY_WORKER_PROTOCOL_VERSION,
            PREVIOUS_WORKER_PROTOCOL_VERSION,
            WORKER_PROTOCOL_VERSION + 1,
            255,
            u16::MAX,
        ] {
            let mut payload = postcard::to_stdvec(&unsupported_version)
                .expect("unsupported version serializes independently");
            payload.push(0xff);
            let mut frame = u32::try_from(payload.len())
                .expect("fixture length fits")
                .to_le_bytes()
                .to_vec();
            frame.extend_from_slice(&payload);
            assert_eq!(
                decode_worker_frame(&frame),
                Err(WorkerProtocolError::UnsupportedVersion(unsupported_version))
            );
        }
    }

    #[test]
    fn oversized_frames_are_rejected_before_payload_decoding() {
        let oversized = envelope(WorkerMessage::Execute {
            plan: vec![0; MAX_WORKER_FRAME_BYTES],
        });
        assert_eq!(
            encode_worker_frame(&oversized),
            Err(WorkerProtocolError::Oversized)
        );

        let mut frame = (u32::try_from(MAX_WORKER_FRAME_BYTES).expect("bounded maximum") + 1)
            .to_le_bytes()
            .to_vec();
        frame.resize(MAX_WORKER_FRAME_BYTES + 5, 0);
        assert_eq!(
            decode_worker_frame(&frame),
            Err(WorkerProtocolError::Oversized)
        );

        let oversized_event = envelope(WorkerMessage::Event {
            event: vec![0; MAX_ENCODED_PREVIEW_BYTES + 1],
        });
        assert_eq!(
            encode_worker_frame(&oversized_event),
            Err(WorkerProtocolError::OversizedEvent)
        );
    }

    #[test]
    fn plugin_results_round_trip_through_the_postcard_envelope() {
        let succeeded = envelope(WorkerMessage::PluginResult {
            outcome: WorkerPluginExecutionOutcome::Succeeded(vec![1, 2, 3]),
        });
        let succeeded_frame =
            encode_worker_frame(&succeeded).expect("plugin result envelope must encode");
        assert_eq!(decode_worker_frame(&succeeded_frame), Ok(succeeded));

        let failed = envelope(WorkerMessage::PluginResult {
            outcome: WorkerPluginExecutionOutcome::Failed(
                WorkerPluginExecutionFailure::CapabilityDenied,
            ),
        });
        let failed_frame =
            encode_worker_frame(&failed).expect("plugin failure envelope must encode");
        assert_eq!(decode_worker_frame(&failed_frame), Ok(failed));

        let trapped = envelope(WorkerMessage::PluginResult {
            outcome: WorkerPluginExecutionOutcome::Failed(WorkerPluginExecutionFailure::Trap {
                diagnostic: "wasm trap: interrupt".to_owned(),
            }),
        });
        let trapped_frame =
            encode_worker_frame(&trapped).expect("bounded trap diagnostic must encode");
        assert_eq!(decode_worker_frame(&trapped_frame), Ok(trapped));

        for diagnostic in [
            String::new(),
            "x".repeat(MAX_WORKER_PLUGIN_DIAGNOSTIC_CHARS + 1),
            "wasm trap\0secret".to_owned(),
        ] {
            assert!(matches!(
                encode_worker_frame(&envelope(WorkerMessage::PluginResult {
                    outcome: WorkerPluginExecutionOutcome::Failed(
                        WorkerPluginExecutionFailure::Trap { diagnostic },
                    ),
                })),
                Err(WorkerProtocolError::InvalidPayload(_))
            ));
        }
    }

    #[test]
    fn output_proposals_are_bounded_on_construction_and_decode() {
        let proposal = WorkerOutputProposal::new(
            Uuid::from_u128(0xfeed),
            b"metadata".to_vec(),
            b"content".to_vec(),
        )
        .expect("bounded proposal must construct");
        let encoded = postcard::to_stdvec(&proposal).expect("bounded proposal must encode");
        assert_eq!(
            postcard::from_bytes::<WorkerOutputProposal>(&encoded),
            Ok(proposal)
        );

        assert_eq!(
            WorkerOutputProposal::new(
                Uuid::nil(),
                vec![0; MAX_WORKER_OUTPUT_METADATA_BYTES + 1],
                Vec::new(),
            ),
            Err(WorkerOutputProposalError::MetadataTooLarge)
        );
        assert_eq!(
            WorkerOutputProposal::new(
                Uuid::nil(),
                Vec::new(),
                vec![0; MAX_WORKER_OUTPUT_CONTENT_BYTES + 1],
            ),
            Err(WorkerOutputProposalError::ContentTooLarge)
        );

        let wire = WorkerOutputProposalWire {
            proposal_id: Uuid::nil(),
            metadata: Vec::new(),
            content: vec![0; MAX_WORKER_OUTPUT_CONTENT_BYTES + 1],
        };
        let encoded = postcard::to_stdvec(&wire).expect("wire fixture serializes");
        assert!(postcard::from_bytes::<WorkerOutputProposal>(&encoded).is_err());
    }

    #[test]
    fn val_cancel_001_worker_projection_and_bounds_are_explicit() {
        let cancellation = crate::CancellationToken::default();
        assert!(cancellation.cancel());
        let cancel = envelope(WorkerMessage::Cancel {
            reason: "operator requested cancellation".to_owned(),
        });
        let frame = encode_worker_frame(&cancel).expect("cancel projection must encode");
        assert_eq!(decode_worker_frame(&frame), Ok(cancel));

        let oversized_event = envelope(WorkerMessage::Event {
            event: vec![0; MAX_ENCODED_PREVIEW_BYTES + 1],
        });
        assert_eq!(
            encode_worker_frame(&oversized_event),
            Err(WorkerProtocolError::OversizedEvent)
        );

        let oversized_frame = envelope(WorkerMessage::Execute {
            plan: vec![0; MAX_WORKER_FRAME_BYTES],
        });
        assert_eq!(
            encode_worker_frame(&oversized_frame),
            Err(WorkerProtocolError::Oversized)
        );
    }

    #[test]
    fn worker_wire_schema_declares_no_host_path_or_typed_root() {
        let source = include_str!("worker_protocol.rs");
        let message_start = source
            .find("pub enum WorkerMessage")
            .expect("worker message declaration");
        let error_start = source[message_start..]
            .find("pub enum WorkerProtocolError")
            .map(|offset| message_start + offset)
            .expect("worker protocol error declaration");
        let wire_declarations = &source[..error_start];

        for prohibited in ["Path", "PathBuf", "AssetRoot", "AssetRoots", "TypedRoot"] {
            assert!(
                !wire_declarations.contains(prohibited),
                "worker wire schema contains prohibited host type {prohibited}"
            );
        }
    }

    #[test]
    fn backend_capabilities_are_bounded_canonical_and_checked_on_decode() {
        let duplicate = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Unary(WorkerUnaryOperationV1::Absolute),
            WorkerTensorRoleV1::Input,
            WorkerDType::F32,
            WorkerLayout::Contiguous,
        )
        .expect("unary is a tensor primitive");
        assert_eq!(
            WorkerBackendCapabilities::new(DeviceKind::Cpu, 0, vec![duplicate, duplicate], vec![],),
            Err(WorkerBackendCapabilityError::Duplicate)
        );
        let canonical =
            WorkerBackendCapabilities::new(DeviceKind::Cpu, 0, vec![duplicate], vec![duplicate])
                .expect("unique declarations canonicalize");
        assert_eq!(canonical.supported(), [duplicate]);
        assert_eq!(canonical.deterministic(), [duplicate]);

        let empty = WorkerBackendCapabilitiesWire {
            device: DeviceKind::Cpu,
            ordinal: 0,
            supported: vec![],
            deterministic: vec![],
            properties: None,
        };
        assert_eq!(
            WorkerBackendCapabilities::try_from(empty.clone()),
            Err(WorkerBackendCapabilityError::Empty)
        );
        let malformed_payload = postcard::to_stdvec(&empty).expect("encode malformed wire fixture");
        assert!(postcard::from_bytes::<WorkerBackendCapabilities>(&malformed_payload).is_err());

        let oversized = WorkerBackendCapabilitiesWire {
            device: DeviceKind::Cpu,
            ordinal: 0,
            supported: vec![duplicate; MAX_WORKER_BACKEND_CAPABILITY_ENTRIES + 1],
            deterministic: vec![],
            properties: None,
        };
        assert_eq!(
            WorkerBackendCapabilities::try_from(oversized.clone()),
            Err(WorkerBackendCapabilityError::Oversized)
        );
        let oversized_payload =
            postcard::to_stdvec(&oversized).expect("encode oversized wire fixture");
        assert!(postcard::from_bytes::<WorkerBackendCapabilities>(&oversized_payload).is_err());
    }

    #[test]
    fn primitive_support_is_versioned_closed_and_event_formats_are_absent() {
        let event = WorkerOperationSupport::for_event_v2(WorkerPrimitiveOperationV2::RecordEvent)
            .expect("record-event is an event primitive");
        assert_eq!(event.version(), WORKER_OPERATION_SUPPORT_VERSION);
        assert_eq!(event.category(), WorkerOperationCategory::Event);
        assert_eq!(event.role(), None);
        assert_eq!(event.dtype(), None);
        assert_eq!(event.layout(), None);

        assert_eq!(
            WorkerOperationSupport::for_tensor_v2(
                WorkerPrimitiveOperationV2::WaitEvent,
                WorkerTensorRoleV1::Output,
                WorkerDType::F32,
                WorkerLayout::Contiguous,
            ),
            Err(WorkerOperationSupportError::EventTensorSignature)
        );
        assert_eq!(
            WorkerOperationSupport::for_event_v2(WorkerPrimitiveOperationV2::Allocation),
            Err(WorkerOperationSupportError::MissingTensorSignature)
        );

        let unsupported_version = WorkerOperationSupportWire {
            version: WORKER_OPERATION_SUPPORT_VERSION + 1,
            operation: WorkerPrimitiveOperationV2::Allocation,
            role: Some(WorkerTensorRoleV1::Output),
            dtype: Some(WorkerDType::F32),
            layout: Some(WorkerLayout::Contiguous),
        };
        assert_eq!(
            WorkerOperationSupport::try_from(unsupported_version),
            Err(WorkerOperationSupportError::UnsupportedVersion(
                WORKER_OPERATION_SUPPORT_VERSION + 1
            ))
        );

        let missing_role = WorkerOperationSupportWire {
            version: WORKER_OPERATION_SUPPORT_VERSION,
            operation: WorkerPrimitiveOperationV2::Allocation,
            role: None,
            dtype: Some(WorkerDType::F32),
            layout: Some(WorkerLayout::Contiguous),
        };
        assert_eq!(
            WorkerOperationSupport::try_from(missing_role),
            Err(WorkerOperationSupportError::MissingTensorSignature)
        );

        let unknown_operation = r#"{
            "version": 2,
            "operation": "future_primitive",
            "role": "input",
            "dtype": "f32",
            "layout": "contiguous"
        }"#;
        assert!(serde_json::from_str::<WorkerOperationSupport>(unknown_operation).is_err());
        let unknown_field = r#"{
            "version": 2,
            "operation": "allocation",
            "role": "output",
            "dtype": "f32",
            "layout": "contiguous",
            "future": true
        }"#;
        assert!(serde_json::from_str::<WorkerOperationSupport>(unknown_field).is_err());
    }

    #[test]
    fn worker_primitive_operation_v2_wire_round_trip() {
        for operation in [
            WorkerPrimitiveOperationV2::Unary(WorkerUnaryOperationV1::Tangent),
            WorkerPrimitiveOperationV2::Unary(WorkerUnaryOperationV1::ArcTangent),
            WorkerPrimitiveOperationV2::Unary(WorkerUnaryOperationV1::ArcHyperbolicTangent),
            WorkerPrimitiveOperationV2::Binary(WorkerBinaryOperationV1::Atan2),
            WorkerPrimitiveOperationV2::Binary(WorkerBinaryOperationV1::LogAddExp),
            WorkerPrimitiveOperationV2::LinearAlgebra(
                WorkerLinearAlgebraOperationV1::MatrixMultiply,
            ),
            WorkerPrimitiveOperationV2::Gather,
            WorkerPrimitiveOperationV2::Scatter,
            WorkerPrimitiveOperationV2::MaskedSelect,
            WorkerPrimitiveOperationV2::Convolution,
            WorkerPrimitiveOperationV2::CustomKernel,
        ] {
            let encoded = postcard::to_stdvec(&operation).expect("primitive operation encodes");
            let decoded: WorkerPrimitiveOperationV2 =
                postcard::from_bytes(&encoded).expect("primitive operation decodes");
            assert_eq!(decoded, operation);
        }
    }

    #[test]
    fn worker_primitive_operation_postcard_discriminants_are_append_only() {
        let fixtures: &[(WorkerPrimitiveOperationV1, &[u8])] = &[
            (WorkerPrimitiveOperationV1::Allocation, &[0]),
            (WorkerPrimitiveOperationV1::Copy, &[1]),
            (WorkerPrimitiveOperationV1::Fill, &[2]),
            (
                WorkerPrimitiveOperationV1::Unary(WorkerUnaryOperationV1::Tangent),
                &[3, 19],
            ),
            (
                WorkerPrimitiveOperationV1::Binary(WorkerBinaryOperationV1::Atan2),
                &[4, 16],
            ),
            (
                WorkerPrimitiveOperationV1::BinaryScalar(WorkerBinaryOperationV1::LogAddExp),
                &[5, 17],
            ),
            (
                WorkerPrimitiveOperationV1::Reduction(
                    WorkerReductionOperationV1::StandardDeviation,
                ),
                &[6, 10],
            ),
            (WorkerPrimitiveOperationV1::Select, &[7]),
            (WorkerPrimitiveOperationV1::Narrow, &[8]),
            (
                WorkerPrimitiveOperationV1::Resize(WorkerResizeModeV1::Lanczos),
                &[9, 4],
            ),
            (WorkerPrimitiveOperationV1::RecordEvent, &[10]),
            (WorkerPrimitiveOperationV1::WaitEvent, &[11]),
        ];

        for (operation, expected) in fixtures {
            let encoded = postcard::to_stdvec(operation).expect("primitive operation encodes");
            assert_eq!(encoded, *expected, "wire bytes changed for {operation:?}");
            let decoded: WorkerPrimitiveOperationV1 =
                postcard::from_bytes(expected).expect("pinned primitive operation decodes");
            assert_eq!(decoded, *operation);
        }

        let linear_algebra = WorkerPrimitiveOperationV2::LinearAlgebra(
            WorkerLinearAlgebraOperationV1::MatrixMultiply,
        );
        assert_eq!(
            postcard::to_stdvec(&linear_algebra).expect("linear algebra operation encodes"),
            [12, 0]
        );
        for (operation, expected) in [
            (WorkerPrimitiveOperationV2::Gather, vec![13]),
            (WorkerPrimitiveOperationV2::Scatter, vec![14]),
            (WorkerPrimitiveOperationV2::MaskedSelect, vec![15]),
            (WorkerPrimitiveOperationV2::Convolution, vec![16]),
            (WorkerPrimitiveOperationV2::CustomKernel, vec![17]),
        ] {
            assert_eq!(
                postcard::to_stdvec(&operation).expect("appended V2 operation encodes"),
                expected
            );
        }
    }

    #[test]
    fn worker_native_device_properties_round_trip_with_only_wire_bounds() {
        let support = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F16,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let properties = WorkerNativeDeviceProperties::new_with_allocation_limit(
            "Cambricon MLU fixture",
            24 * 1024 * 1024,
            20 * 1024 * 1024,
            11,
            0,
            Some("Neuware 1.20".to_owned()),
            true,
        )
        .expect("bounded properties");
        let backend = WorkerBackendCapabilities::new_with_properties(
            DeviceKind::Mlu,
            2,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("bounded backend declaration");
        let message = envelope(WorkerMessage::HelloAck {
            accepted_backend: backend,
        });
        let frame = encode_worker_frame(&message).expect("properties encode in protocol 8");
        assert_eq!(decode_worker_frame(&frame), Ok(message));
        assert_eq!(properties.name(), "Cambricon MLU fixture");
        assert_eq!(properties.total_memory_bytes(), 24 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 20 * 1024 * 1024);
        assert_eq!(properties.major(), 11);
        assert_eq!(properties.minor(), 0);
        assert_eq!(properties.architecture(), Some("Neuware 1.20"));
        assert!(properties.has_fp16());

        assert_eq!(
            WorkerNativeDeviceProperties::new_with_allocation_limit(
                "fixture", 24, 0, 0, 0, None, false,
            ),
            Err(WorkerNativeDevicePropertiesError::InvalidMemoryLimit)
        );
        assert_eq!(
            WorkerNativeDeviceProperties::new_with_allocation_limit(
                "fixture", 24, 25, 0, 0, None, false,
            ),
            Err(WorkerNativeDevicePropertiesError::InvalidMemoryLimit)
        );

        assert_eq!(
            WorkerNativeDeviceProperties::new(
                "x".repeat(MAX_WORKER_DEVICE_PROPERTY_BYTES + 1),
                1,
                0,
                0,
                None,
                false,
            ),
            Err(WorkerNativeDevicePropertiesError::NameOversized)
        );
        let oversized_wire = WorkerNativeDevicePropertiesWire {
            name: "fixture".to_owned(),
            total_memory_bytes: 1,
            allocation_limit_bytes: 1,
            major: 0,
            minor: 0,
            architecture: Some("x".repeat(MAX_WORKER_DEVICE_PROPERTY_BYTES + 1)),
            has_fp16: false,
        };
        let payload = postcard::to_stdvec(&oversized_wire).expect("malformed fixture encodes");
        assert!(postcard::from_bytes::<WorkerNativeDeviceProperties>(&payload).is_err());
    }

    #[test]
    fn directml_native_device_properties_round_trip_through_the_generic_wire_dto() {
        let support = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F16,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let properties = WorkerNativeDeviceProperties::new_with_allocation_limit(
            "DirectML fixture adapter",
            24 * 1024 * 1024 * 1024,
            18 * 1024 * 1024 * 1024,
            1,
            13,
            Some("DXGI adapter LUID 0x0123456789abcdef".to_owned()),
            true,
        )
        .expect("bounded DirectML properties");
        let backend = WorkerBackendCapabilities::new_with_properties(
            DeviceKind::DirectMl,
            0,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("bounded DirectML backend declaration");
        let message = envelope(WorkerMessage::HelloAck {
            accepted_backend: backend,
        });
        let frame = encode_worker_frame(&message).expect("DirectML properties encode");
        assert_eq!(decode_worker_frame(&frame), Ok(message));
        assert_eq!(properties.name(), "DirectML fixture adapter");
        assert_eq!(properties.total_memory_bytes(), 24 * 1024 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 18 * 1024 * 1024 * 1024);
        assert_eq!(properties.major(), 1);
        assert_eq!(properties.minor(), 13);
        assert_eq!(
            properties.architecture(),
            Some("DXGI adapter LUID 0x0123456789abcdef")
        );
        assert!(properties.has_fp16());
    }

    #[test]
    fn npu_native_device_properties_and_effective_ceiling_round_trip_through_the_generic_wire_dto()
    {
        let support = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F16,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let properties = WorkerNativeDeviceProperties::new_with_allocation_limit(
            "Huawei Ascend fixture",
            32 * 1024 * 1024 * 1024,
            12 * 1024 * 1024 * 1024,
            8,
            0,
            Some("AscendCL 8.0.0".to_owned()),
            true,
        )
        .expect("bounded NPU properties");
        let backend = WorkerBackendCapabilities::new_with_properties(
            DeviceKind::Npu,
            2,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("bounded NPU backend declaration");
        let message = envelope(WorkerMessage::HelloAck {
            accepted_backend: backend,
        });
        let frame = encode_worker_frame(&message).expect("NPU properties encode");
        assert_eq!(decode_worker_frame(&frame), Ok(message));
        assert_eq!(properties.name(), "Huawei Ascend fixture");
        assert_eq!(properties.total_memory_bytes(), 32 * 1024 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 12 * 1024 * 1024 * 1024);
        assert_eq!(properties.major(), 8);
        assert_eq!(properties.minor(), 0);
        assert_eq!(properties.architecture(), Some("AscendCL 8.0.0"));
        assert!(properties.has_fp16());
    }

    #[test]
    fn xpu_native_device_properties_and_effective_ceiling_round_trip_through_the_generic_wire_dto()
    {
        let support = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F16,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let properties = WorkerNativeDeviceProperties::new_with_allocation_limit(
            "Intel XPU fixture",
            16 * 1024 * 1024 * 1024,
            6 * 1024 * 1024 * 1024,
            1,
            11,
            Some("Intel 0x8086:0x56a0; oneDNN 3.5.0".to_owned()),
            true,
        )
        .expect("bounded XPU properties");
        let backend = WorkerBackendCapabilities::new_with_properties(
            DeviceKind::Xpu,
            3,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("bounded XPU backend declaration");
        let message = envelope(WorkerMessage::HelloAck {
            accepted_backend: backend,
        });
        let frame = encode_worker_frame(&message).expect("XPU properties encode");
        assert_eq!(decode_worker_frame(&frame), Ok(message));
        assert_eq!(properties.name(), "Intel XPU fixture");
        assert_eq!(properties.total_memory_bytes(), 16 * 1024 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 6 * 1024 * 1024 * 1024);
        assert_eq!(properties.major(), 1);
        assert_eq!(properties.minor(), 11);
        assert_eq!(
            properties.architecture(),
            Some("Intel 0x8086:0x56a0; oneDNN 3.5.0")
        );
        assert!(properties.has_fp16());
    }

    #[test]
    fn cuda_native_device_properties_and_effective_ceiling_round_trip_through_the_generic_wire_dto()
    {
        let support = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F16,
            WorkerLayout::Contiguous,
        )
        .expect("allocation is a tensor primitive");
        let properties = WorkerNativeDeviceProperties::new_with_allocation_limit(
            "NVIDIA CUDA fixture",
            24 * 1024 * 1024 * 1024,
            18 * 1024 * 1024 * 1024,
            12,
            8,
            Some("CUDA driver 12080; NVRTC 12.8".to_owned()),
            true,
        )
        .expect("bounded CUDA properties");
        let backend = WorkerBackendCapabilities::new_with_properties(
            DeviceKind::Cuda,
            3,
            vec![support],
            vec![support],
            Some(properties.clone()),
        )
        .expect("bounded CUDA backend declaration");
        let message = envelope(WorkerMessage::HelloAck {
            accepted_backend: backend,
        });
        let frame = encode_worker_frame(&message).expect("CUDA properties encode");
        assert_eq!(decode_worker_frame(&frame), Ok(message));
        assert_eq!(properties.name(), "NVIDIA CUDA fixture");
        assert_eq!(properties.total_memory_bytes(), 24 * 1024 * 1024 * 1024);
        assert_eq!(properties.allocation_limit_bytes(), 18 * 1024 * 1024 * 1024);
        assert_eq!(properties.major(), 12);
        assert_eq!(properties.minor(), 8);
        assert_eq!(
            properties.architecture(),
            Some("CUDA driver 12080; NVRTC 12.8")
        );
        assert!(properties.has_fp16());
    }

    #[test]
    fn operation_support_schema_versions_reject_mixed_peers_before_use() {
        let legacy = WorkerOperationSupportV1::for_tensor(
            WorkerPrimitiveOperationV1::Allocation,
            WorkerTensorRoleV1::Output,
            WorkerDType::F32,
            WorkerLayout::Contiguous,
        )
        .expect("legacy allocation support is valid");
        let legacy_bytes = postcard::to_stdvec(&legacy).expect("legacy support serializes");
        assert_eq!(legacy_bytes, [1, 0, 1, 1, 1, 1, 1, 0]);
        assert_eq!(
            postcard::from_bytes::<WorkerOperationSupportV1>(&legacy_bytes),
            Ok(legacy)
        );
        assert!(postcard::from_bytes::<WorkerOperationSupport>(&legacy_bytes).is_err());

        let legacy_schema_with_current_operation = WorkerOperationSupportWire {
            version: LEGACY_WORKER_OPERATION_SUPPORT_VERSION,
            operation: WorkerPrimitiveOperationV2::Allocation,
            role: Some(WorkerTensorRoleV1::Output),
            dtype: Some(WorkerDType::F32),
            layout: Some(WorkerLayout::Contiguous),
        };
        assert_eq!(
            WorkerOperationSupport::try_from(legacy_schema_with_current_operation),
            Err(WorkerOperationSupportError::UnsupportedVersion(
                LEGACY_WORKER_OPERATION_SUPPORT_VERSION
            ))
        );

        let current = WorkerOperationSupport::for_tensor_v2(
            WorkerPrimitiveOperationV2::LinearAlgebra(
                WorkerLinearAlgebraOperationV1::MatrixMultiply,
            ),
            WorkerTensorRoleV1::Input,
            WorkerDType::F32,
            WorkerLayout::Contiguous,
        )
        .expect("current linear algebra support is valid");
        let current_bytes = postcard::to_stdvec(&current).expect("current support serializes");
        assert!(postcard::from_bytes::<WorkerOperationSupportV1>(&current_bytes).is_err());
    }
}
