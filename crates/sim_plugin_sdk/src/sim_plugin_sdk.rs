mod type_ids;

pub use type_ids::{
    BUILT_IN_TYPES, CanonicalTypeId, TypeEvolutionRule, TypeRegistration, TypeRegistry,
    TypeRegistryError, ValueFamily, ValueRepresentation,
};

#[cfg(any(feature = "signing-tooling", test))]
use ring::signature::{Ed25519KeyPair, KeyPair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sim_nodes::NodeDescriptor;
pub use sim_tensor::{DType, DeviceId, Layout, StreamId, TensorDescriptor, TensorError};
pub use sim_types::DeviceKind;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[cfg(any(feature = "signing-tooling", test))]
use zeroize::Zeroizing;

pub const COMPONENT_API_VERSION: ApiVersion = ApiVersion::new(1, 0, 0);
pub const COMPONENT_WORLD: &str = "zed:sim-plugin@1.0.0";
pub const PROVIDER_COMPONENT_WORLD: &str = "zed:sim-provider-plugin@1.0.0";
pub const PROVIDER_COMPONENT_API_VERSION_V2: ApiVersion = ApiVersion::new(2, 0, 0);
pub const PROVIDER_COMPONENT_WORLD_V2: &str = "zed:sim-provider-plugin@2.0.0";
pub const PROVIDER_MANIFEST_SCHEMA_VERSION_V2: u16 = 2;
pub const PROVIDER_STREAMING_API_FEATURE_V2: &str = "provider.streaming.v2";
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;
pub const MAX_MANIFEST_NODES: usize = 4_096;
pub const MAX_PORTS_PER_NODE: usize = 1_024;
pub const MAX_LEGACY_IDENTIFIERS: usize = 16_384;
pub const MAX_LEGACY_TRANSLATIONS_PER_MAPPING: usize = 1_024;
pub const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PROVIDER_RESULT_RECEIPTS: usize = 1_024;
pub const MAX_PROVIDER_RESULT_RECEIPT_BYTES: usize = 32 * 1024;
pub const MAX_PROVIDER_RESULT_RECEIPT_SET_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PROVIDER_STREAM_HEADERS: usize = 256;
pub const MAX_PROVIDER_STREAM_HEADER_NAME_BYTES: usize = 256;
pub const MAX_PROVIDER_STREAM_HEADER_VALUE_BYTES: usize = 8 * 1024;
pub const MAX_PROVIDER_STREAM_HEADER_BYTES: usize = 256 * 1024;
pub const MAX_PROVIDER_REQUEST_ENDPOINT_BYTES: usize = 2_048;
pub const MAX_PROVIDER_REQUEST_SECRET_ID_BYTES: usize = 1_024;
pub const MAX_PROVIDER_STREAM_BODY_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PROVIDER_ENCODED_VALUE_BYTES: u64 = MAX_MANIFEST_BYTES as u64;
pub const MAX_PROVIDER_STREAM_CHUNK_BYTES: usize = 1024 * 1024;
pub const MAX_PROVIDER_STREAM_NDJSON_LINE_BYTES: usize = 1024 * 1024;
pub const MAX_PROVIDER_STREAM_WAIT_MILLISECONDS: u64 = 60_000;
pub const MAX_PROVIDER_STREAM_PROGRESS_TOTAL: u64 = 1_000_000_000;
pub const MAX_PROVIDER_STREAM_PROGRESS_MESSAGE_BYTES: usize = 1_024;
pub const MAX_PROVIDER_STREAM_UPLOADS: u32 = 128;
pub const MAX_PROVIDER_STREAM_UPLOAD_BODY_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PROVIDER_STREAM_COST_REQUESTS: u32 = 64;
pub const MAX_PROVIDER_COST_REQUEST_BYTES: u64 = 512;
pub const MAX_PROVIDER_COST_RESPONSE_BYTES: u64 =
    1 + 8 + 4 + MAX_PROVIDER_RESULT_RECEIPT_BYTES as u64;
pub const PLUGIN_SIGNATURE_ALGORITHM: &str = "ed25519-v1";
#[cfg(any(feature = "signing-tooling", test))]
pub const ED25519_PRIVATE_KEY_SEED_BYTES: usize = 32;
pub const ED25519_PUBLIC_KEY_BYTES: usize = 32;
pub const ED25519_SIGNATURE_BYTES: usize = 64;

const PROVIDER_RESULT_RECEIPT_SET_DOMAIN: &[u8] = b"zed.sim.provider-result-receipt-set\0";
const PROVIDER_RESULT_RECEIPT_SET_VERSION: u16 = 1;
const PROVIDER_MANIFEST_V2_SIGNATURE_DOMAIN: &[u8] = b"zed.sim.provider-manifest.v2\0";
const PROVIDER_REQUEST_HEAD_V2_CANONICAL_DOMAIN: &str = "zed:sim-provider-request-head@2";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderResultReceiptSet {
    receipts: Vec<Vec<u8>>,
}

impl ProviderResultReceiptSet {
    pub fn new(receipts: Vec<Vec<u8>>) -> Result<Self, PluginContractError> {
        if receipts.is_empty() || receipts.len() > MAX_PROVIDER_RESULT_RECEIPTS {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        let mut unique = BTreeSet::new();
        let mut encoded_length = PROVIDER_RESULT_RECEIPT_SET_DOMAIN
            .len()
            .checked_add(size_of::<u16>() + size_of::<u32>())
            .ok_or(PluginContractError::InvalidProviderReceiptSet)?;
        for receipt in &receipts {
            if receipt.is_empty()
                || receipt.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES
                || !unique.insert(receipt.as_slice())
            {
                return Err(PluginContractError::InvalidProviderReceiptSet);
            }
            encoded_length = encoded_length
                .checked_add(size_of::<u32>())
                .and_then(|length| length.checked_add(receipt.len()))
                .ok_or(PluginContractError::InvalidProviderReceiptSet)?;
        }
        if encoded_length > MAX_PROVIDER_RESULT_RECEIPT_SET_BYTES {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        Ok(Self { receipts })
    }

    pub fn receipts(&self) -> &[Vec<u8>] {
        &self.receipts
    }

    pub fn into_receipts(self) -> Vec<Vec<u8>> {
        self.receipts
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, PluginContractError> {
        let receipt_count = u32::try_from(self.receipts.len())
            .map_err(|_| PluginContractError::InvalidProviderReceiptSet)?;
        let mut bytes = Vec::with_capacity(
            PROVIDER_RESULT_RECEIPT_SET_DOMAIN.len()
                + size_of::<u16>()
                + size_of::<u32>()
                + self
                    .receipts
                    .iter()
                    .map(|receipt| size_of::<u32>() + receipt.len())
                    .sum::<usize>(),
        );
        bytes.extend_from_slice(PROVIDER_RESULT_RECEIPT_SET_DOMAIN);
        bytes.extend_from_slice(&PROVIDER_RESULT_RECEIPT_SET_VERSION.to_le_bytes());
        bytes.extend_from_slice(&receipt_count.to_le_bytes());
        for receipt in &self.receipts {
            let receipt_length = u32::try_from(receipt.len())
                .map_err(|_| PluginContractError::InvalidProviderReceiptSet)?;
            bytes.extend_from_slice(&receipt_length.to_le_bytes());
            bytes.extend_from_slice(receipt);
        }
        if bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_SET_BYTES {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PluginContractError> {
        if bytes.is_empty() || bytes.len() > MAX_PROVIDER_RESULT_RECEIPT_SET_BYTES {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        let mut remaining = bytes;
        take_exact(&mut remaining, PROVIDER_RESULT_RECEIPT_SET_DOMAIN.len())
            .filter(|domain| *domain == PROVIDER_RESULT_RECEIPT_SET_DOMAIN)
            .ok_or(PluginContractError::InvalidProviderReceiptSet)?;
        let version = u16::from_le_bytes(
            take_exact(&mut remaining, size_of::<u16>())
                .and_then(|value| value.try_into().ok())
                .ok_or(PluginContractError::InvalidProviderReceiptSet)?,
        );
        if version != PROVIDER_RESULT_RECEIPT_SET_VERSION {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        let receipt_count = u32::from_le_bytes(
            take_exact(&mut remaining, size_of::<u32>())
                .and_then(|value| value.try_into().ok())
                .ok_or(PluginContractError::InvalidProviderReceiptSet)?,
        );
        let receipt_count = usize::try_from(receipt_count)
            .map_err(|_| PluginContractError::InvalidProviderReceiptSet)?;
        if receipt_count == 0 || receipt_count > MAX_PROVIDER_RESULT_RECEIPTS {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        let mut receipts = Vec::with_capacity(receipt_count);
        for _ in 0..receipt_count {
            let receipt_length = u32::from_le_bytes(
                take_exact(&mut remaining, size_of::<u32>())
                    .and_then(|value| value.try_into().ok())
                    .ok_or(PluginContractError::InvalidProviderReceiptSet)?,
            );
            let receipt_length = usize::try_from(receipt_length)
                .map_err(|_| PluginContractError::InvalidProviderReceiptSet)?;
            let receipt = take_exact(&mut remaining, receipt_length)
                .ok_or(PluginContractError::InvalidProviderReceiptSet)?
                .to_vec();
            receipts.push(receipt);
        }
        if !remaining.is_empty() {
            return Err(PluginContractError::InvalidProviderReceiptSet);
        }
        Self::new(receipts)
    }
}

fn take_exact<'a>(bytes: &mut &'a [u8], length: usize) -> Option<&'a [u8]> {
    if bytes.len() < length {
        return None;
    }
    let (value, remaining) = bytes.split_at(length);
    *bytes = remaining;
    Some(value)
}

#[cfg(any(feature = "signing-tooling", test))]
pub struct PluginSigningKey {
    key_id: String,
    seed: Zeroizing<[u8; ED25519_PRIVATE_KEY_SEED_BYTES]>,
}

#[cfg(any(feature = "signing-tooling", test))]
impl PluginSigningKey {
    pub fn new(
        key_id: impl Into<String>,
        seed: impl AsRef<[u8]>,
    ) -> Result<Self, PluginContractError> {
        let key_id = key_id.into();
        let seed: [u8; ED25519_PRIVATE_KEY_SEED_BYTES] = seed
            .as_ref()
            .try_into()
            .map_err(|_| PluginContractError::InvalidSigningKey)?;
        if !valid_authority_identifier(&key_id) {
            return Err(PluginContractError::InvalidSigningKey);
        }
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map_err(|_| PluginContractError::InvalidSigningKey)?;
        Ok(Self {
            key_id,
            seed: Zeroizing::new(seed),
        })
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn verification_key_bytes(
        &self,
    ) -> Result<[u8; ED25519_PUBLIC_KEY_BYTES], PluginContractError> {
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| PluginContractError::InvalidSigningKey)?;
        key_pair
            .public_key()
            .as_ref()
            .try_into()
            .map_err(|_| PluginContractError::InvalidSigningKey)
    }

    pub fn sign_manifest(&self, manifest: &PluginManifest) -> Result<String, PluginContractError> {
        if manifest.signature.algorithm != PLUGIN_SIGNATURE_ALGORITHM
            || manifest.signature.key_id != self.key_id
        {
            return Err(PluginContractError::InvalidSignatureMetadata);
        }
        manifest.validate(&TypeRegistry::built_in()?)?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| PluginContractError::InvalidSigningKey)?;
        Ok(encode_hex(
            key_pair.sign(&manifest.signing_payload()).as_ref(),
        ))
    }

    pub fn sign_provider_manifest_v2(
        &self,
        manifest: &ProviderPluginManifestV2,
    ) -> Result<String, PluginContractError> {
        if manifest.signature.algorithm != PLUGIN_SIGNATURE_ALGORITHM
            || manifest.signature.key_id != self.key_id
        {
            return Err(PluginContractError::InvalidSignatureMetadata);
        }
        manifest.validate(&TypeRegistry::built_in()?)?;
        let key_pair = Ed25519KeyPair::from_seed_unchecked(self.seed.as_ref())
            .map_err(|_| PluginContractError::InvalidSigningKey)?;
        Ok(encode_hex(
            key_pair.sign(&manifest.signing_payload()?).as_ref(),
        ))
    }
}

#[cfg(any(feature = "signing-tooling", test))]
impl fmt::Debug for PluginSigningKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PluginSigningKey")
            .field("key_id", &self.key_id)
            .field("seed", &"[REDACTED]")
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl ApiVersion {
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for ApiVersion {
    type Err = PluginContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut components = value.split('.');
        let major = parse_version_component(value, components.next())?;
        let minor = parse_version_component(value, components.next())?;
        let patch = parse_version_component(value, components.next())?;
        if components.next().is_some() {
            return Err(PluginContractError::InvalidVersion(value.to_owned()));
        }
        Ok(Self::new(major, minor, patch))
    }
}

fn parse_version_component(
    full_version: &str,
    component: Option<&str>,
) -> Result<u16, PluginContractError> {
    let component = component
        .filter(|component| !component.is_empty())
        .ok_or_else(|| PluginContractError::InvalidVersion(full_version.to_owned()))?;
    if component.len() > 1 && component.starts_with('0') {
        return Err(PluginContractError::InvalidVersion(full_version.to_owned()));
    }
    component
        .parse::<u16>()
        .map_err(|_| PluginContractError::InvalidVersion(full_version.to_owned()))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiRequirement {
    pub major: u16,
    pub minimum_minor: u16,
    pub maximum_minor: u16,
    pub required_features: Vec<String>,
}

impl ApiRequirement {
    pub fn negotiate(
        &self,
        host_version: ApiVersion,
        host_features: &BTreeSet<String>,
    ) -> Result<NegotiatedApi, PluginContractError> {
        if self.major != host_version.major
            || host_version.minor < self.minimum_minor
            || host_version.minor > self.maximum_minor
        {
            return Err(PluginContractError::UnsupportedApi {
                requested: format!(
                    "{}.{}..={}",
                    self.major, self.minimum_minor, self.maximum_minor
                ),
                host: host_version.to_string(),
            });
        }
        let mut features = BTreeSet::new();
        for feature in &self.required_features {
            if !valid_dotted_identifier(feature) || !host_features.contains(feature) {
                return Err(PluginContractError::MissingApiFeature(feature.clone()));
            }
            features.insert(feature.clone());
        }
        Ok(NegotiatedApi {
            version: host_version,
            features,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NegotiatedApi {
    pub version: ApiVersion,
    pub features: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PortDirection {
    Input,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PortCardinality {
    Singular,
    List,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PortPresence {
    Required,
    Optional,
    Hidden,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PortSerialization {
    Inline,
    Handle,
    ArtifactReference,
    OpaquePreserved,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<ScalarValue>),
    Record(Vec<(String, ScalarValue)>),
}

impl ScalarValue {
    pub fn abi_bytes(&self) -> Result<Vec<u8>, PluginContractError> {
        validate_scalar(self)
            .map_err(|()| PluginContractError::InvalidValue("invalid scalar value".to_owned()))?;
        serde_json::to_vec(self)
            .map_err(|error| PluginContractError::InvalidValue(error.to_string()))
    }

    pub fn from_abi_bytes(bytes: &[u8]) -> Result<Self, PluginContractError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(PluginContractError::InvalidValue(
                "scalar projection is too large".to_owned(),
            ));
        }
        let value: Self = serde_json::from_slice(bytes)
            .map_err(|error| PluginContractError::InvalidValue(error.to_string()))?;
        validate_scalar(&value)
            .map_err(|()| PluginContractError::InvalidValue("invalid scalar value".to_owned()))?;
        Ok(value)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginPort {
    pub id: String,
    pub name: String,
    pub direction: PortDirection,
    pub type_id: CanonicalTypeId,
    pub cardinality: PortCardinality,
    pub presence: PortPresence,
    pub hidden: bool,
    pub lazy: bool,
    pub default: Option<ScalarValue>,
    pub serialization: PortSerialization,
    pub accepted_legacy_names: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeterminismPolicy {
    Deterministic,
    Seeded,
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CachePolicy {
    InputIdentity,
    Never,
    PluginKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EffectPolicy {
    Pure,
    Transactional,
    Provider,
}

pub const PROVIDER_BINDING_SCHEMA_VERSION: u16 = 1;
pub const PROVIDER_BINDING_API_FEATURE: &str = "provider.bindings.v1";
const PROVIDER_BINDING_CANONICAL_DOMAIN: &str = "zed:sim-provider-binding-set@1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderBindingClaim {
    pub feature_id: String,
    pub node_id: String,
    pub contract_sha256: String,
    pub transport_schema: CanonicalTypeId,
    pub materializer_schema: CanonicalTypeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderBindingSet {
    pub schema_version: u16,
    pub implementation_namespace: String,
    pub bindings_sha256: String,
    pub bindings: Vec<ProviderBindingClaim>,
}

impl ProviderBindingSet {
    pub fn canonical_binding_bytes(&self) -> Result<Vec<u8>, PluginContractError> {
        let mut writer = CanonicalWriter::with_limit(MAX_MANIFEST_BYTES);
        self.write_canonical(&mut writer);
        if writer.overflowed() {
            return Err(PluginContractError::ManifestTooLarge);
        }
        Ok(writer.finish())
    }

    pub fn canonical_bindings_sha256(&self) -> Result<String, PluginContractError> {
        Ok(format!(
            "{:x}",
            Sha256::digest(self.canonical_binding_bytes()?)
        ))
    }

    fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.string(PROVIDER_BINDING_CANONICAL_DOMAIN);
        writer.u16(self.schema_version);
        writer.string(&self.implementation_namespace);
        writer.usize(self.bindings.len());
        for binding in &self.bindings {
            writer.string(&binding.feature_id);
            writer.string(&binding.node_id);
            writer.string(&binding.contract_sha256);
            writer.string(&binding.transport_schema.to_string());
            writer.string(&binding.materializer_schema.to_string());
        }
    }

    fn validate(
        &self,
        implementation_namespace: &str,
        required_features: &[String],
        nodes: &[PluginNode],
    ) -> Result<(), PluginContractError> {
        if self.schema_version != PROVIDER_BINDING_SCHEMA_VERSION
            || self.implementation_namespace != implementation_namespace
            || !required_features
                .iter()
                .any(|feature| feature == PROVIDER_BINDING_API_FEATURE)
            || self.bindings.is_empty()
            || self.bindings.len() > MAX_MANIFEST_NODES
        {
            return Err(PluginContractError::InvalidProviderBindingSet);
        }
        validate_sha256(&self.bindings_sha256)?;
        let mut feature_ids = BTreeSet::new();
        let mut previous_node_id: Option<&str> = None;
        for binding in &self.bindings {
            if !valid_feature_id(&binding.feature_id)
                || !valid_dotted_identifier(&binding.node_id)
                || !feature_ids.insert(binding.feature_id.as_str())
                || previous_node_id.is_some_and(|previous| previous >= binding.node_id.as_str())
            {
                return Err(PluginContractError::InvalidProviderBindingSet);
            }
            validate_sha256(&binding.contract_sha256)?;
            let Some(node) = nodes.iter().find(|node| node.id == binding.node_id) else {
                return Err(PluginContractError::InvalidProviderBindingSet);
            };
            if node.determinism != DeterminismPolicy::External
                || node.effects != EffectPolicy::Provider
                || node.cache != CachePolicy::Never
            {
                return Err(PluginContractError::InvalidProviderBindingSet);
            }
            previous_node_id = Some(binding.node_id.as_str());
        }
        if self.bindings_sha256 != self.canonical_bindings_sha256()? {
            return Err(PluginContractError::InvalidProviderBindingSet);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ProviderHttpMethodV2 {
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
pub struct ProviderHeaderV2 {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStreamHandleV2 {
    pub invocation: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInvocationContextV2 {
    pub invocation: u64,
    pub generation: u32,
}

impl ProviderInvocationContextV2 {
    pub fn validate(self) -> Result<(), ProviderStreamingContractError> {
        if self.invocation == 0 || self.generation == 0 {
            return Err(ProviderStreamingContractError::InvalidHandle);
        }
        Ok(())
    }
}

impl ProviderStreamHandleV2 {
    pub fn validate(self) -> Result<(), ProviderStreamingContractError> {
        if self.invocation == 0 || self.slot == 0 || self.generation == 0 {
            return Err(ProviderStreamingContractError::InvalidHandle);
        }
        Ok(())
    }
}

fn require_matching_stream_handle(
    actual: ProviderStreamHandleV2,
    expected: ProviderStreamHandleV2,
) -> Result<(), ProviderStreamingContractError> {
    actual.validate()?;
    if actual.invocation != expected.invocation || actual.slot != expected.slot {
        return Err(ProviderStreamingContractError::ForeignHandle);
    }
    if actual.generation != expected.generation {
        return Err(ProviderStreamingContractError::RevokedHandle);
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderStreamingContractV2 {
    pub methods: Vec<ProviderHttpMethodV2>,
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

impl ProviderStreamingContractV2 {
    pub fn validate(&self) -> Result<(), ProviderStreamingContractError> {
        if self.methods.is_empty()
            || self.methods.len() > 7
            || self
                .methods
                .windows(2)
                .any(|methods| methods[0] >= methods[1])
            || self.maximum_headers == 0
            || usize::from(self.maximum_headers) > MAX_PROVIDER_STREAM_HEADERS
            || self.maximum_header_bytes == 0
            || usize::try_from(self.maximum_header_bytes)
                .map_or(true, |value| value > MAX_PROVIDER_STREAM_HEADER_BYTES)
            || self.maximum_request_body_bytes == 0
            || self.maximum_request_body_bytes > MAX_PROVIDER_STREAM_BODY_BYTES
            || self.maximum_response_body_bytes == 0
            || self.maximum_response_body_bytes > MAX_PROVIDER_STREAM_BODY_BYTES
            || self.maximum_chunk_bytes == 0
            || usize::try_from(self.maximum_chunk_bytes)
                .map_or(true, |value| value > MAX_PROVIDER_STREAM_CHUNK_BYTES)
            || self.maximum_ndjson_line_bytes == 0
            || self.maximum_ndjson_line_bytes > self.maximum_chunk_bytes
            || usize::try_from(self.maximum_ndjson_line_bytes)
                .map_or(true, |value| value > MAX_PROVIDER_STREAM_NDJSON_LINE_BYTES)
            || self.maximum_wait_milliseconds == 0
            || self.maximum_wait_milliseconds > MAX_PROVIDER_STREAM_WAIT_MILLISECONDS
            || self.maximum_progress_total == 0
            || self.maximum_progress_total > MAX_PROVIDER_STREAM_PROGRESS_TOTAL
            || self.maximum_uploads > MAX_PROVIDER_STREAM_UPLOADS
            || self.maximum_upload_body_bytes > MAX_PROVIDER_STREAM_UPLOAD_BODY_BYTES
            || self.maximum_cost_requests > MAX_PROVIDER_STREAM_COST_REQUESTS
            || self.uploads != (self.maximum_uploads != 0 && self.maximum_upload_body_bytes != 0)
            || (!self.uploads && (self.maximum_uploads != 0 || self.maximum_upload_body_bytes != 0))
            || self.cost_requests != (self.maximum_cost_requests != 0)
        {
            return Err(ProviderStreamingContractError::InvalidContract);
        }
        Ok(())
    }

    fn admits_method(&self, method: ProviderHttpMethodV2) -> bool {
        self.methods.binary_search(&method).is_ok()
    }

    fn write_canonical(&self, writer: &mut CanonicalWriter) {
        writer.usize(self.methods.len());
        for method in &self.methods {
            writer.byte(*method as u8);
        }
        writer.u16(self.maximum_headers);
        writer.u32(self.maximum_header_bytes);
        writer.u64(self.maximum_request_body_bytes);
        writer.u64(self.maximum_response_body_bytes);
        writer.u32(self.maximum_chunk_bytes);
        writer.u32(self.maximum_ndjson_line_bytes);
        writer.u64(self.maximum_wait_milliseconds);
        writer.u32(self.maximum_uploads);
        writer.u64(self.maximum_upload_body_bytes);
        writer.u32(self.maximum_cost_requests);
        writer.u64(self.maximum_progress_total);
        writer.byte(u8::from(self.uploads));
        writer.byte(u8::from(self.cost_requests));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRequestHeadV2 {
    pub endpoint: String,
    pub secret_id: Option<String>,
    pub method: ProviderHttpMethodV2,
    pub headers: Vec<ProviderHeaderV2>,
    pub declared_body_bytes: Option<u64>,
}

impl ProviderRequestHeadV2 {
    pub fn validate_for_contract(
        &self,
        contract: &ProviderStreamingContractV2,
    ) -> Result<(), ProviderStreamingContractError> {
        if !valid_provider_request_authority(&self.endpoint, MAX_PROVIDER_REQUEST_ENDPOINT_BYTES)
            || self.secret_id.as_deref().is_some_and(|secret_id| {
                !valid_provider_request_authority(secret_id, MAX_PROVIDER_REQUEST_SECRET_ID_BYTES)
            })
        {
            return Err(ProviderStreamingContractError::InvalidRequestAuthority);
        }
        if !contract.admits_method(self.method) {
            return Err(ProviderStreamingContractError::InvalidMethod);
        }
        validate_provider_headers(&self.headers, contract)?;
        if self
            .declared_body_bytes
            .is_some_and(|bytes| bytes > contract.maximum_request_body_bytes)
            || (self.method == ProviderHttpMethodV2::Head
                && self.declared_body_bytes.is_some_and(|bytes| bytes != 0))
        {
            return Err(ProviderStreamingContractError::BodyLimit);
        }
        Ok(())
    }

    pub fn canonical_bytes(
        &self,
        contract: &ProviderStreamingContractV2,
    ) -> Result<Vec<u8>, ProviderStreamingContractError> {
        contract.validate()?;
        self.validate_for_contract(contract)?;
        let mut writer = CanonicalWriter::with_limit(MAX_MANIFEST_BYTES);
        writer.string(PROVIDER_REQUEST_HEAD_V2_CANONICAL_DOMAIN);
        writer.string(&self.endpoint);
        writer.optional_string(self.secret_id.as_deref());
        writer.byte(self.method as u8);
        writer.usize(self.headers.len());
        for header in &self.headers {
            writer.string(&header.name);
            writer.string(&header.value);
        }
        match self.declared_body_bytes {
            Some(bytes) => {
                writer.byte(1);
                writer.u64(bytes);
            }
            None => writer.byte(0),
        }
        if writer.overflowed() {
            return Err(ProviderStreamingContractError::InvalidContract);
        }
        Ok(writer.finish())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderRequestChunkV2 {
    pub handle: ProviderStreamHandleV2,
    pub sequence: u64,
    pub bytes: Vec<u8>,
    pub end: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResponseHeadV2 {
    pub status: u16,
    pub headers: Vec<ProviderHeaderV2>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum ProviderResponseChunkV2 {
    Binary(Vec<u8>),
    Text(String),
    NdjsonLine(String),
}

impl ProviderResponseChunkV2 {
    fn byte_length(&self) -> usize {
        match self {
            Self::Binary(bytes) => bytes.len(),
            Self::Text(text) | Self::NdjsonLine(text) => text.len(),
        }
    }

    fn validate(
        &self,
        contract: &ProviderStreamingContractV2,
    ) -> Result<(), ProviderStreamingContractError> {
        let byte_length = self.byte_length();
        if byte_length == 0
            || u32::try_from(byte_length).map_or(true, |value| value > contract.maximum_chunk_bytes)
        {
            return Err(ProviderStreamingContractError::ChunkLimit);
        }
        if let Self::NdjsonLine(line) = self {
            if line.contains(['\n', '\r'])
                || u32::try_from(byte_length)
                    .map_or(true, |value| value > contract.maximum_ndjson_line_bytes)
                || serde_json::from_str::<serde_json::Value>(line).is_err()
            {
                return Err(ProviderStreamingContractError::InvalidNdjsonLine);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWaitRequestV2 {
    pub handle: ProviderStreamHandleV2,
    pub after_sequence: Option<u64>,
    pub timeout_milliseconds: u64,
}

impl ProviderWaitRequestV2 {
    fn validate_for_handle(
        &self,
        contract: &ProviderStreamingContractV2,
        handle: ProviderStreamHandleV2,
    ) -> Result<(), ProviderStreamingContractError> {
        require_matching_stream_handle(self.handle, handle)?;
        if self.timeout_milliseconds == 0
            || self.timeout_milliseconds > contract.maximum_wait_milliseconds
        {
            return Err(ProviderStreamingContractError::WaitLimit);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum ProviderWaitOutcomeV2 {
    Frame(ProviderResponseFrameV2),
    TimedOut,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderUploadRequestV2 {
    pub handle: ProviderStreamHandleV2,
    pub port_id: String,
    pub media_type: String,
    pub byte_length: u64,
    pub content_sha256: String,
}

impl ProviderUploadRequestV2 {
    fn validate_for_handle(
        &self,
        contract: &ProviderStreamingContractV2,
        handle: ProviderStreamHandleV2,
    ) -> Result<(), ProviderStreamingContractError> {
        require_matching_stream_handle(self.handle, handle)?;
        if !contract.uploads
            || !valid_dotted_identifier(&self.port_id)
            || !valid_media_type(&self.media_type)
            || self.byte_length == 0
            || self.byte_length > contract.maximum_request_body_bytes
            || validate_sha256(&self.content_sha256).is_err()
        {
            return Err(ProviderStreamingContractError::InvalidUpload);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostRequestV2 {
    pub handle: ProviderStreamHandleV2,
    pub operation: String,
    pub currency: String,
    pub maximum_microunits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCostResponseV2 {
    pub accepted: bool,
    pub approved_microunits: u64,
    pub receipt: Vec<u8>,
}

impl ProviderCostResponseV2 {
    fn validate_for_request(
        &self,
        request: &ProviderCostRequestV2,
    ) -> Result<(), ProviderStreamingContractError> {
        let accepted = self.accepted
            && self.approved_microunits != 0
            && self.approved_microunits <= request.maximum_microunits
            && !self.receipt.is_empty()
            && self.receipt.len() <= MAX_PROVIDER_RESULT_RECEIPT_BYTES;
        let denied = !self.accepted && self.approved_microunits == 0 && self.receipt.is_empty();
        if !accepted && !denied {
            return Err(ProviderStreamingContractError::InvalidCostRequest);
        }
        Ok(())
    }
}

impl ProviderCostRequestV2 {
    fn validate_for_handle(
        &self,
        contract: &ProviderStreamingContractV2,
        handle: ProviderStreamHandleV2,
    ) -> Result<(), ProviderStreamingContractError> {
        require_matching_stream_handle(self.handle, handle)?;
        if !contract.cost_requests
            || !valid_dotted_identifier(&self.operation)
            || self.currency.len() != 3
            || !self.currency.bytes().all(|byte| byte.is_ascii_uppercase())
            || self.maximum_microunits == 0
        {
            return Err(ProviderStreamingContractError::InvalidCostRequest);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderProgressV2 {
    pub handle: ProviderStreamHandleV2,
    pub sequence: u64,
    pub completed: u64,
    pub total: u64,
    pub message: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ProviderStreamTerminalV2 {
    Completed { receipt: Vec<u8> },
    Failed { code: String, message: String },
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "kebab-case",
    deny_unknown_fields
)]
pub enum ProviderResponseFrameEventV2 {
    Head(ProviderResponseHeadV2),
    Chunk(ProviderResponseChunkV2),
    Terminal(ProviderStreamTerminalV2),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResponseFrameV2 {
    pub handle: ProviderStreamHandleV2,
    pub sequence: u64,
    pub event: ProviderResponseFrameEventV2,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderEncodedValueV2 {
    pub type_id: CanonicalTypeId,
    pub family: ValueFamily,
    pub abi_bytes: Vec<u8>,
}

impl ProviderEncodedValueV2 {
    fn validate(&self, registry: &TypeRegistry) -> Result<(), ProviderStreamingContractError> {
        registry
            .require_family(&self.type_id, self.family)
            .map_err(|_| ProviderStreamingContractError::InvalidInvocationResult)?;
        if self.abi_bytes.is_empty()
            || u64::try_from(self.abi_bytes.len())
                .map_or(true, |length| length > MAX_PROVIDER_ENCODED_VALUE_BYTES)
        {
            return Err(ProviderStreamingContractError::InvalidInvocationResult);
        }
        let value = PluginValue::from_abi_bytes(&self.abi_bytes, registry)
            .map_err(|_| ProviderStreamingContractError::InvalidInvocationResult)?;
        if value.type_id() != &self.type_id || value.family() != self.family {
            return Err(ProviderStreamingContractError::InvalidInvocationResult);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderMaterializedOutputV2 {
    pub port_id: String,
    pub value: ProviderEncodedValueV2,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderInvocationResultV2 {
    pub outputs: Vec<ProviderMaterializedOutputV2>,
    pub receipt: Vec<u8>,
}

impl ProviderInvocationResultV2 {
    pub fn validate(&self, registry: &TypeRegistry) -> Result<(), ProviderStreamingContractError> {
        if self.outputs.len() > MAX_PORTS_PER_NODE
            || self.receipt.is_empty()
            || self.receipt.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES
        {
            return Err(ProviderStreamingContractError::InvalidInvocationResult);
        }
        let mut previous_port_id: Option<&str> = None;
        let mut aggregate_bytes = 0u64;
        for output in &self.outputs {
            if !valid_dotted_identifier(&output.port_id)
                || previous_port_id.is_some_and(|previous| previous >= output.port_id.as_str())
            {
                return Err(ProviderStreamingContractError::InvalidInvocationResult);
            }
            output.value.validate(registry)?;
            aggregate_bytes = aggregate_bytes
                .checked_add(
                    u64::try_from(output.value.abi_bytes.len())
                        .map_err(|_| ProviderStreamingContractError::InvalidInvocationResult)?,
                )
                .ok_or(ProviderStreamingContractError::InvalidInvocationResult)?;
            if aggregate_bytes > MAX_PROVIDER_STREAM_BODY_BYTES {
                return Err(ProviderStreamingContractError::InvalidInvocationResult);
            }
            previous_port_id = Some(output.port_id.as_str());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProviderStreamingContractError {
    InvalidContract,
    InvalidHandle,
    ForeignHandle,
    RevokedHandle,
    InvalidMethod,
    InvalidHeaders,
    BodyLimit,
    ChunkLimit,
    InvalidNdjsonLine,
    InvalidSequence,
    InvalidOrder,
    WaitLimit,
    InvalidUpload,
    InvalidCostRequest,
    InvalidProgress,
    InvalidTerminal,
    InvalidInvocationResult,
    InvalidRequestAuthority,
}

impl fmt::Display for ProviderStreamingContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidContract => "invalid provider streaming contract",
            Self::InvalidHandle => "invalid provider stream handle",
            Self::ForeignHandle => "provider stream handle belongs to another invocation",
            Self::RevokedHandle => "provider stream handle was revoked",
            Self::InvalidMethod => "provider HTTP method is not declared",
            Self::InvalidHeaders => "invalid or oversized provider headers",
            Self::BodyLimit => "provider body exceeds its cumulative bound",
            Self::ChunkLimit => "provider chunk exceeds its bound",
            Self::InvalidNdjsonLine => "invalid provider NDJSON line",
            Self::InvalidSequence => "provider stream sequence is not monotonic",
            Self::InvalidOrder => "provider stream event is out of order",
            Self::WaitLimit => "provider wait exceeds its bound",
            Self::InvalidUpload => "invalid provider upload request",
            Self::InvalidCostRequest => "invalid provider cost request",
            Self::InvalidProgress => "invalid provider progress",
            Self::InvalidTerminal => "invalid provider terminal state",
            Self::InvalidInvocationResult => "invalid provider invocation result",
            Self::InvalidRequestAuthority => "invalid provider request authority",
        })
    }
}

impl Error for ProviderStreamingContractError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[repr(u8)]
pub enum ProviderStreamErrorV2 {
    Cancelled,
    TimedOut,
    HostFailure,
    InvalidContract,
    InvalidHandle,
    ForeignHandle,
    RevokedHandle,
    InvalidMethod,
    InvalidHeaders,
    BodyLimit,
    ChunkLimit,
    InvalidNdjsonLine,
    InvalidSequence,
    InvalidOrder,
    WaitLimit,
    InvalidUpload,
    InvalidCostRequest,
    InvalidProgress,
    InvalidTerminal,
    InvalidInvocationResult,
    InvalidRequestAuthority,
}

impl From<&ProviderStreamingContractError> for ProviderStreamErrorV2 {
    fn from(error: &ProviderStreamingContractError) -> Self {
        match error {
            ProviderStreamingContractError::InvalidContract => Self::InvalidContract,
            ProviderStreamingContractError::InvalidHandle => Self::InvalidHandle,
            ProviderStreamingContractError::ForeignHandle => Self::ForeignHandle,
            ProviderStreamingContractError::RevokedHandle => Self::RevokedHandle,
            ProviderStreamingContractError::InvalidMethod => Self::InvalidMethod,
            ProviderStreamingContractError::InvalidHeaders => Self::InvalidHeaders,
            ProviderStreamingContractError::BodyLimit => Self::BodyLimit,
            ProviderStreamingContractError::ChunkLimit => Self::ChunkLimit,
            ProviderStreamingContractError::InvalidNdjsonLine => Self::InvalidNdjsonLine,
            ProviderStreamingContractError::InvalidSequence => Self::InvalidSequence,
            ProviderStreamingContractError::InvalidOrder => Self::InvalidOrder,
            ProviderStreamingContractError::WaitLimit => Self::WaitLimit,
            ProviderStreamingContractError::InvalidUpload => Self::InvalidUpload,
            ProviderStreamingContractError::InvalidCostRequest => Self::InvalidCostRequest,
            ProviderStreamingContractError::InvalidProgress => Self::InvalidProgress,
            ProviderStreamingContractError::InvalidTerminal => Self::InvalidTerminal,
            ProviderStreamingContractError::InvalidInvocationResult => {
                Self::InvalidInvocationResult
            }
            ProviderStreamingContractError::InvalidRequestAuthority => {
                Self::InvalidRequestAuthority
            }
        }
    }
}

#[derive(Debug)]
pub struct ProviderStreamValidatorV2 {
    contract: ProviderStreamingContractV2,
    handle: ProviderStreamHandleV2,
    request_method: ProviderHttpMethodV2,
    declared_body_bytes: Option<u64>,
    request_bytes: u64,
    response_bytes: u64,
    next_request_sequence: u64,
    next_response_sequence: u64,
    last_response_sequence: Option<u64>,
    next_progress_sequence: u64,
    request_finished: bool,
    response_started: bool,
    response_status: Option<u16>,
    terminal: bool,
    uploads: u32,
    upload_bytes: u64,
    upload_handles: BTreeSet<ProviderStreamHandleV2>,
    upload_completions: Vec<Arc<AtomicBool>>,
    cost_requests: u32,
    progress_completed: u64,
    progress_total: Option<u64>,
    revoked: Arc<AtomicBool>,
}

impl ProviderStreamValidatorV2 {
    pub fn checked(
        contract: ProviderStreamingContractV2,
        context: ProviderInvocationContextV2,
        host_handle: ProviderStreamHandleV2,
        request: &ProviderRequestHeadV2,
    ) -> Result<Self, ProviderStreamingContractError> {
        contract.validate()?;
        context.validate()?;
        host_handle.validate()?;
        if context.invocation != host_handle.invocation {
            return Err(ProviderStreamingContractError::ForeignHandle);
        }
        if context.generation != host_handle.generation {
            return Err(ProviderStreamingContractError::RevokedHandle);
        }
        request.validate_for_contract(&contract)?;
        Ok(Self {
            contract,
            handle: host_handle,
            request_method: request.method,
            declared_body_bytes: request.declared_body_bytes,
            request_bytes: 0,
            response_bytes: 0,
            next_request_sequence: 0,
            next_response_sequence: 0,
            last_response_sequence: None,
            next_progress_sequence: 0,
            request_finished: false,
            response_started: false,
            response_status: None,
            terminal: false,
            uploads: 0,
            upload_bytes: 0,
            upload_handles: BTreeSet::new(),
            upload_completions: Vec::new(),
            cost_requests: 0,
            progress_completed: 0,
            progress_total: None,
            revoked: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn retry(
        &mut self,
        host_handle: ProviderStreamHandleV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(self.handle)?;
        host_handle.validate()?;
        if host_handle.invocation != self.handle.invocation || host_handle.slot == self.handle.slot
        {
            return Err(ProviderStreamingContractError::ForeignHandle);
        }
        if host_handle.generation != self.handle.generation {
            return Err(ProviderStreamingContractError::RevokedHandle);
        }
        if !self.request_finished
            || !self.response_started
            || self.response_bytes != 0
            || self.last_response_sequence != Some(0)
            || !self
                .response_status
                .is_some_and(|status| matches!(status, 408 | 429 | 500 | 502 | 503 | 504))
            || self
                .upload_completions
                .iter()
                .any(|completed| !completed.load(Ordering::Acquire))
        {
            return Err(ProviderStreamingContractError::InvalidOrder);
        }
        self.revoked.store(true, Ordering::Release);
        self.handle = host_handle;
        self.request_bytes = 0;
        self.response_bytes = 0;
        self.next_request_sequence = 0;
        self.next_response_sequence = 0;
        self.last_response_sequence = None;
        self.request_finished = false;
        self.response_started = false;
        self.response_status = None;
        self.terminal = false;
        self.upload_handles.clear();
        self.upload_completions.clear();
        self.revoked = Arc::new(AtomicBool::new(false));
        Ok(())
    }

    pub fn write_request_chunk(
        &mut self,
        chunk: &ProviderRequestChunkV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(chunk.handle)?;
        if self.request_finished || self.response_started {
            return Err(ProviderStreamingContractError::InvalidOrder);
        }
        if chunk.sequence != self.next_request_sequence
            || chunk.bytes.len() > MAX_PROVIDER_STREAM_CHUNK_BYTES
            || (chunk.bytes.is_empty() && !chunk.end)
        {
            return Err(if chunk.sequence != self.next_request_sequence {
                ProviderStreamingContractError::InvalidSequence
            } else {
                ProviderStreamingContractError::ChunkLimit
            });
        }
        if u32::try_from(chunk.bytes.len())
            .map_or(true, |length| length > self.contract.maximum_chunk_bytes)
        {
            return Err(ProviderStreamingContractError::ChunkLimit);
        }
        let request_bytes = self
            .request_bytes
            .checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| ProviderStreamingContractError::BodyLimit)?,
            )
            .ok_or(ProviderStreamingContractError::BodyLimit)?;
        if request_bytes > self.contract.maximum_request_body_bytes
            || (self.request_method == ProviderHttpMethodV2::Head && request_bytes != 0)
            || self
                .declared_body_bytes
                .is_some_and(|declared| request_bytes > declared)
            || (chunk.end
                && self
                    .declared_body_bytes
                    .is_some_and(|declared| request_bytes != declared))
        {
            return Err(ProviderStreamingContractError::BodyLimit);
        }
        let next_sequence = chunk
            .sequence
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidSequence)?;
        self.request_bytes = request_bytes;
        self.next_request_sequence = next_sequence;
        self.request_finished = chunk.end;
        Ok(())
    }

    pub fn accept_wait(
        &mut self,
        request: &ProviderWaitRequestV2,
        outcome: ProviderWaitOutcomeV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(request.handle)?;
        request.validate_for_handle(&self.contract, self.handle)?;
        if request.after_sequence != self.last_response_sequence {
            return Err(ProviderStreamingContractError::InvalidSequence);
        }
        match outcome {
            ProviderWaitOutcomeV2::Frame(frame) => self.accept_response_frame(&frame)?,
            ProviderWaitOutcomeV2::Cancelled => {
                self.terminal = true;
                self.revoked.store(true, Ordering::Release);
            }
            ProviderWaitOutcomeV2::TimedOut => {}
        }
        Ok(())
    }

    pub fn start_upload(
        &mut self,
        request: &ProviderUploadRequestV2,
        upload_handle: ProviderStreamHandleV2,
    ) -> Result<ProviderUploadValidatorV2, ProviderStreamingContractError> {
        self.require_active_handle(request.handle)?;
        request.validate_for_handle(&self.contract, self.handle)?;
        upload_handle.validate()?;
        if upload_handle.invocation != self.handle.invocation
            || upload_handle.slot == self.handle.slot
            || self.upload_handles.contains(&upload_handle)
        {
            return Err(ProviderStreamingContractError::ForeignHandle);
        }
        if upload_handle.generation != self.handle.generation {
            return Err(ProviderStreamingContractError::RevokedHandle);
        }
        let uploads = self
            .uploads
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidUpload)?;
        if uploads > self.contract.maximum_uploads {
            return Err(ProviderStreamingContractError::InvalidUpload);
        }
        let upload_bytes = self
            .upload_bytes
            .checked_add(request.byte_length)
            .ok_or(ProviderStreamingContractError::InvalidUpload)?;
        if upload_bytes > self.contract.maximum_upload_body_bytes {
            return Err(ProviderStreamingContractError::InvalidUpload);
        }
        self.uploads = uploads;
        self.upload_bytes = upload_bytes;
        self.upload_handles.insert(upload_handle);
        let completed = Arc::new(AtomicBool::new(false));
        self.upload_completions.push(completed.clone());
        Ok(ProviderUploadValidatorV2::new(
            upload_handle,
            request.byte_length,
            request.content_sha256.clone(),
            self.contract.maximum_chunk_bytes,
            self.revoked.clone(),
            completed,
        ))
    }

    pub fn accept_cost_request(
        &mut self,
        request: &ProviderCostRequestV2,
        response: &ProviderCostResponseV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(request.handle)?;
        request.validate_for_handle(&self.contract, self.handle)?;
        response.validate_for_request(request)?;
        let cost_requests = self
            .cost_requests
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidCostRequest)?;
        if cost_requests > self.contract.maximum_cost_requests {
            return Err(ProviderStreamingContractError::InvalidCostRequest);
        }
        self.cost_requests = cost_requests;
        Ok(())
    }

    pub fn accept_progress(
        &mut self,
        progress: &ProviderProgressV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(progress.handle)?;
        if progress.sequence != self.next_progress_sequence
            || progress.total == 0
            || progress.total > self.contract.maximum_progress_total
            || progress.completed < self.progress_completed
            || progress.completed > progress.total
            || self
                .progress_total
                .is_some_and(|total| total != progress.total)
            || progress.message.as_ref().is_some_and(|message| {
                message.is_empty() || message.len() > MAX_PROVIDER_STREAM_PROGRESS_MESSAGE_BYTES
            })
        {
            return Err(ProviderStreamingContractError::InvalidProgress);
        }
        let next_sequence = progress
            .sequence
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidProgress)?;
        self.next_progress_sequence = next_sequence;
        self.progress_completed = progress.completed;
        self.progress_total = Some(progress.total);
        Ok(())
    }

    fn accept_response_frame(
        &mut self,
        frame: &ProviderResponseFrameV2,
    ) -> Result<(), ProviderStreamingContractError> {
        self.require_active_handle(frame.handle)?;
        if frame.sequence != self.next_response_sequence {
            return Err(ProviderStreamingContractError::InvalidSequence);
        }
        match &frame.event {
            ProviderResponseFrameEventV2::Head(head) => {
                if !self.request_finished
                    || self.response_started
                    || !(200..=599).contains(&head.status)
                {
                    return Err(ProviderStreamingContractError::InvalidOrder);
                }
                validate_provider_headers(&head.headers, &self.contract)?;
                self.response_started = true;
                self.response_status = Some(head.status);
            }
            ProviderResponseFrameEventV2::Chunk(chunk) => {
                if !self.response_started
                    || self.request_method == ProviderHttpMethodV2::Head
                    || self
                        .response_status
                        .is_some_and(|status| matches!(status, 204 | 205 | 304))
                {
                    return Err(ProviderStreamingContractError::InvalidOrder);
                }
                chunk.validate(&self.contract)?;
                let response_bytes = self
                    .response_bytes
                    .checked_add(
                        u64::try_from(chunk.byte_length())
                            .map_err(|_| ProviderStreamingContractError::BodyLimit)?,
                    )
                    .ok_or(ProviderStreamingContractError::BodyLimit)?;
                if response_bytes > self.contract.maximum_response_body_bytes {
                    return Err(ProviderStreamingContractError::BodyLimit);
                }
                self.response_bytes = response_bytes;
            }
            ProviderResponseFrameEventV2::Terminal(terminal) => {
                self.validate_terminal(terminal)?;
                self.terminal = true;
                self.revoked.store(true, Ordering::Release);
            }
        }
        self.last_response_sequence = Some(frame.sequence);
        self.next_response_sequence = frame
            .sequence
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidSequence)?;
        Ok(())
    }

    fn validate_terminal(
        &self,
        terminal: &ProviderStreamTerminalV2,
    ) -> Result<(), ProviderStreamingContractError> {
        match terminal {
            ProviderStreamTerminalV2::Completed { receipt } => {
                if !self.request_finished
                    || !self.response_started
                    || self
                        .upload_completions
                        .iter()
                        .any(|completed| !completed.load(Ordering::Acquire))
                    || receipt.is_empty()
                    || receipt.len() > MAX_PROVIDER_RESULT_RECEIPT_BYTES
                {
                    return Err(ProviderStreamingContractError::InvalidTerminal);
                }
            }
            ProviderStreamTerminalV2::Failed { code, message } => {
                if !valid_dotted_identifier(code)
                    || message.is_empty()
                    || message.len() > MAX_PROVIDER_STREAM_HEADER_VALUE_BYTES
                {
                    return Err(ProviderStreamingContractError::InvalidTerminal);
                }
            }
            ProviderStreamTerminalV2::Cancelled => {}
        }
        Ok(())
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn revoke(&mut self) {
        self.terminal = true;
        self.revoked.store(true, Ordering::Release);
    }

    pub fn request_bytes(&self) -> u64 {
        self.request_bytes
    }

    pub fn response_bytes(&self) -> u64 {
        self.response_bytes
    }

    fn require_active_handle(
        &self,
        handle: ProviderStreamHandleV2,
    ) -> Result<(), ProviderStreamingContractError> {
        handle.validate()?;
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProviderStreamingContractError::RevokedHandle);
        }
        require_matching_stream_handle(handle, self.handle)?;
        if self.terminal {
            return Err(ProviderStreamingContractError::InvalidOrder);
        }
        Ok(())
    }
}

pub struct ProviderUploadValidatorV2 {
    handle: ProviderStreamHandleV2,
    expected_bytes: u64,
    expected_sha256: String,
    maximum_chunk_bytes: u32,
    received_bytes: u64,
    next_sequence: u64,
    digest: Sha256,
    terminal: bool,
    revoked: Arc<AtomicBool>,
    completed: Arc<AtomicBool>,
}

impl ProviderUploadValidatorV2 {
    fn new(
        handle: ProviderStreamHandleV2,
        expected_bytes: u64,
        expected_sha256: String,
        maximum_chunk_bytes: u32,
        revoked: Arc<AtomicBool>,
        completed: Arc<AtomicBool>,
    ) -> Self {
        Self {
            handle,
            expected_bytes,
            expected_sha256,
            maximum_chunk_bytes,
            received_bytes: 0,
            next_sequence: 0,
            digest: Sha256::new(),
            terminal: false,
            revoked,
            completed,
        }
    }

    pub fn write_chunk(
        &mut self,
        chunk: &ProviderRequestChunkV2,
    ) -> Result<(), ProviderStreamingContractError> {
        chunk.handle.validate()?;
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProviderStreamingContractError::RevokedHandle);
        }
        require_matching_stream_handle(chunk.handle, self.handle)?;
        if self.terminal
            || chunk.sequence != self.next_sequence
            || (chunk.bytes.is_empty() && !chunk.end)
        {
            return Err(if chunk.sequence != self.next_sequence {
                ProviderStreamingContractError::InvalidSequence
            } else {
                ProviderStreamingContractError::InvalidOrder
            });
        }
        if u32::try_from(chunk.bytes.len()).map_or(true, |length| length > self.maximum_chunk_bytes)
        {
            return Err(ProviderStreamingContractError::ChunkLimit);
        }
        let received_bytes = self
            .received_bytes
            .checked_add(
                u64::try_from(chunk.bytes.len())
                    .map_err(|_| ProviderStreamingContractError::BodyLimit)?,
            )
            .ok_or(ProviderStreamingContractError::BodyLimit)?;
        if received_bytes > self.expected_bytes
            || (chunk.end && received_bytes != self.expected_bytes)
        {
            return Err(ProviderStreamingContractError::BodyLimit);
        }
        let next_sequence = chunk
            .sequence
            .checked_add(1)
            .ok_or(ProviderStreamingContractError::InvalidSequence)?;
        let mut digest = self.digest.clone();
        digest.update(&chunk.bytes);
        if chunk.end && format!("{:x}", digest.clone().finalize()) != self.expected_sha256 {
            return Err(ProviderStreamingContractError::InvalidUpload);
        }
        self.received_bytes = received_bytes;
        self.next_sequence = next_sequence;
        self.digest = digest;
        self.terminal = chunk.end;
        if chunk.end {
            self.completed.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
}

fn validate_provider_headers(
    headers: &[ProviderHeaderV2],
    contract: &ProviderStreamingContractV2,
) -> Result<(), ProviderStreamingContractError> {
    if headers.len() > usize::from(contract.maximum_headers) {
        return Err(ProviderStreamingContractError::InvalidHeaders);
    }
    let mut aggregate = 0usize;
    for header in headers {
        if header.name.is_empty()
            || header.name.len() > MAX_PROVIDER_STREAM_HEADER_NAME_BYTES
            || !header.name.bytes().all(valid_http_token_byte)
            || header.value.len() > MAX_PROVIDER_STREAM_HEADER_VALUE_BYTES
            || header.value.bytes().any(|byte| {
                byte == b'\r' || byte == b'\n' || byte == 0x7f || (byte < 0x20 && byte != b'\t')
            })
        {
            return Err(ProviderStreamingContractError::InvalidHeaders);
        }
        aggregate = aggregate
            .checked_add(header.name.len())
            .and_then(|bytes| bytes.checked_add(header.value.len()))
            .ok_or(ProviderStreamingContractError::InvalidHeaders)?;
    }
    if aggregate > usize::try_from(contract.maximum_header_bytes).unwrap_or(usize::MAX) {
        return Err(ProviderStreamingContractError::InvalidHeaders);
    }
    Ok(())
}

fn valid_provider_request_authority(value: &str, maximum_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= maximum_bytes
        && value.is_ascii()
        && value.trim() == value
        && value.bytes().all(|byte| byte.is_ascii_graphic())
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginNode {
    pub id: String,
    pub version: ApiVersion,
    pub display_name: String,
    pub category: String,
    pub ports: Vec<PluginPort>,
    pub determinism: DeterminismPolicy,
    pub cache: CachePolicy,
    pub effects: EffectPolicy,
}

impl PluginNode {
    pub fn try_from_legacy_descriptor(
        descriptor: &NodeDescriptor,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        let mut ports = Vec::with_capacity(descriptor.inputs.len() + descriptor.outputs.len());
        for (direction, descriptors) in [
            (PortDirection::Input, descriptor.inputs.as_slice()),
            (PortDirection::Output, descriptor.outputs.as_slice()),
        ] {
            for port in descriptors {
                let type_id = registry.resolve(&port.type_name)?.clone();
                let serialization = match registry.family(&type_id)? {
                    ValueFamily::Scalar => PortSerialization::Inline,
                    ValueFamily::Tensor | ValueFamily::Model => PortSerialization::Handle,
                    ValueFamily::Artifact => PortSerialization::ArtifactReference,
                };
                ports.push(PluginPort {
                    id: normalized_port_id(&port.name)?,
                    name: port.name.clone(),
                    direction,
                    type_id,
                    cardinality: PortCardinality::Singular,
                    presence: if port.required {
                        PortPresence::Required
                    } else {
                        PortPresence::Optional
                    },
                    hidden: false,
                    lazy: false,
                    default: None,
                    serialization,
                    accepted_legacy_names: Vec::new(),
                });
            }
        }
        Ok(Self {
            id: descriptor.type_name.clone(),
            version: ApiVersion::new(1, 0, 0),
            display_name: descriptor.display_name.clone(),
            category: "legacy".to_owned(),
            ports,
            determinism: DeterminismPolicy::Deterministic,
            cache: CachePolicy::InputIdentity,
            effects: EffectPolicy::Pure,
        })
    }
}

fn normalized_port_id(name: &str) -> Result<String, PluginContractError> {
    let mut result = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            result.push(character.to_ascii_lowercase());
        } else if character == '_' || character == '-' || character == ' ' {
            if !result.ends_with('-') {
                result.push('-');
            }
        } else {
            return Err(PluginContractError::InvalidIdentifier(name.to_owned()));
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if valid_dotted_identifier(&result) {
        Ok(result)
    } else {
        Err(PluginContractError::InvalidIdentifier(name.to_owned()))
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityKind {
    Filesystem,
    NetworkProvider,
    Secret,
    Clock,
    Randomness,
    Model,
    TransactionalOutput,
    SanitizedLog,
    DeclarativeUi,
    Route,
    ProviderUpload,
    ProviderCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityQuota {
    pub maximum_operations: u64,
    pub maximum_request_bytes: u64,
    pub maximum_response_bytes: u64,
    pub maximum_total_bytes: u64,
    pub maximum_handles: u32,
    pub timeout_milliseconds: u64,
}

impl CapabilityQuota {
    pub fn validate(&self) -> Result<(), PluginContractError> {
        if self.maximum_operations == 0
            || self.maximum_request_bytes == 0
            || self.maximum_response_bytes == 0
            || self.maximum_total_bytes == 0
            || self.maximum_handles == 0
            || self.timeout_milliseconds == 0
            || self.maximum_request_bytes > self.maximum_total_bytes
            || self.maximum_response_bytes > self.maximum_total_bytes
        {
            return Err(PluginContractError::InvalidQuota);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityRequest {
    pub kind: CapabilityKind,
    pub scope: String,
    pub quota: CapabilityQuota,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestSignature {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestProvenance {
    pub source: String,
    pub publisher: String,
    pub registry: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyMapping {
    pub legacy_identifier: String,
    pub node_id: String,
    pub node_version: ApiVersion,
    #[serde(default)]
    pub legacy_widget_names: Vec<String>,
    #[serde(default)]
    pub input_translations: Vec<LegacyInputTranslation>,
    #[serde(default)]
    pub output_translations: Vec<LegacyOutputTranslation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum LegacyInputTranslation {
    Rename {
        target_port_id: String,
        legacy_input_id: String,
    },
    Constant {
        target_port_id: String,
        value: ScalarValue,
    },
}

impl LegacyInputTranslation {
    pub fn target_port_id(&self) -> &str {
        match self {
            Self::Rename { target_port_id, .. } | Self::Constant { target_port_id, .. } => {
                target_port_id
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LegacyOutputTranslation {
    pub target_port_index: u32,
    pub legacy_output_index: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentLegacyMapping {
    pub legacy_identifier: String,
    pub node_id: String,
    pub node_version: ApiVersion,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UiContribution {
    pub id: String,
    pub surface: String,
    pub state_schema: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteDeclaration {
    pub id: String,
    pub method: String,
    pub path: String,
    pub maximum_request_bytes: u64,
    pub maximum_response_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginManifest {
    pub schema_version: u16,
    pub identifier: String,
    pub plugin_version: ApiVersion,
    pub api: ApiRequirement,
    pub digest_sha256: String,
    pub signature: ManifestSignature,
    pub provenance: ManifestProvenance,
    pub nodes: Vec<PluginNode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_binding: Option<ProviderBindingSet>,
    pub capabilities: Vec<CapabilityRequest>,
    pub ui: Vec<UiContribution>,
    pub routes: Vec<RouteDeclaration>,
    pub legacy_mappings: Vec<LegacyMapping>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderPluginManifestV2 {
    pub schema_version: u16,
    pub component_world: String,
    pub manifest: PluginManifest,
    pub streaming: ProviderStreamingContractV2,
    pub signature: ManifestSignature,
}

impl ProviderPluginManifestV2 {
    pub fn validate(&self, registry: &TypeRegistry) -> Result<(), PluginContractError> {
        if self.schema_version != PROVIDER_MANIFEST_SCHEMA_VERSION_V2 {
            return Err(PluginContractError::UnsupportedManifestSchema(
                self.schema_version,
            ));
        }
        if self.component_world != PROVIDER_COMPONENT_WORLD_V2
            || self.manifest.provider_binding.is_none()
            || !self
                .manifest
                .api
                .required_features
                .iter()
                .any(|feature| feature == PROVIDER_STREAMING_API_FEATURE_V2)
        {
            return Err(PluginContractError::InvalidProviderStreamingContract);
        }
        self.manifest.validate(registry)?;
        self.streaming
            .validate()
            .map_err(|_| PluginContractError::InvalidProviderStreamingContract)?;
        let upload_capabilities = self
            .manifest
            .capabilities
            .iter()
            .filter(|capability| capability.kind == CapabilityKind::ProviderUpload)
            .collect::<Vec<_>>();
        let cost_capabilities = self
            .manifest
            .capabilities
            .iter()
            .filter(|capability| capability.kind == CapabilityKind::ProviderCost)
            .collect::<Vec<_>>();
        let upload_capability_matches = match upload_capabilities.as_slice() {
            [capability] if self.streaming.uploads => {
                capability.quota.maximum_operations >= u64::from(self.streaming.maximum_uploads)
                    && capability.quota.maximum_request_bytes
                        >= self
                            .streaming
                            .maximum_request_body_bytes
                            .min(self.streaming.maximum_upload_body_bytes)
                    && capability.quota.maximum_total_bytes
                        >= self.streaming.maximum_upload_body_bytes
                    && capability.quota.maximum_handles >= self.streaming.maximum_uploads
                    && capability.quota.timeout_milliseconds
                        >= self.streaming.maximum_wait_milliseconds
            }
            [] => !self.streaming.uploads,
            _ => false,
        };
        let cost_capability_matches = match cost_capabilities.as_slice() {
            [capability] if self.streaming.cost_requests => {
                let maximum_cost_bytes = u64::from(self.streaming.maximum_cost_requests)
                    .checked_mul(
                        MAX_PROVIDER_COST_REQUEST_BYTES + MAX_PROVIDER_COST_RESPONSE_BYTES,
                    );
                capability.quota.maximum_operations
                    >= u64::from(self.streaming.maximum_cost_requests)
                    && capability.quota.maximum_handles != 0
                    && capability.quota.maximum_request_bytes >= MAX_PROVIDER_COST_REQUEST_BYTES
                    && capability.quota.maximum_response_bytes >= MAX_PROVIDER_COST_RESPONSE_BYTES
                    && maximum_cost_bytes.is_some_and(|maximum_cost_bytes| {
                        capability.quota.maximum_total_bytes >= maximum_cost_bytes
                    })
                    && capability.quota.timeout_milliseconds
                        >= self.streaming.maximum_wait_milliseconds
            }
            [] => !self.streaming.cost_requests,
            _ => false,
        };
        if !upload_capability_matches || !cost_capability_matches {
            return Err(PluginContractError::InvalidProviderStreamingContract);
        }
        if self.signature.algorithm != PLUGIN_SIGNATURE_ALGORITHM
            || !valid_authority_identifier(&self.signature.key_id)
            || self.signature.value.len() != ED25519_SIGNATURE_BYTES * 2
            || !self
                .signature
                .value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PluginContractError::InvalidSignatureMetadata);
        }
        self.signing_payload()?;
        Ok(())
    }

    pub fn signing_payload(&self) -> Result<Vec<u8>, PluginContractError> {
        let manifest_payload = self.manifest.signing_payload();
        let mut writer = CanonicalWriter::with_limit(MAX_MANIFEST_BYTES);
        writer.raw(PROVIDER_MANIFEST_V2_SIGNATURE_DOMAIN);
        writer.u16(self.schema_version);
        writer.string(&self.component_world);
        writer.usize(manifest_payload.len());
        writer.raw(&manifest_payload);
        writer.string(&self.manifest.signature.value);
        self.streaming.write_canonical(&mut writer);
        writer.string(&self.signature.algorithm);
        writer.string(&self.signature.key_id);
        if writer.overflowed() {
            return Err(PluginContractError::ManifestTooLarge);
        }
        Ok(writer.finish())
    }

    pub fn component_projection(
        &self,
    ) -> Result<ProviderComponentManifestProjectionV2, PluginContractError> {
        Ok(ProviderComponentManifestProjectionV2 {
            schema_version: self.schema_version,
            component_world: self.component_world.clone(),
            manifest: self.manifest.component_projection(),
            provider_binding: self
                .manifest
                .provider_binding
                .clone()
                .ok_or(PluginContractError::InvalidProviderStreamingContract)?,
            streaming: self.streaming.clone(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentManifestProjection {
    pub component_world: String,
    pub schema_version: u16,
    pub identifier: String,
    pub plugin_version: ApiVersion,
    pub api: ApiRequirement,
    pub nodes: Vec<PluginNode>,
    pub capabilities: Vec<CapabilityRequest>,
    pub ui: Vec<UiContribution>,
    pub routes: Vec<RouteDeclaration>,
    pub legacy_mappings: Vec<ComponentLegacyMapping>,
}

impl ComponentManifestProjection {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PluginContractError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| PluginContractError::InvalidComponentProjection(error.to_string()))?;
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(PluginContractError::ManifestTooLarge);
        }
        Ok(bytes)
    }

    pub fn validate_for_manifest(
        &self,
        manifest: &PluginManifest,
        registry: &TypeRegistry,
    ) -> Result<(), PluginContractError> {
        manifest.validate(registry)?;
        self.canonical_bytes()?;
        if self != &manifest.component_projection() {
            return Err(PluginContractError::ComponentProjectionMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderComponentManifestProjectionV2 {
    pub schema_version: u16,
    pub component_world: String,
    pub manifest: ComponentManifestProjection,
    pub provider_binding: ProviderBindingSet,
    pub streaming: ProviderStreamingContractV2,
}

impl ProviderComponentManifestProjectionV2 {
    pub fn validate_for_manifest(
        &self,
        manifest: &ProviderPluginManifestV2,
        registry: &TypeRegistry,
    ) -> Result<(), PluginContractError> {
        manifest.validate(registry)?;
        if self != &manifest.component_projection()? {
            return Err(PluginContractError::ComponentProjectionMismatch);
        }
        Ok(())
    }
}

impl PluginManifest {
    pub fn validate(&self, registry: &TypeRegistry) -> Result<(), PluginContractError> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(PluginContractError::UnsupportedManifestSchema(
                self.schema_version,
            ));
        }
        if !valid_authority_identifier(&self.identifier) {
            return Err(PluginContractError::InvalidIdentifier(
                self.identifier.clone(),
            ));
        }
        if self.plugin_version.major == 0 {
            return Err(PluginContractError::InvalidVersion(
                self.plugin_version.to_string(),
            ));
        }
        if self.api.major == 0 || self.api.minimum_minor > self.api.maximum_minor {
            return Err(PluginContractError::InvalidApiRange);
        }
        if self.api.required_features.len() > 256
            || self
                .api
                .required_features
                .iter()
                .any(|feature| !valid_dotted_identifier(feature))
            || self
                .api
                .required_features
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != self.api.required_features.len()
        {
            return Err(PluginContractError::InvalidApiFeatures);
        }
        validate_sha256(&self.digest_sha256)?;
        if self.signature.algorithm != PLUGIN_SIGNATURE_ALGORITHM
            || !valid_authority_identifier(&self.signature.key_id)
            || self.signature.value.len() != ED25519_SIGNATURE_BYTES * 2
            || !self
                .signature
                .value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(PluginContractError::InvalidSignatureMetadata);
        }
        if self.provenance.source.is_empty()
            || self.provenance.publisher.is_empty()
            || self.provenance.source.len() > 2_048
            || self.provenance.publisher.len() > 256
            || self
                .provenance
                .registry
                .as_ref()
                .is_some_and(|registry| registry.is_empty() || registry.len() > 2_048)
        {
            return Err(PluginContractError::InvalidProvenance);
        }
        if self.nodes.is_empty() || self.nodes.len() > MAX_MANIFEST_NODES {
            return Err(PluginContractError::InvalidNodeCount(self.nodes.len()));
        }

        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            if !valid_dotted_identifier(&node.id) || !node_ids.insert(node.id.as_str()) {
                return Err(PluginContractError::DuplicateOrInvalidNode(node.id.clone()));
            }
            if node.version.major == 0
                || node.display_name.is_empty()
                || node.display_name.len() > 256
                || node.category.is_empty()
                || node.category.len() > 512
            {
                return Err(PluginContractError::InvalidNodeMetadata(node.id.clone()));
            }
            if node.ports.len() > MAX_PORTS_PER_NODE {
                return Err(PluginContractError::InvalidPortCount {
                    node: node.id.clone(),
                    count: node.ports.len(),
                });
            }
            let mut port_ids = BTreeSet::new();
            for port in &node.ports {
                if !valid_dotted_identifier(&port.id)
                    || port.name.is_empty()
                    || port.name.len() > 256
                    || !port_ids.insert(port.id.as_str())
                {
                    return Err(PluginContractError::DuplicateOrInvalidPort {
                        node: node.id.clone(),
                        port: port.id.clone(),
                    });
                }
                let family = registry.family(&port.type_id)?;
                let valid_serialization = matches!(
                    (family, port.serialization),
                    (ValueFamily::Scalar, PortSerialization::Inline)
                        | (ValueFamily::Tensor, PortSerialization::Handle)
                        | (ValueFamily::Artifact, PortSerialization::ArtifactReference)
                        | (ValueFamily::Model, PortSerialization::Handle)
                );
                if !valid_serialization {
                    return Err(PluginContractError::InvalidPortSerialization {
                        port: port.id.clone(),
                        family,
                        serialization: port.serialization,
                    });
                }
                if port.default.is_some()
                    && (family != ValueFamily::Scalar
                        || port.direction != PortDirection::Input
                        || port.cardinality != PortCardinality::Singular)
                {
                    return Err(PluginContractError::InvalidPortDefault(port.id.clone()));
                }
                if let Some(default) = &port.default {
                    validate_scalar(default)
                        .map_err(|_| PluginContractError::InvalidPortDefault(port.id.clone()))?;
                    validate_scalar_type(&port.type_id, default)
                        .map_err(|_| PluginContractError::InvalidPortDefault(port.id.clone()))?;
                }
                if port.hidden != (port.presence == PortPresence::Hidden) {
                    return Err(PluginContractError::InconsistentHiddenPort(port.id.clone()));
                }
                let mut aliases = BTreeSet::new();
                if port.accepted_legacy_names.len() > 256 {
                    return Err(PluginContractError::DuplicatePortAlias(port.id.clone()));
                }
                for alias in &port.accepted_legacy_names {
                    if alias.is_empty() || alias.len() > 256 || !aliases.insert(alias) {
                        return Err(PluginContractError::DuplicatePortAlias(alias.clone()));
                    }
                }
            }
            let mut aliases = BTreeMap::new();
            for port in &node.ports {
                for alias in &port.accepted_legacy_names {
                    if port_ids.contains(alias.as_str())
                        || aliases.insert(alias.as_str(), port.id.as_str()).is_some()
                    {
                        return Err(PluginContractError::DuplicatePortAlias(alias.clone()));
                    }
                }
            }
        }

        if let Some(provider_binding) = &self.provider_binding {
            provider_binding.validate(
                &self.identifier,
                &self.api.required_features,
                &self.nodes,
            )?;
        }

        if self.capabilities.len() > 1_024 {
            return Err(PluginContractError::DuplicateOrInvalidCapability);
        }
        let mut capabilities = BTreeSet::new();
        for capability in &self.capabilities {
            capability.quota.validate()?;
            if capability.scope.is_empty()
                || capability.scope.len() > 1_024
                || !capabilities.insert((capability.kind, capability.scope.as_str()))
            {
                return Err(PluginContractError::DuplicateOrInvalidCapability);
            }
        }
        if self.legacy_mappings.len() > MAX_LEGACY_IDENTIFIERS {
            return Err(PluginContractError::TooManyLegacyMappings);
        }
        let mut mappings = BTreeSet::new();
        for mapping in &self.legacy_mappings {
            let node = self
                .nodes
                .iter()
                .find(|node| node.id == mapping.node_id && node.version == mapping.node_version);
            if mapping.legacy_identifier.is_empty()
                || mapping.legacy_identifier.len() > 512
                || mapping.legacy_identifier.chars().any(char::is_control)
                || !node_ids.contains(mapping.node_id.as_str())
                || node.is_none()
                || !mappings.insert(mapping.legacy_identifier.as_str())
            {
                return Err(PluginContractError::DuplicateOrInvalidLegacyMapping(
                    mapping.legacy_identifier.clone(),
                ));
            }
            let node = node.ok_or_else(|| {
                PluginContractError::DuplicateOrInvalidLegacyMapping(
                    mapping.legacy_identifier.clone(),
                )
            })?;
            validate_legacy_translations(mapping, node, registry)?;
        }
        if self.ui.len() > 1_024 {
            return Err(PluginContractError::InvalidUiContribution(
                "too-many-contributions".to_owned(),
            ));
        }
        let mut ui_ids = BTreeSet::new();
        for contribution in &self.ui {
            if !valid_dotted_identifier(&contribution.id)
                || contribution.surface.is_empty()
                || contribution.surface.len() > 256
                || contribution.state_schema.len() > 65_536
                || !ui_ids.insert(contribution.id.as_str())
            {
                return Err(PluginContractError::InvalidUiContribution(
                    contribution.id.clone(),
                ));
            }
        }
        if self.routes.len() > 1_024 {
            return Err(PluginContractError::InvalidRoute(
                "too-many-routes".to_owned(),
            ));
        }
        let mut route_ids = BTreeSet::new();
        for route in &self.routes {
            if !valid_dotted_identifier(&route.id)
                || !matches!(
                    route.method.as_str(),
                    "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
                )
                || !valid_route_path(&route.path)
                || route.maximum_request_bytes == 0
                || route.maximum_response_bytes == 0
                || !route_ids.insert(route.id.as_str())
            {
                return Err(PluginContractError::InvalidRoute(route.id.clone()));
            }
        }
        for capability in &self.capabilities {
            let references_declared_surface = match capability.kind {
                CapabilityKind::DeclarativeUi => ui_ids.contains(capability.scope.as_str()),
                CapabilityKind::Route => route_ids.contains(capability.scope.as_str()),
                _ => true,
            };
            if !references_declared_surface {
                return Err(PluginContractError::DuplicateOrInvalidCapability);
            }
        }
        let mut writer = CanonicalWriter::with_limit(MAX_MANIFEST_BYTES);
        self.write_signing_payload(&mut writer);
        writer.usize(self.signature.value.len());
        writer.raw(self.signature.value.as_bytes());
        if writer.overflowed() {
            return Err(PluginContractError::ManifestTooLarge);
        }
        Ok(())
    }

    pub fn signing_payload(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::default();
        self.write_signing_payload(&mut writer);
        writer.finish()
    }

    fn write_signing_payload(&self, writer: &mut CanonicalWriter) {
        writer.u16(self.schema_version);
        writer.string(&self.identifier);
        writer.version(self.plugin_version);
        writer.u16(self.api.major);
        writer.u16(self.api.minimum_minor);
        writer.u16(self.api.maximum_minor);
        writer.strings(&self.api.required_features);
        writer.string(&self.digest_sha256);
        writer.string(&self.signature.algorithm);
        writer.string(&self.signature.key_id);
        writer.string(&self.provenance.source);
        writer.string(&self.provenance.publisher);
        writer.optional_string(self.provenance.registry.as_deref());
        self.write_declarations(writer);
        if let Some(provider_binding) = &self.provider_binding {
            writer.string(PROVIDER_BINDING_CANONICAL_DOMAIN);
            writer.string(&provider_binding.bindings_sha256);
            provider_binding.write_canonical(writer);
        }
    }

    fn write_declarations(&self, writer: &mut CanonicalWriter) {
        writer.usize(self.nodes.len());
        for node in &self.nodes {
            writer.string(&node.id);
            writer.version(node.version);
            writer.string(&node.display_name);
            writer.string(&node.category);
            writer.byte(node.determinism as u8);
            writer.byte(node.cache as u8);
            writer.byte(node.effects as u8);
            writer.usize(node.ports.len());
            for port in &node.ports {
                writer.string(&port.id);
                writer.string(&port.name);
                writer.byte(port.direction as u8);
                writer.string(&port.type_id.to_string());
                writer.byte(port.cardinality as u8);
                writer.byte(port.presence as u8);
                writer.byte(u8::from(port.hidden));
                writer.byte(u8::from(port.lazy));
                writer.byte(port.serialization as u8);
                writer.strings(&port.accepted_legacy_names);
                writer.scalar(port.default.as_ref());
            }
        }
        writer.usize(self.capabilities.len());
        for capability in &self.capabilities {
            writer.byte(capability.kind as u8);
            writer.string(&capability.scope);
            writer.u64(capability.quota.maximum_operations);
            writer.u64(capability.quota.maximum_request_bytes);
            writer.u64(capability.quota.maximum_response_bytes);
            writer.u64(capability.quota.maximum_total_bytes);
            writer.u32(capability.quota.maximum_handles);
            writer.u64(capability.quota.timeout_milliseconds);
        }
        writer.usize(self.ui.len());
        for ui in &self.ui {
            writer.string(&ui.id);
            writer.string(&ui.surface);
            writer.string(&ui.state_schema);
        }
        writer.usize(self.routes.len());
        for route in &self.routes {
            writer.string(&route.id);
            writer.string(&route.method);
            writer.string(&route.path);
            writer.u64(route.maximum_request_bytes);
            writer.u64(route.maximum_response_bytes);
        }
        writer.usize(self.legacy_mappings.len());
        for mapping in &self.legacy_mappings {
            writer.string(&mapping.legacy_identifier);
            writer.string(&mapping.node_id);
            writer.version(mapping.node_version);
            writer.strings(&mapping.legacy_widget_names);
            writer.usize(mapping.input_translations.len());
            for translation in &mapping.input_translations {
                match translation {
                    LegacyInputTranslation::Rename {
                        target_port_id,
                        legacy_input_id,
                    } => {
                        writer.byte(0);
                        writer.string(target_port_id);
                        writer.string(legacy_input_id);
                    }
                    LegacyInputTranslation::Constant {
                        target_port_id,
                        value,
                    } => {
                        writer.byte(1);
                        writer.string(target_port_id);
                        writer.scalar(Some(value));
                    }
                }
            }
            writer.usize(mapping.output_translations.len());
            for translation in &mapping.output_translations {
                writer.u32(translation.target_port_index);
                writer.u32(translation.legacy_output_index);
            }
        }
    }

    pub fn component_projection(&self) -> ComponentManifestProjection {
        ComponentManifestProjection {
            component_world: if self.provider_binding.is_some() {
                PROVIDER_COMPONENT_WORLD.to_owned()
            } else {
                COMPONENT_WORLD.to_owned()
            },
            schema_version: self.schema_version,
            identifier: self.identifier.clone(),
            plugin_version: self.plugin_version,
            api: self.api.clone(),
            nodes: self.nodes.clone(),
            capabilities: self.capabilities.clone(),
            ui: self.ui.clone(),
            routes: self.routes.clone(),
            legacy_mappings: self
                .legacy_mappings
                .iter()
                .map(|mapping| ComponentLegacyMapping {
                    legacy_identifier: mapping.legacy_identifier.clone(),
                    node_id: mapping.node_id.clone(),
                    node_version: mapping.node_version,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TensorValueWire")]
pub struct TensorValue {
    descriptor: TensorDescriptor,
    byte_length: u64,
    digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorValueWire {
    descriptor: TensorDescriptor,
    byte_length: u64,
    digest: String,
}

impl TryFrom<TensorValueWire> for TensorValue {
    type Error = PluginContractError;

    fn try_from(value: TensorValueWire) -> Result<Self, Self::Error> {
        Self::new(value.descriptor, value.byte_length, value.digest)
    }
}

impl TensorValue {
    pub fn new(
        descriptor: TensorDescriptor,
        byte_length: u64,
        digest: impl Into<String>,
    ) -> Result<Self, PluginContractError> {
        let value = Self {
            descriptor,
            byte_length,
            digest: digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn byte_length(&self) -> u64 {
        self.byte_length
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    pub fn minimum_backing_byte_length(&self) -> Result<u64, TensorError> {
        self.descriptor.minimum_backing_byte_length()
    }

    fn validate(&self) -> Result<(), PluginContractError> {
        validate_sha256(&self.digest)?;
        self.descriptor
            .validate_backing_byte_length(self.byte_length)
            .map_err(|error| PluginContractError::InvalidValue(error.to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ArtifactValueWire")]
pub struct ArtifactValue {
    namespace: String,
    identifier: String,
    byte_length: u64,
    digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactValueWire {
    namespace: String,
    identifier: String,
    byte_length: u64,
    digest: String,
}

impl TryFrom<ArtifactValueWire> for ArtifactValue {
    type Error = PluginContractError;

    fn try_from(value: ArtifactValueWire) -> Result<Self, Self::Error> {
        Self::new(
            value.namespace,
            value.identifier,
            value.byte_length,
            value.digest,
        )
    }
}

impl ArtifactValue {
    pub fn new(
        namespace: impl Into<String>,
        identifier: impl Into<String>,
        byte_length: u64,
        digest: impl Into<String>,
    ) -> Result<Self, PluginContractError> {
        let value = Self {
            namespace: namespace.into(),
            identifier: identifier.into(),
            byte_length,
            digest: digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn byte_length(&self) -> u64 {
        self.byte_length
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn validate(&self) -> Result<(), PluginContractError> {
        validate_abi_text(&self.namespace, 256, "artifact namespace")?;
        validate_abi_text(&self.identifier, 4_096, "artifact identifier")?;
        validate_sha256(&self.digest)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ModelValueWire")]
pub struct ModelValue {
    identifier: String,
    format: String,
    digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelValueWire {
    identifier: String,
    format: String,
    digest: String,
}

impl TryFrom<ModelValueWire> for ModelValue {
    type Error = PluginContractError;

    fn try_from(value: ModelValueWire) -> Result<Self, Self::Error> {
        Self::new(value.identifier, value.format, value.digest)
    }
}

impl ModelValue {
    pub fn new(
        identifier: impl Into<String>,
        format: impl Into<String>,
        digest: impl Into<String>,
    ) -> Result<Self, PluginContractError> {
        let value = Self {
            identifier: identifier.into(),
            format: format.into(),
            digest: digest.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn format(&self) -> &str {
        &self.format
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }

    fn validate(&self) -> Result<(), PluginContractError> {
        validate_abi_text(&self.identifier, 4_096, "model identifier")?;
        validate_abi_text(&self.format, 128, "model format")?;
        validate_sha256(&self.digest)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "value", rename_all = "kebab-case")]
pub enum PluginValueRepresentation {
    Scalar(ScalarValue),
    Tensor(TensorValue),
    Artifact(ArtifactValue),
    Model(ModelValue),
}

impl PluginValueRepresentation {
    pub fn family(&self) -> ValueFamily {
        match self {
            Self::Scalar(_) => ValueFamily::Scalar,
            Self::Tensor(_) => ValueFamily::Tensor,
            Self::Artifact(_) => ValueFamily::Artifact,
            Self::Model(_) => ValueFamily::Model,
        }
    }

    fn validate(&self) -> Result<(), PluginContractError> {
        match self {
            Self::Scalar(value) => validate_scalar(value)
                .map_err(|()| PluginContractError::InvalidValue("invalid scalar value".to_owned())),
            Self::Tensor(value) => value.validate(),
            Self::Artifact(value) => value.validate(),
            Self::Model(value) => value.validate(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PluginValue {
    type_id: CanonicalTypeId,
    representation: PluginValueRepresentation,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PluginValueWire {
    type_id: CanonicalTypeId,
    representation: PluginValueRepresentation,
}

impl<'de> Deserialize<'de> for PluginValue {
    fn deserialize<Deserializer>(deserializer: Deserializer) -> Result<Self, Deserializer::Error>
    where
        Deserializer: serde::Deserializer<'de>,
    {
        let wire = PluginValueWire::deserialize(deserializer)?;
        let registry = TypeRegistry::built_in().map_err(serde::de::Error::custom)?;
        Self::new(wire.type_id, wire.representation, &registry).map_err(serde::de::Error::custom)
    }
}

impl PluginValue {
    pub fn new(
        type_id: CanonicalTypeId,
        representation: PluginValueRepresentation,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        registry.require_family(&type_id, representation.family())?;
        representation.validate()?;
        if let PluginValueRepresentation::Scalar(value) = &representation {
            validate_scalar_type(&type_id, value)?;
        }
        Ok(Self {
            type_id,
            representation,
        })
    }

    pub fn scalar(
        type_id: CanonicalTypeId,
        value: ScalarValue,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        Self::new(type_id, PluginValueRepresentation::Scalar(value), registry)
    }

    pub fn tensor(
        type_id: CanonicalTypeId,
        value: TensorValue,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        Self::new(type_id, PluginValueRepresentation::Tensor(value), registry)
    }

    pub fn artifact(
        type_id: CanonicalTypeId,
        value: ArtifactValue,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        Self::new(
            type_id,
            PluginValueRepresentation::Artifact(value),
            registry,
        )
    }

    pub fn model(
        type_id: CanonicalTypeId,
        value: ModelValue,
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        Self::new(type_id, PluginValueRepresentation::Model(value), registry)
    }

    pub fn type_id(&self) -> &CanonicalTypeId {
        &self.type_id
    }

    pub fn representation(&self) -> &PluginValueRepresentation {
        &self.representation
    }

    pub fn family(&self) -> ValueFamily {
        self.representation.family()
    }

    pub fn into_representation(self) -> PluginValueRepresentation {
        self.representation
    }

    pub fn abi_bytes(&self) -> Result<Vec<u8>, PluginContractError> {
        serde_json::to_vec(self)
            .map_err(|error| PluginContractError::InvalidValue(error.to_string()))
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut writer = CanonicalWriter::default();
        writer.string(&self.type_id.to_string());
        writer.plugin_value_representation(&self.representation);
        writer.finish()
    }

    pub fn from_abi_bytes(
        bytes: &[u8],
        registry: &TypeRegistry,
    ) -> Result<Self, PluginContractError> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(PluginContractError::InvalidValue(
                "plugin value projection is too large".to_owned(),
            ));
        }
        let wire: PluginValueWire = serde_json::from_slice(bytes)
            .map_err(|error| PluginContractError::InvalidValue(error.to_string()))?;
        Self::new(wire.type_id, wire.representation, registry)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValueHandle {
    pub invocation: u64,
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputState {
    pub present: bool,
    pub length: u32,
    pub type_id: CanonicalTypeId,
    pub family: ValueFamily,
    pub cardinality: PortCardinality,
    pub presence: PortPresence,
    pub serialization: PortSerialization,
    pub lazy: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CapabilityCall {
    FilesystemRead {
        root: String,
        relative_path: String,
    },
    NetworkProvider {
        provider: String,
        endpoint: String,
        body: Vec<u8>,
        secret_id: Option<String>,
    },
    SecretExists {
        identifier: String,
    },
    ClockNow {
        clock: String,
    },
    RandomBytes {
        stream: String,
        length: u32,
    },
    ModelOpen {
        identifier: String,
    },
    OutputBegin {
        namespace: String,
        name: String,
    },
    OutputWrite {
        transaction: u64,
        bytes: Vec<u8>,
    },
    OutputCommit {
        transaction: u64,
    },
    Log {
        level: String,
        message: String,
    },
    UiSet {
        contribution: String,
        state: Vec<u8>,
    },
    RouteRespond {
        route: String,
        status: u16,
        body: Vec<u8>,
    },
}

impl CapabilityCall {
    pub fn kind(&self) -> CapabilityKind {
        match self {
            Self::FilesystemRead { .. } => CapabilityKind::Filesystem,
            Self::NetworkProvider { .. } => CapabilityKind::NetworkProvider,
            Self::SecretExists { .. } => CapabilityKind::Secret,
            Self::ClockNow { .. } => CapabilityKind::Clock,
            Self::RandomBytes { .. } => CapabilityKind::Randomness,
            Self::ModelOpen { .. } => CapabilityKind::Model,
            Self::OutputBegin { .. } | Self::OutputWrite { .. } | Self::OutputCommit { .. } => {
                CapabilityKind::TransactionalOutput
            }
            Self::Log { .. } => CapabilityKind::SanitizedLog,
            Self::UiSet { .. } => CapabilityKind::DeclarativeUi,
            Self::RouteRespond { .. } => CapabilityKind::Route,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CapabilityResponse {
    Bytes(Vec<u8>),
    Boolean(bool),
    TimestampMilliseconds(u64),
    Handle(u64),
    CommittedArtifact(String),
    Unit,
}

pub trait PluginInvocation {
    fn input_state(&self, port_id: &str) -> Result<InputState, InvocationError>;
    fn read_scalar_input(&self, port_id: &str, index: u32) -> Result<PluginValue, InvocationError>;
    fn take_input(&mut self, port_id: &str, index: u32) -> Result<ValueHandle, InvocationError>;
    fn read_handle(&self, handle: ValueHandle) -> Result<&PluginValue, InvocationError>;
    fn create_output_value(&mut self, value: PluginValue) -> Result<ValueHandle, InvocationError>;
    fn push_output(&mut self, port_id: &str, handle: ValueHandle) -> Result<(), InvocationError>;
    fn finish_output(&mut self, port_id: &str, present: bool) -> Result<(), InvocationError>;
    fn call(&mut self, call: CapabilityCall) -> Result<CapabilityResponse, InvocationError>;
    fn check_cancelled(&self) -> Result<(), InvocationError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CancelReason {
    User,
    Timeout,
    HostShutdown,
    CapabilityRevoked,
}

pub trait RustNodeInstance: Send {
    fn invoke(&mut self, invocation: &mut dyn PluginInvocation) -> Result<(), InvocationError>;
    fn cancel(&mut self, reason: CancelReason);
}

pub trait RustSimPlugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
    fn create_node(&self, node_id: &str) -> Result<Box<dyn RustNodeInstance>, PluginContractError>;
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InvocationError {
    Cancelled,
    TimedOut,
    UnknownPort(String),
    WrongDirection(String),
    MissingRequiredPort(String),
    InvalidCardinality(String),
    IndexOutOfBounds {
        port: String,
        index: u32,
    },
    AlreadyTaken {
        port: String,
        index: u32,
    },
    InvalidHandle,
    RevokedHandle,
    WrongValueFamily {
        port: String,
        expected: ValueFamily,
        actual: ValueFamily,
    },
    OutputAlreadyFinished(String),
    UnfinishedOutput(String),
    CapabilityDenied {
        kind: CapabilityKind,
        scope: String,
    },
    QuotaExceeded {
        kind: CapabilityKind,
        limit: String,
    },
    InvocationQuotaExceeded {
        limit: String,
    },
    InvalidCapabilityRequest(String),
    HostFailure(String),
    PluginFailure(String),
}

impl fmt::Display for InvocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("plugin invocation cancelled"),
            Self::TimedOut => formatter.write_str("plugin invocation timed out"),
            Self::UnknownPort(port) => write!(formatter, "unknown plugin port `{port}`"),
            Self::WrongDirection(port) => write!(formatter, "wrong direction for port `{port}`"),
            Self::MissingRequiredPort(port) => write!(formatter, "missing required port `{port}`"),
            Self::InvalidCardinality(port) => write!(formatter, "invalid cardinality for `{port}`"),
            Self::IndexOutOfBounds { port, index } => {
                write!(formatter, "index {index} is out of bounds for `{port}`")
            }
            Self::AlreadyTaken { port, index } => {
                write!(formatter, "value {index} on `{port}` was already taken")
            }
            Self::InvalidHandle => formatter.write_str("invalid invocation handle"),
            Self::RevokedHandle => formatter.write_str("invocation handle was revoked"),
            Self::WrongValueFamily {
                port,
                expected,
                actual,
            } => write!(
                formatter,
                "port `{port}` expects {expected:?}, received {actual:?}"
            ),
            Self::OutputAlreadyFinished(port) => write!(formatter, "output `{port}` is finished"),
            Self::UnfinishedOutput(port) => write!(formatter, "output `{port}` is not finished"),
            Self::CapabilityDenied { kind, scope } => {
                write!(formatter, "{kind:?} capability denied for `{scope}`")
            }
            Self::QuotaExceeded { kind, limit } => {
                write!(formatter, "{kind:?} capability exceeded {limit} quota")
            }
            Self::InvocationQuotaExceeded { limit } => {
                write!(formatter, "plugin invocation exceeded {limit} quota")
            }
            Self::InvalidCapabilityRequest(message)
            | Self::HostFailure(message)
            | Self::PluginFailure(message) => formatter.write_str(message),
        }
    }
}

impl Error for InvocationError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PluginContractError {
    InvalidVersion(String),
    UnsupportedApi {
        requested: String,
        host: String,
    },
    MissingApiFeature(String),
    UnsupportedManifestSchema(u16),
    InvalidIdentifier(String),
    InvalidApiRange,
    InvalidApiFeatures,
    InvalidDigest,
    InvalidSignatureMetadata,
    #[cfg(any(feature = "signing-tooling", test))]
    InvalidSigningKey,
    InvalidProvenance,
    InvalidProviderBindingSet,
    InvalidProviderStreamingContract,
    InvalidProviderReceiptSet,
    InvalidNodeCount(usize),
    DuplicateOrInvalidNode(String),
    InvalidNodeMetadata(String),
    InvalidPortCount {
        node: String,
        count: usize,
    },
    DuplicateOrInvalidPort {
        node: String,
        port: String,
    },
    InvalidPortDefault(String),
    InvalidPortSerialization {
        port: String,
        family: ValueFamily,
        serialization: PortSerialization,
    },
    InconsistentHiddenPort(String),
    DuplicatePortAlias(String),
    InvalidQuota,
    DuplicateOrInvalidCapability,
    TooManyLegacyMappings,
    DuplicateOrInvalidLegacyMapping(String),
    InvalidLegacyTranslation(String),
    InvalidUiContribution(String),
    InvalidRoute(String),
    ManifestTooLarge,
    InvalidValue(String),
    InvalidComponentProjection(String),
    ComponentProjectionMismatch,
    UnknownNode(String),
    TypeRegistry(TypeRegistryError),
}

impl fmt::Display for PluginContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion(version) => {
                write!(formatter, "invalid semantic version `{version}`")
            }
            Self::UnsupportedApi { requested, host } => {
                write!(
                    formatter,
                    "plugin API `{requested}` is incompatible with host `{host}`"
                )
            }
            Self::MissingApiFeature(feature) => {
                write!(formatter, "missing API feature `{feature}`")
            }
            Self::UnsupportedManifestSchema(version) => {
                write!(formatter, "unsupported plugin manifest schema {version}")
            }
            Self::InvalidIdentifier(value) => write!(formatter, "invalid identifier `{value}`"),
            Self::InvalidApiRange => formatter.write_str("invalid plugin API minor-version range"),
            Self::InvalidApiFeatures => formatter.write_str("invalid plugin API feature set"),
            Self::InvalidDigest => formatter.write_str("invalid SHA-256 digest"),
            Self::InvalidSignatureMetadata => {
                formatter.write_str("invalid manifest signature metadata")
            }
            #[cfg(any(feature = "signing-tooling", test))]
            Self::InvalidSigningKey => formatter.write_str("invalid Ed25519 signing key"),
            Self::InvalidProvenance => formatter.write_str("invalid manifest provenance"),
            Self::InvalidProviderBindingSet => {
                formatter.write_str("invalid signed provider binding set")
            }
            Self::InvalidProviderStreamingContract => {
                formatter.write_str("invalid provider streaming component contract")
            }
            Self::InvalidProviderReceiptSet => {
                formatter.write_str("invalid provider result receipt set")
            }
            Self::InvalidNodeCount(count) => write!(formatter, "invalid plugin node count {count}"),
            Self::DuplicateOrInvalidNode(node) => {
                write!(formatter, "duplicate or invalid node `{node}`")
            }
            Self::InvalidNodeMetadata(node) => {
                write!(formatter, "node `{node}` has invalid metadata")
            }
            Self::InvalidPortCount { node, count } => {
                write!(formatter, "node `{node}` has invalid port count {count}")
            }
            Self::DuplicateOrInvalidPort { node, port } => {
                write!(
                    formatter,
                    "node `{node}` has duplicate or invalid port `{port}`"
                )
            }
            Self::InvalidPortDefault(port) => {
                write!(formatter, "port `{port}` has an invalid default")
            }
            Self::InvalidPortSerialization {
                port,
                family,
                serialization,
            } => write!(
                formatter,
                "port `{port}` cannot use {serialization:?} serialization for {family:?} values"
            ),
            Self::InconsistentHiddenPort(port) => {
                write!(formatter, "port `{port}` has inconsistent hidden state")
            }
            Self::DuplicatePortAlias(alias) => write!(formatter, "duplicate port alias `{alias}`"),
            Self::InvalidQuota => formatter.write_str("invalid capability quota"),
            Self::DuplicateOrInvalidCapability => {
                formatter.write_str("duplicate or invalid capability request")
            }
            Self::TooManyLegacyMappings => formatter.write_str("too many legacy mappings"),
            Self::DuplicateOrInvalidLegacyMapping(mapping) => {
                write!(formatter, "duplicate or invalid legacy mapping `{mapping}`")
            }
            Self::InvalidLegacyTranslation(mapping) => {
                write!(
                    formatter,
                    "legacy mapping `{mapping}` has an invalid translation"
                )
            }
            Self::InvalidUiContribution(identifier) => {
                write!(formatter, "invalid UI contribution `{identifier}`")
            }
            Self::InvalidRoute(route) => write!(formatter, "invalid route `{route}`"),
            Self::ManifestTooLarge => formatter.write_str("plugin manifest is too large"),
            Self::InvalidValue(message) => write!(formatter, "invalid plugin value: {message}"),
            Self::InvalidComponentProjection(message) => {
                write!(
                    formatter,
                    "invalid component manifest projection: {message}"
                )
            }
            Self::ComponentProjectionMismatch => formatter
                .write_str("component manifest projection does not match its signed manifest"),
            Self::UnknownNode(node) => write!(formatter, "unknown plugin node `{node}`"),
            Self::TypeRegistry(error) => error.fmt(formatter),
        }
    }
}

impl Error for PluginContractError {}

impl From<TypeRegistryError> for PluginContractError {
    fn from(error: TypeRegistryError) -> Self {
        Self::TypeRegistry(error)
    }
}

fn validate_sha256(digest: &str) -> Result<(), PluginContractError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PluginContractError::InvalidDigest);
    }
    Ok(())
}

fn valid_feature_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-')
}

#[cfg(any(feature = "signing-tooling", test))]
fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn validate_abi_text(
    value: &str,
    maximum_bytes: usize,
    field: &str,
) -> Result<(), PluginContractError> {
    if value.is_empty()
        || value.len() > maximum_bytes
        || value != value.trim()
        || value.chars().any(char::is_control)
    {
        return Err(PluginContractError::InvalidValue(format!(
            "{field} is empty, oversized, padded, or contains control characters"
        )));
    }
    Ok(())
}

fn validate_legacy_translations(
    mapping: &LegacyMapping,
    node: &PluginNode,
    registry: &TypeRegistry,
) -> Result<(), PluginContractError> {
    if mapping.legacy_widget_names.len() > MAX_LEGACY_TRANSLATIONS_PER_MAPPING
        || mapping.input_translations.len() > MAX_LEGACY_TRANSLATIONS_PER_MAPPING
        || mapping.output_translations.len() > MAX_LEGACY_TRANSLATIONS_PER_MAPPING
    {
        return Err(PluginContractError::InvalidLegacyTranslation(
            mapping.legacy_identifier.clone(),
        ));
    }
    let valid_legacy_name = |value: &str| {
        !value.is_empty() && value.len() <= 512 && !value.chars().any(char::is_control)
    };
    let mut widget_names = BTreeSet::new();
    if mapping
        .legacy_widget_names
        .iter()
        .any(|name| !valid_legacy_name(name) || !widget_names.insert(name.as_str()))
    {
        return Err(PluginContractError::InvalidLegacyTranslation(
            mapping.legacy_identifier.clone(),
        ));
    }

    let mut target_inputs = BTreeSet::new();
    for translation in &mapping.input_translations {
        let target_port_id = translation.target_port_id();
        let target_port = node
            .ports
            .iter()
            .find(|port| port.direction == PortDirection::Input && port.id == target_port_id)
            .ok_or_else(|| {
                PluginContractError::InvalidLegacyTranslation(mapping.legacy_identifier.clone())
            })?;
        if !target_inputs.insert(target_port_id) {
            return Err(PluginContractError::InvalidLegacyTranslation(
                mapping.legacy_identifier.clone(),
            ));
        }
        match translation {
            LegacyInputTranslation::Rename {
                legacy_input_id, ..
            } => {
                if !valid_legacy_name(legacy_input_id) {
                    return Err(PluginContractError::InvalidLegacyTranslation(
                        mapping.legacy_identifier.clone(),
                    ));
                }
            }
            LegacyInputTranslation::Constant { value, .. } => {
                if target_port.cardinality != PortCardinality::Singular
                    || registry.family(&target_port.type_id)? != ValueFamily::Scalar
                    || target_port.serialization != PortSerialization::Inline
                    || validate_scalar(value).is_err()
                    || validate_scalar_type(&target_port.type_id, value).is_err()
                {
                    return Err(PluginContractError::InvalidLegacyTranslation(
                        mapping.legacy_identifier.clone(),
                    ));
                }
            }
        }
    }

    let output_ports = node
        .ports
        .iter()
        .filter(|port| port.direction == PortDirection::Output)
        .collect::<Vec<_>>();
    let mut target_output_indices = BTreeSet::new();
    let mut legacy_output_indices = BTreeSet::new();
    for translation in &mapping.output_translations {
        let target_port_index = usize::try_from(translation.target_port_index).map_err(|_| {
            PluginContractError::InvalidLegacyTranslation(mapping.legacy_identifier.clone())
        })?;
        if output_ports.get(target_port_index).is_none()
            || !target_output_indices.insert(translation.target_port_index)
            || usize::try_from(translation.legacy_output_index)
                .ok()
                .is_none_or(|index| index >= MAX_PORTS_PER_NODE)
            || !legacy_output_indices.insert(translation.legacy_output_index)
        {
            return Err(PluginContractError::InvalidLegacyTranslation(
                mapping.legacy_identifier.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_scalar(root: &ScalarValue) -> Result<(), ()> {
    const MAX_DEPTH: usize = 64;
    const MAX_VALUES: usize = 65_536;
    const MAX_BYTES: usize = 8 * 1024 * 1024;
    let mut stack = vec![(root, 0_usize)];
    let mut values = 0_usize;
    let mut bytes = 0_usize;
    while let Some((value, depth)) = stack.pop() {
        if depth > MAX_DEPTH {
            return Err(());
        }
        values = values.checked_add(1).ok_or(())?;
        if values > MAX_VALUES {
            return Err(());
        }
        match value {
            ScalarValue::String(value) => bytes = bytes.checked_add(value.len()).ok_or(())?,
            ScalarValue::Bytes(value) => bytes = bytes.checked_add(value.len()).ok_or(())?,
            ScalarValue::List(children) => {
                for child in children {
                    stack.push((child, depth + 1));
                }
            }
            ScalarValue::Record(fields) => {
                let mut keys = BTreeSet::new();
                for (key, child) in fields {
                    if key.is_empty() || !keys.insert(key) {
                        return Err(());
                    }
                    bytes = bytes.checked_add(key.len()).ok_or(())?;
                    stack.push((child, depth + 1));
                }
            }
            ScalarValue::Float(value) if !value.is_finite() => return Err(()),
            ScalarValue::Null
            | ScalarValue::Boolean(_)
            | ScalarValue::Integer(_)
            | ScalarValue::Float(_) => {}
        }
        if bytes > MAX_BYTES {
            return Err(());
        }
    }
    Ok(())
}

fn validate_scalar_type(
    type_id: &CanonicalTypeId,
    value: &ScalarValue,
) -> Result<(), PluginContractError> {
    let valid = match type_id.to_string().as_str() {
        "sim:boolean@1" => matches!(value, ScalarValue::Boolean(_)),
        "sim:integer@1" => matches!(value, ScalarValue::Integer(_)),
        "sim:float@1" => matches!(value, ScalarValue::Float(_)),
        "sim:string@1" => matches!(value, ScalarValue::String(_)),
        "sim:array@1" => matches!(value, ScalarValue::List(_)),
        "sim:dictionary@1" => matches!(value, ScalarValue::Record(_)),
        "sim:float-list@1" => matches!(
            value,
            ScalarValue::List(values)
                if values.iter().all(|value| matches!(value, ScalarValue::Float(_)))
        ),
        _ => true,
    };
    if !valid {
        return Err(PluginContractError::InvalidValue(format!(
            "scalar representation does not match type `{type_id}`"
        )));
    }
    Ok(())
}

fn valid_dotted_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value.split('.').all(|segment| {
            !segment.is_empty()
                && segment.len() <= 64
                && segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
                && segment
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphanumeric)
        })
}

fn valid_authority_identifier(value: &str) -> bool {
    if value.is_empty() || value.len() > 256 || !value.is_ascii() {
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

fn valid_route_path(path: &str) -> bool {
    path.starts_with('/')
        && path.len() <= 1_024
        && !path.contains('\0')
        && !path.contains('\\')
        && !path
            .split('/')
            .any(|segment| segment == ".." || segment == ".")
}

#[derive(Default)]
struct CanonicalWriter {
    bytes: Vec<u8>,
    maximum_bytes: Option<usize>,
    overflowed: bool,
}

impl CanonicalWriter {
    fn with_limit(maximum_bytes: usize) -> Self {
        Self {
            bytes: Vec::new(),
            maximum_bytes: Some(maximum_bytes),
            overflowed: false,
        }
    }

    fn raw(&mut self, value: &[u8]) {
        if self.overflowed {
            return;
        }
        let Some(new_length) = self.bytes.len().checked_add(value.len()) else {
            self.overflowed = true;
            return;
        };
        if self
            .maximum_bytes
            .is_some_and(|maximum_bytes| new_length > maximum_bytes)
        {
            self.overflowed = true;
            return;
        }
        self.bytes.extend_from_slice(value);
    }

    fn byte(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(u64::try_from(value).unwrap_or(u64::MAX));
    }

    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }

    fn optional_string(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.byte(1);
                self.string(value);
            }
            None => self.byte(0),
        }
    }

    fn strings(&mut self, values: &[String]) {
        self.usize(values.len());
        for value in values {
            self.string(value);
        }
    }

    fn version(&mut self, version: ApiVersion) {
        self.u16(version.major);
        self.u16(version.minor);
        self.u16(version.patch);
    }

    fn scalar(&mut self, value: Option<&ScalarValue>) {
        match value {
            None => self.byte(0),
            Some(ScalarValue::Null) => self.byte(1),
            Some(ScalarValue::Boolean(value)) => {
                self.byte(2);
                self.byte(u8::from(*value));
            }
            Some(ScalarValue::Integer(value)) => {
                self.byte(3);
                self.raw(&value.to_le_bytes());
            }
            Some(ScalarValue::Float(value)) => {
                self.byte(4);
                self.raw(&value.to_bits().to_le_bytes());
            }
            Some(ScalarValue::String(value)) => {
                self.byte(5);
                self.string(value);
            }
            Some(ScalarValue::Bytes(value)) => {
                self.byte(6);
                self.usize(value.len());
                self.raw(value);
            }
            Some(ScalarValue::List(values)) => {
                self.byte(7);
                self.usize(values.len());
                for value in values {
                    self.scalar(Some(value));
                }
            }
            Some(ScalarValue::Record(values)) => {
                self.byte(8);
                self.usize(values.len());
                for (key, value) in values {
                    self.string(key);
                    self.scalar(Some(value));
                }
            }
        }
    }

    fn plugin_value_representation(&mut self, value: &PluginValueRepresentation) {
        match value {
            PluginValueRepresentation::Scalar(value) => {
                self.byte(0);
                self.scalar(Some(value));
            }
            PluginValueRepresentation::Tensor(value) => {
                self.byte(1);
                self.tensor_descriptor(&value.descriptor);
                self.u64(value.byte_length);
                self.string(&value.digest);
            }
            PluginValueRepresentation::Artifact(value) => {
                self.byte(2);
                self.string(&value.namespace);
                self.string(&value.identifier);
                self.u64(value.byte_length);
                self.string(&value.digest);
            }
            PluginValueRepresentation::Model(value) => {
                self.byte(3);
                self.string(&value.identifier);
                self.string(&value.format);
                self.string(&value.digest);
            }
        }
    }

    fn tensor_descriptor(&mut self, descriptor: &TensorDescriptor) {
        self.usize(descriptor.shape().len());
        for dimension in descriptor.shape() {
            self.u64(*dimension);
        }
        self.usize(descriptor.strides().len());
        for stride in descriptor.strides() {
            self.raw(&stride.to_le_bytes());
        }
        self.u64(descriptor.offset_elements());
        self.byte(match descriptor.dtype() {
            DType::F64 => 0,
            DType::F32 => 1,
            DType::F16 => 2,
            DType::Bf16 => 3,
            DType::I64 => 4,
            DType::I32 => 5,
            DType::I16 => 6,
            DType::I8 => 7,
            DType::U64 => 8,
            DType::U32 => 9,
            DType::U16 => 10,
            DType::U8 => 11,
            DType::Bool => 12,
            DType::Complex64 => 13,
            DType::Complex128 => 14,
            DType::Float8E4m3Fn => 15,
            DType::Float8E5m2 => 16,
            DType::Float8E4m3Fnuz => 17,
            DType::Float8E5m2Fnuz => 18,
            DType::Float8E8m0Fnu => 19,
        });
        self.byte(match descriptor.layout() {
            Layout::Contiguous => 0,
            Layout::ChannelsLast => 1,
            Layout::ChannelsLast3d => 2,
            Layout::Strided => 3,
        });
        self.byte(match descriptor.device().kind() {
            DeviceKind::Cpu => 0,
            DeviceKind::Cuda => 1,
            DeviceKind::Rocm => 2,
            DeviceKind::Metal => 3,
            DeviceKind::DirectMl => 4,
            DeviceKind::Xpu => 5,
            DeviceKind::Npu => 6,
            DeviceKind::Mlu => 7,
            DeviceKind::CoreX => 8,
        });
        self.u32(descriptor.device().ordinal());
        self.u64(descriptor.stream().get());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn overflowed(&self) -> bool {
        self.overflowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_v2_schema_definition<T: Serialize>(
        schema: &serde_json::Value,
        v1_schema: &serde_json::Value,
        definition: &str,
        value: &T,
    ) -> Result<(), Box<dyn Error>> {
        let definition_schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": format!("#/$defs/{definition}"),
            "$defs": schema["$defs"].clone(),
        });
        let mut v1_schema = v1_schema.clone();
        v1_schema["$id"] = serde_json::Value::String(
            "https://zed.dev/schema/sim-plugin-manifest-v1.json".to_owned(),
        );
        let validator = jsonschema::options()
            .with_resource(
                "https://zed.dev/schema/sim-plugin-manifest-v1.json",
                jsonschema::Resource::from_contents(v1_schema),
            )
            .build(&definition_schema)?;
        let serialized = serde_json::to_value(value)?;
        if !validator.is_valid(&serialized) {
            return Err(format!(
                "serialized Rust value does not match v2 schema definition `{definition}`: {serialized}"
            )
            .into());
        }
        Ok(())
    }

    fn manifest_fixture(registry: &TypeRegistry) -> Result<PluginManifest, Box<dyn Error>> {
        let string_type = registry.resolve("STRING")?.clone();
        Ok(PluginManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            identifier: "test.typed-plugin".to_owned(),
            plugin_version: ApiVersion::new(1, 2, 3),
            api: ApiRequirement {
                major: 1,
                minimum_minor: 0,
                maximum_minor: 0,
                required_features: vec!["ports.list".to_owned()],
            },
            digest_sha256: "1".repeat(64),
            signature: ManifestSignature {
                algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                key_id: "test-key".to_owned(),
                value: "2".repeat(ED25519_SIGNATURE_BYTES * 2),
            },
            provenance: ManifestProvenance {
                source: "test fixture".to_owned(),
                publisher: "Zed tests".to_owned(),
                registry: None,
            },
            provider_binding: None,
            nodes: vec![PluginNode {
                id: "echo".to_owned(),
                version: ApiVersion::new(1, 0, 0),
                display_name: "Echo".to_owned(),
                category: "test".to_owned(),
                ports: vec![
                    PluginPort {
                        id: "text-in".to_owned(),
                        name: "Text input".to_owned(),
                        direction: PortDirection::Input,
                        type_id: string_type.clone(),
                        cardinality: PortCardinality::Singular,
                        presence: PortPresence::Optional,
                        hidden: false,
                        lazy: false,
                        default: Some(ScalarValue::String("default".to_owned())),
                        serialization: PortSerialization::Inline,
                        accepted_legacy_names: vec!["legacy-text".to_owned()],
                    },
                    PluginPort {
                        id: "text-out".to_owned(),
                        name: "Text output".to_owned(),
                        direction: PortDirection::Output,
                        type_id: string_type,
                        cardinality: PortCardinality::Singular,
                        presence: PortPresence::Optional,
                        hidden: false,
                        lazy: false,
                        default: None,
                        serialization: PortSerialization::Inline,
                        accepted_legacy_names: Vec::new(),
                    },
                ],
                determinism: DeterminismPolicy::Deterministic,
                cache: CachePolicy::InputIdentity,
                effects: EffectPolicy::Pure,
            }],
            capabilities: Vec::new(),
            ui: Vec::new(),
            routes: Vec::new(),
            legacy_mappings: vec![LegacyMapping {
                legacy_identifier: "LegacyEcho".to_owned(),
                node_id: "echo".to_owned(),
                node_version: ApiVersion::new(1, 0, 0),
                legacy_widget_names: vec!["legacy-text".to_owned()],
                input_translations: vec![LegacyInputTranslation::Rename {
                    target_port_id: "text-in".to_owned(),
                    legacy_input_id: "legacy-text".to_owned(),
                }],
                output_translations: vec![LegacyOutputTranslation {
                    target_port_index: 0,
                    legacy_output_index: 2,
                }],
            }],
        })
    }

    fn provider_streaming_contract_fixture() -> ProviderStreamingContractV2 {
        ProviderStreamingContractV2 {
            methods: vec![ProviderHttpMethodV2::Get, ProviderHttpMethodV2::Post],
            maximum_headers: 8,
            maximum_header_bytes: 1_024,
            maximum_request_body_bytes: 16,
            maximum_response_body_bytes: 64,
            maximum_chunk_bytes: 16,
            maximum_ndjson_line_bytes: 16,
            maximum_wait_milliseconds: 1_000,
            maximum_uploads: 2,
            maximum_upload_body_bytes: 8,
            maximum_cost_requests: 2,
            maximum_progress_total: 100,
            uploads: true,
            cost_requests: true,
        }
    }

    fn provider_manifest_v2_fixture(
        registry: &TypeRegistry,
    ) -> Result<ProviderPluginManifestV2, Box<dyn Error>> {
        let mut manifest = manifest_fixture(registry)?;
        manifest.api.required_features.extend([
            PROVIDER_BINDING_API_FEATURE.to_owned(),
            PROVIDER_STREAMING_API_FEATURE_V2.to_owned(),
        ]);
        manifest.nodes[0].determinism = DeterminismPolicy::External;
        manifest.nodes[0].cache = CachePolicy::Never;
        manifest.nodes[0].effects = EffectPolicy::Provider;
        manifest.capabilities = vec![
            CapabilityRequest {
                kind: CapabilityKind::ProviderUpload,
                scope: "provider.upload".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 2,
                    maximum_request_bytes: 8,
                    maximum_response_bytes: 1,
                    maximum_total_bytes: 8,
                    maximum_handles: 2,
                    timeout_milliseconds: 1_000,
                },
            },
            CapabilityRequest {
                kind: CapabilityKind::ProviderCost,
                scope: "provider.cost".to_owned(),
                quota: CapabilityQuota {
                    maximum_operations: 2,
                    maximum_request_bytes: MAX_PROVIDER_COST_REQUEST_BYTES,
                    maximum_response_bytes: MAX_PROVIDER_COST_RESPONSE_BYTES,
                    maximum_total_bytes: 2
                        * (MAX_PROVIDER_COST_REQUEST_BYTES + MAX_PROVIDER_COST_RESPONSE_BYTES),
                    maximum_handles: 1,
                    timeout_milliseconds: 1_000,
                },
            },
        ];
        let mut provider_binding = ProviderBindingSet {
            schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: manifest.identifier.clone(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![ProviderBindingClaim {
                feature_id: "COMFY-NODE-0001".to_owned(),
                node_id: manifest.nodes[0].id.clone(),
                contract_sha256: "3".repeat(64),
                transport_schema: "zed:sim-provider-transport@1".parse()?,
                materializer_schema: "zed:sim-provider-materializer@1".parse()?,
            }],
        };
        provider_binding.bindings_sha256 = provider_binding.canonical_bindings_sha256()?;
        manifest.provider_binding = Some(provider_binding);
        Ok(ProviderPluginManifestV2 {
            schema_version: PROVIDER_MANIFEST_SCHEMA_VERSION_V2,
            component_world: PROVIDER_COMPONENT_WORLD_V2.to_owned(),
            manifest,
            streaming: provider_streaming_contract_fixture(),
            signature: ManifestSignature {
                algorithm: PLUGIN_SIGNATURE_ALGORITHM.to_owned(),
                key_id: "test-key".to_owned(),
                value: "4".repeat(ED25519_SIGNATURE_BYTES * 2),
            },
        })
    }

    #[test]
    fn semantic_versions_are_strict() -> Result<(), Box<dyn Error>> {
        assert_eq!("1.2.3".parse::<ApiVersion>()?, ApiVersion::new(1, 2, 3));
        for invalid in ["1", "1.2", "1.2.3.4", "01.2.3", "1.x.3"] {
            assert!(invalid.parse::<ApiVersion>().is_err());
        }
        Ok(())
    }

    #[test]
    fn sdk_signing_keys_are_private_deterministic_and_ed25519_only() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let manifest = manifest_fixture(&registry)?;
        let key = PluginSigningKey::new("test-key", b"0123456789abcdef0123456789abcdef")?;
        let signature = key.sign_manifest(&manifest)?;
        assert_eq!(signature.len(), ED25519_SIGNATURE_BYTES * 2);
        assert_eq!(signature, key.sign_manifest(&manifest)?);
        assert_eq!(
            key.verification_key_bytes()?.len(),
            ED25519_PUBLIC_KEY_BYTES
        );
        assert!(!format!("{key:?}").contains("0123456789abcdef"));
        assert!(PluginSigningKey::new("test-key", [0_u8; 31]).is_err());
        for invalid_key_id in ["Test-Key", "test-key-", "test..key"] {
            assert!(PluginSigningKey::new(invalid_key_id, [0_u8; 32]).is_err());
        }
        Ok(())
    }

    #[test]
    fn manifest_authority_identifiers_match_runtime_grammar() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        for invalid_identifier in ["Test.plugin", "test.plugin-", "test..plugin"] {
            let mut manifest = manifest_fixture(&registry)?;
            manifest.identifier = invalid_identifier.to_owned();
            assert!(matches!(
                manifest.validate(&registry),
                Err(PluginContractError::InvalidIdentifier(identifier))
                    if identifier == invalid_identifier
            ));
        }
        for invalid_key_id in ["Test-key", "test-key_", "test--key"] {
            let mut manifest = manifest_fixture(&registry)?;
            manifest.signature.key_id = invalid_key_id.to_owned();
            assert_eq!(
                manifest.validate(&registry),
                Err(PluginContractError::InvalidSignatureMetadata)
            );
        }
        Ok(())
    }

    #[test]
    fn api_negotiation_requires_major_minor_and_features() -> Result<(), Box<dyn Error>> {
        let requirement = ApiRequirement {
            major: 1,
            minimum_minor: 0,
            maximum_minor: 2,
            required_features: vec!["ports.list".to_owned()],
        };
        let features = BTreeSet::from(["ports.list".to_owned()]);
        let negotiated = requirement.negotiate(ApiVersion::new(1, 1, 7), &features)?;
        assert_eq!(negotiated.version, ApiVersion::new(1, 1, 7));
        assert!(
            requirement
                .negotiate(ApiVersion::new(2, 0, 0), &features)
                .is_err()
        );
        assert!(
            requirement
                .negotiate(ApiVersion::new(1, 1, 0), &BTreeSet::new())
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn bounded_canonical_writer_stops_before_excess_allocation() {
        let mut writer = CanonicalWriter::with_limit(4);
        writer.raw(&[1, 2, 3, 4]);
        assert!(!writer.overflowed());
        writer.raw(&[5]);
        assert!(writer.overflowed());
        assert_eq!(writer.finish(), [1, 2, 3, 4]);
    }

    #[test]
    fn provider_result_receipt_sets_are_canonical_ordered_and_bounded() -> Result<(), Box<dyn Error>>
    {
        let receipts = ProviderResultReceiptSet::new(vec![
            b"first-sealed-receipt".to_vec(),
            b"second-sealed-receipt".to_vec(),
        ])?;
        let encoded = receipts.to_bytes()?;
        assert_eq!(ProviderResultReceiptSet::from_bytes(&encoded)?, receipts);

        let mut trailing = encoded.clone();
        trailing.push(0);
        assert_eq!(
            ProviderResultReceiptSet::from_bytes(&trailing),
            Err(PluginContractError::InvalidProviderReceiptSet)
        );
        let mut wrong_version = encoded;
        wrong_version[PROVIDER_RESULT_RECEIPT_SET_DOMAIN.len()] = 2;
        assert_eq!(
            ProviderResultReceiptSet::from_bytes(&wrong_version),
            Err(PluginContractError::InvalidProviderReceiptSet)
        );
        assert_eq!(
            ProviderResultReceiptSet::new(Vec::new()),
            Err(PluginContractError::InvalidProviderReceiptSet)
        );
        assert_eq!(
            ProviderResultReceiptSet::new(vec![b"duplicate".to_vec(), b"duplicate".to_vec()]),
            Err(PluginContractError::InvalidProviderReceiptSet)
        );
        assert_eq!(
            ProviderResultReceiptSet::new(vec![vec![b'x'; MAX_PROVIDER_RESULT_RECEIPT_BYTES + 1]]),
            Err(PluginContractError::InvalidProviderReceiptSet)
        );
        Ok(())
    }

    #[test]
    fn typed_values_bind_exact_ids_and_checked_representations() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let string_type = registry.resolve("STRING")?.clone();
        let float_type = registry.resolve("FLOAT")?.clone();
        let image_type = registry.resolve("IMAGE")?.clone();

        let value = PluginValue::scalar(
            string_type.clone(),
            ScalarValue::String("hello".to_owned()),
            &registry,
        )?;
        assert_eq!(value.type_id(), &string_type);
        assert_eq!(value.family(), ValueFamily::Scalar);
        assert_eq!(
            PluginValue::from_abi_bytes(&value.abi_bytes()?, &registry)?,
            value
        );
        assert!(
            PluginValue::scalar(
                float_type,
                ScalarValue::String("not-a-float".to_owned()),
                &registry,
            )
            .is_err()
        );
        assert!(
            PluginValue::scalar(
                image_type,
                ScalarValue::String("not-a-tensor".to_owned()),
                &registry,
            )
            .is_err()
        );

        let mut unknown = serde_json::to_value(&value)?;
        unknown
            .as_object_mut()
            .ok_or("plugin value was not a JSON object")?
            .insert("unknown".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<PluginValue>(unknown).is_err());
        assert!(
            TensorValue::new(
                TensorDescriptor::contiguous(
                    vec![1],
                    DType::F32,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                3,
                "1".repeat(64),
            )
            .is_err()
        );
        assert!(ArtifactValue::new("input", "../escape", 1, "1".repeat(64)).is_ok());
        assert!(ArtifactValue::new(" input", "asset", 1, "1".repeat(64)).is_err());

        let valid_tensor = TensorValue::new(
            TensorDescriptor::contiguous(vec![1], DType::F32, DeviceId::CPU, StreamId::DEFAULT)?,
            4,
            "1".repeat(64),
        )?;
        let mut invalid_tensor = serde_json::to_value(valid_tensor)?;
        invalid_tensor["byte_length"] = serde_json::json!(3);
        assert!(serde_json::from_value::<TensorValue>(invalid_tensor).is_err());

        let mut invalid_artifact =
            serde_json::to_value(ArtifactValue::new("input", "asset", 1, "1".repeat(64))?)?;
        invalid_artifact["digest"] = serde_json::json!("not-a-digest");
        assert!(serde_json::from_value::<ArtifactValue>(invalid_artifact).is_err());

        let mut invalid_model =
            serde_json::to_value(ModelValue::new("model", "safetensors", "1".repeat(64))?)?;
        invalid_model["unknown"] = serde_json::json!(true);
        assert!(serde_json::from_value::<ModelValue>(invalid_model).is_err());
        Ok(())
    }

    #[test]
    fn legacy_descriptor_adapter_uses_canonical_family_serialization() -> Result<(), Box<dyn Error>>
    {
        let registry = TypeRegistry::built_in()?;
        let descriptor = NodeDescriptor {
            type_name: "legacy-node".to_owned(),
            display_name: "Legacy node".to_owned(),
            inputs: [
                ("text", "String"),
                ("image", "Image"),
                ("asset", "File3DAny"),
                ("model", "Model"),
            ]
            .into_iter()
            .map(|(name, type_name)| sim_nodes::PortDescriptor {
                name: name.to_owned(),
                type_name: type_name.to_owned(),
                required: true,
            })
            .collect(),
            outputs: Vec::new(),
        };
        let node = PluginNode::try_from_legacy_descriptor(&descriptor, &registry)?;
        assert_eq!(
            node.ports
                .iter()
                .map(|port| port.serialization)
                .collect::<Vec<_>>(),
            [
                PortSerialization::Inline,
                PortSerialization::Handle,
                PortSerialization::ArtifactReference,
                PortSerialization::Handle,
            ]
        );
        Ok(())
    }

    #[test]
    fn manifests_reject_port_legacy_and_schema_drift() -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let manifest = manifest_fixture(&registry)?;
        manifest.validate(&registry)?;
        manifest
            .component_projection()
            .validate_for_manifest(&manifest, &registry)?;

        let mut duplicate_direction = manifest.clone();
        duplicate_direction.nodes[0].ports[1].id = "text-in".to_owned();
        assert!(matches!(
            duplicate_direction.validate(&registry),
            Err(PluginContractError::DuplicateOrInvalidPort { .. })
        ));

        let mut alias_collision = manifest.clone();
        alias_collision.nodes[0].ports[1]
            .accepted_legacy_names
            .push("legacy-text".to_owned());
        assert!(matches!(
            alias_collision.validate(&registry),
            Err(PluginContractError::DuplicatePortAlias(_))
        ));

        let mut identifier_alias_collision = manifest.clone();
        identifier_alias_collision.nodes[0].ports[1]
            .accepted_legacy_names
            .push("text-in".to_owned());
        assert!(matches!(
            identifier_alias_collision.validate(&registry),
            Err(PluginContractError::DuplicatePortAlias(_))
        ));

        let mut wrong_serialization = manifest.clone();
        wrong_serialization.nodes[0].ports[0].serialization = PortSerialization::Handle;
        assert!(matches!(
            wrong_serialization.validate(&registry),
            Err(PluginContractError::InvalidPortSerialization { .. })
        ));

        let mut wrong_legacy_version = manifest.clone();
        wrong_legacy_version.legacy_mappings[0].node_version = ApiVersion::new(1, 0, 1);
        assert!(matches!(
            wrong_legacy_version.validate(&registry),
            Err(PluginContractError::DuplicateOrInvalidLegacyMapping(_))
        ));

        let mut constant_translation = manifest.clone();
        constant_translation.legacy_mappings[0].input_translations =
            vec![LegacyInputTranslation::Constant {
                target_port_id: "text-in".to_owned(),
                value: ScalarValue::String("fixed".to_owned()),
            }];
        assert!(constant_translation.validate(&registry).is_ok());

        let mut wrong_constant_type = constant_translation.clone();
        wrong_constant_type.legacy_mappings[0].input_translations =
            vec![LegacyInputTranslation::Constant {
                target_port_id: "text-in".to_owned(),
                value: ScalarValue::Integer(7),
            }];
        assert!(matches!(
            wrong_constant_type.validate(&registry),
            Err(PluginContractError::InvalidLegacyTranslation(mapping))
                if mapping == "LegacyEcho"
        ));

        let mut unknown_translation_target = manifest.clone();
        unknown_translation_target.legacy_mappings[0].input_translations =
            vec![LegacyInputTranslation::Rename {
                target_port_id: "missing-port".to_owned(),
                legacy_input_id: "legacy-text".to_owned(),
            }];
        assert!(matches!(
            unknown_translation_target.validate(&registry),
            Err(PluginContractError::InvalidLegacyTranslation(mapping))
                if mapping == "LegacyEcho"
        ));

        let mut invalid_output_translation = manifest.clone();
        invalid_output_translation.legacy_mappings[0].output_translations =
            vec![LegacyOutputTranslation {
                target_port_index: 8,
                legacy_output_index: 0,
            }];
        assert!(matches!(
            invalid_output_translation.validate(&registry),
            Err(PluginContractError::InvalidLegacyTranslation(mapping))
                if mapping == "LegacyEcho"
        ));

        let mut duplicate_output_target = manifest.clone();
        duplicate_output_target.legacy_mappings[0].output_translations = vec![
            LegacyOutputTranslation {
                target_port_index: 0,
                legacy_output_index: 0,
            },
            LegacyOutputTranslation {
                target_port_index: 0,
                legacy_output_index: 1,
            },
        ];
        assert!(matches!(
            duplicate_output_target.validate(&registry),
            Err(PluginContractError::InvalidLegacyTranslation(mapping))
                if mapping == "LegacyEcho"
        ));

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v1.schema.json"))?;
        let validator = jsonschema::validator_for(&schema)?;
        let manifest_json = serde_json::to_value(&manifest)?;
        assert!(validator.is_valid(&manifest_json));
        let component_projection_json = serde_json::to_value(manifest.component_projection())?;
        let component_mapping = component_projection_json
            .get("legacy_mappings")
            .and_then(serde_json::Value::as_array)
            .and_then(|mappings| mappings.first())
            .and_then(serde_json::Value::as_object)
            .ok_or("component legacy mapping projection is absent")?;
        assert_eq!(component_mapping.len(), 3);
        assert!(!component_mapping.contains_key("legacy_widget_names"));
        assert!(!component_mapping.contains_key("input_translations"));
        assert!(!component_mapping.contains_key("output_translations"));
        let mut unknown_manifest = manifest_json;
        unknown_manifest
            .as_object_mut()
            .ok_or("manifest was not a JSON object")?
            .insert(
                "ambient_authority".to_owned(),
                serde_json::Value::Bool(true),
            );
        assert!(!validator.is_valid(&unknown_manifest));
        assert!(serde_json::from_value::<PluginManifest>(unknown_manifest).is_err());

        for invalid_type_id in [
            format!("{}:value@1", "a".repeat(65)),
            "sim:value@65536".to_owned(),
        ] {
            let mut invalid_manifest = serde_json::to_value(&manifest)?;
            invalid_manifest["nodes"][0]["ports"][0]["type_id"] =
                serde_json::Value::String(invalid_type_id);
            assert!(!validator.is_valid(&invalid_manifest));
            assert!(serde_json::from_value::<PluginManifest>(invalid_manifest).is_err());
        }
        Ok(())
    }

    #[test]
    fn signed_provider_binding_sets_are_ordered_bounded_and_node_exact()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let mut manifest = manifest_fixture(&registry)?;
        let legacy_signing_payload = manifest.signing_payload();
        let mut legacy_writer = CanonicalWriter::default();
        legacy_writer.u16(manifest.schema_version);
        legacy_writer.string(&manifest.identifier);
        legacy_writer.version(manifest.plugin_version);
        legacy_writer.u16(manifest.api.major);
        legacy_writer.u16(manifest.api.minimum_minor);
        legacy_writer.u16(manifest.api.maximum_minor);
        legacy_writer.strings(&manifest.api.required_features);
        legacy_writer.string(&manifest.digest_sha256);
        legacy_writer.string(&manifest.signature.algorithm);
        legacy_writer.string(&manifest.signature.key_id);
        legacy_writer.string(&manifest.provenance.source);
        legacy_writer.string(&manifest.provenance.publisher);
        legacy_writer.optional_string(manifest.provenance.registry.as_deref());
        manifest.write_declarations(&mut legacy_writer);
        assert_eq!(legacy_signing_payload, legacy_writer.finish());
        manifest
            .api
            .required_features
            .push(PROVIDER_BINDING_API_FEATURE.to_owned());
        manifest.nodes[0].determinism = DeterminismPolicy::External;
        manifest.nodes[0].cache = CachePolicy::Never;
        manifest.nodes[0].effects = EffectPolicy::Provider;
        let binding = ProviderBindingClaim {
            feature_id: "COMFY-NODE-0001".to_owned(),
            node_id: manifest.nodes[0].id.clone(),
            contract_sha256: "3".repeat(64),
            transport_schema: "zed:sim-provider-transport@1".parse()?,
            materializer_schema: "zed:sim-provider-materializer@1".parse()?,
        };
        let mut provider_binding = ProviderBindingSet {
            schema_version: PROVIDER_BINDING_SCHEMA_VERSION,
            implementation_namespace: manifest.identifier.clone(),
            bindings_sha256: "0".repeat(64),
            bindings: vec![binding],
        };
        provider_binding.bindings_sha256 = provider_binding.canonical_bindings_sha256()?;
        manifest.provider_binding = Some(provider_binding);
        manifest.validate(&registry)?;
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v1.schema.json"))?;
        let validator = jsonschema::validator_for(&schema)?;
        assert!(validator.is_valid(&serde_json::to_value(&manifest)?));
        assert_eq!(
            manifest.component_projection().component_world,
            PROVIDER_COMPONENT_WORLD
        );
        let canonical = manifest
            .provider_binding
            .as_ref()
            .ok_or("provider binding disappeared")?
            .canonical_binding_bytes()?;
        assert_eq!(
            canonical,
            manifest
                .provider_binding
                .as_ref()
                .ok_or("provider binding disappeared")?
                .canonical_binding_bytes()?
        );
        let signed = manifest.signing_payload();
        let mut changed = manifest.clone();
        changed
            .provider_binding
            .as_mut()
            .ok_or("provider binding disappeared")?
            .bindings[0]
            .contract_sha256 = "5".repeat(64);
        assert_ne!(signed, changed.signing_payload());

        let mut forged_set_digest = manifest.clone();
        forged_set_digest
            .provider_binding
            .as_mut()
            .ok_or("provider binding disappeared")?
            .bindings_sha256 = "4".repeat(64);
        assert_eq!(
            forged_set_digest.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );

        let mut wrong_policy = manifest.clone();
        wrong_policy.nodes[0].cache = CachePolicy::InputIdentity;
        assert_eq!(
            wrong_policy.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        let mut unknown_node = manifest.clone();
        unknown_node
            .provider_binding
            .as_mut()
            .ok_or("provider binding disappeared")?
            .bindings[0]
            .node_id = "missing".to_owned();
        assert_eq!(
            unknown_node.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        let mut duplicate = manifest.clone();
        let binding = duplicate
            .provider_binding
            .as_ref()
            .and_then(|set| set.bindings.first())
            .cloned()
            .ok_or("provider binding disappeared")?;
        duplicate
            .provider_binding
            .as_mut()
            .ok_or("provider binding disappeared")?
            .bindings
            .push(binding);
        assert_eq!(
            duplicate.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        let mut wrong_namespace = manifest.clone();
        wrong_namespace
            .provider_binding
            .as_mut()
            .ok_or("provider binding disappeared")?
            .implementation_namespace = "provider.other".to_owned();
        assert_eq!(
            wrong_namespace.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        let mut missing_feature = manifest.clone();
        missing_feature
            .api
            .required_features
            .retain(|feature| feature != PROVIDER_BINDING_API_FEATURE);
        assert_eq!(
            missing_feature.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        let mut wrong_determinism = manifest.clone();
        wrong_determinism.nodes[0].determinism = DeterminismPolicy::Deterministic;
        assert_eq!(
            wrong_determinism.validate(&registry),
            Err(PluginContractError::InvalidProviderBindingSet)
        );
        Ok(())
    }

    #[test]
    fn wit_exposes_typed_manifest_values_and_presence() {
        let wit = include_str!("../wit/sim-plugin.wit");
        for contract in [
            "record manifest-projection",
            "record capability-request",
            "record legacy-mapping",
            "record scalar-value",
            "default: option<scalar-value>",
            "record encoded-value",
            "record provider-binding-claim",
            "record provider-binding-set",
            "record provider-materialized-output",
            "record provider-invocation-response",
            "interface provider-binding",
            "world sim-provider-plugin",
            "read-scalar-input",
            "read-handle: func(handle: value-handle) -> result<encoded-value, invocation-error>",
            "create-output-value",
            "finish-output: func(port-id: string, present: bool)",
            "invocation-quota-exceeded(string)",
            "manifest: func() -> manifest-projection",
        ] {
            assert!(wit.contains(contract), "WIT is missing `{contract}`");
        }
        assert!(!wit.contains("manifest: func() -> list<u8>"));
        assert!(!wit.contains("legacy-widget-names"));
        assert!(!wit.contains("input-translations"));
        assert!(!wit.contains("output-translations"));
    }

    #[test]
    fn provider_streaming_v2_preserves_v1_and_has_typed_schema_parity() -> Result<(), Box<dyn Error>>
    {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../wit/sim-plugin.wit"))
            ),
            "2703627091bb596ed4973903311a91b9c4663ea9da72d854e0a221ad4c3f4866"
        );
        assert_eq!(
            include_bytes!("../wit/sim-plugin.wit"),
            include_bytes!("../wit/provider-v2/deps/sim-plugin/sim-plugin.wit")
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../schema/plugin-manifest-v1.schema.json"))
            ),
            "0b7439ac09d1f1ef8341483a2857da89501135b300204472e0cc997e9bb02a8e"
        );

        let wit = include_str!("../wit/provider-v2/sim-provider-plugin.wit");
        for contract in [
            "record invocation-context",
            "record stream-handle",
            "maximum-upload-body-bytes: u64",
            "record request-head",
            "after-sequence: option<u64>",
            "variant response-frame-event",
            "record response-frame",
            "variant wait-outcome",
            "record upload-request",
            "record upload-completion-request",
            "record upload-completion",
            "record cost-request",
            "record cost-response",
            "record invocation-result",
            "record manifest-v2",
            "manifest: manifest-projection",
            "provider-binding: provider-binding-set",
            "start-request: func(context: invocation-context",
            "complete-upload: func(request: upload-completion-request) -> result<upload-completion, stream-error>",
            "request-cost: func(request: cost-request) -> result<cost-response, stream-error>",
            "invoke: func(context: invocation-context, node-id: string) -> result<invocation-result, stream-error>",
        ] {
            assert!(wit.contains(contract), "v2 WIT is missing `{contract}`");
        }
        let request_head = wit
            .split_once("record request-head {")
            .and_then(|(_, remainder)| remainder.split_once('}'))
            .map(|(record, _)| record)
            .ok_or("v2 request-head record disappeared")?;
        assert!(!request_head.contains("handle:"));
        assert!(!request_head.contains("provider:"));
        let mut previous_field = 0;
        for field in [
            "endpoint: string",
            "secret-id: option<string>",
            "method: http-method",
            "headers: list<header>",
            "declared-body-bytes: option<u64>",
        ] {
            let position = request_head
                .find(field)
                .ok_or_else(|| format!("v2 request-head is missing `{field}`"))?;
            assert!(position >= previous_field);
            previous_field = position;
        }
        let stream_error = wit
            .split_once("enum stream-error {")
            .and_then(|(_, remainder)| remainder.split_once('}'))
            .map(|(enumeration, _)| enumeration)
            .ok_or("v2 stream-error enum disappeared")?;
        let invocation_result_position = stream_error
            .find("invalid-invocation-result")
            .ok_or("v2 stream-error lost invalid-invocation-result")?;
        let request_authority_position = stream_error
            .find("invalid-request-authority")
            .ok_or("v2 stream-error lost invalid-request-authority")?;
        assert!(invocation_result_position < request_authority_position);
        assert!(!wit.contains("manifest-v1: list<u8>"));
        assert!(!wit.contains("invoke: func(node-id: string) -> result<list<u8>"));
        for forbidden in ["native-handle", "secret-bytes", "filesystem-path"] {
            assert!(!wit.contains(forbidden));
        }

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v2.schema.json"))?;
        for definition in [
            "streaming_contract",
            "invocation_context",
            "stream_handle",
            "request_head",
            "request_chunk",
            "response_frame",
            "wait_request",
            "wait_outcome",
            "upload_request",
            "cost_request",
            "cost_response",
            "progress",
            "invocation_result",
            "stream_error",
            "component_manifest_projection_v2",
        ] {
            assert!(
                schema["$defs"].get(definition).is_some(),
                "v2 schema is missing `{definition}`"
            );
        }
        let contract_schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/streaming_contract",
            "$defs": {
                "streaming_contract": schema["$defs"]["streaming_contract"].clone(),
                "http_method": schema["$defs"]["http_method"].clone(),
            },
        });
        let validator = jsonschema::validator_for(&contract_schema)?;
        let contract = provider_streaming_contract_fixture();
        contract.validate()?;
        let contract_value = serde_json::to_value(&contract)?;
        assert!(validator.is_valid(&contract_value));
        let mut unknown_contract = contract_value;
        unknown_contract
            .as_object_mut()
            .ok_or("streaming contract was not an object")?
            .insert("ignored".to_owned(), serde_json::Value::Bool(true));
        assert!(!validator.is_valid(&unknown_contract));
        assert!(serde_json::from_value::<ProviderStreamingContractV2>(unknown_contract).is_err());

        let registry = TypeRegistry::built_in()?;
        let manifest = provider_manifest_v2_fixture(&registry)?;
        manifest.validate(&registry)?;
        let component_projection = manifest.component_projection()?;
        component_projection.validate_for_manifest(&manifest, &registry)?;
        let v1_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v1.schema.json"))?;

        let context = ProviderInvocationContextV2 {
            invocation: 7,
            generation: 3,
        };
        let handle = ProviderStreamHandleV2 {
            invocation: 7,
            slot: 1,
            generation: 3,
        };
        let response_head = ProviderResponseHeadV2 {
            status: 200,
            headers: vec![ProviderHeaderV2 {
                name: "X-Test".to_owned(),
                value: "value".to_owned(),
            }],
        };
        let chunks = [
            ProviderResponseChunkV2::Binary(vec![1]),
            ProviderResponseChunkV2::Text("text".to_owned()),
            ProviderResponseChunkV2::NdjsonLine("{\"ok\":true}".to_owned()),
        ];
        let terminals = [
            ProviderStreamTerminalV2::Completed { receipt: vec![1] },
            ProviderStreamTerminalV2::Failed {
                code: "provider.failed".to_owned(),
                message: "failed".to_owned(),
            },
            ProviderStreamTerminalV2::Cancelled,
        ];
        assert_v2_schema_definition(&schema, &v1_schema, "invocation_context", &context)?;
        assert_v2_schema_definition(&schema, &v1_schema, "stream_handle", &handle)?;
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "request_head",
            &ProviderRequestHeadV2 {
                endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
                secret_id: Some("provider-api-key".to_owned()),
                method: ProviderHttpMethodV2::Post,
                headers: vec![ProviderHeaderV2 {
                    name: "X-Test".to_owned(),
                    value: "value".to_owned(),
                }],
                declared_body_bytes: Some(1),
            },
        )?;
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "request_chunk",
            &ProviderRequestChunkV2 {
                handle,
                sequence: 0,
                bytes: vec![1],
                end: true,
            },
        )?;
        assert_v2_schema_definition(&schema, &v1_schema, "response_head", &response_head)?;
        for chunk in &chunks {
            assert_v2_schema_definition(&schema, &v1_schema, "response_chunk", chunk)?;
        }
        for terminal in &terminals {
            assert_v2_schema_definition(&schema, &v1_schema, "terminal", terminal)?;
        }
        let frame = ProviderResponseFrameV2 {
            handle,
            sequence: 0,
            event: ProviderResponseFrameEventV2::Head(response_head),
        };
        assert_v2_schema_definition(&schema, &v1_schema, "response_frame", &frame)?;
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "wait_request",
            &ProviderWaitRequestV2 {
                handle,
                after_sequence: Some(0),
                timeout_milliseconds: 1,
            },
        )?;
        for outcome in [
            ProviderWaitOutcomeV2::Frame(frame),
            ProviderWaitOutcomeV2::TimedOut,
            ProviderWaitOutcomeV2::Cancelled,
        ] {
            assert_v2_schema_definition(&schema, &v1_schema, "wait_outcome", &outcome)?;
        }
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "upload_request",
            &ProviderUploadRequestV2 {
                handle,
                port_id: "image".to_owned(),
                media_type: "image/png".to_owned(),
                byte_length: 1,
                content_sha256: "0".repeat(64),
            },
        )?;
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "cost_request",
            &ProviderCostRequestV2 {
                handle,
                operation: "generate".to_owned(),
                currency: "USD".to_owned(),
                maximum_microunits: 2,
            },
        )?;
        for response in [
            ProviderCostResponseV2 {
                accepted: true,
                approved_microunits: 1,
                receipt: vec![1],
            },
            ProviderCostResponseV2 {
                accepted: false,
                approved_microunits: 0,
                receipt: Vec::new(),
            },
        ] {
            assert_v2_schema_definition(&schema, &v1_schema, "cost_response", &response)?;
        }
        assert_v2_schema_definition(
            &schema,
            &v1_schema,
            "progress",
            &ProviderProgressV2 {
                handle,
                sequence: 0,
                completed: 1,
                total: 2,
                message: Some("running".to_owned()),
            },
        )?;
        let type_id = registry.resolve("STRING")?.clone();
        let encoded = PluginValue::scalar(
            type_id.clone(),
            ScalarValue::String("done".to_owned()),
            &registry,
        )?;
        let invocation_result = ProviderInvocationResultV2 {
            outputs: vec![ProviderMaterializedOutputV2 {
                port_id: "text".to_owned(),
                value: ProviderEncodedValueV2 {
                    type_id,
                    family: ValueFamily::Scalar,
                    abi_bytes: encoded.abi_bytes()?,
                },
            }],
            receipt: vec![1],
        };
        assert_v2_schema_definition(&schema, &v1_schema, "invocation_result", &invocation_result)?;
        for error in [
            ProviderStreamErrorV2::Cancelled,
            ProviderStreamErrorV2::HostFailure,
            ProviderStreamErrorV2::InvalidInvocationResult,
            ProviderStreamErrorV2::InvalidRequestAuthority,
        ] {
            assert_v2_schema_definition(&schema, &v1_schema, "stream_error", &error)?;
        }
        assert_eq!(
            schema["$defs"]["encoded_value"]["properties"]["abi_bytes"]["maxItems"].as_u64(),
            Some(MAX_PROVIDER_ENCODED_VALUE_BYTES)
        );
        let mut v1_schema_resource = v1_schema;
        v1_schema_resource["$id"] = serde_json::Value::String(
            "https://zed.dev/schema/sim-plugin-manifest-v1.json".to_owned(),
        );
        let root_validator = jsonschema::options()
            .with_resource(
                "https://zed.dev/schema/sim-plugin-manifest-v1.json",
                jsonschema::Resource::from_contents(v1_schema_resource.clone()),
            )
            .build(&schema)?;
        let manifest_value = serde_json::to_value(&manifest)?;
        let root_errors = root_validator
            .iter_errors(&manifest_value)
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        assert!(root_errors.is_empty(), "{root_errors:#?}");
        let component_schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/component_manifest_projection_v2",
            "$defs": schema["$defs"].clone(),
        });
        let component_validator = jsonschema::options()
            .with_resource(
                "https://zed.dev/schema/sim-plugin-manifest-v1.json",
                jsonschema::Resource::from_contents(v1_schema_resource),
            )
            .build(&component_schema)?;
        assert!(component_validator.is_valid(&serde_json::to_value(&component_projection)?));
        let signing_payload = manifest.signing_payload()?;
        let mut changed = manifest.clone();
        changed.streaming.maximum_response_body_bytes += 1;
        assert_ne!(signing_payload, changed.signing_payload()?);
        let key = PluginSigningKey::new("test-key", b"0123456789abcdef0123456789abcdef")?;
        assert_eq!(
            key.sign_provider_manifest_v2(&manifest)?,
            key.sign_provider_manifest_v2(&manifest)?
        );

        let mut missing_upload_grant = manifest.clone();
        missing_upload_grant
            .manifest
            .capabilities
            .retain(|capability| capability.kind != CapabilityKind::ProviderUpload);
        assert_eq!(
            missing_upload_grant.validate(&registry),
            Err(PluginContractError::InvalidProviderStreamingContract)
        );
        let mut undersized_upload_grant = manifest.clone();
        undersized_upload_grant
            .manifest
            .capabilities
            .iter_mut()
            .find(|capability| capability.kind == CapabilityKind::ProviderUpload)
            .ok_or("upload capability disappeared")?
            .quota
            .maximum_request_bytes = 7;
        assert_eq!(
            undersized_upload_grant.validate(&registry),
            Err(PluginContractError::InvalidProviderStreamingContract)
        );
        for (maximum_request_bytes, maximum_response_bytes, maximum_total_bytes) in [
            (
                MAX_PROVIDER_COST_REQUEST_BYTES - 1,
                MAX_PROVIDER_COST_RESPONSE_BYTES,
                2 * (MAX_PROVIDER_COST_REQUEST_BYTES + MAX_PROVIDER_COST_RESPONSE_BYTES),
            ),
            (
                MAX_PROVIDER_COST_REQUEST_BYTES,
                MAX_PROVIDER_COST_RESPONSE_BYTES - 1,
                2 * (MAX_PROVIDER_COST_REQUEST_BYTES + MAX_PROVIDER_COST_RESPONSE_BYTES),
            ),
            (
                MAX_PROVIDER_COST_REQUEST_BYTES,
                MAX_PROVIDER_COST_RESPONSE_BYTES,
                2 * (MAX_PROVIDER_COST_REQUEST_BYTES + MAX_PROVIDER_COST_RESPONSE_BYTES) - 1,
            ),
        ] {
            let mut undersized_cost_grant = manifest.clone();
            let quota = &mut undersized_cost_grant
                .manifest
                .capabilities
                .iter_mut()
                .find(|capability| capability.kind == CapabilityKind::ProviderCost)
                .ok_or("cost capability disappeared")?
                .quota;
            quota.maximum_request_bytes = maximum_request_bytes;
            quota.maximum_response_bytes = maximum_response_bytes;
            quota.maximum_total_bytes = maximum_total_bytes;
            assert_eq!(
                undersized_cost_grant.validate(&registry),
                Err(PluginContractError::InvalidProviderStreamingContract)
            );
        }
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_imports_only_canonical_invocation_inputs() -> Result<(), Box<dyn Error>>
    {
        assert_eq!(
            include_bytes!("../wit/sim-plugin.wit"),
            include_bytes!("../wit/provider-v2/deps/sim-plugin/sim-plugin.wit")
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../wit/sim-plugin.wit"))
            ),
            "2703627091bb596ed4973903311a91b9c4663ea9da72d854e0a221ad4c3f4866"
        );
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(include_bytes!("../schema/plugin-manifest-v1.schema.json"))
            ),
            "0b7439ac09d1f1ef8341483a2857da89501135b300204472e0cc997e9bb02a8e"
        );

        let wit = include_str!("../wit/provider-v2/sim-provider-plugin.wit");
        let input_host = wit
            .split_once("interface invocation-input-host {")
            .and_then(|(_, remainder)| remainder.split_once("\n}"))
            .map(|(interface, _)| interface)
            .ok_or("provider-v2 invocation-input-host disappeared")?;
        let canonical_use = "use zed:sim-plugin/types@1.0.0.{encoded-value, input-state, invocation-error, value-handle};";
        assert_eq!(input_host.matches(canonical_use).count(), 1);

        let mut previous_position = 0;
        for method in [
            "get-input-state: func(port-id: string) -> result<input-state, invocation-error>",
            "read-scalar-input: func(port-id: string, index: u32) -> result<encoded-value, invocation-error>",
            "take-input: func(port-id: string, index: u32) -> result<value-handle, invocation-error>",
            "read-handle: func(handle: value-handle) -> result<encoded-value, invocation-error>",
            "read-media-input: func(handle: value-handle) -> result<media-content, invocation-error>",
            "media-input-count: func(handle: value-handle) -> result<u32, invocation-error>",
            "read-media-input-at: func(handle: value-handle, batch-index: u32) -> result<media-content, invocation-error>",
            "secret-exists: func(identifier: string) -> result<bool, invocation-error>",
            "check-cancelled: func() -> result<_, invocation-error>",
        ] {
            let position = input_host
                .find(method)
                .ok_or_else(|| format!("provider-v2 invocation input host lost `{method}`"))?;
            assert!(position >= previous_position);
            previous_position = position;
        }
        assert_eq!(input_host.matches(": func(").count(), 9);
        let input_host_without_presence_query = input_host.replace(
            "secret-exists: func(identifier: string) -> result<bool, invocation-error>;",
            "",
        );
        for forbidden in [
            "provider:",
            "endpoint:",
            "secret",
            "start-request",
            "provider-request",
            "create-output-value",
            "push-output",
            "finish-output",
            "filesystem-read",
            "secret-exists",
            "clock-now",
            "random-bytes",
            "model-open",
            "output-begin",
            "output-write",
            "output-commit",
            "log:",
            "ui-set",
            "route-respond",
            "authorize",
        ] {
            assert!(
                !input_host_without_presence_query.contains(forbidden),
                "provider-v2 input-only host exposed `{forbidden}`"
            );
        }

        let world = wit
            .split_once("world sim-provider-plugin {")
            .and_then(|(_, remainder)| remainder.split_once('}'))
            .map(|(world, _)| world)
            .ok_or("provider-v2 world disappeared")?;
        assert_eq!(world.matches("import invocation-input-host;").count(), 1);
        assert_eq!(world.matches("import provider-streaming-host;").count(), 1);
        assert_eq!(world.matches("export ").count(), 1);
        assert!(
            world.find("import invocation-input-host;")
                < world.find("import provider-streaming-host;")
        );
        assert!(world.contains("export provider-node;"));

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v2.schema.json"))?;
        for duplicate_authority in ["input_state", "value_handle", "invocation_error"] {
            assert!(
                schema["$defs"].get(duplicate_authority).is_none(),
                "provider-v2 schema duplicated canonical `{duplicate_authority}` semantics"
            );
        }
        let schema_source = include_str!("../schema/plugin-manifest-v2.schema.json");
        assert_eq!(
            schema_source.matches("#/$defs/encoded_value").count(),
            1,
            "provider-v2 encoded-value schema must remain output-only"
        );
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_request_authority_is_canonical_bounded_and_host_derived()
    -> Result<(), Box<dyn Error>> {
        let contract = provider_streaming_contract_fixture();
        let context = ProviderInvocationContextV2 {
            invocation: 19,
            generation: 2,
        };
        let handle = ProviderStreamHandleV2 {
            invocation: 19,
            slot: 1,
            generation: 2,
        };
        let request = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: Some("provider-api-key".to_owned()),
            method: ProviderHttpMethodV2::Post,
            headers: vec![
                ProviderHeaderV2 {
                    name: "X-First".to_owned(),
                    value: "one".to_owned(),
                },
                ProviderHeaderV2 {
                    name: "X-Second".to_owned(),
                    value: "two".to_owned(),
                },
            ],
            declared_body_bytes: Some(4),
        };
        ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &request)?;
        let canonical = request.canonical_bytes(&contract)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&canonical)),
            "caa751f44c76725b6214cdea8f78b2c7d230770a6c29ca472a97f88f0515a56d"
        );
        for mutation in [
            ProviderRequestHeadV2 {
                endpoint: "https://api.provider.invalid/v1/other".to_owned(),
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                secret_id: None,
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                secret_id: Some("different-provider-api-key".to_owned()),
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                method: ProviderHttpMethodV2::Get,
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                headers: request.headers.iter().cloned().rev().collect(),
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                headers: vec![
                    ProviderHeaderV2 {
                        name: "X-Changed".to_owned(),
                        value: "one".to_owned(),
                    },
                    request.headers[1].clone(),
                ],
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                headers: vec![
                    ProviderHeaderV2 {
                        name: "X-First".to_owned(),
                        value: "changed".to_owned(),
                    },
                    request.headers[1].clone(),
                ],
                ..request.clone()
            },
            ProviderRequestHeadV2 {
                declared_body_bytes: Some(3),
                ..request.clone()
            },
        ] {
            assert_ne!(mutation.canonical_bytes(&contract)?, canonical);
        }

        for (endpoint, secret_id) in [
            (String::new(), None),
            (" https://api.provider.invalid".to_owned(), None),
            ("https://api.provider.invalid\n".to_owned(), None),
            ("https://api.provider.invalid/é".to_owned(), None),
            ("x".repeat(MAX_PROVIDER_REQUEST_ENDPOINT_BYTES + 1), None),
            (
                "https://api.provider.invalid".to_owned(),
                Some(String::new()),
            ),
            (
                "https://api.provider.invalid".to_owned(),
                Some("secret id".to_owned()),
            ),
            (
                "https://api.provider.invalid".to_owned(),
                Some("é".to_owned()),
            ),
            (
                "https://api.provider.invalid".to_owned(),
                Some("x".repeat(MAX_PROVIDER_REQUEST_SECRET_ID_BYTES + 1)),
            ),
        ] {
            let invalid = ProviderRequestHeadV2 {
                endpoint,
                secret_id,
                ..request.clone()
            };
            assert_eq!(
                ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &invalid,)
                    .err(),
                Some(ProviderStreamingContractError::InvalidRequestAuthority)
            );
        }

        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v2.schema.json"))?;
        let mut v1_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schema/plugin-manifest-v1.schema.json"))?;
        v1_schema["$id"] = serde_json::Value::String(
            "https://zed.dev/schema/sim-plugin-manifest-v1.json".to_owned(),
        );
        let request_schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "#/$defs/request_head",
            "$defs": schema["$defs"].clone(),
        });
        let validator = jsonschema::options()
            .with_resource(
                "https://zed.dev/schema/sim-plugin-manifest-v1.json",
                jsonschema::Resource::from_contents(v1_schema),
            )
            .build(&request_schema)?;
        let serialized = serde_json::to_value(&request)?;
        assert!(validator.is_valid(&serialized));
        assert!(serialized.get("provider").is_none());
        for invalid_value in [
            serde_json::Value::Null,
            serde_json::Value::String(String::new()),
            serde_json::Value::String("https://api.provider.invalid/é".to_owned()),
            serde_json::Value::String("x".repeat(MAX_PROVIDER_REQUEST_ENDPOINT_BYTES + 1)),
        ] {
            let mut invalid = serialized.clone();
            invalid["endpoint"] = invalid_value;
            assert!(!validator.is_valid(&invalid));
        }
        let mut missing_endpoint = serialized.clone();
        missing_endpoint
            .as_object_mut()
            .ok_or("request head was not an object")?
            .remove("endpoint");
        assert!(!validator.is_valid(&missing_endpoint));
        let mut missing_secret = serialized.clone();
        missing_secret
            .as_object_mut()
            .ok_or("request head was not an object")?
            .remove("secret_id");
        assert!(!validator.is_valid(&missing_secret));
        let mut null_secret = serialized.clone();
        null_secret["secret_id"] = serde_json::Value::Null;
        assert!(validator.is_valid(&null_secret));
        for invalid_value in [
            serde_json::Value::String(String::new()),
            serde_json::Value::String("secret id".to_owned()),
            serde_json::Value::String("é".to_owned()),
            serde_json::Value::String("x".repeat(MAX_PROVIDER_REQUEST_SECRET_ID_BYTES + 1)),
        ] {
            let mut invalid = serialized.clone();
            invalid["secret_id"] = invalid_value;
            assert!(!validator.is_valid(&invalid));
        }
        let boundary = ProviderRequestHeadV2 {
            endpoint: "x".repeat(MAX_PROVIDER_REQUEST_ENDPOINT_BYTES),
            secret_id: Some("s".repeat(MAX_PROVIDER_REQUEST_SECRET_ID_BYTES)),
            ..request
        };
        boundary.validate_for_contract(&contract)?;
        assert!(validator.is_valid(&serde_json::to_value(boundary)?));
        let mut component_provider = serialized;
        component_provider
            .as_object_mut()
            .ok_or("request head was not an object")?
            .insert(
                "provider".to_owned(),
                serde_json::Value::String("forged-provider".to_owned()),
            );
        assert!(!validator.is_valid(&component_provider));
        assert!(serde_json::from_value::<ProviderRequestHeadV2>(component_provider).is_err());
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_is_generation_scoped_ordered_and_bounded() -> Result<(), Box<dyn Error>>
    {
        let contract = provider_streaming_contract_fixture();
        let context = ProviderInvocationContextV2 {
            invocation: 7,
            generation: 3,
        };
        let handle = ProviderStreamHandleV2 {
            invocation: 7,
            slot: 1,
            generation: 3,
        };
        let request = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: Some("provider-api-key".to_owned()),
            method: ProviderHttpMethodV2::Post,
            headers: vec![
                ProviderHeaderV2 {
                    name: "X-Test".to_owned(),
                    value: "first".to_owned(),
                },
                ProviderHeaderV2 {
                    name: "x-test".to_owned(),
                    value: "second".to_owned(),
                },
            ],
            declared_body_bytes: Some(4),
        };
        let mut validator =
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &request)?;
        let stale_handle = ProviderStreamHandleV2 {
            generation: 2,
            ..handle
        };
        assert_eq!(
            ProviderStreamValidatorV2::checked(contract.clone(), context, stale_handle, &request)
                .err(),
            Some(ProviderStreamingContractError::RevokedHandle)
        );
        assert_eq!(
            ProviderStreamValidatorV2::checked(
                contract.clone(),
                context,
                ProviderStreamHandleV2 {
                    invocation: 8,
                    ..handle
                },
                &request,
            )
            .err(),
            Some(ProviderStreamingContractError::ForeignHandle)
        );
        let mut invalid_header = request.clone();
        invalid_header.headers[0].value = "bad\u{7f}".to_owned();
        assert_eq!(
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &invalid_header)
                .err(),
            Some(ProviderStreamingContractError::InvalidHeaders)
        );
        assert_eq!(
            validator.write_request_chunk(&ProviderRequestChunkV2 {
                handle,
                sequence: 0,
                bytes: Vec::new(),
                end: false,
            }),
            Err(ProviderStreamingContractError::ChunkLimit)
        );
        validator.write_request_chunk(&ProviderRequestChunkV2 {
            handle,
            sequence: 0,
            bytes: b"da".to_vec(),
            end: false,
        })?;
        validator.write_request_chunk(&ProviderRequestChunkV2 {
            handle,
            sequence: 1,
            bytes: b"ta".to_vec(),
            end: true,
        })?;
        assert_eq!(validator.request_bytes(), 4);

        let upload_request = ProviderUploadRequestV2 {
            handle,
            port_id: "image".to_owned(),
            media_type: "image/png".to_owned(),
            byte_length: 4,
            content_sha256: format!("{:x}", Sha256::digest(b"file")),
        };
        let upload_handle = ProviderStreamHandleV2 { slot: 2, ..handle };
        let mut upload = validator.start_upload(&upload_request, upload_handle)?;
        assert_eq!(
            validator.start_upload(&upload_request, upload_handle).err(),
            Some(ProviderStreamingContractError::ForeignHandle)
        );
        upload.write_chunk(&ProviderRequestChunkV2 {
            handle: upload_handle,
            sequence: 0,
            bytes: b"file".to_vec(),
            end: true,
        })?;
        assert!(upload.is_terminal());

        let second_upload_request = ProviderUploadRequestV2 {
            byte_length: 1,
            content_sha256: format!("{:x}", Sha256::digest(b"x")),
            ..upload_request.clone()
        };
        let second_upload_handle = ProviderStreamHandleV2 { slot: 3, ..handle };
        let mut second_upload =
            validator.start_upload(&second_upload_request, second_upload_handle)?;
        let over_aggregate_request = ProviderUploadRequestV2 {
            byte_length: 4,
            content_sha256: format!("{:x}", Sha256::digest(b"more")),
            ..upload_request.clone()
        };
        assert_eq!(
            validator
                .start_upload(
                    &over_aggregate_request,
                    ProviderStreamHandleV2 { slot: 4, ..handle },
                )
                .err(),
            Some(ProviderStreamingContractError::InvalidUpload)
        );

        let cost_request = ProviderCostRequestV2 {
            handle,
            operation: "generate.image".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 50,
        };
        let accepted_cost = ProviderCostResponseV2 {
            accepted: true,
            approved_microunits: 40,
            receipt: b"cost-receipt".to_vec(),
        };
        validator.accept_cost_request(&cost_request, &accepted_cost)?;
        validator.accept_cost_request(
            &cost_request,
            &ProviderCostResponseV2 {
                accepted: false,
                approved_microunits: 0,
                receipt: Vec::new(),
            },
        )?;
        assert_eq!(
            validator.accept_cost_request(&cost_request, &accepted_cost),
            Err(ProviderStreamingContractError::InvalidCostRequest)
        );
        assert_eq!(
            validator.accept_cost_request(
                &cost_request,
                &ProviderCostResponseV2 {
                    accepted: true,
                    approved_microunits: 51,
                    receipt: b"cost-receipt".to_vec(),
                },
            ),
            Err(ProviderStreamingContractError::InvalidCostRequest)
        );

        validator.accept_progress(&ProviderProgressV2 {
            handle,
            sequence: 0,
            completed: 10,
            total: 100,
            message: Some("started".to_owned()),
        })?;
        assert_eq!(
            validator.accept_progress(&ProviderProgressV2 {
                handle,
                sequence: 1,
                completed: 9,
                total: 100,
                message: None,
            }),
            Err(ProviderStreamingContractError::InvalidProgress)
        );
        validator.accept_progress(&ProviderProgressV2 {
            handle,
            sequence: 1,
            completed: 100,
            total: 100,
            message: None,
        })?;

        let first_wait = ProviderWaitRequestV2 {
            handle,
            after_sequence: None,
            timeout_milliseconds: 100,
        };
        validator.accept_wait(
            &first_wait,
            ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: 0,
                event: ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                    status: 200,
                    headers: Vec::new(),
                }),
            }),
        )?;
        let second_wait = ProviderWaitRequestV2 {
            after_sequence: Some(0),
            ..first_wait.clone()
        };
        validator.accept_wait(
            &second_wait,
            ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: 1,
                event: ProviderResponseFrameEventV2::Chunk(ProviderResponseChunkV2::NdjsonLine(
                    "{\"ok\":true}".to_owned(),
                )),
            }),
        )?;
        let terminal_wait = ProviderWaitRequestV2 {
            after_sequence: Some(1),
            ..first_wait.clone()
        };
        let completed_frame = ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
            handle,
            sequence: 2,
            event: ProviderResponseFrameEventV2::Terminal(ProviderStreamTerminalV2::Completed {
                receipt: b"result-receipt".to_vec(),
            }),
        });
        assert_eq!(
            validator.accept_wait(&terminal_wait, completed_frame.clone()),
            Err(ProviderStreamingContractError::InvalidTerminal)
        );
        second_upload.write_chunk(&ProviderRequestChunkV2 {
            handle: second_upload_handle,
            sequence: 0,
            bytes: b"x".to_vec(),
            end: true,
        })?;
        validator.accept_wait(&terminal_wait, completed_frame)?;
        assert!(validator.is_terminal());
        assert_eq!(validator.response_bytes(), 11);
        assert_eq!(
            upload.write_chunk(&ProviderRequestChunkV2 {
                handle: upload_handle,
                sequence: 1,
                bytes: Vec::new(),
                end: true,
            }),
            Err(ProviderStreamingContractError::RevokedHandle)
        );

        let mut informational = ProviderStreamValidatorV2::checked(
            contract.clone(),
            context,
            handle,
            &ProviderRequestHeadV2 {
                declared_body_bytes: Some(0),
                ..request.clone()
            },
        )?;
        informational.write_request_chunk(&ProviderRequestChunkV2 {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        assert_eq!(
            informational.accept_wait(
                &first_wait,
                ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                    handle,
                    sequence: 0,
                    event: ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                        status: 103,
                        headers: Vec::new(),
                    }),
                }),
            ),
            Err(ProviderStreamingContractError::InvalidOrder)
        );

        let mut cancelled =
            ProviderStreamValidatorV2::checked(contract, context, handle, &request)?;
        let mut cancelled_upload = cancelled.start_upload(&upload_request, upload_handle)?;
        cancelled.accept_wait(&first_wait, ProviderWaitOutcomeV2::Cancelled)?;
        assert_eq!(
            cancelled_upload.write_chunk(&ProviderRequestChunkV2 {
                handle: upload_handle,
                sequence: 0,
                bytes: b"file".to_vec(),
                end: true,
            }),
            Err(ProviderStreamingContractError::RevokedHandle)
        );

        let mut unknown_chunk = serde_json::to_value(ProviderRequestChunkV2 {
            handle,
            sequence: 0,
            bytes: b"data".to_vec(),
            end: true,
        })?;
        unknown_chunk
            .as_object_mut()
            .ok_or("request chunk was not an object")?
            .insert("ignored".to_owned(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<ProviderRequestChunkV2>(unknown_chunk).is_err());
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_retry_revokes_old_handles_and_preserves_cumulative_limits()
    -> Result<(), Box<dyn Error>> {
        let contract = provider_streaming_contract_fixture();
        let context = ProviderInvocationContextV2 {
            invocation: 31,
            generation: 5,
        };
        let handle = ProviderStreamHandleV2 {
            invocation: context.invocation,
            slot: 1,
            generation: context.generation,
        };
        let request = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Post,
            headers: Vec::new(),
            declared_body_bytes: Some(0),
        };
        let mut validator =
            ProviderStreamValidatorV2::checked(contract, context, handle, &request)?;
        validator.write_request_chunk(&ProviderRequestChunkV2 {
            handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        let upload_handle = ProviderStreamHandleV2 { slot: 2, ..handle };
        let upload_request = ProviderUploadRequestV2 {
            handle,
            port_id: "image".to_owned(),
            media_type: "image/png".to_owned(),
            byte_length: 1,
            content_sha256: format!("{:x}", Sha256::digest(b"x")),
        };
        let mut upload = validator.start_upload(&upload_request, upload_handle)?;
        upload.write_chunk(&ProviderRequestChunkV2 {
            handle: upload_handle,
            sequence: 0,
            bytes: b"x".to_vec(),
            end: true,
        })?;
        let cost_request = ProviderCostRequestV2 {
            handle,
            operation: "generate.image".to_owned(),
            currency: "USD".to_owned(),
            maximum_microunits: 10,
        };
        validator.accept_cost_request(
            &cost_request,
            &ProviderCostResponseV2 {
                accepted: false,
                approved_microunits: 0,
                receipt: Vec::new(),
            },
        )?;
        validator.accept_progress(&ProviderProgressV2 {
            handle,
            sequence: 0,
            completed: 1,
            total: 2,
            message: None,
        })?;
        validator.accept_wait(
            &ProviderWaitRequestV2 {
                handle,
                after_sequence: None,
                timeout_milliseconds: 10,
            },
            ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: 0,
                event: ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                    status: 503,
                    headers: Vec::new(),
                }),
            }),
        )?;
        let retry_handle = ProviderStreamHandleV2 { slot: 3, ..handle };
        validator.retry(retry_handle)?;
        assert_eq!(
            upload.write_chunk(&ProviderRequestChunkV2 {
                handle: upload_handle,
                sequence: 1,
                bytes: Vec::new(),
                end: true,
            }),
            Err(ProviderStreamingContractError::RevokedHandle)
        );
        validator.write_request_chunk(&ProviderRequestChunkV2 {
            handle: retry_handle,
            sequence: 0,
            bytes: Vec::new(),
            end: true,
        })?;
        validator.accept_progress(&ProviderProgressV2 {
            handle: retry_handle,
            sequence: 1,
            completed: 2,
            total: 2,
            message: None,
        })?;
        let second_upload_request = ProviderUploadRequestV2 {
            handle: retry_handle,
            ..upload_request
        };
        let second_upload_handle = ProviderStreamHandleV2 {
            slot: 4,
            ..retry_handle
        };
        validator.start_upload(&second_upload_request, second_upload_handle)?;
        assert_eq!(
            validator
                .start_upload(
                    &second_upload_request,
                    ProviderStreamHandleV2 {
                        slot: 5,
                        ..retry_handle
                    },
                )
                .err(),
            Some(ProviderStreamingContractError::InvalidUpload)
        );
        let retry_cost_request = ProviderCostRequestV2 {
            handle: retry_handle,
            ..cost_request
        };
        validator.accept_cost_request(
            &retry_cost_request,
            &ProviderCostResponseV2 {
                accepted: false,
                approved_microunits: 0,
                receipt: Vec::new(),
            },
        )?;
        assert_eq!(
            validator.accept_cost_request(
                &retry_cost_request,
                &ProviderCostResponseV2 {
                    accepted: false,
                    approved_microunits: 0,
                    receipt: Vec::new(),
                },
            ),
            Err(ProviderStreamingContractError::InvalidCostRequest)
        );
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_rejects_every_bounded_lifecycle_drift() -> Result<(), Box<dyn Error>> {
        let context = ProviderInvocationContextV2 {
            invocation: 11,
            generation: 5,
        };
        let handle = ProviderStreamHandleV2 {
            invocation: 11,
            slot: 1,
            generation: 5,
        };
        let methods = vec![
            ProviderHttpMethodV2::Delete,
            ProviderHttpMethodV2::Get,
            ProviderHttpMethodV2::Head,
            ProviderHttpMethodV2::Options,
            ProviderHttpMethodV2::Patch,
            ProviderHttpMethodV2::Post,
            ProviderHttpMethodV2::Put,
        ];
        let mut all_methods = provider_streaming_contract_fixture();
        all_methods.methods = methods.clone();
        all_methods.validate()?;
        for method in methods {
            let declared_body_bytes = (method == ProviderHttpMethodV2::Head).then_some(0);
            ProviderStreamValidatorV2::checked(
                all_methods.clone(),
                context,
                handle,
                &ProviderRequestHeadV2 {
                    endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
                    secret_id: None,
                    method,
                    headers: Vec::new(),
                    declared_body_bytes,
                },
            )?;
        }
        let mut unordered_methods = all_methods.clone();
        unordered_methods.methods.swap(0, 1);
        assert_eq!(
            unordered_methods.validate(),
            Err(ProviderStreamingContractError::InvalidContract)
        );

        let contract = provider_streaming_contract_fixture();
        let request = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: Some("provider-api-key".to_owned()),
            method: ProviderHttpMethodV2::Post,
            headers: Vec::new(),
            declared_body_bytes: Some(3),
        };
        let mut body_validator =
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &request)?;
        assert_eq!(
            body_validator.write_request_chunk(&ProviderRequestChunkV2 {
                handle,
                sequence: 0,
                bytes: b"four".to_vec(),
                end: true,
            }),
            Err(ProviderStreamingContractError::BodyLimit)
        );
        assert_eq!(
            body_validator.write_request_chunk(&ProviderRequestChunkV2 {
                handle,
                sequence: 0,
                bytes: vec![0; 17],
                end: false,
            }),
            Err(ProviderStreamingContractError::ChunkLimit)
        );
        let head_with_body = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Head,
            headers: Vec::new(),
            declared_body_bytes: Some(1),
        };
        assert_eq!(
            ProviderStreamValidatorV2::checked(all_methods, context, handle, &head_with_body,)
                .err(),
            Some(ProviderStreamingContractError::BodyLimit)
        );

        let open_request = ProviderRequestHeadV2 {
            endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
            secret_id: None,
            method: ProviderHttpMethodV2::Post,
            headers: Vec::new(),
            declared_body_bytes: None,
        };
        let mut waiting =
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &open_request)?;
        assert_eq!(
            waiting.accept_wait(
                &ProviderWaitRequestV2 {
                    handle,
                    after_sequence: Some(0),
                    timeout_milliseconds: 1,
                },
                ProviderWaitOutcomeV2::TimedOut,
            ),
            Err(ProviderStreamingContractError::InvalidSequence)
        );
        assert_eq!(
            waiting.accept_wait(
                &ProviderWaitRequestV2 {
                    handle,
                    after_sequence: None,
                    timeout_milliseconds: 0,
                },
                ProviderWaitOutcomeV2::TimedOut,
            ),
            Err(ProviderStreamingContractError::WaitLimit)
        );
        assert_eq!(
            waiting.accept_wait(
                &ProviderWaitRequestV2 {
                    handle: ProviderStreamHandleV2 {
                        generation: 4,
                        ..handle
                    },
                    after_sequence: None,
                    timeout_milliseconds: 1,
                },
                ProviderWaitOutcomeV2::TimedOut,
            ),
            Err(ProviderStreamingContractError::RevokedHandle)
        );
        assert_eq!(
            waiting.accept_wait(
                &ProviderWaitRequestV2 {
                    handle: ProviderStreamHandleV2 { slot: 9, ..handle },
                    after_sequence: None,
                    timeout_milliseconds: 1,
                },
                ProviderWaitOutcomeV2::TimedOut,
            ),
            Err(ProviderStreamingContractError::ForeignHandle)
        );
        waiting.accept_wait(
            &ProviderWaitRequestV2 {
                handle,
                after_sequence: None,
                timeout_milliseconds: 1,
            },
            ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                handle,
                sequence: 0,
                event: ProviderResponseFrameEventV2::Terminal(ProviderStreamTerminalV2::Failed {
                    code: "provider.failed".to_owned(),
                    message: "failed before a response".to_owned(),
                }),
            }),
        )?;
        assert_eq!(
            waiting.accept_progress(&ProviderProgressV2 {
                handle,
                sequence: 0,
                completed: 0,
                total: 1,
                message: None,
            }),
            Err(ProviderStreamingContractError::RevokedHandle)
        );

        let ready_response =
            |method: ProviderHttpMethodV2,
             status: u16,
             response_limit: u64|
             -> Result<ProviderStreamValidatorV2, ProviderStreamingContractError> {
                let mut contract = provider_streaming_contract_fixture();
                contract.methods = vec![method];
                contract.maximum_response_body_bytes = response_limit;
                let mut validator = ProviderStreamValidatorV2::checked(
                    contract,
                    context,
                    handle,
                    &ProviderRequestHeadV2 {
                        endpoint: "https://api.provider.invalid/v1/generate".to_owned(),
                        secret_id: None,
                        method,
                        headers: Vec::new(),
                        declared_body_bytes: Some(0),
                    },
                )?;
                validator.write_request_chunk(&ProviderRequestChunkV2 {
                    handle,
                    sequence: 0,
                    bytes: Vec::new(),
                    end: true,
                })?;
                validator.accept_wait(
                    &ProviderWaitRequestV2 {
                        handle,
                        after_sequence: None,
                        timeout_milliseconds: 1,
                    },
                    ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                        handle,
                        sequence: 0,
                        event: ProviderResponseFrameEventV2::Head(ProviderResponseHeadV2 {
                            status,
                            headers: Vec::new(),
                        }),
                    }),
                )?;
                Ok(validator)
            };
        let chunk_wait = ProviderWaitRequestV2 {
            handle,
            after_sequence: Some(0),
            timeout_milliseconds: 1,
        };
        for (method, status) in [
            (ProviderHttpMethodV2::Head, 200),
            (ProviderHttpMethodV2::Post, 204),
            (ProviderHttpMethodV2::Post, 205),
            (ProviderHttpMethodV2::Post, 304),
        ] {
            let mut no_body = ready_response(method, status, 64)?;
            assert_eq!(
                no_body.accept_wait(
                    &chunk_wait,
                    ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                        handle,
                        sequence: 1,
                        event: ProviderResponseFrameEventV2::Chunk(
                            ProviderResponseChunkV2::Binary(vec![1]),
                        ),
                    }),
                ),
                Err(ProviderStreamingContractError::InvalidOrder)
            );
        }
        let mut invalid_ndjson = ready_response(ProviderHttpMethodV2::Post, 200, 64)?;
        assert_eq!(
            invalid_ndjson.accept_wait(
                &chunk_wait,
                ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                    handle,
                    sequence: 1,
                    event: ProviderResponseFrameEventV2::Chunk(
                        ProviderResponseChunkV2::NdjsonLine("not-json".to_owned()),
                    ),
                }),
            ),
            Err(ProviderStreamingContractError::InvalidNdjsonLine)
        );
        let mut response_overflow = ready_response(ProviderHttpMethodV2::Post, 200, 8)?;
        assert_eq!(
            response_overflow.accept_wait(
                &chunk_wait,
                ProviderWaitOutcomeV2::Frame(ProviderResponseFrameV2 {
                    handle,
                    sequence: 1,
                    event: ProviderResponseFrameEventV2::Chunk(ProviderResponseChunkV2::Binary(
                        vec![0; 9]
                    ),),
                }),
            ),
            Err(ProviderStreamingContractError::BodyLimit)
        );

        let upload_request = ProviderUploadRequestV2 {
            handle,
            port_id: "image".to_owned(),
            media_type: "image/png".to_owned(),
            byte_length: 4,
            content_sha256: format!("{:x}", Sha256::digest(b"file")),
        };
        let mut upload_parent =
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &open_request)?;
        assert_eq!(
            upload_parent
                .start_upload(
                    &upload_request,
                    ProviderStreamHandleV2 {
                        slot: 2,
                        generation: 4,
                        ..handle
                    },
                )
                .err(),
            Some(ProviderStreamingContractError::RevokedHandle)
        );
        assert_eq!(
            upload_parent
                .start_upload(
                    &upload_request,
                    ProviderStreamHandleV2 {
                        invocation: 12,
                        slot: 2,
                        ..handle
                    },
                )
                .err(),
            Some(ProviderStreamingContractError::ForeignHandle)
        );
        let upload_handle = ProviderStreamHandleV2 { slot: 2, ..handle };
        let mut bad_digest = upload_parent.start_upload(&upload_request, upload_handle)?;
        assert_eq!(
            bad_digest.write_chunk(&ProviderRequestChunkV2 {
                handle: upload_handle,
                sequence: 0,
                bytes: b"fail".to_vec(),
                end: true,
            }),
            Err(ProviderStreamingContractError::InvalidUpload)
        );
        assert!(!bad_digest.is_terminal());
        assert_eq!(
            bad_digest.write_chunk(&ProviderRequestChunkV2 {
                handle: ProviderStreamHandleV2 {
                    generation: 4,
                    ..upload_handle
                },
                sequence: 0,
                bytes: b"file".to_vec(),
                end: true,
            }),
            Err(ProviderStreamingContractError::RevokedHandle)
        );

        let mut aggregate_contract = contract.clone();
        aggregate_contract.maximum_uploads = 3;
        aggregate_contract.maximum_upload_body_bytes = 5;
        let mut aggregate_uploads =
            ProviderStreamValidatorV2::checked(aggregate_contract, context, handle, &open_request)?;
        aggregate_uploads.start_upload(&upload_request, upload_handle)?;
        assert_eq!(
            aggregate_uploads
                .start_upload(
                    &ProviderUploadRequestV2 {
                        byte_length: 2,
                        content_sha256: format!("{:x}", Sha256::digest(b"ok")),
                        ..upload_request.clone()
                    },
                    ProviderStreamHandleV2 { slot: 3, ..handle },
                )
                .err(),
            Some(ProviderStreamingContractError::InvalidUpload)
        );

        let mut costs =
            ProviderStreamValidatorV2::checked(contract.clone(), context, handle, &open_request)?;
        let valid_cost_response = ProviderCostResponseV2 {
            accepted: true,
            approved_microunits: 1,
            receipt: b"receipt".to_vec(),
        };
        for invalid_request in [
            ProviderCostRequestV2 {
                handle,
                operation: "generate".to_owned(),
                currency: "usd".to_owned(),
                maximum_microunits: 2,
            },
            ProviderCostRequestV2 {
                handle,
                operation: "generate".to_owned(),
                currency: "USD".to_owned(),
                maximum_microunits: 0,
            },
        ] {
            assert_eq!(
                costs.accept_cost_request(&invalid_request, &valid_cost_response),
                Err(ProviderStreamingContractError::InvalidCostRequest)
            );
        }
        assert_eq!(
            costs.accept_cost_request(
                &ProviderCostRequestV2 {
                    handle,
                    operation: "generate".to_owned(),
                    currency: "USD".to_owned(),
                    maximum_microunits: 2,
                },
                &ProviderCostResponseV2 {
                    accepted: true,
                    approved_microunits: 1,
                    receipt: vec![0; MAX_PROVIDER_RESULT_RECEIPT_BYTES + 1],
                },
            ),
            Err(ProviderStreamingContractError::InvalidCostRequest)
        );

        let mut progress =
            ProviderStreamValidatorV2::checked(contract, context, handle, &open_request)?;
        for invalid_progress in [
            ProviderProgressV2 {
                handle,
                sequence: 1,
                completed: 0,
                total: 1,
                message: None,
            },
            ProviderProgressV2 {
                handle,
                sequence: 0,
                completed: 0,
                total: 101,
                message: None,
            },
            ProviderProgressV2 {
                handle,
                sequence: 0,
                completed: 0,
                total: 1,
                message: Some("x".repeat(MAX_PROVIDER_STREAM_PROGRESS_MESSAGE_BYTES + 1)),
            },
        ] {
            assert_eq!(
                progress.accept_progress(&invalid_progress),
                Err(ProviderStreamingContractError::InvalidProgress)
            );
        }

        let nested_unknown = serde_json::json!({
            "kind": "head",
            "value": {"status": 200, "headers": [], "ignored": true}
        });
        assert!(serde_json::from_value::<ProviderResponseFrameEventV2>(nested_unknown).is_err());
        Ok(())
    }

    #[test]
    fn provider_streaming_v2_materializes_only_canonical_typed_results()
    -> Result<(), Box<dyn Error>> {
        let registry = TypeRegistry::built_in()?;
        let type_id = registry.resolve("STRING")?.clone();
        let value = PluginValue::scalar(
            type_id.clone(),
            ScalarValue::String("done".to_owned()),
            &registry,
        )?;
        let result = ProviderInvocationResultV2 {
            outputs: vec![ProviderMaterializedOutputV2 {
                port_id: "text".to_owned(),
                value: ProviderEncodedValueV2 {
                    type_id,
                    family: ValueFamily::Scalar,
                    abi_bytes: value.abi_bytes()?,
                },
            }],
            receipt: b"sealed-result".to_vec(),
        };
        result.validate(&registry)?;

        let mut malformed = result.clone();
        malformed.outputs[0].value.abi_bytes = b"../secret/native-handle".to_vec();
        assert_eq!(
            malformed.validate(&registry),
            Err(ProviderStreamingContractError::InvalidInvocationResult)
        );
        let mut mismatched = result.clone();
        mismatched.outputs[0].value.type_id = registry.resolve("FLOAT")?.clone();
        assert_eq!(
            mismatched.validate(&registry),
            Err(ProviderStreamingContractError::InvalidInvocationResult)
        );
        let mut duplicate = result;
        duplicate.outputs.push(duplicate.outputs[0].clone());
        assert_eq!(
            duplicate.validate(&registry),
            Err(ProviderStreamingContractError::InvalidInvocationResult)
        );

        let internal_errors = [
            ProviderStreamingContractError::InvalidContract,
            ProviderStreamingContractError::InvalidRequestAuthority,
            ProviderStreamingContractError::InvalidHandle,
            ProviderStreamingContractError::ForeignHandle,
            ProviderStreamingContractError::RevokedHandle,
            ProviderStreamingContractError::InvalidMethod,
            ProviderStreamingContractError::InvalidHeaders,
            ProviderStreamingContractError::BodyLimit,
            ProviderStreamingContractError::ChunkLimit,
            ProviderStreamingContractError::InvalidNdjsonLine,
            ProviderStreamingContractError::InvalidSequence,
            ProviderStreamingContractError::InvalidOrder,
            ProviderStreamingContractError::WaitLimit,
            ProviderStreamingContractError::InvalidUpload,
            ProviderStreamingContractError::InvalidCostRequest,
            ProviderStreamingContractError::InvalidProgress,
            ProviderStreamingContractError::InvalidTerminal,
            ProviderStreamingContractError::InvalidInvocationResult,
        ];
        let wire_errors = internal_errors
            .iter()
            .map(ProviderStreamErrorV2::from)
            .collect::<BTreeSet<_>>();
        assert_eq!(wire_errors.len(), internal_errors.len());
        assert_eq!(
            ProviderStreamErrorV2::from(&ProviderStreamingContractError::InvalidRequestAuthority),
            ProviderStreamErrorV2::InvalidRequestAuthority
        );
        for (index, error) in [
            ProviderStreamErrorV2::Cancelled,
            ProviderStreamErrorV2::TimedOut,
            ProviderStreamErrorV2::HostFailure,
            ProviderStreamErrorV2::InvalidContract,
            ProviderStreamErrorV2::InvalidHandle,
            ProviderStreamErrorV2::ForeignHandle,
            ProviderStreamErrorV2::RevokedHandle,
            ProviderStreamErrorV2::InvalidMethod,
            ProviderStreamErrorV2::InvalidHeaders,
            ProviderStreamErrorV2::BodyLimit,
            ProviderStreamErrorV2::ChunkLimit,
            ProviderStreamErrorV2::InvalidNdjsonLine,
            ProviderStreamErrorV2::InvalidSequence,
            ProviderStreamErrorV2::InvalidOrder,
            ProviderStreamErrorV2::WaitLimit,
            ProviderStreamErrorV2::InvalidUpload,
            ProviderStreamErrorV2::InvalidCostRequest,
            ProviderStreamErrorV2::InvalidProgress,
            ProviderStreamErrorV2::InvalidTerminal,
            ProviderStreamErrorV2::InvalidInvocationResult,
            ProviderStreamErrorV2::InvalidRequestAuthority,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(usize::from(error as u8), index);
        }
        Ok(())
    }

    #[test]
    fn tensor_value_uses_the_canonical_checked_descriptor() -> Result<(), Box<dyn Error>> {
        let descriptor = TensorDescriptor::contiguous(
            vec![1, 2, 3],
            DType::F32,
            DeviceId::CPU,
            StreamId::new(7),
        )?;
        let value = TensorValue::new(descriptor, 24, "1".repeat(64))?;
        let encoded = serde_json::to_vec(&value)?;
        assert_eq!(serde_json::from_slice::<TensorValue>(&encoded)?, value);

        let invalid_layout = serde_json::json!({
            "descriptor": {
                "shape": [1, 2, 3],
                "strides": [1, 2, 3],
                "offset_elements": 0,
                "dtype": "f32",
                "layout": "contiguous",
                "device": {"kind": "cpu", "ordinal": 0},
                "stream": 0
            },
            "byte_length": 24,
            "digest": "1".repeat(64)
        });
        assert!(serde_json::from_value::<TensorValue>(invalid_layout).is_err());

        let unknown_device = serde_json::json!({
            "descriptor": {
                "shape": [1],
                "strides": [1],
                "offset_elements": 0,
                "dtype": "f32",
                "layout": "contiguous",
                "device": {"kind": "future_device", "ordinal": 0},
                "stream": 0
            },
            "byte_length": 4,
            "digest": "1".repeat(64)
        });
        assert!(serde_json::from_value::<TensorValue>(unknown_device).is_err());

        let offset_value = TensorValue::new(
            TensorDescriptor::new_strided(
                vec![1],
                vec![1],
                2,
                DType::F32,
                Layout::Strided,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            12,
            "1".repeat(64),
        )?;
        assert_eq!(offset_value.minimum_backing_byte_length()?, 12);
        Ok(())
    }

    #[test]
    fn tensor_canonical_projection_includes_all_descriptor_metadata() -> Result<(), Box<dyn Error>>
    {
        let registry = TypeRegistry::built_in()?;
        let image_type = registry.resolve("IMAGE")?.clone();
        let value = |descriptor, byte_length, digest: &str| {
            PluginValue::tensor(
                image_type.clone(),
                TensorValue::new(descriptor, byte_length, digest.repeat(64))?,
                &registry,
            )
        };

        let strided = |shape, strides, offset_elements, dtype, layout, device, stream| {
            TensorDescriptor::new_strided(
                shape,
                strides,
                offset_elements,
                dtype,
                layout,
                device,
                stream,
            )
        };

        let baseline_descriptor = strided(
            vec![1],
            vec![1],
            0,
            DType::F32,
            Layout::Strided,
            DeviceId::CPU,
            StreamId::DEFAULT,
        )?;
        let baseline = value(baseline_descriptor.clone(), 16, "2")?.canonical_bytes();
        assert_ne!(
            baseline,
            value(
                strided(
                    vec![2],
                    vec![1],
                    0,
                    DType::F32,
                    Layout::Strided,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                16,
                "2",
            )?
            .canonical_bytes()
        );
        assert_ne!(
            baseline,
            value(
                strided(
                    vec![1],
                    vec![1],
                    2,
                    DType::F32,
                    Layout::Strided,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                16,
                "2",
            )?
            .canonical_bytes()
        );
        assert_ne!(
            baseline,
            value(
                strided(
                    vec![1],
                    vec![1],
                    0,
                    DType::F32,
                    Layout::Contiguous,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                16,
                "2",
            )?
            .canonical_bytes()
        );
        let stride_baseline = value(
            strided(
                vec![2],
                vec![1],
                0,
                DType::F32,
                Layout::Strided,
                DeviceId::CPU,
                StreamId::DEFAULT,
            )?,
            16,
            "2",
        )?
        .canonical_bytes();
        assert_ne!(
            stride_baseline,
            value(
                strided(
                    vec![2],
                    vec![2],
                    0,
                    DType::F32,
                    Layout::Strided,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?,
                16,
                "2",
            )?
            .canonical_bytes()
        );

        let dtypes = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool,
            DType::Complex64,
            DType::Complex128,
            DType::Float8E4m3Fn,
            DType::Float8E5m2,
            DType::Float8E4m3Fnuz,
            DType::Float8E5m2Fnuz,
            DType::Float8E8m0Fnu,
        ];
        let dtype_projections = dtypes
            .into_iter()
            .map(|dtype| -> Result<Vec<u8>, Box<dyn Error>> {
                let descriptor = TensorDescriptor::new_strided(
                    vec![1],
                    vec![1],
                    0,
                    dtype,
                    Layout::Strided,
                    DeviceId::CPU,
                    StreamId::DEFAULT,
                )?;
                Ok(value(descriptor, 16, "2")?.canonical_bytes())
            })
            .collect::<Result<BTreeSet<_>, Box<dyn Error>>>()?;
        assert_eq!(dtype_projections.len(), dtypes.len());

        let device_projections = DeviceKind::ALL
            .into_iter()
            .map(|kind| -> Result<Vec<u8>, Box<dyn Error>> {
                let descriptor = TensorDescriptor::new_strided(
                    vec![1],
                    vec![1],
                    0,
                    DType::F32,
                    Layout::Strided,
                    DeviceId::new(kind, 0),
                    StreamId::DEFAULT,
                )?;
                Ok(value(descriptor, 16, "2")?.canonical_bytes())
            })
            .collect::<Result<BTreeSet<_>, Box<dyn Error>>>()?;
        assert_eq!(device_projections.len(), DeviceKind::ALL.len());

        let ordinal = value(
            strided(
                vec![1],
                vec![1],
                0,
                DType::F32,
                Layout::Strided,
                DeviceId::new(DeviceKind::Cpu, 3),
                StreamId::DEFAULT,
            )?,
            16,
            "2",
        )?
        .canonical_bytes();
        assert_ne!(baseline, ordinal);

        let stream = value(
            strided(
                vec![1],
                vec![1],
                0,
                DType::F32,
                Layout::Strided,
                DeviceId::CPU,
                StreamId::new(9),
            )?,
            16,
            "2",
        )?
        .canonical_bytes();
        assert_ne!(baseline, stream);
        assert_ne!(
            baseline,
            value(baseline_descriptor.clone(), 17, "2")?.canonical_bytes()
        );
        assert_ne!(
            baseline,
            value(baseline_descriptor, 16, "3")?.canonical_bytes()
        );
        Ok(())
    }
}
