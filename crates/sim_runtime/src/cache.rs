use crate::NativeHandleLease;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sim_nodes::NativeValue;
use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

pub const CACHE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CanonicalClipCacheIdentitiesWire")]
pub struct CanonicalClipCacheIdentities {
    tokenizer: String,
    architecture: String,
    artifact: String,
    model: String,
    patch: String,
    execution: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CanonicalVaeCacheIdentitiesWire")]
pub struct CanonicalVaeCacheIdentities {
    identity: String,
    artifact: String,
    patch: String,
    execution: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CanonicalConditioningCacheIdentitiesWire")]
pub struct CanonicalConditioningCacheIdentities {
    conditioning: String,
    guidance: String,
    model_patch: String,
    model_execution: String,
    control: String,
    execution: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CanonicalNativeDiffusionCacheIdentitiesWire")]
pub struct CanonicalNativeDiffusionCacheIdentities {
    model_digest: String,
    tokenizer_digest: String,
    clip: CanonicalClipCacheIdentities,
    vae: CanonicalVaeCacheIdentities,
    conditioning: CanonicalConditioningCacheIdentities,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalClipCacheIdentitiesWire {
    tokenizer: String,
    architecture: String,
    artifact: String,
    model: String,
    patch: String,
    execution: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalVaeCacheIdentitiesWire {
    identity: String,
    artifact: String,
    patch: String,
    execution: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalConditioningCacheIdentitiesWire {
    conditioning: String,
    guidance: String,
    model_patch: String,
    model_execution: String,
    control: String,
    execution: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalNativeDiffusionCacheIdentitiesWire {
    model_digest: String,
    tokenizer_digest: String,
    clip: CanonicalClipCacheIdentities,
    vae: CanonicalVaeCacheIdentities,
    conditioning: CanonicalConditioningCacheIdentities,
}

impl TryFrom<CanonicalClipCacheIdentitiesWire> for CanonicalClipCacheIdentities {
    type Error = NativeCacheError;

    fn try_from(value: CanonicalClipCacheIdentitiesWire) -> Result<Self, Self::Error> {
        Self::checked(
            value.tokenizer,
            value.architecture,
            value.artifact,
            value.model,
            value.patch,
            value.execution,
        )
    }
}

impl TryFrom<CanonicalVaeCacheIdentitiesWire> for CanonicalVaeCacheIdentities {
    type Error = NativeCacheError;

    fn try_from(value: CanonicalVaeCacheIdentitiesWire) -> Result<Self, Self::Error> {
        Self::checked(value.identity, value.artifact, value.patch, value.execution)
    }
}

impl TryFrom<CanonicalConditioningCacheIdentitiesWire> for CanonicalConditioningCacheIdentities {
    type Error = NativeCacheError;

    fn try_from(value: CanonicalConditioningCacheIdentitiesWire) -> Result<Self, Self::Error> {
        let identities = Self::checked(
            value.conditioning,
            value.guidance,
            value.model_patch,
            value.model_execution,
            value.control,
        )?;
        if identities.execution != value.execution {
            return Err(NativeCacheError::DependencyIdentityMismatch);
        }
        Ok(identities)
    }
}

impl CanonicalNativeDiffusionCacheIdentities {
    pub fn checked(
        model_digest: impl Into<String>,
        tokenizer_digest: impl Into<String>,
        clip: CanonicalClipCacheIdentities,
        vae: CanonicalVaeCacheIdentities,
        conditioning: CanonicalConditioningCacheIdentities,
    ) -> Result<Self, NativeCacheError> {
        let model_digest = model_digest.into();
        let tokenizer_digest = tokenizer_digest.into();
        if !is_sha256(&model_digest)
            || !is_sha256(&tokenizer_digest)
            || clip
                .artifact_digests()
                .into_values()
                .chain(vae.artifact_digests().into_values())
                .chain(conditioning.artifact_digests().into_values())
                .any(|digest| !is_sha256(&digest))
        {
            return Err(NativeCacheError::InvalidDependencyIdentity);
        }
        let canonical_conditioning = CanonicalConditioningCacheIdentities::checked(
            conditioning.conditioning(),
            conditioning.guidance(),
            conditioning.model_patch(),
            conditioning.model_execution(),
            conditioning.control(),
        )?;
        if canonical_conditioning != conditioning
            || clip.artifact() != model_digest
            || vae.artifact() != model_digest
            || clip.tokenizer() != tokenizer_digest
        {
            return Err(NativeCacheError::DependencyIdentityMismatch);
        }
        Ok(Self {
            model_digest,
            tokenizer_digest,
            clip,
            vae,
            conditioning,
        })
    }

    pub fn model_digest(&self) -> &str {
        &self.model_digest
    }

    pub fn tokenizer_digest(&self) -> &str {
        &self.tokenizer_digest
    }

    pub fn clip(&self) -> &CanonicalClipCacheIdentities {
        &self.clip
    }

    pub fn vae(&self) -> &CanonicalVaeCacheIdentities {
        &self.vae
    }

    pub fn conditioning(&self) -> &CanonicalConditioningCacheIdentities {
        &self.conditioning
    }

    pub fn artifact_digests(&self) -> BTreeMap<String, String> {
        let mut digests = self.clip.artifact_digests();
        digests.extend(self.vae.artifact_digests());
        digests.extend(self.conditioning.artifact_digests());
        digests.insert("model.safetensors".to_owned(), self.model_digest.clone());
        digests.insert("tokenizer.sd1".to_owned(), self.tokenizer_digest.clone());
        digests
    }

    pub fn require_exact_match(&self, actual: &Self) -> Result<(), NativeCacheError> {
        if self == actual {
            Ok(())
        } else {
            Err(NativeCacheError::DependencyIdentityMismatch)
        }
    }
}

impl TryFrom<CanonicalNativeDiffusionCacheIdentitiesWire>
    for CanonicalNativeDiffusionCacheIdentities
{
    type Error = NativeCacheError;

    fn try_from(value: CanonicalNativeDiffusionCacheIdentitiesWire) -> Result<Self, Self::Error> {
        Self::checked(
            value.model_digest,
            value.tokenizer_digest,
            value.clip,
            value.vae,
            value.conditioning,
        )
    }
}

impl CanonicalConditioningCacheIdentities {
    pub fn checked(
        conditioning: impl Into<String>,
        guidance: impl Into<String>,
        model_patch: impl Into<String>,
        model_execution: impl Into<String>,
        control: impl Into<String>,
    ) -> Result<Self, NativeCacheError> {
        let conditioning = conditioning.into();
        let guidance = guidance.into();
        let model_patch = model_patch.into();
        let model_execution = model_execution.into();
        let control = control.into();
        if [
            conditioning.as_str(),
            guidance.as_str(),
            model_patch.as_str(),
            model_execution.as_str(),
            control.as_str(),
        ]
        .into_iter()
        .any(|digest| !is_sha256(digest))
        {
            return Err(NativeCacheError::InvalidDependencyIdentity);
        }
        let mut hasher = Sha256::new();
        hasher.update(b"zed.sim.conditioning-execution.v1\0");
        for digest in [
            conditioning.as_str(),
            guidance.as_str(),
            model_patch.as_str(),
            model_execution.as_str(),
            control.as_str(),
        ] {
            hasher.update(digest.as_bytes());
            hasher.update([0]);
        }
        Ok(Self {
            conditioning,
            guidance,
            model_patch,
            model_execution,
            control,
            execution: format!("{:x}", hasher.finalize()),
        })
    }

    pub fn artifact_digests(&self) -> BTreeMap<String, String> {
        self.ordered()
            .into_iter()
            .map(|(name, digest)| (name.to_owned(), digest.to_owned()))
            .collect()
    }

    pub fn conditioning(&self) -> &str {
        &self.conditioning
    }

    pub fn guidance(&self) -> &str {
        &self.guidance
    }

    pub fn model_patch(&self) -> &str {
        &self.model_patch
    }

    pub fn model_execution(&self) -> &str {
        &self.model_execution
    }

    pub fn control(&self) -> &str {
        &self.control
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    pub fn require_exact_match(&self, actual: &Self) -> Result<(), NativeCacheError> {
        if self == actual {
            Ok(())
        } else {
            Err(NativeCacheError::DependencyIdentityMismatch)
        }
    }

    fn ordered(&self) -> [(&'static str, &str); 6] {
        [
            ("conditioning.abi", &self.conditioning),
            ("conditioning.control", &self.control),
            ("conditioning.execution", &self.execution),
            ("conditioning.guidance", &self.guidance),
            ("conditioning.model-execution", &self.model_execution),
            ("conditioning.model-patch", &self.model_patch),
        ]
    }
}

impl CanonicalVaeCacheIdentities {
    pub fn checked(
        identity: impl Into<String>,
        artifact: impl Into<String>,
        patch: impl Into<String>,
        execution: impl Into<String>,
    ) -> Result<Self, NativeCacheError> {
        let identities = Self {
            identity: identity.into(),
            artifact: artifact.into(),
            patch: patch.into(),
            execution: execution.into(),
        };
        if identities
            .ordered()
            .into_iter()
            .any(|(_, digest)| !is_sha256(digest))
        {
            return Err(NativeCacheError::InvalidDependencyIdentity);
        }
        Ok(identities)
    }

    pub fn artifact_digests(&self) -> BTreeMap<String, String> {
        self.ordered()
            .into_iter()
            .map(|(name, digest)| (name.to_owned(), digest.to_owned()))
            .collect()
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn artifact(&self) -> &str {
        &self.artifact
    }

    pub fn patch(&self) -> &str {
        &self.patch
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    pub fn require_exact_match(&self, actual: &Self) -> Result<(), NativeCacheError> {
        if self == actual {
            Ok(())
        } else {
            Err(NativeCacheError::DependencyIdentityMismatch)
        }
    }

    fn ordered(&self) -> [(&'static str, &str); 4] {
        [
            ("vae.artifact", &self.artifact),
            ("vae.execution", &self.execution),
            ("vae.identity", &self.identity),
            ("vae.patch", &self.patch),
        ]
    }
}

impl CanonicalClipCacheIdentities {
    pub fn checked(
        tokenizer: impl Into<String>,
        architecture: impl Into<String>,
        artifact: impl Into<String>,
        model: impl Into<String>,
        patch: impl Into<String>,
        execution: impl Into<String>,
    ) -> Result<Self, NativeCacheError> {
        let identities = Self {
            tokenizer: tokenizer.into(),
            architecture: architecture.into(),
            artifact: artifact.into(),
            model: model.into(),
            patch: patch.into(),
            execution: execution.into(),
        };
        if identities
            .ordered()
            .into_iter()
            .any(|(_, digest)| !is_sha256(digest))
        {
            return Err(NativeCacheError::InvalidDependencyIdentity);
        }
        Ok(identities)
    }

    pub fn artifact_digests(&self) -> BTreeMap<String, String> {
        self.ordered()
            .into_iter()
            .map(|(name, digest)| (name.to_owned(), digest.to_owned()))
            .collect()
    }

    pub fn tokenizer(&self) -> &str {
        &self.tokenizer
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn artifact(&self) -> &str {
        &self.artifact
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn patch(&self) -> &str {
        &self.patch
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    fn ordered(&self) -> [(&'static str, &str); 6] {
        [
            ("clip.architecture", &self.architecture),
            ("clip.artifact", &self.artifact),
            ("clip.execution", &self.execution),
            ("clip.model", &self.model),
            ("clip.patch", &self.patch),
            ("clip.tokenizer", &self.tokenizer),
        ]
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CacheKey {
    pub node_class: String,
    pub implementation_version: String,
    pub demanded_inputs: Vec<(String, String)>,
    #[serde(default)]
    pub demanded_dependencies: Vec<(String, String)>,
    pub artifact_digests: Vec<(String, String)>,
    pub backend: String,
    pub dtype_policy: String,
    pub plugin_digest: Option<String>,
    pub rng_phase: Option<String>,
    pub configuration_token: String,
    pub registry_version: String,
    pub change_token: String,
}

impl CacheKey {
    #[allow(clippy::too_many_arguments)]
    pub fn from_inputs(
        node_class: impl Into<String>,
        implementation_version: impl Into<String>,
        inputs: &BTreeMap<String, NativeValue>,
        artifact_digests: BTreeMap<String, String>,
        backend: impl Into<String>,
        dtype_policy: impl Into<String>,
        plugin_digest: Option<String>,
        rng_phase: Option<String>,
        configuration_token: impl Into<String>,
        registry_version: impl Into<String>,
        change_token: impl Into<String>,
    ) -> Result<Self, NativeCacheError> {
        Self::from_inputs_with_dependencies(
            node_class,
            implementation_version,
            inputs,
            BTreeMap::new(),
            artifact_digests,
            backend,
            dtype_policy,
            plugin_digest,
            rng_phase,
            configuration_token,
            registry_version,
            change_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_inputs_with_dependencies(
        node_class: impl Into<String>,
        implementation_version: impl Into<String>,
        inputs: &BTreeMap<String, NativeValue>,
        demanded_dependencies: BTreeMap<String, String>,
        artifact_digests: BTreeMap<String, String>,
        backend: impl Into<String>,
        dtype_policy: impl Into<String>,
        plugin_digest: Option<String>,
        rng_phase: Option<String>,
        configuration_token: impl Into<String>,
        registry_version: impl Into<String>,
        change_token: impl Into<String>,
    ) -> Result<Self, NativeCacheError> {
        let demanded_inputs = inputs
            .iter()
            .map(|(name, value)| {
                let value = serde_json::to_value(value)
                    .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
                Ok((name.clone(), canonical_json(&value)?))
            })
            .collect::<Result<Vec<_>, NativeCacheError>>()?;
        let node_class = node_class.into();
        let implementation_version = implementation_version.into();
        let backend = backend.into();
        let dtype_policy = dtype_policy.into();
        let configuration_token = configuration_token.into();
        let registry_version = registry_version.into();
        let change_token = change_token.into();
        for (name, value) in [
            ("node class", node_class.as_str()),
            ("implementation version", implementation_version.as_str()),
            ("backend", backend.as_str()),
            ("dtype policy", dtype_policy.as_str()),
            ("configuration token", configuration_token.as_str()),
            ("registry version", registry_version.as_str()),
            ("change token", change_token.as_str()),
        ] {
            if value.is_empty() {
                return Err(NativeCacheError::InvalidDimension(name));
            }
        }
        if demanded_dependencies
            .iter()
            .chain(artifact_digests.iter())
            .any(|(name, identity)| name.is_empty() || identity.is_empty())
            || plugin_digest.as_deref().is_some_and(str::is_empty)
            || rng_phase.as_deref().is_some_and(str::is_empty)
        {
            return Err(NativeCacheError::InvalidDependencyIdentity);
        }
        Ok(Self {
            node_class,
            implementation_version,
            demanded_inputs,
            demanded_dependencies: demanded_dependencies.into_iter().collect(),
            artifact_digests: artifact_digests.into_iter().collect(),
            backend,
            dtype_policy,
            plugin_digest,
            rng_phase,
            configuration_token,
            registry_version,
            change_token,
        })
    }

    pub fn identity(&self) -> Result<String, NativeCacheError> {
        let value = serde_json::to_value(self)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        let canonical = canonical_json(&value)?;
        Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheEntry {
    pub outputs: Vec<NativeValue>,
    pub ui: Option<Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub schema_version: u16,
    pub entry_count: usize,
    pub registry_versions: Vec<String>,
    pub artifact_digests: Vec<(String, String)>,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeCacheError {
    #[error("cache capacity must be non-zero")]
    ZeroCapacity,
    #[error("cache value could not be canonicalized: {0}")]
    Canonicalization(String),
    #[error("cache {0} must be non-empty")]
    InvalidDimension(&'static str),
    #[error("cache dependency identities must have non-empty names and values")]
    InvalidDependencyIdentity,
    #[error("cache dependency identities do not match")]
    DependencyIdentityMismatch,
}

#[derive(Clone, Debug)]
struct CacheRecord {
    entry: CacheEntry,
    handle_lease: Option<NativeHandleLease>,
}

#[derive(Clone, Debug)]
pub struct NativeCache {
    maximum_entries: usize,
    entries: BTreeMap<CacheKey, CacheRecord>,
    least_recently_used: VecDeque<CacheKey>,
}

impl NativeCache {
    pub fn new(maximum_entries: usize) -> Result<Self, NativeCacheError> {
        if maximum_entries == 0 {
            return Err(NativeCacheError::ZeroCapacity);
        }
        Ok(Self {
            maximum_entries,
            entries: BTreeMap::new(),
            least_recently_used: VecDeque::new(),
        })
    }

    pub fn get(&mut self, key: &CacheKey) -> Option<CacheEntry> {
        self.get_with_handle_lease(key).map(|(entry, _)| entry)
    }

    pub(crate) fn get_with_handle_lease(
        &mut self,
        key: &CacheKey,
    ) -> Option<(CacheEntry, Option<NativeHandleLease>)> {
        let record = self.entries.get(key)?;
        let entry = record.entry.clone();
        let lease = record.handle_lease.clone();
        self.touch(key);
        Some((entry, lease))
    }

    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) -> bool {
        self.insert_with_handle_lease(key, entry, None)
    }

    pub(crate) fn insert_with_handle_lease(
        &mut self,
        key: CacheKey,
        entry: CacheEntry,
        handle_lease: Option<NativeHandleLease>,
    ) -> bool {
        if !cache_lease_covers_entry(&entry, handle_lease.as_ref()) {
            return false;
        }
        self.entries.insert(
            key.clone(),
            CacheRecord {
                entry,
                handle_lease,
            },
        );
        self.touch(&key);
        while self.entries.len() > self.maximum_entries {
            let Some(oldest) = self.least_recently_used.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        true
    }

    pub(crate) fn insert_batch_with_handle_leases(
        &mut self,
        entries: Vec<(CacheKey, CacheEntry, Option<NativeHandleLease>)>,
    ) -> bool {
        if entries
            .iter()
            .any(|(_, entry, lease)| !cache_lease_covers_entry(entry, lease.as_ref()))
        {
            return false;
        }
        for (key, entry, lease) in entries {
            if !self.insert_with_handle_lease(key, entry, lease) {
                return false;
            }
        }
        true
    }

    pub(crate) fn remove(&mut self, key: &CacheKey) -> bool {
        self.least_recently_used
            .retain(|candidate| candidate != key);
        self.entries.remove(key).is_some()
    }

    pub fn invalidate_node(&mut self, node_class: &str) -> usize {
        self.retain(|key| key.node_class != node_class)
    }

    pub fn invalidate_registry(&mut self, registry_version: &str) -> usize {
        self.retain(|key| key.registry_version == registry_version)
    }

    pub fn invalidate_artifact(&mut self, path: &str, current_digest: &str) -> usize {
        self.retain(|key| {
            key.artifact_digests
                .iter()
                .all(|(dependency_path, digest)| {
                    dependency_path != path || digest == current_digest
                })
        })
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.least_recently_used.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn metadata(&self) -> CacheMetadata {
        let registry_versions = self
            .entries
            .keys()
            .map(|key| key.registry_version.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let artifact_digests = self
            .entries
            .keys()
            .flat_map(|key| key.artifact_digests.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        CacheMetadata {
            schema_version: CACHE_SCHEMA_VERSION,
            entry_count: self.entries.len(),
            registry_versions,
            artifact_digests,
        }
    }

    fn touch(&mut self, key: &CacheKey) {
        self.least_recently_used
            .retain(|candidate| candidate != key);
        self.least_recently_used.push_back(key.clone());
    }

    fn retain(&mut self, mut keep: impl FnMut(&CacheKey) -> bool) -> usize {
        let before = self.entries.len();
        self.entries.retain(|key, _| keep(key));
        self.least_recently_used
            .retain(|key| self.entries.contains_key(key));
        before.saturating_sub(self.entries.len())
    }
}

fn cache_lease_covers_entry(entry: &CacheEntry, lease: Option<&NativeHandleLease>) -> bool {
    let contains_handles = entry.outputs.iter().any(native_value_contains_handle);
    match (contains_handles, lease) {
        (false, None) => true,
        (true, Some(lease)) => lease.covers_values(&entry.outputs),
        _ => false,
    }
}

fn native_value_contains_handle(value: &NativeValue) -> bool {
    match value {
        NativeValue::Handle { .. } => true,
        NativeValue::List { values } => values.iter().any(native_value_contains_handle),
        NativeValue::Primitive { .. } | NativeValue::PreservedUnknown { .. } => false,
    }
}

pub fn canonical_json(value: &Value) -> Result<String, NativeCacheError> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string())),
        Value::Array(values) => {
            let values = values
                .iter()
                .map(canonical_json)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", values.join(",")))
        }
        Value::Object(values) => {
            let mut fields = values
                .iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{}:{}",
                        serde_json::to_string(key).map_err(|error| {
                            NativeCacheError::Canonicalization(error.to_string())
                        })?,
                        canonical_json(value)?
                    ))
                })
                .collect::<Result<Vec<_>, NativeCacheError>>()?;
            fields.sort();
            Ok(format!("{{{}}}", fields.join(",")))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    fn native(value: Value) -> NativeValue {
        NativeValue::PreservedUnknown {
            type_name: "zed.json@1".to_owned(),
            value,
        }
    }

    fn key(node: &str, input: Value) -> Result<CacheKey, NativeCacheError> {
        CacheKey::from_inputs(
            node,
            "1",
            &BTreeMap::from([("input".to_owned(), native(input))]),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )
    }

    #[test]
    fn val_domain_004_canonical_inputs_ignore_object_insertion_order()
    -> Result<(), NativeCacheError> {
        let first = json!({"b": 2, "a": 1});
        let second: Value = serde_json::from_str(r#"{"a":1,"b":2}"#)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        assert_eq!(key("Node", first)?, key("Node", second)?);
        Ok(())
    }

    #[test]
    fn val_domain_004_lru_and_targeted_invalidation_are_deterministic()
    -> Result<(), NativeCacheError> {
        let mut cache = NativeCache::new(2)?;
        let first = key("First", json!(1))?;
        let second = key("Second", json!(2))?;
        let third = key("Third", json!(3))?;
        cache.insert(
            first.clone(),
            CacheEntry {
                outputs: vec![native(json!(1))],
                ui: None,
            },
        );
        cache.insert(
            second.clone(),
            CacheEntry {
                outputs: vec![native(json!(2))],
                ui: None,
            },
        );
        assert!(cache.get(&first).is_some());
        cache.insert(
            third.clone(),
            CacheEntry {
                outputs: vec![native(json!(3))],
                ui: None,
            },
        );
        assert!(cache.get(&second).is_none());
        assert_eq!(cache.invalidate_node("First"), 1);
        assert!(cache.get(&third).is_some());
        Ok(())
    }

    #[test]
    fn val_domain_004_dependency_identities_are_canonical_and_validated()
    -> Result<(), NativeCacheError> {
        let inputs = BTreeMap::from([("value".to_owned(), native(json!(7)))]);
        let first = CacheKey::from_inputs_with_dependencies(
            "Output",
            "1",
            &inputs,
            BTreeMap::from([("value".to_owned(), "source-v1".to_owned())]),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        let second = CacheKey::from_inputs_with_dependencies(
            "Output",
            "1",
            &inputs,
            BTreeMap::from([("value".to_owned(), "source-v2".to_owned())]),
            BTreeMap::new(),
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        assert_ne!(first, second);
        assert_ne!(first.identity()?, second.identity()?);
        assert!(matches!(
            CacheKey::from_inputs_with_dependencies(
                "Output",
                "1",
                &inputs,
                BTreeMap::from([("value".to_owned(), String::new())]),
                BTreeMap::new(),
                "cpu",
                "f32",
                None,
                None,
                "config-v1",
                "registry-v1",
                "stable",
            ),
            Err(NativeCacheError::InvalidDependencyIdentity)
        ));
        Ok(())
    }

    #[test]
    fn structured_dotted_dependency_identity_is_part_of_the_cache_key()
    -> Result<(), NativeCacheError> {
        let inputs = BTreeMap::from([
            ("resize_type".to_owned(), native(json!("match size"))),
            ("resize_type.crop".to_owned(), native(json!("center"))),
            (
                "resize_type.match".to_owned(),
                native(json!("resolved-image")),
            ),
        ]);
        let key = |dependency: &str| {
            CacheKey::from_inputs_with_dependencies(
                "ResizeImageMaskNode",
                "1",
                &inputs,
                BTreeMap::from([("resize_type.match".to_owned(), dependency.to_owned())]),
                BTreeMap::new(),
                "cpu",
                "f32",
                None,
                None,
                "config-v1",
                "registry-v1",
                "stable",
            )
        };
        let first = key("image-source-v1")?;
        let second = key("image-source-v2")?;
        assert_ne!(first, second);
        assert_ne!(first.identity()?, second.identity()?);
        Ok(())
    }

    #[test]
    fn native_model_resource_execution_cache_binds_order_identity_and_eviction()
    -> Result<(), NativeCacheError> {
        let resource_key = |sources: [&str; 2], semantic_digest: &str| {
            CacheKey::from_inputs_with_dependencies(
                "DualCLIPLoader",
                "native-model-resource-v1",
                &BTreeMap::from([("sources".to_owned(), native(json!(sources)))]),
                BTreeMap::new(),
                BTreeMap::from([(
                    "model_resource.semantic".to_owned(),
                    semantic_digest.to_owned(),
                )]),
                "cpu",
                "f32",
                None,
                None,
                "config-v1",
                "registry-v1",
                "stable",
            )
        };
        let first = resource_key(
            ["clip-l.safetensors", "clip-g.safetensors"],
            &"1".repeat(64),
        )?;
        let reordered = resource_key(
            ["clip-g.safetensors", "clip-l.safetensors"],
            &"1".repeat(64),
        )?;
        let changed = resource_key(
            ["clip-l.safetensors", "clip-g.safetensors"],
            &"2".repeat(64),
        )?;
        assert_ne!(first.identity()?, reordered.identity()?);
        assert_ne!(first.identity()?, changed.identity()?);

        let mut cache = NativeCache::new(1)?;
        let first_entry = CacheEntry {
            outputs: vec![native(json!("first"))],
            ui: None,
        };
        assert!(cache.insert(first.clone(), first_entry.clone()));
        assert_eq!(cache.get(&first), Some(first_entry));
        assert!(cache.insert(
            changed.clone(),
            CacheEntry {
                outputs: vec![native(json!("changed"))],
                ui: None,
            },
        ));
        assert!(cache.get(&first).is_none());
        assert!(cache.get(&changed).is_some());
        Ok(())
    }

    pub(crate) fn val_domain_004_cache_case_results()
    -> Result<Vec<(&'static str, bool)>, NativeCacheError> {
        val_domain_004_canonical_inputs_ignore_object_insertion_order()?;
        val_domain_004_lru_and_targeted_invalidation_are_deterministic()?;
        val_domain_004_dependency_identities_are_canonical_and_validated()?;
        Ok(vec![
            ("cache_canonical_declared_inputs", true),
            ("cache_lru_targeted_invalidation", true),
            ("cache_demanded_dependency_identity", true),
        ])
    }

    #[test]
    fn canonical_clip_cache_identities_bind_every_execution_owner() -> Result<(), NativeCacheError>
    {
        let identities = CanonicalClipCacheIdentities::checked(
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64),
            "6".repeat(64),
        )?;
        let dependencies = identities.artifact_digests();
        assert_eq!(dependencies.len(), 6);
        assert_eq!(dependencies.get("clip.tokenizer"), Some(&"1".repeat(64)));
        let base = CacheKey::from_inputs_with_dependencies(
            "CLIPTextEncode",
            "1",
            &BTreeMap::from([("text".to_owned(), native(json!("a test")))]),
            BTreeMap::new(),
            dependencies,
            "cpu",
            "f32",
            None,
            None,
            "config-v1",
            "registry-v1",
            "stable",
        )?;
        for changed in 0..6 {
            let mut values = ["1", "2", "3", "4", "5", "6"].map(|value| value.repeat(64));
            values[changed] = "a".repeat(64);
            let changed = CanonicalClipCacheIdentities::checked(
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
                values[4].clone(),
                values[5].clone(),
            )?;
            let key = CacheKey::from_inputs_with_dependencies(
                "CLIPTextEncode",
                "1",
                &BTreeMap::from([("text".to_owned(), native(json!("a test")))]),
                BTreeMap::new(),
                changed.artifact_digests(),
                "cpu",
                "f32",
                None,
                None,
                "config-v1",
                "registry-v1",
                "stable",
            )?;
            assert_ne!(base.identity()?, key.identity()?);
        }
        assert!(
            CanonicalClipCacheIdentities::checked(
                "not-a-digest",
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64),
                "6".repeat(64),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn canonical_vae_cache_identities_bind_every_execution_owner() -> Result<(), NativeCacheError> {
        let identities = CanonicalVaeCacheIdentities::checked(
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
        )?;
        let dependencies = identities.artifact_digests();
        assert_eq!(dependencies.len(), 4);
        assert_eq!(dependencies.get("vae.identity"), Some(&"1".repeat(64)));
        assert_eq!(dependencies.get("vae.execution"), Some(&"4".repeat(64)));
        identities.require_exact_match(&identities)?;
        for changed in 0..4 {
            let mut values = ["1", "2", "3", "4"].map(|value| value.repeat(64));
            values[changed] = "a".repeat(64);
            let changed = CanonicalVaeCacheIdentities::checked(
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
            )?;
            assert_ne!(identities, changed);
            assert_eq!(
                identities.require_exact_match(&changed),
                Err(NativeCacheError::DependencyIdentityMismatch)
            );
        }
        assert!(
            CanonicalVaeCacheIdentities::checked(
                "not-a-digest",
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn canonical_conditioning_cache_identities_bind_every_execution_owner()
    -> Result<(), NativeCacheError> {
        let identities = CanonicalConditioningCacheIdentities::checked(
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64),
        )?;
        let dependencies = identities.artifact_digests();
        assert_eq!(dependencies.len(), 6);
        assert_eq!(
            dependencies.get("conditioning.control"),
            Some(&"5".repeat(64))
        );
        assert_eq!(
            dependencies.get("conditioning.execution"),
            Some(&identities.execution().to_owned())
        );
        identities.require_exact_match(&identities)?;
        for changed in 0..5 {
            let mut values = ["1", "2", "3", "4", "5"].map(|value| value.repeat(64));
            values[changed] = "a".repeat(64);
            let changed = CanonicalConditioningCacheIdentities::checked(
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                values[3].clone(),
                values[4].clone(),
            )?;
            assert_ne!(identities, changed);
            assert_eq!(
                identities.require_exact_match(&changed),
                Err(NativeCacheError::DependencyIdentityMismatch)
            );
        }
        assert!(
            CanonicalConditioningCacheIdentities::checked(
                "not-a-digest",
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64),
                "5".repeat(64),
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn canonical_native_diffusion_cache_identities_bind_one_checked_snapshot()
    -> Result<(), NativeCacheError> {
        let model = "1".repeat(64);
        let tokenizer = "2".repeat(64);
        let clip = CanonicalClipCacheIdentities::checked(
            tokenizer.clone(),
            "3".repeat(64),
            model.clone(),
            "4".repeat(64),
            "5".repeat(64),
            "6".repeat(64),
        )?;
        let vae = CanonicalVaeCacheIdentities::checked(
            "7".repeat(64),
            model.clone(),
            "8".repeat(64),
            "9".repeat(64),
        )?;
        let conditioning = CanonicalConditioningCacheIdentities::checked(
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            "d".repeat(64),
            "e".repeat(64),
        )?;
        let identities = CanonicalNativeDiffusionCacheIdentities::checked(
            model.clone(),
            tokenizer.clone(),
            clip.clone(),
            vae.clone(),
            conditioning.clone(),
        )?;
        assert_eq!(identities.model_digest(), model);
        assert_eq!(identities.tokenizer_digest(), tokenizer);
        assert_eq!(identities.clip(), &clip);
        assert_eq!(identities.vae(), &vae);
        assert_eq!(identities.conditioning(), &conditioning);
        assert_ne!(clip.patch(), vae.patch());
        assert_ne!(clip.patch(), conditioning.model_patch());
        assert_ne!(vae.patch(), conditioning.model_patch());
        let artifact_digests = identities.artifact_digests();
        assert_eq!(artifact_digests.len(), 18);
        assert_eq!(artifact_digests.get("model.safetensors"), Some(&model));
        assert_eq!(artifact_digests.get("tokenizer.sd1"), Some(&tokenizer));
        assert_eq!(
            artifact_digests.get("clip.execution"),
            Some(&"6".repeat(64))
        );
        assert_eq!(artifact_digests.get("vae.execution"), Some(&"9".repeat(64)));
        assert_eq!(
            artifact_digests.get("conditioning.execution"),
            Some(&conditioning.execution().to_owned())
        );
        let encoded = serde_json::to_vec(&identities)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        let decoded: CanonicalNativeDiffusionCacheIdentities = serde_json::from_slice(&encoded)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        identities.require_exact_match(&decoded)?;

        let mut forged = serde_json::to_value(&identities)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        forged["clip"]["execution"] = json!("NOT-A-LOWERCASE-SHA256");
        assert!(serde_json::from_value::<CanonicalNativeDiffusionCacheIdentities>(forged).is_err());
        let mut forged = serde_json::to_value(&identities)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        forged["conditioning"]["execution"] = json!("f".repeat(64));
        assert!(serde_json::from_value::<CanonicalNativeDiffusionCacheIdentities>(forged).is_err());

        let mut invalid_clip = serde_json::to_value(&clip)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        invalid_clip["execution"] = json!("NOT-A-LOWERCASE-SHA256");
        assert!(serde_json::from_value::<CanonicalClipCacheIdentities>(invalid_clip).is_err());

        let mut forged_conditioning = serde_json::to_value(&conditioning)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        forged_conditioning["execution"] = json!("f".repeat(64));
        assert!(
            serde_json::from_value::<CanonicalConditioningCacheIdentities>(forged_conditioning)
                .is_err()
        );

        let mut unknown_leaf = serde_json::to_value(&vae)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        unknown_leaf["unexpected"] = json!(true);
        assert!(serde_json::from_value::<CanonicalVaeCacheIdentities>(unknown_leaf).is_err());

        let mut unknown_aggregate = serde_json::to_value(&identities)
            .map_err(|error| NativeCacheError::Canonicalization(error.to_string()))?;
        unknown_aggregate["unexpected"] = json!(true);
        assert!(
            serde_json::from_value::<CanonicalNativeDiffusionCacheIdentities>(unknown_aggregate)
                .is_err()
        );

        for changed in 0..5 {
            let changed_model = if changed == 0 {
                "f".repeat(64)
            } else {
                model.clone()
            };
            let changed_tokenizer = if changed == 1 {
                "f".repeat(64)
            } else {
                tokenizer.clone()
            };
            let changed_clip = if changed == 2 {
                CanonicalClipCacheIdentities::checked(
                    changed_tokenizer.clone(),
                    "3".repeat(64),
                    changed_model.clone(),
                    "4".repeat(64),
                    "5".repeat(64),
                    "f".repeat(64),
                )?
            } else {
                CanonicalClipCacheIdentities::checked(
                    changed_tokenizer.clone(),
                    "3".repeat(64),
                    changed_model.clone(),
                    "4".repeat(64),
                    "5".repeat(64),
                    "6".repeat(64),
                )?
            };
            let changed_vae = CanonicalVaeCacheIdentities::checked(
                "7".repeat(64),
                changed_model.clone(),
                "8".repeat(64),
                if changed == 3 {
                    "f".repeat(64)
                } else {
                    "9".repeat(64)
                },
            )?;
            let changed_conditioning = CanonicalConditioningCacheIdentities::checked(
                "a".repeat(64),
                "b".repeat(64),
                "c".repeat(64),
                "d".repeat(64),
                if changed == 4 {
                    "f".repeat(64)
                } else {
                    "e".repeat(64)
                },
            )?;
            let changed_identities = CanonicalNativeDiffusionCacheIdentities::checked(
                changed_model,
                changed_tokenizer,
                changed_clip,
                changed_vae,
                changed_conditioning,
            )?;
            assert_eq!(
                identities.require_exact_match(&changed_identities),
                Err(NativeCacheError::DependencyIdentityMismatch)
            );
        }
        assert_eq!(
            CanonicalNativeDiffusionCacheIdentities::checked(
                "F".repeat(64),
                tokenizer.clone(),
                clip.clone(),
                vae.clone(),
                conditioning.clone(),
            ),
            Err(NativeCacheError::InvalidDependencyIdentity)
        );
        let clip_tokenizer_mismatch = CanonicalClipCacheIdentities::checked(
            "f".repeat(64),
            clip.architecture(),
            clip.artifact(),
            clip.model(),
            clip.patch(),
            clip.execution(),
        )?;
        assert_eq!(
            CanonicalNativeDiffusionCacheIdentities::checked(
                model.clone(),
                tokenizer.clone(),
                clip_tokenizer_mismatch,
                vae.clone(),
                conditioning.clone(),
            ),
            Err(NativeCacheError::DependencyIdentityMismatch)
        );
        let clip_artifact_mismatch = CanonicalClipCacheIdentities::checked(
            clip.tokenizer(),
            clip.architecture(),
            "f".repeat(64),
            clip.model(),
            clip.patch(),
            clip.execution(),
        )?;
        assert_eq!(
            CanonicalNativeDiffusionCacheIdentities::checked(
                model.clone(),
                tokenizer.clone(),
                clip_artifact_mismatch,
                vae.clone(),
                conditioning.clone(),
            ),
            Err(NativeCacheError::DependencyIdentityMismatch)
        );
        let vae_artifact_mismatch = CanonicalVaeCacheIdentities::checked(
            vae.identity(),
            "f".repeat(64),
            vae.patch(),
            vae.execution(),
        )?;
        assert_eq!(
            CanonicalNativeDiffusionCacheIdentities::checked(
                model,
                tokenizer,
                clip,
                vae_artifact_mismatch,
                conditioning,
            ),
            Err(NativeCacheError::DependencyIdentityMismatch)
        );
        Ok(())
    }
}
