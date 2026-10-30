use std::{collections::BTreeMap, error::Error, fmt};

use crate::team::{NostrEventId, NostrPublicKey};

pub const MAX_MANAGED_AGENT_GENERATION: u64 = (1_u64 << 53) - 1;

const MAX_RUNTIME_ID_BYTES: usize = 128;
const MAX_PROVIDER_ID_BYTES: usize = 128;
const MAX_MODEL_ID_BYTES: usize = 16_384;
const MAX_ENVIRONMENT_VARIABLES: usize = 256;
const MAX_ENVIRONMENT_VARIABLE_NAME_BYTES: usize = 256;
const MAX_REFERENCE_ID_BYTES: usize = 512;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeId(String);

impl RuntimeId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ManagedAgentConfigError> {
        let value = value.into();
        if !valid_component_id(&value, MAX_RUNTIME_ID_BYTES) {
            return Err(ManagedAgentConfigError::InvalidRuntime);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProviderId(String);

impl ProviderId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ManagedAgentConfigError> {
        let value = value.into();
        if !valid_component_id(&value, MAX_PROVIDER_ID_BYTES) {
            return Err(ManagedAgentConfigError::InvalidProvider);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelId(String);

impl ModelId {
    pub fn parse(value: impl Into<String>) -> Result<Self, ManagedAgentConfigError> {
        let value = value.into();
        if value.trim().is_empty()
            || value.len() > MAX_MODEL_ID_BYTES
            || value.contains('\0')
            || value.chars().any(char::is_control)
        {
            return Err(ManagedAgentConfigError::InvalidModel);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EnvironmentVariableName(String);

impl EnvironmentVariableName {
    pub fn parse(value: impl Into<String>) -> Result<Self, ManagedAgentConfigError> {
        let value = value.into();
        let mut bytes = value.bytes();
        let Some(first) = bytes.next() else {
            return Err(ManagedAgentConfigError::InvalidEnvironmentVariable);
        };
        if value.len() > MAX_ENVIRONMENT_VARIABLE_NAME_BYTES
            || !(first.is_ascii_alphabetic() || first == b'_')
            || bytes.any(|byte| !(byte.is_ascii_alphanumeric() || byte == b'_'))
        {
            return Err(ManagedAgentConfigError::InvalidEnvironmentVariable);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProtectedCredentialReference(String);

impl ProtectedCredentialReference {
    pub fn parse(value: impl Into<String>) -> Result<Self, ManagedAgentConfigError> {
        let value = value.into();
        if !valid_reference_id(&value) {
            return Err(ManagedAgentConfigError::InvalidEnvironmentReference);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ProtectedCredentialReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ProtectedCredentialReference(<redacted>)")
    }
}

#[derive(Clone, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum EnvironmentReference {
    ProcessEnvironment(EnvironmentVariableName),
    ProtectedCredential(ProtectedCredentialReference),
}

impl fmt::Debug for EnvironmentReference {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProcessEnvironment(_) => formatter.write_str("ProcessEnvironment(<redacted>)"),
            Self::ProtectedCredential(_) => formatter.write_str("ProtectedCredential(<redacted>)"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct ManagedAgentConfiguration {
    runtime: RuntimeId,
    provider: Option<ProviderId>,
    model: Option<ModelId>,
    environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
}

impl ManagedAgentConfiguration {
    pub fn new(
        runtime: RuntimeId,
        provider: Option<ProviderId>,
        model: Option<ModelId>,
        environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
    ) -> Result<Self, ManagedAgentConfigError> {
        if environment.len() > MAX_ENVIRONMENT_VARIABLES {
            return Err(ManagedAgentConfigError::TooManyEnvironmentVariables);
        }
        Ok(Self {
            runtime,
            provider,
            model,
            environment,
        })
    }

    pub fn runtime(&self) -> &RuntimeId {
        &self.runtime
    }

    pub fn provider(&self) -> Option<&ProviderId> {
        self.provider.as_ref()
    }

    pub fn model(&self) -> Option<&ModelId> {
        self.model.as_ref()
    }

    pub fn environment(&self) -> &BTreeMap<EnvironmentVariableName, EnvironmentReference> {
        &self.environment
    }
}

impl fmt::Debug for ManagedAgentConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagedAgentConfiguration")
            .field("runtime", &self.runtime)
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("environment_entries", &self.environment.len())
            .finish()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ManagedAgentVersion {
    generation: u64,
    event_id: NostrEventId,
}

impl ManagedAgentVersion {
    pub fn new(generation: u64, event_id: NostrEventId) -> Result<Self, ManagedAgentConfigError> {
        if generation == 0 || generation > MAX_MANAGED_AGENT_GENERATION {
            return Err(ManagedAgentConfigError::InvalidGeneration);
        }
        Ok(Self {
            generation,
            event_id,
        })
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn event_id(&self) -> &NostrEventId {
        &self.event_id
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ManagedAgentState {
    Active(ManagedAgentConfiguration),
    Deleted { deleted_at: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrivateManagedAgentRecord {
    owner_public_key: NostrPublicKey,
    agent_public_key: NostrPublicKey,
    version: ManagedAgentVersion,
    previous_event_id: Option<NostrEventId>,
    state: ManagedAgentState,
}

impl PrivateManagedAgentRecord {
    pub fn new(
        owner_public_key: NostrPublicKey,
        agent_public_key: NostrPublicKey,
        initial_event_id: NostrEventId,
        configuration: ManagedAgentConfiguration,
    ) -> Result<Self, ManagedAgentConfigError> {
        Self::hydrate(
            owner_public_key,
            agent_public_key,
            ManagedAgentVersion::new(1, initial_event_id)?,
            None,
            ManagedAgentState::Active(configuration),
        )
    }

    pub fn hydrate(
        owner_public_key: NostrPublicKey,
        agent_public_key: NostrPublicKey,
        version: ManagedAgentVersion,
        previous_event_id: Option<NostrEventId>,
        state: ManagedAgentState,
    ) -> Result<Self, ManagedAgentConfigError> {
        if owner_public_key == agent_public_key {
            return Err(ManagedAgentConfigError::InvalidIdentityBinding);
        }
        if (version.generation == 1) != previous_event_id.is_none() {
            return Err(ManagedAgentConfigError::InvalidPredecessor);
        }
        if matches!(state, ManagedAgentState::Deleted { deleted_at: 0 }) {
            return Err(ManagedAgentConfigError::InvalidDeletionTime);
        }
        Ok(Self {
            owner_public_key,
            agent_public_key,
            version,
            previous_event_id,
            state,
        })
    }

    pub fn owner_public_key(&self) -> &NostrPublicKey {
        &self.owner_public_key
    }

    pub fn agent_public_key(&self) -> &NostrPublicKey {
        &self.agent_public_key
    }

    pub fn version(&self) -> &ManagedAgentVersion {
        &self.version
    }

    pub fn previous_event_id(&self) -> Option<&NostrEventId> {
        self.previous_event_id.as_ref()
    }

    pub fn state(&self) -> &ManagedAgentState {
        &self.state
    }

    pub fn replace(
        &mut self,
        expected_version: &ManagedAgentVersion,
        next_event_id: NostrEventId,
        configuration: ManagedAgentConfiguration,
    ) -> Result<(), ManagedAgentConfigError> {
        if !matches!(self.state, ManagedAgentState::Active(_)) {
            return Err(ManagedAgentConfigError::DeletedRecord);
        }
        self.transition(
            expected_version,
            next_event_id,
            ManagedAgentState::Active(configuration),
        )
    }

    pub fn delete(
        &mut self,
        expected_version: &ManagedAgentVersion,
        next_event_id: NostrEventId,
        deleted_at: u64,
    ) -> Result<(), ManagedAgentConfigError> {
        if deleted_at == 0 {
            return Err(ManagedAgentConfigError::InvalidDeletionTime);
        }
        if !matches!(self.state, ManagedAgentState::Active(_)) {
            return Err(ManagedAgentConfigError::DeletedRecord);
        }
        self.transition(
            expected_version,
            next_event_id,
            ManagedAgentState::Deleted { deleted_at },
        )
    }

    fn transition(
        &mut self,
        expected_version: &ManagedAgentVersion,
        next_event_id: NostrEventId,
        next_state: ManagedAgentState,
    ) -> Result<(), ManagedAgentConfigError> {
        if expected_version != &self.version {
            return Err(ManagedAgentConfigError::StaleVersion);
        }
        if next_event_id == self.version.event_id {
            return Err(ManagedAgentConfigError::DuplicateEventId);
        }
        let generation = self
            .version
            .generation
            .checked_add(1)
            .filter(|generation| *generation <= MAX_MANAGED_AGENT_GENERATION)
            .ok_or(ManagedAgentConfigError::GenerationExhausted)?;
        let previous_event_id = self.version.event_id.clone();
        self.version = ManagedAgentVersion {
            generation,
            event_id: next_event_id,
        };
        self.previous_event_id = Some(previous_event_id);
        self.state = next_state;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManagedAgentConfigError {
    InvalidRuntime,
    InvalidProvider,
    InvalidModel,
    InvalidEnvironmentVariable,
    InvalidEnvironmentReference,
    TooManyEnvironmentVariables,
    InvalidGeneration,
    InvalidPredecessor,
    InvalidIdentityBinding,
    InvalidDeletionTime,
    StaleVersion,
    DuplicateEventId,
    GenerationExhausted,
    DeletedRecord,
}

impl fmt::Display for ManagedAgentConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidRuntime => "invalid managed-agent runtime",
            Self::InvalidProvider => "invalid managed-agent provider",
            Self::InvalidModel => "invalid managed-agent model",
            Self::InvalidEnvironmentVariable => "invalid environment variable name",
            Self::InvalidEnvironmentReference => "invalid environment reference",
            Self::TooManyEnvironmentVariables => "too many environment variables",
            Self::InvalidGeneration => "invalid managed-agent generation",
            Self::InvalidPredecessor => "invalid managed-agent predecessor",
            Self::InvalidIdentityBinding => "invalid managed-agent identity binding",
            Self::InvalidDeletionTime => "invalid managed-agent deletion time",
            Self::StaleVersion => "managed-agent version is stale",
            Self::DuplicateEventId => "managed-agent event ID was already used",
            Self::GenerationExhausted => "managed-agent generation is exhausted",
            Self::DeletedRecord => "managed-agent record is deleted",
        };
        formatter.write_str(message)
    }
}

impl Error for ManagedAgentConfigError {}

fn valid_component_id(value: &str, maximum_bytes: usize) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    value.len() <= maximum_bytes
        && (first.is_ascii_lowercase() || first.is_ascii_digit())
        && bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-')
        })
}

fn valid_reference_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_REFERENCE_ID_BYTES
        && !value
            .chars()
            .any(|character| character.is_control() || character.is_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn public_key(byte: char) -> NostrPublicKey {
        NostrPublicKey::parse(byte.to_string().repeat(64)).expect("fixture key must be valid")
    }

    fn event_id(byte: char) -> NostrEventId {
        NostrEventId::parse(byte.to_string().repeat(64)).expect("fixture event ID must be valid")
    }

    fn configuration() -> ManagedAgentConfiguration {
        let mut environment = BTreeMap::new();
        environment.insert(
            EnvironmentVariableName::parse("ANTHROPIC_API_KEY")
                .expect("fixture environment name must be valid"),
            EnvironmentReference::ProtectedCredential(
                ProtectedCredentialReference::parse("credentials/anthropic/default")
                    .expect("fixture credential reference must be valid"),
            ),
        );
        environment.insert(
            EnvironmentVariableName::parse("HTTP_PROXY")
                .expect("fixture environment name must be valid"),
            EnvironmentReference::ProcessEnvironment(
                EnvironmentVariableName::parse("CORPORATE_HTTP_PROXY")
                    .expect("fixture source environment name must be valid"),
            ),
        );
        ManagedAgentConfiguration::new(
            RuntimeId::parse("claude-code").expect("fixture runtime must be valid"),
            Some(ProviderId::parse("anthropic").expect("fixture provider must be valid")),
            Some(ModelId::parse("claude-opus-4-1").expect("fixture model must be valid")),
            environment,
        )
        .expect("fixture configuration must be valid")
    }

    #[test]
    fn compare_and_swap_advances_generation_and_predecessor() {
        let mut record = PrivateManagedAgentRecord::new(
            public_key('1'),
            public_key('2'),
            event_id('3'),
            configuration(),
        )
        .expect("fixture record must be valid");
        let expected = record.version().clone();

        record
            .replace(&expected, event_id('4'), configuration())
            .expect("current expected version must update");

        assert_eq!(record.version().generation(), 2);
        assert_eq!(record.version().event_id(), &event_id('4'));
        assert_eq!(record.previous_event_id(), Some(&event_id('3')));
        assert_eq!(
            PrivateManagedAgentRecord::hydrate(
                record.owner_public_key().clone(),
                record.agent_public_key().clone(),
                record.version().clone(),
                record.previous_event_id().cloned(),
                record.state().clone(),
            ),
            Ok(record)
        );
    }

    #[test]
    fn stale_update_is_rejected_without_mutation() {
        let mut record = PrivateManagedAgentRecord::new(
            public_key('1'),
            public_key('2'),
            event_id('3'),
            configuration(),
        )
        .expect("fixture record must be valid");
        let initial = record.clone();
        let stale = ManagedAgentVersion::new(1, event_id('4'))
            .expect("fixture stale version must be valid");

        assert_eq!(
            record.replace(&stale, event_id('5'), configuration()),
            Err(ManagedAgentConfigError::StaleVersion)
        );
        assert_eq!(record, initial);
    }

    #[test]
    fn invalid_provider_and_model_are_rejected() {
        assert_eq!(
            ProviderId::parse("../Anthropic"),
            Err(ManagedAgentConfigError::InvalidProvider)
        );
        assert_eq!(
            ModelId::parse(" \t\n"),
            Err(ManagedAgentConfigError::InvalidModel)
        );
        assert_eq!(
            ModelId::parse("model\0override"),
            Err(ManagedAgentConfigError::InvalidModel)
        );
    }

    #[test]
    fn environment_storage_contains_only_references_and_redacts_diagnostics() {
        let configuration = configuration();
        let credential = configuration
            .environment()
            .get(
                &EnvironmentVariableName::parse("ANTHROPIC_API_KEY")
                    .expect("fixture environment name must be valid"),
            )
            .expect("fixture credential binding must exist");

        assert!(matches!(
            credential,
            EnvironmentReference::ProtectedCredential(reference)
                if reference.as_str() == "credentials/anthropic/default"
        ));
        let debug = format!("{configuration:?}");
        assert!(!debug.contains("credentials/anthropic/default"));
        assert!(!debug.contains("CORPORATE_HTTP_PROXY"));
        assert!(debug.contains("environment_entries: 2"));
    }
}
