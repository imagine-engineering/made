use std::{collections::BTreeMap, ffi::OsString, fmt, path::PathBuf};

use agent_settings::managed_agent::{
    EnvironmentReference, EnvironmentVariableName, ProtectedCredentialReference,
};
use bech32::{Bech32, Hrp};
use credentials_provider::CredentialsProvider;
use gpui::AsyncApp;
use remote::agent_provider_lifecycle::AgentProviderDeployInput;
use secp256k1::{Keypair, Secp256k1, SecretKey, XOnlyPublicKey};
use serde_json::{Map, Number, Value};
use zeroize::{Zeroize as _, Zeroizing};

const MAX_PROVIDER_CONFIG_FIELDS: usize = 20;
const MAX_PROVIDER_CONFIG_BYTES: usize = 64 * 1024;
const MAX_PROVIDER_CONFIG_KEY_BYTES: usize = 128;
const MAX_REMOTE_ENVIRONMENT_ENTRIES: usize = 256;
const MAX_RESOLVED_ENVIRONMENT_BYTES: usize = 512 * 1024;
const MAX_RESOLVED_SECRET_BYTES: usize = 64 * 1024;
const NSEC_HRP: &str = "nsec";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoteProviderConfigValue {
    String(String),
    Number(Number),
    Boolean(bool),
}

impl RemoteProviderConfigValue {
    fn into_json(self) -> Value {
        match self {
            Self::String(value) => Value::String(value),
            Self::Number(value) => Value::Number(value),
            Self::Boolean(value) => Value::Bool(value),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct RemoteProviderProjectConfiguration {
    fields: BTreeMap<String, RemoteProviderConfigValue>,
}

impl RemoteProviderProjectConfiguration {
    pub fn new(
        fields: BTreeMap<String, RemoteProviderConfigValue>,
    ) -> Result<Self, RemoteProviderConfigError> {
        if fields.len() > MAX_PROVIDER_CONFIG_FIELDS {
            return Err(RemoteProviderConfigError::TooManyProviderConfigFields);
        }
        for key in fields.keys() {
            if key.is_empty()
                || key.len() > MAX_PROVIDER_CONFIG_KEY_BYTES
                || key.chars().any(char::is_control)
            {
                return Err(RemoteProviderConfigError::InvalidProviderConfigKey);
            }
            if split_config_key(key).iter().any(|word| {
                matches!(
                    word.as_str(),
                    "secret" | "password" | "token" | "key" | "credential"
                )
            }) {
                return Err(RemoteProviderConfigError::SecretLikeProviderConfigKey);
            }
        }
        let configuration = Self { fields };
        let encoded = serde_json::to_vec(&configuration.to_json())
            .map_err(|_| RemoteProviderConfigError::InvalidProviderConfiguration)?;
        if encoded.len() > MAX_PROVIDER_CONFIG_BYTES {
            return Err(RemoteProviderConfigError::ProviderConfigurationTooLarge);
        }
        Ok(configuration)
    }

    pub fn to_json(&self) -> Value {
        Value::Object(
            self.fields
                .clone()
                .into_iter()
                .map(|(key, value)| (key, value.into_json()))
                .collect(),
        )
    }

    pub fn field_count(&self) -> usize {
        self.fields.len()
    }
}

impl fmt::Debug for RemoteProviderProjectConfiguration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteProviderProjectConfiguration")
            .field("field_count", &self.fields.len())
            .finish()
    }
}

#[derive(Clone)]
pub struct RemoteProviderSecretReferences {
    identity: ProtectedCredentialReference,
    auth_tag: Option<ProtectedCredentialReference>,
    environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
}

impl RemoteProviderSecretReferences {
    pub fn new(
        identity: ProtectedCredentialReference,
        auth_tag: Option<ProtectedCredentialReference>,
        environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
    ) -> Result<Self, RemoteProviderConfigError> {
        if environment.len() > MAX_REMOTE_ENVIRONMENT_ENTRIES {
            return Err(RemoteProviderConfigError::TooManyEnvironmentReferences);
        }
        Ok(Self {
            identity,
            auth_tag,
            environment,
        })
    }

    pub fn identity(&self) -> &ProtectedCredentialReference {
        &self.identity
    }

    pub fn auth_tag(&self) -> Option<&ProtectedCredentialReference> {
        self.auth_tag.as_ref()
    }

    pub fn environment(&self) -> &BTreeMap<EnvironmentVariableName, EnvironmentReference> {
        &self.environment
    }
}

impl fmt::Debug for RemoteProviderSecretReferences {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteProviderSecretReferences")
            .field("identity", &"<redacted-reference>")
            .field(
                "auth_tag",
                &self.auth_tag.as_ref().map(|_| "<redacted-reference>"),
            )
            .field("environment_entries", &self.environment.len())
            .finish()
    }
}

pub struct RemoteProviderDeployTemplate {
    operation_id: String,
    work_directory: PathBuf,
    agent_identity: [u8; 32],
    agent: Map<String, Value>,
    provider_configuration: RemoteProviderProjectConfiguration,
    secret_references: RemoteProviderSecretReferences,
}

impl RemoteProviderDeployTemplate {
    pub fn new(
        operation_id: impl Into<String>,
        work_directory: PathBuf,
        agent_identity: &str,
        agent: Map<String, Value>,
        provider_configuration: RemoteProviderProjectConfiguration,
        secret_references: RemoteProviderSecretReferences,
    ) -> Result<Self, RemoteProviderConfigError> {
        validate_secret_free_agent_template(&agent)?;
        Ok(Self {
            operation_id: operation_id.into(),
            work_directory,
            agent_identity: parse_agent_identity(agent_identity)?,
            agent,
            provider_configuration,
            secret_references,
        })
    }

    pub fn agent_identity(&self) -> String {
        encode_hex(&self.agent_identity)
    }

    pub fn provider_configuration(&self) -> &RemoteProviderProjectConfiguration {
        &self.provider_configuration
    }

    pub fn secret_references(&self) -> &RemoteProviderSecretReferences {
        &self.secret_references
    }

    pub async fn resolve(
        &self,
        credentials: &dyn CredentialsProvider,
        cx: &AsyncApp,
    ) -> Result<AgentProviderDeployInput, RemoteProviderConfigError> {
        self.resolve_with_environment(credentials, &SystemProcessEnvironment, cx)
            .await
    }

    pub async fn resolve_with_environment(
        &self,
        credentials: &dyn CredentialsProvider,
        process_environment: &dyn RemoteProviderProcessEnvironment,
        cx: &AsyncApp,
    ) -> Result<AgentProviderDeployInput, RemoteProviderConfigError> {
        let identity = read_credential(
            credentials,
            &self.secret_references.identity,
            CredentialPurpose::Identity,
            cx,
        )
        .await?;
        let private_key_nsec = resolve_identity(&identity, self.agent_identity)?;

        let auth_tag = if let Some(reference) = self.secret_references.auth_tag.as_ref() {
            Some(resolve_text_secret(
                read_credential(credentials, reference, CredentialPurpose::AuthTag, cx).await?,
                CredentialPurpose::AuthTag,
            )?)
        } else {
            None
        };

        let mut resolved_environment = BTreeMap::new();
        let mut environment_bytes = 0_usize;
        for (name, reference) in &self.secret_references.environment {
            let value = match reference {
                EnvironmentReference::ProcessEnvironment(source) => {
                    let value = process_environment
                        .read(source.as_str())
                        .ok_or(RemoteProviderConfigError::MissingProcessEnvironment)?;
                    let value = value
                        .into_string()
                        .map_err(|_| RemoteProviderConfigError::InvalidProcessEnvironment)?;
                    if value.is_empty() {
                        return Err(RemoteProviderConfigError::EmptyEnvironmentSecret);
                    }
                    Zeroizing::new(value)
                }
                EnvironmentReference::ProtectedCredential(reference) => resolve_text_secret(
                    read_credential(credentials, reference, CredentialPurpose::Environment, cx)
                        .await?,
                    CredentialPurpose::Environment,
                )?,
            };
            if value.len() > MAX_RESOLVED_SECRET_BYTES {
                return Err(RemoteProviderConfigError::ResolvedSecretTooLarge);
            }
            environment_bytes = environment_bytes
                .checked_add(name.as_str().len())
                .and_then(|total| total.checked_add(value.len()))
                .ok_or(RemoteProviderConfigError::ResolvedEnvironmentTooLarge)?;
            if environment_bytes > MAX_RESOLVED_ENVIRONMENT_BYTES {
                return Err(RemoteProviderConfigError::ResolvedEnvironmentTooLarge);
            }
            resolved_environment.insert(name.as_str().to_owned(), value);
        }

        let mut agent = self.agent.clone();
        agent.insert(
            "private_key_nsec".to_owned(),
            Value::String(private_key_nsec.to_string()),
        );
        if let Some(auth_tag) = auth_tag {
            agent.insert("auth_tag".to_owned(), Value::String(auth_tag.to_string()));
        }
        let environment = resolved_environment
            .into_iter()
            .map(|(name, value)| (name, Value::String(value.to_string())))
            .collect::<Map<_, _>>();
        agent.insert("env_vars".to_owned(), Value::Object(environment.clone()));
        if let Some(Value::Object(launch)) = agent.get_mut("launch") {
            launch.insert("env".to_owned(), Value::Object(environment));
        }

        Ok(AgentProviderDeployInput {
            operation_id: self.operation_id.clone(),
            work_directory: self.work_directory.clone(),
            agent: Value::Object(agent),
            provider_config: self.provider_configuration.to_json(),
        })
    }
}

impl fmt::Debug for RemoteProviderDeployTemplate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RemoteProviderDeployTemplate")
            .field("operation_id", &self.operation_id)
            .field("agent_identity", &self.agent_identity())
            .field("agent_field_count", &self.agent.len())
            .field("provider_configuration", &self.provider_configuration)
            .field("secret_references", &self.secret_references)
            .finish()
    }
}

pub trait RemoteProviderProcessEnvironment: Send + Sync {
    fn read(&self, name: &str) -> Option<OsString>;
}

pub struct SystemProcessEnvironment;

impl RemoteProviderProcessEnvironment for SystemProcessEnvironment {
    fn read(&self, name: &str) -> Option<OsString> {
        std::env::var_os(name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CredentialPurpose {
    Identity,
    AuthTag,
    Environment,
}

async fn read_credential(
    credentials: &dyn CredentialsProvider,
    reference: &ProtectedCredentialReference,
    purpose: CredentialPurpose,
    cx: &AsyncApp,
) -> Result<Zeroizing<Vec<u8>>, RemoteProviderConfigError> {
    let credential = credentials
        .read_credentials(reference.as_str(), cx)
        .await
        .map_err(|_| RemoteProviderConfigError::ProtectedStorageUnavailable)?;
    let Some((mut username, secret)) = credential else {
        return Err(match purpose {
            CredentialPurpose::Identity => RemoteProviderConfigError::MissingIdentityCredential,
            CredentialPurpose::AuthTag => RemoteProviderConfigError::MissingAuthTagCredential,
            CredentialPurpose::Environment => {
                RemoteProviderConfigError::MissingEnvironmentCredential
            }
        });
    };
    username.zeroize();
    let secret = Zeroizing::new(secret);
    if secret.len() > MAX_RESOLVED_SECRET_BYTES {
        return Err(RemoteProviderConfigError::ResolvedSecretTooLarge);
    }
    Ok(secret)
}

fn resolve_identity(
    secret: &[u8],
    expected_identity: [u8; 32],
) -> Result<Zeroizing<String>, RemoteProviderConfigError> {
    if secret.is_empty() {
        return Err(RemoteProviderConfigError::EmptyIdentityCredential);
    }
    if secret.len() != 32 {
        return Err(RemoteProviderConfigError::InvalidIdentityCredential);
    }
    let secret_key = SecretKey::from_slice(secret)
        .map_err(|_| RemoteProviderConfigError::InvalidIdentityCredential)?;
    let secp = Secp256k1::new();
    let keypair = Keypair::from_secret_key(&secp, &secret_key);
    let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
    if public_key.serialize() != expected_identity {
        return Err(RemoteProviderConfigError::IdentityCredentialMismatch);
    }
    let human_readable_part =
        Hrp::parse(NSEC_HRP).map_err(|_| RemoteProviderConfigError::InvalidIdentityCredential)?;
    bech32::encode::<Bech32>(human_readable_part, &secret_key.secret_bytes())
        .map(Zeroizing::new)
        .map_err(|_| RemoteProviderConfigError::InvalidIdentityCredential)
}

fn resolve_text_secret(
    secret: Zeroizing<Vec<u8>>,
    purpose: CredentialPurpose,
) -> Result<Zeroizing<String>, RemoteProviderConfigError> {
    if secret.is_empty() {
        return Err(match purpose {
            CredentialPurpose::Identity => RemoteProviderConfigError::EmptyIdentityCredential,
            CredentialPurpose::AuthTag => RemoteProviderConfigError::EmptyAuthTagCredential,
            CredentialPurpose::Environment => RemoteProviderConfigError::EmptyEnvironmentSecret,
        });
    }
    let value = std::str::from_utf8(secret.as_ref()).map_err(|_| match purpose {
        CredentialPurpose::Identity => RemoteProviderConfigError::InvalidIdentityCredential,
        CredentialPurpose::AuthTag => RemoteProviderConfigError::InvalidAuthTagCredential,
        CredentialPurpose::Environment => RemoteProviderConfigError::InvalidEnvironmentSecret,
    })?;
    Ok(Zeroizing::new(value.to_owned()))
}

fn validate_secret_free_agent_template(
    agent: &Map<String, Value>,
) -> Result<(), RemoteProviderConfigError> {
    if ["private_key_nsec", "auth_tag", "env_vars"]
        .iter()
        .any(|field| agent.contains_key(*field))
    {
        return Err(RemoteProviderConfigError::InlineSecretMaterial);
    }
    if let Some(launch) = agent.get("launch") {
        let Value::Object(launch) = launch else {
            return Err(RemoteProviderConfigError::InvalidAgentTemplate);
        };
        if launch.contains_key("env") {
            return Err(RemoteProviderConfigError::InlineSecretMaterial);
        }
    }
    Ok(())
}

fn parse_agent_identity(value: &str) -> Result<[u8; 32], RemoteProviderConfigError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RemoteProviderConfigError::InvalidAgentIdentity);
    }
    let mut identity = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        identity[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(identity)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => 0,
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn split_config_key(key: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let characters = key.chars().collect::<Vec<_>>();
    for (index, character) in characters.iter().copied().enumerate() {
        if matches!(character, '_' | '-' | '.') {
            if !current.is_empty() {
                words.push(current.to_lowercase());
                current.clear();
            }
            continue;
        }
        if character.is_uppercase() {
            let previous_is_lowercase = current.chars().last().is_some_and(char::is_lowercase);
            let acronym_ends = current.chars().last().is_some_and(char::is_uppercase)
                && characters
                    .get(index + 1)
                    .is_some_and(|next| next.is_lowercase());
            if previous_is_lowercase || acronym_ends {
                words.push(current.to_lowercase());
                current.clear();
            }
        }
        current.push(character);
    }
    if !current.is_empty() {
        words.push(current.to_lowercase());
    }
    words
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteProviderConfigError {
    TooManyProviderConfigFields,
    InvalidProviderConfigKey,
    SecretLikeProviderConfigKey,
    InvalidProviderConfiguration,
    ProviderConfigurationTooLarge,
    InvalidAgentIdentity,
    InvalidAgentTemplate,
    InlineSecretMaterial,
    TooManyEnvironmentReferences,
    ProtectedStorageUnavailable,
    MissingIdentityCredential,
    MissingAuthTagCredential,
    MissingEnvironmentCredential,
    EmptyIdentityCredential,
    EmptyAuthTagCredential,
    EmptyEnvironmentSecret,
    InvalidIdentityCredential,
    IdentityCredentialMismatch,
    InvalidAuthTagCredential,
    InvalidEnvironmentSecret,
    MissingProcessEnvironment,
    InvalidProcessEnvironment,
    ResolvedSecretTooLarge,
    ResolvedEnvironmentTooLarge,
}

impl fmt::Display for RemoteProviderConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::TooManyProviderConfigFields => {
                "remote provider configuration has too many fields"
            }
            Self::InvalidProviderConfigKey => {
                "remote provider configuration contains an invalid field name"
            }
            Self::SecretLikeProviderConfigKey => {
                "remote provider configuration must not contain secret-like fields"
            }
            Self::InvalidProviderConfiguration => "remote provider configuration is invalid",
            Self::ProviderConfigurationTooLarge => {
                "remote provider configuration exceeds its size limit"
            }
            Self::InvalidAgentIdentity => {
                "remote agent identity must be 64 lowercase hexadecimal characters"
            }
            Self::InvalidAgentTemplate => "remote agent launch template is invalid",
            Self::InlineSecretMaterial => {
                "remote agent launch template must contain references instead of secret values"
            }
            Self::TooManyEnvironmentReferences => {
                "remote agent launch template has too many environment references"
            }
            Self::ProtectedStorageUnavailable => "protected credential storage is unavailable",
            Self::MissingIdentityCredential => "remote agent identity credential is missing",
            Self::MissingAuthTagCredential => "remote agent authorization credential is missing",
            Self::MissingEnvironmentCredential => "remote agent environment credential is missing",
            Self::EmptyIdentityCredential => "remote agent identity credential is empty",
            Self::EmptyAuthTagCredential => "remote agent authorization credential is empty",
            Self::EmptyEnvironmentSecret => "remote agent environment credential is empty",
            Self::InvalidIdentityCredential => "remote agent identity credential is invalid",
            Self::IdentityCredentialMismatch => {
                "remote agent identity credential does not match the selected agent"
            }
            Self::InvalidAuthTagCredential => "remote agent authorization credential is invalid",
            Self::InvalidEnvironmentSecret => "remote agent environment credential is invalid",
            Self::MissingProcessEnvironment => {
                "a referenced process environment value is unavailable"
            }
            Self::InvalidProcessEnvironment => "a referenced process environment value is invalid",
            Self::ResolvedSecretTooLarge => {
                "a resolved remote agent credential exceeds its size limit"
            }
            Self::ResolvedEnvironmentTooLarge => {
                "the resolved remote agent environment exceeds its size limit"
            }
        })
    }
}

impl std::error::Error for RemoteProviderConfigError {}

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin, sync::Mutex};

    use anyhow::{Result, anyhow};
    use gpui::TestAppContext;
    use serde_json::json;

    use super::*;

    const IDENTITY_REFERENCE: &str = "credentials/agent/identity-private";
    const ENVIRONMENT_REFERENCE: &str = "credentials/provider/sentinel-environment";
    const SECRET_SENTINEL: &str = "sentinel-provider-secret";

    enum FakeCredentialState {
        Locked,
        Values(BTreeMap<String, Vec<u8>>),
    }

    struct FakeCredentialsProvider {
        state: Mutex<FakeCredentialState>,
    }

    impl FakeCredentialsProvider {
        fn locked() -> Self {
            Self {
                state: Mutex::new(FakeCredentialState::Locked),
            }
        }

        fn values(values: impl IntoIterator<Item = (String, Vec<u8>)>) -> Self {
            Self {
                state: Mutex::new(FakeCredentialState::Values(values.into_iter().collect())),
            }
        }
    }

    impl CredentialsProvider for FakeCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>> {
            Box::pin(async move {
                match &*self.state.lock().map_err(|_| anyhow!("locked state"))? {
                    FakeCredentialState::Locked => {
                        Err(anyhow!("keyring locked: {url} {SECRET_SENTINEL}"))
                    }
                    FakeCredentialState::Values(values) => Ok(values
                        .get(url)
                        .cloned()
                        .map(|secret| ("remote-agent".to_owned(), secret))),
                }
            })
        }

        fn write_credentials<'a>(
            &'a self,
            _url: &'a str,
            _username: &'a str,
            _password: &'a [u8],
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Err(anyhow!("read-only fixture")) })
        }

        fn delete_credentials<'a>(
            &'a self,
            _url: &'a str,
            _cx: &'a AsyncApp,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Err(anyhow!("read-only fixture")) })
        }
    }

    struct EmptyProcessEnvironment;

    impl RemoteProviderProcessEnvironment for EmptyProcessEnvironment {
        fn read(&self, _name: &str) -> Option<OsString> {
            None
        }
    }

    fn credential_reference(value: &str) -> ProtectedCredentialReference {
        ProtectedCredentialReference::parse(value).expect("fixture credential reference")
    }

    fn resolution_error(
        result: Result<AgentProviderDeployInput, RemoteProviderConfigError>,
    ) -> RemoteProviderConfigError {
        match result {
            Ok(_) => panic!("credential resolution unexpectedly succeeded"),
            Err(error) => error,
        }
    }

    fn identity(secret: [u8; 32]) -> String {
        let secret = SecretKey::from_slice(&secret).expect("fixture private key");
        let keypair = Keypair::from_secret_key(&Secp256k1::new(), &secret);
        let (public_key, _) = XOnlyPublicKey::from_keypair(&keypair);
        encode_hex(&public_key.serialize())
    }

    fn provider_configuration() -> RemoteProviderProjectConfiguration {
        RemoteProviderProjectConfiguration::new(BTreeMap::from([
            (
                "image".to_owned(),
                RemoteProviderConfigValue::String("image@sha256:abc".to_owned()),
            ),
            (
                "inactivity_seconds".to_owned(),
                RemoteProviderConfigValue::Number(Number::from(3600)),
            ),
        ]))
        .expect("fixture provider configuration")
    }

    fn template(
        secret: [u8; 32],
        environment: BTreeMap<EnvironmentVariableName, EnvironmentReference>,
    ) -> RemoteProviderDeployTemplate {
        let Value::Object(agent) = json!({
            "relay_url": "wss://relay.example",
            "launch": {
                "command": "goose",
                "owner_pubkey": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }
        }) else {
            panic!("fixture agent object")
        };
        RemoteProviderDeployTemplate::new(
            "operation-1",
            PathBuf::from("/tmp/project"),
            &identity(secret),
            agent,
            provider_configuration(),
            RemoteProviderSecretReferences::new(
                credential_reference(IDENTITY_REFERENCE),
                None,
                environment,
            )
            .expect("fixture secret references"),
        )
        .expect("fixture deploy template")
    }

    #[gpui::test]
    async fn locked_keyring_fails_with_redacted_diagnostic(cx: &mut TestAppContext) {
        let template = template([7; 32], BTreeMap::new());
        let error = resolution_error(
            template
                .resolve_with_environment(
                    &FakeCredentialsProvider::locked(),
                    &EmptyProcessEnvironment,
                    &cx.to_async(),
                )
                .await,
        );

        assert_eq!(
            error,
            RemoteProviderConfigError::ProtectedStorageUnavailable
        );
        let diagnostic = format!("{error:?}: {error}");
        assert!(!diagnostic.contains(IDENTITY_REFERENCE));
        assert!(!diagnostic.contains(SECRET_SENTINEL));
    }

    #[gpui::test]
    async fn missing_empty_and_mismatched_identity_fail_before_deploy(cx: &mut TestAppContext) {
        let template = template([7; 32], BTreeMap::new());
        let missing = FakeCredentialsProvider::values([]);
        assert_eq!(
            resolution_error(
                template
                    .resolve_with_environment(&missing, &EmptyProcessEnvironment, &cx.to_async())
                    .await,
            ),
            RemoteProviderConfigError::MissingIdentityCredential
        );

        let empty = FakeCredentialsProvider::values([(IDENTITY_REFERENCE.to_owned(), Vec::new())]);
        assert_eq!(
            resolution_error(
                template
                    .resolve_with_environment(&empty, &EmptyProcessEnvironment, &cx.to_async())
                    .await,
            ),
            RemoteProviderConfigError::EmptyIdentityCredential
        );

        let mismatched =
            FakeCredentialsProvider::values([(IDENTITY_REFERENCE.to_owned(), [8_u8; 32].to_vec())]);
        assert_eq!(
            resolution_error(
                template
                    .resolve_with_environment(
                        &mismatched,
                        &EmptyProcessEnvironment,
                        &cx.to_async(),
                    )
                    .await,
            ),
            RemoteProviderConfigError::IdentityCredentialMismatch
        );
    }

    #[gpui::test]
    async fn credentials_resolve_only_into_the_ephemeral_deploy_input(cx: &mut TestAppContext) {
        let secret = [7; 32];
        let environment_name =
            EnvironmentVariableName::parse("MODEL_API_TOKEN").expect("fixture environment name");
        let environment = BTreeMap::from([(
            environment_name,
            EnvironmentReference::ProtectedCredential(credential_reference(ENVIRONMENT_REFERENCE)),
        )]);
        let template = template(secret, environment);
        let credentials = FakeCredentialsProvider::values([
            (IDENTITY_REFERENCE.to_owned(), secret.to_vec()),
            (
                ENVIRONMENT_REFERENCE.to_owned(),
                SECRET_SENTINEL.as_bytes().to_vec(),
            ),
        ]);

        let input = template
            .resolve_with_environment(&credentials, &EmptyProcessEnvironment, &cx.to_async())
            .await
            .expect("credentials should resolve");

        assert_eq!(input.agent["env_vars"]["MODEL_API_TOKEN"], SECRET_SENTINEL);
        assert_eq!(
            input.agent["launch"]["env"]["MODEL_API_TOKEN"],
            SECRET_SENTINEL
        );
        assert!(
            input.agent["private_key_nsec"]
                .as_str()
                .is_some_and(|value| value.starts_with("nsec1"))
        );
        let durable = format!("{template:?} {:?}", template.provider_configuration());
        assert!(!durable.contains(IDENTITY_REFERENCE));
        assert!(!durable.contains(ENVIRONMENT_REFERENCE));
        assert!(!durable.contains(SECRET_SENTINEL));
        assert!(
            !template
                .provider_configuration()
                .to_json()
                .to_string()
                .contains(SECRET_SENTINEL)
        );
    }

    #[test]
    fn provider_configuration_is_typed_scalar_only_and_secret_free() {
        let configuration = provider_configuration();
        assert_eq!(configuration.field_count(), 2);
        assert_eq!(configuration.to_json()["inactivity_seconds"], 3600);
        assert!(configuration.to_json()["inactivity_seconds"].is_number());

        let secret_key = RemoteProviderProjectConfiguration::new(BTreeMap::from([(
            "clientSecret".to_owned(),
            RemoteProviderConfigValue::String(SECRET_SENTINEL.to_owned()),
        )]))
        .expect_err("secret-like config key must fail");
        let nested_value_cannot_be_constructed = format!("{secret_key:?}: {secret_key}");
        assert_eq!(
            secret_key,
            RemoteProviderConfigError::SecretLikeProviderConfigKey
        );
        assert!(!nested_value_cannot_be_constructed.contains(SECRET_SENTINEL));
        assert!(!nested_value_cannot_be_constructed.contains("clientSecret"));
    }
}
