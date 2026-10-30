use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

use url::Url;

const PUBLIC_URL: &str = "COLLABORATION_PUBLIC_URL";
const DATABASE_URL: &str = "COLLABORATION_DATABASE_URL";
const READ_DATABASE_URL: &str = "COLLABORATION_READ_DATABASE_URL";
const REDIS_URL: &str = "COLLABORATION_REDIS_URL";
const OBJECT_ENDPOINT: &str = "COLLABORATION_OBJECT_ENDPOINT";
const OBJECT_REGION: &str = "COLLABORATION_OBJECT_REGION";
const OBJECT_BUCKET: &str = "COLLABORATION_OBJECT_BUCKET";
const OBJECT_ACCESS_KEY: &str = "COLLABORATION_OBJECT_ACCESS_KEY";
const OBJECT_SECRET_KEY: &str = "COLLABORATION_OBJECT_SECRET_KEY";
const OBJECT_ADDRESSING_STYLE: &str = "COLLABORATION_OBJECT_ADDRESSING_STYLE";
const GIT_REPOSITORY_PATH: &str = "COLLABORATION_GIT_REPOSITORY_PATH";
const GIT_HOOK_SECRET: &str = "COLLABORATION_GIT_HOOK_SECRET";
const REPLICA_COUNT: &str = "COLLABORATION_REPLICA_COUNT";
const PUSH_ENABLED: &str = "COLLABORATION_PUSH_ENABLED";
const PUSH_URL: &str = "COLLABORATION_PUSH_URL";
const PUSH_CREDENTIAL: &str = "COLLABORATION_PUSH_CREDENTIAL";
const PAIRING_ENABLED: &str = "COLLABORATION_PAIRING_ENABLED";
const PAIRING_URL: &str = "COLLABORATION_PAIRING_URL";
const RELAY_MESH_ENABLED: &str = "COLLABORATION_RELAY_MESH_ENABLED";
const RELAY_MESH_PEERS: &str = "COLLABORATION_RELAY_MESH_PEERS";
const RELAY_MESH_TRUST_ROOT: &str = "COLLABORATION_RELAY_MESH_TRUST_ROOT";
const MAX_MESH_PEERS: usize = 32;

#[derive(Clone, Eq, PartialEq)]
pub struct SecretValue(String);

impl SecretValue {
    fn new(value: String, field: &'static str) -> Result<Self, CollaborationConfigError> {
        if value.is_empty() || value.len() > 8_192 || contains_control_character(&value) {
            return Err(CollaborationConfigError::invalid(field));
        }
        Ok(Self(value))
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretValue([REDACTED])")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectAddressingStyle {
    Path,
    VirtualHosted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationDatabaseConfig {
    writer_url: SecretValue,
    reader_url: Option<SecretValue>,
}

impl CollaborationDatabaseConfig {
    pub fn writer_url(&self) -> &SecretValue {
        &self.writer_url
    }

    pub fn reader_url(&self) -> Option<&SecretValue> {
        self.reader_url.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationObjectStoreConfig {
    endpoint: Url,
    region: String,
    bucket: String,
    addressing_style: ObjectAddressingStyle,
    access_key: SecretValue,
    secret_key: SecretValue,
}

impl CollaborationObjectStoreConfig {
    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    pub fn addressing_style(&self) -> ObjectAddressingStyle {
        self.addressing_style
    }

    pub fn access_key(&self) -> &SecretValue {
        &self.access_key
    }

    pub fn secret_key(&self) -> &SecretValue {
        &self.secret_key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationGitConfig {
    repository_path: PathBuf,
    hook_secret: Option<SecretValue>,
}

impl CollaborationGitConfig {
    pub fn repository_path(&self) -> &Path {
        &self.repository_path
    }

    pub fn hook_secret(&self) -> Option<&SecretValue> {
        self.hook_secret.as_ref()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationPushConfig {
    endpoint: Url,
    credential: SecretValue,
}

impl CollaborationPushConfig {
    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    pub fn credential(&self) -> &SecretValue {
        &self.credential
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationPairingConfig {
    relay_url: Url,
}

impl CollaborationPairingConfig {
    pub fn relay_url(&self) -> &Url {
        &self.relay_url
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationRelayMeshConfig {
    peers: Vec<Url>,
    trust_root: SecretValue,
}

impl CollaborationRelayMeshConfig {
    pub fn peers(&self) -> &[Url] {
        &self.peers
    }

    pub fn trust_root(&self) -> &SecretValue {
        &self.trust_root
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollaborationServiceConfig {
    public_url: Url,
    replica_count: u16,
    database: CollaborationDatabaseConfig,
    redis_url: Option<SecretValue>,
    object_store: CollaborationObjectStoreConfig,
    git: CollaborationGitConfig,
    push: Option<CollaborationPushConfig>,
    pairing: Option<CollaborationPairingConfig>,
    relay_mesh: Option<CollaborationRelayMeshConfig>,
}

impl CollaborationServiceConfig {
    pub fn from_env() -> Result<Self, CollaborationConfigError> {
        Self::from_variables(std::env::vars())
    }

    pub fn from_variables(
        variables: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Result<Self, CollaborationConfigError> {
        let variables = variables
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect::<BTreeMap<_, _>>();
        let public_url = service_url(required(&variables, PUBLIC_URL)?, PUBLIC_URL)?;
        let replica_count = optional_integer(&variables, REPLICA_COUNT, 1, 1, 256)?;
        let writer_url = database_secret(required(&variables, DATABASE_URL)?, DATABASE_URL)?;
        let reader_url = optional(&variables, READ_DATABASE_URL)
            .map(|value| database_secret(value, READ_DATABASE_URL))
            .transpose()?;
        let redis_url = optional(&variables, REDIS_URL)
            .map(|value| redis_secret(value, REDIS_URL))
            .transpose()?;
        let object_store = CollaborationObjectStoreConfig {
            endpoint: service_url(required(&variables, OBJECT_ENDPOINT)?, OBJECT_ENDPOINT)?,
            region: identifier(required(&variables, OBJECT_REGION)?, OBJECT_REGION, 64)?,
            bucket: bucket(required(&variables, OBJECT_BUCKET)?)?,
            addressing_style: match optional(&variables, OBJECT_ADDRESSING_STYLE).unwrap_or("path")
            {
                "path" => ObjectAddressingStyle::Path,
                "virtual-hosted" => ObjectAddressingStyle::VirtualHosted,
                _ => return Err(CollaborationConfigError::invalid(OBJECT_ADDRESSING_STYLE)),
            },
            access_key: SecretValue::new(
                required(&variables, OBJECT_ACCESS_KEY)?.to_owned(),
                OBJECT_ACCESS_KEY,
            )?,
            secret_key: SecretValue::new(
                required(&variables, OBJECT_SECRET_KEY)?.to_owned(),
                OBJECT_SECRET_KEY,
            )?,
        };
        let repository_path = PathBuf::from(required(&variables, GIT_REPOSITORY_PATH)?);
        if !repository_path.is_absolute() {
            return Err(CollaborationConfigError::invalid(GIT_REPOSITORY_PATH));
        }
        let hook_secret = optional(&variables, GIT_HOOK_SECRET)
            .map(|value| SecretValue::new(value.to_owned(), GIT_HOOK_SECRET))
            .transpose()?;
        if replica_count > 1 && hook_secret.is_none() {
            return Err(CollaborationConfigError::missing(GIT_HOOK_SECRET));
        }
        let push = if optional_boolean(&variables, PUSH_ENABLED, false)? {
            Some(CollaborationPushConfig {
                endpoint: service_url(required(&variables, PUSH_URL)?, PUSH_URL)?,
                credential: SecretValue::new(
                    required(&variables, PUSH_CREDENTIAL)?.to_owned(),
                    PUSH_CREDENTIAL,
                )?,
            })
        } else {
            reject_disabled_values(&variables, PUSH_URL, PUSH_CREDENTIAL)?;
            None
        };
        let pairing = if optional_boolean(&variables, PAIRING_ENABLED, false)? {
            Some(CollaborationPairingConfig {
                relay_url: websocket_url(required(&variables, PAIRING_URL)?, PAIRING_URL)?,
            })
        } else {
            reject_disabled_values(&variables, PAIRING_URL, PAIRING_URL)?;
            None
        };
        let relay_mesh = if optional_boolean(&variables, RELAY_MESH_ENABLED, false)? {
            if replica_count < 2 || redis_url.is_none() {
                return Err(CollaborationConfigError::incompatible(RELAY_MESH_ENABLED));
            }
            let peers = mesh_peers(required(&variables, RELAY_MESH_PEERS)?)?;
            Some(CollaborationRelayMeshConfig {
                peers,
                trust_root: SecretValue::new(
                    required(&variables, RELAY_MESH_TRUST_ROOT)?.to_owned(),
                    RELAY_MESH_TRUST_ROOT,
                )?,
            })
        } else {
            reject_disabled_values(&variables, RELAY_MESH_PEERS, RELAY_MESH_TRUST_ROOT)?;
            None
        };
        Ok(Self {
            public_url,
            replica_count,
            database: CollaborationDatabaseConfig {
                writer_url,
                reader_url,
            },
            redis_url,
            object_store,
            git: CollaborationGitConfig {
                repository_path,
                hook_secret,
            },
            push,
            pairing,
            relay_mesh,
        })
    }

    pub fn public_url(&self) -> &Url {
        &self.public_url
    }

    pub fn replica_count(&self) -> u16 {
        self.replica_count
    }

    pub fn database(&self) -> &CollaborationDatabaseConfig {
        &self.database
    }

    pub fn redis_url(&self) -> Option<&SecretValue> {
        self.redis_url.as_ref()
    }

    pub fn object_store(&self) -> &CollaborationObjectStoreConfig {
        &self.object_store
    }

    pub fn git(&self) -> &CollaborationGitConfig {
        &self.git
    }

    pub fn push(&self) -> Option<&CollaborationPushConfig> {
        self.push.as_ref()
    }

    pub fn pairing(&self) -> Option<&CollaborationPairingConfig> {
        self.pairing.as_ref()
    }

    pub fn relay_mesh(&self) -> Option<&CollaborationRelayMeshConfig> {
        self.relay_mesh.as_ref()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollaborationConfigErrorKind {
    Missing,
    Invalid,
    Incompatible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CollaborationConfigError {
    kind: CollaborationConfigErrorKind,
    field: &'static str,
}

impl CollaborationConfigError {
    const fn missing(field: &'static str) -> Self {
        Self {
            kind: CollaborationConfigErrorKind::Missing,
            field,
        }
    }

    const fn invalid(field: &'static str) -> Self {
        Self {
            kind: CollaborationConfigErrorKind::Invalid,
            field,
        }
    }

    const fn incompatible(field: &'static str) -> Self {
        Self {
            kind: CollaborationConfigErrorKind::Incompatible,
            field,
        }
    }

    pub fn kind(self) -> CollaborationConfigErrorKind {
        self.kind
    }

    pub fn field(self) -> &'static str {
        self.field
    }

    pub const fn diagnostic_code(self) -> &'static str {
        match self.kind {
            CollaborationConfigErrorKind::Missing => "collaboration_config_missing",
            CollaborationConfigErrorKind::Invalid => "collaboration_config_invalid",
            CollaborationConfigErrorKind::Incompatible => "collaboration_config_incompatible",
        }
    }
}

impl fmt::Display for CollaborationConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "collaboration configuration rejected ({}: {})",
            self.diagnostic_code(),
            self.field
        )
    }
}

impl Error for CollaborationConfigError {}

fn required<'a>(
    variables: &'a BTreeMap<String, String>,
    field: &'static str,
) -> Result<&'a str, CollaborationConfigError> {
    optional(variables, field).ok_or_else(|| CollaborationConfigError::missing(field))
}

fn optional<'a>(variables: &'a BTreeMap<String, String>, field: &str) -> Option<&'a str> {
    variables
        .get(field)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

fn optional_boolean(
    variables: &BTreeMap<String, String>,
    field: &'static str,
    default: bool,
) -> Result<bool, CollaborationConfigError> {
    match optional(variables, field) {
        None => Ok(default),
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(_) => Err(CollaborationConfigError::invalid(field)),
    }
}

fn optional_integer(
    variables: &BTreeMap<String, String>,
    field: &'static str,
    default: u16,
    minimum: u16,
    maximum: u16,
) -> Result<u16, CollaborationConfigError> {
    let Some(value) = optional(variables, field) else {
        return Ok(default);
    };
    let value = value
        .parse::<u16>()
        .map_err(|_| CollaborationConfigError::invalid(field))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(CollaborationConfigError::invalid(field));
    }
    Ok(value)
}

fn database_secret(
    value: &str,
    field: &'static str,
) -> Result<SecretValue, CollaborationConfigError> {
    let url = Url::parse(value).map_err(|_| CollaborationConfigError::invalid(field))?;
    if !matches!(url.scheme(), "postgres" | "postgresql")
        || url.host_str().is_none()
        || url.path().len() <= 1
    {
        return Err(CollaborationConfigError::invalid(field));
    }
    SecretValue::new(value.to_owned(), field)
}

fn redis_secret(value: &str, field: &'static str) -> Result<SecretValue, CollaborationConfigError> {
    let url = Url::parse(value).map_err(|_| CollaborationConfigError::invalid(field))?;
    if !matches!(url.scheme(), "redis" | "rediss") || url.host_str().is_none() {
        return Err(CollaborationConfigError::invalid(field));
    }
    SecretValue::new(value.to_owned(), field)
}

fn service_url(value: &str, field: &'static str) -> Result<Url, CollaborationConfigError> {
    let url = Url::parse(value).map_err(|_| CollaborationConfigError::invalid(field))?;
    if url.username() != ""
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !secure_or_loopback(&url, "https", "http")
    {
        return Err(CollaborationConfigError::invalid(field));
    }
    Ok(url)
}

fn websocket_url(value: &str, field: &'static str) -> Result<Url, CollaborationConfigError> {
    let url = Url::parse(value).map_err(|_| CollaborationConfigError::invalid(field))?;
    if url.username() != ""
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !secure_or_loopback(&url, "wss", "ws")
    {
        return Err(CollaborationConfigError::invalid(field));
    }
    Ok(url)
}

fn secure_or_loopback(url: &Url, secure: &str, insecure: &str) -> bool {
    url.scheme() == secure
        || (url.scheme() == insecure
            && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1")))
}

fn identifier(
    value: &str,
    field: &'static str,
    maximum: usize,
) -> Result<String, CollaborationConfigError> {
    if value.is_empty()
        || value.len() > maximum
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err(CollaborationConfigError::invalid(field));
    }
    Ok(value.to_owned())
}

fn bucket(value: &str) -> Result<String, CollaborationConfigError> {
    if !(3..=63).contains(&value.len())
        || value.starts_with(['.', '-'])
        || value.ends_with(['.', '-'])
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b".-".contains(&byte))
    {
        return Err(CollaborationConfigError::invalid(OBJECT_BUCKET));
    }
    Ok(value.to_owned())
}

fn mesh_peers(value: &str) -> Result<Vec<Url>, CollaborationConfigError> {
    let values = value.split(',').collect::<Vec<_>>();
    if values.is_empty() || values.len() > MAX_MESH_PEERS {
        return Err(CollaborationConfigError::invalid(RELAY_MESH_PEERS));
    }
    let mut unique = BTreeSet::new();
    let mut peers = Vec::with_capacity(values.len());
    for value in values {
        if value.trim() != value {
            return Err(CollaborationConfigError::invalid(RELAY_MESH_PEERS));
        }
        let peer = service_url(value, RELAY_MESH_PEERS)?;
        if !unique.insert(peer.as_str().to_owned()) {
            return Err(CollaborationConfigError::invalid(RELAY_MESH_PEERS));
        }
        peers.push(peer);
    }
    Ok(peers)
}

fn reject_disabled_values(
    variables: &BTreeMap<String, String>,
    first: &'static str,
    second: &'static str,
) -> Result<(), CollaborationConfigError> {
    if optional(variables, first).is_some() || optional(variables, second).is_some() {
        return Err(CollaborationConfigError::incompatible(first));
    }
    Ok(())
}

fn contains_control_character(value: &str) -> bool {
    value
        .chars()
        .any(|character| character.is_control() || character == '\u{7f}')
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATABASE_CANARY: &str = "postgres://collab:database-secret@db.internal/collab";
    const OBJECT_SECRET_CANARY: &str = "object-secret-canary";
    const REDIS_CANARY: &str = "rediss://:redis-secret@redis.internal/0";

    fn valid_variables() -> Vec<(&'static str, &'static str)> {
        vec![
            (PUBLIC_URL, "https://collab.example.test"),
            (DATABASE_URL, DATABASE_CANARY),
            (
                READ_DATABASE_URL,
                "postgres://reader:secret@read.internal/collab",
            ),
            (REDIS_URL, REDIS_CANARY),
            (OBJECT_ENDPOINT, "https://objects.example.test"),
            (OBJECT_REGION, "eu-west-2"),
            (OBJECT_BUCKET, "collaboration-objects"),
            (OBJECT_ACCESS_KEY, "object-access-canary"),
            (OBJECT_SECRET_KEY, OBJECT_SECRET_CANARY),
            (OBJECT_ADDRESSING_STYLE, "virtual-hosted"),
            (GIT_REPOSITORY_PATH, "/var/lib/collab/git"),
            (GIT_HOOK_SECRET, "git-hook-secret-canary"),
            (REPLICA_COUNT, "2"),
            (PUSH_ENABLED, "true"),
            (PUSH_URL, "https://push.example.test"),
            (PUSH_CREDENTIAL, "push-secret-canary"),
            (PAIRING_ENABLED, "true"),
            (PAIRING_URL, "wss://pair.example.test"),
            (RELAY_MESH_ENABLED, "true"),
            (
                RELAY_MESH_PEERS,
                "https://mesh-a.example.test,https://mesh-b.example.test",
            ),
            (RELAY_MESH_TRUST_ROOT, "mesh-trust-secret-canary"),
        ]
    }

    #[test]
    fn valid_configuration_covers_every_canonical_dependency() {
        let config = CollaborationServiceConfig::from_variables(valid_variables())
            .expect("valid collaboration configuration");

        assert_eq!(config.public_url().as_str(), "https://collab.example.test/");
        assert_eq!(config.replica_count(), 2);
        assert_eq!(config.database().writer_url().expose(), DATABASE_CANARY);
        assert!(config.database().reader_url().is_some());
        assert_eq!(config.redis_url().expect("redis").expose(), REDIS_CANARY);
        assert_eq!(config.object_store().bucket(), "collaboration-objects");
        assert_eq!(
            config.object_store().addressing_style(),
            ObjectAddressingStyle::VirtualHosted
        );
        assert_eq!(
            config.git().repository_path(),
            Path::new("/var/lib/collab/git")
        );
        assert!(config.git().hook_secret().is_some());
        assert_eq!(
            config.push().expect("push").endpoint().as_str(),
            "https://push.example.test/"
        );
        assert_eq!(
            config.pairing().expect("pairing").relay_url().scheme(),
            "wss"
        );
        assert_eq!(config.relay_mesh().expect("mesh").peers().len(), 2);
    }

    #[test]
    fn missing_secret_has_a_stable_value_free_diagnostic() {
        let variables = valid_variables()
            .into_iter()
            .filter(|(key, _)| *key != OBJECT_SECRET_KEY);
        let error = CollaborationServiceConfig::from_variables(variables)
            .expect_err("missing object secret must fail");

        assert_eq!(error.kind(), CollaborationConfigErrorKind::Missing);
        assert_eq!(error.field(), OBJECT_SECRET_KEY);
        assert_eq!(error.diagnostic_code(), "collaboration_config_missing");
        assert!(!error.to_string().contains(OBJECT_SECRET_CANARY));
    }

    #[test]
    fn relay_mesh_rejects_an_incompatible_single_replica_configuration() {
        let variables = valid_variables().into_iter().map(|(key, value)| {
            if key == REPLICA_COUNT {
                (key, "1")
            } else {
                (key, value)
            }
        });
        let error = CollaborationServiceConfig::from_variables(variables)
            .expect_err("single replica mesh must fail");

        assert_eq!(error.kind(), CollaborationConfigErrorKind::Incompatible);
        assert_eq!(error.field(), RELAY_MESH_ENABLED);
    }

    #[test]
    fn debug_and_failure_surfaces_redact_every_secret() {
        let config = CollaborationServiceConfig::from_variables(valid_variables())
            .expect("valid collaboration configuration");
        let diagnostic = format!("{config:?}");

        for canary in [DATABASE_CANARY, OBJECT_SECRET_CANARY, REDIS_CANARY] {
            assert!(!diagnostic.contains(canary));
        }
        assert!(diagnostic.contains("[REDACTED]"));

        let variables = valid_variables().into_iter().map(|(key, value)| {
            if key == PAIRING_URL {
                (key, "wss://pair.example.test?secret=pair-canary")
            } else {
                (key, value)
            }
        });
        let error = CollaborationServiceConfig::from_variables(variables)
            .expect_err("credential-bearing pairing URL must fail");
        let diagnostic = format!("{error:?} {error}");
        assert!(!diagnostic.contains("pair-canary"));
        assert_eq!(error.field(), PAIRING_URL);
    }
}
